import logging
import os
import time
import torch
import torch.nn as nn
import random
import numpy as np
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from utils.reranking import cheb_gr_reranking, re_ranking

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    scaler = amp.GradScaler()

    # 开始训练循环
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        scheduler.step(epoch)
        model.train()
        
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)

            # --- 阶段 2: 引入 BIPO (批次内行人遮挡增强) ---
            # 适当降低概率至 0.2，确保模型能够收敛
            if cfg.DATASETS.BIPO and random.random() < 0.2:
                batch_size = img.size(0)
                indices = torch.randperm(batch_size)
                for i in range(batch_size):
                    src_idx = indices[i]
                    # 确保不是同一个 ID 的球员，模拟真实的拥挤遮挡
                    if vid[i] != vid[src_idx]:
                        # 随机截取 40x40 的局部 patch
                        h_start = random.randint(0, img.size(2) - 40)
                        w_start = random.randint(0, img.size(3) - 40)
                        patch = img[src_idx, :, h_start:h_start+40, w_start:w_start+40]
                        img[i, :, h_start:h_start+40, w_start:w_start+40] = patch

            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view )
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            # 计算准确率
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if not cfg.MODEL.DIST_TRAIN:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        # 保存权重
        if epoch % checkpoint_period == 0:
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch))
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)

        # --- 核心修正：验证环节必须调用 do_inference 以激活 Cheb-GR ---
        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    do_inference(cfg, model, val_loader, num_query)
            else:
                do_inference(cfg, model, val_loader, num_query)
            torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    # --- 阶段 4: 集成超低延迟重排序 (Cheb-GR) ---
    if cfg.TEST.RE_RANKING_TYPE in ['cheb_gr', 'yes']:
        # 提取特征
        feats = torch.cat(evaluator.feats, dim=0)
        
        # 分离 Query 和 Gallery
        qf = feats[:num_query]
        gf = feats[num_query:]
        
        # 计算欧氏距离矩阵
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()

        if cfg.TEST.RE_RANKING_TYPE == 'cheb_gr':
            # Cheb-GR: 线性复杂度重排序，完美契合 30ms 红线
            logger.info("Applying Cheb-GR re-ranking...")
            distmat = cheb_gr_reranking(distmat, kappa=cfg.TEST.KAPPA)
        else:
            # 兼容旧版 k-reciprocal
            logger.info("Applying k-reciprocal re-ranking...")
            distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        
        # 使用重排序矩阵计算最终指标
        cmc, mAP, _, _, _, _, _ = evaluator.compute(distmat=distmat)
    else:
        # 无重排序模式
        cmc, mAP, _, _, _, _, _ = evaluator.compute()

    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    
    return cmc[0], cmc[4]
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

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        scheduler.step(epoch)
        model.train()
        
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            if optimizer_center is not None:
                optimizer_center.zero_grad()
            
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)

            # --- 阶段 2: 向量化 BIPO (不拖慢 GPU 速度的安全实现) ---
            if cfg.DATASETS.BIPO and random.random() < 0.2:
                with torch.no_grad():
                    shifted_img = torch.roll(img, shifts=1, dims=0)
                    shifted_vid = torch.roll(vid, shifts=1, dims=0)
                    diff_mask = (vid != shifted_vid)
                    if diff_mask.any():
                        h_start = random.randint(0, img.size(2) - 40)
                        w_start = random.randint(0, img.size(3) - 40)
                        patch = shifted_img[:, :, h_start:h_start+40, w_start:w_start+40]
                        img[diff_mask, :, h_start:h_start+40, w_start:w_start+40] = patch[diff_mask]

            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view )
                loss = loss_fn(score, feat, target, target_cam)

            # --- 致命 Bug 2 修复区：AMP 正确的更新逻辑 ---
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            if cfg.MODEL.IF_WITH_CENTER == 'yes' or 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                if optimizer_center is not None:
                    for param in center_criterion.parameters():
                        param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                    scaler.step(optimizer_center)

            # 绝对不能写两次！只能在这里执行唯一的一次 update
            scaler.update()
            # ----------------------------------------------------

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

        if epoch % checkpoint_period == 0:
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch))
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    do_inference(cfg, model, val_loader, num_query)
            else:
                do_inference(cfg, model, val_loader, num_query)
            torch.cuda.empty_cache()


def do_inference(cfg, model, val_loader, num_query):
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

    if cfg.TEST.RE_RANKING_TYPE in ['cheb_gr', 'yes']:
        feats = torch.cat(evaluator.feats, dim=0)
        
        # --- 致命 Bug 1 修复区：强制执行 L2 归一化！ ---
        if cfg.TEST.FEAT_NORM == 'yes':
            logger.info("Applying L2 Normalization before re-ranking...")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # ---------------------------------------------
        
        qf = feats[:num_query]
        gf = feats[num_query:]
        
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()

        if cfg.TEST.RE_RANKING_TYPE == 'cheb_gr':
            logger.info("Applying Cheb-GR re-ranking...")
            distmat = cheb_gr_reranking(distmat, kappa=cfg.TEST.KAPPA)
        else:
            logger.info("Applying k-reciprocal re-ranking...")
            distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        
        cmc, mAP, _, _, _, _, _ = evaluator.compute(distmat=distmat)
    else:
        cmc, mAP, _, _, _, _, _ = evaluator.compute()

    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    
    return cmc[0], cmc[4]
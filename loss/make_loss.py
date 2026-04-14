# encoding: utf-8
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss

def make_loss(cfg, num_classes):    
    sampler = cfg.DATALOADER.SAMPLER
    # 修复：ViT-Base 的输出维度是 768，不是 2048
    feat_dim = 768 
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)
    
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)
    else:
        print('expected METRIC_LOSS_TYPE should be triplet')

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)

    if cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam=None):
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                if isinstance(score, list):
                    ID_LOSS = [xent(scor, target) for scor in score[1:]]
                    ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                    ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                else:
                    ID_LOSS = xent(score, target)
            else:
                if isinstance(score, list):
                    ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                    ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                    ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                else:
                    ID_LOSS = F.cross_entropy(score, target)

            if isinstance(feat, list):
                TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
            else:
                TRI_LOSS = triplet(feat, target)[0]

            # 核心修复：将 CenterLoss 真正加入计算图
            if cfg.MODEL.IF_WITH_CENTER == 'yes' or cfg.MODEL.IF_WITH_CENTER == 'on':
                if isinstance(feat, list):
                    CENTER_LOSS = [center_criterion(feats, target) for feats in feat[1:]]
                    CENTER_LOSS = sum(CENTER_LOSS) / len(CENTER_LOSS)
                    CENTER_LOSS = 0.5 * CENTER_LOSS + 0.5 * center_criterion(feat[0], target)
                else:
                    CENTER_LOSS = center_criterion(feat, target)
                
                # 返回三种 Loss 的加权总和
                return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                       cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS + \
                       cfg.SOLVER.CENTER_LOSS_WEIGHT * CENTER_LOSS
            else:
                return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                       cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

    return loss_func, center_criterion
import torch
import torchvision.transforms as T
import random
import math
import numpy as np
from PIL import Image

class RandomErasing(object):
    """ 原版随机擦除实现 """
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img
        return img

class MotionBlur(object):
    """ 修正版：针对篮球场景的物理运动模糊（带局部遮罩保护） """
    def __init__(self, p=0.25, l=3, f=7): # 建议 p 降至 0.25 避免欠拟合
        self.p = p
        self.l = l 
        self.f = f 

    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        img_np = np.array(img)
        h, w, c = img_np.shape
        
        # 1. 局部区域自适应提取: 确定球员主体区域，保护背景
        y1, y2 = int(h * 0.1), int(h * 0.9)
        x1, x2 = int(w * 0.2), int(w * 0.8)
        player_region = img_np[y1:y2, x1:x2].copy()

        # 2. 方向性位移选择
        p_dir = random.random()
        dx, dy = ((-1, 0) if p_dir < 0.4 else (1, 0) if p_dir < 0.8 else random.choice([(-1, 1), (1, 1)]))

        # 3. 视觉残影卷积 (修正：仅对球员区域应用物理位移)
        blurred_region = player_region.copy()
        for i in range(self.l, 0, -1):
            shift_x, shift_y = dx * i * self.f, dy * i * self.f
            # 使用 roll 并截断边缘，模拟真实物理滞后
            temp_region = np.roll(player_region, shift=(shift_y, shift_x), axis=(0, 1))
            alpha = 1.0 / (i + 1)
            blurred_region = (blurred_region * (1 - alpha) + temp_region * alpha).astype(np.uint8)
        
        # 将模糊后的球员区域贴回原图，保持背景清晰
        new_img = img_np.copy()
        new_img[y1:y2, x1:x2] = blurred_region
            
        return Image.fromarray(new_img)

def build_transforms(cfg, is_train=True):
    res = []
    res.append(T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3))
    
    if is_train:
        res.append(T.RandomHorizontalFlip(p=cfg.INPUT.PROB))
        res.append(T.Pad(cfg.INPUT.PADDING))
        res.append(T.RandomCrop(cfg.INPUT.SIZE_TRAIN))
        
        # 1. ColorJitter: 应对反光
        res.append(T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0.01))
        
        # 2. MotionBlur: 使用修正后的局部模糊
        # 注意：此处推荐在 YAML 配置中将 cfg.INPUT.MOTION_BLUR_PROB 设为 0.25
        res.append(MotionBlur(p=cfg.INPUT.MOTION_BLUR_PROB))
        
        res.append(T.ToTensor())
        res.append(T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD))
        
        # 3. RandomErasing
        res.append(RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.RE_ID))
    else:
        res.append(T.ToTensor())
        res.append(T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD))
        
    return T.Compose(res)
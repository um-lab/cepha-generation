import random
from typing import Dict, Optional, Union, Tuple, List

import torch
import cv2
from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomBlur(BaseTransform):
    def __init__(self) -> None:
        super().__init__()

    def transform(self, results: Dict):
        # 对图像进行高斯模糊
        # cv2.GaussianBlur(src, ksize, sigmaX, sigmaY=None, borderType=None) -> dst
        # 其中：
        # - src是输入的图像
        # - ksize是高斯核的大小（必须是奇数和正数，例如：(3,3)、(5,5)、(7,7)等）
        # - sigmaX是X方向上的标准偏差
        # - sigmaY是Y方向上的标准偏差，如果sigmaY是0，则将其设置为与sigmaX相同的值；如果它们都是零，那么它们是从ksize.width和ksize.height计算得出的

        # torch.save(results, '/home/jianbingshen/guodongqian/CL-Detection2023-MMPose/augmentation_debug/before_results.pth')
        ksize = random.choice(list(range(3, 16, 2)))
        blurred_image = cv2.GaussianBlur(results['img'], (ksize, ksize), 0)
        results['img'] = blurred_image
        # torch.save(results, '/home/jianbingshen/guodongqian/CL-Detection2023-MMPose/augmentation_debug/after_results.pth')

        return results

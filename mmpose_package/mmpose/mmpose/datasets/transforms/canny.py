import random
from typing import Dict

import numpy as np
import cv2
from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class CannyChannel(BaseTransform):
    def __init__(self) -> None:
        super().__init__()

    def transform(self, results: Dict):
        gray_img = cv2.cvtColor(results['img'], cv2.COLOR_BGR2GRAY)
        # threshold1 = np.randomint()
        # threshold2 = np.randomint()

        edges = cv2.Canny(gray_img, threshold1=20, threshold2=45)
        new_image = np.dstack((results['img'], edges))
        results['img'] = new_image

        # torch.save([gray_img, edges], '/home/jianbingshen/guodongqian/CL-Detection2023-MMPose/augmentation_debug/canny.pth')

        return results
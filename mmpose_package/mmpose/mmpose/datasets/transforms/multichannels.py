import numpy as np
from typing import Dict, Optional, Union, Tuple, List

from mmcv.transforms import BaseTransform

from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MultiChannels(BaseTransform):

    """
    alpha:调节对比度。大于1增加对比度，黑白部分更明显；小于1降低对比度，黑白部分更接近。
    beta:调节亮度。大于0增加亮度；小于1降低亮度。
    """
    def __init__(self, alpha, beta) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        image = results['img'].copy()
        image_increase_contrast = np.clip(image.astype(np.float32) * self.alpha, 0, 255).astype(np.uint8)
        image_decrease_contrast = np.clip(image.astype(np.float32) / self.alpha, 0, 255).astype(np.uint8)
        image_increase_brightness = np.clip(image.astype(np.float32) + self.beta, 0, 255).astype(np.uint8)
        image_decrease_brightness = np.clip(image.astype(np.float32) - self.beta, 0, 255).astype(np.uint8)

        # output_path = '/home/jianbingshen/guodongqian/CL-Detection2023-MMPose/visualize/multichannels/'
        # cv2.imwrite(output_path + str(results['img_id']) + '.png', image)
        # cv2.imwrite(output_path + str(results['img_id']) + '_increase_contrast.png', image_increase_contrast)
        # cv2.imwrite(output_path + str(results['img_id']) + '_decrease_contrast.png', image_decrease_contrast)
        # cv2.imwrite(output_path + str(results['img_id']) + '_increase_brightness.png', image_increase_brightness)
        # cv2.imwrite(output_path + str(results['img_id']) + '_decrease_brightness.png', image_decrease_brightness)

        new_image = np.concatenate((image, image_increase_contrast, image_decrease_contrast, image_increase_brightness, image_decrease_brightness), axis=2)
        results['img'] = new_image
        return results





import numpy as np
from typing import Dict, Optional, Union, Tuple, List

import torch
import cv2
from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomShift(BaseTransform):
    def __init__(self) -> None:
        super().__init__()

    def transform(self, results: Dict):
        # torch.save(results, '/home/jianbingshen/guodongqian/CL-Detection2023-MMPose/randomshift_debug/before_results.pth')
        # 设置 padding 大小
        padding_size = np.random.randint(100, 300)
        padded_img = np.pad(results['img'], ((padding_size, padding_size), (padding_size, padding_size), (0, 0)), mode='constant')

        # 更新关键点坐标和图片
        new_keypoints = results['keypoints'].copy()
        new_keypoints[:, :, 0] += padding_size
        new_keypoints[:, :, 1] += padding_size


        # 确定关键点的最大和最小的 x 和 y 坐标
        min_x = np.min(new_keypoints[:, :, 0])
        max_x = np.max(new_keypoints[:, :, 0])
        min_y = np.min(new_keypoints[:, :, 1])
        max_y = np.max(new_keypoints[:, :, 1])


        # 确定裁剪的范围
        margin = 20
        crop_x_start = np.random.randint(0, min_x - margin)
        crop_x_end = np.random.randint(max_x + margin, padded_img.shape[1])
        crop_y_start = np.random.randint(0, min_y - margin)
        crop_y_end = np.random.randint(max_y + margin, padded_img.shape[0])

        # 裁剪图片
        cropped_img = padded_img[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

        # 更新关键点坐标
        new_keypoints[:, :, 0] -= crop_x_start
        new_keypoints[:, :, 1] -= crop_y_start
        results['img'] = cropped_img
        results['keypoints'] = new_keypoints
        results['bbox'] = np.array([[0, 0, cropped_img.shape[1], cropped_img.shape[0]]], dtype='float32')

        # 在图片上绘制转换坐标后的gt点
        # img_gt = cropped_img.copy()
        # keypoints = results['keypoints'][0].copy()
        # for i in range(0, len(keypoints)):
        #     x, y = keypoints[i][0], keypoints[i][1]
            # 在图片上绘制标注点
            # cv2.circle(img_gt, (int(x), int(y)), 6, (0, 255, 0), -1)
            # 在标注点旁边添加序号
            # cv2.putText(img_gt, str(i + 1), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 保存带有标注点的图片
        # if not cv2.imwrite('/home/jianbingshen/guodongqian/CL-Detection2023-MMPose/randomshift_debug/' + str(results['img_id']) + '.png', img_gt):
        #     print(f"Error saving image {self.output_dir}")

        # torch.save(results, '/home/jianbingshen/guodongqian/CL-Detection2023-MMPose/randomshift_debug/after_results.pth')
        return results

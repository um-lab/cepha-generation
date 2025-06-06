import numpy as np
from typing import Dict, Optional, Union, Tuple, List

import torch
import cv2
from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomRotation(BaseTransform):
    def __init__(self, rotation_range) -> None:
        super().__init__()
        self.rotation_range = rotation_range

    def transform(self, results: Dict):
        # torch.save(results, '/home/jianbingshen/guodongqian/CL-Detection2023-MMPose/augmentation_debug/before_results.pth')
        # 设置 padding 大小
        padding_size = 100
        padded_img = np.pad(results['img'], ((padding_size, padding_size), (padding_size, padding_size), (0, 0)),
                            mode='constant')

        # 更新关键点坐标和图片
        new_keypoints = results['keypoints'].copy()
        new_keypoints[:, :, 0] += padding_size
        new_keypoints[:, :, 1] += padding_size

        # 1. 从指定范围中随机选择旋转角度
        angle = np.random.randint(-self.rotation_range, self.rotation_range)

        # 2. 获取图像中心坐标
        # center = (results['img_shape'][1] // 2, results['img_shape'][0] // 2)
        center = (padded_img.shape[1] // 2, padded_img.shape[0] // 2)

        # 3. 使用OpenCV计算旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1)

        # 4. 旋转图片
        # rotated_image = cv2.warpAffine(results['img'], M, (results['img_shape'][1], results['img_shape'][0]))
        rotated_image = cv2.warpAffine(padded_img, M, (padded_img.shape[1], padded_img.shape[0]))

        # 5. 更新关键点坐标
        # keypoints = np.array([cv2.transform(
        #     np.array([[[results['keypoints'][0][i][0], results['keypoints'][0][i][1]]]]), M)[0][0]
        #                       for i in range(results['keypoints'].shape[1])])
        keypoints = np.array([cv2.transform(
            np.array([[[new_keypoints[0][i][0], new_keypoints[0][i][1]]]]), M)[0][0]
                              for i in range(new_keypoints.shape[1])])

        results['img'] = rotated_image
        results['keypoints'] = keypoints.reshape(1, *keypoints.shape)
        results['bbox'] = np.array([[0, 0, rotated_image.shape[1], rotated_image.shape[0]]], dtype='float32')

        # 在图片上绘制转换坐标后的gt点
        # img_gt = rotated_image.copy()
        # keypoints = results['keypoints'][0].copy()
        # for i in range(0, len(keypoints)):
        #     x, y = keypoints[i][0], keypoints[i][1]
            # 在图片上绘制标注点
            # cv2.circle(img_gt, (int(x), int(y)), 6, (0, 255, 0), -1)
            # 在标注点旁边添加序号
            # cv2.putText(img_gt, str(i + 1), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 保存带有标注点的图片
        # if not cv2.imwrite('/home/jianbingshen/guodongqian/CL-Detection2023-MMPose/randomrotation_debug/' + str(results['img_id']) + '.png', img_gt):
        #     print(f"Error saving image {self.output_dir}")

        # torch.save(results, '/home/jianbingshen/guodongqian/CL-Detection2023-MMPose/augmentation_debug/after_results.pth')
        return results
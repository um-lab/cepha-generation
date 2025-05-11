import os
from typing import Dict, Optional

import cv2
import numpy as np
from mmcv.transforms import BaseTransform

from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class Registration(BaseTransform):
    def __init__(self,
                 ref_points_index: list,
                 ref_points_coord: list,
                 ref_image_size: tuple,
                 output_dir: str,
                 train: bool,
                 noise: int = 10) -> None:
        super().__init__()

        self.ref_points_index = ref_points_index
        self.ref_points_coord = ref_points_coord
        self.ref_image_size = ref_image_size
        self.output_dir = output_dir
        self.train = train
        self.noise = noise


    def transform(self, results: Dict) -> Optional[dict]:
        output_path = os.path.join(self.output_dir, os.path.basename(results['img_path']))

        # Load images
        image = results['img']

        # Calculate transformation matrix
        src_points = np.float32([results['keypoints'][0][i] for i in self.ref_points_index])
        dst_points = np.float32(self.ref_points_coord)

        if self.train:
            offsets = np.random.randint(-10, 11, size=src_points.shape)
            src_points = src_points + offsets
            while np.any((src_points < 0) | (src_points[:, 0] >= image.shape[1]) | (src_points[:, 1] >= image.shape[0])):
                print("Some points are out of bounds!")
                src_points = src_points - offsets
                offsets = np.random.randint(-5, 6, size=src_points.shape)
                src_points = src_points + offsets

        M = cv2.estimateAffinePartial2D(src_points, dst_points)[0]
        aligned_img = cv2.warpAffine(image, M, self.ref_image_size, flags=cv2.INTER_CUBIC)


        # Update keypoints and bbox
        results['original_keypoints'] = results['keypoints']
        keypoints = np.array([cv2.transform(
            np.array([[[results['keypoints'][0][i][0], results['keypoints'][0][i][1]]]]), M)[0][0]
                              for i in range(results['keypoints'].shape[1])])
        bbox = np.array([[0, 0, aligned_img.shape[1], aligned_img.shape[0]]], dtype=np.float32)

        # 求逆变换
        # 生成3x3矩阵
        M_3_3 = np.vstack([M, [0, 0, 1]])
        # 求逆
        inverse_M = np.linalg.inv(M_3_3)
        # 将逆矩阵缩减到2x3形式以供 cv2.warpAffine 使用
        inverse_M = inverse_M[:2, :]

        # Update image information
        results['img'] = aligned_img
        results['keypoints'] = keypoints.reshape(1, *keypoints.shape)
        results['bbox'] = bbox
        results['M'] = inverse_M

        # 在对齐后的图片上绘制转换坐标后的gt点
        # img_gt = aligned_img.copy()
        # for i in range(0, len(keypoints)):
        #     x, y = keypoints[i][0], keypoints[i][1]
            # 在图片上绘制标注点
            # cv2.circle(img_gt, (int(x), int(y)), 3, (0, 255, 0), -1)
            # 在标注点旁边添加序号
            # cv2.putText(img_gt, str(i + 1), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 保存带有标注点的图片
        # if not cv2.imwrite(output_path, img_gt):
        #     print(f"Error saving image {self.output_dir}")

        return results
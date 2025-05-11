# !/usr/bin/env python
# -*- coding:utf-8 -*-

import warnings
from typing import Dict, Optional, Sequence, Union
import torch
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmpose.registry import METRICS
from ..functional import (keypoint_auc, keypoint_epe, keypoint_nme,
                          keypoint_pck_accuracy)


@METRICS.register_module()
class CephalometricMetric(BaseMetric):
    """Cephalometric evaluation metric.

    Calculate the Mean Radius Error of keypoints.

    Note:
        - length of dataset: N
        - num_keypoints: K
        - number of keypoint dimensions: D (typically D = 2)

    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be ``'cpu'`` or
            ``'gpu'``. Default: ``'cpu'``.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, ``self.default_prefix``
            will be used instead. Default: ``None``.
    """

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            # predicted keypoints coordinates, [1, K, D]
            pred_coords = data_sample['pred_instances']['keypoints']
            # ground truth data_info
            gt = data_sample['gt_instances']
            # spacing
            spacing = data_sample['spacing']
            # ground truth keypoints coordinates, [1, K, D]
            gt_coords = gt['keypoints']
            # ground truth keypoints_visible, [1, K, 1]
            mask = gt['keypoints_visible'].astype(bool).reshape(1, -1)

            result = {
                'pred_coords': pred_coords,
                'gt_coords': gt_coords,
                'mask': mask,
                'spacing': spacing
            }

            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # pred_coords: [N, K, D]
        pred_coords = np.concatenate([result['pred_coords'] for result in results])
        # gt_coords: [N, K, D]
        gt_coords = np.concatenate([result['gt_coords'] for result in results])
        # mask: [N, K]
        # mask = np.concatenate([result['mask'] for result in results])
        # spacing: [N, 1]
        spacing = np.asarray([[result['spacing']] for result in results])
        # torch.save([gt_coords, pred_coords, spacing], '/mnt/guodongqian/CL-Detection2023-MMPose/debug/new-cepha.pth')
        logger.info(f'Evaluating {self.__class__.__name__}...')

        # calculate the prediction keypoints error
        n_kpt = np.prod(np.shape(gt_coords)[:-1])
        each_kpt_error = np.sqrt(np.sum(np.square(pred_coords - gt_coords), axis=2)) * spacing

        # the mean radial error metric
        mre = np.sum(each_kpt_error) / n_kpt
        mre_std = np.std(each_kpt_error)
        mre_without_spacing = np.sum(np.sqrt(np.sum(np.square(pred_coords - gt_coords), axis=2))) / n_kpt

        # the success detection rate metric
        sdr2_0 = np.sum(each_kpt_error <= 2.0) / n_kpt * 100
        sdr2_5 = np.sum(each_kpt_error <= 2.5) / n_kpt * 100
        sdr3_0 = np.sum(each_kpt_error <= 3.0) / n_kpt * 100
        sdr4_0 = np.sum(each_kpt_error <= 4.0) / n_kpt * 100
        print('=> each_kpt_error: ')
        each_kpt_error = list(np.sum(each_kpt_error, axis=0, dtype='int') / each_kpt_error.shape[0])
        sorted_error = sorted(range(len(each_kpt_error)), key=lambda i: each_kpt_error[i])
        for index, value in enumerate(each_kpt_error, start=1):  # 注意，这里设置start=1，让序号从1开始
            print(f"{index}: {value}")
        print("min_5：")
        for i in range(5):
            index = sorted_error[i] + 1  # 因为序号从1开始，所以需要+1
            value = each_kpt_error[sorted_error[i]]
            print(f"{index}: {value}")

        # 输出最大的五个值
        print("max_5：")
        for i in range(1, 6):
            index = sorted_error[-i] + 1  # 因为序号从1开始，所以需要+1
            value = each_kpt_error[sorted_error[-i]]
            print(f"{index}: {value}")


        metrics = dict()
        metrics['MRE'] = mre
        metrics['MRE_without_spacing'] = mre_without_spacing
        metrics['SDR 2.0mm'] = sdr2_0
        metrics['SDR 2.5mm'] = sdr2_5
        metrics['SDR 3.0mm'] = sdr3_0
        metrics['SDR 4.0mm'] = sdr4_0

        print('=>mre_without_spacing：' + str(mre_without_spacing))
        print('=> {:<24} :  {} = {:0.3f} ± {:0.3f} mm'.format('Mean Radial Error', 'MRE', mre, mre_std))
        print('=> {:<24} :  SDR 2.0mm = {:0.3f}% | SDR 2.5mm = {:0.3f}% | SDR 3mm = {:0.3f}% | SDR 4mm = {:0.3f}%'
              .format('Success Detection Rate', sdr2_0, sdr2_5, sdr3_0, sdr4_0))

        return metrics







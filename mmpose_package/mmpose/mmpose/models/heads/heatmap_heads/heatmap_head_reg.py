# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmengine.structures import PixelData, InstanceData
from torch import Tensor, nn

from mmpose.evaluation.functional import pose_pck_accuracy
from mmpose.models.utils.tta import flip_heatmaps
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions, InstanceList)
from ..base_head import BaseHead
from ...losses.regression_loss import _transpose_and_gather_feat

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class HeatmapHeadReg(BaseHead):
    """Top-down heatmap head introduced in `Simple Baselines`_ by Xiao et al
    (2018). The head is composed of a few deconvolutional layers followed by a
    convolutional layer to generate heatmaps from low-resolution feature maps.

    Args:
        in_channels (int | Sequence[int]): Number of channels in the input
            feature map
        out_channels (int): Number of channels in the output heatmap
        deconv_out_channels (Sequence[int], optional): The output channel
            number of each deconv layer. Defaults to ``(256, 256, 256)``
        deconv_kernel_sizes (Sequence[int | tuple], optional): The kernel size
            of each deconv layer. Each element should be either an integer for
            both height and width dimensions, or a tuple of two integers for
            the height and the width dimension respectively.Defaults to
            ``(4, 4, 4)``
        conv_out_channels (Sequence[int], optional): The output channel number
            of each intermediate conv layer. ``None`` means no intermediate
            conv layer between deconv layers and the final conv layer.
            Defaults to ``None``
        conv_kernel_sizes (Sequence[int | tuple], optional): The kernel size
            of each intermediate conv layer. Defaults to ``None``
        final_layer (dict): Arguments of the final Conv2d layer.
            Defaults to ``dict(kernel_size=1)``
        heatmap_loss (Config): Config of the keypoint loss. Defaults to use
            :class:`KeypointMSELoss`
        offset_loss (Config):
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings
        extra (dict, optional): Extra configurations.
            Defaults to ``None``

    . _`Simple Baselines`: https://arxiv.org/abs/1804.06208
    """

    _version = 2

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 out_channels: int,
                 keypoint_nums: int,
                 ref_keypoints: dict,
                 deconv_out_channels: OptIntSeq = (256, 256, 256),
                 deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
                 conv_out_channels: OptIntSeq = None,
                 conv_kernel_sizes: OptIntSeq = None,
                 final_layer: dict = dict(kernel_size=1),
                 heatmap_loss: ConfigType = dict(type='KeypointMSELoss', use_target_weight=True),
                 offset_loss: ConfigType = dict(type='RegLoss'),
                 decoder: OptConfigType = None,
                 init_cfg: OptConfigType = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.keypoint_nums = keypoint_nums
        self.ref_keypoints = ref_keypoints
        self.heatmap_loss_module = MODELS.build(heatmap_loss)
        self.offset_loss_module = MODELS.build(offset_loss)
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        if deconv_out_channels:
            if deconv_kernel_sizes is None or len(deconv_out_channels) != len(
                    deconv_kernel_sizes):
                raise ValueError(
                    '"deconv_out_channels" and "deconv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {deconv_out_channels} and '
                    f'{deconv_kernel_sizes}')

            self.deconv_layers = self._make_deconv_layers(
                in_channels=in_channels,
                layer_out_channels=deconv_out_channels,
                layer_kernel_sizes=deconv_kernel_sizes,
            )
            in_channels = deconv_out_channels[-1]
        else:
            self.deconv_layers = nn.Identity()

        if conv_out_channels:
            if conv_kernel_sizes is None or len(conv_out_channels) != len(
                    conv_kernel_sizes):
                raise ValueError(
                    '"conv_out_channels" and "conv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {conv_out_channels} and '
                    f'{conv_kernel_sizes}')

            self.conv_layers = self._make_conv_layers(
                in_channels=in_channels,
                layer_out_channels=conv_out_channels,
                layer_kernel_sizes=conv_kernel_sizes)
            in_channels = conv_out_channels[-1]
        else:
            self.conv_layers = nn.Identity()

        if final_layer is not None:
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1)
            cfg.update(final_layer)
            self.final_layer = build_conv_layer(cfg)
        else:
            self.final_layer = nn.Identity()

        # Register the hook to automatically convert old version state dicts
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def _make_conv_layers(self, in_channels: int,
                          layer_out_channels: Sequence[int],
                          layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create convolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            padding = (kernel_size - 1) // 2
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding)
            layers.append(build_conv_layer(cfg))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _make_deconv_layers(self, in_channels: int,
                            layer_out_channels: Sequence[int],
                            layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create deconvolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise ValueError(f'Unsupported kernel size {kernel_size} for'
                                 'deconvlutional layers in '
                                 f'{self.__class__.__name__}')
            cfg = dict(
                type='deconv',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False)
            layers.append(build_upsample_layer(cfg))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(
                type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1)
        ]
        return init_cfg

    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the heatmap.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tensor: output heatmap.
        """
        x = feats[-1]

        x = self.deconv_layers(x)
        x = self.conv_layers(x)
        x = self.final_layer(x)

        return x

    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-stage features in TTA)
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}

        Returns:
            Union[InstanceList | Tuple[InstanceList | PixelDataList]]: If
            ``test_cfg['output_heatmap']==True``, return both pose and heatmap
            prediction; otherwise only return the pose prediction.

            The pose prediction is a list of ``InstanceData``, each contains
            the following fields:

                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)

            The heatmap prediction is a list of ``PixelData``, each contains
            the following fields:

                - heatmaps (Tensor): The predicted heatmaps in shape (K, h, w)
        """

        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats
            _batch_heatmaps = self.forward(_feats)
            _batch_heatmaps_flip = flip_heatmaps(
                self.forward(_feats_flip),
                flip_mode=test_cfg.get('flip_mode', 'heatmap'),
                flip_indices=flip_indices,
                shift_heatmap=test_cfg.get('shift_heatmap', False))
            batch_heatmaps = (_batch_heatmaps + _batch_heatmaps_flip) * 0.5
        else:
            batch_combined = self.forward(feats)

        preds = self.decode(batch_combined)

        if test_cfg.get('output_heatmaps', False):
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_combined.detach()
            ]
            return preds, pred_fields
        else:
            return preds

    def decode(self, heatmaps: Tuple[Tensor]) -> InstanceList:
        if self.decoder is None:
            raise RuntimeError(
                f'The decoder has not been set in {self.__class__.__name__}. '
                'Please set the decoder configs in the init parameters to '
                'enable head methods `head.predict()` and `head.decode()`')

        # 将点的序号转化为下标
        ref_keypoints_index = {}
        for key, value_list in self.ref_keypoints.items():
            # 对键和值都减一，并添加到新字典中
            new_key = key - 1
            new_value = [val - 1 for val in value_list]
            ref_keypoints_index[new_key] = new_value

        batch_size = heatmaps.shape[0]
        batch_heatmaps = heatmaps[:, :self.keypoint_nums, :, :]  # heatmap部分
        batch_offsets = heatmaps[:, self.keypoint_nums:, :, :]  # offset部分
        batch_offsets = batch_offsets.cpu()
        batch_heatmaps_np = to_numpy(batch_heatmaps, unzip=True)
        # batch_offsets_np = to_numpy(batch_offsets, unzip=True)
        batch_keypoints = []
        batch_scores = []
        batch_predicted_points = [] # test
        batch_ind = [] # test
        batch_keypoints_original = [] # test


        for b in range(batch_size):
            # keypoints是在input_size尺度下的预测坐标值，keypoints_original是在heatmap_size尺度下的坐标值，两者相差一个scale_factor
            keypoints, scores, keypoints_original = self.decoder.decode(batch_heatmaps_np[b], reg=True)
            keypoints_original = np.round(keypoints_original)

            all_ind = []
            all_ref_points = []
            all_predicted_points = []
            # todo:修改为多目标点的regression
            for key in ref_keypoints_index:
                ind = []
                for j in ref_keypoints_index[key]:
                    ct = keypoints_original[0][j]
                    ct_int = ct.astype(np.int32)
                    ind.append(ct_int[1] * batch_heatmaps_np[b].shape[1] + ct_int[0])
                ind = torch.tensor(ind)
                all_ind.append(ind)
                all_ref_points.append(torch.tensor(keypoints[0][ref_keypoints_index[key]]))  # 参考点的预测坐标

            for key, ind, ref_points, i in zip(ref_keypoints_index, all_ind, all_ref_points, range(0, batch_offsets.shape[1], 2)):
                offset = batch_offsets[b, i:i + 2]
                offset = offset.permute(1, 2, 0).contiguous()
                offset = offset.view(-1, 2)
                ind = ind.unsqueeze(1).expand(ind.size(0), 2)
                offset = offset.gather(0, ind)
                # 将归一化的偏移量转换为在input_size下的偏移量
                offset[:, 0] *= self.decoder.input_size[0]
                offset[:, 1] *= self.decoder.input_size[1]
                # 参考点偏移量取平均值计算得到目标点坐标
                predicted_points = ref_points + offset
                target_point = torch.mean(predicted_points, dim=0)
                predicted_points = to_numpy(predicted_points)
                all_predicted_points.append(predicted_points)
                # 取regression的结果作为最终预测
                keypoints[0, key] = target_point.numpy()
                # 取heatmap和regression的结果进行平均作为最终预测
                # keypoints[0, key] = (target_point.numpy() + keypoints[0, key]) / 2

            all_predicted_points = np.array(all_predicted_points)
            batch_predicted_points.append(all_predicted_points.reshape(1, *all_predicted_points.shape))
            batch_ind.append(all_ind)
            batch_keypoints.append(keypoints)
            batch_scores.append(scores)
            batch_keypoints_original.append(keypoints_original)

        preds = [InstanceData(keypoints=keypoints, keypoint_scores=scores, reg_results=predicted_points)
                 for keypoints, scores, predicted_points in zip(batch_keypoints, batch_scores, batch_predicted_points)]
        # torch.save([preds, heatmaps, batch_ind, batch_keypoints_original], '/home/jianbingshen/guodongqian/CL-Detection2023-MMPose/debug_reg/HeatmapHeadReg.pth')

        return  preds


    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            feats (Tuple[Tensor]): The multi-stage features
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            train_cfg (dict): The runtime config for training process.
                Defaults to {}

        Returns:
            dict: A dictionary of losses.
        """
        pred_fields_combined = self.forward(feats)  # 包含了全部map的tensor，形状为(N,C,W,H)，(N,38,W,H)为heatmap，剩余的channels表示offset
        pred_fields_heatmap = pred_fields_combined[:, :self.keypoint_nums, :, :]  # heatmap部分
        pred_fields_offset = pred_fields_combined[:, self.keypoint_nums:, :, :]  # offset部分

        gt_heatmaps = torch.stack([d.gt_fields.heatmaps for d in batch_data_samples])
        gt_offsets = torch.cat([d.gt_instance_labels.offsets for d in batch_data_samples])
        keypoint_weights = torch.cat([d.gt_instance_labels.keypoint_weights for d in batch_data_samples])
        ind = torch.cat([d.gt_instance_labels.ind for d in batch_data_samples])


        # calculate losses
        losses = dict()
        heatmap_loss = self.heatmap_loss_module(pred_fields_heatmap, gt_heatmaps, keypoint_weights)
        offest_loss = []
        for i, j in zip(range(0, pred_fields_offset.shape[1], 2), range(len(self.ref_keypoints))):
            offest_loss.append(self.offset_loss_module(pred_fields_offset[:, i:i+2, :, :], ind[:, j, :], gt_offsets[:, j, :, :]))
        stacked_offest_loss = torch.stack(offest_loss, dim=0)
        offest_loss = torch.mean(stacked_offest_loss)


        losses.update(loss_kpt=heatmap_loss, loss_offest=offest_loss)

        # calculate accuracy
        if train_cfg.get('compute_acc', True):
            _, avg_acc, _ = pose_pck_accuracy(
                output=to_numpy(pred_fields_heatmap),
                target=to_numpy(gt_heatmaps),
                mask=to_numpy(keypoint_weights) > 0)

            acc_pose = torch.tensor(avg_acc, device=gt_heatmaps.device)
            losses.update(acc_pose=acc_pose)

        return losses

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                                  **kwargs):
        """A hook function to convert old-version state dict of
        :class:`DeepposeRegressionHead` (before MMPose v1.0.0) to a
        compatible format of :class:`RegressionHead`.

        The hook will be automatically registered during initialization.
        """
        version = local_meta.get('version', None)
        if version and version >= self._version:
            return

        # convert old-version state dict
        keys = list(state_dict.keys())
        for _k in keys:
            if not _k.startswith(prefix):
                continue
            v = state_dict.pop(_k)
            k = _k[len(prefix):]
            # In old version, "final_layer" includes both intermediate
            # conv layers (new "conv_layers") and final conv layers (new
            # "final_layer").
            #
            # If there is no intermediate conv layer, old "final_layer" will
            # have keys like "final_layer.xxx", which should be still
            # named "final_layer.xxx";
            #
            # If there are intermediate conv layers, old "final_layer"  will
            # have keys like "final_layer.n.xxx", where the weights of the last
            # one should be renamed "final_layer.xxx", and others should be
            # renamed "conv_layers.n.xxx"
            k_parts = k.split('.')
            if k_parts[0] == 'final_layer':
                if len(k_parts) == 3:
                    assert isinstance(self.conv_layers, nn.Sequential)
                    idx = int(k_parts[1])
                    if idx < len(self.conv_layers):
                        # final_layer.n.xxx -> conv_layers.n.xxx
                        k_new = 'conv_layers.' + '.'.join(k_parts[1:])
                    else:
                        # final_layer.n.xxx -> final_layer.xxx
                        k_new = 'final_layer.' + k_parts[2]
                else:
                    # final_layer.xxx remains final_layer.xxx
                    k_new = k
            else:
                k_new = k

            state_dict[prefix + k_new] = v

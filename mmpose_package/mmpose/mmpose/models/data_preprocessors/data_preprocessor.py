# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import math
import torch
import torch.nn.functional as F

from mmengine import is_seq_of
from mmengine.model import ImgDataPreprocessor, stack_batch

from mmpose.registry import MODELS


@MODELS.register_module()
class PoseDataPreprocessor(ImgDataPreprocessor):
    """Image pre-processor for pose estimation tasks."""

@MODELS.register_module()
class MultiChannelsDataPreprocessor(ImgDataPreprocessor):
    def forward(self, data: dict, training: bool = False) -> Union[dict, list]:
        data = self.cast_data(data)  # type: ignore
        _batch_inputs = data['inputs']
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_inputs = []
            for _batch_input in _batch_inputs:
                chunks = torch.chunk(_batch_input, 5, dim=0)
                processed_chunks = []
                for chunk in chunks:
                    # channel transform
                    if self._channel_conversion:
                        chunk = chunk[[2, 1, 0], ...]
                        # Convert to float after channel conversion to ensure efficiency
                        chunk = chunk.float()
                        # Normalization.
                        if self._enable_normalize:
                            if self.mean.shape[0] == 3:
                                assert chunk.dim() == 3 and chunk.shape[0] == 3, (
                                    'If the mean has 3 values, the input tensor '
                                    'should in shape of (3, H, W), but got the tensor '
                                    f'with shape {chunk.shape}')
                            chunk = (chunk - self.mean) / self.std
                    processed_chunks.append(chunk)
                _batch_input = torch.cat(processed_chunks, dim=0)
                batch_inputs.append(_batch_input)
            # Pad and stack Tensor.
            batch_inputs = stack_batch(batch_inputs, self.pad_size_divisor, self.pad_value)
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            chunks = torch.chunk(_batch_inputs, 5, dim=0)
            processed_chunks = []
            for chunk in chunks:
                if self._channel_conversion:
                    chunk = chunk[:, [2, 1, 0], ...]
                # Convert to float after channel conversion to ensure efficiency
                chunk = chunk.float()
                if self._enable_normalize:
                    chunk = (chunk - self.mean) / self.std
                h, w = chunk.shape[2:]
                target_h = math.ceil(h / self.pad_size_divisor) * self.pad_size_divisor
                target_w = math.ceil(w / self.pad_size_divisor) * self.pad_size_divisor
                pad_h = target_h - h
                pad_w = target_w - w
                chunk = F.pad(chunk, (0, pad_w, 0, pad_h), 'constant', self.pad_value)
                processed_chunks.append(chunk)
            _batch_inputs = torch.cat(processed_chunks, dim=0)
        else:
            raise TypeError('Output of `cast_data` should be a dict of '
                            'list/tuple with inputs and data_samples, '
                            f'but got {type(data)}： {data}')
        data['inputs'] = batch_inputs
        data.setdefault('data_samples', None)
        return data


@MODELS.register_module()
class CannyChannelsDataPreprocessor(ImgDataPreprocessor):
    def forward(self, data: dict, training: bool = False) -> Union[dict, list]:
        data = self.cast_data(data)  # type: ignore
        _batch_inputs = []
        _batch_inputs_canny = []
        for input in data['inputs']:
            _batch_inputs.append(input[:3, :, :])
            _batch_inputs_canny.append(input[3:, :, :])
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_inputs = []
            for _batch_input, _batch_input_canny  in zip(_batch_inputs, _batch_inputs_canny):
                # channel transform
                if self._channel_conversion:
                    _batch_input = _batch_input[[2, 1, 0], ...]
                # Convert to float after channel conversion to ensure
                # efficiency
                _batch_input = _batch_input.float()
                _batch_input_canny = _batch_input_canny.float()
                # Normalization.
                if self._enable_normalize:
                    if self.mean.shape[0] == 3:
                        assert _batch_input.dim(
                        ) == 3 and _batch_input.shape[0] == 3, (
                            'If the mean has 3 values, the input tensor '
                            'should in shape of (3, H, W), but got the tensor '
                            f'with shape {_batch_input.shape}')
                    _batch_input = (_batch_input - self.mean) / self.std
                    _batch_input_canny = _batch_input_canny / 255.0
                    _batch_input = torch.cat((_batch_input, _batch_input_canny), dim=0)
                batch_inputs.append(_batch_input)
            # Pad and stack Tensor.
            batch_inputs = stack_batch(batch_inputs, self.pad_size_divisor,
                                       self.pad_value)
        data['inputs'] = batch_inputs
        data.setdefault('data_samples', None)
        return data


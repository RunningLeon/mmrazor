# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from mmrazor.models.architectures.dynamic_ops.bricks.dynamic_conv import \
    DynamicConv2d
from mmrazor.models.architectures.dynamic_ops.bricks.dynamic_linear import \
    DynamicLinear
from mmrazor.models.task_modules.estimators.counters import (Conv2dCounter,
                                                             LinearCounter)
from mmrazor.registry import TASK_UTILS


class GroupFisherMixin:
    """The mixin class for GroupFisher ops."""

    def _init(self) -> None:
        self.handlers: list = []
        self.recorded_input: List = []
        self.recorded_grad: List = []
        self.recorded_out_shape: List = []

    def forward_hook_wrapper(self):
        """Wrap the hook used in forward."""

        def forward_hook(module: GroupFisherMixin, input, output):
            module.recorded_out_shape.append(output.shape)
            module.recorded_input.append(input[0])

        return forward_hook

    def backward_hook_wrapper(self):
        """Wrap the hook used in backward."""

        def backward_hook(module: GroupFisherMixin, grad_in, grad_out):
            module.recorded_grad.insert(0, grad_in[0])

        return backward_hook

    def start_record(self: torch.nn.Module) -> None:
        """Start recording information during forward and backward."""
        self.end_record()  # ensure to run start_record only once
        self.handlers.append(
            self.register_forward_hook(self.forward_hook_wrapper()))
        self.handlers.append(
            self.register_backward_hook(self.backward_hook_wrapper()))

    def end_record(self):
        """Stop recording information during forward and backward."""
        for handle in self.handlers:
            handle.remove()

    def reset_recorded(self):
        """Reset the recorded information."""
        self.recorded_input = []
        self.recorded_grad = []
        self.recorded_out_shape = []

    @property
    def delta_flop_of_a_out_channel(self):
        raise NotImplementedError()

    @property
    def delta_flop_of_a_in_channel(self):
        raise NotImplementedError()

    @property
    def delta_memory_of_a_out_channel(self):
        raise NotImplementedError()


class GroupFisherConv2d(DynamicConv2d, GroupFisherMixin):
    """The Dynamic Conv2d operation used in GroupFisher Algorithm."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init()

    @property
    def delta_flop_of_a_out_channel(self) -> torch.Tensor:
        """Calculate the summation of flops when prune an out_channel."""
        delta_flop_sum = 0
        for shape in self.recorded_out_shape:
            _, _, h, w = shape
            in_c = self.mutable_attrs['in_channels'].current_mask.float().sum()
            delta_flop = h * w * self.kernel_size[0] * self.kernel_size[
                1] * in_c
            delta_flop_sum += delta_flop
        return delta_flop_sum

    @property
    def delta_flop_of_a_in_channel(self):
        """Calculate the summation of flops when prune an in_channel."""
        delta_flop_sum = 0
        for shape in self.recorded_out_shape:
            _, out_c, h, w = shape
            delta_flop = out_c * h * w * self.kernel_size[
                0] * self.kernel_size[1]
            delta_flop_sum += delta_flop
        return delta_flop_sum

    @property
    def delta_memory_of_a_out_channel(self):
        """Calculate the summation of memory when prune a channel."""
        delta_flop_sum = 0
        for shape in self.recorded_out_shape:
            _, _, h, w = shape
            delta_flop_sum += h * w
        return delta_flop_sum


class GroupFisherLinear(DynamicLinear, GroupFisherMixin):
    """The Dynamic Linear operation used in GroupFisher Algorithm."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init()

    @property
    def delta_flop_of_a_out_channel(self):
        """Calculate the summation of flops when prune an out_channel."""
        in_c = self.mutable_attrs['in_channels'].current_mask.float().sum()
        return in_c * len(self.recorded_out_shape)

    @property
    def delta_flop_of_a_in_channel(self):
        """Calculate the summation of flops when prune an in_channel."""
        out_c = self.mutable_attrs['out_channels'].current_mask.float().sum()
        return out_c * len(self.recorded_out_shape)

    @property
    def delta_memory_of_a_out_channel(self):
        """Calculate the summation of memory when prune a channel."""
        return 1 * len(self.recorded_out_shape)


@TASK_UTILS.register_module()
class DynamicConv2dCounter(Conv2dCounter):
    """The counter of GroupFisherConv2d."""

    @staticmethod
    def add_count_hook(module: nn.Conv2d, input: Tuple[torch.Tensor],
                       output: torch.Tensor) -> None:
        """Count the flops and params of a DynamicConv2d.

        Args:
            module (nn.Conv2d): A Conv2d module.
            input (Tuple[torch.Tensor]): Input of this module.
            output (torch.Tensor): Output of this module.
        """
        batch_size = input[0].shape[0]
        output_dims = list(output.shape[2:])

        kernel_dims = list(module.kernel_size)

        out_channels = module.mutable_attrs['out_channels'].activated_channels
        in_channels = module.mutable_attrs['in_channels'].activated_channels

        groups = module.groups

        filters_per_channel = out_channels / groups
        conv_per_position_flops = int(
            np.prod(kernel_dims)) * in_channels * filters_per_channel

        active_elements_count = batch_size * int(np.prod(output_dims))

        overall_conv_flops = conv_per_position_flops * active_elements_count
        overall_params = conv_per_position_flops

        bias_flops = 0
        overall_params = conv_per_position_flops
        if module.bias is not None:
            bias_flops = out_channels * active_elements_count
            overall_params += out_channels

        overall_flops = overall_conv_flops + bias_flops

        module.__flops__ += overall_flops
        module.__params__ += int(overall_params)


@TASK_UTILS.register_module()
class GroupFisherConv2dCounter(DynamicConv2dCounter):
    """Counter of GroupFisherConv2d."""
    pass


@TASK_UTILS.register_module()
class GroupFisherLinearCounter(LinearCounter):
    """Counter of GroupFisherLinear."""
    pass

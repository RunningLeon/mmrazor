# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn
from mmengine.model.utils import _BatchNormXd
from mmengine.utils.dl_utils.parrots_wrapper import \
    SyncBatchNorm as EngineSyncBatchNorm
from torch import distributed as dist

import mmrazor.models.architectures.dynamic_ops as dynamic_ops
from mmrazor.models.mutables.mutable_channel.mutable_channel_container import \
    MutableChannelContainer
from mmrazor.models.mutables.mutable_channel.units.l1_mutable_channel_unit import \
    L1MutableChannelUnit  # noqa
from mmrazor.registry import MODELS
from .group_fisher_ops import (GroupFisherConv2d, GroupFisherLinear,
                               GroupFisherMixin)


@MODELS.register_module()
class GroupFisherChannelUnit(L1MutableChannelUnit):
    """ChannelUnit for GroupFisher Pruning Algorithm."""

    # init

    def __init__(
        self,
        num_channels: int,
        choice_mode='number',
        divisor=1,
        min_value=1,
        min_ratio=0.9,
        detla_type='flop',
    ) -> None:
        super().__init__(num_channels, choice_mode, divisor, min_value,
                         min_ratio)
        _fisher_info = torch.zeros([self.num_channels])
        self.register_buffer('fisher_info', _fisher_info)
        self.fisher_info: torch.Tensor

        self.hook_handles: List = []

        assert detla_type in ['flop', 'act', 'none']
        self.delta_type = detla_type

    def prepare_for_pruning(self, model: nn.Module):
        """Prepare for pruning, including register mutable channels."""
        # register MutableMask
        self._replace_with_dynamic_ops(
            model, {
                nn.Conv2d: GroupFisherConv2d,
                nn.BatchNorm2d: dynamic_ops.DynamicBatchNorm2d,
                nn.Linear: GroupFisherLinear,
                nn.SyncBatchNorm: dynamic_ops.DynamicSyncBatchNorm,
                EngineSyncBatchNorm: dynamic_ops.DynamicSyncBatchNorm,
                _BatchNormXd: dynamic_ops.DynamicBatchNormXd,
            })
        self._register_channel_container(model, MutableChannelContainer)
        self._register_mutable_channel(self.mutable_channel)

    # prune

    def try_to_prune_min_fisher(self) -> bool:
        """Prune the channel.

        Args:
            channel (int): Index of the pruned channel.
        """
        if self.mutable_channel.activated_channels > 1:
            fisher = self.normalized_fisher_info
            index = fisher.argmin()
            self.mutable_channel.mask.scatter_(0, index, 0.0)
            return True
        else:
            return False

    # fisher information recorded

    def start_record_fisher_info(self):
        for channel in self.input_related + self.output_related:
            module = channel.module
            if isinstance(module, GroupFisherMixin):
                module.start_record()

    def end_record_fisher_info(self):
        for channel in self.input_related + self.output_related:
            module = channel.module
            if isinstance(module, GroupFisherMixin):
                module.end_record()

    def reset_recorded(self):
        for channel in self.input_related + self.output_related:
            module = channel.module
            if isinstance(module, GroupFisherMixin):
                module.reset_recorded()

    # fisher related computation

    def reset_fisher_info(self):
        self.fisher_info.zero_()

    @torch.no_grad()
    def update_fisher_info(self):
        for channel in self.input_related:
            module = channel.module
            if isinstance(module, GroupFisherMixin):
                batch_fisher = self.current_batch_fisher
                self.fisher_info += batch_fisher
        if dist.is_initialized():
            dist.all_reduce(self.fisher_info)

    @property
    def normalized_fisher_info(self):
        return self._get_normalized_fisher_info(normal_type=self.delta_type)

    @property
    def current_batch_fisher(self):
        with torch.no_grad():
            fisher = 0
            for channel in self.input_related:
                if isinstance(channel.module, GroupFisherMixin):
                    fisher = fisher + self._fisher_of_a_module(
                        channel.module, channel.start, channel.end)
            return (fisher**2).sum(0)

    @torch.no_grad()
    def _fisher_of_a_module(self, module, start: int, end: int):
        assert isinstance(module, GroupFisherMixin)
        assert len(module.recorded_input) > 0 and \
            len(module.recorded_input) == len(module.recorded_grad)
        fisher_sum: torch.Tensor = 0
        for input, grad_input in zip(module.recorded_input,
                                     module.recorded_grad):
            fisher: torch.Tensor = input * grad_input
            fisher = fisher.sum(dim=[i for i in range(2, len(fisher.shape))])
            fisher_sum = fisher_sum + fisher

        # expand to full num_channel
        batch_size = fisher_sum.shape[0]
        mask = self.mutable_channel.current_mask.unsqueeze(0).expand(
            [batch_size, self.num_channels])
        zeros = fisher_sum.new_zeros([batch_size, self.num_channels])
        fisher_sum = zeros.masked_scatter_(mask, fisher_sum)
        return fisher_sum

    @property
    def _delta_flop_of_a_channel(self):
        delta_flop = 0
        for channel in self.output_related:
            if isinstance(channel.module, GroupFisherMixin):
                delta_flop += channel.module.delta_flop_of_a_out_channel
        for channel in self.input_related:
            if isinstance(channel.module, GroupFisherMixin):
                delta_flop += channel.module.delta_flop_of_a_in_channel
        return delta_flop

    @property
    def _delta_memory_of_a_channel(self):
        delta_memory = 0
        for channel in self.output_related:
            if isinstance(channel.module, GroupFisherMixin):
                delta_memory += channel.module.delta_memory_of_a_out_channel
        return delta_memory

    def _get_normalized_fisher_info(self, normal_type='flop'):
        fisher = self.fisher_info.double()
        mask = self.mutable_channel.current_mask
        n_mask = (1 - mask.float()).bool()
        fisher.masked_fill_(n_mask, fisher.max() + 1)
        if normal_type == 'flop':
            delta_flop = self._delta_flop_of_a_channel
            fisher = fisher / (float(delta_flop) / 1e9)
        elif normal_type == 'act':
            delta_memory = self._delta_memory_of_a_channel
            fisher = fisher / (delta_memory / 1e6)
        elif normal_type == 'none':
            pass
        else:
            raise NotImplementedError(normal_type)
        return fisher

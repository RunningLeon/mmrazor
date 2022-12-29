# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Union

import torch
import torch.nn as nn

from mmrazor.registry import MODELS
from .l1_mutable_channel_unit import L1MutableChannelUnit


@MODELS.register_module()
class GroupFisherChannelUnit(L1MutableChannelUnit):
    """ChannelUnit for GroupFisher Pruning Algorithm."""

    def find_pruned_channel(self, accum_fisher: torch.Tensor, delta: int,
                            info_min: float) -> Dict:
        """Find the channel with the minimum fisher information to be pruned.

        Args:
            accum_fisher (torch.Tensor): Fisher information of this unit.
            delta (int): Delta value of this unit.
            info_min (float): The current minimum value of fisher information.
        """
        group = self.name
        in_mask = self.mutable_channel.current_mask
        fisher = accum_fisher.double()
        fisher /= float(delta / 1e9)
        module_info = self.update_module_info(group, fisher, in_mask, info_min)

        return module_info

    def update_module_info(self, module: Union[nn.Module, int],
                           fisher: torch.Tensor, in_mask: torch.Tensor,
                           info_min: torch.Tensor) -> Dict:
        """Update the information of module to find the channel to be pruned.

        Args:
            module (Union[nn.Module | int] ): Conv module of model or index of
                self.group.
            fisher(torch.Tensor): the fisher information of module's in_mask.
            in_mask (torch.Tensor): the squeeze in_mask of modules.
            info (Dict): store the channel of which module need to pruning.
                module: the module has channel need to pruning.
                channel: the index of channel need to pruning.
                min : the value of fisher / delta.
        """
        module_info = {}
        if fisher.sum() > 0 and in_mask.sum() > 0:
            nonzero = in_mask.nonzero().view(-1)
            fisher = fisher[nonzero]
            min_value, argmin = fisher.min(dim=0)
            if min_value < info_min:
                module_info['module'] = module
                module_info['channel'] = nonzero[argmin]
                module_info['min'] = min_value
        return module_info

    def prune(self, channel: int) -> None:
        """Prune the channel.

        Args:
            channel (int): Index of the pruned channel.
        """
        cur_mask = self.mutable_channel.current_mask
        cur_mask[channel] = False
        self.mutable_channel.current_choice = cur_mask

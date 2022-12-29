# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Dict, Tuple, Type, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from mmengine.logging import print_log

from mmrazor.models.mutables import L1MutableChannelUnit
from mmrazor.registry import MODELS
from .channel_mutator import ChannelMutator, ChannelUnitType


@MODELS.register_module()
class GroupFisherChannelMutator(ChannelMutator[L1MutableChannelUnit]):
    """Channel mutator for GroupFisher pruning algorithm.

    Args:
        channel_unit_cfg (Union[dict, Type[ChannelUnitType]], optional):
            Config of MutableChannelUnits. Defaults to
            dict(type='L1MutableChannelUnit',
                 default_args=dict(choice_mode='ratio')).
        parse_cfg (Dict): The config of the tracer to parse the model.
            Defaults to dict(type='ChannelAnalyzer',
                             demo_input=(1, 3, 224, 224),
                             tracer_type='FxTracer').
        batch_size (int): The batch_size when pruning model. Defaults to 2.
    """

    def __init__(self,
                 channel_unit_cfg: Union[dict, Type[ChannelUnitType]] = dict(
                     type='L1MutableChannelUnit',
                     default_args=dict(choice_mode='ratio')),
                 parse_cfg: Dict = dict(
                     type='ChannelAnalyzer',
                     demo_input=(1, 3, 224, 224),
                     tracer_type='FxTracer'),
                 batch_size: int = 2,
                 **kwargs) -> None:
        super().__init__(channel_unit_cfg, parse_cfg, **kwargs)

        self.batch_size = batch_size

    def _map_conv_name(self,
                       named_modules: Dict[str, nn.ModuleDict]) -> OrderedDict:
        """Map the conv modules with their names.

        Args:
            named_modules (Dict[str, nn.ModuleDict]): named_modules of the
                architecture.
        """
        conv2name = OrderedDict()
        for unit in self.units:
            for in_channel in unit.input_related:
                src_module = in_channel.name
                if src_module not in named_modules:
                    continue
                module = named_modules[src_module]
                if isinstance(module, nn.Conv2d):
                    conv2name[module] = src_module

            for out_channel in unit.output_related:
                src_module = out_channel.name
                if src_module not in named_modules:
                    continue
                module = named_modules[src_module]
                if isinstance(module, nn.Conv2d):
                    conv2name[module] = src_module
        return conv2name

    def _register_hooks(self, named_modules: Dict[str, nn.ModuleDict]) -> None:
        """Register forward and backward hook to Conv module.

        Args:
            named_modules (Dict[str, nn.ModuleDict]): named_modules of the
                architecture.
        """
        for unit in self.units:
            for in_channel in unit.input_related:
                src_module = in_channel.name
                if src_module not in named_modules:
                    continue

                module = named_modules[src_module]
                if not isinstance(module, nn.Conv2d):
                    continue
                module.register_forward_hook(self._save_input_forward_hook)
                module.register_backward_hook(
                    self._compute_fisher_backward_hook)

    def _save_input_forward_hook(self, module: nn.Module,
                                 inputs: Tuple[torch.Tensor],
                                 outputs: torch.Tensor) -> None:
        """Save the input and flops and acts for computing fisher and flops or
        acts.

        Args:
            module (nn.Module): module of the register hook.
            inputs (Tuple[torch.Tensor]): input of the module.
            outputs (torch.Tensor): output of the module.
        """
        n, oc, oh, ow = outputs.shape
        ic = module.in_channels // module.groups
        kh, kw = module.kernel_size
        self.flops[module] += np.prod([n, oc, oh, ow, ic, kh, kw])
        self.acts[module] += np.prod([n, oc, oh, ow])
        # a conv module may has several inputs in graph,for example
        # head in Retinanet
        if inputs[0].requires_grad:
            self.conv_inputs[module].append(inputs)

    def _compute_fisher_backward_hook(self, module: nn.Module,
                                      grad_input: tuple, *args) -> None:
        """Compute the fisher information of each module during backward.

        Args:
            module (nn.Module): module of the register hook.
            grad_input (tuple): tuple contains grad of input and parameters,
                grad_input[0]is the grad of input in Pytorch 1.3, it seems
                has changed in higher version.
        """

        def compute_fisher(input, grad_input):
            grads = input * grad_input
            grads = grads.sum(-1).sum(-1)
            return grads

        if module in self.conv_names and grad_input[0] is not None:
            feature = self.conv_inputs[module].pop(-1)[0]
            grad_feature = grad_input[0]
            # avoid that last batch is't full,
            # but actually it's always full in mmdetection.
            cur_mask = module.mutable_attrs.in_channels.current_mask
            cur_mask = cur_mask.to(feature.device)
            self.temp_fisher_info[module] = self.temp_fisher_info[module].to(
                feature.device)
            self.temp_fisher_info[module][:grad_input[0].size(0),
                                          cur_mask] += compute_fisher(
                                              feature, grad_feature)

    def init_flops_acts(self) -> None:
        """Clear the flops and acts of model in each iter."""
        for module, _ in self.conv_names.items():
            self.flops[module] = 0
            self.acts[module] = 0

    def init_temp_fishers(self) -> None:
        """Clear fisher info of single conv and group."""
        for module, _ in self.conv_names.items():
            self.temp_fisher_info[module].zero_()
        for unit in self.units:
            group = unit.name
            self.temp_fisher_info[group].zero_()

    def init_group_infos(self, module_dict):
        self.module_dict = module_dict
        self.conv_names = self._map_conv_name(module_dict)
        # The key of self.conv_inputs is conv module, and value of it
        # is list of conv's input_features in forward process
        self.conv_inputs: Dict[nn.Module, list] = {}
        # The key of self.flops is conv module, and value of it
        # is the summation of conv's flops in forward process
        self.flops: Dict[nn.Module, int] = {}
        # The key of self.acts is conv module, and value of it
        # is number of all the out feature's activations(N*C*H*W)
        # in forward process
        self.acts: Dict[nn.Module, int] = {}
        # The key of self.temp_fisher_info is conv module, and value
        # is a temporary variable used to estimate fisher.
        self.temp_fisher_info: Dict[nn.Module, torch.Tensor] = {}
        # The key of self.batch_fishers is conv module, and value
        # is the estimation of fisher by single batch.
        self.batch_fishers: Dict[nn.Module, torch.Tensor] = {}
        # The key of self.accum_fishers is conv module, and value
        # is the estimation of parameter's fisher by all the batch
        # during number of self.interval iterations.
        self.accum_fishers: Dict[nn.Module, torch.Tensor] = {}

        # Init fisher info for all convs.
        for conv, _ in self.conv_names.items():
            self.conv_inputs[conv] = []
            self.temp_fisher_info[conv] = conv.weight.data.new_zeros(
                self.batch_size, conv.in_channels)
            self.accum_fishers[conv] = conv.weight.data.new_zeros(
                conv.in_channels)

        # Init fisher info for all units (or called groups).
        self.current_unit_channel = dict()
        for unit in self.units:
            group = unit.name
            self.current_unit_channel[
                group] = unit.mutable_channel.num_channels
            self.temp_fisher_info[group] = conv.weight.data.new_zeros(
                self.batch_size, unit.mutable_channel.num_channels)
            self.accum_fishers[group] = conv.weight.data.new_zeros(
                unit.mutable_channel.num_channels)

        self._register_hooks(self.module_dict)
        self.init_flops_acts()
        self.init_temp_fishers()

    def find_pruning_channel(self, module: Union[nn.Module,
                                                 int], fisher: torch.Tensor,
                             in_mask: torch.Tensor, info: Dict) -> Dict:
        """Find the the channel of a model to pruning.

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
            if min_value < info['min']:
                module_info['module'] = module
                module_info['channel'] = nonzero[argmin]
                module_info['min'] = min_value
        return module_info

    def init_accum_fishers(self) -> None:
        """Clear accumulated fisher info."""
        for module, name in self.conv_names.items():
            self.accum_fishers[module].zero_()
        for unit in self.units:
            group = unit.name
            self.accum_fishers[group].zero_()

    def channel_prune(self, delta) -> None:
        """Select the channel in model with smallest fisher / delta set
        corresponding in_mask 0."""

        info = {'module': None, 'channel': None, 'min': 1e9}

        for unit in self.units:
            group = unit.name
            in_mask = unit.mutable_channel.current_mask
            fisher = self.accum_fishers[group].double()
            if delta == 'flops':
                fisher /= float(self.flops[group] / 1e9)
            elif delta == 'acts':
                fisher /= float(self.acts[group] / 1e6)
            info.update(
                self.find_pruning_channel(group, fisher, in_mask, info))

        module, channel = info['module'], info['channel']
        for unit in self.units:
            group = unit.name
            if module == group:
                cur_mask = unit.mutable_channel.current_mask
                cur_mask[channel] = False
                unit.mutable_channel.current_choice = cur_mask
                self.current_unit_channel[group] -= 1
                break

        flops, acts = self.compute_flops_acts()
        if dist.is_initialized() and dist.get_rank() == 0:
            print_log(f'slim {module} {channel}th channel, flops {flops:.2f},'
                      f'acts {acts:.2f}')
        self.init_accum_fishers()

    def accumulate_fishers(self) -> None:
        """Accumulate all the fisher information during self.interval
        iterations."""
        for module, _ in self.conv_names.items():
            self.accum_fishers[module] += self.batch_fishers[module].cpu()
        for unit in self.units:
            group = unit.name
            self.accum_fishers[group] += self.batch_fishers[group].cpu()

    def reduce_batch_fishers(self) -> None:
        """Collect batch fisher information from all ranks."""
        for module, _ in self.conv_names.items():
            dist.all_reduce(self.batch_fishers[module])
        for unit in self.units:
            group = unit.name
            dist.all_reduce(self.batch_fishers[group])

    def accumulate_batch_fishers(self) -> None:
        """Accumulate all module.in_mask's fisher and flops in the same
        group."""
        for unit in self.units:
            self.flops[unit.name] = 0
            self.acts[unit.name] = 0

            activated_channels = unit.mutable_channel.activated_channels
            for input_channel in unit.input_related:
                if input_channel.name not in self.module_dict:
                    continue
                module = self.module_dict[input_channel.name]
                if not isinstance(module, nn.Conv2d):
                    continue
                module_fisher = self.temp_fisher_info[module]
                self.temp_fisher_info[unit.name] += module_fisher.cpu()
                delta_flops = self.flops[module] // module.in_channels // \
                    module.out_channels * activated_channels
                self.flops[unit.name] += delta_flops

            self.batch_fishers[unit.name] = (
                self.temp_fisher_info[unit.name]**2).sum(0).to(
                    module.weight.device)

            for output_channel in unit.output_related:
                if output_channel.name not in self.module_dict:
                    continue
                module = self.module_dict[output_channel.name]
                if not isinstance(module, nn.Conv2d):
                    continue
                delta_flops = self.flops[module] // module.out_channels // \
                    module.in_channels * activated_channels
                self.flops[unit.name] += delta_flops
                acts = self.acts[module] // module.out_channels
                self.acts[unit.name] += acts

        for module, _ in self.conv_names.items():
            self.batch_fishers[module] = (
                self.temp_fisher_info[module]**2).sum(0).to(
                    module.weight.device)

    def cal_group_fishers(self) -> None:
        self.accumulate_batch_fishers()
        if dist.is_initialized():
            self.reduce_batch_fishers()
        self.accumulate_fishers()
        self.init_temp_fishers()

    def compute_flops_acts(self):
        """Computing the flops and activation remains."""
        flops = 0
        max_flops = 0
        acts = 0
        max_acts = 0

        for module, _ in self.conv_names.items():
            max_flop = self.flops[module]
            in_channels = module.in_channels
            out_channels = module.out_channels

            act_in_channels = module.mutable_attrs[
                'in_channels'].activated_channels
            act_out_channels = module.mutable_attrs[
                'out_channels'].activated_channels
            flops += max_flop / (in_channels * out_channels) * (
                act_in_channels * act_out_channels)
            max_flops += max_flop
            max_act = self.acts[module]
            acts += max_act / out_channels * act_out_channels
            max_acts += max_act
        return flops / max_flops, acts / max_acts

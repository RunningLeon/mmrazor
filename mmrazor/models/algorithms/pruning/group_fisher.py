# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from mmengine import Config, MessageHub
from mmengine.logging import print_log
from mmengine.model import BaseModel
from mmengine.runner import save_checkpoint
from mmengine.structures import BaseDataElement

from mmrazor.models.mutators import ChannelMutator
from mmrazor.registry import MODELS
from ..base import BaseAlgorithm

LossResults = Dict[str, torch.Tensor]
TensorResults = Union[Tuple[torch.Tensor], torch.Tensor]
PredictResults = List[BaseDataElement]
ForwardResults = Union[LossResults, TensorResults, PredictResults]


@MODELS.register_module()
class GroupFisher(BaseAlgorithm):
    """`Group Fisher Pruning for Practical Network Compression`.
    https://arxiv.org/pdf/2108.00708.pdf.

    Args:
        architecture (Union[BaseModel, Dict]): The model to be pruned.
        pruning (bool): When True, the model is in the pruning process, when
            False, the model is in the finetune process. Defaults to True.
        mutator (Union[Dict, ChannelMutator], optional): The config
            of a mutator. Defaults to dict( type='ChannelMutator',
            channel_unit_cfg=dict( type='SequentialMutableChannelUnit')).
        delta (str): "acts" or "flops", prune the model by activations or
            flops. Defaults to "acts".
        interval (int): The interval of  pruning two channels. Defaults to 10.
        batch_size (int): The batch_size when pruning model. Defaults to 2.
        save_ckpt_delta_thr (list): Checkpoint would be saved when
            the delta reached specific value in the list.
            Defaults to [0.75, 0.5, 0.25].
        data_preprocessor (Optional[Union[Dict, nn.Module]], optional):
            Defaults to None.
        init_cfg (Optional[Dict], optional): init config for the model.
            Defaults to None.
    """

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 pruning: bool = True,
                 mutator: Union[Dict, ChannelMutator] = dict(
                     type='ChannelMutator',
                     channel_unit_cfg=dict(
                         type='SequentialMutableChannelUnit')),
                 delta: str = 'acts',
                 interval: int = 10,
                 batch_size: int = 2,
                 save_ckpt_delta_thr: list = [0.75, 0.5, 0.25],
                 data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
                 init_cfg: Optional[Dict] = None) -> None:

        super().__init__(architecture, data_preprocessor, init_cfg)
        # using sync bn or normal bn
        import torch.distributed as dist
        if dist.is_initialized():
            print_log('Convert Bn to SyncBn.')
            self.architecture = nn.SyncBatchNorm.convert_sync_batchnorm(
                self.architecture)
        else:
            from mmengine.model import revert_sync_batchnorm
            self.architecture = revert_sync_batchnorm(self.architecture)

        self.pruning = pruning
        self.interval = interval
        self.batch_size = batch_size
        self.save_ckpt_delta_thr = save_ckpt_delta_thr

        # mutator
        self.mutator: ChannelMutator = MODELS.build(mutator)
        self.mutator.prepare_from_supernet(self.architecture)

        if self.pruning:
            self.module_dict = dict(self.architecture.named_modules())
            self.conv_names = self._map_conv_name(self.module_dict)
            self.delta = delta
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
            self.delta = delta

            # Init fisher info for all convs.
            for conv, _ in self.conv_names.items():
                self.conv_inputs[conv] = []
                self.temp_fisher_info[conv] = conv.weight.data.new_zeros(
                    self.batch_size, conv.in_channels)
                self.accum_fishers[conv] = conv.weight.data.new_zeros(
                    conv.in_channels)

            # Init fisher info for all units (or called groups).
            self.current_unit_channel = dict()
            for unit in self.mutator.units:
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

    def _map_conv_name(self,
                       named_modules: Dict[str, nn.ModuleDict]) -> OrderedDict:
        """Map the conv modules with their names.

        Args:
            named_modules (Dict[str, nn.ModuleDict]): named_modules of the
                architecture.
        """
        conv2name = OrderedDict()
        for unit in self.mutator.units:
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
        for unit in self.mutator.units:
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

    def _compute_fisher_backward_hook(self, module, grad_input, *args) -> None:
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

    def accumulate_fishers(self) -> None:
        """Accumulate all the fisher information during self.interval
        iterations."""
        for module, _ in self.conv_names.items():
            self.accum_fishers[module] += self.batch_fishers[module].cpu()
        for unit in self.mutator.units:
            group = unit.name
            self.accum_fishers[group] += self.batch_fishers[group].cpu()

    def reduce_fishers(self) -> None:
        """Collect fisher information from all ranks."""
        for module, _ in self.conv_names.items():
            dist.all_reduce(self.batch_fishers[module])
        for unit in self.mutator.units:
            group = unit.name
            dist.all_reduce(self.batch_fishers[group])

    def group_fishers(self) -> None:
        """Accumulate all module.in_mask's fisher and flops in the same
        group."""
        for unit in self.mutator.units:
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

    def init_flops_acts(self) -> None:
        """Clear the flops and acts of model in each iter."""
        for module, _ in self.conv_names.items():
            self.flops[module] = 0
            self.acts[module] = 0

    def init_temp_fishers(self) -> None:
        """Clear fisher info of single conv and group."""
        for module, _ in self.conv_names.items():
            self.temp_fisher_info[module].zero_()
        for unit in self.mutator.units:
            group = unit.name
            self.temp_fisher_info[group].zero_()

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

    def init_accum_fishers(self) -> None:
        """Clear accumulated fisher info."""
        for module, name in self.conv_names.items():
            self.accum_fishers[module].zero_()
        for unit in self.mutator.units:
            group = unit.name
            self.accum_fishers[group].zero_()

    def channel_prune(self) -> None:
        """Select the channel in model with smallest fisher / delta set
        corresponding in_mask 0."""

        info = {'module': None, 'channel': None, 'min': 1e9}

        for unit in self.mutator.units:
            group = unit.name
            in_mask = unit.mutable_channel.current_mask
            fisher = self.accum_fishers[group].double()
            if self.delta == 'flops':
                fisher /= float(self.flops[group] / 1e9)
            elif self.delta == 'acts':
                fisher /= float(self.acts[group] / 1e6)
            info.update(
                self.find_pruning_channel(group, fisher, in_mask, info))

        module, channel = info['module'], info['channel']
        for unit in self.mutator.units:
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

    def save_delta_ckpt(self, delta: float, cfg: Config) -> None:
        """Save checkpoint according to delta thres.

        Args:
            delta (float): The value of delta.
            cfg (Config): The config dictionary.
        """
        self.save_ckpt_delta_thr.pop(0)
        ckpt = {'state_dict': self.state_dict()}
        save_path = f'{cfg.work_dir}/{self.delta}_{delta:.2f}.pth'
        save_checkpoint(ckpt, save_path)
        print_log(f'Save checkpoint to {save_path}')

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'tensor') -> ForwardResults:
        """Forward.

        Args:
            batch_inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.
            mode (str): mode should be one of ``loss``, ``predict`` and
                ``tensor``
                - ``loss``: Called by ``train_step`` and return loss ``dict``
                  used for logging
                - ``predict``: Called by ``val_step`` and ``test_step``
                  and return list of ``BaseDataElement`` results used for
                  computing metric.
                - ``tensor``: Called by custom use to get ``Tensor`` type
                  results.
        """
        if self.pruning:
            hub = MessageHub.get_current_instance()
            cur_iter = hub.runtime_info['iter']
            if cur_iter > 0:
                self.group_fishers()
                if dist.is_initialized():
                    self.reduce_fishers()
                self.accumulate_fishers()
                self.init_temp_fishers()

                # pruning
                if cur_iter % self.interval == 0:
                    self.channel_prune()
                    self.init_accum_fishers()
                    flops, acts = self.compute_flops_acts()

                    if len(self.save_ckpt_delta_thr) > 0:
                        cfg = Config.fromstring(hub.runtime_info['cfg'], '.py')
                        if self.delta == 'flops':
                            if flops < self.save_ckpt_delta_thr[0]:
                                self.save_delta_ckpt(flops, cfg)
                        else:
                            if acts < self.save_ckpt_delta_thr[0]:
                                self.save_delta_ckpt(acts, cfg)
                self.init_flops_acts()

        return super().forward(inputs, data_samples, mode)

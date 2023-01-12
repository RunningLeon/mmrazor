# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from mmengine import Config, MessageHub
from mmengine.dist import master_only
from mmengine.logging import print_log
from mmengine.model import BaseModel
from mmengine.runner import save_checkpoint
from mmengine.structures import BaseDataElement

from mmrazor.models.algorithms.base import BaseAlgorithm
from mmrazor.models.task_modules import ResourceEstimator
from mmrazor.registry import MODELS, TASK_UTILS
from .group_fisher_channel_mutator import GroupFisherChannelMutator

LossResults = Dict[str, torch.Tensor]
TensorResults = Union[Tuple[torch.Tensor], torch.Tensor]
PredictResults = List[BaseDataElement]
ForwardResults = Union[LossResults, TensorResults, PredictResults]


@MODELS.register_module()
class GroupFisherAlgorithm(BaseAlgorithm):
    """`Group Fisher Pruning for Practical Network Compression`.
    https://arxiv.org/pdf/2108.00708.pdf.

    Args:
        architecture (Union[BaseModel, Dict]): The model to be pruned.
        pruning (bool): When True, the model is in the pruning process, when
            False, the model is in the finetune process. Defaults to True.
        mutator (Union[Dict, ChannelMutator], optional): The config
            of a mutator. Defaults to dict( type='GroupFisherChannelMutator',
            channel_unit_cfg=dict( type='GroupFisherChannelUnit')).
        interval (int): The interval of  pruning two channels. Defaults to 10.
        delta (str): "acts" or "flops", prune the model by activations or
            flops. Defaults to "flops".
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
                 mutator: Union[Dict, GroupFisherChannelMutator] = dict(
                     type='GroupFisherChannelMutator',
                     channel_unit_cfg=dict(type='GroupFisherChannelUnit')),
                 interval: int = 10,
                 delta: str = 'flops',
                 save_ckpt_delta_thr: list = [0.75, 0.5, 0.25],
                 esitmator=dict(
                     type='ResourceEstimator',
                     flops_params_cfg=dict(input_shape=(1, 3, 224, 224), )),
                 data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
                 init_cfg: Optional[Dict] = None) -> None:

        super().__init__(architecture, data_preprocessor, init_cfg)

        self.pruning = pruning
        self.interval = interval
        self.delta = delta
        self.save_ckpt_delta_thr = save_ckpt_delta_thr
        self.need_restart_record_info = False

        if self.pruning:
            # resource_estimator
            self.resource_estimator: ResourceEstimator = TASK_UTILS.build(
                esitmator)
            self.origin_delta = self.estimate_delta()

        # using sync bn or normal bn
        if dist.is_initialized():
            print_log('Convert Bn to SyncBn.')
            self.architecture = nn.SyncBatchNorm.convert_sync_batchnorm(
                self.architecture)
        else:
            from mmengine.model import revert_sync_batchnorm
            self.architecture = revert_sync_batchnorm(self.architecture)

        # mutator
        self.mutator: GroupFisherChannelMutator = MODELS.build(mutator)
        self.mutator.prepare_from_supernet(self.architecture)

    @master_only
    def save_delta_ckpt(self, delta_percent: float, cfg: Config) -> None:
        """Save checkpoint according to delta thres.

        Args:
            delta (float): Percentage of current value to original value.
            cfg (Config): The config dictionary.
        """
        self.save_ckpt_delta_thr.pop(0)
        ckpt = {'state_dict': self.state_dict()}
        save_path = f'{cfg.work_dir}/{self.delta}_{delta_percent:.2f}.pth'
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
        if self.training and self.pruning:
            hub = MessageHub.get_current_instance()
            cur_iter = hub.runtime_info['iter']

            if cur_iter == 0 or self.need_restart_record_info:
                self.mutator.start_record_info()
                self.need_restart_record_info = False

            if cur_iter > 0:
                self.mutator.update_fisher()
                # pruning
                if cur_iter % self.interval == 0:
                    self.mutator.try_prune()
                    self.mutator.reset_fisher_info()
                    if len(self.save_ckpt_delta_thr) > 0:
                        # avoid recording input and grad during estimating.
                        self.mutator.end_record_info()
                        current_delta = self.estimate_delta()
                        self.mutator.start_record_info()
                        percent = current_delta / self.origin_delta
                        if percent < self.save_ckpt_delta_thr[0]:
                            cfg = Config.fromstring(hub.runtime_info['cfg'],
                                                    '.py')
                            self.save_delta_ckpt(percent, cfg)
                self.mutator.reset_recorded_info()

        # finetune or validation
        else:
            self.mutator.end_record_info()
            self.need_restart_record_info = True

        res = super().forward(inputs, data_samples, mode)

        return res

    def estimate_delta(self):
        """Get the current value of delta."""
        return self.resource_estimator.estimate(
            self.architecture)[self.delta]  # type: ignore

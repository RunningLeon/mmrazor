# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.dist import master_only
from mmengine.hooks import Hook
from mmengine.runner import Runner, save_checkpoint

from mmrazor.models.task_modules.demo_inputs import DefaultDemoInput
from mmrazor.models.task_modules.estimators import ResourceEstimator
from mmrazor.registry import HOOKS, TASK_UTILS
from mmrazor.utils import print_log
from ..utils import RuntimeInfo, get_model_from_runner, is_pruning_algorithm


@HOOKS.register_module()
class PruningStructureHook(Hook):

    def __init__(self, by_epoch=True, interval=1) -> None:
        super().__init__()
        self.by_epoch = by_epoch
        self.interval = interval

    def show_unit_info(self, algorithm):
        if is_pruning_algorithm(algorithm):
            chices = algorithm.mutator.choice_template
            import json
            print_log(json.dumps(chices, indent=4))

            for unit in algorithm.mutator.mutable_units:
                if hasattr(unit, 'importance'):
                    imp = unit.importance()
                    print_log(
                        f'{unit.name}: \t{imp.min().item()}\t{imp.max().item()}'  # noqa
                    )

    @master_only
    def show(self, runner):
        # print structure info
        algorithm = get_model_from_runner(runner)
        if is_pruning_algorithm(algorithm):
            self.show_unit_info(algorithm)

    # hook points

    def after_train_epoch(self, runner) -> None:
        if self.by_epoch and RuntimeInfo.epoch() % self.interval == 0:
            self.show(runner)

    def after_train_iter(self, runner, batch_idx: int, data_batch,
                         outputs) -> None:
        if not self.by_epoch and RuntimeInfo.iter() % self.interval == 0:
            self.show(runner)


@HOOKS.register_module()
class ResourceInfoHook(Hook):

    # init

    def __init__(self,
                 demo_input=DefaultDemoInput([1, 3, 224, 224]),
                 interval=10,
                 delta_type='flops',
                 save_ckpt_delta_thr=[0.5],
                 early_stop=True) -> None:
        super().__init__()
        if isinstance(demo_input, dict):
            demo_input = TASK_UTILS.build(demo_input)

        self.demo_input = demo_input
        self.save_ckpt_delta_thr = sorted(
            save_ckpt_delta_thr, reverse=True)  # big to small
        self.delta_type = delta_type
        self.early_stop = early_stop
        self.estimator: ResourceEstimator = TASK_UTILS.build(
            dict(
                _scope_='mmrazor',
                type='ResourceEstimator',
                flops_params_cfg=dict(
                    input_shape=tuple(demo_input.input_shape), )))
        self.interval = interval
        self.origin_delta = None

    def before_run(self, runner) -> None:
        model = get_model_from_runner(runner)
        self.origin_delta = self._evaluate(model)[self.delta_type]
        print_log(f'get original {self.delta_type}: {self.origin_delta}')

    # save checkpoint

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs=None) -> None:
        if RuntimeInfo.iter() % self.interval == 0 and len(
                self.save_ckpt_delta_thr) > 0:
            model = get_model_from_runner(runner)
            current_delta = self._evaluate(model)[self.delta_type]
            percent = current_delta / self.origin_delta
            if percent < self.save_ckpt_delta_thr[0]:
                self._save_checkpoint(model, runner.work_dir,
                                      self.save_ckpt_delta_thr.pop(0))
        if self.early_stop and len(self.save_ckpt_delta_thr) == 0:
            exit()

    # show info

    @master_only
    def after_train_epoch(self, runner) -> None:
        model = get_model_from_runner(runner)
        current_delta = self._evaluate(model)[self.delta_type]
        print_log(
            f'current {self.delta_type}: {current_delta} / {self.origin_delta}'  # noqa
        )

    #

    def _evaluate(self, model: nn.Module):
        with torch.no_grad():
            training = model.training
            model.eval()
            res = self.estimator.estimate(model)
            if training:
                model.train()
            return res

    @master_only
    def _save_checkpoint(self, model, path, delta_percent):
        ckpt = {'state_dict': model.state_dict()}
        save_path = f'{path}/{self.delta_type}_{delta_percent:.2f}.pth'
        save_checkpoint(ckpt, save_path)
        print_log(
            f'Save checkpoint to {save_path} with {self._evaluate(model)}'  # noqa
        )
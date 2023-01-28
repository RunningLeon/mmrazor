# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dist import dist
from mmengine.hooks import Hook
from torch import distributed as torch_dist

from mmrazor.registry import HOOKS
from mmrazor.utils import print_log


@HOOKS.register_module()
class PruneHook(Hook):

    def get_model(self, runner):
        if torch_dist.is_initialized():
            return runner.model.module
        else:
            return runner.model

    def after_train_epoch(self, runner) -> None:
        self.print_info(runner)

    # def after_train_iter(self,
    #                      runner,
    #                      batch_idx: int,
    #                      data_batch=None,
    #                      outputs=None) -> None:
    #     self.print_info(runner)

    def print_info(self, runner):
        if dist.get_rank() == 0:
            model = self.get_model(runner)
            if hasattr(model, 'current_flop') and hasattr(
                    model, 'origin_delta'):
                print_log(f'flops: {model.current_flop}, {model.origin_flop}')
                print_log(
                    f'params: {model.current_param}, {model.origin_param}')

            chices = model.mutator.choice_template
            import json
            print_log(json.dumps(chices, indent=4))

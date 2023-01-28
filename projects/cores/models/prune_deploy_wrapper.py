import json
import types

import torch.nn as nn
from mmengine.model import BaseModel

from mmrazor.models import BaseAlgorithm
from mmrazor.registry import MODELS
from mmrazor.structures.subnet.fix_subnet import (export_fix_subnet,
                                                  load_fix_subnet)
from mmrazor.utils import print_log


def clean_params_init_info(model: nn.Module):
    if hasattr(model, '_params_init_info'):
        delattr(model, '_params_init_info')
    for module in model.modules():
        if hasattr(module, '_params_init_info'):
            delattr(module, '_params_init_info')


def empty_init_weights(model):
    pass


@MODELS.register_module()
def PruneDeployWrapper(algorithm):
    algorithm: BaseAlgorithm = MODELS.build(algorithm)
    algorithm.init_weights()
    clean_params_init_info(algorithm)
    print_log(json.dumps(algorithm.mutator.choice_template, indent=4))

    if hasattr(algorithm, 'to_static'):
        model = algorithm.to_static()
    else:
        mutables = export_fix_subnet(algorithm.architecture)[0]
        load_fix_subnet(algorithm.architecture, mutables)
        model = algorithm.architecture

    model.data_preprocessor = algorithm.data_preprocessor
    if isinstance(model, BaseModel):
        model.init_cfg = None
        model.init_weights = types.MethodType(empty_init_weights, model)
    return model

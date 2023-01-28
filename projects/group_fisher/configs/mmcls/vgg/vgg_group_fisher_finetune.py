_base_ = '../../../../models/vgg/configs/vgg_pretrain.py'
custom_imports = dict(imports=['projects.group_fisher'])

architecture = _base_.model
# `pruned_path` need to be updated.
pruned_path = './work_dirs/vgg_group_fisher_prune/flops_0.30.pth'
architecture.update({'data_preprocessor': _base_.data_preprocessor})
data_preprocessor = None

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GroupFisherAlgorithm',
    architecture=architecture,
    pruning=False,
    mutator=dict(
        type='GroupFisherChannelMutator',
        parse_cfg=dict(type='ChannelAnalyzer', tracer_type='FxTracer'),
        channel_unit_cfg=dict(type='GroupFisherChannelUnit')),
    init_cfg=dict(type='Pretrained', checkpoint=pruned_path),
)

custom_hooks = [
    dict(type='mmrazor.PruneHook'),
]

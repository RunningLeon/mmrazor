_base_ = 'mmcls::resnet/resnet50_8xb32_in1k.py'
custom_imports = dict(imports=['projects.group_fisher'])

pruned_path = './work_dirs/resnet_group_fisher_prune/flops_0.50.pth'

architecture = _base_.model
architecture.update({
    'data_preprocessor': _base_.data_preprocessor,
})
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

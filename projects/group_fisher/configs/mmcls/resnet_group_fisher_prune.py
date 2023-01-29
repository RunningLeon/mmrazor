_base_ = 'mmcls::resnet/resnet50_8xb32_in1k.py'
custom_imports = dict(imports=['projects'])
architecture = _base_.model
pretrained_path = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth'  # noqa
architecture.init_cfg = dict(type='Pretrained', checkpoint=pretrained_path)
architecture.update({
    'data_preprocessor': _base_.data_preprocessor,
})
data_preprocessor = None

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GroupFisherAlgorithm',
    architecture=architecture,
    interval=25,
    mutator=dict(
        type='GroupFisherChannelMutator',
        parse_cfg=dict(type='ChannelAnalyzer', tracer_type='FxTracer'),
        channel_unit_cfg=dict(
            type='GroupFisherChannelUnit',
            detla_type='act',
        ),
    ),
)
model_wrapper_cfg = dict(
    type='mmrazor.GroupFisherDDP',
    broadcast_buffers=False,
)
# update optimizer

optim_wrapper = dict(optimizer=dict(lr=0.004, ))
param_scheduler = None

custom_hooks = [
    dict(type='mmrazor.PruningStructureHook'),
    dict(
        type='mmrazor.ResourceInfoHook',
        interval=25,
        demo_input=dict(
            type='mmrazor.DefaultDemoInput',
            input_shape=[1, 3, 224, 224],
        ),
        save_ckpt_delta_thr=[0.75, 0.50],
    ),
]

# original
"""
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0001,
        _scope_='mmcls'))
param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=True,
    milestones=[30, 60, 90],
    gamma=0.1,
    _scope_='mmcls')
"""

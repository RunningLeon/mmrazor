_base_ = 'mmcls::resnet/resnet50_8xb32_in1k.py'
custom_imports = dict(imports=['projects.group_fisher'])
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
    interval=10,
    delta='flops',
    save_ckpt_delta_thr=[0.75, 0.5, 0.25],
    esitmator=dict(
        type='ResourceEstimator',
        flops_params_cfg=dict(input_shape=(1, 3, 224, 224), )),
    mutator=dict(
        type='GroupFisherChannelMutator',
        parse_cfg=dict(type='ChannelAnalyzer', tracer_type='FxTracer'),
        channel_unit_cfg=dict(
            type='GroupFisherChannelUnit',
            detla_type='act',
            default_args=dict(detla_type='act', )),
        min_ratio=0.0,
    ),
)

optim_wrapper = dict(optimizer=dict(lr=0.001))  # origin 0.1
custom_hooks = [
    dict(type='mmrazor.PruneHook'),
]

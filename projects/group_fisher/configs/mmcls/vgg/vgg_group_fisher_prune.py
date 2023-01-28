_base_ = '../../../../models/vgg/configs/vgg_pretrain.py'
custom_imports = dict(imports=['projects'])

pretrained_path = './work_dirs/pretrained/vgg_pretrained.pth'  # noqa

architecture = _base_.model
architecture.init_cfg = dict(type='Pretrained', checkpoint=pretrained_path)
architecture.update({'data_preprocessor': _base_.data_preprocessor})
data_preprocessor = None

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GroupFisherAlgorithm',
    architecture=architecture,
    interval=4,
    mutator=dict(
        type='GroupFisherChannelMutator',
        parse_cfg=dict(type='ChannelAnalyzer', tracer_type='FxTracer'),
        channel_unit_cfg=dict(
            type='GroupFisherChannelUnit',
            default_args={'detla_type': 'act'},
        ),
    ),
)
custom_hooks = [
    dict(type='mmrazor.PruningStructureHook'),
    dict(
        type='mmrazor.ResourceInfoHook',
        demo_input=dict(
            type='mmrazor.DefaultDemoInput',
            input_shape=[1, 3, 32, 32],
        ),
        save_ckpt_delta_thr=[0.5, 0.4, 0.3],
    ),
]

optim_wrapper = dict(optimizer=dict(lr=0.0001))

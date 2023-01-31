_base_ = 'mmseg::deeplabv3plus/deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024.py'  # noqa

custom_imports = dict(imports=['projects'])

architecture = _base_.model
pretrained_path = 'https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r18-d8_512x1024_80k_cityscapes/deeplabv3plus_r18-d8_512x1024_80k_cityscapes_20201226_080942-cff257fe.pth'  # noqa
architecture.init_cfg = dict(type='Pretrained', checkpoint=pretrained_path)
architecture.backbone.frozen_stages = -1
if hasattr(_base_, 'data_preprocessor'):
    architecture.update({'data_preprocessor': _base_.data_preprocessor})
    data_preprocessor = None

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GroupFisherAlgorithm',
    architecture=architecture,
    interval=10,
    mutator=dict(
        type='GroupFisherChannelMutator',
        parse_cfg=dict(type='ChannelAnalyzer', tracer_type='FxTracer'),
        channel_unit_cfg=dict(
            type='GroupFisherChannelUnit',
            default_args=dict(detla_type='flop', ),
        ),
    ),
)

model_wrapper_cfg = dict(
    type='mmrazor.GroupFisherDDP',
    broadcast_buffers=False,
)

optim_wrapper = dict(optimizer=dict(lr=0.002))

custom_hooks = [
    dict(type='mmrazor.PruningStructureHook'),
    dict(
        type='mmrazor.ResourceInfoHook',
        interval=10,
        demo_input=dict(
            type='mmrazor.DefaultDemoInput',
            input_shape=[1, 3, 1024, 512],
        ),
        save_ckpt_delta_thr=[0.75, 0.5],
    ),
]

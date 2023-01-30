_base_ = './pose_s_pretrain.py'
custom_imports = dict(imports=['projects'])

architecture = _base_.model
architecture['_scope_'] = _base_.default_scope

if hasattr(_base_, 'data_preprocessor'):
    architecture.update({'data_preprocessor': _base_.data_preprocessor})
    data_preprocessor = None

pretrained_path = './work_dirs/pretrained/rtmpose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth'  # noqa
architecture.init_cfg = dict(type='Pretrained', checkpoint=pretrained_path)

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GroupFisherAlgorithm',
    architecture=architecture,
    interval=4,
    mutator=dict(
        type='GroupFisherChannelMutator',
        parse_cfg=dict(
            type='ChannelAnalyzer',
            tracer_type='FxTracer',
            demo_input=[1, 3, 256, 192]),
        channel_unit_cfg=dict(
            type='GroupFisherChannelUnit',
            default_args=dict(detla_type='act', ),
        ),
    ),
)

model_wrapper_cfg = dict(
    type='mmrazor.GroupFisherDDP',
    broadcast_buffers=False,
)

optim_wrapper = dict(optimizer=dict(lr=4e-4))

custom_hooks = [
    dict(type='mmrazor.PruningStructureHook'),
    dict(
        type='mmrazor.ResourceInfoHook',
        interval=4,
        demo_input=dict(
            type='mmrazor.DefaultDemoInput',
            input_shape=[1, 3, 256, 192],
        ),
        save_ckpt_delta_thr=[0.53, 0.50],
    ),
]

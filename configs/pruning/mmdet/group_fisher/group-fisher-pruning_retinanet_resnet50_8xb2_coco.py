_base_ = 'mmdet::retinanet/retinanet_r50_fpn_1x_coco.py'

pretrained_path = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'  # noqa

architecture = _base_.model
architecture.init_cfg = dict(type='Pretrained', checkpoint=pretrained_path)
architecture.backbone.frozen_stages = -1
model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GroupFisher',
    architecture=architecture,
    interval=10,
    delta='flops',
    save_ckpt_delta_thr=[0.75, 0.5, 0.25],
    mutator=dict(
        type='GroupFisherChannelMutator',
        parse_cfg=dict(type='ChannelAnalyzer', tracer_type='FxTracer'),
        channel_unit_cfg=dict(
            type='L1MutableChannelUnit',
            default_args=dict(choice_mode='ratio')),
        batch_size=2),
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001))

find_unused_parameters = True

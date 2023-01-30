_base_ = [
    'mmcls::resnet/resnet50_8xb32-coslr_in1k.py',
]

# add label smooth
model = dict(
    head=dict(
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, loss_weight=1.0), ))

# to 120 epochs
param_scheduler = dict(T_max=140, end=140)

train_cfg = dict(max_epochs=140)

# 120 epoch : 77.29
# 140 epoch : 77.55

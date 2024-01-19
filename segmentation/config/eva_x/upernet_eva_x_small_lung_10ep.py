_base_ = [
    '../__base__/models/upernet_eva_x_small.py',
    '../__base__/datasets/lung.py',
    '../__base__/runtime/default_runtime.py',
    '../__base__/schedules/schedule_10ep_adamw.py'
]

model = dict(
    backbone = dict(
        pretrained='pretrained/eva_x_small_patch16_merged520k_mim.pt'),
    decode_head=dict(
        num_classes=2, loss_decode=dict(use_sigmoid=True), out_channels=1),
    auxiliary_head=None,
    test_cfg=dict(mode='whole'))
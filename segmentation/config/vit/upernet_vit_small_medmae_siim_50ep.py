_base_ = [
    '../__base__/models/upernet_medical_mae_small.py',
    '../__base__/datasets/siim.py',
    '../__base__/runtime/default_runtime.py',
    '../__base__/schedules/schedule_50ep_adamw.py'
]

model = dict(
    backbone = dict(
        weight_init='/home/jingfengyao/code/medical/EVA-X/reproduce/pretrained/vit-s_CXR_0.3M_mae.pth'),
    decode_head=dict(
        num_classes=2, loss_decode=dict(use_sigmoid=True), out_channels=1),
    auxiliary_head=None,
    test_cfg=dict(mode='whole'))
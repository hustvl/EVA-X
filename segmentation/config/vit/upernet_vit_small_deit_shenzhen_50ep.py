_base_ = [
    '../__base__/models/upernet_medical_mae_small.py',
    '../__base__/datasets/shenzhen.py',
    '../__base__/runtime/default_runtime.py',
    '../__base__/schedules/schedule_50ep_adamw.py'
]

model = dict(
    backbone = dict(
        weight_init='/home/jingfengyao/code/medical/mmsegmentation/projects/unet_cxr/pretrained/deit_small_patch16_224-cd65a155.pth'),
    decode_head=dict(
        num_classes=2, loss_decode=dict(use_sigmoid=True), out_channels=1),
    auxiliary_head=None,
    test_cfg=dict(mode='whole'))
dataset_type = 'CustomDataset'
data_root =  '/home/jingfengyao/code/medical/mmsegmentation/projects/unet_cxr/data/lung_seg'
img_scale = (512, 512)
img_norm_cfg = dict(
    mean=[0.49185243*255, 0.49185243*255, 0.49185243*255], std=[0.28509309*255, 0.28509309*255, 0.28509309*255], to_rgb=True)
img_scale=(512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        img_ratios=[1.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='Train/Images',
        ann_dir='Train/Masks',
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=['normal', 'lung'],
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='Test/Images',
        ann_dir='Test/Masks',
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=['normal', 'lung'],
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='Test/Images',
        ann_dir='Test/Masks',
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=['normal', 'lung'],
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
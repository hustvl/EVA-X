# optimizer
optimizer = dict(
                 type='AdamW', 
                 lr=6e-5,
                 betas=(0.9, 0.999), 
                 weight_decay=0.05,
                )
optimizer_config = dict()
# learning policy
lr_config = dict(
                policy='poly',
                warmup='linear',
                warmup_iters=10,
                warmup_ratio=1e-6,
                power=1.0, min_lr=0.0, by_epoch=True)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=10)
checkpoint_config = dict(by_epoch=True, interval=2)
evaluation = dict(by_epoch=True, interval=2, metric=['mDice', 'mIoU'], pre_eval=True)
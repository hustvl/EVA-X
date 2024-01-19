# a vitdet-like model, simple but effective

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='EVA_X',
        img_size=512, 
        patch_size=16, 
        in_chans=3,
        embed_dim=384, 
        depth=12,
        num_heads=6,
        mlp_ratio=4*2/3,
        qkv_bias=True, 
        drop_path_rate=0.15, 
        init_values=None, 
        use_checkpoint=False, 
        use_abs_pos_emb=True, 
        use_rel_pos_bias=False, 
        use_shared_rel_pos_bias=False,
        rope=True,
        pt_hw_seq_len=14,
        intp_freq=True,
        swiglu=True,
        subln=False,
        xattn=True,
        naiveswiglu=False,
        pretrained=None,
        use_last_feat_only=True,   # different to eva02-seg, we use last feature only (borrow from vitdet, thanks a lot)
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[384, 384, 384, 384],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=256,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=None,
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

find_unused_parameters = False
"""Stage 1: ordinary SegFormer-B2 CE pretraining on Cityscapes.

The data/optimizer/LR definitions inherit the already audited 256x512 B0
experiment. The model is replaced by a normal MMSeg EncoderDecoder and no EMA
hook is registered in this stage.
"""

_base_ = ['./ddps_cityscapes_256x512_800ep.py']

imagenet_pretrained = '/home/igarashi_25/DDPS/pretrained/mit_b2.pth'
work_dir = (
    '/home/igarashi_25/playground_2/DSDFM/DDPS/result/'
    'segformer_b2_pretrain_256x512_800ep')

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    _delete_=True,
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='MixVisionTransformerCustomInitWeights',
        pretrained=imagenet_pretrained,
        strict_pretrained=True,
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        # Matches the public DDPS-B2 first-prediction feature head.
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        ignore_index=255,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# The B0 base already defines 800*743 max_iters and 50-epoch checkpoints.
interval_100ep = 74300  # 100 * 743
evaluation = dict(
    interval=interval_100ep,
    metric='mIoU',
    pre_eval=True,
    save_best='mIoU')

# Explicitly remove the Stage-2 EMA hook inherited from the B0 base.
custom_hooks = []

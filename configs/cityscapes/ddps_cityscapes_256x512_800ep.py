"""DDPS MiT-B0 on standard 19-class Cityscapes at 256 x 512.

This follows the released DDPS Cityscapes MiT-B0 multi-step architecture and
optimizer.  It uses MMSegmentation's standard train IDs 0--18/ignore 255 and
the official MMSeg MiT-B0 Cityscapes checkpoint for the frozen feature path.
"""

_base_ = ['../_base_/default_runtime.py']

repo_root = '/home/igarashi_25/DDPS'
data_root = '/home/igarashi_25/datasets/cityscapes'
pretrained = (
    f'{repo_root}/pretrained/'
    'segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-'
    'e7f88502.pth')
work_dir = (
    '/home/igarashi_25/playground_2/DSDFM/DDPS/result/'
    'cityscapes_256x512_800ep_b4')

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoderDiffusion',
    pretrained=pretrained,
    freeze_parameters=['backbone', 'decode_head'],
    backbone=dict(
        type='MixVisionTransformerCustomInitWeights',
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
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
        type='SegformerHeadUnetFCHeadMultiStep',
        pretrained=pretrained,
        dim=128,
        out_dim=256,
        unet_channels=272,
        dim_mults=[1, 1, 1],
        cat_embedding_dim=16,
        diffusion_timesteps=20,
        inference_timesteps=20,
        collect_timesteps=list(range(20)),
        guidance_scale=1.0,
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
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

# BGR uint8 [0, 255] is converted once to RGB, then normalized.  These are
# exactly torchvision ImageNet mean/std multiplied by 255.
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
image_size = (256, 512)  # (height, width), for auditing/documentation

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    # MMCV img_scale is (width, height).
    dict(type='Resize', img_scale=(512, 256), keep_ratio=False),
    dict(
        type='TorchvisionColorJitter',
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

# The dataset adapter loads annotations before MultiScaleFlipAug so Resize
# transforms image (bilinear) and mask (nearest) identically.  Collect drops
# the mask before inference; pre_eval reads the original label by sample index.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 256),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),
]

dataset_type = 'CityscapesDiffusionDataset'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline),
    train_dataloader=dict(
        samples_per_gpu=4,
        workers_per_gpu=4,
        persistent_workers=True),
    val_dataloader=dict(
        samples_per_gpu=1,
        workers_per_gpu=2,
        persistent_workers=False),
    test_dataloader=dict(
        samples_per_gpu=1,
        workers_per_gpu=2,
        persistent_workers=False))

# The observed dataset has 2975 train samples.  The DDPS API uses drop_last,
# so len(train_dataloader) = floor(2975 / 4) = 743 on one GPU.  Runtime checks
# in train_multi_steps.py abort if either observed count changes.
expected_train_size = 2975
target_epochs = 800
iters_per_epoch = 743
max_iters = target_epochs * iters_per_epoch  # 594400
interval_50ep = 50 * iters_per_epoch  # 37150
interval_200ep = 200 * iters_per_epoch  # 148600

# Preserve the released Cityscapes MiT-B0 DDPS optimizer/LR; no batch LR scale.
optimizer = dict(
    type='AdamW', lr=0.00015, betas=[0.9, 0.96], weight_decay=0.045)
optimizer_config = dict()
fp16 = dict(loss_scale='dynamic')
lr_config = dict(
    # CamelCase keeps MMCV from converting this custom hook name to
    # ``Epoch_CosineLrUpdaterHook``.
    policy='EpochCosine',
    warmup_epochs=10,
    warmup_ratio=0.1,
    target_epochs=800,
    iters_per_epoch=743,
    eta_min=1e-6,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=max_iters)
checkpoint_config = dict(
    by_epoch=False, interval=interval_50ep, max_keep_ckpts=16)
evaluation = dict(
    interval=interval_200ep,
    metric='mIoU',
    pre_eval=True,
    save_best='mIoU')
custom_hooks = [
    dict(
        type='ConstantMomentumEMAHook',
        momentum=0.01,
        interval=25,
        eval_interval=interval_200ep,
        auto_resume=False,
        # After optimizer (40), before checkpoint (50).
        priority=49)
]
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])

gpu_ids = [0]  # physical GPU 1 is remapped to logical cuda:0 by the scripts
auto_resume = True
cudnn_benchmark = True
workflow = [('train', 1)]

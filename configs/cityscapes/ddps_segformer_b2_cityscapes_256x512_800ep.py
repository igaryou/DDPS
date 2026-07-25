"""Stage 2: released DDPS SegFormer-B2 on 19-class Cityscapes.

The DDPS-B2 dimensions match the repository's released ADE20K B2 multistep
config (the same architecture reported as about 65.5M parameters): dim=256,
out_dim=256, channels=256, cat_embedding_dim=16, unet_channels=272 and
three [1,1,1] UNet levels. Only the dataset/classes/timesteps follow the
existing audited Cityscapes B0 experiment.
"""

_base_ = ['./ddps_cityscapes_256x512_800ep.py']

stage1_checkpoint = (
    '/home/igarashi_25/playground_2/DSDFM/DDPS/result/'
    'segformer_b2_pretrain_256x512_800ep/latest.pth')
work_dir = (
    '/home/igarashi_25/playground_2/DSDFM/DDPS/result/'
    'ddps_segformer_b2_cityscapes_256x512_800ep_b4')

model = dict(
    # Load Stage-1 independently into backbone and first-prediction head.
    pretrained=None,
    backbone=dict(
        pretrained=stage1_checkpoint,
        strict_pretrained=True,
        embed_dims=64,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8]),
    decode_head=dict(
        pretrained=stage1_checkpoint,
        strict_pretrained=True,
        train_mask_embedding=True,
        dim=256,
        out_dim=256,
        unet_channels=272,
        dim_mults=[1, 1, 1],
        cat_embedding_dim=16,
        in_channels=[64, 128, 320, 512],
        channels=256,
        diffusion_timesteps=20,
        inference_timesteps=20,
        collect_timesteps=list(range(20))))

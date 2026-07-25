#!/usr/bin/env python3
"""Audit Stage-2 DDPS-B2 loading, freezing, schedule and tensor shapes."""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))

import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.cnn.utils import revert_sync_batchnorm
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
import mmseg_custom  # noqa: F401

from audit_b2_common import audit_lr, batch_tensors, build_train_loader


def count(module, trainable_only=False):
    return sum(parameter.numel() for parameter in module.parameters()
               if not trainable_only or parameter.requires_grad)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config', nargs='?',
        default=os.path.join(
            REPO, 'configs/cityscapes/'
            'ddps_segformer_b2_cityscapes_256x512_800ep.py'))
    parser.add_argument('--stage1-checkpoint')
    parser.add_argument('--forward', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.stage1_checkpoint:
        cfg.model.backbone.pretrained = args.stage1_checkpoint
        cfg.model.decode_head.pretrained = args.stage1_checkpoint
    checkpoint = cfg.model.backbone.pretrained
    assert checkpoint == cfg.model.decode_head.pretrained
    assert os.path.isfile(checkpoint), f'Missing Stage-1 checkpoint: {checkpoint}'

    assert cfg.model.type == 'EncoderDecoderDiffusion'
    backbone_cfg = cfg.model.backbone
    stage_channels = [backbone_cfg.embed_dims * heads
                      for heads in backbone_cfg.num_heads]
    assert backbone_cfg.num_layers == [3, 4, 6, 3]
    assert stage_channels == [64, 128, 320, 512]
    head_cfg = cfg.model.decode_head
    assert head_cfg.in_channels == stage_channels
    assert (head_cfg.dim, head_cfg.out_dim, head_cfg.unet_channels) == (
        256, 256, 272)
    assert head_cfg.dim_mults == [1, 1, 1]
    assert head_cfg.channels == 256 and head_cfg.cat_embedding_dim == 16
    assert head_cfg.unet_channels == (
        head_cfg.out_dim + head_cfg.cat_embedding_dim)
    assert head_cfg.diffusion_timesteps == 20
    assert head_cfg.num_classes == 19 and head_cfg.ignore_index == 255

    train, loader = build_train_loader(cfg)
    val = build_dataset(cfg.data.val, dict(test_mode=True))
    model = build_segmentor(cfg.model)
    model.init_weights()
    head = model.decode_head
    backbone_report = model.backbone.pretrained_load_report
    head_report = head.pretrained_load_report
    for report in (backbone_report, head_report):
        assert report and not report['missing_required_keys']
        assert not report['unexpected_keys']
        assert not report['shape_mismatch_keys']

    total = count(model)
    trainable = count(model, True)
    frozen = total - trainable
    backbone_trainable = count(model.backbone, True)
    first_head_trainable = sum(
        parameter.numel()
        for name, parameter in head.named_parameters()
        if name.startswith(('convs.', 'fusion_conv.'))
        and parameter.requires_grad)
    unet_trainable = count(head.unet, True)
    conv_trainable = count(head.conv_seg_new, True)
    embedding_trainable = count(head.embed, True)
    assert backbone_trainable == 0
    assert first_head_trainable == 0
    assert unet_trainable > 0
    assert conv_trainable > 0 and embedding_trainable > 0

    released_reference = 65_500_000
    print('model: EncoderDecoderDiffusion / DDPS SegFormer-B2')
    print('stage channels:', stage_channels)
    print('DDPS dimensions:', {
        key: head_cfg[key] for key in
        ('dim', 'out_dim', 'unet_channels', 'dim_mults', 'channels',
         'cat_embedding_dim')})
    print('dataset train/val:', len(train), len(val))
    print('batch/drop_last/loader:', loader.batch_size, loader.drop_last,
          len(loader))
    print('target epochs/max_iters:', cfg.lr_config.target_epochs,
          cfg.runner.max_iters)
    print('checkpoint/evaluation/EMA-swap interval:',
          cfg.checkpoint_config.interval, cfg.evaluation.interval,
          cfg.custom_hooks[0].eval_interval)
    print('Stage-1 checkpoint:', checkpoint)
    print('backbone loaded/allowed-missing/unexpected/mismatch:',
          len(backbone_report['loaded_keys']),
          len(backbone_report['allowed_missing_keys']),
          len(backbone_report['unexpected_keys']),
          len(backbone_report['shape_mismatch_keys']))
    print('first-head loaded/allowed-missing/allowed-unused/unexpected/mismatch:',
          len(head_report['loaded_keys']),
          len(head_report['allowed_missing_keys']),
          len(head_report['allowed_unused_source_keys']),
          len(head_report['unexpected_keys']),
          len(head_report['shape_mismatch_keys']))
    print('parameters total/trainable/frozen:', total, trainable, frozen)
    print('released ~65.5M difference:', total - released_reference)
    print('trainable partition backbone/first-head/unet/conv-new/embed:',
          backbone_trainable, first_head_trainable, unet_trainable,
          conv_trainable, embedding_trainable)
    print('diffusion/inference/collect timesteps:',
          head_cfg.diffusion_timesteps, head_cfg.inference_timesteps,
          head_cfg.collect_timesteps)
    print('EMA momentum/effective-decay/update/eval-swap:',
          cfg.custom_hooks[0].momentum,
          1 - cfg.custom_hooks[0].momentum,
          cfg.custom_hooks[0].interval,
          cfg.custom_hooks[0].eval_interval)

    audit_lr(cfg, len(loader))
    images, masks, metadata = batch_tensors(next(iter(loader)))
    assert tuple(images.shape) == (4, 3, 256, 512)
    assert tuple(masks.shape) == (4, 1, 256, 512)
    print('batch image/mask:', tuple(images.shape), tuple(masks.shape),
          images.dtype, masks.dtype)

    if args.forward:
        if not torch.cuda.is_available():
            raise RuntimeError('--forward requested but CUDA is unavailable')
        model = revert_sync_batchnorm(model).cuda().train()
        with torch.inference_mode():
            features = model.extract_feat(images.cuda())
            logits_low = model.decode_head(features, masks.cuda())
            logits = F.interpolate(
                logits_low.float(), size=(256, 512), mode='bilinear',
                align_corners=False)
        assert tuple(logits.shape) == (4, 19, 256, 512)
        assert torch.isfinite(logits).all()
        print('logits:', tuple(logits.shape), logits.dtype,
              'finite=', bool(torch.isfinite(logits).all()))
        print('max CUDA memory MiB:',
              torch.cuda.max_memory_allocated() / 1024**2)

    print('Stage-2 audit: passed')


if __name__ == '__main__':
    main()

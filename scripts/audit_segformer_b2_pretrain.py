#!/usr/bin/env python3
"""Audit Stage-1 SegFormer-B2 configuration, data and initialization."""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))

import torch
from mmcv import Config
from mmcv.cnn.utils import revert_sync_batchnorm
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
import mmseg_custom  # noqa: F401

from audit_b2_common import audit_lr, batch_tensors, build_train_loader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config', nargs='?',
        default=os.path.join(
            REPO, 'configs/cityscapes/'
            'segformer_b2_cityscapes_256x512_800ep_pretrain.py'))
    parser.add_argument('--forward', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    assert cfg.model.type == 'EncoderDecoder'
    backbone = cfg.model.backbone
    stage_channels = [backbone.embed_dims * heads
                      for heads in backbone.num_heads]
    assert backbone.num_layers == [3, 4, 6, 3]
    assert stage_channels == [64, 128, 320, 512]
    assert cfg.model.decode_head.in_channels == stage_channels
    assert cfg.model.decode_head.channels == 256
    assert cfg.model.decode_head.num_classes == 19
    assert cfg.model.decode_head.ignore_index == 255
    assert os.path.isfile(backbone.pretrained), (
        f'Missing ImageNet checkpoint: {backbone.pretrained}')

    train, loader = build_train_loader(cfg)
    val = build_dataset(cfg.data.val, dict(test_mode=True))
    assert len(train.CLASSES) == 19 and train.ignore_index == 255
    assert len(loader) == cfg.lr_config.iters_per_epoch
    print('model: EncoderDecoder / SegFormer-B2')
    print('stage channels:', stage_channels)
    print('dataset train/val:', len(train), len(val))
    print('batch/drop_last/loader:', loader.batch_size, loader.drop_last,
          len(loader))
    print('target epochs/max_iters:', cfg.lr_config.target_epochs,
          cfg.runner.max_iters)
    print('checkpoint/evaluation interval:', cfg.checkpoint_config.interval,
          cfg.evaluation.interval)
    print('ImageNet checkpoint:', backbone.pretrained,
          os.path.getsize(backbone.pretrained), 'bytes')
    print('pipeline:', [step['type'] for step in cfg.train_pipeline])
    print('normalization:', cfg.img_norm_cfg)

    model = build_segmentor(cfg.model)
    model.init_weights()
    report = model.backbone.pretrained_load_report
    assert report and not report['missing_required_keys']
    assert not report['unexpected_keys'] and not report['shape_mismatch_keys']
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    assert total == trainable
    print('parameters total/trainable/frozen:', total, trainable,
          total - trainable)
    print('backbone load loaded/missing/unexpected/mismatch:',
          len(report['loaded_keys']), len(report['missing_required_keys']),
          len(report['unexpected_keys']), len(report['shape_mismatch_keys']))

    audit_lr(cfg, len(loader))
    images, masks, metadata = batch_tensors(next(iter(loader)))
    assert tuple(images.shape) == (4, 3, 256, 512)
    assert tuple(masks.shape) == (4, 1, 256, 512)
    print('batch image/mask:', tuple(images.shape), tuple(masks.shape),
          images.dtype, masks.dtype)

    if args.forward:
        if not torch.cuda.is_available():
            raise RuntimeError('--forward requested but CUDA is unavailable')
        model = revert_sync_batchnorm(model).cuda().eval()
        with torch.inference_mode():
            logits = model.encode_decode(images.cuda(), metadata)
        assert tuple(logits.shape) == (4, 19, 256, 512)
        assert torch.isfinite(logits).all()
        print('logits:', tuple(logits.shape), logits.dtype,
              'finite=', bool(torch.isfinite(logits).all()))
        print('max CUDA memory MiB:',
              torch.cuda.max_memory_allocated() / 1024**2)

    print('Stage-1 audit: passed')


if __name__ == '__main__':
    main()

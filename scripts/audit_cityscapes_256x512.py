"""Construct the configured dataset/loader and print the preflight audit."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from mmcv import Config
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
import mmseg_custom  # noqa: F401: register custom modules
from mmseg_custom.core.hook.lr_epoch import EpochCosineLrUpdaterHook


def main(path):
    cfg = Config.fromfile(path)
    train = build_dataset(cfg.data.train)
    val = build_dataset(cfg.data.val, dict(test_mode=True))
    loader = build_dataloader(
        train, samples_per_gpu=cfg.data.train_dataloader.samples_per_gpu,
        workers_per_gpu=cfg.data.train_dataloader.workers_per_gpu,
        num_gpus=1, dist=False, shuffle=False, drop_last=True,
        persistent_workers=cfg.data.train_dataloader.get(
            'persistent_workers', False))
    model = build_segmentor(cfg.model)
    n = sum(p.numel() for p in model.parameters())
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model:', cfg.model.type)
    print('backbone:', cfg.model.backbone.type)
    print('parameters:', n, 'trainable:', t)
    print('train/val size:', len(train), len(val))
    print('batch train/val:', loader.batch_size,
          cfg.data.val_dataloader.samples_per_gpu)
    print('len(train_dataloader):', len(loader), 'drop_last:', loader.drop_last)
    print(f'{cfg.lr_config.target_epochs} epoch iters:',
          cfg.lr_config.target_epochs * len(loader))
    print('runner.max_iters:', cfg.runner.max_iters)
    print('image size (H,W):', (256, 512))
    print('classes/ignore:', len(train.CLASSES), train.ignore_index)
    print('optimizer:', cfg.optimizer)
    print('lr/scheduler:', cfg.optimizer.lr, cfg.lr_config)
    lr_cfg = cfg.lr_config
    lr_hook = EpochCosineLrUpdaterHook(
        target_epochs=lr_cfg.target_epochs,
        iters_per_epoch=lr_cfg.iters_per_epoch,
        eta_min=lr_cfg.eta_min,
        warmup_epochs=lr_cfg.warmup_epochs,
        warmup_ratio=lr_cfg.warmup_ratio,
        by_epoch=False)
    print('scheduler class: EpochCosineLrUpdaterHook')
    print('scheduler formula: LinearLR(start_factor=0.1) -> CosineAnnealingLR')
    warmup_iters = lr_cfg.warmup_epochs * lr_cfg.iters_per_epoch
    print('warmup epochs/iters:', lr_cfg.warmup_epochs, warmup_iters)
    print('base/min lr:', cfg.optimizer.lr, lr_cfg.eta_min)
    audit_epochs = [epoch for epoch in
                    [0, 1, 10, 50, 100, 200, 400, 600, 799, 800]
                    if epoch <= lr_cfg.target_epochs]
    if lr_cfg.target_epochs not in audit_epochs:
        audit_epochs.append(lr_cfg.target_epochs)
    for epoch in audit_epochs:
        value = lr_hook.get_epoch_lr(cfg.optimizer.lr, epoch)
        print(f'lr epoch={epoch} iter={epoch * len(loader)} '
              f'group=all lr={value:.10e}')
    assert cfg.runner.max_iters == lr_cfg.target_epochs * len(loader)
    if lr_cfg.target_epochs == 800:
        assert cfg.runner.max_iters == 800 * 743 == 594400

    # Verify that the LR is exactly constant at the beginning, middle, and
    # end of representative epochs.
    constant_epochs = sorted(set([
        0, 1, lr_cfg.warmup_epochs - 1,
        lr_cfg.warmup_epochs, lr_cfg.target_epochs - 1]))
    for epoch in constant_epochs:
        start = epoch * len(loader)
        middle = start + len(loader) // 2
        end = start + len(loader) - 1
        values = [lr_hook.get_iter_lr(cfg.optimizer.lr, point)
                  for point in (start, middle, end)]
        assert values[0] == values[1] == values[2]
        print(f'epoch-constant epoch={epoch} '
              f'iters=({start},{middle},{end}) lr={values[0]:.10e}')

    assert lr_hook.get_iter_lr(cfg.optimizer.lr, 0) == \
        lr_hook.get_iter_lr(cfg.optimizer.lr, len(loader) - 1)
    assert lr_hook.get_iter_lr(cfg.optimizer.lr, len(loader)) == \
        lr_hook.get_iter_lr(cfg.optimizer.lr, 2 * len(loader) - 1)
    assert lr_hook.get_iter_lr(cfg.optimizer.lr, warmup_iters - 1) != \
        lr_hook.get_iter_lr(cfg.optimizer.lr, warmup_iters)
    print('epoch-boundary asserts: passed')

    # Compare every epoch boundary with the actual PyTorch scheduler used by
    # CCDM: SequentialLR(LinearLR -> CosineAnnealingLR).
    reference_param = torch.nn.Parameter(torch.tensor(0.0))
    reference_optimizer = torch.optim.SGD(
        [reference_param], lr=cfg.optimizer.lr)
    reference_scheduler = torch.optim.lr_scheduler.SequentialLR(
        reference_optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                reference_optimizer,
                start_factor=lr_cfg.warmup_ratio,
                end_factor=1.0,
                total_iters=lr_cfg.warmup_epochs),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                reference_optimizer,
                T_max=lr_cfg.target_epochs - lr_cfg.warmup_epochs,
                eta_min=lr_cfg.eta_min),
        ],
        milestones=[lr_cfg.warmup_epochs])
    for epoch in range(lr_cfg.target_epochs + 1):
        expected = reference_optimizer.param_groups[0]['lr']
        actual = lr_hook.get_epoch_lr(cfg.optimizer.lr, epoch)
        assert abs(actual - expected) <= 1e-12
        if epoch < lr_cfg.target_epochs:
            reference_optimizer.step()
            reference_scheduler.step()
    print('PyTorch SequentialLR epoch-boundary comparison: passed')

    print('evaluation interval epochs/iters:',
          cfg.evaluation.interval / len(loader), cfg.evaluation.interval)
    print('checkpoint interval epochs/iters:',
          cfg.checkpoint_config.interval / len(loader),
          cfg.checkpoint_config.interval)
    print('best metric/timestep:', 'mIoU', cfg.model.decode_head.collect_timesteps[-1])
    print('EMA momentum/decay/update/eval-swap:',
          cfg.custom_hooks[0].momentum, 1 - cfg.custom_hooks[0].momentum,
          cfg.custom_hooks[0].interval, cfg.custom_hooks[0].eval_interval)
    print('work_dir:', cfg.work_dir)
    print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'))
    print('cuda:', torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
    sample = train[0]
    print('sample img/mask:', sample['img'].data.shape,
          sample['gt_semantic_seg'].data.shape)


if __name__ == '__main__':
    main(sys.argv[1])

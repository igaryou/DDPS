"""Shared, side-effect-free audit helpers for the B2 Cityscapes configs."""

import torch

from mmseg.datasets import build_dataloader, build_dataset
from mmseg_custom.core.hook.lr_epoch import EpochCosineLrUpdaterHook


def build_train_loader(cfg):
    dataset = build_dataset(cfg.data.train)
    loader_cfg = cfg.data.train_dataloader
    loader = build_dataloader(
        dataset,
        samples_per_gpu=loader_cfg.samples_per_gpu,
        workers_per_gpu=loader_cfg.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False,
        drop_last=True,
        persistent_workers=loader_cfg.get('persistent_workers', False))
    return dataset, loader


def audit_lr(cfg, loader_length):
    lr_cfg = cfg.lr_config
    hook = EpochCosineLrUpdaterHook(
        target_epochs=lr_cfg.target_epochs,
        iters_per_epoch=lr_cfg.iters_per_epoch,
        eta_min=lr_cfg.eta_min,
        warmup_epochs=lr_cfg.warmup_epochs,
        warmup_ratio=lr_cfg.warmup_ratio,
        by_epoch=False)
    print('scheduler: EpochCosineLrUpdaterHook')
    print('formula: epoch-wise LinearLR -> epoch-wise CosineAnnealingLR')
    print('warmup epochs/iters:', lr_cfg.warmup_epochs,
          lr_cfg.warmup_epochs * lr_cfg.iters_per_epoch)
    print('base/min LR:', cfg.optimizer.lr, lr_cfg.eta_min)

    reference_parameter = torch.nn.Parameter(torch.tensor(0.0))
    reference_optimizer = torch.optim.SGD(
        [reference_parameter], lr=cfg.optimizer.lr)
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

    checkpoints = {0, 1, lr_cfg.warmup_epochs, lr_cfg.target_epochs}
    if lr_cfg.target_epochs >= 800:
        checkpoints.update({10, 50, 100, 200, 400, 600, 799, 800})
    for epoch in range(lr_cfg.target_epochs + 1):
        actual = hook.get_epoch_lr(cfg.optimizer.lr, epoch)
        expected = reference_optimizer.param_groups[0]['lr']
        assert abs(actual - expected) <= 1e-12, (epoch, actual, expected)
        if epoch in checkpoints:
            print(f'lr epoch={epoch} iter={epoch * loader_length}: '
                  f'{actual:.12e}')
        if epoch < lr_cfg.target_epochs:
            reference_optimizer.step()
            reference_scheduler.step()

    representative_epochs = sorted({
        0, max(0, lr_cfg.warmup_epochs - 1), lr_cfg.warmup_epochs,
        max(0, lr_cfg.target_epochs - 1)})
    for epoch in representative_epochs:
        start = epoch * loader_length
        middle = start + loader_length // 2
        end = start + loader_length - 1
        values = [hook.get_iter_lr(cfg.optimizer.lr, iteration)
                  for iteration in (start, middle, end)]
        assert values[0] == values[1] == values[2]
        print(f'epoch-constant epoch={epoch} '
              f'iters=({start},{middle},{end}) lr={values[0]:.12e}')

    assert cfg.runner.max_iters == lr_cfg.target_epochs * loader_length
    assert lr_cfg.iters_per_epoch == loader_length
    print('PyTorch SequentialLR boundary comparison: passed')
    return hook


def batch_tensors(batch):
    images = batch['img'].data[0]
    masks = batch['gt_semantic_seg'].data[0]
    metadata = batch['img_metas'].data[0]
    return images, masks, metadata

import math

from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.lr_updater import LrUpdaterHook


@HOOKS.register_module()
class EpochCosineLrUpdaterHook(LrUpdaterHook):
    """CCDM's epoch-wise LinearLR warmup followed by cosine annealing.

    MMCV calls this hook every iteration, but the schedule uses
    ``runner.iter // iters_per_epoch``.  The LR is therefore constant for all
    iterations in one epoch and changes only at an epoch boundary.  Warmup is
    calculated here as well; MMCV's iteration-wise warmup is disabled.
    """

    def __init__(self,
                 target_epochs,
                 iters_per_epoch,
                 eta_min=0.0,
                 warmup_epochs=0,
                 warmup_ratio=0.1,
                 **kwargs):
        self.target_epochs = int(target_epochs)
        self.iters_per_epoch = int(iters_per_epoch)
        self.eta_min = float(eta_min)
        self.schedule_warmup_epochs = int(warmup_epochs)
        self.warmup_ratio = float(warmup_ratio)
        if self.target_epochs <= 0 or self.iters_per_epoch <= 0:
            raise ValueError('target_epochs and iters_per_epoch must be > 0')
        if not 0 <= self.schedule_warmup_epochs < self.target_epochs:
            raise ValueError('warmup_epochs must be in [0, target_epochs)')
        if not 0 < self.warmup_ratio <= 1:
            raise ValueError('warmup_ratio must be in (0, 1]')

        # Silently accepting MMCV warmup here would apply a second,
        # iteration-wise warmup.  Reject it so config mistakes fail early.
        mmcv_warmup = kwargs.pop('warmup', None)
        mmcv_warmup_iters = kwargs.pop('warmup_iters', 0)
        if mmcv_warmup is not None or mmcv_warmup_iters not in (0, None):
            raise ValueError(
                'EpochCosineLrUpdaterHook implements warmup internally; '
                'remove warmup and warmup_iters from lr_config')
        super().__init__(
            warmup=None,
            warmup_iters=0,
            warmup_ratio=self.warmup_ratio,
            **kwargs)

    def get_lr(self, runner, base_lr):
        epoch_index = runner.iter // self.iters_per_epoch
        return self.get_epoch_lr(base_lr, epoch_index)

    def get_epoch_lr(self, base_lr, epoch_index):
        """LR used while training the given zero-based epoch index."""
        epoch_index = int(epoch_index)
        if epoch_index < self.schedule_warmup_epochs:
            factor = self.warmup_ratio + (
                1.0 - self.warmup_ratio) * (
                    epoch_index / self.schedule_warmup_epochs)
            return base_lr * factor

        cosine_epochs = self.target_epochs - self.schedule_warmup_epochs
        progress = min(max(
            (epoch_index - self.schedule_warmup_epochs) / cosine_epochs,
            0.0), 1.0)
        return self.eta_min + 0.5 * (base_lr - self.eta_min) * (
            1.0 + math.cos(math.pi * progress))

    def get_iter_lr(self, base_lr, iteration):
        """Convenience method used by the preflight audit."""
        class _Runner:
            iter = iteration
        return self.get_lr(_Runner(), base_lr)

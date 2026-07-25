"""Eight-image Stage-2 integration config (2 iterations per epoch)."""

_base_ = ['./ddps_segformer_b2_cityscapes_256x512_800ep.py']

debug_train_split = '/home/igarashi_25/DDPS/configs/cityscapes/debug_train.txt'
debug_val_split = '/home/igarashi_25/DDPS/configs/cityscapes/debug_val.txt'

model = dict(
    decode_head=dict(
        inference_timesteps=2,
        collect_timesteps=[0, 19]))

data = dict(
    workers_per_gpu=0,
    train=dict(split=debug_train_split),
    val=dict(split=debug_val_split),
    test=dict(split=debug_val_split),
    train_dataloader=dict(
        samples_per_gpu=4,
        workers_per_gpu=0,
        persistent_workers=False),
    val_dataloader=dict(
        samples_per_gpu=1,
        workers_per_gpu=0,
        persistent_workers=False),
    test_dataloader=dict(
        samples_per_gpu=1,
        workers_per_gpu=0,
        persistent_workers=False))

expected_train_size = 8
target_epochs = 2
iters_per_epoch = 2
max_iters = target_epochs * iters_per_epoch
runner = dict(max_iters=max_iters)
checkpoint_config = dict(interval=iters_per_epoch, max_keep_ckpts=4)
evaluation = dict(interval=iters_per_epoch)
custom_hooks = [
    dict(
        type='ConstantMomentumEMAHook',
        momentum=0.01,
        interval=1,
        eval_interval=iters_per_epoch,
        auto_resume=False,
        priority=49)
]
lr_config = dict(
    target_epochs=target_epochs,
    iters_per_epoch=iters_per_epoch,
    warmup_epochs=1)
log_config = dict(interval=1)
work_dir = (
    '/home/igarashi_25/playground_2/DSDFM/DDPS/result/'
    'ddps_segformer_b2_cityscapes_256x512_debug')

"""Validated loading helpers for the two-stage SegFormer/DDPS workflow."""

from collections import OrderedDict
from collections.abc import Mapping
import os

import torch

from mmseg.utils import get_root_logger


def _read_state_dict(checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f'Pretrained checkpoint does not exist: {checkpoint_path}')

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(
            f'Checkpoint must contain a mapping, got {type(checkpoint)!r}: '
            f'{checkpoint_path}')
    state_dict = checkpoint.get('state_dict', checkpoint)
    if not isinstance(state_dict, Mapping):
        raise RuntimeError(
            f'Checkpoint state_dict must be a mapping: {checkpoint_path}')

    cleaned = OrderedDict()
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise RuntimeError(
                f'Non-string state_dict key {key!r}: {checkpoint_path}')
        while key.startswith('module.'):
            key = key[len('module.'):]
        cleaned[key] = value
    return cleaned


def load_validated_submodule(
        module,
        checkpoint_path,
        checkpoint_prefix,
        description,
        required_target_prefixes=None,
        allowed_missing_target_prefixes=(),
        allowed_unused_source_prefixes=(),
        fail_on_unexpected=True):
    """Load a checkpoint submodule only after key and shape validation.

    Official OpenMMLab ImageNet MiT checkpoints are raw backbone state dicts,
    while Stage-1 checkpoints use full MMSeg ``backbone.``/``decode_head.``
    prefixes. Both forms are accepted and classified explicitly.
    """
    logger = get_root_logger()
    source_all = _read_state_dict(checkpoint_path)
    qualified_prefix = checkpoint_prefix.rstrip('.') + '.'
    has_qualified_keys = any(
        key.startswith(qualified_prefix) for key in source_all)
    if has_qualified_keys:
        source = OrderedDict(
            (key[len(qualified_prefix):], value)
            for key, value in source_all.items()
            if key.startswith(qualified_prefix))
    else:
        source = source_all

    target = module.state_dict()
    if required_target_prefixes is None:
        required = set(target)
    else:
        required = {
            key for key in target
            if key.startswith(tuple(required_target_prefixes))
        }
    if not required:
        raise RuntimeError(
            f'{description}: no required target keys matched '
            f'{required_target_prefixes!r}')

    loaded = OrderedDict()
    shape_mismatch = []
    allowed_unused = []
    unexpected = []
    for key, value in source.items():
        if key in target:
            if tuple(value.shape) != tuple(target[key].shape):
                shape_mismatch.append(
                    f'{key}: checkpoint={tuple(value.shape)}, '
                    f'model={tuple(target[key].shape)}')
            else:
                loaded[key] = value
        elif key.startswith(tuple(allowed_unused_source_prefixes)):
            allowed_unused.append(key)
        else:
            unexpected.append(key)

    missing_required = sorted(required.difference(loaded))
    non_required = set(target).difference(required)
    allowed_missing = sorted(
        key for key in non_required
        if key.startswith(tuple(allowed_missing_target_prefixes)))
    unclassified_missing = sorted(non_required.difference(allowed_missing))
    shape_mismatch = sorted(shape_mismatch)
    allowed_unused = sorted(allowed_unused)
    unexpected = sorted(unexpected)

    report = dict(
        checkpoint=os.path.abspath(checkpoint_path),
        description=description,
        loaded_keys=tuple(loaded),
        allowed_missing_keys=tuple(allowed_missing),
        unclassified_missing_keys=tuple(unclassified_missing),
        allowed_unused_source_keys=tuple(allowed_unused),
        unexpected_keys=tuple(unexpected),
        shape_mismatch_keys=tuple(shape_mismatch),
        missing_required_keys=tuple(missing_required))
    module.pretrained_load_report = report

    logger.info('%s checkpoint: %s', description, report['checkpoint'])
    logger.info('%s loaded pretrained keys (%d): %s', description,
                len(loaded), list(loaded))
    logger.info('%s allowed missing keys (%d): %s', description,
                len(allowed_missing), allowed_missing)
    logger.info('%s unclassified missing keys (%d): %s', description,
                len(unclassified_missing), unclassified_missing)
    logger.info('%s allowed unused source keys (%d): %s', description,
                len(allowed_unused), allowed_unused)
    logger.info('%s unexpected keys (%d): %s', description,
                len(unexpected), unexpected)
    logger.info('%s shape mismatch keys (%d): %s', description,
                len(shape_mismatch), shape_mismatch)
    logger.info('%s missing required keys (%d): %s', description,
                len(missing_required), missing_required)

    errors = []
    if missing_required:
        errors.append(f'missing required keys={missing_required}')
    if unclassified_missing:
        errors.append(f'unclassified missing keys={unclassified_missing}')
    if shape_mismatch:
        errors.append(f'shape mismatches={shape_mismatch}')
    if fail_on_unexpected and unexpected:
        errors.append(f'unexpected keys={unexpected}')
    if errors:
        raise RuntimeError(
            f'Validated pretrained load failed for {description}: ' +
            '; '.join(errors))

    # strict=False is safe here: classification was completed above and only
    # compatible tensors are passed to PyTorch.
    module.load_state_dict(loaded, strict=False)
    return report

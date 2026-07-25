#!/usr/bin/env bash
set -euo pipefail

unset PYTORCH_CUDA_ALLOC_CONF
export CUDA_VISIBLE_DEVICES=1

REPO_DIR="/home/igarashi_25/DDPS"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
CONFIG="${REPO_DIR}/configs/cityscapes/segformer_b2_cityscapes_256x512_800ep_pretrain.py"
WORK_DIR="/home/igarashi_25/playground_2/DSDFM/DDPS/result/segformer_b2_pretrain_256x512_800ep"
CHECKPOINT="${1:-${WORK_DIR}/latest.pth}"

cd "${REPO_DIR}"
exec .venv/bin/python tools/test.py "${CONFIG}" "${CHECKPOINT}" \
    --work-dir "${WORK_DIR}/evaluation" --eval mIoU

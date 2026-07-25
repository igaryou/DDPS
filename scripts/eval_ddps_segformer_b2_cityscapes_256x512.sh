#!/usr/bin/env bash
set -euo pipefail

unset PYTORCH_CUDA_ALLOC_CONF
export CUDA_VISIBLE_DEVICES=1

REPO_DIR="/home/igarashi_25/DDPS"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
CONFIG="${REPO_DIR}/configs/cityscapes/ddps_segformer_b2_cityscapes_256x512_800ep.py"
WORK_DIR="/home/igarashi_25/playground_2/DSDFM/DDPS/result/ddps_segformer_b2_cityscapes_256x512_800ep_b4"
CHECKPOINT="${1:-${WORK_DIR}/latest.pth}"

cd "${REPO_DIR}"
exec .venv/bin/python tools/test_diffusion.py "${CONFIG}" "${CHECKPOINT}" \
    --work-dir "${WORK_DIR}/evaluation" --eval mIoU mAcc aAcc

#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

REPO_DIR="/home/igarashi_25/DDPS"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
CONFIG="${REPO_DIR}/configs/cityscapes/ddps_cityscapes_256x512_800ep.py"
WORK_DIR="/home/igarashi_25/playground_2/DSDFM/DDPS/result/cityscapes_256x512_800ep_b4"

args=("${CONFIG}" "--work-dir" "${WORK_DIR}")
if [[ "${1:-}" == "--resume" || "${1:-}" == "--resume-from" ]]; then
    [[ $# -ge 2 ]] || { echo "Missing checkpoint path" >&2; exit 2; }
    args+=("--resume-from" "$2")
    shift 2
fi
[[ $# -eq 0 ]] || { echo "Unknown arguments: $*" >&2; exit 2; }

cd "${REPO_DIR}"
exec .venv/bin/python tools/train_diffusion.py "${args[@]}" --auto-resume

#!/usr/bin/env bash
set -euo pipefail

unset PYTORCH_CUDA_ALLOC_CONF
export CUDA_VISIBLE_DEVICES=1

REPO_DIR="/home/igarashi_25/DDPS"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
CONFIG="${REPO_DIR}/configs/cityscapes/ddps_segformer_b2_cityscapes_256x512_800ep.py"
WORK_DIR="/home/igarashi_25/playground_2/DSDFM/DDPS/result/ddps_segformer_b2_cityscapes_256x512_800ep_b4"
STAGE1_CHECKPOINT="${DDPS_STAGE1_CHECKPOINT:-/home/igarashi_25/playground_2/DSDFM/DDPS/result/segformer_b2_pretrain_256x512_800ep/latest.pth}"
RESUME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume|--resume-from)
            [[ $# -ge 2 ]] || { echo "Missing resume checkpoint path" >&2; exit 2; }
            RESUME="$2"
            shift 2
            ;;
        --stage1-checkpoint|--pretrained)
            [[ $# -ge 2 ]] || { echo "Missing Stage-1 checkpoint path" >&2; exit 2; }
            STAGE1_CHECKPOINT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

args=("${CONFIG}" "--work-dir" "${WORK_DIR}")
if [[ -n "${RESUME}" ]]; then
    args+=("--resume-from" "${RESUME}")
fi
args+=("--cfg-options"
       "model.backbone.pretrained=${STAGE1_CHECKPOINT}"
       "model.decode_head.pretrained=${STAGE1_CHECKPOINT}")

cd "${REPO_DIR}"
exec .venv/bin/python tools/train_diffusion.py "${args[@]}" --auto-resume

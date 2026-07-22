#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

export HF_LEROBOT_HOME=/mnt/g1_training_dataset
export OPENPI_DATA_HOME=/mnt/openpi_base
export HF_DATASETS_CACHE=/mnt/hf-datasets-cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE=disabled

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE="${BATCH_SIZE:-192}"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    scripts/train_pytorch.py pi05_g1_sonic_full_movedoor_4cam \
    --exp-name=walk_clean_desk_mix_full_70_torch_ddp \
    --pytorch-weight-path=/mnt/openpi_base/openpi-pytorch/pi05_base \
    --pytorch-training-precision=float32 \
    --pytorch-autocast \
    --num-train-steps=1000 \
    --save-interval=5000 \
    --keep-period=25000 \
    --num-workers=4 \
    --batch-size="${BATCH_SIZE}" \
    --fsdp-devices=1

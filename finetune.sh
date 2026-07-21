#!/usr/bin/env bash
. .venv/bin/activate 
export HF_LEROBOT_HOME=/mnt/g1_training_dataset
export OPENPI_DATA_HOME=/mnt/openpi_base
export HF_DATASETS_CACHE=/mnt/hf-datasets-cache
# export HTTP_PROXY=http://127.0.0.1:7890
# export HTTPS_PROXY=http://127.0.0.1:7890
# export http_proxy=http://127.0.0.1:7890
# export https_proxy=http://127.0.0.1:7890
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
python scripts/train.py pi05_g1_sonic_full_walk_clean_desk_mix \
--exp-name walk_clean_desk_mix_full_70 \
--num-train-steps 50000 \
--save-interval 5000 \
--keep-period 25000 \
--num_workers=32 \
--fsdp_devices=8 \
--batch_size=256

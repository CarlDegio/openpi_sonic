#!/usr/bin/env bash
. .venv/bin/activate 
export HF_LEROBOT_HOME=/mnt/g1_training_dataset
export OPENPI_DATA_HOME=/mnt/openpi_base
export HF_DATASETS_CACHE=/mnt/hf-datasets-cache
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
python scripts/train.py pi05_g1_sonic_full_collect_pillow_4cam \
--exp-name collect_pillow_full \
--num-train-steps 30000 \
--save-interval 5000 \
--keep-period 15000 \
--num_workers=32 \
--fsdp_devices=8 \
--batch_size=256
# Pi0.5 PyTorch DDP Fine-tuning Design

## Goal

Add a PyTorch DDP launch script for the existing
`pi05_g1_sonic_full_walk_clean_desk_mix` training configuration while preserving
the initialization behavior of the working JAX training path.

## Weight Initialization

The converted base checkpoint at
`/mnt/openpi_base/openpi-pytorch/pi05_base` contains 32-dimensional action
projection weights, while the G1 SONIC configuration constructs a model with a
78-dimensional action space.

The PyTorch loader will mirror `ShapeAwareCheckpointWeightLoader`:

- Build the target 78-dimensional PyTorch model first.
- Load every checkpoint tensor whose name and shape match the target model.
- Keep the target model's initialized value only for shape-mismatched parameters
  matching the config's `reinit_mismatched_regexes`.
- Reject every other shape mismatch, missing parameter, or unexpected parameter.
- Preserve the existing FP32 checkpoint and model validation.

For the selected configuration, only `action_in_proj` and `action_out_proj` are
allowed to remain newly initialized. The model architecture itself is unchanged.

## DDP Launch Script

Create `finetune_torch.sh` with the same dataset and offline Hugging Face
environment settings as `finetune.sh`. It will launch
`scripts/train_pytorch.py` through single-node `torchrun` using:

- Config: `pi05_g1_sonic_full_walk_clean_desk_mix`
- Initial weights: `/mnt/openpi_base/openpi-pytorch/pi05_base`
- Precision: `float32`
- Default processes: 8
- Default global batch size: 32
- Training steps: 50,000
- Save interval: 5,000
- Keep period: 25,000
- Data-loader workers: 32

`NPROC_PER_NODE` and `BATCH_SIZE` will be environment-variable overrides so
memory experiments do not require editing the script.

## Validation

Add focused tests for shape-aware PyTorch state loading:

- Allowed action projection mismatches preserve initialized target values.
- Matching backbone tensors load from the checkpoint.
- Unapproved shape mismatches fail.
- Unexpected and unexplained missing keys fail.

Validate the shell script with `bash -n`. A full multi-GPU training run is not
part of automated validation, but the existing trainer's early-step memory logs
will report allocation after model creation and backward.

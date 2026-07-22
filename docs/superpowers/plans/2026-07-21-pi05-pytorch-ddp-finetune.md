# Pi0.5 PyTorch DDP Fine-tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add JAX-equivalent shape-aware PyTorch base-weight loading and an eight-GPU DDP launcher for the G1 SONIC mixed dataset.

**Architecture:** The trainer constructs the configured 78-dimensional target model, then a focused loader copies shape-compatible safetensors values while retaining initialized values only for mismatch patterns declared by `ShapeAwareCheckpointWeightLoader`. The shell launcher supplies the converted FP32 base checkpoint and exposes process count and global batch size as environment overrides.

**Tech Stack:** Python 3.11, PyTorch, safetensors, pytest, Bash, torchrun.

## Global Constraints

- Do not change the model architecture or G1 data transformations.
- Preserve full-FP32 validation.
- Reject every mismatch not explicitly allowed by the config.
- Do not create a Git commit; the user will review the working tree first.

---

### Task 1: Shape-aware PyTorch Weight Loading

**Files:**
- Modify: `scripts/train_pytorch.py`
- Create: `scripts/train_pytorch_weight_loading_test.py`

**Interfaces:**
- Consumes: `model: torch.nn.Module`, `checkpoint_path: pathlib.Path | str`, and the configured `WeightLoader`.
- Produces: `_load_initial_weights(model, checkpoint_path, weight_loader) -> None`.

- [ ] **Step 1: Write failing loader tests**

Create tiny linear models and safetensors checkpoints that verify matching backbone values load, allowed `action_in_proj` and `action_out_proj` mismatches retain initialized values, and unapproved mismatches or unexpected keys raise `RuntimeError`.

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/pytest -q scripts/train_pytorch_weight_loading_test.py`

Expected: FAIL because `_load_initial_weights` does not exist.

- [ ] **Step 3: Implement minimal loader**

Load tensors with `safetensors.torch.load_file`, compare against the unwrapped model state, filter only mismatch names accepted by `weight_loader.reinit_mismatched_regexes`, call `load_state_dict(..., strict=False)`, and validate that missing and unexpected keys are exactly explained.

- [ ] **Step 4: Use loader during initial fine-tuning setup**

Replace the direct `safetensors.torch.load_model` call in `train_loop` while retaining `_validate_fp32_training_state` before and after loading.

- [ ] **Step 5: Run focused and precision tests**

Run: `.venv/bin/pytest -q scripts/train_pytorch_weight_loading_test.py scripts/train_pytorch_precision_test.py`

Expected: all tests PASS.

### Task 2: DDP Launch Script

**Files:**
- Create: `finetune_torch.sh`

**Interfaces:**
- Consumes: optional `NPROC_PER_NODE` and `BATCH_SIZE` environment variables.
- Produces: a single-node `torchrun` launch of `scripts/train_pytorch.py`.

- [ ] **Step 1: Create the launcher**

Activate `.venv`, set the existing dataset/cache/offline environment, default `NPROC_PER_NODE=8` and `BATCH_SIZE=32`, and launch `pi05_g1_sonic_full_walk_clean_desk_mix` with the converted FP32 checkpoint, 50,000 steps, 5,000-step saves, 25,000-step retention, and 32 workers.

- [ ] **Step 2: Validate shell syntax and CLI arguments**

Run: `bash -n finetune_torch.sh` and `.venv/bin/python scripts/train_pytorch.py pi05_g1_sonic_full_walk_clean_desk_mix --help`.

Expected: shell syntax succeeds and every option used by the launcher appears in CLI help.

- [ ] **Step 3: Review the final diff**

Run: `git diff --check` and `git diff -- scripts/train_pytorch.py scripts/train_pytorch_weight_loading_test.py finetune_torch.sh`.

Expected: no whitespace errors and only the approved loader, tests, and launcher changes.

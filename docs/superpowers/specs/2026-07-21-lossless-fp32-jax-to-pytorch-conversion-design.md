# Lossless FP32 JAX-to-PyTorch Conversion Design

## Context

`examples/convert_jax_model_to_pytorch.py` restores JAX/Orbax parameters as
FP32, but constructs `PI0Pytorch` from the original model config before loading
them. The default config dtype is BF16. PyTorch `load_state_dict` copies source
tensors into the destination parameter dtype, so most FP32 values can be
rounded to BF16 before the script later calls `model.to(torch.float32)`. The
saved file then reports FP32 tensors even though many values only retain BF16
precision.

The target workflow is full-FP32 PyTorch training initialized from pi0 or pi0.5
JAX weights without conversion-induced rounding.

The PyTorch action expert also currently constructs a tied language-model head,
then removes the embedding to which that head was tied. The expert forward path
always calls the decoder model directly and never uses this orphaned head. JAX
therefore has no source tensor for it, but it remains as a large randomly
initialized parameter in the FP32 PyTorch state dict. A strict conversion
coverage check must remove this FP32-only artifact rather than bless it as a
valid missing key.

## Goals

- Preserve every mapped FP32 JAX parameter value exactly through PyTorch model
  loading and safetensors serialization.
- Cover both pi0 and pi0.5 through their shared conversion path.
- Fail conversion instead of silently retaining unexplained randomly
  initialized parameters.
- Prevent an FP32 PyTorch training run from silently loading a BF16 or FP16
  safetensors checkpoint.
- Keep the training-side change local to the PyTorch entrypoint.

## Non-goals

- Do not change JAX model code, JAX checkpoint loading, JAX training, or any
  pi0/pi0.5 JAX configuration.
- Do not add or modify fields in `TrainConfig`.
- Do not add AMP, autocast, `GradScaler`, FP32 master weights, or a new mixed
  precision policy.
- Do not implement float16 conversion. The existing float16 request continues
  to fail as unsupported.
- Do not redesign parameter-name or layout mappings beyond changes required to
  eliminate unexplained missing parameters.

## Selected Approach

Use the requested output precision to construct the PyTorch model before
calling `load_state_dict`.

The converter creates a detached configuration with:

```python
conversion_config = dataclasses.replace(
    model_config,
    dtype=precision,
    pytorch_compile_mode=None,
)
```

It then constructs `PI0Pytorch(conversion_config)` and loads the mapped state
dict. In FP32 mode, every destination parameter is FP32 before the first copy,
so there is no intermediate BF16 rounding. Disabling compilation avoids
compiling an inference method for a model used only as a conversion container.

BF16 conversion retains the existing behavior, including the action expert's
legacy head and the final whole-model BF16 cast. In FP32 models only, the unused
action-expert `lm_head` is removed. A narrowly scoped load-state pre-hook drops
that one legacy key when an older FP32 PyTorch checkpoint is loaded, so existing
FP32 checkpoints remain loadable. Float16 retains the existing
unsupported-precision error.

An alternative that always constructs an FP32 model was rejected because it
would impose the FP32 peak-memory cost on BF16 conversions. Directly writing
the mapped state dict was rejected because it would weaken model shape checks
and complicate tied-weight handling.

## Conversion Data Flow

1. Restore Orbax parameters as NumPy FP32 arrays using the existing restore
   path.
2. Apply the existing JAX-to-PyTorch name and layout transformations.
3. Construct a pi0 or pi0.5 PyTorch model using a copied config whose dtype is
   the requested output precision.
4. For FP32, remove the unused action-expert `lm_head`, load the mapped tensors,
   and validate the returned incompatible-key report. BF16 keeps its current
   permissive load behavior.
5. In FP32 mode, compare every mapped floating tensor against the corresponding
   model tensor with exact equality.
6. Apply the existing final output cast. For FP32 this is idempotent; for BF16
   it preserves current behavior.
7. Save with `safetensors.torch.save_model` so tied storage remains supported.
8. Read the safetensors header and verify that every stored floating tensor is
   FP32 when FP32 output was requested.

## State-Dict Integrity Policy

- For lossless FP32 conversion, unexpected input keys are always fatal.
- A missing model key is accepted only when it is a tied alias whose storage is
  demonstrably shared with a successfully loaded key.
- Any missing non-aliased parameter is fatal, including parameters that the
  current forward path does not use. The unused action-expert `lm_head` is
  removed from FP32 models before this validation, so the converter does not
  publish random trainable state or add an exception to the coverage rule.
- Validation errors identify every offending key rather than reporting only
  the first one.

The coverage check is FP32-only. This preserves the exact BF16 conversion path,
whose legacy action-expert head has no JAX counterpart. Tightening BF16 state
coverage is outside this change.

For FP32 output, each mapped floating source tensor must satisfy all of the
following after model loading:

- The destination key exists.
- Source and destination shapes match.
- Source and destination dtypes are FP32.
- `torch.equal(source, destination)` is true.

If equality fails, the error includes the key, both dtypes, both shapes, and
the maximum absolute difference.

## PyTorch Training Guards

Training guards live only in `scripts/train_pytorch.py` and use the existing
`config.pytorch_training_precision` value.

When the requested training precision is FP32:

1. Before loading `pytorch_weight_path/model.safetensors`, inspect its header
   and reject every stored floating dtype other than FP32.
2. Before loading a resume checkpoint, apply the same source-file check.
3. After model construction and any model-weight load, require all floating
   parameters and floating buffers to be `torch.float32`.
4. Run the model-state check before creating a new optimizer and again after
   resume restoration.

Integer and boolean tensors are permitted. BF16 training bypasses these FP32
guards and keeps its current behavior.

The source-file check is required because loading a BF16 tensor into an FP32
parameter produces an FP32 container but cannot restore the discarded BF16
mantissa bits.

## Error Handling

Failures are raised before a converted checkpoint is advertised as complete or
before an FP32 training optimizer step can run. Messages group errors by:

- unsupported precision;
- unexpected or unexplained missing state-dict keys;
- FP32 value mismatch;
- non-FP32 safetensors entries; and
- non-FP32 model parameters or buffers.

Partial output directories retain the existing conversion behavior; no new
cleanup or checkpoint transaction mechanism is introduced.

## Test Strategy

Fast regression tests use a small fake PyTorch model rather than allocating the
full multi-billion-parameter pi0 model.

Converter tests cover:

- The old ordering reproduces FP32-to-BF16 rounding before an FP32 upcast.
- FP32 target construction preserves representative source values exactly.
- Both pi0 and pi0.5 configs are copied with the requested dtype without
  mutating their original objects.
- `pytorch_compile_mode` is disabled only on the copied conversion config.
- BF16 output retains the current final dtype behavior.
- Unexpected keys and missing non-aliased parameters are rejected.
- A mechanically verified tied alias is accepted.
- FP32 models remove the unused action-expert head and accept that one key from
  legacy FP32 checkpoints through the compatibility hook.
- BF16 models retain the legacy head and existing conversion behavior.
- FP32 equality errors report the tensor name and maximum absolute difference.
- The saved-header validator accepts FP32 plus integer/boolean tensors and
  rejects BF16/FP16 tensors.

Training-guard tests cover:

- FP32 mode accepts an all-FP32 model and checkpoint header.
- FP32 mode rejects a BF16 parameter, floating buffer, or checkpoint tensor.
- BF16 mode does not invoke the FP32-only rejection path.
- DDP-wrapped and unwrapped model access follows the existing training helper
  pattern.

The slow acceptance test converts
`/mnt/openpi_base/openpi-assets/checkpoints/pi05_base` to FP32, verifies the
saved header, loads it into an FP32 `PI0Pytorch`, and confirms that every
floating parameter and buffer is FP32. It also runs the converter's exact
per-key equality validation during conversion.

## Acceptance Criteria

- FP32 conversion never copies an FP32 mapped tensor into a BF16/FP16
  destination parameter.
- Every mapped FP32 tensor is exactly equal before and after model loading.
- Every floating tensor stored in the FP32 safetensors file has dtype FP32.
- pi0 and pi0.5 use the same corrected conversion path.
- FP32 training rejects non-FP32 source checkpoints and non-FP32 in-memory
  model state before optimization.
- BF16 conversion and BF16 training behavior remain unchanged, including the
  existing expert-head state layout.
- JAX code and configuration have no diff.

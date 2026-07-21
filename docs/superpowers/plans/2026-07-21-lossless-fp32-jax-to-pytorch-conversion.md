# Lossless FP32 JAX-to-PyTorch Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert pi0 and pi0.5 JAX checkpoints to genuine FP32 PyTorch checkpoints without an intermediate BF16 copy, and prevent FP32 PyTorch training from silently loading reduced-precision state.

**Architecture:** The converter will construct `PI0Pytorch` from a copied config whose dtype already matches the requested output precision, then validate state-dict coverage and exact FP32 values before saving. FP32 models will omit the action expert's unused, JAX-less `lm_head`, with a narrow compatibility hook for older FP32 PyTorch checkpoints; BF16 keeps its current layout. The PyTorch training entrypoint will add local, configuration-free guards that inspect safetensors headers before loading and inspect model state before optimization. JAX code and configuration remain untouched.

**Tech Stack:** Python 3.11, PyTorch 2.7.1, safetensors 0.5.3, Orbax 0.11.13, pytest 8, Ruff.

## Global Constraints

- Do not change JAX model code, JAX checkpoint loading, JAX training, or any pi0/pi0.5 JAX configuration.
- Do not add or modify fields in `TrainConfig`.
- Keep the primary behavioral change in `examples/convert_jax_model_to_pytorch.py`.
- Keep training guards local to `scripts/train_pytorch.py` and use the existing `pytorch_training_precision` value.
- Cover both pi0 and pi0.5 through the same conversion helper.
- Preserve current BF16 behavior and state layout.
- Leave float16 unsupported and preserve its current `ValueError` behavior.
- Do not add autocast, `GradScaler`, FP32 master weights, or another precision mode.
- Preserve unrelated user changes in the dirty worktree.

## File Structure

- Modify `examples/convert_jax_model_to_pytorch.py`: target-dtype model construction, state-dict integrity checks, exact FP32 checks, and saved-header checks.
- Create `scripts/convert_jax_model_to_pytorch_test.py`: small-model regression tests for conversion ordering, pi0/pi0.5 config copying, state-dict coverage, tied aliases, and safetensors dtype validation.
- Modify `src/openpi/models_pytorch/gemma_pytorch.py`: remove the unused action-expert `lm_head` in FP32 models only and ignore that one legacy key while loading older FP32 checkpoints.
- Create `src/openpi/models_pytorch/gemma_pytorch_test.py`: focused tests for FP32 head removal compatibility without allocating a full pi0 model.
- Modify `scripts/train_pytorch.py`: FP32 safetensors and in-memory model guards at initial load and resume boundaries.
- Create `scripts/train_pytorch_precision_test.py`: focused unit tests for the PyTorch-only training guards.

---

### Task 1: Construct the Conversion Model at the Requested Precision

**Files:**
- Modify: `examples/convert_jax_model_to_pytorch.py:29-47, 422-527`
- Create: `scripts/convert_jax_model_to_pytorch_test.py`

**Interfaces:**
- Consumes: `openpi.models.pi0_config.Pi0Config`, requested precision string, and `dict[str, torch.Tensor]` produced by the existing mapping functions.
- Produces: `_create_converted_model(model_config, precision, state_dict) -> torch.nn.Module` and `_validate_exact_fp32_values(model, state_dict) -> None`.

- [ ] **Step 1: Add regression tests that reproduce the old rounding and specify the corrected behavior**

Create `scripts/convert_jax_model_to_pytorch_test.py` with the following initial content:

```python
import dataclasses

import pytest
import torch

from examples import convert_jax_model_to_pytorch as converter


@dataclasses.dataclass(frozen=True)
class _FakeConfig:
    dtype: str = "bfloat16"
    pi05: bool = False
    pytorch_compile_mode: str | None = "max-autotune"


class _TinyModel(torch.nn.Module):
    def __init__(self, config: _FakeConfig):
        super().__init__()
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[config.dtype]
        self.weight = torch.nn.Parameter(torch.empty((1, 4), dtype=dtype))
        self.bias = torch.nn.Parameter(torch.empty((1,), dtype=dtype))
        self.config = config


def _source_state() -> dict[str, torch.Tensor]:
    return {
        "weight": torch.tensor([[1.001, 1.003, 0.1234567, -3.1415927]], dtype=torch.float32),
        "bias": torch.tensor([0.33333334], dtype=torch.float32),
    }


def test_old_load_order_loses_fp32_precision():
    source = _source_state()
    model = _TinyModel(_FakeConfig())
    model.load_state_dict(source)
    model.to(torch.float32)

    assert not torch.equal(model.weight, source["weight"])


@pytest.mark.parametrize("pi05", [False, True])
def test_create_converted_model_preserves_fp32_values_and_original_config(monkeypatch, pi05: bool):
    monkeypatch.setattr(converter.openpi.models_pytorch.pi0_pytorch, "PI0Pytorch", _TinyModel)
    config = _FakeConfig(pi05=pi05)
    source = _source_state()

    model = converter._create_converted_model(config, "float32", source)

    assert model.config.dtype == "float32"
    assert model.config.pi05 is pi05
    assert model.config.pytorch_compile_mode is None
    assert config.dtype == "bfloat16"
    assert config.pytorch_compile_mode == "max-autotune"
    assert torch.equal(model.weight, source["weight"])
    assert torch.equal(model.bias, source["bias"])


def test_create_converted_model_preserves_bfloat16_output(monkeypatch):
    monkeypatch.setattr(converter.openpi.models_pytorch.pi0_pytorch, "PI0Pytorch", _TinyModel)
    source = _source_state()

    model = converter._create_converted_model(_FakeConfig(), "bfloat16", source)

    assert model.weight.dtype == torch.bfloat16
    assert model.bias.dtype == torch.bfloat16
    assert torch.equal(model.weight, source["weight"].to(torch.bfloat16))


class _PerturbingTinyModel(_TinyModel):
    def load_state_dict(self, state_dict, strict=True, assign=False):
        incompatible = super().load_state_dict(state_dict, strict=strict, assign=assign)
        with torch.no_grad():
            self.weight.add_(0.25)
        return incompatible


def test_exact_fp32_validation_reports_key_and_error(monkeypatch):
    monkeypatch.setattr(converter.openpi.models_pytorch.pi0_pytorch, "PI0Pytorch", _PerturbingTinyModel)

    with pytest.raises(RuntimeError, match=r"weight.*max_abs_diff=0.25"):
        converter._create_converted_model(_FakeConfig(), "float32", _source_state())


def test_float16_remains_unsupported(monkeypatch):
    monkeypatch.setattr(converter.openpi.models_pytorch.pi0_pytorch, "PI0Pytorch", _TinyModel)

    with pytest.raises(ValueError, match="Invalid precision: float16"):
        converter._create_converted_model(_FakeConfig(), "float16", _source_state())
```

- [ ] **Step 2: Run the focused tests and verify that the new helper is missing**

Run:

```bash
JAX_PLATFORMS=cpu .venv/bin/pytest scripts/convert_jax_model_to_pytorch_test.py -v
```

Expected: the old-order test passes, while the new tests fail with `AttributeError: module ... has no attribute '_create_converted_model'`.

- [ ] **Step 3: Implement requested-precision construction and exact FP32 validation**

Add `import dataclasses` to `examples/convert_jax_model_to_pytorch.py`, then add these helpers immediately before `convert_pi0_checkpoint`:

```python
_TORCH_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _validate_exact_fp32_values(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    model_state = model.state_dict()
    errors = []
    for key, source in state_dict.items():
        if not source.is_floating_point():
            continue
        destination = model_state.get(key)
        if destination is None:
            errors.append(f"{key}: destination key is missing")
            continue
        if source.shape != destination.shape:
            errors.append(f"{key}: source_shape={tuple(source.shape)} destination_shape={tuple(destination.shape)}")
            continue
        if source.dtype != torch.float32 or destination.dtype != torch.float32:
            errors.append(f"{key}: source_dtype={source.dtype} destination_dtype={destination.dtype}")
            continue
        if not torch.equal(source, destination):
            max_abs_diff = torch.max(torch.abs(source - destination)).item()
            errors.append(f"{key}: max_abs_diff={max_abs_diff}")
    if errors:
        raise RuntimeError("FP32 conversion changed mapped values:\n" + "\n".join(errors))


def _create_converted_model(
    model_config: openpi.models.pi0_config.Pi0Config,
    precision: str,
    state_dict: dict[str, torch.Tensor],
) -> torch.nn.Module:
    target_dtype = _TORCH_DTYPES.get(precision)
    if target_dtype is None:
        raise ValueError(f"Invalid precision: {precision}")

    conversion_config = dataclasses.replace(
        model_config,
        dtype=precision,
        pytorch_compile_mode=None,
    )
    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(conversion_config)
    model.load_state_dict(state_dict, strict=False)

    if precision == "float32":
        _validate_exact_fp32_values(model, state_dict)

    return model.to(target_dtype)
```

Replace the model construction, load, and final cast at the current lines 513-527 with:

```python
    # Combine all parameters (no prefix needed for our model structure)
    all_params = {**paligemma_params, **gemma_params, **projection_params}

    # Construct at the requested dtype before copying values into parameters.
    pi0_model = _create_converted_model(model_config, precision, all_params)
```

- [ ] **Step 4: Run the focused tests and verify exact FP32 preservation**

Run:

```bash
JAX_PLATFORMS=cpu .venv/bin/pytest scripts/convert_jax_model_to_pytorch_test.py -v
```

Expected: `6 passed`.

- [ ] **Step 5: Run formatting and lint checks for the task files**

Run:

```bash
.venv/bin/ruff format examples/convert_jax_model_to_pytorch.py scripts/convert_jax_model_to_pytorch_test.py
.venv/bin/ruff check examples/convert_jax_model_to_pytorch.py scripts/convert_jax_model_to_pytorch_test.py
```

Expected: formatting succeeds and Ruff reports `All checks passed!`.

- [ ] **Step 6: Commit the precision-ordering fix**

```bash
git add examples/convert_jax_model_to_pytorch.py scripts/convert_jax_model_to_pytorch_test.py
git commit -m "fix: preserve fp32 values during checkpoint conversion"
```

---

### Task 2: Remove the FP32-Only Orphan Head and Validate Checkpoint Integrity

**Files:**
- Modify: `src/openpi/models_pytorch/gemma_pytorch.py:10-61`
- Create: `src/openpi/models_pytorch/gemma_pytorch_test.py`
- Modify: `examples/convert_jax_model_to_pytorch.py`
- Modify: `scripts/convert_jax_model_to_pytorch_test.py`

**Interfaces:**
- Consumes: the `_create_converted_model` and `_validate_exact_fp32_values` helpers from Task 1.
- Produces: `_remove_orphaned_expert_lm_head(module) -> None`, `_validate_state_dict_coverage(model, state_dict, incompatible_keys) -> None`, and `_validate_fp32_safetensors(path) -> None`.

- [ ] **Step 1: Add failing tests for FP32 expert-head removal and legacy loading**

Create `src/openpi/models_pytorch/gemma_pytorch_test.py` using tiny `torch.nn.Module` fixtures. Do not construct the full PaliGemma model. The helper-level tests should build this shape:

```python
class _TinyExpertContainer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.gemma_expert = torch.nn.Module()
        self.gemma_expert.lm_head = torch.nn.Linear(1, 1, bias=False)
```

Test these behaviors through the planned `_remove_orphaned_expert_lm_head` helper:

```python
def test_remove_orphaned_expert_lm_head_removes_parameter():
    model = _TinyExpertContainer()

    gemma_pytorch._remove_orphaned_expert_lm_head(model)

    assert model.gemma_expert.lm_head is None
    assert "gemma_expert.lm_head.weight" not in model.state_dict()


def test_remove_orphaned_expert_lm_head_accepts_legacy_key_when_nested():
    model = torch.nn.Module()
    model.child = _TinyExpertContainer()
    gemma_pytorch._remove_orphaned_expert_lm_head(model.child)
    legacy_state = {
        "child.weight": torch.full((1,), 2.0),
        "child.gemma_expert.lm_head.weight": torch.full((1, 1), 3.0),
    }

    model.load_state_dict(legacy_state, strict=True)

    assert torch.equal(model.child.weight, torch.full((1,), 2.0))
```

Also monkeypatch `PaliGemmaForConditionalGeneration` and `GemmaForCausalLM` with tiny constructor-compatible modules, then instantiate `PaliGemmaWithExpertModel` with minimal fake VLM/expert configs. Assert that `precision="float32"` removes `gemma_expert.lm_head`, while `precision="bfloat16"` retains it and casts it to BF16. This proves the precision branch without allocating the real model.

Run:

```bash
JAX_PLATFORMS=cpu .venv/bin/pytest src/openpi/models_pytorch/gemma_pytorch_test.py -v
```

Expected: the tests fail because `_remove_orphaned_expert_lm_head` does not exist and the FP32 constructor still retains the head.

- [ ] **Step 2: Implement FP32-only removal with a one-key compatibility hook**

Add a module-level helper to `src/openpi/models_pytorch/gemma_pytorch.py`. It must:

1. Set `module.gemma_expert.lm_head = None` so the unused parameter is absent from FP32 `state_dict()` output.
2. Register a `load_state_dict` pre-hook on `module` that removes exactly `f"{prefix}gemma_expert.lm_head.weight"` from an incoming legacy state dict.
3. Leave every other unexpected key untouched so strict loading still rejects it.

Call the helper in `PaliGemmaWithExpertModel.__init__` only when `precision == "float32"`, immediately after `self.gemma_expert.model.embed_tokens = None`. Do not call it for BF16.

Run:

```bash
JAX_PLATFORMS=cpu .venv/bin/pytest src/openpi/models_pytorch/gemma_pytorch_test.py -v
```

Expected: all expert-head tests pass; record the exact count from pytest rather than hard-coding it in the handoff.

- [ ] **Step 3: Add failing coverage and safetensors-header tests**

Append to `scripts/convert_jax_model_to_pytorch_test.py`:

```python
from safetensors.torch import save_file


class _TiedTinyModel(torch.nn.Module):
    def __init__(self, config: _FakeConfig):
        super().__init__()
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[config.dtype]
        self.weight = torch.nn.Parameter(torch.empty((1, 4), dtype=dtype))
        self.alias = self.weight
        self.config = config


def test_unexpected_state_dict_key_is_rejected(monkeypatch):
    monkeypatch.setattr(converter.openpi.models_pytorch.pi0_pytorch, "PI0Pytorch", _TinyModel)
    source = {**_source_state(), "unexpected": torch.ones(1)}

    with pytest.raises(RuntimeError, match=r"unexpected_keys=.*unexpected"):
        converter._create_converted_model(_FakeConfig(), "float32", source)


def test_missing_non_alias_parameter_is_rejected(monkeypatch):
    monkeypatch.setattr(converter.openpi.models_pytorch.pi0_pytorch, "PI0Pytorch", _TinyModel)
    source = {"weight": _source_state()["weight"]}

    with pytest.raises(RuntimeError, match=r"missing_keys=.*bias"):
        converter._create_converted_model(_FakeConfig(), "float32", source)


def test_missing_tied_alias_is_accepted(monkeypatch):
    monkeypatch.setattr(converter.openpi.models_pytorch.pi0_pytorch, "PI0Pytorch", _TiedTinyModel)
    source = {"weight": _source_state()["weight"]}

    model = converter._create_converted_model(_FakeConfig(), "float32", source)

    assert torch.equal(model.alias, source["weight"])


def test_fp32_safetensors_header_accepts_float32_and_integer_tensors(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({"weight": torch.ones(2), "index": torch.ones(1, dtype=torch.int64)}, path)

    converter._validate_fp32_safetensors(path)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float64])
def test_fp32_safetensors_header_rejects_non_fp32_float(tmp_path, dtype: torch.dtype):
    path = tmp_path / "model.safetensors"
    save_file({"weight": torch.ones(2, dtype=dtype)}, path)

    with pytest.raises(RuntimeError, match=r"weight"):
        converter._validate_fp32_safetensors(path)
```

Move the new `save_file` import into the import block after running Ruff formatting.

- [ ] **Step 4: Run the new converter tests and verify the missing validation failures**

Run:

```bash
JAX_PLATFORMS=cpu .venv/bin/pytest scripts/convert_jax_model_to_pytorch_test.py -v
```

Expected: the three coverage tests and the header tests fail because the validators do not exist or are not called.

- [ ] **Step 5: Implement mechanical tied-alias coverage validation for FP32 only**

Add `from safetensors import safe_open` to the converter imports. Add these definitions above `_validate_exact_fp32_values`:

```python
def _storage_signature(tensor: torch.Tensor) -> tuple:
    storage = tensor.untyped_storage()
    return (
        storage.data_ptr(),
        storage.nbytes(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


def _validate_state_dict_coverage(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    incompatible_keys,
) -> None:
    model_state = model.state_dict()
    loaded_storage = {
        _storage_signature(model_state[key])
        for key in state_dict
        if key in model_state
    }
    unexplained_missing = [
        key
        for key in incompatible_keys.missing_keys
        if key not in model_state or _storage_signature(model_state[key]) not in loaded_storage
    ]
    if incompatible_keys.unexpected_keys or unexplained_missing:
        raise RuntimeError(
            "Converted state dict does not cover the model: "
            f"missing_keys={sorted(unexplained_missing)}, "
            f"unexpected_keys={sorted(incompatible_keys.unexpected_keys)}"
        )
```

Change `_create_converted_model` to capture and validate the load result before checking FP32 values:

```python
    incompatible_keys = model.load_state_dict(state_dict, strict=False)

    if precision == "float32":
        _validate_state_dict_coverage(model, state_dict, incompatible_keys)
        _validate_exact_fp32_values(model, state_dict)
```

Keeping `_validate_state_dict_coverage` inside the FP32 branch is deliberate: BF16 retains its existing permissive load behavior and its legacy expert head, which has no JAX source key.

- [ ] **Step 6: Implement header-only FP32 safetensors validation and wire it after saving**

Add this helper after `_validate_state_dict_coverage`:

```python
def _validate_fp32_safetensors(path: pathlib.Path | str) -> None:
    invalid = {}
    with safe_open(path, framework="pt", device="cpu") as tensors:
        for key in tensors.keys():
            dtype = tensors.get_slice(key).get_dtype()
            if (dtype.startswith("F") or dtype == "BF16") and dtype != "F32":
                invalid[key] = dtype
    if invalid:
        details = ", ".join(f"{key}={dtype}" for key, dtype in sorted(invalid.items()))
        raise RuntimeError(f"Expected an FP32 safetensors file, found non-FP32 floating tensors: {details}")
```

Replace the save call with a named path and validate only FP32 output:

```python
    model_path = pathlib.Path(output_path) / "model.safetensors"
    safetensors.torch.save_model(pi0_model, model_path)
    if precision == "float32":
        _validate_fp32_safetensors(model_path)
```

- [ ] **Step 7: Run focused tests, then all lightweight project tests affected by imports**

Run:

```bash
JAX_PLATFORMS=cpu .venv/bin/pytest scripts/convert_jax_model_to_pytorch_test.py src/openpi/models_pytorch/gemma_pytorch_test.py src/openpi/models/model_test.py -v
```

Expected: all selected tests pass. The exact number is the sum reported by pytest; there must be zero failures and zero errors.

- [ ] **Step 8: Run formatting and lint checks**

```bash
.venv/bin/ruff format examples/convert_jax_model_to_pytorch.py scripts/convert_jax_model_to_pytorch_test.py src/openpi/models_pytorch/gemma_pytorch.py src/openpi/models_pytorch/gemma_pytorch_test.py
.venv/bin/ruff check examples/convert_jax_model_to_pytorch.py scripts/convert_jax_model_to_pytorch_test.py src/openpi/models_pytorch/gemma_pytorch.py src/openpi/models_pytorch/gemma_pytorch_test.py
```

Expected: `All checks passed!`.

- [ ] **Step 9: Commit integrity validation and FP32 model cleanup**

```bash
git add examples/convert_jax_model_to_pytorch.py scripts/convert_jax_model_to_pytorch_test.py src/openpi/models_pytorch/gemma_pytorch.py src/openpi/models_pytorch/gemma_pytorch_test.py
git commit -m "fix: validate fp32 checkpoint integrity"
```

---

### Task 3: Add FP32-Only Guards to the PyTorch Training Entrypoint

**Files:**
- Modify: `scripts/train_pytorch.py:25-45, 135-250, 392-470`
- Create: `scripts/train_pytorch_precision_test.py`

**Interfaces:**
- Consumes: an optional safetensors path, a PyTorch model that may be DDP-wrapped, and the existing precision string.
- Produces: `_validate_fp32_safetensors(path) -> None`, `_validate_fp32_model(model) -> None`, and `_validate_fp32_training_state(model, precision, checkpoint_path=None) -> None`.

- [ ] **Step 1: Add failing unit tests for source-file, parameter, buffer, BF16 bypass, and wrapper behavior**

Create `scripts/train_pytorch_precision_test.py`:

```python
import pathlib

import pytest
import torch
from safetensors.torch import save_file

from scripts import train_pytorch


class _TinyTrainingModel(torch.nn.Module):
    def __init__(self, parameter_dtype=torch.float32, buffer_dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, dtype=parameter_dtype))
        self.register_buffer("running", torch.ones(2, dtype=buffer_dtype))
        self.register_buffer("index", torch.ones(1, dtype=torch.int64))


def test_fp32_training_state_accepts_fp32_model_and_file(tmp_path: pathlib.Path):
    path = tmp_path / "model.safetensors"
    save_file({"weight": torch.ones(2), "index": torch.ones(1, dtype=torch.int64)}, path)

    train_pytorch._validate_fp32_training_state(_TinyTrainingModel(), "float32", path)


def test_fp32_training_state_rejects_bfloat16_source_file(tmp_path: pathlib.Path):
    path = tmp_path / "model.safetensors"
    save_file({"weight": torch.ones(2, dtype=torch.bfloat16)}, path)

    with pytest.raises(RuntimeError, match=r"weight=BF16"):
        train_pytorch._validate_fp32_training_state(_TinyTrainingModel(), "float32", path)


@pytest.mark.parametrize(
    ("parameter_dtype", "buffer_dtype", "expected"),
    [
        (torch.bfloat16, torch.float32, "parameter:weight"),
        (torch.float32, torch.float16, "buffer:running"),
    ],
)
def test_fp32_training_state_rejects_non_fp32_model_state(parameter_dtype, buffer_dtype, expected):
    model = _TinyTrainingModel(parameter_dtype=parameter_dtype, buffer_dtype=buffer_dtype)

    with pytest.raises(RuntimeError, match=expected):
        train_pytorch._validate_fp32_training_state(model, "float32")


def test_bfloat16_training_bypasses_fp32_guards(tmp_path: pathlib.Path):
    path = tmp_path / "model.safetensors"
    save_file({"weight": torch.ones(2, dtype=torch.bfloat16)}, path)
    model = _TinyTrainingModel(parameter_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)

    train_pytorch._validate_fp32_training_state(model, "bfloat16", path)


def test_fp32_training_state_unwraps_ddp(monkeypatch):
    class _FakeDDP(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    monkeypatch.setattr(train_pytorch.torch.nn.parallel, "DistributedDataParallel", _FakeDDP)

    train_pytorch._validate_fp32_training_state(_FakeDDP(_TinyTrainingModel()), "float32")
```

- [ ] **Step 2: Run the focused tests and verify the guard is missing**

Run:

```bash
JAX_PLATFORMS=cpu .venv/bin/pytest scripts/train_pytorch_precision_test.py -v
```

Expected: tests fail with `AttributeError` for `_validate_fp32_training_state`.

- [ ] **Step 3: Implement local FP32 file and model guards**

Add `from safetensors import safe_open` to `scripts/train_pytorch.py`. Add these definitions after `get_model_parameters`:

```python
def _validate_fp32_safetensors(path: pathlib.Path | str) -> None:
    invalid = {}
    with safe_open(path, framework="pt", device="cpu") as tensors:
        for key in tensors.keys():
            dtype = tensors.get_slice(key).get_dtype()
            if (dtype.startswith("F") or dtype == "BF16") and dtype != "F32":
                invalid[key] = dtype
    if invalid:
        details = ", ".join(f"{key}={dtype}" for key, dtype in sorted(invalid.items()))
        raise RuntimeError(f"FP32 training requires an FP32 checkpoint; found: {details}")


def _validate_fp32_model(model: torch.nn.Module) -> None:
    model_to_check = (
        model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    )
    invalid = []
    for name, parameter in model_to_check.named_parameters():
        if parameter.is_floating_point() and parameter.dtype != torch.float32:
            invalid.append(f"parameter:{name}={parameter.dtype}")
    for name, buffer in model_to_check.named_buffers():
        if buffer.is_floating_point() and buffer.dtype != torch.float32:
            invalid.append(f"buffer:{name}={buffer.dtype}")
    if invalid:
        raise RuntimeError("FP32 training found non-FP32 model state: " + ", ".join(sorted(invalid)))


def _validate_fp32_training_state(
    model: torch.nn.Module,
    precision: str,
    checkpoint_path: pathlib.Path | str | None = None,
) -> None:
    if precision != "float32":
        return
    if checkpoint_path is not None:
        _validate_fp32_safetensors(checkpoint_path)
    _validate_fp32_model(model)
```

- [ ] **Step 4: Wire the guard around initial PyTorch weight loading and before optimizer creation**

Change the initial load block to validate the source before loading and validate the model after loading:

```python
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")

        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        _validate_fp32_training_state(model, config.pytorch_training_precision, model_path)
        safetensors.torch.load_model(
            (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model), model_path
        )
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    _validate_fp32_training_state(model, config.pytorch_training_precision)

    # Optimizer + learning rate schedule from config
```

This leaves all `TrainConfig` definitions unchanged.

- [ ] **Step 5: Wire the same guard into resume loading**

Change the signature:

```python
def load_checkpoint(model, optimizer, checkpoint_dir, device, *, precision: str):
```

Immediately before `safetensors.torch.load_model` in `load_checkpoint`, add:

```python
            _validate_fp32_training_state(model, precision, safetensors_path)
```

Immediately after that load succeeds, add:

```python
            _validate_fp32_training_state(model, precision)
```

Change the resume call site to:

```python
        global_step = load_checkpoint(
            model,
            optim,
            config.checkpoint_dir,
            device,
            precision=config.pytorch_training_precision,
        )
```

- [ ] **Step 6: Run training-guard tests**

Run:

```bash
JAX_PLATFORMS=cpu .venv/bin/pytest scripts/train_pytorch_precision_test.py -v
```

Expected: `6 passed`.

- [ ] **Step 7: Run existing script tests and static checks**

Run:

```bash
JAX_PLATFORMS=cpu .venv/bin/pytest scripts/train_pytorch_precision_test.py scripts/convert_jax_model_to_pytorch_test.py -v
.venv/bin/ruff format scripts/train_pytorch.py scripts/train_pytorch_precision_test.py
.venv/bin/ruff check scripts/train_pytorch.py scripts/train_pytorch_precision_test.py
```

Expected: pytest reports zero failures and zero errors; Ruff reports `All checks passed!`.

- [ ] **Step 8: Verify that JAX and configuration files were not modified**

Run:

```bash
git show --name-only --format= HEAD
```

Expected: the training-guard commit lists only `scripts/train_pytorch.py` and `scripts/train_pytorch_precision_test.py`. Also run `git status --short` and confirm the user's pre-existing edits to `src/openpi/models/model.py` and `src/openpi/training/config.py` remain uncommitted and untouched.

- [ ] **Step 9: Commit the PyTorch training guards**

```bash
git add scripts/train_pytorch.py scripts/train_pytorch_precision_test.py
git commit -m "fix: guard full-fp32 pytorch training state"
```

---

### Task 4: Run Full Verification and the pi0.5 Base Slow Acceptance Test

**Files:**
- Verify: `examples/convert_jax_model_to_pytorch.py`
- Verify: `src/openpi/models_pytorch/gemma_pytorch.py`
- Verify: `scripts/train_pytorch.py`
- Verify: `scripts/convert_jax_model_to_pytorch_test.py`
- Verify: `src/openpi/models_pytorch/gemma_pytorch_test.py`
- Verify: `scripts/train_pytorch_precision_test.py`
- Reference checkpoint: `/mnt/openpi_base/openpi-assets/checkpoints/pi05_base`
- Temporary output: `/mnt/openpi/.cache/pi05_base_pytorch_fp32`

**Interfaces:**
- Consumes: all helpers and tests from Tasks 1-3 plus the local pi0.5 base Orbax checkpoint.
- Produces: verified FP32 safetensors output and evidence that an FP32 training model retains only FP32 floating state.

- [ ] **Step 1: Run the complete focused test and lint suite**

```bash
JAX_PLATFORMS=cpu .venv/bin/pytest scripts/convert_jax_model_to_pytorch_test.py scripts/train_pytorch_precision_test.py src/openpi/models_pytorch/gemma_pytorch_test.py src/openpi/models/model_test.py -v
.venv/bin/ruff check examples/convert_jax_model_to_pytorch.py scripts/convert_jax_model_to_pytorch_test.py src/openpi/models_pytorch/gemma_pytorch.py src/openpi/models_pytorch/gemma_pytorch_test.py scripts/train_pytorch.py scripts/train_pytorch_precision_test.py
git diff --check
```

Expected: pytest has zero failures/errors, Ruff reports `All checks passed!`, and `git diff --check` has no output.

- [ ] **Step 2: Confirm the required Transformers replacements are installed**

Run:

```bash
.venv/bin/python -c "from transformers.models.siglip import check; assert check.check_whether_transformers_replace_is_installed_correctly()"
```

Expected: exit code 0. If the import or assertion fails, stop and request approval before applying the repository-documented Transformers replacement; do not silently modify the environment.

- [ ] **Step 3: Convert the local pi0.5 base checkpoint to FP32**

Run:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= .venv/bin/python examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir /mnt/openpi_base/openpi-assets/checkpoints/pi05_base \
  --config_name pi05_aloha \
  --output_path /mnt/openpi/.cache/pi05_base_pytorch_fp32 \
  --precision float32
```

Expected: conversion completes, the exact per-key FP32 check does not raise, and `/mnt/openpi/.cache/pi05_base_pytorch_fp32/model.safetensors` is created.

- [ ] **Step 4: Independently inspect the saved safetensors header**

Run:

```bash
.venv/bin/python -c 'from safetensors import safe_open; p="/mnt/openpi/.cache/pi05_base_pytorch_fp32/model.safetensors"; f=safe_open(p, framework="pt", device="cpu"); bad={k:d for k in f.keys() if ((d:=f.get_slice(k).get_dtype()).startswith("F") or d=="BF16") and d!="F32"}; print("tensor_count", len(list(f.keys()))); print("non_fp32_floating", bad); assert not bad'
```

Expected: `non_fp32_floating {}`.

- [ ] **Step 5: Load the file into a full-FP32 pi0.5 model and inspect in-memory state**

Run:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= .venv/bin/python -c 'import dataclasses, torch; import safetensors.torch; from openpi.training import config as c; from openpi.models_pytorch.pi0_pytorch import PI0Pytorch; cfg=dataclasses.replace(c.get_config("pi05_aloha").model, dtype="float32", pytorch_compile_mode=None); model=PI0Pytorch(cfg); safetensors.torch.load_model(model, "/mnt/openpi/.cache/pi05_base_pytorch_fp32/model.safetensors"); bad={f"parameter:{n}":str(p.dtype) for n,p in model.named_parameters() if p.is_floating_point() and p.dtype != torch.float32}; bad.update({f"buffer:{n}":str(b.dtype) for n,b in model.named_buffers() if b.is_floating_point() and b.dtype != torch.float32}); print("non_fp32_model_state", bad); assert not bad'
```

Expected: `non_fp32_model_state {}`.

- [ ] **Step 6: Inspect the final diff and preserve unrelated worktree changes**

Run:

```bash
git status --short
git diff HEAD~3 -- examples/convert_jax_model_to_pytorch.py scripts/convert_jax_model_to_pytorch_test.py src/openpi/models_pytorch/gemma_pytorch.py src/openpi/models_pytorch/gemma_pytorch_test.py scripts/train_pytorch.py scripts/train_pytorch_precision_test.py
git log -3 --oneline
```

Expected: only the six planned implementation/test files are part of the new task commits. Existing unrelated modified and untracked files remain untouched. If commit ancestry contains unrelated work, inspect each task commit with `git show --stat <commit>` instead of relying on `HEAD~3`.

- [ ] **Step 7: Record verification evidence in the handoff**

The handoff must include the exact pytest pass count, Ruff result, conversion command exit status, safetensors tensor count, `non_fp32_floating {}`, and `non_fp32_model_state {}`. Do not claim end-to-end success if the environment replacement check or slow checkpoint conversion was skipped.

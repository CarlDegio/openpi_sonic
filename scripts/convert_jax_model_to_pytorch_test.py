import dataclasses

import pytest
from safetensors.torch import save_file
import torch

from examples import convert_jax_model_to_pytorch as converter

# Tests intentionally exercise private conversion helpers.
# ruff: noqa: SLF001


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
def test_create_converted_model_preserves_fp32_values_and_original_config(monkeypatch, pi05):
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
    def load_state_dict(self, state_dict, strict=True, assign=False):  # noqa: FBT002
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

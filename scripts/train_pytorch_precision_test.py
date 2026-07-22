import pathlib

import pytest
from safetensors.torch import save_file
import torch

from scripts import train_pytorch

# Tests intentionally exercise private precision-validation helpers.
# ruff: noqa: SLF001


class _TinyTrainingModel(torch.nn.Module):
    def __init__(self, parameter_dtype=torch.float32, buffer_dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, dtype=parameter_dtype))
        self.register_buffer("running", torch.ones(2, dtype=buffer_dtype))
        self.register_buffer("index", torch.ones(1, dtype=torch.int64))


class _TinyAutocastModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, observation, actions):
        return self.linear(observation + actions)


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


@pytest.mark.parametrize(("enabled", "output_dtype"), [(False, torch.float32), (True, torch.bfloat16)])
def test_forward_autocast_keeps_fp32_master_weights(enabled, output_dtype):
    model = _TinyAutocastModel()
    observation = torch.ones((2, 2))
    actions = torch.ones((2, 2))

    output = train_pytorch._forward_with_autocast(
        model,
        observation,
        actions,
        enabled=enabled,
        device_type="cpu",
    )

    assert output.dtype == output_dtype
    assert model.linear.weight.dtype == torch.float32

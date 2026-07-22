import pathlib

import pytest
from safetensors.torch import save_file
import torch

from openpi.training import weight_loaders
from scripts import train_pytorch

# Tests intentionally exercise a private initial-weight loading helper.
# ruff: noqa: SLF001


class _TinyActionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2, bias=False)
        self.action_in_proj = torch.nn.Linear(3, 4)
        self.action_out_proj = torch.nn.Linear(4, 3)


class _TinyTiedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((2, 2)))
        self.alias = self.weight


def _shape_aware_loader() -> weight_loaders.ShapeAwareCheckpointWeightLoader:
    return weight_loaders.ShapeAwareCheckpointWeightLoader(
        "unused",
        reinit_mismatched_regexes=(
            ".*action_in_proj.*",
            ".*action_out_proj.*",
        ),
    )


def _base_checkpoint() -> dict[str, torch.Tensor]:
    return {
        "backbone.weight": torch.full((2, 2), 7.0),
        "action_in_proj.weight": torch.full((4, 2), 11.0),
        "action_in_proj.bias": torch.full((4,), 13.0),
        "action_out_proj.weight": torch.full((2, 4), 17.0),
        "action_out_proj.bias": torch.full((2,), 19.0),
    }


def test_shape_aware_load_keeps_only_allowed_mismatched_initialization(tmp_path: pathlib.Path):
    model = _TinyActionModel()
    initial_action_in_weight = model.action_in_proj.weight.detach().clone()
    initial_action_out_weight = model.action_out_proj.weight.detach().clone()
    initial_action_out_bias = model.action_out_proj.bias.detach().clone()
    checkpoint_path = tmp_path / "model.safetensors"
    save_file(_base_checkpoint(), checkpoint_path)

    train_pytorch._load_initial_weights(model, checkpoint_path, _shape_aware_loader())

    assert torch.equal(model.backbone.weight, torch.full((2, 2), 7.0))
    assert torch.equal(model.action_in_proj.bias, torch.full((4,), 13.0))
    assert torch.equal(model.action_in_proj.weight, initial_action_in_weight)
    assert torch.equal(model.action_out_proj.weight, initial_action_out_weight)
    assert torch.equal(model.action_out_proj.bias, initial_action_out_bias)


def test_shape_aware_load_rejects_unapproved_shape_mismatch(tmp_path: pathlib.Path):
    model = _TinyActionModel()
    checkpoint = _base_checkpoint()
    checkpoint["backbone.weight"] = torch.ones((1, 2))
    checkpoint_path = tmp_path / "model.safetensors"
    save_file(checkpoint, checkpoint_path)

    with pytest.raises(RuntimeError, match=r"backbone\.weight.*checkpoint=\(1, 2\).*model=\(2, 2\)"):
        train_pytorch._load_initial_weights(model, checkpoint_path, _shape_aware_loader())


def test_shape_aware_load_rejects_unexpected_checkpoint_key(tmp_path: pathlib.Path):
    model = _TinyActionModel()
    checkpoint = _base_checkpoint()
    checkpoint["unexpected"] = torch.ones(1)
    checkpoint_path = tmp_path / "model.safetensors"
    save_file(checkpoint, checkpoint_path)

    with pytest.raises(RuntimeError, match=r"unexpected_keys=.*unexpected"):
        train_pytorch._load_initial_weights(model, checkpoint_path, _shape_aware_loader())


def test_shape_aware_load_rejects_unexplained_missing_key(tmp_path: pathlib.Path):
    model = _TinyActionModel()
    checkpoint = _base_checkpoint()
    del checkpoint["backbone.weight"]
    checkpoint_path = tmp_path / "model.safetensors"
    save_file(checkpoint, checkpoint_path)

    with pytest.raises(RuntimeError, match=r"missing_keys=.*backbone\.weight"):
        train_pytorch._load_initial_weights(model, checkpoint_path, _shape_aware_loader())


def test_initial_load_accepts_missing_tied_alias(tmp_path: pathlib.Path):
    model = _TinyTiedModel()
    checkpoint_path = tmp_path / "model.safetensors"
    save_file({"weight": torch.full((2, 2), 23.0)}, checkpoint_path)

    train_pytorch._load_initial_weights(model, checkpoint_path, weight_loaders.NoOpWeightLoader())

    assert torch.equal(model.alias, torch.full((2, 2), 23.0))

from types import SimpleNamespace

import pytest
import torch

from openpi.models_pytorch import gemma_pytorch

# Tests intentionally exercise a private compatibility helper.
# ruff: noqa: SLF001


class _TinyExpertContainer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.gemma_expert = torch.nn.Module()
        self.gemma_expert.lm_head = torch.nn.Linear(1, 1, bias=False)


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


def test_remove_orphaned_expert_lm_head_does_not_hide_other_unexpected_keys():
    model = _TinyExpertContainer()
    gemma_pytorch._remove_orphaned_expert_lm_head(model)
    state = {
        "weight": torch.full((1,), 2.0),
        "gemma_expert.lm_head.weight": torch.full((1, 1), 3.0),
        "unexpected": torch.ones(1),
    }

    with pytest.raises(RuntimeError, match="Unexpected key.*unexpected"):
        model.load_state_dict(state, strict=True)


class _FakePaliGemma(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weight = torch.nn.Parameter(torch.ones(1))


class _FakeGemma(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(2, 2)
        self.lm_head = torch.nn.Linear(2, 2, bias=False)


def _model_configs():
    common = {
        "width": 4,
        "mlp_dim": 8,
        "num_heads": 2,
        "depth": 1,
        "num_kv_heads": 1,
        "head_dim": 2,
    }
    return SimpleNamespace(**common), SimpleNamespace(**common)


@pytest.mark.parametrize(
    ("precision", "head_is_present", "head_dtype"),
    [
        ("float32", False, None),
        ("bfloat16", True, torch.bfloat16),
    ],
)
def test_model_removes_expert_lm_head_only_for_fp32(monkeypatch, precision, head_is_present, head_dtype):
    monkeypatch.setattr(gemma_pytorch, "PaliGemmaForConditionalGeneration", _FakePaliGemma)
    monkeypatch.setattr(gemma_pytorch, "GemmaForCausalLM", _FakeGemma)
    vlm_config, action_expert_config = _model_configs()

    model = gemma_pytorch.PaliGemmaWithExpertModel(
        vlm_config,
        action_expert_config,
        precision=precision,
    )

    head = model.gemma_expert.lm_head
    assert (head is not None) is head_is_present
    if head is not None:
        assert head.weight.dtype == head_dtype


def test_sdpa_attention_matches_eager_grouped_query_attention():
    torch.manual_seed(7)
    query = torch.randn(2, 4, 5, 8, requires_grad=True)
    key = torch.randn(2, 2, 5, 8, requires_grad=True)
    value = torch.randn(2, 2, 5, 8, requires_grad=True)
    allowed = torch.tril(torch.ones(5, 5, dtype=torch.bool))
    allowed[:2, :2] = True
    attention_mask = torch.where(allowed[None, None], 0.0, -1e4)
    scaling = 0.37

    repeated_key = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
    repeated_value = value.repeat_interleave(query.shape[1] // value.shape[1], dim=1)
    weights = torch.matmul(query, repeated_key.transpose(2, 3)) * scaling
    weights = torch.softmax(weights + attention_mask, dim=-1, dtype=torch.float32)
    expected = torch.matmul(weights, repeated_value).transpose(1, 2).contiguous()

    actual = gemma_pytorch._sdpa_attention(query, key, value, attention_mask, scaling)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    expected_grads = torch.autograd.grad(expected.square().sum(), (query, key, value), retain_graph=True)
    actual_grads = torch.autograd.grad(actual.square().sum(), (query, key, value))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-5)


def test_sdpa_attention_uses_gqa_and_query_dtype_mask(monkeypatch):
    captured = {}

    def fake_sdpa(query, key, value, *, attn_mask, scale, enable_gqa):
        captured.update(attn_mask=attn_mask, scale=scale, enable_gqa=enable_gqa)
        return torch.zeros_like(query)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)
    query = torch.randn(1, 4, 3, 8)
    key = torch.randn(1, 2, 3, 8)
    value = torch.randn(1, 2, 3, 8)
    attention_mask = torch.zeros(1, 1, 3, 3, dtype=torch.float64)

    output = gemma_pytorch._sdpa_attention(query, key, value, attention_mask, 0.5)

    assert captured["attn_mask"].dtype == query.dtype
    assert captured["scale"] == 0.5
    assert captured["enable_gqa"] is True
    assert output.shape == (1, 3, 4, 8)

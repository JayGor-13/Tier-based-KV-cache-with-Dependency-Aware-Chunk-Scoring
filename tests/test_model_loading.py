from types import SimpleNamespace

import pytest
import torch

transformers = pytest.importorskip("transformers")

from src.models.cache_utils import load_hf_model_and_tokenizer, resolve_torch_dtype


class _Tokenizer:
    pad_token_id = None
    eos_token_id = 1
    eos_token = "<eos>"
    pad_token = None


class _Model:
    def __init__(self, attention_backend="eager"):
        self.config = SimpleNamespace(_attn_implementation=attention_backend)
        self.device = None
        self.training = True

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.training = False
        return self


def test_auto_dtype_preserves_checkpoint_native_selection():
    assert resolve_torch_dtype("auto", device="cuda") is None
    assert resolve_torch_dtype("auto", device="cpu") is None


def test_loader_does_not_silently_drop_requested_attention_backend(monkeypatch):
    calls = []
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: _Tokenizer()),
    )

    def fail_load(*_args, **kwargs):
        calls.append(kwargs)
        raise TypeError("backend unsupported")

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(fail_load),
    )

    with pytest.raises(TypeError):
        load_hf_model_and_tokenizer(
            "fixture/model",
            device="cpu",
            dtype="bfloat16",
            attn_implementation="eager",
        )

    assert calls
    assert all(call.get("attn_implementation") == "eager" for call in calls)


def test_loader_records_checkpoint_native_auto_and_verifies_backend(monkeypatch):
    calls = []
    model = _Model("eager")
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: _Tokenizer()),
    )

    def load(*_args, **kwargs):
        calls.append(kwargs)
        return model

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(load),
    )

    bundle = load_hf_model_and_tokenizer(
        "fixture/model",
        device="cpu",
        dtype="auto",
        attn_implementation="eager",
    )

    assert calls[0]["dtype"] == "auto"
    assert calls[0]["attn_implementation"] == "eager"
    assert bundle.model is model
    assert model.training is False


def test_loader_rejects_mismatched_resolved_backend(monkeypatch):
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: _Tokenizer()),
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: _Model("sdpa")),
    )

    with pytest.raises(RuntimeError, match="was not honored"):
        load_hf_model_and_tokenizer(
            "fixture/model",
            device="cpu",
            dtype="float32",
            attn_implementation="eager",
        )

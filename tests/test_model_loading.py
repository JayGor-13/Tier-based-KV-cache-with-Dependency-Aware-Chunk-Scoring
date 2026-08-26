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
    def __init__(self, attention_backend="eager", *, loaded_in_4bit=False):
        self.config = SimpleNamespace(_attn_implementation=attention_backend)
        self.device = None
        self.training = True
        self.is_loaded_in_4bit = loaded_in_4bit
        self.to_calls = []

    def to(self, device):
        self.to_calls.append(device)
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


def test_loader_builds_bnb_4bit_config_and_skips_model_to(monkeypatch):
    calls = []
    quantization_configs = []
    model = _Model("eager", loaded_in_4bit=True)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: _Tokenizer()),
    )

    def make_quantization_config(**kwargs):
        config = SimpleNamespace(**kwargs)
        quantization_configs.append(config)
        return config

    monkeypatch.setattr(transformers, "BitsAndBytesConfig", make_quantization_config)

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
        device="cuda",
        attn_implementation="eager",
        quantization="bnb-4bit",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    assert quantization_configs[0].load_in_4bit is True
    assert quantization_configs[0].bnb_4bit_compute_dtype is torch.float16
    assert calls[0]["quantization_config"] is quantization_configs[0]
    assert calls[0]["device_map"] == {"": 0}
    assert calls[0]["dtype"] is torch.float16
    assert model.to_calls == []
    assert bundle.load_metadata["quantization"] == "bnb-4bit"
    assert bundle.load_metadata["bnb_4bit_compute_dtype"] == "float16"


def test_loader_rejects_bnb_4bit_on_cpu():
    with pytest.raises(ValueError, match="requires a CUDA device"):
        load_hf_model_and_tokenizer(
            "fixture/model",
            device="cpu",
            quantization="bnb-4bit",
        )

import sys
from types import SimpleNamespace

import torch

from benchmarks.model_preflight import hub_model_preflight, loaded_model_preflight


class TinyModel(torch.nn.Module):
    def __init__(
        self,
        *,
        sliding_window=None,
        dtype=torch.float32,
        backend="eager",
        loaded_in_4bit=False,
        quantization_config=None,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4, 4, dtype=dtype))
        self.is_loaded_in_4bit = loaded_in_4bit
        self.config = SimpleNamespace(
            max_position_embeddings=4096,
            sliding_window=sliding_window,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            hidden_size=8,
            _attn_implementation=backend,
            quantization_config=quantization_config,
        )


def test_cpu_preflight_records_memory_estimate_without_requiring_cuda():
    report = loaded_model_preflight(
        model=TinyModel(),
        device=torch.device("cpu"),
        required_context=1024,
        prefill_block_size=16,
        require_cuda=False,
    )

    assert report["passed"] is True
    assert report["estimated_full_kv_bytes"] > 0
    assert report["gpu_total_memory_bytes"] is None


def test_preflight_rejects_truncated_sliding_cache():
    report = loaded_model_preflight(
        model=TinyModel(sliding_window=512),
        device=torch.device("cpu"),
        required_context=1024,
        prefill_block_size=16,
        require_cuda=False,
    )

    assert report["passed"] is False
    assert "sliding_window=512" in report["issues"][0]


def test_hub_preflight_resolves_immutable_revision_and_weight_size(monkeypatch):
    info = SimpleNamespace(
        sha="abc123",
        siblings=[
            SimpleNamespace(rfilename="model-1.safetensors", size=10),
            SimpleNamespace(rfilename="model-2.safetensors", size=20),
            SimpleNamespace(rfilename="config.json", size=3),
        ],
    )
    fake_module = SimpleNamespace(
        HfApi=lambda: SimpleNamespace(model_info=lambda **_kwargs: info)
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    report = hub_model_preflight(
        model_name="org/model",
        revision=None,
        token="secret",
    )

    assert report["passed"] is True
    assert report["resolved_revision"] == "abc123"
    assert report["repository_weight_bytes"] == 30

    quantized_report = hub_model_preflight(
        model_name="org/model",
        revision=None,
        token="secret",
        quantization="bnb-4bit",
    )
    assert quantized_report["quantization"] == "bnb-4bit"
    assert quantized_report["estimated_loaded_weight_bytes"] == 7


def test_preflight_records_actual_dtype_and_backend():
    report = loaded_model_preflight(
        model=TinyModel(dtype=torch.bfloat16),
        device=torch.device("cpu"),
        required_context=128,
        prefill_block_size=16,
        require_cuda=False,
        requested_dtype="bfloat16",
        requested_attention_backend="eager",
        require_unquantized=True,
    )

    assert report["passed"] is True
    assert report["dominant_parameter_dtype"] == "torch.bfloat16"
    assert report["parameter_dtypes"] == ["torch.bfloat16"]
    assert report["attention_backend"] == "eager"
    assert report["is_quantized"] is False


def test_preflight_rejects_requested_dtype_and_backend_mismatch():
    report = loaded_model_preflight(
        model=TinyModel(dtype=torch.float16, backend="sdpa"),
        device=torch.device("cpu"),
        required_context=128,
        prefill_block_size=16,
        require_cuda=False,
        requested_dtype="bfloat16",
        requested_attention_backend="eager",
    )

    assert report["passed"] is False
    assert any("requested dtype" in issue for issue in report["issues"])
    assert any("requested attention backend" in issue for issue in report["issues"])


def test_preflight_records_intended_4bit_compute_contract():
    model = TinyModel(
        dtype=torch.float16,
        loaded_in_4bit=True,
        quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
        },
    )

    report = loaded_model_preflight(
        model=model,
        device=torch.device("cpu"),
        required_context=128,
        prefill_block_size=16,
        require_cuda=False,
        requested_attention_backend="eager",
        requested_quantization="bnb-4bit",
        requested_quantization_compute_dtype="float16",
        require_unquantized=False,
    )

    assert report["passed"] is True
    assert report["loaded_in_4bit"] is True
    assert report["effective_compute_dtype"] == "torch.float16"
    assert report["quantization_compute_dtype"] == "torch.float16"


def test_preflight_rejects_missing_requested_4bit_loading():
    report = loaded_model_preflight(
        model=TinyModel(dtype=torch.float16),
        device=torch.device("cpu"),
        required_context=128,
        prefill_block_size=16,
        require_cuda=False,
        requested_quantization="bnb-4bit",
        requested_quantization_compute_dtype="float16",
    )

    assert report["passed"] is False
    assert any("4-bit loading was not resolved" in issue for issue in report["issues"])

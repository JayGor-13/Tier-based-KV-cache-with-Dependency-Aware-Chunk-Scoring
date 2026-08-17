import sys
from types import SimpleNamespace

import torch

from benchmarks.model_preflight import hub_model_preflight, loaded_model_preflight


class TinyModel(torch.nn.Module):
    def __init__(self, *, sliding_window=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4, 4))
        self.config = SimpleNamespace(
            max_position_embeddings=4096,
            sliding_window=sliding_window,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            hidden_size=8,
            _attn_implementation="eager",
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

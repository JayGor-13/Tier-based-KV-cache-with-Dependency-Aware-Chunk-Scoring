import pytest

from benchmarks.experiment_identity import build_run_identity, identity_sha256


def _identity(**execution_overrides):
    execution = {
        "model": "model",
        "revision": "abc",
        "dtype": "torch.bfloat16",
        "backend": "eager",
        **execution_overrides,
    }
    return build_run_identity(
        execution_contract=execution,
        sample_input={"record_sha256": "record", "input_token_sha256": "tokens"},
        method_config={"method": "tdc_kv", "retention": 0.3},
        seed=42,
    )


def test_run_identity_is_stable_for_mapping_order():
    left = identity_sha256({"a": 1, "b": {"x": 2, "y": 3}})
    right = identity_sha256({"b": {"y": 3, "x": 2}, "a": 1})
    assert left == right


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("revision", "def"),
        ("dtype", "torch.float16"),
        ("backend", "sdpa"),
        ("code_commit", "new"),
    ],
)
def test_execution_contract_changes_run_key(field, value):
    assert _identity()["run_key"] != _identity(**{field: value})["run_key"]


def test_identity_rejects_nonfinite_configuration():
    with pytest.raises(ValueError, match="NaN or Infinity"):
        identity_sha256({"retention": float("nan")})

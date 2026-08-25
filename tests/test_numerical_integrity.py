import pytest
import torch

from benchmarks.io_utils import strict_json_dumps
from benchmarks.numerical_validation import generation_health
from src.core.numerical import (
    NumericalIntegrityError,
    require_finite_tensor,
    validate_token_ids,
)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_require_finite_tensor_rejects_nonfinite_with_context(bad):
    tensor = torch.tensor([1.0, bad], dtype=torch.float32)

    with pytest.raises(NumericalIntegrityError) as captured:
        require_finite_tensor(
            "logits",
            tensor,
            stage="prefill",
            block_index=3,
        )

    diagnostic = captured.value.to_dict()
    assert diagnostic["context"]["stage"] == "prefill"
    assert diagnostic["context"]["block_index"] == 3
    assert diagnostic["tensor_health"]["nonfinite_count"] == 1


def test_generation_health_detects_limit_length_repetition():
    health = generation_health([0] * 96, max_new_tokens=96)
    assert health["degenerate_repetition"] is True
    assert health["maximum_non_special_token_fraction"] == 1.0


def test_generation_health_does_not_reject_short_or_varied_generation():
    assert generation_health([0] * 7, max_new_tokens=7)["degenerate_repetition"] is False
    assert generation_health([1, 2] * 8, max_new_tokens=16)["degenerate_repetition"] is False


def test_strict_json_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        strict_json_dumps({"score": float("nan")})


def test_validate_token_ids_rejects_fractional_or_float_tensor_ids():
    with pytest.raises(NumericalIntegrityError, match="integer token IDs"):
        validate_token_ids([1, 2.5], vocab_size=10)
    with pytest.raises(NumericalIntegrityError, match="integer tensor dtype"):
        validate_token_ids(torch.tensor([1.0, 2.0]), vocab_size=10)


def test_validate_token_ids_accepts_batch_size_one_integer_tensor():
    assert validate_token_ids(torch.tensor([[1, 2, 3]]), vocab_size=10) == (1, 2, 3)

import hashlib

import pytest

from benchmarks.gsm8k_protocol import (
    CHUNKKV_GSM8K_8SHOT_PROTOCOL,
    CHUNKKV_GSM8K_EXEMPLARS,
    build_chunkkv_gsm8k_prompt,
    canonicalize_protocol,
    protocol_metadata,
    validate_protocol_for_adapter,
)


def test_chunkkv_prompt_matches_versioned_eight_shot_fixture():
    prompt = build_chunkkv_gsm8k_prompt("What is 2 + 2?")

    assert len(CHUNKKV_GSM8K_EXEMPLARS) == 8
    assert prompt.count("Question:") == 9
    assert prompt.count("The answer is") == 8
    assert prompt.startswith("Question: There are 15 trees in the grove.")
    assert prompt.endswith("Question: What is 2 + 2?\n")
    assert "####" not in prompt
    assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == (
        "e5658d864c0ac8e3e9f717779644d6d0a142ce02d91ff01ebc8cdfefd5f210f1"
    )


def test_protocol_aliases_resolve_to_one_stable_identifier():
    assert canonicalize_protocol("gsm8k-chunkkv") == CHUNKKV_GSM8K_8SHOT_PROTOCOL
    assert canonicalize_protocol("chunkkv_gsm8k") == CHUNKKV_GSM8K_8SHOT_PROTOCOL


def test_chunkkv_protocol_rejects_non_gsm8k_adapters():
    with pytest.raises(ValueError, match="requires adapter `gsm8k`"):
        validate_protocol_for_adapter(CHUNKKV_GSM8K_8SHOT_PROTOCOL, "hotpotqa")


def test_protocol_metadata_is_defensive_and_auditable():
    first = protocol_metadata(CHUNKKV_GSM8K_8SHOT_PROTOCOL)
    first["shots"] = 0
    second = protocol_metadata(CHUNKKV_GSM8K_8SHOT_PROTOCOL)

    assert second["shots"] == 8
    assert second["prompt_source"] == "ChunkKV Appendix G, Table 30"
    assert second["judge"] == "gsm8k_final_numeric_exact_match"

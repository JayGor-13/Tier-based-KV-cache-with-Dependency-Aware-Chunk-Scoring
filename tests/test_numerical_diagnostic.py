from scripts.diagnose_hf_numerics import (
    diagnostic_cases,
    diagnostic_record_passes,
    first_differing_token,
)


def test_diagnostic_matrix_has_stable_a_through_h_contract():
    cases = diagnostic_cases()

    assert [case["case"] for case in cases] == list("ABCDEFGH")
    assert [case["dtype"] for case in cases[:4]] == ["float16"] * 4
    assert [case["dtype"] for case in cases[4:]] == ["bfloat16"] * 4
    assert cases[3]["blocks"] == [128, 16]
    assert cases[7]["blocks"] == [128, 16]


def test_first_differing_token_handles_values_and_length():
    assert first_differing_token([1, 2], [1, 3]) == 1
    assert first_differing_token([1, 2], [1, 2, 3]) == 2
    assert first_differing_token([1, 2], [1, 2]) is None


def test_custom_diagnostic_requires_prompt_next_token_and_generation_parity():
    record = {
        "status": "ok",
        "path": "custom_fullkv",
        "native_prompt_token_match": True,
        "native_token_match": True,
        "direct_next_token_match": True,
        "generation_health": {"degenerate_repetition": False},
    }
    assert diagnostic_record_passes(record) is True

    record["native_token_match"] = False
    assert diagnostic_record_passes(record) is False

from benchmarks.tuning import select_best_configuration


def _run(alpha, prediction, policy_ms):
    return {
        "status": "ok",
        "method": "tdc_kv",
        "dataset": "niah",
        "gold": "KEY",
        "evicted_prediction": prediction,
        "config": {"alpha": alpha, "theta": 0.3, "recent_window": 16},
        "runtime": {"stages": {"policy": {"elapsed_ms": policy_ms}}},
    }


def test_tuning_selects_quality_before_speed():
    selection = select_best_configuration(
        {"runs": [_run(0.5, "wrong", 1.0), _run(0.6, "KEY", 3.0)]}
    )
    assert selection["selected"]["configuration"]["alpha"] == 0.6

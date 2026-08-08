import torch

from benchmarks.runtime_metrics import combine_measurements, measure_call


def test_cpu_measurement_has_time_and_no_cuda_memory():
    result = measure_call(lambda: 7, device=torch.device("cpu"))

    assert result.value == 7
    assert result.measurement.elapsed_ms >= 0.0
    assert result.measurement.cuda_device is None
    assert result.measurement.peak_allocated_bytes is None


def test_combined_measurements_reports_totals():
    first = measure_call(lambda: None, device="cpu").measurement
    second = measure_call(lambda: None, device="cpu").measurement

    combined = combine_measurements(first=first, second=second)

    assert set(combined["stages"]) == {"first", "second"}
    assert combined["total_measured_ms"] >= 0.0
    assert combined["max_peak_allocated_bytes"] is None

import pytest
from perflab.metrics import compute_percentiles, compute_throughput, compute_samples_per_sec


def test_compute_percentiles():
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    result = compute_percentiles(values)
    assert abs(result["p50"] - 55.0) < 0.1
    assert abs(result["p90"] - 91.0) < 0.1
    assert abs(result["p95"] - 95.5) < 0.1
    assert abs(result["p99"] - 99.1) < 0.1


def test_compute_percentiles_empty():
    result = compute_percentiles([])
    assert result["p50"] == 0.0
    assert result["p90"] == 0.0
    assert result["p95"] == 0.0
    assert result["p99"] == 0.0


def test_compute_percentiles_single():
    result = compute_percentiles([42])
    assert result["p50"] == 42.0
    assert result["p90"] == 42.0


def test_compute_throughput():
    throughput = compute_throughput(1000, 500)
    assert throughput == 2000.0


def test_compute_throughput_zero_time():
    throughput = compute_throughput(1000, 0)
    assert throughput == 0.0


def test_compute_samples_per_sec():
    samples_per_sec = compute_samples_per_sec(500, 1000)
    assert samples_per_sec == 500.0

import pytest
from perflab.recommend import (
    filter_valid_runs,
    compute_balanced_score,
    get_best_for_latency_budget,
    get_best_for_max_throughput,
    get_best_balanced,
    generate_recommendations,
)


def test_filter_valid_runs():
    runs = [
        {"latency_p95": 10, "throughput_rps": 100},
        {"latency_p95": 20},
        {"throughput_rps": 50},
        {"latency_p95": 15, "throughput_rps": 80},
    ]
    valid = filter_valid_runs(runs)
    assert len(valid) == 2
    assert valid[0]["latency_p95"] == 10
    assert valid[1]["latency_p95"] == 15


def test_compute_balanced_score():
    run = {"latency_p95": 10, "throughput_rps": 100}
    score = compute_balanced_score(run)
    assert score == 10.0


def test_compute_balanced_score_zero_latency():
    run = {"latency_p95": 0, "throughput_rps": 100}
    score = compute_balanced_score(run)
    assert score == 100.0


def test_get_best_for_latency_budget():
    runs = [
        {"latency_p95": 10, "throughput_rps": 100},
        {"latency_p95": 20, "throughput_rps": 200},
        {"latency_p95": 30, "throughput_rps": 300},
    ]
    best = get_best_for_latency_budget(runs, "p95", 25)
    assert best["latency_p95"] == 20
    assert best["throughput_rps"] == 200


def test_get_best_for_latency_budget_none():
    runs = [
        {"latency_p95": 100, "throughput_rps": 100},
    ]
    best = get_best_for_latency_budget(runs, "p95", 50)
    assert best is None


def test_get_best_for_max_throughput():
    runs = [
        {"latency_p95": 10, "throughput_rps": 100},
        {"latency_p95": 20, "throughput_rps": 300},
        {"latency_p95": 30, "throughput_rps": 200},
    ]
    best = get_best_for_max_throughput(runs)
    assert best["throughput_rps"] == 300


def test_get_best_balanced():
    runs = [
        {"latency_p95": 10, "throughput_rps": 100},
        {"latency_p95": 20, "throughput_rps": 300},
        {"latency_p95": 5, "throughput_rps": 50},
    ]
    best = get_best_balanced(runs)
    assert best["latency_p95"] == 20
    assert best["throughput_rps"] == 300


def test_generate_recommendations():
    runs = [
        {"model": "resnet18", "latency_p95": 10, "throughput_rps": 100, "batch_size": 1, "compile": False, "threads": 4},
        {"model": "resnet18", "latency_p95": 20, "throughput_rps": 300, "batch_size": 4, "compile": True, "threads": 8},
        {"model": "mobilenet", "latency_p95": 5, "throughput_rps": 150, "batch_size": 2, "compile": False, "threads": 2},
    ]
    result = generate_recommendations(runs, "balanced")
    assert len(result["recommendations"]) == 2
    assert "resnet18" in result["details"]
    assert "mobilenet" in result["details"]

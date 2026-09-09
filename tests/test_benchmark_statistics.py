import copy

import pytest

from aegis_norm.benchmarks.kernel import cases, pair_orders, repetitions_for
from aegis_norm.benchmarks.statistics import summarize


def samples():
    return [
        {
            "case_id": "case",
            "rows": 1,
            "hidden": 7,
            "dtype": "float16",
            "eps": 1e-5,
            "seed": 2026,
            "timing_scope": "ordinary_operator_warm_cache",
            "pair_id": pair,
            "arm": arm,
            "status": "completed",
            "repetitions": 100,
            "elapsed_ms": time,
            "host_elapsed_ms": time * 2,
        }
        for pair in range(4)
        for arm, time in [("reference", 2.0), ("native", 1.0)]
    ]


def test_paired_ratio_and_units():
    result = summarize(samples(), trials=4)
    assert result["elapsed_ms"]["ratio_of_medians"] == 2
    assert result["elapsed_ms"]["paired_bootstrap_95ci"] == [2, 2]
    assert result["elapsed_ms"]["arms"]["native"]["median_us"] == 10
    assert result["host_elapsed_ms"]["arms"]["native"]["median_us"] == 20


def test_comparisons_cannot_be_mixed():
    rows = samples()
    rows[0]["comparison"] = "geometry128"
    with pytest.raises(ValueError, match="identical workload"):
        summarize(rows, trials=4)


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "repetitions", "nan", "negative", "status", "shape"]
)
def test_invalid_pairs_are_rejected(change):
    rows = copy.deepcopy(samples())
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "repetitions":
        rows[0]["repetitions"] = 20
    elif change == "nan":
        rows[0]["elapsed_ms"] = float("nan")
    elif change == "negative":
        rows[0]["host_elapsed_ms"] = -1
    elif change == "status":
        rows[0]["status"] = "failed"
    elif change == "shape":
        rows[0]["hidden"] = 8
    with pytest.raises(ValueError):
        summarize(rows, trials=4)


def test_outliers_are_preserved_and_pair_bootstrap_is_reproducible():
    rows = samples()
    rows[-1]["elapsed_ms"] = 100
    first = summarize(rows, trials=4, resamples=1000)
    assert first == summarize(rows, trials=4, resamples=1000)
    assert first["elapsed_ms"]["arms"]["native"]["max_us"] == 1000
    assert first["elapsed_ms"]["comparison"] == "inconclusive"


def test_calibration_and_profiles():
    assert repetitions_for([0.1, 0.2]) == 100
    assert repetitions_for([0.00001, 0.001]) == 10000
    assert repetitions_for([1]) == 20
    with pytest.raises(ValueError):
        repetitions_for([0])
    assert len(cases("smoke")) == 6 and len(cases("required")) == 72
    assert len(cases("safety")) == 16 and all(c["offset"] == 1 for c in cases("safety"))
    assert pair_orders(30, 2026) == pair_orders(30, 2026)
    assert {tuple(order) for order in pair_orders(30, 2026)} == {
        ("reference", "native"),
        ("native", "reference"),
    }

"""Paired statistics with explicit validation and no numerical dependencies."""

import math
import random
from statistics import median


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def distribution(values):
    return {
        "count": len(values),
        "median_us": median(values),
        "p95_us": percentile(values, 0.95),
        "min_us": min(values),
        "max_us": max(values),
        "iqr_us": percentile(values, 0.75) - percentile(values, 0.25),
    }


def summarize(samples, *, trials, resamples=10000):
    """Validate one case and bootstrap pair indices, never inner repetitions."""
    if type(trials) is not int or trials < 2 or type(resamples) is not int or resamples < 1:
        raise ValueError("Invalid trial/bootstrap count")
    pairs = {}
    signatures = {
        (row.get("comparison", "eager"),)
        + tuple(
            row[key]
            for key in ("case_id", "rows", "hidden", "dtype", "eps", "seed", "timing_scope")
        )
        for row in samples
    }
    if len(signatures) != 1:
        raise ValueError("Samples must describe one identical workload")
    for row in samples:
        arm, pair = row["arm"], row["pair_id"]
        if arm not in ("reference", "native") or type(pair) is not int or not 0 <= pair < trials:
            raise ValueError("Invalid arm or pair ID")
        if row["status"] != "completed":
            raise ValueError("Incomplete sample")
        repetitions = row["repetitions"]
        if type(repetitions) is not int or repetitions <= 0:
            raise ValueError("Invalid repetition count")
        for key in ("elapsed_ms", "host_elapsed_ms"):
            if isinstance(row[key], bool) or not math.isfinite(row[key]) or row[key] <= 0:
                raise ValueError("Invalid timing")
        pair_data = pairs.setdefault(pair, {})
        if arm in pair_data:
            raise ValueError("Duplicate arm in pair")
        pair_data[arm] = row
    if len(pairs) != trials or any(set(p) != {"reference", "native"} for p in pairs.values()):
        raise ValueError("Missing paired observations")
    repetitions = {row["repetitions"] for row in samples}
    if len(repetitions) != 1:
        raise ValueError("Repetitions must be frozen for the case")
    result = {}
    for timing in ("elapsed_ms", "host_elapsed_ms"):
        arms = {
            arm: [
                pairs[i][arm][timing] * 1000 / pairs[i][arm]["repetitions"] for i in range(trials)
            ]
            for arm in ("reference", "native")
        }
        rng = random.Random(2026)
        ratios = []
        for _ in range(resamples):
            indices = [rng.randrange(trials) for _ in range(trials)]
            ratios.append(
                median([arms["reference"][i] for i in indices])
                / median([arms["native"][i] for i in indices])
            )
        interval = [percentile(ratios, 0.025), percentile(ratios, 0.975)]
        result[timing] = {
            "arms": {arm: distribution(values) for arm, values in arms.items()},
            "ratio_of_medians": median(arms["reference"]) / median(arms["native"]),
            "paired_bootstrap_95ci": interval,
            "comparison": "improved"
            if interval[0] > 1
            else "regressed"
            if interval[1] < 1
            else "inconclusive",
        }
    return result

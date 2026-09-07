"""F04 measurement foundation: paired ordinary-operator timing on one T4."""

import argparse
import hashlib
import json
import math
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .statistics import summarize

PROTOCOL = "f04-baseline-v1"


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def cases(profile):
    if profile == "smoke":
        rows, widths, dtypes = [1, 32, 128], [2048, 4096], ["float16"]
    elif profile == "required":
        rows, widths, dtypes = (
            [1, 2, 4, 8, 16, 32, 128, 512, 2048],
            [2048, 4096, 5120, 8192],
            ["float16", "float32"],
        )
    elif profile == "safety":
        rows, widths, dtypes = [1, 128], [96, 257, 1023, 4097], ["float16", "float32"]
    else:
        raise ValueError("Unknown profile")
    return [
        {
            "case_id": f"r{r}-h{h}-{d}",
            "rows": r,
            "hidden": h,
            "dtype": d,
            "offset": 1 if profile == "safety" else 0,
            "eps": 1e-5,
        }
        for r in rows
        for h in widths
        for d in dtypes
    ]


def pair_orders(trials, seed):
    rng = random.Random(seed)
    return [rng.sample(["reference", "native"], 2) for _ in range(trials)]


def repetitions_for(milliseconds_per_call):
    if not milliseconds_per_call or any(
        not math.isfinite(t) or t <= 0 for t in milliseconds_per_call
    ):
        raise ValueError("Calibration must contain positive finite timings")
    return max(20, min(10000, math.ceil(10 / min(milliseconds_per_call))))


def reference(x, weight, eps):
    import torch

    x32 = x.float()
    return weight * (x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + eps)).to(x.dtype)


def errors(actual, expected):
    import torch

    if (
        actual.shape != expected.shape
        or actual.dtype != expected.dtype
        or actual.device != expected.device
    ):
        raise RuntimeError("Baseline output metadata does not match")
    atol, rtol = (2e-3, 2e-3) if expected.dtype == torch.float16 else (1e-6, 2e-5)
    delta = (actual.float() - expected.float()).abs()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    return {
        "passed": finite and bool(torch.all(delta <= atol + rtol * expected.float().abs())),
        "atol": atol,
        "rtol": rtol,
        "finite": finite,
        "max_absolute": float(delta.max()) if finite else None,
        "max_relative": float((delta / expected.float().abs().clamp_min(1e-8)).max())
        if finite
        else None,
        "rms_error": float(delta.square().mean().sqrt()) if finite else None,
        "failing_count": int((delta > atol + rtol * expected.float().abs()).sum())
        if finite
        else None,
        "worst_flat_index": int(delta.reshape(-1).argmax()) if finite else None,
    }


def timed_group(fn, repetitions):
    import torch

    stream = torch.cuda.current_stream()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # Initialize event resources and drain preceding work outside the timed region.
    start.record(stream)
    end.record(stream)
    end.synchronize()
    beginning = time.perf_counter_ns()
    start.record(stream)
    for _ in range(repetitions):
        fn()
    end.record(stream)
    end.synchronize()
    host_ms = (time.perf_counter_ns() - beginning) / 1e6
    return {"elapsed_ms": start.elapsed_time(end), "host_elapsed_ms": host_ms}


def run_case(case, *, seed, trials, emit):
    import torch

    from aegis_norm import explain_dispatch

    dtype = getattr(torch, case["dtype"])
    generator = torch.Generator(device="cuda").manual_seed(seed)
    count = case["rows"] * case["hidden"]
    x = torch.randn(count + case["offset"], device="cuda", dtype=dtype, generator=generator)
    x = x[case["offset"] :].view(case["rows"], case["hidden"])
    weight = torch.randn(case["hidden"], device="cuda", dtype=dtype, generator=generator)
    eps = case["eps"]
    assert explain_dispatch(x, weight, eps, backend="cuda").backend == "cuda"
    arms = {
        "reference": lambda: reference(x, weight, eps),
        "native": lambda: torch.ops.aegis_norm.rms_norm.default(x, weight, eps),
    }
    check = errors(arms["native"](), arms["reference"]())
    emit("correctness", {**case, "seed": seed, **check})
    if not check["passed"]:
        raise RuntimeError("Numerical gate failed for " + case["case_id"])
    # Related implementation: a passing tolerance check does not prove identical casting.
    related = torch.nn.functional.rms_norm(x, (case["hidden"],), weight, eps)
    emit(
        "related-baseline",
        {
            **case,
            "baseline": "torch.nn.functional.rms_norm",
            "version": torch.__version__,
            "timed": False,
            "semantics": "related_casting_not_guaranteed",
            **errors(related, arms["reference"]()),
        },
    )
    for fn in arms.values():
        for _ in range(100):
            fn()
    calibration = {arm: timed_group(fn, 20) for arm, fn in arms.items()}
    repetitions = repetitions_for([c["elapsed_ms"] / 20 for c in calibration.values()])
    emit("calibration", {**case, "measurements": calibration, "repetitions": repetitions})
    samples = []
    for pair, order in enumerate(pair_orders(trials, seed)):
        for position, arm in enumerate(order):
            row = {
                **case,
                "seed": seed,
                "trial_id": pair,
                "pair_id": pair,
                "arm": arm,
                "position": position,
                "status": "completed",
                "repetitions": repetitions,
                "timing_scope": "ordinary_operator_warm_cache",
                **timed_group(arms[arm], repetitions),
            }
            emit("samples", row)
            samples.append(row)
    return {
        **case,
        "correctness_passed": True,
        "repetitions": repetitions,
        "groups_below_10ms": sum(s["elapsed_ms"] < 10 for s in samples),
        "statistics": summarize(samples, trials=trials),
    }


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def finish(output, manifest, summaries):
    write_json(
        output / "summary.json",
        {
            "protocol": PROTOCOL,
            "cases": summaries,
            "claim_status": "requires_review"
            if manifest["status"] == "completed"
            else "not_measured",
        },
    )
    manifest["finished_at"] = utc_now()
    manifest["artifacts"] = sorted(
        p.name
        for p in output.iterdir()
        if p.is_file() and p.name not in ("manifest.json", "checksums.sha256")
    )
    write_json(output / "manifest.json", manifest)
    hashes = "".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n"
        for p in sorted(output.iterdir())
        if p.is_file() and p.name != "checksums.sha256"
    )
    (output / "checksums.sha256").write_text(hashes, encoding="utf-8")


def run(output_root, *, profile="smoke", trials=30):
    from aegis_norm import load_native
    from aegis_norm.preflight import collect, command, dependency_snapshot

    matrix = cases(profile)
    if type(trials) is not int or not 2 <= trials <= 100:
        raise ValueError("trials must be 2..100")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = Path(output_root) / run_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "protocol_version": PROTOCOL,
        "run_id": run_id,
        "parent_run_id": None,
        "started_at": utc_now(),
        "finished_at": None,
        "status": "running",
        "code_commit": None,
        "environment": None,
        "model": None,
        "configuration": {
            "profile": profile,
            "trials": trials,
            "warmups": 100,
            "seed": 2026,
            "cases": matrix,
            "timing_scope": "ordinary_operator_warm_cache",
        },
        "limitations": [
            "Provisional F04 baseline harness, not full F09 release evaluation",
            "Compiler and third-party paired baselines deferred to F09",
            "Warm-cache allocated outputs; event time may include Python submission gaps",
            "No measured DRAM traffic, pure-kernel duration, model or serving result",
            "Smoke/nonstandard trial counts do not support headline performance claims",
        ],
    }
    write_json(output / "manifest.json", manifest)
    summaries = []

    def emit(name, row):
        with (output / (name + ".jsonl")).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"run_id": run_id, **row}, allow_nan=False) + "\n")
            stream.flush()

    try:
        manifest["code_commit"] = command(["git", "rev-parse", "HEAD"])
        manifest["worktree"] = command(["git", "status", "--porcelain"])
        if (
            manifest["code_commit"]["returncode"] != 0
            or manifest["worktree"]["returncode"] != 0
            or manifest["worktree"]["output"].strip()
        ):
            raise RuntimeError("Run from a clean pinned repository")
        package = Path(__file__).resolve().parents[1]
        manifest["source_sha256"] = {
            p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(package.rglob("*"))
            if p.suffix in (".py", ".cpp", ".cu")
        }
        manifest["environment"] = collect()
        (output / "environment.txt").write_text(dependency_snapshot(), encoding="utf-8")
        write_json(output / "manifest.json", manifest)
        if manifest["environment"]["blockers"]:
            raise RuntimeError(
                "Preflight blockers: " + "; ".join(manifest["environment"]["blockers"])
            )
        build_start = time.perf_counter()
        load_native()
        manifest["build_load_seconds"] = time.perf_counter() - build_start
        write_json(output / "manifest.json", manifest)
        import torch

        with torch.inference_mode():
            for index, case in enumerate(matrix):
                summaries.append(run_case(case, seed=2026 + index, trials=trials, emit=emit))
                write_json(
                    output / "summary.json", {"cases": summaries, "claim_status": "requires_review"}
                )
                print(f"Completed {index + 1}/{len(matrix)}: {case['case_id']}", flush=True)
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        finish(output, manifest, summaries)
        print("Measurement artifacts:", output, flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "required", "safety"], default="smoke")
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    run(args.output_root, profile=args.profile, trials=args.trials)


if __name__ == "__main__":
    main()

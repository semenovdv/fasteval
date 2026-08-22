"""Statistics over repeated runs: dispersion, confidence intervals, comparisons.

Pure standard library. The philosophy matches the tool's honesty rules: we
report descriptive statistics rigorously and use Wilson score intervals for
proportions; pairwise model comparison states whether intervals overlap
instead of fabricating p-values from tiny samples.
"""

import json
import math
from statistics import mean, median
from typing import Any

from .models import RunResult

__all__ = ["compute_statistics", "describe_numeric", "wilson_interval"]

_Z_95 = 1.959963984540054


def describe_numeric(values: list[float]) -> dict[str, float | None] | None:
    """Mean, sample std, median, min, max and coefficient of variation."""
    if not values:
        return None
    result: dict[str, float | None] = {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }
    if len(values) > 1:
        std = _sample_std(values)
        result["std"] = std
        result["cv_percent"] = (std / result["mean"] * 100) if result["mean"] else None
    else:
        result["std"] = None
        result["cv_percent"] = None
    return result


def wilson_interval(successes: int, total: int, z: float = _Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if total <= 0:
        return (0.0, 1.0)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z / denominator * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (max(0.0, center - half), min(1.0, center + half))


def _sample_std(values: list[float]) -> float:
    m = mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _result_label(row: RunResult) -> str:
    return f"{row.provider}/{row.model} ({row.reasoning_effort or 'off'})"


def _canonical(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def _output_consistency(rows: list[RunResult]) -> float | None:
    """Share of cases whose attempts produced identical successful outputs."""
    by_case: dict[str, list[str]] = {}
    for row in rows:
        if row.ok and row.output is not None:
            by_case.setdefault(row.case_id, []).append(_canonical(row.output))
    multi = [outputs for outputs in by_case.values() if len(outputs) > 1]
    if not multi:
        return None
    agreeing = sum(1 for outputs in multi if len(set(outputs)) == 1)
    return agreeing / len(multi)


def compute_statistics(results: list[RunResult]) -> dict[str, Any]:
    """Aggregate per-model statistics across attempts and cases.

    Grouping key is the full cell identity ``provider/model (effort)``, so
    different reasoning efforts of one model are compared separately.
    """
    grouped: dict[str, list[RunResult]] = {}
    for row in results:
        grouped.setdefault(_result_label(row), []).append(row)

    per_model: dict[str, dict[str, Any]] = {}
    for label, rows in sorted(grouped.items()):
        ok_rows = [row for row in rows if row.ok]
        latencies = [row.latency_ms for row in ok_rows if row.latency_ms is not None]
        throughputs = [row.tokens_per_second for row in ok_rows if row.tokens_per_second is not None]
        scored = [row for row in rows if row.evaluation and row.evaluation.get("passed") is not None]
        passed = sum(1 for row in scored if row.evaluation and row.evaluation["passed"])

        entry: dict[str, Any] = {
            "runs": len(rows),
            "ok_runs": len(ok_rows),
            "errors": len(rows) - len(ok_rows),
            "latency_ms": describe_numeric(latencies),
            "tokens_per_second": describe_numeric(throughputs),
            "cost_usd_per_run": describe_numeric(
                [row.total_cost_usd for row in rows if row.total_cost_usd is not None]
            ),
            "scored": len(scored),
        }
        if scored:
            low, high = wilson_interval(passed, len(scored))
            entry["pass_rate"] = passed / len(scored)
            entry["pass_rate_ci95"] = [round(low, 4), round(high, 4)]
        else:
            entry["pass_rate"] = None
            entry["pass_rate_ci95"] = None
        consistency = _output_consistency(ok_rows)
        entry["output_consistency"] = round(consistency, 4) if consistency is not None else None
        per_model[label] = entry

    return {
        "per_model": per_model,
        "pairwise": _pairwise_pass_comparison(per_model),
    }


def _pairwise_pass_comparison(per_model: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare models with scored results via pass-rate Wilson intervals.

    Overlapping intervals mean the observed difference is not distinguishable
    from noise at this sample size — reported as a verdict instead of a
    fabricated p-value.
    """
    scored_models = [(label, stats) for label, stats in sorted(per_model.items()) if stats.get("pass_rate") is not None]
    pairs: list[dict[str, Any]] = []
    for i, (a_label, a) in enumerate(scored_models):
        for b_label, b in scored_models[i + 1 :]:
            a_lo, a_hi = a["pass_rate_ci95"]
            b_lo, b_hi = b["pass_rate_ci95"]
            delta = round(a["pass_rate"] - b["pass_rate"], 4)
            overlap = not (a_lo > b_hi or b_lo > a_hi)
            verdict = "not distinguishable at this sample size" if overlap else ("A leads" if delta > 0 else "B leads")
            pairs.append(
                {
                    "a": a_label,
                    "b": b_label,
                    "delta_pass_rate": delta,
                    "a_ci95": a["pass_rate_ci95"],
                    "b_ci95": b["pass_rate_ci95"],
                    "verdict": verdict,
                }
            )
    return pairs

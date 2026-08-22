import json
from pathlib import Path

import pytest

from fastevals.config import RunConfig
from fastevals.models import RunResult
from fastevals.stats import compute_statistics, describe_numeric, wilson_interval


def row(model: str, effort: str = "low", output="x", passed=None, latency=100.0, cost=None):
    evaluation = None if passed is None else {"evaluator": "exact_match", "passed": passed, "detail": None}
    return RunResult(
        provider="openai",
        model=model,
        reasoning_effort=effort,
        output=output,
        case_id=f"case-{model}",
        attempt=1,
        evaluation=evaluation,
        latency_ms=latency,
        output_cost_usd=cost,
    )


def test_describe_numeric_known_values():
    stats = describe_numeric([10.0, 20.0, 30.0])
    assert stats["mean"] == 20.0
    assert stats["median"] == 20.0
    assert stats["min"] == 10.0 and stats["max"] == 30.0
    assert stats["std"] == pytest.approx(10.0)
    assert stats["cv_percent"] == pytest.approx(50.0)


def test_describe_numeric_empty_and_single():
    assert describe_numeric([]) is None
    single = describe_numeric([5.0])
    assert single["mean"] == 5.0
    assert single["std"] is None and single["cv_percent"] is None


def test_wilson_interval_known_example():
    # 8/10 successes: published Wilson 95% interval ~ [0.490, 0.943]
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.490, abs=0.01)
    assert high == pytest.approx(0.943, abs=0.01)


def test_wilson_interval_edges():
    assert wilson_interval(0, 5)[0] == 0.0
    assert wilson_interval(5, 5)[1] == 1.0
    assert wilson_interval(3, 0) == (0.0, 1.0)  # degenerate guard


def test_compute_statistics_groups_by_cell_identity():
    rows = [
        row("gpt-a", "low", output="same", latency=100),
        row("gpt-a", "low", output="same", latency=200),
        row("gpt-b", "low", output="diff1", latency=300),
        row("gpt-b", "high", output="x", latency=50),
    ]
    stats = compute_statistics(rows)
    per_model = stats["per_model"]
    assert set(per_model) == {
        "openai/gpt-a (low)",
        "openai/gpt-b (low)",
        "openai/gpt-b (high)",
    }
    luna = per_model["openai/gpt-a (low)"]
    assert luna["runs"] == 2 and luna["ok_runs"] == 2
    assert luna["latency_ms"]["mean"] == 150.0
    assert luna["output_consistency"] == 1.0


def test_output_consistency_detects_divergence():
    rows = [
        row("gpt-a", output="answer-v1"),
        row("gpt-a", output="answer-v2"),
    ]
    stats = compute_statistics(rows)["per_model"]
    assert stats["openai/gpt-a (low)"]["output_consistency"] == 0.0


def test_pass_rate_ci_present_only_for_scored_models():
    rows = [
        row("scored", passed=True),
        row("scored", passed=False),
        row("unscored", output="free text"),
    ]
    per_model = compute_statistics(rows)["per_model"]
    assert per_model["openai/scored (low)"]["pass_rate"] == pytest.approx(0.5)
    lo, hi = per_model["openai/scored (low)"]["pass_rate_ci95"]
    assert 0 < lo < hi < 1
    assert per_model["openai/unscored (low)"]["pass_rate"] is None


def test_pairwise_verdicts_separate_vs_noise():
    # clearly separated pass rates with enough samples -> A leads
    separated = []
    for _ in range(6):
        separated.append(row("strong", passed=True))
        separated.append(row("weak", passed=False))
    pairs = compute_statistics(separated)["pairwise"]
    assert len(pairs) == 1
    assert pairs[0]["verdict"] == "A leads"
    assert pairs[0]["delta_pass_rate"] == pytest.approx(1.0)

    # identical rates -> intervals overlap, honest verdict
    tie = [row("m1", passed=True), row("m2", passed=True)]
    tie_pairs = compute_statistics(tie)["pairwise"]
    assert tie_pairs[0]["verdict"].startswith("not distinguishable")


def _multi_attempt_results() -> list[RunResult]:
    results: list[RunResult] = []
    for attempt in range(1, 4):
        for model, out in (("stable", "same answer"), ("flaky", f"answer-{attempt}")):
            r = row(model, output=out)
            r.case_id = "case-001"
            r.attempt = attempt
            results.append(r)
    return results


# --- integration through save_report / run.json / report html -----------------


@pytest.mark.asyncio
async def test_statistics_flow_into_artifacts(tmp_path, fake_llm, api_key, openai_registry):
    from fastevals.report import save_report
    from fastevals.runner import run_evals

    calls = fake_llm(text="deterministic answer")
    config = RunConfig(
        prompt="hi",
        providers=frozenset({"openai"}),
        registry=str(openai_registry),
        nruns=3,
    )
    results = await run_evals(config)
    assert len(calls) == 6  # 3 attempts x off|low

    json_path, html_path = save_report(config, results, tmp_path)
    payload = json.loads(Path(json_path).read_text())
    stats = payload["statistics"]
    assert all(entry["runs"] == 3 for entry in stats["per_model"].values())
    stable_key = next(k for k in stats["per_model"] if "gpt-test" in k)
    assert stats["per_model"][stable_key]["output_consistency"] == 1.0

    html = Path(html_path).read_text()
    assert ">Statistics<" in html
    assert "Pairwise pass-rate comparison" in html
    assert "Wilson score" in html

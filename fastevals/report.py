"""Standalone single-file HTML report generation."""

import csv
import io
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import RunConfig
from .models import RunResult
from .stats import compute_statistics

__all__ = ["save_report"]


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _result_label(row: RunResult) -> str:
    return f"{row.provider}/{row.model} ({row.reasoning_effort or 'off'})"


def _format_usd(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 0.0001:
        return f"${value:.2e}"
    return f"${value:.6f}"


def _format_number(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.2f}{suffix}"
    return f"{value:,}{suffix}"


def _status_badge(row: RunResult) -> tuple[str, str]:
    if row.error:
        return "error", "Error"
    return "ok", "OK"


def _pass_badge(row: RunResult) -> str:
    if not row.evaluation or row.evaluation.get("passed") is None:
        return "<span class='muted'>—</span>"
    if row.evaluation["passed"]:
        return "<span class='badge ok'>Pass</span>"
    return "<span class='badge error'>Fail</span>"


def _output_body(row: RunResult) -> str:
    if row.error:
        return _escape(str(row.error))
    if isinstance(row.output, (dict, list)):
        return _escape(json.dumps(row.output, ensure_ascii=False, indent=2))
    return _escape(str(row.output or ""))


def _render_config_items(config: RunConfig) -> str:
    """Render only the configuration rows that actually have a value."""
    items = [
        ("Prompt", config.prompt.strip() if config.prompt else None),
        ("Dataset", config.dataset),
        ("Runs per case", str(config.nruns) if config.nruns > 1 else None),
        ("File", config.file),
        ("Image", config.image),
        (
            "Structured output",
            json.dumps(config.structured_output, ensure_ascii=False, indent=2) if config.structured_output else None,
        ),
    ]
    visible = [(label, value) for label, value in items if value]
    return "".join(
        f"<div class='config-item'><span class='config-key'>{_escape(label)}</span>"
        f"<div class='config-value'>{_escape(str(value))}</div></div>"
        for label, value in visible
    )


def _render_summary_cards(results: list[RunResult]) -> str:
    ok_rows = [row for row in results if row.ok]
    total_cost = sum(row.total_cost_usd or 0 for row in results)
    latencies = [row.latency_ms for row in ok_rows if row.latency_ms is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    total_tokens = sum(
        (row.input_tokens or 0) + (row.output_tokens or 0) + (row.reasoning_tokens or 0) for row in results
    )
    cards = [
        ("Models", str(len(results)), "Runs in this matrix"),
        ("Success", f"{len(ok_rows)}/{len(results)}", "Completed without error"),
        ("Total cost", _format_usd(total_cost), "Sum across all runs"),
        ("Avg latency", _format_number(avg_latency, " ms"), "Mean end-to-end time"),
        ("Total tokens", _format_number(total_tokens), "Input + output + reasoning"),
    ]
    if len(ok_rows) >= 2:
        fastest = min(ok_rows, key=lambda row: row.latency_ms if row.latency_ms is not None else float("inf"))
        priced = [row for row in ok_rows if row.total_cost_usd is not None]
        quickest = max(ok_rows, key=lambda row: row.tokens_per_second or 0)
        cards += [
            ("Fastest", _result_label(fastest), _format_number(fastest.latency_ms, " ms")),
            ("Top throughput", _result_label(quickest), _format_number(quickest.tokens_per_second, " tok/s")),
        ]
        if priced:
            cheapest = min(priced, key=lambda row: row.total_cost_usd or float("inf"))
            cards.append(("Cheapest", _result_label(cheapest), _format_usd(cheapest.total_cost_usd)))
    return "".join(
        f"<div class='card stat-card'><div class='stat-label'>{_escape(label)}</div>"
        f"<div class='stat-value'>{_escape(value)}</div>"
        f"<div class='stat-hint'>{_escape(hint)}</div></div>"
        for label, value, hint in cards
    )


def _model_aggregates(results: list[RunResult]) -> list[dict[str, Any]]:
    grouped: dict[str, list[RunResult]] = {}
    for row in results:
        grouped.setdefault(_result_label(row), []).append(row)
    aggregates = []
    for label, rows in sorted(grouped.items()):
        ok_rows = [row for row in rows if row.ok]
        latencies = [row.latency_ms for row in ok_rows if row.latency_ms is not None]
        scored = [
            row.evaluation["passed"] for row in rows if row.evaluation and row.evaluation.get("passed") is not None
        ]
        aggregates.append(
            {
                "label": label,
                "runs": len(rows),
                "success_rate": len(ok_rows) / len(rows),
                "pass_rate": (sum(1 for passed in scored if passed) / len(scored)) if scored else None,
                "avg_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
                "total_cost_usd": sum(row.total_cost_usd or 0 for row in rows),
            }
        )
    return aggregates


def _render_aggregate_table(results: list[RunResult]) -> str:
    rows = []
    for aggregate in _model_aggregates(results):
        pass_rate = aggregate["pass_rate"]
        rows.append(
            "<tr>"
            f"<td>{_escape(aggregate['label'])}</td>"
            f"<td>{aggregate['runs']}</td>"
            f"<td>{aggregate['success_rate'] * 100:.0f}%</td>"
            f"<td>{f'{pass_rate * 100:.0f}%' if pass_rate is not None else '—'}</td>"
            f"<td>{_format_number(aggregate['avg_latency_ms'], ' ms')}</td>"
            f"<td>{_format_usd(aggregate['total_cost_usd'])}</td>"
            "</tr>"
        )
    return (
        "<table class='comparison-table'>"
        "<thead><tr><th>Model</th><th>Runs</th><th>Success</th>"
        "<th>Pass rate</th><th>Avg latency</th><th>Total cost</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_statistics_card(results: list[RunResult], config: RunConfig) -> str:
    """Per-model dispersion, pass-rate confidence intervals and pairwise verdicts."""
    stats = compute_statistics(results)
    rows = []
    for label, entry in stats["per_model"].items():
        latency = entry["latency_ms"]
        tps = entry["tokens_per_second"]
        cost = entry["cost_usd_per_run"]
        if entry.get("pass_rate") is not None:
            lo, hi = entry["pass_rate_ci95"]
            pass_cell = f"{entry['pass_rate'] * 100:.0f}% <span class='muted'>[{lo * 100:.0f}-{hi * 100:.0f}%]</span>"
        else:
            pass_cell = "<span class='muted'>—</span>"

        def fmt(stat: dict[str, Any] | None, unit: str = "") -> str:
            if not stat or stat.get("mean") is None:
                return "<span class='muted'>—</span>"
            cell = f"{stat['mean']:,.1f}{unit}"
            if stat.get("std") is not None:
                cv = f" (±{stat['std']:,.1f})" if stat["std"] else ""
                return f"{cell}{cv}"
            return cell

        consistency = (
            f"{entry['output_consistency'] * 100:.0f}%"
            if entry.get("output_consistency") is not None
            else "<span class='muted'>—</span>"
        )
        rows.append(
            "<tr>"
            f"<td>{_escape(label)}</td><td>{entry['runs']}</td>"
            f"<td>{pass_cell}</td>"
            f"<td>{fmt(latency, ' ms')}</td>"
            f"<td>{fmt(tps, ' t/s')}</td>"
            f"<td>{fmt(cost)}</td>"
            f"<td>{consistency}</td>"
            "</tr>"
        )

    pair_rows_list: list[str] = []
    for pair in stats["pairwise"]:
        marker = "&#9989; " if pair["verdict"].startswith(("A ", "B ")) else "&#10134; "
        pair_rows_list.append(
            "<tr>"
            f"<td>{_escape(pair['a'])}</td><td>{_escape(pair['b'])}</td>"
            f"<td>{pair['delta_pass_rate'] * 100:+.0f}%</td>"
            f"<td>{marker}{_escape(pair['verdict'])}</td>"
            "</tr>"
        )
    pair_rows = "".join(pair_rows_list) or (
        "<tr><td colspan='4'><span class='muted'>No scored cases to compare.</span></td></tr>"
    )

    head_stats = (
        "<tr><th>Model</th><th>Runs</th><th>Pass rate [95% CI]</th>"
        "<th>Latency</th><th>Throughput</th><th>Cost / run</th><th>Output consistency</th></tr>"
    )
    head_pairs = "<tr><th>Model A</th><th>Model B</th><th>Δ pass rate</th><th>Verdict</th></tr>"
    note = (
        "<p class='muted' style='margin:10px 0 0'>Pass-rate intervals are Wilson score "
        "(95%). Overlapping intervals mean the difference is not distinguishable from "
        "noise at this sample size — add more cases or runs before concluding.</p>"
    )
    pairs_block = (
        f"<h3 class='section-title' style='font-size:15px;margin-top:18px'>Pairwise pass-rate comparison</h2>"
        f"<div class='table-wrap'><table class='comparison-table'><thead>{head_pairs}</thead>"
        f"<tbody>{pair_rows}</tbody></table></div>{note}"
    )
    return (
        "<section class='card' style='margin-bottom:20px'>"
        "<h2 class='section-title'>Statistics</h2>"
        f"<div class='table-wrap'><table class='comparison-table'><thead>{head_stats}</thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>" + pairs_block + "</section>"
    )


def _render_comparison_table(results: list[RunResult], show_case: bool, show_eval: bool) -> str:
    rows = []
    for row in results:
        status_class, status_text = _status_badge(row)
        cells = [f"<td><span class='badge {status_class}'>{status_text}</span></td>"]
        if show_case:
            cells.append(f"<td>{_escape(row.case_id)}</td><td>{row.attempt}</td>")
        cells += [
            f"<td>{_escape(row.provider)}</td>",
            f"<td>{_escape(row.model)}</td>",
            f"<td>{_escape(row.reasoning_effort or 'off')}</td>",
            f"<td>{_format_number(row.latency_ms, ' ms')}</td>",
            f"<td>{_format_number(row.time_to_first_token_ms, ' ms')}</td>",
            f"<td>{_format_number(row.tokens_per_second, ' t/s')}</td>",
            f"<td>{_format_number(row.input_tokens)}</td>",
            f"<td>{_format_number(row.output_tokens)}</td>",
            f"<td>{_format_number(row.reasoning_tokens)}</td>",
            f"<td>{_format_number(row.cached_tokens)}</td>",
            f"<td>{_format_usd(row.total_cost_usd)}</td>",
            f"<td>{_escape(row.finish_reason or '—')}</td>",
        ]
        if show_eval:
            cells.append(f"<td>{_pass_badge(row)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    head_cells = ["Status"]
    if show_case:
        head_cells += ["Case", "Attempt"]
    head_cells += [
        "Provider",
        "Model",
        "Reasoning",
        "Latency",
        "TTFT",
        "Throughput",
        "In",
        "Out",
        "Reason",
        "Cached",
        "Cost",
        "Finish",
    ]
    if show_eval:
        head_cells.append("Eval")
    head = "".join(f"<th class='sortable'>{label}</th>" for label in head_cells)
    return (
        "<div class='table-tools'>"
        "<input id='table-filter' type='search' placeholder='Filter rows…'>"
        "<span class='table-tools-spacer'></span>"
        "<button id='export-csv' type='button'>Export CSV</button>"
        "<button id='export-md' type='button'>Export Markdown</button>"
        "</div>"
        "<table class='comparison-table' id='comparison-table'>"
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_result_cards(results: list[RunResult]) -> str:
    cards = []
    for index, row in enumerate(results, start=1):
        status_class, status_text = _status_badge(row)
        metrics = [
            ("Latency", _format_number(row.latency_ms, " ms")),
            ("Throughput", _format_number(row.tokens_per_second, " tok/s")),
            ("Input tokens", _format_number(row.input_tokens)),
            ("Output tokens", _format_number(row.output_tokens)),
            ("Reasoning tokens", _format_number(row.reasoning_tokens)),
            ("Cached tokens", _format_number(row.cached_tokens)),
            ("Input cost", _format_usd(row.input_cost_usd)),
            ("Output cost", _format_usd(row.output_cost_usd)),
            ("Reasoning cost", _format_usd(row.reasoning_cost_usd)),
            ("Cached cost", _format_usd(row.cached_cost_usd)),
            ("Total cost", _format_usd(row.total_cost_usd)),
            ("Finish reason", row.finish_reason or "—"),
            ("Response ID", row.response_id or "—"),
        ]
        metric_grid = "".join(
            f"<div class='metric'><span class='metric-key'>{_escape(label)}</span>"
            f"<span class='metric-value'>{_escape(str(value))}</span></div>"
            for label, value in metrics
        )
        eval_note = ""
        if row.evaluation and row.evaluation.get("detail"):
            eval_note = f"<p class='eval-detail'>{_escape(str(row.evaluation['detail']))}</p>"
        meta_line = _escape(f"{row.case_id} · attempt {row.attempt}")
        reasoning_label = _escape(row.reasoning_effort or "off")
        cards.append(
            f"<article class='result-card {status_class}'>"
            f"<div class='result-head'>"
            f"<div><h2>#{index} {_escape(row.provider)} / {_escape(row.model)}</h2>"
            f"<p class='meta'>{meta_line} · reasoning {reasoning_label}</p></div>"
            f"<span class='badge {status_class}'>{status_text}</span>"
            f"</div>"
            f"<div class='metric-grid'>{metric_grid}</div>"
            f"{eval_note}"
            f"<pre class='output-block'>{_output_body(row)}</pre>"
            f"</article>"
        )
    return "".join(cards)


def _chart_payload(results: list[RunResult]) -> dict[str, Any]:
    labels = [_result_label(row) for row in results]
    return {
        "labels": labels,
        "latency_ms": [row.latency_ms or 0 for row in results],
        "ttft_ms": [row.time_to_first_token_ms or 0 for row in results],
        "tokens_per_second": [row.tokens_per_second or 0 for row in results],
        "input_tokens": [row.input_tokens or 0 for row in results],
        "output_tokens": [row.output_tokens or 0 for row in results],
        "reasoning_tokens": [row.reasoning_tokens or 0 for row in results],
        "cached_tokens": [row.cached_tokens or 0 for row in results],
        "input_cost_usd": [row.input_cost_usd or 0 for row in results],
        "output_cost_usd": [row.output_cost_usd or 0 for row in results],
        "reasoning_cost_usd": [row.reasoning_cost_usd or 0 for row in results],
        "cached_cost_usd": [row.cached_cost_usd or 0 for row in results],
        "total_cost_usd": [row.total_cost_usd or 0 for row in results],
        "statuses": ["error" if row.error else "ok" for row in results],
    }


def _table_tools_script() -> str:
    return """
    const table = document.getElementById("comparison-table");
    if (table) {
      const tbody = table.tBodies[0];
      const filterInput = document.getElementById("table-filter");
      filterInput.addEventListener("input", () => {
        const query = filterInput.value.toLowerCase();
        for (const row of tbody.rows) {
          row.style.display = row.textContent.toLowerCase().includes(query) ? "" : "none";
        }
      });
      table.tHead.querySelectorAll("th").forEach((th, index) => {
        let ascending = true;
        th.addEventListener("click", () => {
          const parse = text => {
            const num = parseFloat(text.replace(/[^0-9.\\-eE]/g, ""));
            return Number.isNaN(num) ? null : num;
          };
          const rows = [...tbody.rows].filter(row => row.style.display !== "none");
          rows.sort((a, b) => {
            const av = parse(a.cells[index].textContent);
            const bv = parse(b.cells[index].textContent);
            if (av === null && bv === null) return a.cells[index].textContent.localeCompare(b.cells[index].textContent);
            if (av === null) return 1;
            if (bv === null) return -1;
            return ascending ? av - bv : bv - av;
          });
          rows.forEach(row => tbody.appendChild(row));
          table.tHead.querySelectorAll("th").forEach(other => other.classList.remove("sorted-asc", "sorted-desc"));
          th.classList.add(ascending ? "sorted-asc" : "sorted-desc");
          ascending = !ascending;
        });
      });
      const exportRows = () => [...table.tHead.rows[0].cells].map(cell => cell.textContent);
      const exportData = () => [...tbody.rows]
        .filter(row => row.style.display !== "none")
        .map(row => [...row.cells].map(cell => cell.textContent.trim()));
      document.getElementById("export-csv").addEventListener("click", () => {
        const lines = [exportRows(), ...exportData()].map(cells =>
          cells.map(cell => '"' + cell.replace(/"/g, '""') + '"').join(",")
        );
        download("fastevals-report.csv", lines.join("\\n"), "text/csv");
      });
      document.getElementById("export-md").addEventListener("click", () => {
        const escapePipe = text => text.replace(/\\|/g, "\\\\|");
        const header = exportRows();
        const rows = exportData().map(cells => cells.map(escapePipe));
        const lines = [
          "| " + header.map(escapePipe).join(" | ") + " |",
          "| " + header.map(() => "---").join(" | ") + " |",
          ...rows.map(cells => "| " + cells.join(" | ") + " |"),
        ];
        download("fastevals-report.md", lines.join("\\n"), "text/markdown");
      });
      function download(filename, content, mime) {
        const link = document.createElement("a");
        link.href = URL.createObjectURL(new Blob([content], { type: mime }));
        link.download = filename;
        link.click();
        URL.revokeObjectURL(link.href);
      }
    }
    """


def _config_section(config: RunConfig) -> str:
    items = _render_config_items(config)
    if not items:
        return ""
    return (
        "<section class='card' style='margin-bottom:20px'>"
        "<h2 class='section-title'>Run configuration</h2>"
        f"<div class='config-grid'>{items}</div></section>"
    )


def render_html_report(config: RunConfig, results: list[RunResult], created_at: str) -> str:
    from . import __version__

    chart_data = json.dumps(_chart_payload(results), ensure_ascii=False)
    ok_count = sum(1 for row in results if row.ok)
    overall_status = "ok" if ok_count == len(results) else ("partial" if ok_count else "error")
    overall_label = {"ok": "All passed", "partial": "Partial success", "error": "All failed"}[overall_status]
    show_case = len({row.case_id for row in results}) > 1 or config.nruns > 1
    show_eval = any(row.evaluation and row.evaluation.get("passed") is not None for row in results)
    show_aggregates = show_case
    aggregate_section = (
        f"<section class='card' style='margin-bottom:20px'>"
        f"<h2 class='section-title'>Model aggregates</h2>"
        f"<div class='table-wrap'>{_render_aggregate_table(results)}</div></section>"
        if show_aggregates
        else ""
    )
    stats_section = _render_statistics_card(results, config) if show_aggregates else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>fastevals report</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #eef2f8;
      --panel: #ffffff;
      --text: #152033;
      --muted: #667085;
      --line: #dbe3ef;
      --accent: #2563eb;
      --ok: #15803d;
      --ok-bg: #ecfdf3;
      --error: #b42318;
      --error-bg: #fef3f2;
      --warn: #b54708;
      --warn-bg: #fffaeb;
      --shadow: 0 10px 30px rgba(21, 32, 51, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 15px/1.5 Inter, ui-sans-serif, system-ui, sans-serif;
      color: var(--text);
      background: linear-gradient(180deg, #f8fbff 0%, var(--bg) 220px, var(--bg) 100%);
    }}
    .page {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    .hero, .card, .result-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
    }}
    .hero {{
      padding: 28px;
      margin-bottom: 20px;
    }}
    .hero-top {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      flex-wrap: wrap;
    }}
    h1, h2 {{ margin: 0 0 8px; }}
    .meta {{ color: var(--muted); margin: 0; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }}
    .badge.ok {{ background: var(--ok-bg); color: var(--ok); }}
    .badge.error {{ background: var(--error-bg); color: var(--error); }}
    .badge.partial {{ background: var(--warn-bg); color: var(--warn); }}
    .grid {{
      display: grid;
      gap: 16px;
    }}
    .stats-grid {{
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      margin-bottom: 20px;
    }}
    .charts-grid {{
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      margin-bottom: 20px;
    }}
    .card {{
      padding: 20px;
    }}
    .stat-card .stat-label {{
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .stat-card .stat-value {{
      font-size: 20px;
      font-weight: 700;
      margin: 8px 0 4px;
      word-break: break-word;
    }}
    .stat-card .stat-hint {{
      color: var(--muted);
      font-size: 13px;
    }}
    .section-title {{
      margin: 0 0 14px;
      font-size: 18px;
    }}
    .config-grid {{
      display: grid;
      gap: 14px;
    }}
    .config-item {{
      display: grid;
      gap: 6px;
    }}
    .config-key {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 700;
    }}
    .config-value {{
      white-space: pre-wrap;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 14px;
    }}
    .muted {{ color: var(--muted); }}
    .table-tools {{
      display: flex;
      gap: 8px;
      align-items: center;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }}
    .table-tools-spacer {{ flex: 1; }}
    .table-tools input {{
      flex: 0 1 260px;
      padding: 8px 12px;
      border: 1px solid var(--line);
      border-radius: 10px;
      font: inherit;
    }}
    .table-tools button {{
      padding: 8px 14px;
      border: 1px solid var(--line);
      background: #f8fafc;
      border-radius: 10px;
      font: inherit;
      font-weight: 600;
      cursor: pointer;
    }}
    .table-tools button:hover {{ background: #eef2f8; }}
    .comparison-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .comparison-table th, .comparison-table td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    .comparison-table th {{
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .comparison-table th.sortable {{ cursor: pointer; user-select: none; white-space: nowrap; }}
    .comparison-table th.sortable:hover {{ color: var(--accent); }}
    .comparison-table th.sorted-asc::after {{ content: " ▲"; }}
    .comparison-table th.sorted-desc::after {{ content: " ▼"; }}
    .comparison-table tbody tr:hover {{
      background: #f8fafc;
    }}
    .table-wrap {{
      overflow: auto;
    }}
    .result-card {{
      padding: 22px;
      margin-bottom: 16px;
    }}
    .result-card.error {{
      border-color: #fecdca;
    }}
    .result-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 16px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .metric {{
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
    }}
    .metric-key {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 4px;
    }}
    .metric-value {{
      font-weight: 600;
      word-break: break-word;
    }}
    .eval-detail {{
      color: var(--error);
      margin: 0 0 12px;
      font-size: 13px;
    }}
    .output-block {{
      white-space: pre-wrap;
      margin: 0;
      background: #101827;
      color: #e8eef8;
      border-radius: 12px;
      padding: 16px;
      overflow: auto;
    }}
    canvas {{
      width: 100% !important;
      height: 280px !important;
    }}
  </style>
</head>
<body>
  <div class="page">
    <header class="hero">
      <div class="hero-top">
        <div>
          <h1>fastevals report</h1>
          <p class="meta">{len(results)} model run(s) · fastevals v{__version__} · {_escape(created_at)}</p>
        </div>
        <span class="badge {overall_status}">{overall_label}</span>
      </div>
    </header>

    <section class="grid stats-grid">
      {_render_summary_cards(results)}
    </section>

    {_config_section(config)}

    <section class="grid charts-grid">
      <div class="card"><h2 class="section-title">Latency</h2><canvas id="latencyChart"></canvas></div>
      <div class="card"><h2 class="section-title">Time to first token</h2><canvas id="ttftChart"></canvas></div>
      <div class="card"><h2 class="section-title">Throughput</h2><canvas id="throughputChart"></canvas></div>
      <div class="card"><h2 class="section-title">Token usage</h2><canvas id="tokensChart"></canvas></div>
      <div class="card"><h2 class="section-title">Cost breakdown</h2><canvas id="costChart"></canvas></div>
    </section>

    <section class="card" style="margin-bottom:20px">
      <h2 class="section-title">Comparison table</h2>
      <div class="table-wrap">{_render_comparison_table(results, show_case, show_eval)}</div>
    </section>

    {aggregate_section}
    {stats_section}

    <section>
      <h2 class="section-title">Detailed results</h2>
      {_render_result_cards(results)}
    </section>
  </div>

  <script>
    const chartData = {chart_data};
    const statusColors = chartData.statuses.map(status => status === "ok" ? "#2563eb" : "#ef4444");

    Chart.defaults.font.family = "Inter, ui-sans-serif, system-ui, sans-serif";
    Chart.defaults.color = "#667085";

    new Chart(document.getElementById("latencyChart"), {{
      type: "bar",
      data: {{
        labels: chartData.labels,
        datasets: [{{
          label: "Latency (ms)",
          data: chartData.latency_ms,
          backgroundColor: statusColors,
          borderRadius: 8,
        }}],
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: "ms" }} }} }},
      }},
    }});

    new Chart(document.getElementById("ttftChart"), {{
      type: "bar",
      data: {{
        labels: chartData.labels,
        datasets: [{{
          label: "TTFT (ms)",
          data: chartData.ttft_ms,
          backgroundColor: "#7c3aed",
          borderRadius: 8,
        }}],
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: "ms" }} }} }},
      }},
    }});

    new Chart(document.getElementById("throughputChart"), {{
      type: "bar",
      data: {{
        labels: chartData.labels,
        datasets: [{{
          label: "Tokens / second",
          data: chartData.tokens_per_second,
          backgroundColor: "#059669",
          borderRadius: 8,
        }}],
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: "tok/s" }} }} }},
      }},
    }});

    new Chart(document.getElementById("tokensChart"), {{
      type: "bar",
      data: {{
        labels: chartData.labels,
        datasets: [
          {{ label: "Input", data: chartData.input_tokens, backgroundColor: "#2563eb", stack: "tokens" }},
          {{ label: "Output", data: chartData.output_tokens, backgroundColor: "#7c3aed", stack: "tokens" }},
          {{ label: "Reasoning", data: chartData.reasoning_tokens, backgroundColor: "#d97706", stack: "tokens" }},
          {{ label: "Cached", data: chartData.cached_tokens, backgroundColor: "#94a3b8", stack: "tokens" }},
        ],
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
          x: {{ stacked: true }},
          y: {{ stacked: true, beginAtZero: true, title: {{ display: true, text: "tokens" }} }},
        }},
      }},
    }});

    new Chart(document.getElementById("costChart"), {{
      type: "bar",
      data: {{
        labels: chartData.labels,
        datasets: [
          {{ label: "Input", data: chartData.input_cost_usd, backgroundColor: "#2563eb", stack: "cost" }},
          {{ label: "Output", data: chartData.output_cost_usd, backgroundColor: "#7c3aed", stack: "cost" }},
          {{ label: "Reasoning", data: chartData.reasoning_cost_usd, backgroundColor: "#d97706", stack: "cost" }},
          {{ label: "Cached", data: chartData.cached_cost_usd, backgroundColor: "#94a3b8", stack: "cost" }},
        ],
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
          x: {{ stacked: true }},
          y: {{ stacked: true, beginAtZero: true, title: {{ display: true, text: "USD" }} }},
        }},
      }},
    }});

    {_table_tools_script()}
  </script>
</body>
</html>"""


def _dataset_payload(config: RunConfig) -> list[dict[str, Any]] | None:
    if not config.dataset:
        return None
    from .dataset import load_dataset

    return [case.as_dict() for case in load_dataset(config.dataset)]


def _markdown_export(results: list[RunResult]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter="|")
    header = ["status", "case", "attempt", "provider", "model", "reasoning", "latency_ms", "total_cost_usd", "eval"]
    writer.writerow(header)
    for row in results:
        writer.writerow(
            [
                "error" if row.error else "ok",
                row.case_id,
                row.attempt,
                row.provider,
                row.model,
                row.reasoning_effort,
                row.latency_ms,
                row.total_cost_usd,
                (row.evaluation or {}).get("passed"),
            ]
        )
    return buffer.getvalue()


def save_report(
    config: RunConfig,
    results: list[RunResult],
    output_dir: str | Path,
    include_html: bool = True,
) -> tuple[Path, Path | None]:
    """Persist ``run.json`` and the standalone HTML report for one run."""
    from . import __version__

    run_id = f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(UTC).isoformat()
    payload = {
        "fastevals_version": __version__,
        "prompt": config.prompt or None,
        "dataset": config.dataset,
        "cases": _dataset_payload(config),
        "nruns": config.nruns,
        "file": config.file,
        "image": config.image,
        "structured_output": config.structured_output,
        "created_at": created_at,
        "results": [row.as_dict() for row in results],
        "statistics": compute_statistics(results),
        "markdown_summary": _markdown_export(results),
    }
    json_path = run_dir / "run.json"
    html_path = run_dir / "report.html"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    if include_html:
        html_path.write_text(render_html_report(config, results, created_at))
    return json_path, html_path if include_html else None

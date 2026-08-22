# fastevals — Detailed Usage Guide

Everything the tool can do, in one place. For a quick overview see the
[README](https://github.com/semenovdv/fastevals#readme); this document is the full manual.

---

## Table of contents

1. [Concepts](#1-concepts)
2. [Installation & API keys](#2-installation--api-keys)
3. [CLI reference](#3-cli-reference)
4. [Model registry (TOML)](#4-model-registry-toml)
5. [Choosing what to compare: providers, selectors, tags](#5-choosing-what-to-compare)
6. [Datasets and evaluators](#6-datasets-and-evaluators)
7. [Structured output](#7-structured-output)
8. [Attachments: files, images, PDFs](#8-attachments)
9. [Reports and artifacts](#9-reports-and-artifacts)
10. [MCP server (agents)](#10-mcp-server-agents)
11. [Python API](#11-python-api)
12. [Exit codes and error handling](#12-exit-codes-and-error-handling)
13. [Configuration files & environment](#13-configuration-files-environment)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Concepts

| Concept | Meaning |
|---|---|
| **Run** | One execution of an evaluation matrix: every selected model × every case × every attempt |
| **Cell** | A single model call inside the matrix (`case × attempt × provider/model/effort`) |
| **Registry** | TOML file describing which models exist and their pricing |
| **Selector** | `provider/model[@effort]` string that picks specific cells without editing the registry |
| **Tag** | Named, saved list of selectors — a reusable model suite |
| **Evaluator** | Deterministic check applied to each output (`exact_match`, `contains`, `json_valid`, `regex`) |

A run never mixes these up: token buckets are disjoint (`input` excludes
`cached`, `output` excludes `reasoning`), so costs add up honestly.

## 2. Installation & API keys

```bash
python3 -m pip install fastevals            # CLI + library + bundled registry
python3 -m pip install 'fastevals[mcp]'     # + MCP server for AI agents
```

API keys are read from environment variables only. Which variable each model
uses is defined in the registry (`api_key_env`); the default fallback is
`<PROVIDER>_API_KEY` (`OPENAI_API_KEY`, `GEMINI_API_KEY`,
`OPENROUTER_API_KEY`).

`.env` files are loaded automatically, in priority order:

1. real environment variables always win,
2. `./.env` in your current directory,
3. `~/.config/fastevals/.env` — global store for MCP servers spawned by
   Claude Desktop (they run with `cwd=/` and cannot see project files).

Copy `.env.example` from the repository as a template. Keys are never written
to reports, logs or error messages.

## 3. CLI reference

```
fastevals [-p PROMPT] [-s SCHEMA] [-f FILE] [-i IMAGE] [-t TAG]
          [--providers LIST] [-m MODELS] [-r REGISTRY] [-d DATASET]
          [-n NRUNS] [-c N] [-o DIR] [-l | --version]
```

| Flag | Default | Description |
|---|---|---|
| `-p, --prompt` | — | Task prompt. Omit when `--dataset` supplies prompts |
| `-s, --structured-output` | — | Compact schema; compiles to strict JSON Schema (see §7) |
| `-f, --file PATH` | — | Attachment: image, PDF or text file (§8) |
| `-i, --image PATH` | — | Image attachment; may be combined with `--file` |
| `-t, --tag NAME` | — | Use a saved suite instead of `--models` (§5.3) |
| `--providers LIST` | `all` | Pipe-separated: `openai\|gemini\|openrouter\|all` |
| `-m, --models LIST` | — | Pipe-separated selectors, provider required (§5.2) |
| `-r, --registry PATH` | auto | Registry override (§4) |
| `-d, --dataset PATH` | — | JSONL/CSV evaluation cases (§6) |
| `-n, --nruns N` | 1 | Attempts per case — stability checks |
| `-c, --concurrency N` | 4 | Max parallel model calls |
| `-o, --out DIR` | `runs` | Where run artifacts are written |
| `-l, --list-models` | — | Print resolved registry as JSON and exit |
| `--version` | — | Print version and exit |

Matrix size = cases × nruns × cells-selected. Exit code is `0` when every
cell succeeded, `1` otherwise — safe to script against.

### Tag subcommands

```bash
fastevals tag add NAME -m "SELECTORS" [-d DESCRIPTION] [-r REGISTRY]
fastevals tag list            # saved suites + built-ins preview
fastevals tag show NAME       # one suite as JSON
fastevals tag remove NAME     # delete a saved suite
```

Selectors passed to `tag add` are validated against the registry
immediately; a tag that references unknown models cannot be saved. Tag names
`auto-*` are reserved for built-in suites (§5.4).

## 4. Model registry (TOML)

Resolution order: `--registry PATH` → `./config/models.toml` in your project
→ the registry bundled inside the package. The bundled copy ships with
`openai/gpt-5.6-luna` plus paste-ready reference blocks.

Every table defines one model:

```toml
["openai:gpt-5.6-luna"]              # table name is free-form; ids shown by --list-models
provider = "openai"                  # required: openai | gemini | openrouter
model = "gpt-5.6-luna"               # required: exact id the provider API accepts
api_key_env = "OPENAI_API_KEY"       # optional; defaults to <PROVIDER>_API_KEY
reasoning_efforts = "none|low"       # expands into one cell per effort
reasoning_parameter = "reasoning.effort"  # informational
input_cost_usd_per_mtok = 1.0        # pricing below is USD per 1M tokens
cached_input_cost_usd_per_mtok = 0.1 # alias accepted: cached_cost_usd_per_mtok
cached_write_cost_usd_per_mtok = 0.0 # reserved for cache-write accounting
output_cost_usd_per_mtok = 6.0
reasoning_cost_usd_per_mtok = 0      # optional; falls back to the output rate
timeout_s = 120                      # per-call timeout
max_retries = 2                      # extra attempts on transient failures
```

Validation rules enforced at load time:

- `model` is required; unknown keys fail with the key names listed
  (catches typos like `reasoning_efftort`);
- empty registries are rejected;
- `timeout_s ≥ 1`, `max_retries ≥ 0`.

Cost semantics: buckets are disjoint — `input_tokens = prompt − cached`,
`output_tokens = completion − reasoning` — so nothing is billed twice.
Reasoning tokens fall back to the output rate when no dedicated rate exists.
A bucket with zero tokens or no configured price reports `null`, never `0`.

## 5. Choosing what to compare

### 5.1 By provider

```bash
fastevals --providers openai ...          # every registry entry of openai
fastevals --providers "openai|gemini" ... # union
fastevals ...                             # default: all providers in the registry
```

### 5.2 By selector — exact cells

Grammar: `provider/model[@effort[,effort…]]`, selectors joined with `|`.

```bash
fastevals -m "openai/gpt-5.6-luna@high" ...
fastevals -m "openai/gpt-5.6-luna@none,low|openai/gpt-5.6-sol@low" ...
```

Rules:

- `model` matches **exactly** (case-insensitive) the registry `model`
  field — use official ids like `gpt-5.6-luna`, never nicknames;
- the full entry id printed by `--list-models` may be pasted verbatim:
  `openai:gpt-5.6-luna@low`;
- the provider part must match exactly — bare `gpt-5.6-luna` is rejected
  because several providers often serve the same model string, and the
  error lists every matching entry id so the fix is copy-paste;
- omitting `@efforts` selects all efforts of the model;
- unknown selectors fail listing all available ids.

### 5.3 By tag — reusable suites

```bash
fastevals tag add cheap -m "openai/gpt-5.6-luna@none|openai/gpt-5.6-luna@low" -d "Smoke tier"
fastevals --tag cheap -p "Summarize this"
```

Suites live in `~/.config/fastevals/tags.toml` (override with
`FASTEVAL_TAGS_FILE`), shared across terminals, MCP clients and the Python
API. Four built-in suites adapt to whatever your registry contains:

| Built-in | Cells |
|---|---|
| `auto-fast` | lightest effort of every model |
| `auto-deep` | deepest effort of every model |
| `auto-cheap` | cheapest model at its lightest effort |
| `auto-flagship` | priciest model across all efforts |

`--tag` and `--models` are mutually exclusive.

## 6. Datasets and evaluators

JSONL (one JSON object per line) or CSV (with a header row):

```jsonl
{"id": "capital-france", "prompt": "Capital of France? City only.", "expected": "Paris", "evaluator": "exact_match"}
{"id": "json-output", "prompt": "Return {\"status\": \"ok\"} as JSON.", "evaluator": "json_valid"}
{"id": "has-year", "prompt": "Mention the current year.", "evaluator": "regex", "pattern": "20[0-9]{2}"}
```

```csv
prompt,expected,evaluator
Capital of France? City only.,Paris,exact_match
```

Only `prompt` is required. Fields:

| Field | Purpose |
|---|---|
| `id` | Case label in reports (default `case-001`, `case-002`, …) |
| `prompt` | Required task text |
| `expected` | Reference answer for text-based evaluators |
| `evaluator` | `exact_match` · `contains` · `json_valid` · `regex` |
| `pattern` | Regex source for the `regex` evaluator |

Blank lines are skipped. Every output receives a verdict
(`evaluation.passed` / `detail`) in `run.json`, the HTML report aggregates
pass rates per model, and structured outputs already parsed as JSON count as
valid for `json_valid`.

### Statistics across repeated runs

With `-n/--nruns N` (N ≥ 2) reports gain a **Statistics** section and
`run.json` gains a `statistics` object, so you can tell signal from noise:

- **Per-model dispersion** — mean ± std and coefficient of variation for
  latency, throughput and cost across attempts.
- **Pass rate with Wilson 95% CI** — e.g. `67% [30–90%]`; a wide interval at
  small samples is shown honestly instead of a deceptively exact number.
- **Output consistency** — share of cases where all attempts of a model
  produced identical outputs (deterministic models score 100%).
- **Pairwise comparison** — for every model pair with scored cases: Δ pass
  rate plus a verdict. Overlapping Wilson intervals are reported as *not
  distinguishable at this sample size* rather than as a fabricated p-value;
  add more cases or runs before concluding that one model leads.

Methodology note: intervals are Wilson score intervals; no distributional
assumptions beyond the binomial proportion. The tool deliberately avoids
p-value theatre on tiny samples.

Repeat runs: `-n/--nruns 3` executes every case three times.

## 7. Structured output

Compact schema syntax compiles to a strict JSON Schema, is sent to the
provider as `response_format`, and **every response is validated locally**
before it reaches `run.json`:

```bash
fastevals -p "Extract invoice fields" \
  -s 'invoice_number:str("Unique identifier"),total:float("Amount incl. tax"),line_items:str[]("Items"),notes:str?' \
  --providers openai
```

| Syntax | JSON Schema |
|---|---|
| `name:str` (aliases `string`, `int`→`integer`, `float`→`number`, `bool`→`boolean`) | scalar type |
| `name:object`, `name:any` | free-form object / any value |
| `name:str[]` | array of strings |
| `notes:str?` | optional — removed from `required` |
| `desc:str("why")` | description hint for the model |

On validation failure the cell is marked as errored with the schema message;
the raw text stays visible in the report.

## 8. Attachments

```bash
fastevals -i screenshot.png -m "..."    # vision part (image_url)
fastevals -f contract.pdf  ...          # OpenAI-style file part
fastevals -f notes.md      ...          # decoded UTF-8, inlined into the prompt
```

Limit: 20 MB per attachment, validated before any network call. Unsupported
binary types fail with an explicit message.

## 9. Reports and artifacts

Each run creates `OUT/<timestamp>-<id>/` containing:

- **`run.json`** — machine-readable: config echo, resolved dataset cases,
  and per-cell rows with `output`, disjoint token buckets, per-bucket costs,
  `total_cost_usd`, `latency_ms`, `time_to_first_token_ms`,
  `tokens_per_second`, `evaluation{passed,detail}`, `error`;
- **`report.html`** — standalone dashboard (Chart.js from CDN): summary
  cards incl. fastest/cheapest/top-throughput, TTFT + latency + throughput +
  tokens + cost charts, sortable/filterable comparison table with CSV &
  Markdown export, detailed result cards, per-model aggregates when running
  datasets. Empty configuration rows are omitted entirely.

TTFT comes from streamed completions; providers that cannot stream report it
honestly as unavailable rather than faking it.

## 10. MCP server (agents)

```bash
pip install 'fastevals[mcp]'
claude mcp add fastevals -- fastevals-mcp
```

Tools exposed:

| Tool | Notes |
|---|---|
| `list_models(registry?)` | ids, efforts, pricing |
| `add_tag(name, models, description?, registry?)` | validates before saving |
| `list_tags()` | saved suites + built-ins |
| `remove_tag(name)` | built-ins are protected |
| `run_evaluation(prompt?, providers?, models?, tag?, cases?, dataset?, file?, image?, structured_output?, nruns?, registry?, out?, output_limit?)` | the runner; returns per-cell summary |
| `get_run(json_path)` | pass rate, errors, cost of one saved run |
| `list_runs(out?, limit?)` | recent runs, newest first |

Agent-specific behaviour:

- **inline `cases`**: `[{"prompt": ..., "expected": ..., "evaluator": ...}]`
  — shell-less clients (Claude Desktop) can evaluate many prompts without
  creating files;
- **`output_limit`** (default 400 chars) truncates long outputs *in the
  response only*; structured values stay native and full text remains in
  artifacts;
- tags created via MCP are immediately usable from the CLI and vice versa.

## 11. Python API

```python
import asyncio
from fastevals import RunConfig, run_evals, save_report

config = RunConfig(
    prompt="Summarize eval best practices",   # required unless dataset is set
    providers=frozenset({"openai"}),          # or frozenset({"all"})
    models={"openai/gpt-5.6-luna@none"},      # or tag="auto-fast" (not both)
    dataset=None,                             # str path, alternative to prompt
    nruns=1,
    max_concurrency=4,
    out="runs",
)
results = await run_evals(config)             # list[RunResult]
save_report(config, results, "runs")
```

`RunConfig` validates eagerly — blank prompts without a dataset, missing
attachment paths, unknown providers, `tag`+`models` conflicts raise before
any network call. `RunResult` exposes `.ok`, `.total_cost_usd`,
`.as_dict()` alongside every metric field.

Tag management from Python:

```python
from fastevals import save_tag, load_tags, remove_tag, resolve_tag

save_tag("cheap", ["openai/gpt-5.6-luna@none"], description="Smoke tier")
selectors = resolve_tag("auto-deep")           # built-ins work standalone
remove_tag("cheap")
```

## 12. Exit codes and error handling

| Code | Meaning |
|---|---|
| `0` | every cell completed successfully |
| `1` | at least one cell failed, or configuration was rejected |
| `2` | argparse usage error (unknown flag, malformed value) |

All failures print machine-readable JSON (`{"ok": false, "error": ...}`).
One failing provider never cancels the matrix — remaining cells complete and
the failure is reported per-cell. Transient provider errors retry with
exponential backoff (`max_retries`, default two extra attempts). Configuration
errors (missing key, bad selector) are never retried.

## 13. Configuration files & environment

| Path / var | Purpose |
|---|---|
| `./config/models.toml` | project registry override |
| `~/.config/fasteval/tags.toml` (var: `FASTEVAL_TAGS_FILE`) | saved suites |
| `~/.config/fastevals/.env` | global agent credentials store |
| `./.env` | project credentials store |

Precedence for keys: environment → project `.env` → global `.env`.

## 14. Troubleshooting

| Message | Cause & fix |
|---|---|
| `Missing API key environment variable: OPENAI_API_KEY` | export the key or put it in `./.env`; for Claude Desktop use `~/.config/fastevals/.env` |
| `Unknown provider(s): X` | typo, or provider missing from the registry; supported: `openai`, `gemini`, `openrouter` |
| `No models found for provider(s)` | registry has no entries for the selection — extend `config/models.toml` or fix `--registry` |
| `Selector 'X' omits the provider. Matching entries: …` | qualify as `provider/model`, or paste an id from `--list-models` |
| `Unknown tag 'X'. Available — saved: …; built-in: …` | typo in `--tag`; pick from the list |
| `Model output is not valid JSON` / `failed schema validation` | model ignored the schema — tighten descriptions, lower temperature via provider defaults, or switch to a stronger model cell |
| `Attachment exceeds 20 MB limit` | shrink the file; images/PDF/text only |
| `LiteLLM is not installed` | `pip install fastevals` includes it since 0.1.3 — upgrade |

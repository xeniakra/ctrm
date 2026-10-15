# Optiflow CTRM recap extractor

This repository contains a local-model prototype for extracting one physical
oil-products deal from a recap. Its design prioritises evidence and safe
abstention: unsupported mandatory values stay `null` and produce a focused
trader question.

## Payment terms

`deal.payment_terms` is `null` when the recap does not state usable terms;
otherwise it contains the original wording plus a structured payment schedule:

```json
{
  "as_written": "90 pct provisional ... balance within 5 days after final pricing",
  "late_interest": null,
  "payments": [
    {
      "document_condition": {
        "document_type": "documents",
        "relation": "against"
      },
      "due": {
        "day_basis": "banking_days",
        "days": 3,
        "event": "bill_of_lading",
        "qualifier": "within",
        "relation": "from"
      },
      "instrument": null,
      "irrevocable": null,
      "portion": {"kind": "percentage", "percentage": 90},
      "stage": "provisional"
    }
  ]
}
```

Unstated attributes remain `null`; in particular, `balance` is not converted
to an inferred percentage. A payment field may be supported by multiple exact,
single-line quotations. Each quotation is located independently against the
unchanged recap, so forwarded prefixes and line breaks never change offsets.

## Repository map

- `src/optiflow_ctrm/` — product code. The LLM adapter is isolated from
  deterministic validation, reference-data lookup, and calculations.
- `evals/` — committed evaluation cases and deterministic field-level scorer.
- `tests/` — unit and integration tests; these complement, rather than replace,
  the evaluation suite.
- `WRITEUP.md` — architecture, evaluation protocol and results, experiment log,
  MCP evidence, workflow history, reflections, limitations, and next steps.
- `agent_logs/` — genuine, unedited agent conversation exports.
- `data/` — supplied synthetic task data; do not alter the source fixtures.

## Install llama.cpp (macOS)

The default runtime requires the `llama-server` executable from llama.cpp;
installing this project's Python dependencies does not install it. With
[Homebrew](https://brew.sh/) installed, run:

```bash
brew install llama.cpp
llama-server --version
```

See the [Homebrew llama.cpp formula](https://formulae.brew.sh/formula/llama.cpp)
for package details. Homebrew installs its current version; the benchmark in
`WRITEUP.md` used build 10809, so a new installation may use a different build.
The evaluation runner records the actual runtime version.

If `llama-server` is not on `PATH`, pass its executable path with
`--binary /path/to/llama-server` to either server or evaluation command below.

## Commands

```bash
# from the repository root, create the local environment (Python 3.11+)
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,mcp]'

# first run: download the pinned Qwen3-4B GGUF and run the evaluation
python scripts/run_local_eval.py

# subsequent runs: use the verified cached model without downloading
python scripts/run_local_eval.py --offline

# Qwen3-30B-A3B FP8 evaluation via vLLM (set parallelism to visible GPU count)
pip install -e '.[big-model]'
python scripts/run_local_eval.py \
  --runtime vllm \
  --tensor-parallel-size 1 \
  --ctx-size 8192 \
  --startup-timeout 1800

pytest
python -m optiflow_ctrm.mcp.server
python -m mcp_client_demonstration.agent --bl-date 2026-09-14
```

The evaluation runner downloads the model into `models/`, verifies its SHA-256,
starts the local server, and stops it after evaluation. To keep a server running
for the MCP commands, run this in a separate terminal with the same environment
activated:

```bash
python scripts/start_llama_server.py --offline
```

This serves the cached model at `http://127.0.0.1:8080`. Omit `--offline` if the
model has not been downloaded yet. More options are in [scripts/README.md](scripts/README.md).

The vLLM runner pins `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` and persists its
Hugging Face download under `models/huggingface`. After the first successful
download, add `--offline` to prohibit network access and require the cache.

The eval report includes field-level quality, p50/p95 end-to-end latency,
aggregate completion throughput, and peak RAM/VRAM when the platform exposes
them. See [WRITEUP.md](WRITEUP.md) for results, design decisions, explicit
submission placeholders, and the supporting documentation.

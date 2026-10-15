# Scripts

Keep only small, reproducible operational commands here:

- start the pinned local inference server
- run the evaluation suite
- run the MCP demonstration

Scripts should print the model revision, quantisation, seed, and command
arguments used. Avoid hidden shell state and never embed secrets.

## Qwen llama.cpp server

The Q4_K_M GGUF artifact is pinned to an immutable Hugging Face revision and
verified by SHA-256. By default it is cached under `models/`, which is ignored
by Git but persists across server and evaluation runs.

Download or verify the cached model:

```bash
python scripts/model_cache.py
```

Start a locally bound llama.cpp server (this downloads only if the verified
artifact is not already cached):

```bash
python scripts/start_llama_server.py
```

`llama-server` must be installed and on `PATH`; alternatively pass
`--binary /path/to/llama-server`. Useful options include:

```bash
python scripts/start_llama_server.py --offline --threads 8
python scripts/start_llama_server.py --print-command
python scripts/start_llama_server.py --cache-dir /persistent/model/cache
```

Interrupted downloads remain as `.part` files and resume on the next run.
Completed files are reused only after their checksum passes.

Run the complete download → server → health check → evaluation → shutdown
workflow with one command:

```bash
python scripts/run_local_eval.py
```

The evaluation report and llama.cpp log are stored under `evals/results/`.
Each `<run>.cases/<recap>.json` contains only the extraction result: `deal`,
`questions`, and `provenance`. Failed extractions are stored as JSON `null`.
Expected values, scores, differences, errors, and raw attempts are stored in
the aggregate `<run>.json` evaluation report.
The script refuses to take ownership of a server already listening on its port,
unless that listener is verified as `llama-server`. By default a verified old
server is stopped before the new one starts. A non-llama listener is never
killed. Use `--no-replace-existing` to disable replacement. The script always
stops the new process it started, including after an evaluation error.

## MCP valuation demo

With the local inference server running, start the interactive MCP client:

```bash
python -m mcp_client_demonstration.agent
```

It launches `optiflow_ctrm.mcp.server` over stdio, has the local model translate
requests into the two schema-constrained MCP calls, and requests the actual B/L
date before resolving recap 02's relative pricing period. See
`mcp_client_demonstration/README.md` for the explicit-date reproducible example.

# Optiflow CTRM recap extraction — write-up

This document follows the six parts of `TASK.md`. 

## 1. Extractor

The input is unchanged recap text. The output is one strict Pydantic/JSON deal
object containing the fields required by `data/field_spec.md`, clarification
questions, and field-level provenance. Unknown JSON properties are rejected.

Every non-null extracted leaf has evidence. Source evidence records Python
character offsets `[start, end)` into the exact, unmodified input. Reference
evidence records a supplied CSV plus its key. Derived reference values can
therefore be traced to both recap wording and catalog data without pretending
that catalog values appeared in the recap.

 Extractor flow visual:  `./extractor-flow.svg`

 The flow starts with an LLM extraction and an additional round of repair for missing fields. It is followed by reference/evidence matching, which also has 2 rounds. If there is no evidence, the value becomes null. It then assembles the finalized JSON and adds questions for the null values.


### Implementation map

| Path | Responsibility |
| --- | --- |
| `models.py` | Canonical API models and JSON Schema |
| `extractor.py` | Prompt, local inference, and controlled repair orchestration |
| `provenance.py` | Evidence construction and provenance validation |
| `validation.py` | Commercial rules and safe abstention |
| `references.py` | Product and quotation reference-data resolution |
| `pricing.py` | Quote lookup and deterministic price aggregation |
| `runtime/small_model/client.py` | Qwen3-4B/llama.cpp constrained-inference transport |
| `runtime/big_model/client.py` | Qwen3-30B-A3B/vLLM constrained-inference transport |
| `mcp/server.py` | Thin MCP adapter over the application functions |

The MCP layer exposes the same application functions used by tests and other
callers; it is not a second extraction or pricing implementation.

## 2. Self-hosted model and output-shape enforcement

The default extraction model is the open-weights
`Qwen/Qwen3-4B-Instruct-2507`, using the pinned Q4_K_M GGUF artifact served by
llama.cpp on localhost. Its dense 4B architecture and 4-bit quantisation make
it a practical baseline for the available local hardware, with a small memory
footprint and instruction tuning suited to structured extraction.

The larger comparison candidate is
`Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`, pinned to revision
`e87d00b8a62c184ac41942416fc9e861a84ad71b` and configured for self-hosted vLLM.
Its mixture-of-experts architecture has about 30B total parameters but activates
about 3B per token, offering a larger parameter pool without the per-token
compute of a dense 30B model. The FP8 checkpoint reduces weight memory relative
to BF16, although the full expert weights still need to be accommodated.

These two models were selected to compare a compact local baseline with a
larger GPU candidate from the same Qwen3 instruction-tuned release family.
Both use the same extraction schema and evaluation fixtures. This tests whether
the larger model improves extraction quality enough to justify its deployment
cost; it does not isolate parameter count, since architecture, quantisation and
runtime also differ. Both configurations' performance was tested on an
NVIDIA L40S. The small model was used mainly for local development on macOS.

Output control happens in layers:

1. Pydantic models in `src/optiflow_ctrm/models.py` define the canonical JSON
   Schema and reject unknown keys.
2. The llama.cpp runtime adapter sends that schema as `response_format`.
   llama.cpp compiles it to a grammar and constrains token generation, so the
   model cannot emit arbitrary prose or a structurally unrelated JSON shape.
3. Pydantic validates the returned value again in the application.
4. `extractor.py` may retry a failed generation and request a constrained
   repair. Deterministic evidence and commercial validation run after that.

These mechanisms are complementary rather than four sequential modes. JSON
Schema is the application contract; grammar-constrained decoding is how the
selected server enforces it during generation; Pydantic validates at the trust
boundary; retry-and-repair is used only when an attempt still fails validation
or required evidence checks. Prompt-only JSON was rejected because it permits
malformed output. Native model-specific function calling was not used for the
extractor because the same schema path works across the llama.cpp and vLLM
adapters and is easier to measure.

For the MCP demo, the local model selects one action at a time using a JSON
Schema assembled from discovered MCP tools and local calculator actions. The
server validates tool inputs, and `parse_recap` uses the extraction schema and
validation described above. The client loop is described in Part 5.

## 3. Evaluation harness

The suite contains 18 committed cases: the three supplied recaps plus channel,
formatting, amendment, missing-field, ambiguity, unit, pricing, and distractor
variants. Run it with:

```bash
python scripts/run_local_eval.py --offline
```

The report scores leaf fields rather than judging whether the document “looks
right.” It separately counts correct values, incorrect values, correct
abstentions, wrong abstentions, and unsupported non-null values (“confident
nonsense”). It also reports schema validity, provenance validity, and question
recall. Expected outputs are committed.

Question recall currently matches clarification questions by their target JSON
Pointer. It does **not** independently prove that arbitrary question wording is
specific or contains every concept a trader needs to answer. The repository
policy that questions should be required-information-specific and answerable by
a trader is an interpretation of `TASK.md`'s “specific question” requirement,
recorded in `evals/AGENTS.md`; it is not a protocol supplied verbatim by the
task. Canonical expected questions are human-reviewed, but semantic scoring of
generated wording remains a limitation.

### Fixture metadata

- `id` is the stable case identifier used in reports.
- `source.path` identifies the immutable recap input.
- `source.sha256` (sometimes described informally as `source_hash`) is the
  SHA-256 digest of the exact source bytes. The loader refuses to evaluate a
  silently changed source against stale offsets and expectations.
- `tags` are optional slice labels such as `must_abstain`, `mixed_units`, or
  `amendment`; they are useful for grouped analysis but are not required by
  `TASK.md` and are not currently stored in the fixture JSON.
- `rationale` explains the commercial or model failure a case is intended to
  catch. Rationales live in `evals/cases_explain/*.md`, separate from the
  executable expected-output JSON.

The latest complete full-suite JSON report is retained in `evals/results`.

The recaps have been generated automatically by a separate agent with no access to the code to cover as many scenarios as possible and verified manually to check the intended scenario.

## 4. Performance and cost

Metrics below are end-to-end per recap. Aggregate tokens/s is total completion
tokens divided by summed successful inference-request duration. Peak RAM is
the maximum summed resident memory observed for the owned model-server process
tree. 

| Configuration | Run | Cases | Accuracy | Unsupported non-null | p50 | p95 | Aggregate tokens/s | Peak RAM | Peak VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B Q4_K_M / llama.cpp | `20260910T155926Z` | 18 | 99.5% | 0 | 35.531 s | 49.331 s | 26.4 | 7265.9 MiB | Unavailable |
| Qwen3-30B-A3B FP8 / vLLM, 1× L40S | `20260910T183209Z` | 18 | 97.8% | 1 | 5.376 s | 10.413 s | 134.9 | 11765.6 MiB | Unavailable |

Both configurations produced schema-valid and provenance-valid output for all
18 cases and recorded zero wrong abstentions.

The vLLM configuration delivered 6.61× lower median latency, 4.74× lower p95
latency, and 5.11× higher aggregate completion throughput. The
larger model, however, did not improve quality or abstention safety on this
suite (though it was already high). 
The measurements are retained in `stat_llama_4b` and `stat_vllm_30b`.

Observed optimization work: reducing context improved the measured pipeline by
roughly 10%; warm-up did not materially improve it; the attempted llama.cpp
batch, thread, and Flash Attention changes did not show a useful gain. A real
GPU follow-up should rerun both configurations on controlled hardware with
working VRAM accounting, then investigate prefix caching and tensor parallelism.
The earlier optimization claims should be tied to retained run IDs before
submission.

## 5. MCP server and provisional valuation

The MCP server exposes exactly the two requested tools:

1. `parse_recap(text)`
2. `get_price_quote(quote_code, date_or_period, statistic)`

The client gives the local model the user's question, server instructions, and
available action schemas. Its decision loop is:

```text
local model selects an action
  -> client executes the selected MCP tool or local action
  -> result is returned to the model for its next decision
  -> model eventually requests the valuation calculator to produce the answer
```

The MCP server loads `src/optiflow_ctrm/mcp/provisional_valuation.md` into its
initialization `instructions`. This guide explains how to use the extracted
deal and quote, resolve B/L-relative dates, convert units, add the differential,
apply FX, and distinguish cargo value from provisional payment. The client
passes these instructions and the discovered tool descriptions to the model.

The model chooses the MCP tool and its arguments. For the recap text, it uses
the host reference `$recap`; the client substitutes the unmodified stored text.
Quote arguments pass directly to the MCP server, without comparison against a
preconstructed request. Local actions `ask_bl_date`, `resolve_pricing_period`,
and `calculate_provisional_value` obtain user input and perform deterministic
arithmetic using stored results. The model chooses when to request them.

There is one inference per decision, with an eight-action cap to stop loops.
Malformed actions and tool or calculator errors are reported; the client does
not insert fallback calls. Extraction clarification questions stop valuation
and are returned to the trader. The calculator still verifies that the quote
matches the deal before using it in financial arithmetic.

The local LLM server must be running before starting the demo. From the
repository root, with the project virtual environment activated, start it in
one terminal and leave it running:

```bash
python scripts/start_llama_server.py --offline --no-replace-existing
```

This serves the cached model at `http://127.0.0.1:8080`. Omit `--offline` on
the first run if the model has not been downloaded yet.

In a second terminal, also at the repository root with the virtual environment
activated, run the demo with an explicit scenario B/L date:

```bash
python -m mcp_client_demonstration.agent --bl-date 2026-09-14
```

The demo starts its MCP server automatically; the LLM server is the separate
process started above.

Tests use a scripted model and simulated MCP session to verify action dispatch,
feedback of tool results, alternative action order, original recap preservation,
and explicit failure without fallback. Financial expectations remain the same.
These tests verify the client implementation, not live-model reliability.


### Rules files
The actual rules are in `AGENTS.md`, `src/optiflow_ctrm/AGENTS.md`,
and `evals/AGENTS.md`. They are scoped to repository-wide safety, product code,
and evaluation work. Documentation-integrity rules formerly scoped under
`docs/` were consolidated into the root rules when that directory was removed.

### Task list and mid-flight changes

1. Design the schema, provenance, questions, and audit trail.
2. Hand-author expected outputs for the three supplied recaps.
3. Implement normalization and field-level scoring.
4. Expand to 15–20 cases.
5. Establish the first local-model baseline.
6. Implement constrained extraction and validation.
7. Implement price/product lookup and deterministic calculation.
8. Add both MCP tools.
9. Exercise the provisional-value workflow.
10. Test tool-call reliability across diverse cases.
11. Benchmark two model/configuration variants.
12. Record latency, throughput, memory, quality, and spend.
13. Finish README, experiment log, reflections, and agent logs.
14. Verify the repository from a clean run.

The task list has kept almost the same order throughout the process, although "Expand to 15–20 cases" happened mainly at the end to use time efficiently and focus on developing the main blocks.

Subsequent client changes: the redundant translation/comparison layer was
removed. A first revision used a fixed Python sequence; the user corrected
that interpretation, and model-controlled action selection was implemented.
The reliability work in item 10 describes the earlier experiment retained
above; repeated trials are no longer part of running the demo.


### Agent conversations

Under ./agent_log

### Agent Development cost

Limited by OpenAI subscription

### Three reflections

#### 1. Code that passed tests but was still wrong

Describe the evidence, why the tests missed it, and the workflow change made as
a result.

In recap 2, the extracted payment term contained two installments spread across two quoted lines: 90% provisional payment against documents and the balance after final pricing. The code originally represented this as one joined string, but its provenance covered only the first physical line. The tests still passed because they didn't check whether this specific property worked correctly. 
Currently: the model identifies the complete payment wording. Deterministic code then matches it across quoted source lines, preserving separate offsets into the original recap, and parses the installments. The end boundary comes from the model’s selected wording; there is no independent rule for detecting the end of the payment section. Regression tests check that recap 2 includes both clauses and their supporting spans.

It creates a structure of multiple installments:

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
            "portion": {
              "kind": "percentage",
              "percentage": 90
            },
            "stage": "provisional"
          },
          {
            "document_condition": null,
            "due": {
              "day_basis": "days",
              "days": 5,
              "event": "final_pricing",
              "qualifier": "within",
              "relation": "after"
            },
            "instrument": null,
            "irrevocable": null,
            "portion": {
              "kind": "balance",
              "percentage": null
            },
            "stage": "final"
          }
        ]

Structuring the payment schedule made each installment’s amount, deadline, and triggering event explicit and testable.

#### 2. Three rules and why they exist

Choose three actual rules from the committed `AGENTS.md` files and explain the
concrete failure mode each prevents.

1. Do not invent commercial values. Missing or ambiguous mandatory data must be
   `null` with a field-specific question.

This is a basic guideline and one of the strongest for the whole project. Whether the property is calculated by an LLM or deterministic code, it should be verified.

2. `pricing.py` and `references.py` are deterministic. They must not call an
   LLM or choose missing commercial assumptions.

   These rules make sure only certain parts are able to call an LLM.

3. Add a focused test or evaluation case before changing extraction behaviour.

  Extractor logic is the most complex part of the system. Every little change might break previously working examples. Therefore, it should be tested thoroughly.

  
#### 3. What I refuse to delegate

In terms of writing the code, I do not refuse to delegate to the agent. I refuse to give up my understanding of the system, instructions and the covered examples. To make sure I understand it correctly, I work with multiple agents, while a separate agent, which is not involved in the development, is used for sanity tests/questions.


### Future improvements

1. Model inference performance improvement was not tested thoroughly. Specifically, the big model configuration might be improved by batching and threading on multiple GPUs and different hardware.

2. Choice/comparison of different models. The models here have been selected quite arbitrarily by the number of params. However, they also have differences in runtime, architecture and quantisation, and if we really want to isolate how each one of these conditions impacts performance, we need to test only one change each time.

3. Testing each layer separately. For example, the extractor currently contains separate decision branches like an LLM extracting values and finding evidence + deterministic logic supporting them. Each of these steps needs to be tested separately in terms of performance and accuracy.

4. Testing additional eval scenarios. The goal is to explore and cover different cases, like those that lead to diverse extracted values, cover different conversation styles/platforms, contain conflicting information, cause correct/missing abstention, or increase the risks of invented answers. However, due to limited time and familiarity with the domain, the current examples might not cover it all.




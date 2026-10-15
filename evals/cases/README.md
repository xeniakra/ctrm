# Evaluation fixtures

Each case is a committed JSON file with:

- `id` — stable case identifier.
- `source` — canonical raw recap path and SHA-256. The runner loads this exact
  unmodified text; offsets in expected provenance are measured against it.
- `expected` — human-reviewed deal values, questions, provenance expectations,
  and permitted alternatives where truly necessary.
- `tags` — e.g. `clean`, `amendment`, `must_abstain`, `unit_conversion`.
- `rationale` — why the case exists and the business risk it detects.

Keep source recap text out of prompts and code. Do not use evaluation cases as
few-shot examples for the model.

Suggested first case IDs:

```text
01_clean_telegram
02_forwarded_amendments
03_incomplete_must_abstain
04_missing_differential
05_ambiguous_delivered_term
06_unrelated_deal_distraction
```

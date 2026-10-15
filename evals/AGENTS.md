# Evaluation rules

The root `AGENTS.md` also applies here.

1. Expected outputs are human-reviewed ground truth. Do not edit an expectation
   merely because the current model fails it.
2. Each fixture needs a stable ID and source hash. Each case needs a companion
   rationale in `cases_explain/` describing the commercial failure it detects.
   Tags are optional analysis labels, not part of the current fixture contract.
3. Keep fixture source text identical to the supplied recap. Provenance offsets
   must be measured against that exact original text.
4. Do not use evaluation fixtures as few-shot prompt examples or tuning data.
5. Normalisation and scoring must be deterministic. Do not use an LLM as the
   reported correctness judge.
6. Score field correctness, correct abstention, wrong abstention, and
   unsupported non-null values separately. The latter is the most serious error.
7. Score question presence by field, not exact prose. Human review must still
   require questions to identify the missing information clearly enough for a
   trader to answer; do not claim the current field-path metric proves wording
   quality.
8. A new parsing rule, prompt change, or model configuration needs a relevant
   committed case before it can be claimed as an improvement.

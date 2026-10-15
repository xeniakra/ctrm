# Working rules

1. Do not invent commercial values. Missing or ambiguous mandatory data must be
   `null` with a field-specific question.
2. Preserve the original recap text. Every non-null extracted value needs an
   offset span against that unmodified text; do not calculate offsets after
   normalising or trimming the input.
3. Keep deterministic business logic outside the model. CSV lookups, unit
   conversion, price aggregation, FX arithmetic, and validation are ordinary
   code, not model reasoning.
5. Add or update an evaluation fixture before changing a prompt, model setting,
   or parsing rule. Never weaken expected output to suit one model result.
6. Record actual model version, quantisation, runtime version, seed, and
   inference parameters with every benchmark.
7. Keep agent logs and task-plan changes factual. Do not reconstruct a clean
   history after the fact.

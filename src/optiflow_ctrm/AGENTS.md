# Application-code rules

The root `AGENTS.md` also applies here.

1. Keep the canonical external JSON shape in `models.py`. Do not introduce
   endpoint-specific response dictionaries.
2. `extractor.py` may call a self-hosted model only through `runtime/`. It owns
   prompting, constrained-output requests, parsing, and at most one repair
   attempt; a second invalid result must fail safely.
3. `pricing.py` and `references.py` are deterministic. They must not call an
   LLM or choose missing commercial assumptions.
4. Preserve units as extracted. Convert BBL and MT only in calculation code and
   return the product-reference factor and formula used.
5. Validate source evidence against the original, unmodified recap string. A
   non-null extracted commercial value without evidence is invalid.
7. Keep `mcp/server.py` thin: it delegates to shared application functions and
   does not reimplement parsing, validation, or pricing.
8. Add a focused test or evaluation case before changing extraction behaviour.


# Field specification

What the extracted deal object has to cover. You design the schema — this is the coverage list and the rules, not a JSON template.

## Mandatory fields

A deal object is **not usable** if any of these is missing or invented. If the recap does not state one, the field must come back `null` with a specific question attached.

| Field | Notes |
|---|---|
| `recap_date` | the date the recap was agreed. In a forwarded thread, this is not necessarily the date of the newest message. |
| `buyer` | legal entity as written |
| `seller` | legal entity as written |
| `product` | as written, plus a link to `products.csv` where you can resolve it |
| `quantity.value` | numeric |
| `quantity.unit` | `MT` or `BBL` |
| `quantity.tolerance_pct` | e.g. 10 |
| `quantity.tolerance_option` | `seller` / `buyer` / `null` |
| `delivery_term.incoterm` | one of EXW, FAS, FOB, CFR, CIF, DAP, DDP — or `null` if what is written is not an Incoterm |
| `delivery_term.place` | named port or place |
| `delivery_window.from` / `.to` | the laycan |
| `pricing.quote_code` | the publication quotation, resolvable against `price_quotes.csv` |
| `pricing.statistic` | `mean` / `high` / `low` / `mean_of_high` |
| `pricing.period` | either a B/L-relative expression (e.g. `BL+0..BL+3`) or an explicit date range |
| `pricing.differential.value` | signed, negative for a discount |
| `pricing.differential.unit` | `USD/MT` or `USD/BBL` |
| `payment_terms` | free text is acceptable, but the structure you impose is part of what we look at |
| `currency` | ISO code |

## Conditional fields

Capture when present; absence is not an error.

| Field | Notes |
|---|---|
| `fx.rate` / `fx.basis` | required only when `currency` is not USD |
| `quality_spec` | product specification or the governing spec document |
| `inspection.appointed_by` / `.cost_split` | e.g. `50/50`, `seller` |
| `demurrage.rate_per_day` / `.currency` | |
| `law` / `arbitration` | |
| `vessel` | if nominated at recap stage |
| `notes` | anything commercially material that does not fit above |

## Rules

1. **Never infer a number.** If the recap does not state the differential, the statistic, or the pricing period, it is `null` + a question. An industry-typical default is not an answer.
3. **One recap per output.** A message may mention other deals in passing. Those are not part of this deal object.
4. **Units are not interchangeable.** If quantity is in barrels and the differential is per tonne, both are recorded as written. Conversion happens at calculation time, using `products.csv`, and the factor used is recorded.
5. **Not-an-Incoterm is not an Incoterm.** `delivered Rotterdam`, `ex tank`, `into vessel` are trader shorthand. Record what was written, set `incoterm` to `null`, and ask.
6. **Every field carries provenance.** For each extracted value, keep the span of source text it came from. Confidence scores are optional; source spans are not.
7. **Questions are structured output too.** When a mandatory field is `null`, emit a question a trader can answer in one line — naming the field and what is missing.

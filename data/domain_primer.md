# The domain in 10 minutes

Enough oil-products trading to do the task. None of it is rocket science, all of it is specific.

## The recap

A physical oil-products deal starts as a conversation. When the trader and the counterparty agree, one of them writes a **recap**: a short free-text message that fixes the commercial terms. It is not a contract — the contract is drafted from it later — but it is the moment the deal becomes real, and every downstream artefact (contract, pricing, nomination, invoice, P&L) inherits its fields.

Recaps arrive as Telegram messages, WhatsApp text, or forwarded email threads. They are written by humans in a hurry, in trader shorthand, and they get amended in the same thread.

## Quantity and tolerance

Quantity is never exact — you cannot load a vessel to the litre, and product is lost in transit. So quantity comes with a **tolerance**, typically ±10%, and the recap says whose option it is:

- `30,000 MT +/- 10% at seller's option` — the seller decides the final loaded quantity within that band.
- `in buyer's option` — the buyer decides.

Units are mixed. Crude and some products are quoted in **barrels (bbl)**, others in **metric tonnes (MT)**. Conversion is product-specific, by density: see `products.csv` for barrels per tonne. A recap can state quantity in one unit and price in the other — that is normal, and it is on you to keep them straight.

## Incoterms (2020)

The delivery term decides who pays for what, who carries the risk, and where title passes. This is the single most important field in a recap. The ones you will see:

| Term | Meaning in one line |
|---|---|
| `FOB <port>` | seller delivers on board at the load port; buyer pays freight and insurance. ~70% of our deals. |
| `CFR <port>` | seller pays freight to the discharge port; risk still passes at load. |
| `CIF <port>` | as CFR, plus the seller pays insurance. |
| `DAP <place>` | seller delivers, unloaded-ready, at a named place inland or at destination. |
| `EXW <place>` | buyer collects from the seller's premises. |
| `FAS <port>` | seller delivers alongside the vessel, not on board. |

The term is always written with a named place, and the place matters as much as the term. Traders also write things like "delivered Rotterdam", which is *not* an Incoterm and can mean DAP or CFR — that kind of phrasing is a question for the trader, not a guess.

## Pricing: a formula, not a number

Physical deals are rarely priced at a fixed figure. The price is a **formula**:

```
<publication quotation> over <pricing period>, <statistic> +/- <differential>
```

For example: `Mean of Platts FOB MED Italy Gasoil 0.1% over the 5 days after B/L, minus USD 12.50/MT`.

The pieces:

- **Publication** — Platts or Argus publish daily quotations for each product and location. Each quotation has a low and a high; `mean` is the midpoint. `data/price_quotes.csv` is a synthetic version of this.
- **Pricing period** — the days over which the quotations are taken. Often expressed relative to the **Bill of Lading** date: `B/L + 3` means the three days after the B/L date; `B/L -1/+3` means one day before to three days after. Sometimes it is a plain calendar window instead.
- **Statistic** — mean, high, low, or mean of high.
- **Differential** — a premium or discount, in USD per MT or per bbl. `minus 12.50` and `-12.50` and `at a discount of USD 12.50/MT` are the same thing.

A **provisional invoice** is issued before the pricing period closes, using an estimate; the **final invoice** is issued once the quotations are published and the formula can be evaluated. That is why the price fields have to be captured as a formula and not collapsed into a single number.

## Currency and FX

Everything is ultimately computed in USD. If a deal is denominated in EUR, the recap either names a fixed agreed FX rate or says which published rate applies on which date (usually the same B/L dating). USD/AED is pegged at 3.6725.

## Payment terms

Typical shapes: `100% prepayment before loading`, `net 30 days from B/L date against documents`, `LC at sight`, `90% provisional within 3 days of B/L, balance on final pricing`. Late payment carries a fixed interest rate, e.g. 12% p.a.

## The rest of the recap

- **Quality clauses** — the product specification, and which spec document governs.
- **Inspection** — an independent inspector (SGS, Intertek, Saybolt) measures quantity and quality at load and/or discharge. The recap says who appoints and who pays, usually "50/50" or "at seller's expense".
- **Laycan / delivery window** — the date range in which the vessel must present for loading.
- **Demurrage** — the daily rate payable if loading or discharge takes longer than the allowed laytime.
- **Law and arbitration** — e.g. "English law, LMAA arbitration, London".

## Why the abstain behaviour matters

If a system silently invents a differential of `-2.50` where the recap said "price tbc as per our call", the error does not surface in testing. It surfaces in a provisional invoice, in the trader's P&L, in a dispute with the counterparty. In this domain a confident wrong number costs more than a missing one, and "I don't know, here is what I need" is a correct answer that the system must be able to give.

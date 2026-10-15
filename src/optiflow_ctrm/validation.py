"""Deterministic normalization before canonical model validation.

This module only applies rules that are explicit in the field specification or
unambiguous syntax in the source. It never supplies a commercial default.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import re
from typing import Any

from optiflow_ctrm.models import MANDATORY_PATHS
from optiflow_ctrm.provenance import SourceSpan, SourceSpanMap, add_source_span, merge_source_spans
from optiflow_ctrm.provenance import commercial_source
from optiflow_ctrm.source_rules import explicit_rules, parse_bl


_QUESTION_TEXT = {
    "/deal/recap_date": "What date was this recap agreed?",
    "/deal/buyer": "What is the buyer's full legal entity name?",
    "/deal/seller": "What is the seller's full legal entity name?",
    "/deal/product/as_written": "What product was agreed, as written in the recap?",
    "/deal/quantity/value": "What exact contractual quantity was agreed?",
    "/deal/quantity/unit": "Is the agreed quantity stated in MT or BBL?",
    "/deal/quantity/tolerance_pct": "What quantity tolerance percentage applies?",
    "/deal/quantity/tolerance_option": (
        "Whose option applies to the quantity tolerance: buyer or seller?"
    ),
    "/deal/delivery_term/incoterm": "What agreed Incoterm applies?",
    "/deal/delivery_term/place": "What is the agreed delivery port or place?",
    "/deal/delivery_window/from": "What is the exact delivery-window start date?",
    "/deal/delivery_window/to": "What is the exact delivery-window end date?",
    "/deal/pricing/quote_code": "Which exact publication quotation is the pricing basis?",
    "/deal/pricing/statistic": "Which quotation statistic applies: mean, high, low, or mean of high?",
    "/deal/pricing/period": "What pricing period applies, including any B/L-relative dates?",
    "/deal/pricing/differential/value": "What is the signed price differential value?",
    "/deal/pricing/differential/unit": "Is the differential stated in USD/MT or USD/BBL?",
    "/deal/payment_terms": "What are the agreed payment terms?",
    "/deal/currency": "What is the deal currency?",
    "/deal/fx/rate": "What agreed FX rate applies?",
    "/deal/fx/basis": "What is the agreed FX basis?",
}


def normalize_generated_output(
    value: Any,
    source: str,
    existing_spans: SourceSpanMap | None = None,
) -> tuple[Any, SourceSpanMap]:
    """Normalize one model result without making unsupported assumptions.

    Incomplete objects are left incomplete so schema validation can trigger the
    single model repair. Complete-shaped objects get deterministic safety rules
    and canonical clarification questions.
    """

    spans = merge_source_spans(existing_spans or {})
    if not isinstance(value, dict) or not isinstance(value.get("deal"), dict):
        return value, spans
    result = deepcopy(value)
    deal = result["deal"]
    source = commercial_source(source)

    _normalize_explicit_pricing(deal, source, spans)
    _normalize_abstentions(deal, source, spans)
    _normalize_quantity_option(deal, source, spans)
    _normalize_recap_date(deal, source, spans)
    _normalize_explicit_terms(deal, source, spans)
    _normalize_payment_terms(deal, source, spans)
    _normalize_notes(deal, source, spans)
    explicit_rules(deal, source, spans)
    _rebuild_questions(result)
    return result, spans


def _normalize_explicit_pricing(deal: dict[str, Any], source: str, spans: SourceSpanMap) -> None:
    pricing = deal.get("pricing")
    if not isinstance(pricing, dict):
        return
    differential = pricing.get("differential")
    if not isinstance(differential, dict):
        return

    parsed_period = _parse_bl_period(source)
    if parsed_period:
        period, span = parsed_period
        pricing["period"] = period
        add_source_span(spans, "/deal/pricing/period", span)

    parsed_statistic = _parse_pricing_statistic(source)
    if parsed_statistic:
        statistic, span = parsed_statistic
        pricing["statistic"] = statistic
        add_source_span(spans, "/deal/pricing/statistic", span)

    _canonicalize_period_expression(pricing)


def _canonicalize_period_expression(pricing: dict[str, Any]) -> None:
    """Serialize a period from its typed components, independent of source wording."""

    period = pricing.get("period")
    if not isinstance(period, dict):
        return
    if period.get("kind") == "bl_relative":
        start = period.get("from_offset_days")
        end = period.get("to_offset_days")
        if (
            isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
        ):
            period["expression"] = f"BL{start:+d}..BL{end:+d}"
    elif period.get("kind") == "date_range":
        start = period.get("from_date", period.get("from"))
        end = period.get("to_date", period.get("to"))
        if isinstance(start, str) and isinstance(end, str):
            period["expression"] = f"{start}..{end}"


def _parse_bl_period(source: str) -> tuple[dict[str, Any], SourceSpan] | None:
    return parse_bl(source)


def _parse_pricing_statistic(source: str) -> tuple[str, SourceSpan] | None:
    patterns = (
        ("mean_of_high", r"\bmean\s+of\s+the\s+HIGH\s+quotations\b"),
        ("high", r"\bHIGH\s+(?:of\s+)?(?:Argus|Platts)\b"),
        ("low", r"\blow\s+quotation\b"),
        ("mean", r"\bmean\s+(?:of\s+)?(?:Argus|Platts)\b"),
    )
    for value, pattern in patterns:
        match = re.search(pattern, source, flags=re.IGNORECASE)
        if match:
            return value, match.span()
    return None


def _normalize_abstentions(deal: dict[str, Any], source: str, spans: SourceSpanMap) -> None:
    lower = source.casefold()

    if re.search(r"\bbuyer\s+us\b", source, flags=re.IGNORECASE):
        deal["buyer"] = None

    quantity = deal.get("quantity")
    ambiguous_quantity = re.search(
        r"\b\d+\s*-\s*\d+\s+cargoes\s+of\s+about\s+[\d,]+\s*(?:mt|bbl)\s+each\b",
        source,
        flags=re.IGNORECASE,
    )
    if ambiguous_quantity and isinstance(quantity, dict):
        quantity["value"] = None
        deal["notes"] = [ambiguous_quantity.group(0)]
        add_source_span(spans, "/deal/quantity", ambiguous_quantity.span())
        add_source_span(spans, "/deal/notes", ambiguous_quantity.span())

    window = deal.get("delivery_window")
    if "second half of" in lower and isinstance(window, dict):
        window["from"] = None
        window["to"] = None

    if re.search(r"\bpayment\s+as\s+usual\b", source, flags=re.IGNORECASE):
        deal["payment_terms"] = None

    if re.search(r"\bprice\s+tbc\b", source, flags=re.IGNORECASE):
        pricing = deal.get("pricing")
        if isinstance(pricing, dict):
            pricing["quote_code"] = None
            pricing["statistic"] = None
            pricing["period"] = None
            if isinstance(pricing.get("differential"), dict):
                pricing["differential"]["value"] = None
                pricing["differential"]["unit"] = None

    product = deal.get("product")
    quality = deal.get("quality_spec")
    if (
        isinstance(product, dict)
        and isinstance(product.get("as_written"), str)
        and isinstance(quality, str)
        and quality.casefold() in product["as_written"].casefold()
    ):
        deal["quality_spec"] = None


def _normalize_quantity_option(
    deal: dict[str, Any], source: str, spans: SourceSpanMap
) -> None:
    """Resolve closed SO/BO shorthand only inside an explicit tolerance clause."""

    matches = list(
        re.finditer(
            r"(?im)^[^\r\n]*(?:\+/-|±)[^\r\n]*?\b(SO|BO)\b",
            source,
        )
    )
    resolved = {
        "so": "seller",
        "bo": "buyer",
    }
    values = {resolved[match.group(1).casefold()] for match in matches}
    quantity = deal.get("quantity")
    if len(values) != 1 or not isinstance(quantity, dict):
        return
    quantity["tolerance_option"] = values.pop()
    for match in matches:
        add_source_span(spans, "/deal/quantity/tolerance_option", match.span(1))


def _normalize_recap_date(deal: dict[str, Any], source: str, spans: SourceSpanMap) -> None:
    parsed = _parse_recap_date(source)
    if parsed:
        value, span = parsed
        deal["recap_date"] = value
        add_source_span(spans, "/deal/recap_date", span)


_DATE_TEXT = (
    r"(?:\d{4}-\d{1,2}-\d{1,2}|"
    r"\d{1,2}[./-]\d{1,2}[./-](?:\d{4}|\d{2})|"
    r"\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\s+(?:\d{4}|\d{2}))"
)


def _parse_recap_date(source: str) -> tuple[str, SourceSpan] | None:
    contextual_patterns = (
        rf"recap\s+of\s+our\s+call\s+today\s+({_DATE_TEXT})",
        rf"(?:trade\s+|deal\s+|recap\s+|main\s+recap\s+)?agreed"
        rf"(?:\s+today|\s+on)?\s*[,.:/-]?\s*({_DATE_TEXT})",
        rf"confirmed(?:\s+on)?\s*[,.:/-]?\s*({_DATE_TEXT})",
        rf"agreement\s+dated\s*[,.:/-]?\s*({_DATE_TEXT})",
        rf"CONFIRMED\s*[-—:]\s*RECAP\s+({_DATE_TEXT})",
    )
    for pattern in contextual_patterns:
        match = re.search(pattern, source, flags=re.IGNORECASE)
        if match:
            parsed = _parse_date_text(match.group(1))
            if parsed is not None:
                return parsed, match.span(1)

    header_patterns = (
        rf"^Date:\s*(?:[A-Za-z]{{3}},\s*)?({_DATE_TEXT})",
        rf"^Sent:\s*(?:[A-Za-z]+,\s*)?({_DATE_TEXT})",
        rf"(?:Telegram|WhatsApp|transcript|recap|BLOTTER)[^\r\n]*?({_DATE_TEXT})",
        rf"DEAL\s+MEMO\s*/\s*({_DATE_TEXT})",
        rf"^#?trading-recaps\s*[—-]\s*({_DATE_TEXT})",
        rf"^\[SMS\s+export\s+({_DATE_TEXT})",
    )
    for pattern in header_patterns:
        match = re.search(pattern, source, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            parsed = _parse_date_text(match.group(1))
            if parsed is not None:
                return parsed, match.span(1)
    return None


def _parse_date_text(value: str) -> str | None:
    normalized = re.sub(r"\s+", " ", value.strip())
    for format_string in (
        "%Y-%m-%d",
        "%d.%m.%Y",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d.%m.%y",
        "%d/%m/%y",
        "%d-%m-%y",
        "%d %B %Y",
        "%d %b %Y",
        "%d %B %y",
        "%d %b %y",
    ):
        try:
            return datetime.strptime(normalized, format_string).date().isoformat()
        except ValueError:
            continue
    return None


def _normalize_explicit_terms(deal: dict[str, Any], source: str, spans: SourceSpanMap) -> None:
    parsed_currency = _parse_currency(source)
    if parsed_currency:
        currency, span = parsed_currency
        deal["currency"] = currency
        add_source_span(spans, "/deal/currency", span)

    fx = deal.get("fx")
    fx_match = re.search(r"fixed\s+agreed\s+rate", source, flags=re.IGNORECASE)
    if isinstance(fx, dict) and fx_match:
        fx["basis"] = "fixed_agreed"
        add_source_span(spans, "/deal/fx/basis", fx_match.span())

    inspection = deal.get("inspection")
    if (
        isinstance(inspection, dict)
        and isinstance(inspection.get("location"), str)
        and inspection["location"].casefold() == "at discharge"
    ):
        inspection["location"] = "discharge"
        location_match = re.search(r"at\s+discharge", source, flags=re.IGNORECASE)
        if location_match:
            add_source_span(spans, "/deal/inspection/location", location_match.span())


_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "fifteen": 15,
    "thirty": 30,
    "ninety": 90,
}
_PAYMENT_NUMBER = r"(?:\d+(?:\.\d+)?|" + "|".join(_NUMBER_WORDS) + r")"


def _normalize_payment_terms(
    deal: dict[str, Any], source: str, spans: SourceSpanMap
) -> None:
    """Parse explicit payment wording and retain every physical source fragment."""

    current = deal.get("payment_terms")
    if not isinstance(current, dict) or not isinstance(current.get("as_written"), str):
        return
    located = _locate_normalized_source_text(source, current["as_written"])
    if located is None:
        deal["payment_terms"] = None
        return
    as_written, source_fragments = located
    parsed = _parse_payment_terms(as_written)
    if parsed is None:
        deal["payment_terms"] = None
        return
    deal["payment_terms"] = parsed
    for span in source_fragments:
        add_source_span(spans, "/deal/payment_terms", span)


def _locate_normalized_source_text(
    source: str, expected: str
) -> tuple[str, list[SourceSpan]] | None:
    """Locate text joined across quoted lines without changing source coordinates."""

    characters: list[str] = []
    origins: list[int | None] = []
    offset = 0
    for line in source.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        prefix = re.match(r"^\s*(?:>\s*)+", content)
        content_start = prefix.end() if prefix else 0
        if characters:
            characters.append(" ")
            origins.append(None)
        for index, character in enumerate(content[content_start:], start=content_start):
            characters.append(character)
            origins.append(offset + index)
        offset += len(line)

    searchable = "".join(characters)
    folded = searchable.casefold()
    needle = expected.casefold()
    starts = []
    cursor = 0
    while True:
        start = folded.find(needle, cursor)
        if start < 0:
            break
        starts.append(start)
        cursor = start + 1
    if len(starts) != 1:
        return None

    start = starts[0]
    end = start + len(expected)
    selected = origins[start:end]
    spans: list[SourceSpan] = []
    span_start: int | None = None
    previous: int | None = None
    for origin in selected + [None]:
        if origin is not None and (previous is None or origin == previous + 1):
            if span_start is None:
                span_start = origin
        else:
            if span_start is not None and previous is not None:
                spans.append((span_start, previous + 1))
            span_start = origin
        previous = origin
    return searchable[start:end], spans


def _parse_payment_terms(as_written: str) -> dict[str, Any] | None:
    clauses = re.split(
        r",\s*(?=(?:the\s+)?balance\b)|\s+and\s+(?=(?:the\s+)?balance\b)",
        as_written,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    payments = [_parse_payment_clause(clause) for clause in clauses if clause.strip()]
    if not payments:
        return None
    interest = re.search(
        rf"\blate\s+interest\s+({_PAYMENT_NUMBER})\s*%\s*p\.a\.",
        as_written,
        flags=re.IGNORECASE,
    )
    late_interest = None
    if interest:
        late_interest = {
            "rate_pct": _payment_number(interest.group(1)),
            "basis": "per_annum",
        }
    return {
        "as_written": as_written,
        "payments": payments,
        "late_interest": late_interest,
    }


def _parse_payment_clause(clause: str) -> dict[str, Any]:
    lower = clause.casefold()
    percentage_match = re.search(
        rf"\b({_PAYMENT_NUMBER})\s*(?:%|pct\b|percent\b)",
        clause,
        flags=re.IGNORECASE,
    )
    if "balance" in lower:
        portion = {"kind": "balance", "percentage": None}
    elif percentage_match:
        portion = {
            "kind": "percentage",
            "percentage": _payment_number(percentage_match.group(1)),
        }
    elif re.search(r"\bfull\b", clause, flags=re.IGNORECASE):
        portion = {"kind": "full", "percentage": None}
    else:
        portion = {"kind": None, "percentage": None}

    if re.search(r"\bL\s*/?\s*C\b", clause, flags=re.IGNORECASE):
        instrument = "letter_of_credit"
    elif re.search(r"\b(?:prepayment|prepaid)\b", clause, flags=re.IGNORECASE):
        instrument = "prepayment"
    elif re.search(r"\bcash\b", clause, flags=re.IGNORECASE):
        instrument = "cash"
    else:
        instrument = None

    due = _parse_payment_due(clause)
    return {
        "portion": portion,
        "stage": (
            "provisional"
            if "provisional" in lower
            else "final" if "balance" in lower else None
        ),
        "instrument": instrument,
        "irrevocable": True if "irrevocable" in lower else None,
        "due": due,
        "document_condition": _parse_document_condition(clause),
    }


def _parse_payment_due(clause: str) -> dict[str, Any] | None:
    lower = clause.casefold()
    if "at sight" in lower:
        return {
            "qualifier": None,
            "days": None,
            "day_basis": None,
            "relation": "at",
            "event": "sight",
        }

    days_match = re.search(
        rf"\b({_PAYMENT_NUMBER})\s+(?:(banking|business)\s+)?days?\b",
        clause,
        flags=re.IGNORECASE,
    )
    if days_match:
        days = int(_payment_number(days_match.group(1)))
        explicit_basis = days_match.group(2)
        day_basis = f"{explicit_basis.casefold()}_days" if explicit_basis else "days"
    else:
        net_match = re.search(rf"\bnet\s+({_PAYMENT_NUMBER})\b", clause, re.IGNORECASE)
        days = int(_payment_number(net_match.group(1))) if net_match else None
        day_basis = "days" if net_match else None

    event = None
    if re.search(r"\bB\s*/?\s*L\b", clause, flags=re.IGNORECASE):
        event = "bill_of_lading"
    elif "final pricing" in lower or ("balance" in lower and "after final" in lower):
        event = "final_pricing"
    elif "each lifting" in lower:
        event = "each_lifting"
    elif "lifting" in lower:
        event = "lifting"
    elif "loading" in lower:
        event = "loading"

    qualifier = (
        "no_later_than"
        if "no later than" in lower
        else "within" if "within" in lower else "net" if re.search(r"\bnet\b", lower) else None
    )
    relation = next(
        (item for item in ("from", "after", "before") if re.search(rf"\b{item}\b", lower)),
        None,
    )
    if relation is None and event == "bill_of_lading" and re.search(
        r"\bof\s+B\s*/?\s*L\b", clause, flags=re.IGNORECASE
    ):
        relation = "of"
    if days is None and event is None and relation is None and qualifier is None:
        return None
    return {
        "qualifier": qualifier,
        "days": days,
        "day_basis": day_basis,
        "relation": relation,
        "event": event,
    }


def _parse_document_condition(clause: str) -> dict[str, str] | None:
    match = re.search(
        r"\bagainst\s+(scanned\s+documents|original\s+documents|originals|"
        r"shipping\s+documents|documents|docs)\b",
        clause,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    document = re.sub(r"\s+", "_", match.group(1).casefold())
    document_type = {
        "docs": "documents",
        "originals": "original_documents",
    }.get(document, document)
    return {"relation": "against", "document_type": document_type}


def _payment_number(value: str) -> float:
    lowered = value.casefold()
    return float(_NUMBER_WORDS[lowered]) if lowered in _NUMBER_WORDS else float(value)


def _parse_currency(source: str) -> tuple[str, SourceSpan] | None:
    patterns = (
        r"\b(?:contract\s+|deal\s+)?currency(?:\s+is)?\s*[:—-]?\s*(USD|EUR)\b",
        r"\bccy\s*[:—-]?\s*(USD|EUR)\b",
        r"(?m)^\s*(USD|EUR)(?=\s*(?:[.;]|at\s+fixed))",
        r"(?m)^\s*(EUR)(?=\s+at\s+fixed)",
        r"(?<=[.;])\s*(USD|EUR)(?=\s*[.;])",
    )
    for pattern in patterns:
        match = re.search(pattern, source, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper(), match.span(1)
    dollar = re.search(r"\bDollar\s+deal\b", source, flags=re.IGNORECASE)
    if dollar:
        return "USD", dollar.span()
    return None


def _normalize_notes(deal: dict[str, Any], source: str, spans: SourceSpanMap) -> None:
    notes = deal.get("notes")
    if not isinstance(notes, list):
        return
    normalized = [
        note
        for note in notes
        if isinstance(note, str)
        and "pls confirm" not in note.casefold()
        and "paperwork" not in note.casefold()
        and "price tbc" not in note.casefold()
        and "inspector is" not in note.casefold()
    ]
    frame_agreement = re.search(r"Rest as per[^\r\n]+", source, flags=re.IGNORECASE)
    if frame_agreement and not any(
        note.casefold() == frame_agreement.group(0).casefold() for note in normalized
    ):
        normalized.append(frame_agreement.group(0))
        add_source_span(spans, "/deal/notes", frame_agreement.span())
    deal["notes"] = normalized


def _rebuild_questions(result: dict[str, Any]) -> None:
    deal = result["deal"]
    missing = [pointer for pointer, path in MANDATORY_PATHS if _dict_path(deal, path) is None]
    currency = deal.get("currency")
    fx = deal.get("fx")
    if currency not in {None, "USD"}:
        if not isinstance(fx, dict) or fx.get("rate") is None:
            missing.append("/deal/fx/rate")
        if not isinstance(fx, dict) or fx.get("basis") is None:
            missing.append("/deal/fx/basis")
    result["questions"] = [
        {"field": pointer, "question": _QUESTION_TEXT[pointer]} for pointer in missing
    ]


def _dict_path(value: Any, path: tuple[str, ...]) -> Any:
    for part in path:
        key = "from" if part == "from_date" else "to" if part == "to_date" else part
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value

"""Deterministic calculator actions and MCP result decoding for the agent."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import json
from pathlib import Path
import re
from typing import Any, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
BL_DATE_QUESTION = {
    "field": "/valuation/bl_date",
    "question": "What is the actual B/L date (YYYY-MM-DD) for this deal?",
}
_MONEY_QUANTUM = Decimal("0.01")
_PRICE_UNIT = re.compile(r"^(?P<currency>[A-Z]{3})/(?P<unit>MT|BBL)$")
class McpToolCaller(Protocol):
    """The part of ``mcp.ClientSession`` used by this orchestration."""

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


class ValuationError(ValueError):
    """Raised when a deal cannot be valued without inventing an input."""


class McpToolCallError(RuntimeError):
    """Raised when an MCP tool reports an error or malformed result."""


def resolve_pricing_period(
    deal: Mapping[str, Any], bl_date: date | str | None = None
) -> dict[str, Any]:
    """Resolve extracted calendar offsets without asking the model to add dates."""
    pricing = _object(_required(deal.get("pricing"), "pricing"), "pricing")
    period = _object(_required(pricing.get("period"), "pricing period"), "pricing period")
    actual_bl_date, start, end = _resolve_period(period, bl_date)
    return {
        "bl_date": actual_bl_date.isoformat() if actual_bl_date else None,
        "date_or_period": f"{start.isoformat()}..{end.isoformat()}",
    }


def calculate_provisional_value(
    deal: Mapping[str, Any],
    quote: Mapping[str, Any],
    bl_date: date | str | None = None,
    *,
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> dict[str, Any]:
    """Calculate from retrieved data, without calling tools or choosing a sequence.

    ``provisional_value`` is the gross cargo value based on the provisional price.
    If the payment terms state a numeric provisional percentage, the separately
    labelled ``provisional_payment`` is also returned.
    """

    pricing = _object(_required(deal.get("pricing"), "pricing"), "pricing")
    period = _object(
        _required(pricing.get("period"), "pricing period"), "pricing period"
    )
    actual_bl_date, period_start, period_end = _resolve_period(period, bl_date)
    quote_code = _text(pricing.get("quote_code"), "pricing quote code")
    statistic = _text(pricing.get("statistic"), "pricing statistic")
    returned_quote_code = _text(quote.get("quote_code"), "returned quote code")
    returned_statistic = _text(quote.get("statistic"), "returned statistic")
    if returned_quote_code.casefold() != quote_code.casefold():
        raise McpToolCallError("get_price_quote returned a different quote code")
    if returned_statistic != statistic:
        raise McpToolCallError("get_price_quote returned a different statistic")
    returned_period = _object(
        _required(quote.get("period"), "returned pricing period"),
        "returned pricing period",
    )
    returned_start = _parse_wire_date(returned_period.get("from"), "returned period start")
    returned_end = _parse_wire_date(returned_period.get("to"), "returned period end")
    if returned_start != period_start or returned_end != period_end:
        raise McpToolCallError("get_price_quote returned a different pricing period")

    quote_unit = _text(quote.get("unit"), "quotation unit")
    quote_currency, price_unit = _split_price_unit(quote_unit, "quotation unit")
    if quote_currency != "USD":
        raise ValuationError(f"unsupported quotation currency: {quote_currency}")

    quantity = _object(_required(deal.get("quantity"), "quantity"), "quantity")
    product = _object(_required(deal.get("product"), "product"), "product")
    quantity_value = _decimal(quantity.get("value"), "quantity value")
    quantity_unit = _text(quantity.get("unit"), "quantity unit")
    differential = _object(
        _required(pricing.get("differential"), "price differential"),
        "price differential",
    )
    differential_value = _decimal(
        differential.get("value"), "price differential"
    )
    differential_unit = _text(differential.get("unit"), "price differential unit")
    differential_currency, differential_denominator = _split_price_unit(
        differential_unit, "price differential unit"
    )
    if differential_currency != quote_currency:
        raise ValuationError(
            "quotation and differential currencies differ; no conversion was agreed"
        )
    conversion_required = (
        quantity_unit != price_unit or differential_denominator != price_unit
    )
    bbl_per_mt = (
        _load_bbl_per_mt(
            _text(product.get("product_code"), "product code"), Path(data_dir)
        )
        if conversion_required
        else None
    )
    pricing_quantity, conversion = _convert_quantity(
        quantity_value, quantity_unit, price_unit, bbl_per_mt
    )
    converted_differential = _convert_per_unit(
        differential_value,
        differential_denominator,
        price_unit,
        bbl_per_mt,
    )

    quotation = _decimal(quote.get("value"), "quotation value")
    unit_price = quotation + converted_differential
    gross_usd = pricing_quantity * unit_price
    deal_currency = _text(deal.get("currency"), "deal currency")
    fx_value = deal.get("fx")
    fx = None if fx_value is None else _object(fx_value, "FX terms")
    deal_value, fx_record = _convert_from_usd(
        gross_usd,
        deal_currency,
        fx,
    )
    rounded_usd = _money(gross_usd)
    rounded_deal_value = _money(deal_value)
    payment_terms_value = deal.get("payment_terms")
    payment_terms = (
        None
        if payment_terms_value is None
        else _object(payment_terms_value, "payment terms")
    )
    provisional_percentage = _read_provisional_percentage(payment_terms)

    result: dict[str, Any] = {
        "status": "completed",
        "bl_date": actual_bl_date.isoformat() if actual_bl_date else None,
        "pricing_period": {
            "from": period_start.isoformat(),
            "to": period_end.isoformat(),
        },
        "price_quote": dict(quote),
        "calculation": {
            "quantity_as_written": {
                "value": float(quantity_value),
                "unit": quantity_unit,
            },
            "pricing_quantity": {
                "value": float(pricing_quantity),
                "unit": price_unit,
            },
            "conversion": conversion,
            "quotation": {"value": float(quotation), "unit": quote_unit},
            "differential_as_written": {
                "value": float(differential_value),
                "unit": differential_unit,
            },
            "differential_in_quote_unit": {
                "value": float(converted_differential),
                "unit": quote_unit,
            },
            "provisional_unit_price": {
                "value": float(unit_price),
                "unit": quote_unit,
            },
            "gross_value_usd": float(rounded_usd),
            "fx": fx_record,
            "formula": _valuation_formula(
                quantity_value,
                quantity_unit,
                pricing_quantity,
                price_unit,
                quotation,
                converted_differential,
                gross_usd,
                deal_currency,
                fx_record,
            ),
        },
        "provisional_value": {
            "amount": float(rounded_deal_value),
            "currency": deal_currency,
        },
        "quantity_basis": (
            "nominal quantity; recap tolerance is +/- "
            f"{_decimal(quantity.get('tolerance_pct'), 'quantity tolerance'):g}% at "
            f"{_text(quantity.get('tolerance_option'), 'tolerance option')}'s option"
        ),
    }
    if provisional_percentage is not None:
        payment = _money(deal_value * provisional_percentage / Decimal("100"))
        result["provisional_payment"] = {
            "percentage": float(provisional_percentage),
            "amount": float(payment),
            "currency": deal_currency,
        }
    else:
        result["provisional_payment"] = None
    return result


async def call_mcp_tool(
    client: McpToolCaller, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    result = await client.call_tool(name, arguments)
    if isinstance(result, Mapping):
        return dict(result)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Mapping):
        return dict(result[1])

    if bool(getattr(result, "isError", False)):
        raise McpToolCallError(f"{name} failed: {_content_text(result)}")
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, Mapping):
        value = dict(structured)
        if set(value) == {"result"} and isinstance(value["result"], Mapping):
            return dict(value["result"])
        return value

    raw_text = _content_text(result)
    try:
        value = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError) as error:
        raise McpToolCallError(f"{name} returned no JSON object") from error
    if not isinstance(value, Mapping):
        raise McpToolCallError(f"{name} returned JSON that was not an object")
    return dict(value)


def _content_text(result: Any) -> str:
    content = getattr(result, "content", ())
    values = [item.text for item in content if isinstance(getattr(item, "text", None), str)]
    return "\n".join(values)


def _resolve_period(
    period: Mapping[str, Any], bl_date: date | str | None
) -> tuple[date | None, date, date]:
    kind = _text(period.get("kind"), "pricing period kind")
    if kind == "date_range":
        return (
            None,
            _parse_wire_date(period.get("from"), "pricing period start"),
            _parse_wire_date(period.get("to"), "pricing period end"),
        )
    if kind != "bl_relative":
        raise ValuationError(f"unsupported pricing period kind: {kind!r}")
    actual_bl_date = _parse_iso_date(bl_date)
    from_offset = _integer(period.get("from_offset_days"), "pricing start offset")
    to_offset = _integer(period.get("to_offset_days"), "pricing end offset")
    if from_offset > to_offset:
        raise ValuationError("pricing-period start offset must not be after its end offset")
    return (
        actual_bl_date,
        actual_bl_date + timedelta(days=from_offset),
        actual_bl_date + timedelta(days=to_offset),
    )


def _parse_iso_date(value: date | str | None) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise ValuationError("B/L date must be an ISO date (YYYY-MM-DD)")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValuationError(f"invalid B/L date: {value!r}") from error


def _parse_wire_date(value: Any, label: str) -> date:
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise McpToolCallError(f"{label} must be an ISO date (YYYY-MM-DD)")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise McpToolCallError(f"invalid {label}: {value!r}") from error


def _load_bbl_per_mt(product_code: str, data_dir: Path) -> Decimal:
    path = data_dir / "products.csv"
    try:
        stream = path.open(newline="", encoding="utf-8")
    except OSError as error:
        raise ValuationError(f"could not read product reference data at {path}") from error

    with stream:
        reader = csv.DictReader(stream)
        required = {"product_code", "bbl_per_mt"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValuationError(f"product reference data is missing columns: {sorted(missing)}")
        matches = [
            row
            for row in reader
            if row["product_code"].strip().casefold() == product_code.casefold()
        ]
    if len(matches) != 1:
        detail = "unknown" if not matches else "duplicate"
        raise ValuationError(f"{detail} product code in products.csv: {product_code!r}")
    factor = _decimal(matches[0]["bbl_per_mt"], "bbl_per_mt")
    if factor <= 0:
        raise ValuationError("bbl_per_mt must be positive")
    return factor


def _convert_quantity(
    value: Decimal,
    source_unit: str,
    target_unit: str,
    bbl_per_mt: Decimal | None,
) -> tuple[Decimal, dict[str, Any]]:
    if source_unit == target_unit:
        return value, {
            "from_unit": source_unit,
            "to_unit": target_unit,
            "bbl_per_mt": None,
            "formula": "quantity unchanged because quantity and quotation units match",
        }
    if bbl_per_mt is None:
        raise ValuationError("bbl_per_mt is required for quantity conversion")
    if source_unit == "BBL" and target_unit == "MT":
        converted = value / bbl_per_mt
        formula = f"{value} BBL / {bbl_per_mt} BBL/MT"
    elif source_unit == "MT" and target_unit == "BBL":
        converted = value * bbl_per_mt
        formula = f"{value} MT * {bbl_per_mt} BBL/MT"
    else:
        raise ValuationError(f"cannot convert quantity from {source_unit} to {target_unit}")
    return converted, {
        "from_unit": source_unit,
        "to_unit": target_unit,
        "bbl_per_mt": float(bbl_per_mt),
        "formula": formula,
    }


def _convert_per_unit(
    value: Decimal,
    source_unit: str,
    target_unit: str,
    bbl_per_mt: Decimal | None,
) -> Decimal:
    if source_unit == target_unit:
        return value
    if bbl_per_mt is None:
        raise ValuationError("bbl_per_mt is required for differential conversion")
    if source_unit == "BBL" and target_unit == "MT":
        return value * bbl_per_mt
    if source_unit == "MT" and target_unit == "BBL":
        return value / bbl_per_mt
    raise ValuationError(f"cannot convert differential from {source_unit} to {target_unit}")


def _convert_from_usd(
    gross_usd: Decimal,
    currency: str,
    fx: Mapping[str, Any] | None,
) -> tuple[Decimal, dict[str, Any]]:
    if currency == "USD":
        return gross_usd, {
            "rate": None,
            "quote": None,
            "basis": "not_required",
            "operation": "none",
        }
    if fx is None:
        raise ValuationError(f"FX terms are required for a {currency} deal")
    rate = _decimal(fx.get("rate"), "FX rate")
    quote = _text(fx.get("quote"), "FX quote").upper().replace(" ", "")
    basis = _text(fx.get("basis"), "FX basis")
    if quote == f"USD/{currency}":
        converted = gross_usd / rate
        operation = "divide USD value by rate"
    elif quote == f"{currency}/USD":
        converted = gross_usd * rate
        operation = "multiply USD value by rate"
    else:
        raise ValuationError(
            f"FX quote {quote!r} does not convert USD to deal currency {currency}"
        )
    return converted, {
        "rate": float(rate),
        "quote": quote,
        "basis": basis,
        "operation": operation,
    }


def _split_price_unit(value: str, label: str) -> tuple[str, str]:
    match = _PRICE_UNIT.fullmatch(value.strip().upper())
    if match is None:
        raise ValuationError(f"{label} must be CURRENCY/MT or CURRENCY/BBL")
    return match.group("currency"), match.group("unit")


def _read_provisional_percentage(payment_terms: Mapping[str, Any] | None) -> Decimal | None:
    if payment_terms is None:
        return None
    payments = payment_terms.get("payments")
    if not isinstance(payments, list):
        raise McpToolCallError("payment terms have no payments list")
    for item in payments:
        payment = _object(item, "payment")
        if payment.get("stage") != "provisional":
            continue
        portion = _object(payment.get("portion"), "payment portion")
        if portion.get("kind") != "percentage":
            continue
        value = _decimal(portion.get("percentage"), "provisional payment percentage")
        if value <= 0 or value > 100:
            raise ValuationError(
                "provisional payment percentage must be above 0 and at most 100"
            )
        return value
    return None


def _valuation_formula(
    quantity: Decimal,
    quantity_unit: str,
    pricing_quantity: Decimal,
    price_unit: str,
    quotation: Decimal,
    differential: Decimal,
    gross_usd: Decimal,
    currency: str,
    fx_record: Mapping[str, Any],
) -> str:
    core = (
        f"{quantity} {quantity_unit} -> {pricing_quantity} {price_unit}; "
        f"{pricing_quantity} * ({quotation} + {differential}) = {gross_usd} USD"
    )
    if currency == "USD":
        return core
    return f"{core}; {fx_record['operation']} {fx_record['rate']} = value in {currency}"


def _decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool):
        raise ValuationError(f"{label} must be numeric")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValuationError(f"{label} must be numeric") from error
    if not number.is_finite():
        raise ValuationError(f"{label} must be finite")
    return number


def _money(value: Decimal) -> Decimal:
    return value.quantize(_MONEY_QUANTUM, rounding=ROUND_HALF_UP)


def _required(value: Any, label: str) -> Any:
    if value is None:
        raise ValuationError(f"{label} is required for valuation")
    return value


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise McpToolCallError(f"{label} must be a JSON object")
    return dict(value)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValuationError(f"{label} must be a non-empty string")
    return value.strip()


def _integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValuationError(f"{label} must be an integer")
    return value

"""Deterministic pricing-period, unit-conversion, FX, and valuation functions.

No LLM call belongs in this module. A B/L-relative period remains unresolved
until the caller supplies an actual B/L date or a documented assumption.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
from typing import Any

from optiflow_ctrm.models import PriceQuoteResult
from optiflow_ctrm.references import DATA_DIR


_REQUIRED_QUOTE_COLUMNS = {
    "date",
    "publication",
    "quote_code",
    "quote_name",
    "unit",
    "low",
    "high",
    "mean",
}
_STATISTIC_COLUMNS = {
    "mean": "mean",
    "high": "high",
    "low": "low",
    "mean_of_high": "high",
}
_ISO_DATE = r"\d{4}-\d{2}-\d{2}"
_DATE_OR_PERIOD = re.compile(
    rf"^(?P<start>{_ISO_DATE})(?:\s*(?:\.\.|/|\bto\b)\s*(?P<end>{_ISO_DATE}))?$",
    flags=re.IGNORECASE,
)


class PriceQuoteError(ValueError):
    """Raised when a quotation request cannot be resolved safely."""


def get_price_quote(
    quote_code: str,
    date_or_period: date | str | Sequence[date | str] | Mapping[str, date | str],
    statistic: str,
    *,
    data_dir: Path | str = DATA_DIR,
) -> dict[str, Any]:
    """Return one quotation or aggregate quotations over an inclusive period.

    ``date_or_period`` accepts an ISO date, an inclusive ISO range such as
    ``2026-08-24..2026-08-28``, a two-item date sequence, or a mapping with
    ``from``/``to`` (or ``from_date``/``to_date``) keys. Calendar dates without
    a published row, such as weekends, are skipped. At least one published
    observation must remain.

    The quotation statistic is a daily series. Period requests return the
    arithmetic mean of that series: ``mean`` selects the CSV's daily midpoint,
    ``low`` and ``high`` select the respective daily side, and
    ``mean_of_high`` explicitly averages the daily high side. No rounding is
    applied beyond conversion of the exact decimal calculation to JSON-safe
    floats.

    B/L-relative expressions are deliberately rejected here because this
    function has no actual B/L date with which to resolve them.
    """

    canonical_code = _validate_quote_code(quote_code)
    statistic_key = _validate_statistic(statistic)
    start_date, end_date = _parse_date_or_period(date_or_period)

    quote_path = Path(data_dir) / "price_quotes.csv"
    rows = _load_quote_rows(quote_path, canonical_code)
    selected_rows = [row for row in rows if start_date <= row["date"] <= end_date]
    if not selected_rows:
        raise PriceQuoteError(
            "no published quotations found for "
            f"{rows[0]['quote_code']} from {start_date.isoformat()} through "
            f"{end_date.isoformat()}"
        )

    source_column = _STATISTIC_COLUMNS[statistic_key]
    selected_values = [row[source_column] for row in selected_rows]
    aggregate = sum(selected_values, start=Decimal("0")) / len(selected_values)
    metadata = selected_rows[0]

    observations = [
        {
            "date": row["date"].isoformat(),
            "low": float(row["low"]),
            "high": float(row["high"]),
            "mean": float(row["mean"]),
            "selected_value": float(row[source_column]),
        }
        for row in selected_rows
    ]
    count = len(observations)
    result = PriceQuoteResult.model_validate(
        {
            "quote_code": metadata["quote_code"],
            "quote_name": metadata["quote_name"],
            "publication": metadata["publication"],
            "unit": metadata["unit"],
            "statistic": statistic_key,
            "period": {"from": start_date, "to": end_date},
            "observations": observations,
            "observation_count": count,
            "aggregation": "arithmetic_mean",
            "source_column": source_column,
            "value": float(aggregate),
            "formula": f"sum({count} published daily {source_column} values) / {count}",
        }
    )
    return result.model_dump(mode="json", by_alias=True)


def _validate_quote_code(quote_code: str) -> str:
    if not isinstance(quote_code, str) or not quote_code.strip():
        raise PriceQuoteError("quote_code must be a non-empty string")
    return quote_code.strip()


def _validate_statistic(statistic: str) -> str:
    if not isinstance(statistic, str):
        raise PriceQuoteError(
            "unsupported statistic; expected one of: high, low, mean, mean_of_high"
        )
    value = statistic.strip().casefold()
    if value not in _STATISTIC_COLUMNS:
        raise PriceQuoteError(
            f"unsupported statistic {statistic!r}; expected one of: high, low, mean, mean_of_high"
        )
    return value


def _parse_date_or_period(
    value: date | str | Sequence[date | str] | Mapping[str, date | str],
) -> tuple[date, date]:
    if isinstance(value, datetime):
        start = end = value.date()
    elif isinstance(value, date):
        start = end = value
    elif isinstance(value, str):
        stripped = value.strip()
        if re.search(r"\bB\s*/?\s*L\b", stripped, flags=re.IGNORECASE):
            raise PriceQuoteError(
                "B/L-relative pricing period requires an actual B/L date before quote lookup"
            )
        match = _DATE_OR_PERIOD.fullmatch(stripped)
        if match is None:
            raise PriceQuoteError(
                "date_or_period must be an ISO date or inclusive ISO date range "
                "(YYYY-MM-DD..YYYY-MM-DD)"
            )
        start = _parse_iso_date(match.group("start"), "start")
        end = _parse_iso_date(match.group("end") or match.group("start"), "end")
    elif isinstance(value, Mapping):
        kind = value.get("kind")
        if kind == "bl_relative":
            raise PriceQuoteError(
                "B/L-relative pricing period requires an actual B/L date before quote lookup"
            )
        start_value = value.get("from", value.get("from_date"))
        end_value = value.get("to", value.get("to_date"))
        if start_value is None or end_value is None:
            raise PriceQuoteError(
                "date_or_period mapping must contain from/to or from_date/to_date"
            )
        start = _parse_iso_date(start_value, "start")
        end = _parse_iso_date(end_value, "end")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        if len(value) != 2:
            raise PriceQuoteError("date_or_period sequence must contain exactly two dates")
        start = _parse_iso_date(value[0], "start")
        end = _parse_iso_date(value[1], "end")
    else:
        raise PriceQuoteError(
            "date_or_period must be an ISO date, inclusive ISO range, or two dates"
        )

    if start > end:
        raise PriceQuoteError("pricing-period start date must not be after its end date")
    return start, end


def _parse_iso_date(value: date | str, label: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or re.fullmatch(_ISO_DATE, value) is None:
        raise PriceQuoteError(f"pricing-period {label} must be an ISO date (YYYY-MM-DD)")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise PriceQuoteError(f"invalid pricing-period {label} date: {value!r}") from error


def _load_quote_rows(path: Path, requested_code: str) -> list[dict[str, Any]]:
    try:
        stream = path.open(newline="", encoding="utf-8")
    except OSError as error:
        raise PriceQuoteError(f"could not read quotation data at {path}") from error

    with stream:
        reader = csv.DictReader(stream)
        missing_columns = _REQUIRED_QUOTE_COLUMNS - set(reader.fieldnames or ())
        if missing_columns:
            raise PriceQuoteError(
                f"quotation data is missing required columns: {sorted(missing_columns)}"
            )

        matches: list[dict[str, Any]] = []
        seen_dates: set[date] = set()
        metadata: tuple[str, str, str, str] | None = None
        for line_number, row in enumerate(reader, start=2):
            if row["quote_code"].casefold() != requested_code.casefold():
                continue
            row_date = _parse_csv_date(row["date"], path, line_number)
            if row_date in seen_dates:
                raise PriceQuoteError(
                    f"duplicate quotation for {row['quote_code']} on {row_date.isoformat()}"
                )
            seen_dates.add(row_date)

            row_metadata = (
                row["quote_code"],
                row["quote_name"],
                row["publication"],
                row["unit"],
            )
            if metadata is None:
                metadata = row_metadata
            elif metadata != row_metadata:
                raise PriceQuoteError(
                    f"inconsistent quotation metadata for quote_code {row['quote_code']}"
                )

            low = _parse_csv_decimal(row["low"], "low", path, line_number)
            high = _parse_csv_decimal(row["high"], "high", path, line_number)
            mean = _parse_csv_decimal(row["mean"], "mean", path, line_number)
            if low > mean or mean > high:
                raise PriceQuoteError(
                    f"invalid low/mean/high ordering in {path} at line {line_number}"
                )
            matches.append(
                {
                    "date": row_date,
                    "publication": row["publication"],
                    "quote_code": row["quote_code"],
                    "quote_name": row["quote_name"],
                    "unit": row["unit"],
                    "low": low,
                    "high": high,
                    "mean": mean,
                }
            )

    if not matches:
        raise PriceQuoteError(f"unknown quote_code: {requested_code!r}")
    return sorted(matches, key=lambda row: row["date"])


def _parse_csv_date(value: str, path: Path, line_number: int) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise PriceQuoteError(
            f"invalid quotation date in {path} at line {line_number}: {value!r}"
        ) from error


def _parse_csv_decimal(value: str, column: str, path: Path, line_number: int) -> Decimal:
    try:
        number = Decimal(value)
    except (InvalidOperation, TypeError) as error:
        raise PriceQuoteError(
            f"invalid {column} value in {path} at line {line_number}: {value!r}"
        ) from error
    if not number.is_finite():
        raise PriceQuoteError(
            f"non-finite {column} value in {path} at line {line_number}: {value!r}"
        )
    return number

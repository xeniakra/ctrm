from datetime import date
from pathlib import Path

import pytest

from optiflow_ctrm.models import PriceQuoteResult
from optiflow_ctrm.pricing import PriceQuoteError, get_price_quote


def test_get_price_quote_aggregates_daily_highs_over_inclusive_period() -> None:
    result = get_price_quote(
        "ARGUS_CIF_NWE_NAPHTHA",
        "2026-08-24..2026-08-28",
        "mean_of_high",
    )

    assert result["quote_code"] == "ARGUS_CIF_NWE_NAPHTHA"
    assert result["unit"] == "USD/MT"
    assert result["period"] == {"from": "2026-08-24", "to": "2026-08-28"}
    assert result["observation_count"] == 5
    assert result["value"] == pytest.approx(588.7)
    assert [observation["date"] for observation in result["observations"]] == [
        "2026-08-24",
        "2026-08-25",
        "2026-08-26",
        "2026-08-27",
        "2026-08-28",
    ]
    assert result["observations"][0]["selected_value"] == 583.86
    assert PriceQuoteResult.model_validate(result).observation_count == 5


def test_get_price_quote_accepts_a_single_date_object() -> None:
    result = get_price_quote(
        "ARGUS_CIF_NWE_NAPHTHA",
        date(2026, 9, 4),
        "high",
    )

    assert result["period"] == {"from": "2026-09-04", "to": "2026-09-04"}
    assert result["observation_count"] == 1
    assert result["value"] == 594.5


@pytest.mark.parametrize(
    ("date_or_period", "message"),
    [
        ("BL-1..BL+3", "B/L-relative"),
        ("2026-08-28..2026-08-24", "start date"),
        ("2026-08-29..2026-08-30", "no published quotations"),
    ],
)
def test_get_price_quote_rejects_unresolvable_periods(date_or_period: str, message: str) -> None:
    with pytest.raises(PriceQuoteError, match=message):
        get_price_quote("ARGUS_CIF_NWE_NAPHTHA", date_or_period, "mean")


def test_get_price_quote_rejects_unknown_quote_and_statistic() -> None:
    with pytest.raises(PriceQuoteError, match="unknown quote_code"):
        get_price_quote("NOT_A_QUOTE", "2026-08-24", "mean")

    with pytest.raises(PriceQuoteError, match="unsupported statistic"):
        get_price_quote("ARGUS_CIF_NWE_NAPHTHA", "2026-08-24", "median")


def test_get_price_quote_validates_duplicate_quote_dates(tmp_path: Path) -> None:
    (tmp_path / "price_quotes.csv").write_text(
        "date,publication,quote_code,quote_name,unit,low,high,mean\n"
        "2026-08-24,Argus,Q,Quote,USD/MT,1,2,1.5\n"
        "2026-08-24,Argus,Q,Quote,USD/MT,1,2,1.5\n",
        encoding="utf-8",
    )

    with pytest.raises(PriceQuoteError, match="duplicate quotation"):
        get_price_quote("Q", "2026-08-24", "mean", data_dir=tmp_path)

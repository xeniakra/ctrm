import ast
import json
from pathlib import Path

import pytest

from mcp_client_demonstration.valuation import (
    McpToolCallError,
    ValuationError,
    calculate_provisional_value,
    resolve_pricing_period,
)
from optiflow_ctrm.pricing import get_price_quote


ROOT = Path(__file__).parents[1]


def _deal() -> dict:
    fixture = json.loads(
        (ROOT / "evals" / "cases" / "02_forwarded_amendments.json").read_text()
    )
    return fixture["expected"]["deal"]


def test_calculator_does_not_import_server_implementation() -> None:
    source = (ROOT / "mcp_client_demonstration" / "valuation.py").read_text()
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(module == "optiflow_ctrm" or module.startswith("optiflow_ctrm.")
                   for module in imported_modules)


def test_period_calculator_requires_actual_bl_date() -> None:
    with pytest.raises(ValuationError, match="B/L date"):
        resolve_pricing_period(_deal())
    assert resolve_pricing_period(_deal(), "2026-09-14") == {
        "bl_date": "2026-09-14",
        "date_or_period": "2026-09-13..2026-09-17",
    }


def test_valuation_applies_units_differential_and_fx() -> None:
    quote = get_price_quote(
        "ARGUS_CIF_NWE_NAPHTHA", "2026-09-13..2026-09-17", "mean_of_high"
    )
    result = calculate_provisional_value(_deal(), quote, bl_date="2026-09-14")

    assert result["status"] == "completed"
    assert result["price_quote"]["observation_count"] == 4
    assert result["calculation"]["quotation"]["value"] == pytest.approx(596.8175)
    assert result["calculation"]["conversion"]["bbl_per_mt"] == 8.9
    assert result["calculation"]["pricing_quantity"] == {
        "value": pytest.approx(24719.1011235955), "unit": "MT",
    }
    assert result["calculation"]["provisional_unit_price"] == {
        "value": pytest.approx(604.8175), "unit": "USD/MT",
    }
    assert result["calculation"]["gross_value_usd"] == pytest.approx(14950544.94)
    assert result["provisional_value"] == {
        "amount": pytest.approx(13776764.60), "currency": "EUR",
    }
    assert result["provisional_payment"] == {
        "percentage": 90.0, "amount": pytest.approx(12399088.14), "currency": "EUR",
    }


def test_calculator_rejects_a_quote_for_a_different_period() -> None:
    quote = get_price_quote("ARGUS_CIF_NWE_NAPHTHA", "2026-09-14", "mean_of_high")
    with pytest.raises(McpToolCallError, match="different pricing period"):
        calculate_provisional_value(_deal(), quote, "2026-09-14")

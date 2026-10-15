"""Expose exactly parse_recap and get_price_quote as MCP tools.

The server delegates to extractor.py and references.py/pricing.py. It should not
duplicate their validation or calculation rules.
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from optiflow_ctrm.extractor import parse_recap as _parse_recap
from optiflow_ctrm.models import ExtractionResult, PriceQuoteResult, PricingStatistic
from optiflow_ctrm.pricing import get_price_quote as _get_price_quote


mcp = FastMCP(
    "Optiflow CTRM",
    instructions=Path(__file__).with_name("provisional_valuation.md").read_text(
        encoding="utf-8"
    ),
)


@mcp.tool(structured_output=True)
def parse_recap(text: str) -> ExtractionResult:
    """Extract one recap without changing its text.

    Non-null deal fields include source provenance. Missing or ambiguous
    mandatory commercial terms remain null and produce field-specific questions.
    B/L date is later operational data, so a B/L-relative pricing period does not
    itself produce a recap clarification question.
    """

    return _parse_recap(text)


@mcp.tool(structured_output=True)
def get_price_quote(
    quote_code: str,
    date_or_period: str,
    statistic: PricingStatistic,
) -> PriceQuoteResult:
    """Read and aggregate published quotations for one date or inclusive period.

    Use an ISO date or an inclusive ``YYYY-MM-DD..YYYY-MM-DD`` period. Calendar
    dates without a published observation are skipped. Resolve B/L-relative
    expressions to calendar dates before calling this tool.
    """

    return PriceQuoteResult.model_validate(
        _get_price_quote(quote_code, date_or_period, statistic)
    )


def main() -> None:
    """Run the MCP server over standard input/output."""

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

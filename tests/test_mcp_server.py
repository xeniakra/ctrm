import asyncio

from optiflow_ctrm.mcp import server
from optiflow_ctrm.models import PriceQuoteResult
from optiflow_ctrm.pricing import get_price_quote as application_get_price_quote


class _DumpableResult:
    def model_dump(self, **kwargs):
        assert kwargs == {"mode": "json", "by_alias": True}
        return {"deal": {"buyer": "Buyer"}, "questions": [], "provenance": []}


def test_mcp_server_registers_exactly_the_two_required_tools() -> None:
    tools = asyncio.run(server.mcp.list_tools())

    assert {tool.name for tool in tools} == {"parse_recap", "get_price_quote"}
    schemas = {tool.name: tool.inputSchema for tool in tools}
    output_schemas = {tool.name: tool.outputSchema for tool in tools}
    assert schemas["parse_recap"]["required"] == ["text"]
    assert schemas["get_price_quote"]["required"] == [
        "quote_code",
        "date_or_period",
        "statistic",
    ]
    assert output_schemas["parse_recap"]["required"] == [
        "deal",
        "questions",
        "provenance",
    ]
    assert output_schemas["get_price_quote"]["required"] == [
        "quote_code",
        "quote_name",
        "publication",
        "unit",
        "statistic",
        "period",
        "observations",
        "observation_count",
        "aggregation",
        "source_column",
        "value",
        "formula",
    ]


def test_mcp_tools_are_thin_json_safe_application_adapters(monkeypatch) -> None:
    parse_calls = []
    quote_calls = []
    parsed = _DumpableResult()
    quote_payload = application_get_price_quote(
        "ARGUS_CIF_NWE_NAPHTHA", "2026-09-14", "mean"
    )

    def fake_parse(text):
        parse_calls.append(text)
        return parsed

    def fake_quote(quote_code, date_or_period, statistic):
        quote_calls.append((quote_code, date_or_period, statistic))
        return quote_payload

    monkeypatch.setattr(server, "_parse_recap", fake_parse)
    monkeypatch.setattr(server, "_get_price_quote", fake_quote)

    assert server.parse_recap("original recap\n") is parsed
    quote = server.get_price_quote("Q", "2026-09-14..2026-09-18", "mean")
    assert isinstance(quote, PriceQuoteResult)
    assert quote.model_dump(mode="json", by_alias=True) == quote_payload
    assert parse_calls == ["original recap\n"]
    assert quote_calls == [("Q", "2026-09-14..2026-09-18", "mean")]

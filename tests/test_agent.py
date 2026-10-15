import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mcp_client_demonstration.agent import ProvisionalValuationAgent, main
from optiflow_ctrm.mcp import server
from optiflow_ctrm.pricing import get_price_quote


ROOT = Path(__file__).parents[1]
FIXTURE = json.loads((ROOT / "tests/fixtures/agent_workflow.json").read_text())


class ScriptedRuntime:
    def __init__(self, actions):
        self.actions = iter(deepcopy(actions))
        self.requests = []

    def generate_json(self, messages, json_schema):
        self.requests.append((deepcopy(messages), deepcopy(json_schema)))
        return SimpleNamespace(value=next(self.actions))


class FakeMcpSession:
    def __init__(self):
        self.calls = []
        self.extraction = json.loads(
            (ROOT / FIXTURE["extraction_fixture"]).read_text()
        )["expected"]

    async def list_tools(self):
        return SimpleNamespace(tools=await server.mcp.list_tools())

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "parse_recap":
            return self.extraction
        if name == "get_price_quote":
            return get_price_quote(**arguments)
        raise AssertionError(f"unexpected tool: {name}")


def run_agent(actions, *, session=None, bl_date=None, max_steps=8):
    session = session or FakeMcpSession()
    runtime = ScriptedRuntime(actions)
    prompts = []

    def answer_prompt(prompt):
        prompts.append(prompt)
        return FIXTURE["bl_date"]

    agent = ProvisionalValuationAgent(
        session, runtime, server.mcp.instructions, answer_prompt, max_steps
    )
    recap = (ROOT / FIXTURE["source"]).read_text()
    result = asyncio.run(agent.answer(FIXTURE["question"], recap, bl_date))
    return result, session, runtime, prompts


def test_model_selects_actions_and_receives_server_instructions_and_results():
    result, session, runtime, prompts = run_agent(FIXTURE["actions"])

    assert result["status"] == "completed"
    assert result["actions"] == FIXTURE["actions"]
    assert result["provisional_value"] == FIXTURE["expected"]["provisional_value"]
    assert result["provisional_payment"] == FIXTURE["expected"]["provisional_payment"]
    assert [name for name, _ in session.calls] == FIXTURE["expected"]["tool_names"]
    assert session.calls[0][1] == {"text": (ROOT / FIXTURE["source"]).read_text()}
    assert session.calls[1][1] == FIXTURE["actions"][3]["arguments"]
    assert len(prompts) == 1 and "actual B/L date" in prompts[0]
    assert len(runtime.requests) == len(FIXTURE["actions"])
    assert server.mcp.instructions in runtime.requests[0][0][0]["content"]
    initial_context = json.loads(runtime.requests[0][0][1]["content"])
    assert initial_context["available_results"] == FIXTURE["initial_results"]
    assert "Available actions:" in runtime.requests[0][0][0]["content"]
    assert json.dumps(runtime.requests[0][1], ensure_ascii=False) in runtime.requests[0][0][0]["content"]
    choices = runtime.requests[0][1]["oneOf"]
    assert {item["properties"]["name"]["const"] for item in choices} == {
        "parse_recap", "get_price_quote", "ask_bl_date",
        "resolve_pricing_period", "calculate_provisional_value",
    }
    after_parse = json.loads(runtime.requests[1][0][-1]["content"])
    assert after_parse["action_result"]["result"]["deal"] == session.extraction["deal"]
    after_dates = json.loads(runtime.requests[3][0][-1]["content"])
    assert after_dates["action_result"]["result"]["date_or_period"] == "2026-09-13..2026-09-17"
    after_quote = json.loads(runtime.requests[4][0][-1]["content"])
    assert after_quote["action_result"]["result"]["value"] == pytest.approx(596.8175)


def test_model_can_choose_a_different_valid_action_order():
    actions = deepcopy(FIXTURE["actions"])
    actions[0], actions[1] = actions[1], actions[0]
    result, _, runtime, prompts = run_agent(actions)
    assert result["status"] == "completed"
    assert result["actions"] == actions
    assert len(prompts) == 1
    assert json.loads(runtime.requests[1][0][-1]["content"])["action_result"] == {
        "name": "ask_bl_date", "result": {"bl_date": FIXTURE["bl_date"]},
    }


@pytest.mark.parametrize("action", [FIXTURE["actions"][2], FIXTURE["actions"][-1]])
def test_premature_calculation_fails_without_scripted_tool_calls(action):
    result, session, runtime, _ = run_agent([action])
    assert result["status"] == "failed"
    assert "parsed deal" in result["error"]
    assert session.calls == []
    assert len(runtime.requests) == 1


def test_model_quote_arguments_are_executed_without_expected_call_comparison():
    actions = deepcopy(FIXTURE["actions"])
    actions[3]["arguments"]["date_or_period"] = "2026-09-14"
    result, session, runtime, _ = run_agent(actions)
    assert session.calls[1][1]["date_or_period"] == "2026-09-14"
    assert result["status"] == "failed"
    assert "different pricing period" in result["error"]
    assert len(session.calls) == 2
    assert len(runtime.requests) == 5


@pytest.mark.parametrize("action", [
    {"name": "unknown_tool", "arguments": {}},
    {"name": "parse_recap", "arguments": {"text": "rewritten recap"}},
    {"name": "ask_bl_date", "arguments": {"date": "2026-09-14"}},
    {"name": "parse_recap"},
    {"name": "parse_recap", "arguments": []},
])
def test_invalid_actions_fail_without_fallback_or_repeated_inference(action):
    result, session, runtime, prompts = run_agent([action])
    assert result["status"] == "failed"
    assert session.calls == []
    assert prompts == []
    assert len(runtime.requests) == 1


def test_repeated_actions_stop_at_the_step_limit():
    result, session, runtime, prompts = run_agent(
        [FIXTURE["actions"][1]] * 3, max_steps=3
    )
    assert result["status"] == "failed"
    assert "within 3 actions" in result["error"]
    assert session.calls == []
    assert len(runtime.requests) == 3
    assert len(prompts) == 1


def test_extraction_questions_stop_valuation():
    session = FakeMcpSession()
    questions = [{"field": "/deal/pricing/differential/value",
                  "question": "What is the agreed price differential?"}]
    session.extraction = {"deal": None, "questions": questions}
    result, session, runtime, _ = run_agent([FIXTURE["actions"][0]], session=session)
    assert result["status"] == "needs_input"
    assert result["questions"] == questions
    assert questions[0]["question"] in result["answer"]
    assert len(session.calls) == 1 and len(runtime.requests) == 1


def test_mcp_error_stops_without_a_fallback_request():
    session = FakeMcpSession()

    async def failing_call(name, arguments):
        session.calls.append((name, arguments))
        return SimpleNamespace(isError=True, content=[SimpleNamespace(text="extraction failed")])

    session.call_tool = failing_call
    result, session, runtime, _ = run_agent([FIXTURE["actions"][0]], session=session)
    assert result["status"] == "failed"
    assert "extraction failed" in result["error"]
    assert len(session.calls) == 1 and len(runtime.requests) == 1


def test_agent_uses_real_mcp_protocol_with_scripted_inference(monkeypatch):
    from mcp.shared.memory import create_connected_server_and_client_session
    from optiflow_ctrm.models import ExtractionResult

    extraction = FakeMcpSession().extraction
    raw_inputs = []

    def parse(text):
        raw_inputs.append(text)
        return ExtractionResult.model_validate(extraction)

    monkeypatch.setattr(server, "_parse_recap", parse)
    actions = deepcopy(FIXTURE["actions"])
    actions.pop(1)
    runtime = ScriptedRuntime(actions)
    recap = (ROOT / FIXTURE["source"]).read_text()

    async def run():
        async with create_connected_server_and_client_session(server.mcp) as session:
            agent = ProvisionalValuationAgent(session, runtime, server.mcp.instructions)
            return await agent.answer(FIXTURE["question"], recap, FIXTURE["bl_date"])

    result = asyncio.run(run())
    assert result["status"] == "completed"
    assert result["provisional_value"] == FIXTURE["expected"]["provisional_value"]
    assert raw_inputs == [recap]


@pytest.mark.parametrize("interactive", [False, True])
def test_cli_uses_model_actions_and_mcp_handshake(monkeypatch, capsys, interactive):
    import mcp
    import mcp.client.stdio

    session = FakeMcpSession()
    actions = deepcopy(FIXTURE["actions"])
    if not interactive:
        actions.pop(1)
    runtime = ScriptedRuntime(actions)
    prompts = []

    async def initialize():
        return SimpleNamespace(instructions=server.mcp.instructions)

    session.initialize = initialize

    @asynccontextmanager
    async def fake_stdio_client(parameters):
        assert parameters.args == ["-m", "optiflow_ctrm.mcp.server"]
        yield ("read", "write")

    @asynccontextmanager
    async def fake_client_session(read, write):
        yield session

    def answer_prompt(prompt):
        prompts.append(prompt)
        return FIXTURE["bl_date"]

    monkeypatch.setattr(mcp.client.stdio, "stdio_client", fake_stdio_client)
    monkeypatch.setattr(mcp, "ClientSession", fake_client_session)
    monkeypatch.setattr("builtins.input", answer_prompt)
    monkeypatch.setattr("mcp_client_demonstration.agent.create_default_runtime", lambda: runtime)
    arguments = ["--recap", str(ROOT / FIXTURE["source"])]
    if not interactive:
        arguments.extend(["--bl-date", FIXTURE["bl_date"]])
    assert main(arguments) == 0
    assert len(prompts) == int(interactive)
    assert len(runtime.requests) == len(actions)
    output = capsys.readouterr().out
    assert "EUR 13,776,764.60" in output
    assert "90% provisional payment: EUR 12,399,088.14" in output

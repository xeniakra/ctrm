# Provisional valuation MCP demo

Start the pinned local llama.cpp model server in one terminal:

```bash
python scripts/start_llama_server.py --offline
```

Then run the client. It starts the MCP server over stdio and gives the local
model the server's instructions and tool schemas. The model chooses each next
action using the results of previous actions. The client executes those actions
and prints the calculated provisional cargo value and payment:

```bash
unalias python
rehash
source /Users/xenia/home_task/.venv/bin/activate
python -m mcp_client_demonstration.agent
```

For a reproducible non-interactive example with the explicit scenario B/L date
14 September 2026:

```bash
python -m mcp_client_demonstration.agent --bl-date 2026-09-14
```

The server's [valuation guide](../src/optiflow_ctrm/mcp/provisional_valuation.md)
is delivered in the MCP initialization response's `instructions` field. The
client includes it in the model's context along with the schemas and descriptions
from `list_tools`. The model chooses between the two MCP tools and three local
actions: `ask_bl_date`, `resolve_pricing_period`, and
`calculate_provisional_value`. Those local actions obtain user input or run
deterministic calculations on stored results; they are not extra MCP tools.

There is one model inference per decision. The client executes the selected
action and returns its result to the model. The model's quote arguments are
sent to MCP as generated. For `parse_recap`, the model supplies the reference
`$recap`, which the client expands to the original text, preserving newlines
and source offsets without asking the model to reproduce the document.

Tool input validation belongs to the server; the client checks the action
envelope and the calculator checks its inputs. A malformed action, tool error,
or missing calculator input ends the run with an explicit error. There are no
expected-call comparisons, repeated translation trials, or fallback requests.
An eight-action limit prevents an endless loop; it is not eight attempts at
the same request. The returned result retains the model's actions for inspection.

Deterministic tests use scripted model responses to verify dispatch, result
feedback, clarification, and failure handling. They do not measure live-model
tool-selection reliability.

The reported `provisional_value` is the gross cargo value at the provisional
price. Because recap 02 separately requires a 90% provisional payment, the demo
also reports `provisional_payment`; it does not silently treat the two values as
the same thing. Calculations use the nominal recap quantity and state its ±5%
buyer-option tolerance rather than selecting a quantity inside that range.

"""Contract invariant for #121616: the tool_call schema must describe only
shapes the dispatcher will accept.

The dispatcher (``resolve_underlying_call``) deliberately rejects a
multi-entry batch naming a local tool — that refusal is pinned by
``test_connector_local_batches.py`` and stays. The schema therefore must not
invite the model to form one: the batch array must be presented as a
connector-only shape, never as the general ``tool_call`` shape.
"""

from tools.tool_search import TOOL_CALL_NAME, bridge_tool_schemas, resolve_underlying_call


def _tool_call_schema():
    return next(td for td in bridge_tool_schemas(1)
                if td["function"]["name"] == TOOL_CALL_NAME)


def test_schema_does_not_invite_a_local_batch():
    """The tool description must not frame the array as the general shape
    ("one entry per invocation; a single call is an array of one" teaches
    N > 1 and leaves the prefix rule to the rejection path)."""
    description = _tool_call_schema()["function"]["description"]
    assert "one entry per invocation" not in description
    assert "an array of one" not in description


def test_calls_property_says_single_entry_for_local_tools():
    """The ``calls`` property must not offer "one or more" to local tools —
    that is the sentence a compliant model batches against."""
    calls = _tool_call_schema()["function"]["parameters"]["properties"]["calls"]
    assert "One local invocation, or one or more connector invocations" \
        not in calls["description"]


def test_model_style_local_batch_is_still_rejected_at_dispatch():
    """Dispatch level: two independent local lookups in one ``calls`` array
    are refused (the enforcement half of the contract)."""
    batch = {"calls": [
        {"name": "session_search", "arguments": {"query": "alpha"}},
        {"name": "todo_list", "arguments": {}},
    ]}
    name, _args, err = resolve_underlying_call(batch)
    assert name is None
    assert err is not None and "exactly one entry" in err

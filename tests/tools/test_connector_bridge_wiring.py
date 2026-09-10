"""Behavior tests for the connector leg of the tool_search bridge.

DI-callable idiom (test_managed_tool_gateway.py precedent): remote legs are
injected as plain callables; no module mocks, no network. Local-only behavior
is pinned byte-identical when the remote leg fails (D32).
"""

import json
import logging

import pytest

from agent.tool_dispatch_helpers import _peel_bridge_call
from tools.tool_gateway.bridge import connector_describe
from tools.tool_search import (
    CONNECTOR_BATCH_SENTINEL,
    ToolSearchConfig,
    assemble_tool_defs,
    dispatch_tool_describe,
    dispatch_tool_search,
    resolve_underlying_call,
)
from tools.tool_search_validation import normalize_tool_call_entries


def _tool_search_description(tool_defs):
    # session_search is in the default defer set, so the bridge activates in
    # both arms for the same reason: a deferrable local tool exists.
    defs = tool_defs + [{"type": "function", "function": {
        "name": "session_search", "description": "Search past sessions", "parameters": {}}}]
    assembled = assemble_tool_defs(
        defs, context_length=200_000, config=ToolSearchConfig.from_raw({"enabled": "on"}))
    assert assembled.activated
    return next(td["function"]["description"] for td in assembled.tool_defs
                if td["function"]["name"] == "tool_search")


def test_tool_search_names_manage_connections_only_when_the_session_has_it():
    """The model learns that connectors__ names belong to accounts managed by
    manage_connections from the tool_search description, but only when that tool is in the
    session. Signed out (or connectors off) the tool is absent and the description must not
    name a tool the model cannot call."""
    with_connections = _tool_search_description(_local_defs())
    assert "manage_connections" in with_connections
    assert "connectors__" in with_connections

    without = _tool_search_description(
        [td for td in _local_defs() if td["function"]["name"] != "manage_connections"])
    assert "manage_connections" not in without


def _local_defs():
    """One deferrable (mcp-toolset) tool, one core-shaped tool."""
    return [
        {"type": "function", "function": {"name": "manage_connections", "parameters": {}}},
        {
            "type": "function",
            "function": {
                "name": "mcp__github__create_issue",
                "description": "Create a GitHub issue",
                "parameters": {"type": "object", "properties": {}, "required": ["title"]},
            },
        },
    ]


# ---------------------------------------------------------------------------
# resolve_underlying_call: batch shapes
# ---------------------------------------------------------------------------


def test_resolve_single_connector_entry_returns_sentinel():
    name, args, err = resolve_underlying_call(
        {"calls": [{"name": "connectors__gmail__SEND_EMAIL", "arguments": {"to": "x"}}]}
    )
    assert err is None
    assert name == CONNECTOR_BATCH_SENTINEL
    assert args["calls"][0]["name"] == "connectors__gmail__SEND_EMAIL"
    assert args["calls"][0]["arguments"] == {"to": "x"}


def test_resolve_multi_local_batch_requires_separate_calls():
    name, args, err = resolve_underlying_call(
        {"calls": [
            {"name": "some_local_tool", "arguments": {}},
            {"name": "another_local", "arguments": {}},
        ]}
    )
    assert name is None
    assert "one entry per tool_call" in err


def test_resolve_legacy_single_shape_unchanged_for_local_names():
    # Non-deferrable local name keeps the historical rejection message.
    name, args, err = resolve_underlying_call({"name": "not_a_real_tool", "arguments": {}})
    assert name is None
    assert "not a deferrable tool" in (err or "")


def test_resolve_legacy_connector_single_shape_routes_to_sentinel():
    name, args, err = resolve_underlying_call(
        {"name": "connectors__gmail__CREATE_EMAIL_DRAFT", "arguments": {}}
    )
    assert err is None
    assert name == CONNECTOR_BATCH_SENTINEL
    assert len(args["calls"]) == 1


@pytest.mark.parametrize(
    "bad,expected_fragment",
    [
        ({}, "requires 'calls'"),
        ({"calls": []}, "non-empty array"),
        ({"calls": [{"arguments": {}}]}, "requires a 'name'"),
        ({"calls": [{"name": "tool_search"}]}, "itself a bridge tool"),
        ({"calls": [{"name": "x", "arguments": "not json {"}]}, "not valid JSON"),
        ({"calls": [{"name": "x", "arguments": 42}]}, "must be an object"),
        ({"calls": "nope"}, "non-empty array"),
    ],
)
def test_normalize_rejects_malformed_batches(bad, expected_fragment):
    entries, err = normalize_tool_call_entries(bad)
    assert entries == []
    assert expected_fragment in (err or "")


# ---------------------------------------------------------------------------
# dispatch_tool_search: remote merge
# ---------------------------------------------------------------------------


def _fake_connector_search(queries):
    assert queries == [{"use_case": "send an email"}]
    return {
        "results": [
            {"index": 1, "use_case": "send an email", "tools": ["GMAIL_SEND_EMAIL", "ORPHAN_TOOL"]},
        ],
        "schemas": {
            "GMAIL_SEND_EMAIL": {
                "connector": "gmail",
                "tool": "GMAIL_SEND_EMAIL",
                "description": "Send an email via gmail",
                "input_schema": {"type": "object", "required": ["to", "subject"]},
            },
            # ORPHAN_TOOL deliberately has no schema entry: without a
            # connector it cannot compose a callable name and must be dropped.
        },
        "connections": [{"connector": "gmail", "connected": False, "description": ""}],
    }


def _registered_local_defs():
    """Deferrable local tools the registry knows, so they enter the BM25 catalog: an issue
    tracker whose descriptions mention email notifications, and an unrelated tool."""
    from tools.registry import registry

    specs = [
        ("mcp__tracker__create_issue", "Create an issue. Sends an email notification to the team."),
        ("mcp__tracker__list_issues", "List issues in a project. Email digests are optional."),
        ("mcp__tracker__archive_project", "Archive a project and its issues."),
    ]
    defs = []
    for name, desc in specs:
        schema = {"name": name, "description": desc,
                  "parameters": {"type": "object", "properties": {"id": {"type": "string"}}}}
        registry.register(name=name, handler=lambda a, **k: "{}", schema=schema, toolset="mcp-tracker")
        defs.append({"type": "function", "function": schema})
    return [{"type": "function", "function": {"name": "manage_connections", "parameters": {}}}] + defs, [n for n, _ in specs]


def test_connector_intent_is_not_starved_by_local_tools_sharing_one_word():
    """The reported bug: with a large local catalog, tools that merely shared 'email' filled
    every slot and the gmail connector tool never appeared. Ranked as one corpus with the
    rarest-token gate ('gmail' is in one document), the connector tool is the only result."""
    from tools.registry import registry

    defs, names = _registered_local_defs()
    try:
        out = json.loads(dispatch_tool_search(
            {"queries": ["send gmail email"], "limit": 5},
            current_tool_defs=defs,
            connector_search=lambda q: {
                "results": [{"use_case": "send gmail email", "tools": ["GMAIL_SEND_EMAIL"]}],
                "schemas": {"GMAIL_SEND_EMAIL": {
                    "connector": "gmail", "tool": "GMAIL_SEND_EMAIL",
                    "description": "Send an email via gmail", "input_schema": {}}},
            }))
        assert out["results"][0]["matches"] == ["connectors__gmail__SEND_EMAIL"]
    finally:
        for n in names:
            registry.deregister(n)


def test_both_sources_answer_within_one_limit():
    """When a local MCP server and a connector both serve the same service, both surface,
    ranked by the same BM25 pass, and `limit` caps the group as a whole."""
    from tools.registry import registry

    defs, names = _registered_local_defs()
    try:
        out = json.loads(dispatch_tool_search(
            {"queries": ["tracker create issue"], "limit": 2},
            current_tool_defs=defs,
            connector_search=lambda q: {
                "results": [{"use_case": "tracker create issue", "tools": ["TRACKER_CREATE_ISSUE"]}],
                "schemas": {"TRACKER_CREATE_ISSUE": {
                    "connector": "tracker", "tool": "TRACKER_CREATE_ISSUE",
                    "description": "Create a tracker issue", "input_schema": {}}},
            }))
        matches = out["results"][0]["matches"]
        assert len(matches) == 2
        assert set(matches) == {"mcp__tracker__create_issue", "connectors__tracker__CREATE_ISSUE"}
    finally:
        for n in names:
            registry.deregister(n)


def test_search_composes_lowercase_connector_from_vendor_cased_schema():
    # The gateway search surface leaks vendor-cased connector slugs for
    # custom toolkits; the composed name must carry the lowercase catalog
    # form or the gateway's own policy gates refuse the call.
    def cased_search(queries):
        return {
            "results": [{"index": 1, "tools": ["CUSTOM_X_READ"]}],
            "schemas": {
                "CUSTOM_X_READ": {
                    "connector": "CUSTOM_X",
                    "tool": "CUSTOM_X_READ",
                    "description": "d",
                    "input_schema": {},
                }
            },
        }

    out = json.loads(
        dispatch_tool_search(
            {"queries": ["custom_x read"]},
            current_tool_defs=_local_defs(),
            connector_search=cased_search,
        )
    )
    assert "connectors__custom_x__READ" in out["results"][0]["matches"]


def test_search_merges_remote_hits_tagged_as_connectors():
    out = json.loads(
        dispatch_tool_search(
            {"queries": ["send an email"]},
            current_tool_defs=_local_defs(),
            connector_search=_fake_connector_search,
        )
    )
    matches = out["results"][0]["matches"]
    composed = "connectors__gmail__SEND_EMAIL"
    assert composed in matches
    assert all("ORPHAN_TOOL" not in m for m in matches)
    record = out["tools"][composed]
    assert record["source"] == "connectors"
    assert record["source_name"] == "gmail"
    assert record["required"] == ["to", "subject"]


@pytest.mark.parametrize("order", [("GMAIL_FETCH_PROFILE", "FETCH_PROFILE"), ("FETCH_PROFILE", "GMAIL_FETCH_PROFILE")])
def test_search_keeps_only_the_twin_a_colliding_name_reaches(order, caplog):
    """Composition is not injective: GMAIL_FETCH_PROFILE and a literal FETCH_PROFILE on
    gmail both compose to connectors__gmail__FETCH_PROFILE, and describe/execute decode
    that name to GMAIL_FETCH_PROFILE. If a vendor ever ships both, search must not
    describe the literal under a name that runs the prefixed tool, whichever the
    gateway listed first, and must say so in the log rather than alias silently."""
    def twins(queries):
        return {
            "results": [{"index": 1, "tools": list(order)}],
            "schemas": {
                "GMAIL_FETCH_PROFILE": {"connector": "gmail", "tool": "GMAIL_FETCH_PROFILE",
                                        "description": "prefixed twin", "input_schema": {}},
                "FETCH_PROFILE": {"connector": "gmail", "tool": "FETCH_PROFILE",
                                  "description": "literal twin", "input_schema": {}},
            },
        }

    with caplog.at_level(logging.WARNING, logger="tools.connector_search"):
        out = json.loads(dispatch_tool_search(
            {"queries": ["gmail fetch profile"]},
            current_tool_defs=_local_defs(),
            connector_search=twins,
        ))
    assert out["results"][0]["matches"] == ["connectors__gmail__FETCH_PROFILE"]
    assert out["tools"]["connectors__gmail__FETCH_PROFILE"]["description"] == "prefixed twin"
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "GMAIL_FETCH_PROFILE" in warnings[0] and "FETCH_PROFILE" in warnings[0]


def test_search_limit_caps_the_group_across_both_legs_and_counts_total():
    def many_hits(queries):
        slugs = [f"CUSTOM_X_TOOL_{i}" for i in range(9)]
        return {
            "results": [{"index": 1, "tools": slugs}],
            "schemas": {
                s: {"connector": "custom_x", "tool": s, "description": "widget", "input_schema": {}}
                for s in slugs
            },
        }

    out = json.loads(
        dispatch_tool_search(
            {"queries": ["custom_x widget"], "limit": 3},
            current_tool_defs=_local_defs(),
            connector_search=many_hits,
        )
    )
    matches = out["results"][0]["matches"]
    assert len(matches) == 3  # limit is the per-query cap across BOTH legs
    assert all(m.startswith("connectors__") for m in matches)
    # total_available counts returned remote tools on top of the local catalog
    # (empty here: the fake def is not registry-backed in this test env).
    assert out["total_available"] == 3


def test_search_drops_remote_group_with_mismatched_use_case_echo():
    def misaligned(queries):
        return {
            "results": [{"index": 1, "use_case": "SOMETHING ELSE", "tools": ["CUSTOM_X_READ"]}],
            "schemas": {
                "CUSTOM_X_READ": {"connector": "custom_x", "tool": "CUSTOM_X_READ", "description": "d", "input_schema": {}}
            },
        }

    out = json.loads(
        dispatch_tool_search(
            {"queries": ["send an email"]},
            current_tool_defs=_local_defs(),
            connector_search=misaligned,
        )
    )
    assert not any(m.startswith("connectors__") for m in out["results"][0]["matches"])


def test_search_identical_to_local_only_when_remote_leg_fails():
    def exploding_search(queries):
        raise RuntimeError("gateway exploded")

    local_only = dispatch_tool_search(
        {"queries": ["send an email"]},
        current_tool_defs=_local_defs(),
        connector_search=lambda queries: {},
    )
    with_failure = dispatch_tool_search(
        {"queries": ["send an email"]},
        current_tool_defs=_local_defs(),
        connector_search=exploding_search,
    )
    assert local_only == with_failure  # byte-identical: D32


def test_search_never_sends_the_gateway_more_use_cases_than_it_accepts():
    """The gateway's search route returns HTTP 502 above 7 use_cases per request, and one
    tool_search call maps to one gateway request. Seven queries reach it in one request;
    eight are refused before any request is made, so the model gets a retry hint and the
    gateway never sees a request it cannot answer."""
    sent = []

    def recording_search(use_cases):
        sent.append(use_cases)
        return {}

    seven = [f"query {i}" for i in range(7)]
    parsed = json.loads(dispatch_tool_search(
        {"queries": seven}, current_tool_defs=_local_defs(), connector_search=recording_search))
    assert "error" not in parsed
    assert sent == [[{"use_case": q} for q in seven]]

    sent.clear()
    parsed = json.loads(dispatch_tool_search(
        {"queries": seven + ["query 7"]}, current_tool_defs=_local_defs(),
        connector_search=recording_search))
    assert "too many queries" in parsed["error"]
    assert sent == []


# ---------------------------------------------------------------------------
# dispatch_tool_describe: remote merge
# ---------------------------------------------------------------------------


def test_describe_merges_remote_schema_and_leaves_misses_in_not_found():
    composed = "connectors__gmail__SEND_EMAIL"
    stale = "connectors__gmail__GONE_TOOL"

    def fake_describe(names):
        assert set(names) == {composed, stale}
        return {"tools": {composed: {"description": "Send an email", "parameters": {"type": "object"}}}}

    out = json.loads(
        dispatch_tool_describe(
            {"names": [composed, stale]},
            current_tool_defs=_local_defs(),
            connector_describe=fake_describe,
        )
    )
    assert out["tools"][composed]["parameters"] == {"type": "object"}
    assert stale in out["not_found"]
    assert "errors" not in out  # a connector miss is stale/unknown, not an error


def test_describe_connector_names_fall_to_not_found_when_dark():
    composed = "connectors__gmail__SEND_EMAIL"
    out = json.loads(
        dispatch_tool_describe(
            {"names": [composed]},
            current_tool_defs=_local_defs(),
            connector_describe=lambda names: {},
        )
    )
    assert out["not_found"] == [composed]


# ---------------------------------------------------------------------------
# planner admission: only PURE connector batches are parallel-safe
# ---------------------------------------------------------------------------


def test_peel_admits_pure_connector_batch_as_sentinel():
    name, args = _peel_bridge_call(
        "tool_call",
        {"calls": [
            {"name": "connectors__gmail__SEND_EMAIL", "arguments": {}},
            {"name": "connectors__slack__POST_MESSAGE", "arguments": {}},
        ]},
    )
    assert name == CONNECTOR_BATCH_SENTINEL


def test_peel_keeps_mixed_and_local_batches_as_sequential_barrier():
    mixed = {"calls": [
        {"name": "connectors__gmail__SEND_EMAIL", "arguments": {}},
        {"name": "write_file", "arguments": {"path": "x"}},
    ]}
    name, args = _peel_bridge_call("tool_call", mixed)
    assert name == "tool_call"  # barrier: local entries never got admission

    all_local = {"calls": [
        {"name": "write_file", "arguments": {"path": "x"}},
        {"name": "read_file", "arguments": {"path": "x"}},
    ]}
    name, _ = _peel_bridge_call("tool_call", all_local)
    assert name == "tool_call"


# ---------------------------------------------------------------------------
# run_remote through production dispatch: vendor-slug restoration, the
# one-pass literal fallback, and the gateway request body
#
# tool_call batches re-enter core dispatch once per connector entry, so each
# entry reaches run_remote alone. The gateway is swapped at the bridge's
# client factory, the same seam the real client is created through.
# ---------------------------------------------------------------------------


def _connectors_on(monkeypatch, client_factory):
    from tools.registry import invalidate_check_fn_cache
    from tools.tool_gateway import bridge, config

    monkeypatch.setattr(config, "connectors_available", lambda: True)
    monkeypatch.setattr(bridge, "connectors_available", lambda: True)
    monkeypatch.setattr(bridge, "_default_client_factory", client_factory)
    invalidate_check_fn_cache()


def _tool_call(calls):
    import model_tools

    return json.loads(model_tools.handle_function_call(
        "tool_call", {"calls": calls}, enabled_toolsets=["connections"], session_id="bridge-session",
        skip_pre_tool_call_hook=True, skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True))


def test_execute_falls_back_once_for_literal_slug_without_touching_siblings(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.calls = []

        def execute(self, planned):
            self.calls.append([plan.tool for plan in planned])
            (plan,) = planned
            if plan.tool == "GRANOLA_FETCH_NOTES":
                return [{"data": None, "error": {"code": "TOOL_NOT_FOUND", "message": "missing"}}]
            if plan.tool == "SLACK_POST_MESSAGE":
                return [{"data": None, "error": {"code": "TOOL_NOT_ALLOWED", "message": "blocked"}}]
            return [{"data": f"ran {plan.tool}", "error": None}]

    client = FakeClient()
    _connectors_on(monkeypatch, lambda: client)
    out = _tool_call([
        {"name": "connectors__gmail__SEND_EMAIL", "arguments": {}},
        {"name": "connectors__granola__FETCH_NOTES", "arguments": {}},
        {"name": "connectors__slack__POST_MESSAGE", "arguments": {}},
    ])

    # Every entry crosses the wire under its restored vendor slug. Only the
    # confirmed TOOL_NOT_FOUND miss is retried, once, under the literal slug;
    # the success and the TOOL_NOT_ALLOWED sibling are never re-sent.
    assert client.calls == [
        ["GMAIL_SEND_EMAIL"], ["GRANOLA_FETCH_NOTES"], ["FETCH_NOTES"], ["SLACK_POST_MESSAGE"]]
    assert out["results"][0]["response"] == "ran GMAIL_SEND_EMAIL"
    assert out["results"][1] == {
        "index": 1, "name": "connectors__granola__FETCH_NOTES", "response": "ran FETCH_NOTES"}
    assert out["results"][2]["error"]["code"] == "TOOL_NOT_ALLOWED"
    assert out["success_count"] == 2 and out["error_count"] == 1


@pytest.mark.parametrize("fail_at", ["GRANOLA_FETCH_NOTES", "FETCH_NOTES"])
def test_execute_transport_failure_degrades_only_the_failing_entry(monkeypatch, fail_at):
    class FakeClient:
        def execute(self, planned):
            (plan,) = planned
            if plan.tool == fail_at:
                raise RuntimeError("transport failed")
            if plan.tool == "GRANOLA_FETCH_NOTES":
                return [{"data": None, "error": {"code": "TOOL_NOT_FOUND", "message": "missing"}}]
            return [{"data": "sibling", "error": None}]

    _connectors_on(monkeypatch, FakeClient)
    out = _tool_call([
        {"name": "connectors__gmail__SEND_EMAIL", "arguments": {}},
        {"name": "connectors__granola__FETCH_NOTES", "arguments": {}},
    ])

    # Whether the primary send or the literal retry blows up, only that entry
    # degrades to PROVIDER_ERROR; the sibling keeps its result.
    assert out["results"][0]["response"] == "sibling"
    assert out["results"][1]["error"]["code"] == "PROVIDER_ERROR"
    assert out["success_count"] == 1 and out["error_count"] == 1


class _FakeResponse:
    def __init__(self, body):
        self.status_code = 200
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


class _RecordingTransport:
    """Records every outgoing request; answers each with `data` echoes."""

    def __init__(self):
        self.requests = []

    def request(self, method, url, *, headers=None, json=None, timeout=None):
        self.requests.append({"method": method, "url": url, "json": json})
        tools = (json or {}).get("tools") or []
        results = [
            {"index": i, "connector": t.get("connector"), "tool": t.get("tool"), "data": "ok"}
            for i, t in enumerate(tools)
        ]
        return _FakeResponse(
            {
                "results": results,
                "successCount": len(results),
                "errorCount": 0,
                "totalCount": len(results),
            }
        )


def _recording_client_factory(transport):
    from tools.tool_gateway.client import ConnectorClient

    return lambda: ConnectorClient(
        transport=transport,
        endpoint_resolver=lambda: "https://tool-gateway.test",
        header_provider=lambda url: {"Authorization": "Bearer nous-token"},
    )


def _sent_tools(transport):
    assert len(transport.requests) == 1
    return transport.requests[0]["json"]["tools"]


def test_hook_rewrite_and_restored_vendor_slug_reach_the_gateway_request_body(monkeypatch):
    import hermes_cli.plugins as plugins

    transport = _RecordingTransport()
    _connectors_on(monkeypatch, _recording_client_factory(transport))
    # A pre_tool_call redaction pass: the secret must never leave the process.
    monkeypatch.setattr(plugins, "_dispatch_pre_tool_call_hooks",
                        lambda name, args, **kw: (None, {**args, "body": "[REDACTED]"}))

    out = _tool_call([{"name": "connectors__gmail__SEND_EMAIL",
                       "arguments": {"to": "x@example.com", "body": "sk-secret"}}])

    assert _sent_tools(transport) == [
        {"connector": "gmail", "tool": "GMAIL_SEND_EMAIL",
         "arguments": {"to": "x@example.com", "body": "[REDACTED]"}},
    ]
    assert "sk-secret" not in json.dumps(transport.requests[0]["json"])
    assert out["results"][0]["response"] == "ok"  # correlation survives the rewrite


# ---------------------------------------------------------------------------
# bridge.connector_describe: composed-name round trip
# ---------------------------------------------------------------------------


def test_connector_describe_maps_slugs_back_to_composed_names():
    class FakeClient:
        def schemas(self, slugs):
            assert slugs == ["GMAIL_SEND_EMAIL", "SEND_EMAIL"]
            return {
                "schemas": {
                    "GMAIL_SEND_EMAIL": {
                        "connector": "gmail",
                        "tool": "GMAIL_SEND_EMAIL",
                        "description": "Send an email",
                        "input_schema": {"type": "object"},
                    }
                },
                "not_found": [],
            }

    out = connector_describe(
        ["connectors__gmail__SEND_EMAIL", "connectors__broken"],
        availability=lambda: True,
        client_factory=lambda: FakeClient(),
    )
    assert out["tools"]["connectors__gmail__SEND_EMAIL"]["parameters"] == {"type": "object"}


def test_connector_describe_colliding_candidate_slugs_resolve_per_name():
    # connectors__first__SECOND_X nominates (FIRST_SECOND_X, SECOND_X) and
    # connectors__second__X nominates (SECOND_X, X): the shared SECOND_X must
    # not be claimed globally by whichever name came first. Each name takes
    # its own best-ranked resolved candidate — here the first name's prefixed
    # primary is unknown to the gateway, so BOTH names land on SECOND_X.
    class FakeClient:
        def schemas(self, slugs):
            assert slugs == ["FIRST_SECOND_X", "SECOND_X", "X"]
            return {
                "schemas": {
                    "SECOND_X": {
                        "connector": "second",
                        "tool": "SECOND_X",
                        "description": "Shared slug",
                        "input_schema": {"type": "object"},
                    }
                },
                "not_found": ["FIRST_SECOND_X", "X"],
            }

    out = connector_describe(
        ["connectors__first__SECOND_X", "connectors__second__X"],
        availability=lambda: True,
        client_factory=lambda: FakeClient(),
    )
    assert set(out["tools"]) == {
        "connectors__first__SECOND_X",
        "connectors__second__X",
    }


def test_connector_describe_prefers_each_names_prefixed_candidate():
    # When BOTH of a name's candidates resolve, the prefixed primary wins —
    # the literal is only the recovery lane for slugs the encoder never
    # stripped.
    class FakeClient:
        def schemas(self, slugs):
            assert slugs == ["GMAIL_X", "X"]
            return {
                "schemas": {
                    "GMAIL_X": {
                        "connector": "gmail",
                        "tool": "GMAIL_X",
                        "description": "prefixed",
                        "input_schema": {"type": "object", "title": "prefixed"},
                    },
                    "X": {
                        "connector": "gmail",
                        "tool": "X",
                        "description": "literal",
                        "input_schema": {"type": "object", "title": "literal"},
                    },
                },
                "not_found": [],
            }

    out = connector_describe(
        ["connectors__gmail__X"],
        availability=lambda: True,
        client_factory=lambda: FakeClient(),
    )
    assert out["tools"]["connectors__gmail__X"]["description"] == "prefixed"


def test_connector_describe_is_empty_on_unavailable_and_exploding_client():
    assert connector_describe(["connectors__g__T"], availability=lambda: False) == {}

    def boom():
        raise RuntimeError("boom")

    assert connector_describe(
        ["connectors__g__T"], availability=lambda: True, client_factory=boom
    ) == {}

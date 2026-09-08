"""Behavior tests for the connector leg of the tool_search bridge.

DI-callable idiom (test_managed_tool_gateway.py precedent): remote legs are
injected as plain callables; no module mocks, no network. Local-only behavior
is pinned byte-identical when the remote leg fails (D32).
"""

import json

import pytest

from agent.tool_dispatch_helpers import _peel_bridge_call
from tools.tool_gateway.bridge import connector_describe, dispatch_calls
from tools.tool_search import (
    CONNECTOR_BATCH_SENTINEL,
    dispatch_tool_describe,
    dispatch_tool_search,
    normalize_tool_call_entries,
    resolve_underlying_call,
)


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
            {"queries": ["anything"]},
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


def test_search_remote_leg_respects_per_query_limit_and_counts_total():
    def many_hits(queries):
        slugs = [f"CUSTOM_X_TOOL_{i}" for i in range(9)]
        return {
            "results": [{"index": 1, "tools": slugs}],
            "schemas": {
                s: {"connector": "custom_x", "tool": s, "description": "d", "input_schema": {}}
                for s in slugs
            },
        }

    out = json.loads(
        dispatch_tool_search(
            {"queries": ["anything"], "limit": 3},
            current_tool_defs=_local_defs(),
            connector_search=many_hits,
        )
    )
    matches = out["results"][0]["matches"]
    assert len(matches) == 3  # limit is the per-query cap across BOTH legs
    assert all(m.startswith("connectors__") for m in matches)
    # total_available counts merged remote tools on top of the local catalog
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
# dispatch_calls: per-entry gates
# ---------------------------------------------------------------------------


def test_pre_dispatch_block_denies_one_remote_entry_and_siblings_run():
    class FakeClient:
        def execute(self, planned):
            assert [p.name for p in planned] == ["connectors__slack__POST_MESSAGE"]
            return [{"data": "posted", "error": None}]

    out = json.loads(
        dispatch_calls(
            [
                {"name": "connectors__gmail__SEND_EMAIL", "arguments": {}},
                {"name": "connectors__slack__POST_MESSAGE", "arguments": {}},
            ],
            local_dispatch=lambda n, a: (True, "{}"),
            pre_dispatch=lambda name, args: (
                ("blocked by policy", None) if "gmail" in name else (None, None)
            ),
            availability=lambda: True,
            client_factory=lambda: FakeClient(),
        )
    )
    assert out["results"][0]["error"]["code"] == "USER_DENIED"
    assert out["results"][1]["response"] == "posted"
    assert out["success_count"] == 1 and out["error_count"] == 1


# ---------------------------------------------------------------------------
# dispatch_calls: vendor-slug restoration and one-pass literal fallback
# ---------------------------------------------------------------------------


def test_execute_falls_back_once_for_literal_slug_without_touching_siblings():
    class FakeClient:
        def __init__(self):
            self.calls = []

        def execute(self, planned):
            self.calls.append(tuple(planned))
            if len(self.calls) == 1:
                return [
                    {"data": "sent", "error": None},
                    {
                        "data": None,
                        "error": {"code": "TOOL_NOT_FOUND", "message": "missing"},
                    },
                    {
                        "data": None,
                        "error": {"code": "TOOL_NOT_ALLOWED", "message": "blocked"},
                    },
                ]
            return [{"data": "literal result", "error": None}]

    client = FakeClient()
    out = json.loads(
        dispatch_calls(
            [
                {"name": "connectors__gmail__SEND_EMAIL", "arguments": {}},
                {"name": "connectors__granola__FETCH_NOTES", "arguments": {}},
                {"name": "connectors__slack__POST_MESSAGE", "arguments": {}},
            ],
            local_dispatch=lambda n, a: (True, "{}"),
            availability=lambda: True,
            client_factory=lambda: client,
        )
    )

    assert [[plan.tool for plan in call] for call in client.calls] == [
        ["GMAIL_SEND_EMAIL", "GRANOLA_FETCH_NOTES", "SLACK_POST_MESSAGE"],
        ["FETCH_NOTES"],
    ]
    assert out["results"][0]["response"] == "sent"
    assert out["results"][1] == {
        "index": 1,
        "name": "connectors__granola__FETCH_NOTES",
        "response": "literal result",
    }
    assert out["results"][2]["error"]["code"] == "TOOL_NOT_ALLOWED"


def test_execute_does_not_fallback_when_primary_candidate_succeeds():
    class FakeClient:
        def __init__(self):
            self.calls = []

        def execute(self, planned):
            self.calls.append(tuple(planned))
            return [{"data": "sent", "error": None}]

    client = FakeClient()
    out = json.loads(
        dispatch_calls(
            [{"name": "connectors__gmail__SEND_EMAIL", "arguments": {}}],
            local_dispatch=lambda n, a: (True, "{}"),
            availability=lambda: True,
            client_factory=lambda: client,
        )
    )

    assert len(client.calls) == 1
    assert client.calls[0][0].tool == "GMAIL_SEND_EMAIL"
    assert out["results"][0]["response"] == "sent"


def test_execute_fallback_failure_degrades_only_the_retried_entry():
    class FakeClient:
        def __init__(self):
            self.call_count = 0

        def execute(self, planned):
            self.call_count += 1
            if self.call_count == 1:
                return [
                    {"data": "sibling", "error": None},
                    {
                        "data": None,
                        "error": {"code": "TOOL_NOT_FOUND", "message": "missing"},
                    },
                ]
            raise RuntimeError("fallback transport failed")

    client = FakeClient()
    out = json.loads(
        dispatch_calls(
            [
                {"name": "connectors__gmail__SEND_EMAIL", "arguments": {}},
                {"name": "connectors__granola__FETCH_NOTES", "arguments": {}},
            ],
            local_dispatch=lambda n, a: (True, "{}"),
            availability=lambda: True,
            client_factory=lambda: client,
        )
    )

    assert client.call_count == 2
    assert out["results"][0]["response"] == "sibling"
    assert out["results"][1]["error"]["code"] == "PROVIDER_ERROR"


# ---------------------------------------------------------------------------
# pre_dispatch argument rewrites (policy sanitization/redaction) reach the WIRE
#
# A pre_tool_call `modify` directive is a policy rewrite; the single-entry
# deferred path applies it before dispatch. These assert on the outgoing
# gateway request body, not on bridge internals: a rewrite that stops at the
# PlannedCall and never reaches the transport is exactly the bug.
# ---------------------------------------------------------------------------


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


def test_pre_dispatch_rewrite_reaches_the_gateway_request_body():
    transport = _RecordingTransport()

    def gate(name, args):
        # A redaction pass: the secret never leaves the process.
        return None, {**args, "body": "[REDACTED]"}

    out = json.loads(
        dispatch_calls(
            [{"name": "connectors__gmail__SEND_EMAIL",
              "arguments": {"to": "x@example.com", "body": "sk-live-secret"}}],
            local_dispatch=lambda n, a: (True, "{}"),
            pre_dispatch=gate,
            availability=lambda: True,
            client_factory=_recording_client_factory(transport),
        )
    )
    assert _sent_tools(transport) == [
        {
            "connector": "gmail",
            "tool": "GMAIL_SEND_EMAIL",
            "arguments": {"to": "x@example.com", "body": "[REDACTED]"},
        }
    ]
    assert "sk-live-secret" not in json.dumps(transport.requests[0]["json"])
    assert out["results"][0]["response"] == "ok"  # correlation survives the rebuild


def test_pre_dispatch_rewrite_does_not_leak_to_sibling_entries():
    transport = _RecordingTransport()

    def gate(name, args):
        if "slack" in name:
            return None, {**args, "channel": "#safe"}
        return None, None  # explicit "no change" for the sibling

    out = json.loads(
        dispatch_calls(
            [
                {"name": "connectors__gmail__SEND_EMAIL", "arguments": {"to": "keep-me"}},
                {"name": "connectors__slack__POST_MESSAGE", "arguments": {"channel": "#raw"}},
            ],
            local_dispatch=lambda n, a: (True, "{}"),
            pre_dispatch=gate,
            availability=lambda: True,
            client_factory=_recording_client_factory(transport),
        )
    )
    assert _sent_tools(transport) == [
        {"connector": "gmail", "tool": "GMAIL_SEND_EMAIL", "arguments": {"to": "keep-me"}},
        {"connector": "slack", "tool": "SLACK_POST_MESSAGE", "arguments": {"channel": "#safe"}},
    ]
    assert [e["response"] for e in out["results"]] == ["ok", "ok"]


def test_pre_dispatch_block_keeps_the_entry_off_the_wire_entirely():
    transport = _RecordingTransport()

    out = json.loads(
        dispatch_calls(
            [
                {"name": "connectors__gmail__SEND_EMAIL", "arguments": {"to": "blocked"}},
                {"name": "connectors__slack__POST_MESSAGE", "arguments": {"channel": "#ok"}},
            ],
            local_dispatch=lambda n, a: (True, "{}"),
            # A block wins even when the same pass also produced a rewrite.
            pre_dispatch=lambda name, args: (
                ("blocked by policy", {"to": "rewritten"}) if "gmail" in name else (None, None)
            ),
            availability=lambda: True,
            client_factory=_recording_client_factory(transport),
        )
    )
    assert _sent_tools(transport) == [
        {"connector": "slack", "tool": "SLACK_POST_MESSAGE", "arguments": {"channel": "#ok"}}
    ]
    assert out["results"][0]["error"]["code"] == "USER_DENIED"
    assert out["results"][0]["error"]["message"] == "blocked by policy"
    assert out["results"][1]["response"] == "ok"


def test_pre_dispatch_rewrite_to_empty_dict_is_a_rewrite_not_a_no_op():
    # "No change" is None; an empty dict is a deliberate strip-all rewrite.
    transport = _RecordingTransport()

    dispatch_calls(
        [{"name": "connectors__gmail__SEND_EMAIL", "arguments": {"to": "x"}}],
        local_dispatch=lambda n, a: (True, "{}"),
        pre_dispatch=lambda name, args: (None, {}),
        availability=lambda: True,
        client_factory=_recording_client_factory(transport),
    )
    assert _sent_tools(transport) == [
        {"connector": "gmail", "tool": "GMAIL_SEND_EMAIL", "arguments": {}}
    ]


def test_pre_dispatch_exception_sends_the_original_arguments():
    transport = _RecordingTransport()

    def exploding_gate(name, args):
        raise RuntimeError("hook blew up")

    dispatch_calls(
        [{"name": "connectors__gmail__SEND_EMAIL", "arguments": {"to": "x"}}],
        local_dispatch=lambda n, a: (True, "{}"),
        pre_dispatch=exploding_gate,
        availability=lambda: True,
        client_factory=_recording_client_factory(transport),
    )
    assert _sent_tools(transport) == [
        {"connector": "gmail", "tool": "GMAIL_SEND_EMAIL", "arguments": {"to": "x"}}
    ]


def test_local_tool_error_results_are_counted_as_errors():
    def local_dispatch(name, arguments):
        # ok=False: the dispatcher declares its own refusal.
        return False, json.dumps(
            {"error": f"'{name}' is not available in this session."}
        )

    out = json.loads(
        dispatch_calls(
            [
                {"name": "denied_tool", "arguments": {}},
                {"name": "another_denied", "arguments": {}},
            ],
            local_dispatch=local_dispatch,
        )
    )
    assert all(e["error"]["code"] == "TOOL_ERROR" for e in out["results"])
    assert all("not available" in e["error"]["message"] for e in out["results"])
    assert out["error_count"] == 2 and out["success_count"] == 0


def test_local_refusal_extras_survive_into_the_error_slot():
    # tool_error takes arbitrary extras; an allow-list of ("parameters",
    # "hint") silently dropped everything else, so the model never saw them.
    def local_dispatch(name, arguments):
        return False, json.dumps(
            {
                "error": "upstream said no",
                "code": 404,
                "parameters": {"title": "string"},
                "hint": "pass a title",
                "retry_after": 30,
            }
        )

    out = json.loads(
        dispatch_calls([{"name": "denied_tool", "arguments": {}}], local_dispatch=local_dispatch)
    )
    slot = out["results"][0]["error"]
    assert slot["code"] == 404  # caller-supplied code is NOT overwritten
    assert slot["message"] == "upstream said no"
    assert slot["parameters"] == {"title": "string"}
    assert slot["hint"] == "pass a title"
    assert slot["retry_after"] == 30  # the key an allow-list would have eaten


def test_successful_result_carrying_an_error_field_is_not_misclassified():
    # A legitimate result may report per-item errors under "error". Sniffing
    # for that key filed the whole call as a failure and threw the payload away.
    def local_dispatch(name, arguments):
        return True, json.dumps(
            {"error": "2 of 5 rows rejected", "rows": [1, 2, 3], "written": 3}
        )

    out = json.loads(
        dispatch_calls([{"name": "bulk_write", "arguments": {}}], local_dispatch=local_dispatch)
    )
    entry = out["results"][0]
    assert "error" not in entry
    assert entry["response"] == {
        "error": "2 of 5 rows rejected",
        "rows": [1, 2, 3],
        "written": 3,
    }
    assert out["success_count"] == 1 and out["error_count"] == 0


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

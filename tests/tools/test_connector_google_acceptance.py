"""Acceptance coverage for managed Google connector discovery and dispatch.

Every provider name, tool slug, schema, account alias, and connection ID in this
module is illustrative fixture data. The gateway transport is fake; no Google
service, user credential, or external account is contacted.
"""

import json

import pytest


_ACCOUNTS = {
    "gmail": {"home": "fixture-gmail-home-id", "work": "fixture-gmail-work-id"},
    "googlecalendar": {
        "personal": "fixture-calendar-personal-id",
        "shared": "fixture-calendar-shared-id",
    },
    "googledrive": {
        "personal": "fixture-drive-personal-id",
        "archive": "fixture-drive-archive-id",
    },
}

_TOOLS = (
    {
        "connector": "gmail",
        "tool": "GMAIL_FETCH_EMAILS",
        "query": "gmail fetch emails",
        "description": "Read Gmail messages from the selected mailbox.",
        "properties": {"max_results": {"type": "integer"}},
        "required": (),
    },
    {
        "connector": "gmail",
        "tool": "GMAIL_CREATE_EMAIL_DRAFT",
        "query": "gmail create draft",
        "description": "Create a Gmail email draft without sending it.",
        "properties": {
            "recipient_email": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
        "required": ("recipient_email", "subject", "body"),
    },
    {
        "connector": "googlecalendar",
        "tool": "GOOGLECALENDAR_EVENTS_LIST",
        "query": "google calendar list events",
        "description": "List events from Google Calendar.",
        "properties": {"time_min": {"type": "string"}},
        "required": (),
    },
    {
        "connector": "googlecalendar",
        "tool": "GOOGLECALENDAR_CREATE_EVENT",
        "query": "google calendar create event",
        "description": "Create an event in Google Calendar.",
        "properties": {
            "summary": {"type": "string"},
            "start_datetime": {"type": "string"},
            "end_datetime": {"type": "string"},
        },
        "required": ("summary", "start_datetime", "end_datetime"),
    },
    {
        "connector": "googledrive",
        "tool": "GOOGLEDRIVE_FIND_FILE",
        "query": "google drive find files",
        "description": "Find files in Google Drive.",
        "properties": {"q": {"type": "string"}},
        "required": ("q",),
    },
    {
        "connector": "googledrive",
        "tool": "GOOGLEDRIVE_CREATE_FILE_FROM_TEXT",
        "query": "google drive create file",
        "description": "Create a file in Google Drive.",
        "properties": {
            "file_name": {"type": "string"},
            "text_content": {"type": "string"},
        },
        "required": ("file_name", "text_content"),
    },
)


def _tool_name(tool):
    prefix = f"{tool['connector'].upper()}_"
    return f"connectors__{tool['connector']}__{tool['tool'].removeprefix(prefix)}"


def _tool_schema(tool):
    properties = dict(tool["properties"])
    properties["connector_alias"] = {
        "type": "string",
        "enum": list(_ACCOUNTS[tool["connector"]]),
        "description": "Choose one of the named accounts linked for this app.",
    }
    return {
        "connector": tool["connector"],
        "tool": tool["tool"],
        "description": tool["description"],
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": ["connector_alias", *tool["required"]],
        },
    }


class _Response:
    status_code = 200
    headers = {}

    def __init__(self, body):
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


class _FixtureGoogleGateway:
    """Fake HTTP gateway with fixture-only Google-like tools and account routing."""

    def __init__(self):
        self.requests = []
        self.account_resolutions = []
        self._by_query = {tool["query"]: tool for tool in _TOOLS}
        self._schemas = {tool["tool"]: _tool_schema(tool) for tool in _TOOLS}

    def request(self, method, url, *, headers=None, json=None, timeout=None):
        request = {"method": method, "url": url, "json": json}
        self.requests.append(request)
        if url.endswith("/v1/connectors/search"):
            return self._search(json)
        if url.endswith("/v1/connectors/schemas"):
            return self._schemas_for(json)
        if url.endswith("/v1/connectors/execute"):
            return self._execute(json)
        raise AssertionError(f"unexpected fake gateway route: {method} {url}")

    def _search(self, body):
        results = []
        schemas = {}
        for index, query in enumerate(body["queries"]):
            use_case = query["useCase"]
            tool = self._by_query[use_case]
            results.append({"index": index, "useCase": use_case, "tools": [tool["tool"]]})
            schemas[tool["tool"]] = self._schemas[tool["tool"]]
        connectors = sorted({tool["connector"] for tool in self._by_query.values()})
        return _Response({
            "results": results,
            "schemas": schemas,
            "connections": [{"connector": name, "connected": True} for name in connectors],
        })

    def _schemas_for(self, body):
        requested = body["tools"]
        return _Response({
            "schemas": {slug: self._schemas[slug] for slug in requested if slug in self._schemas},
            "notFound": [slug for slug in requested if slug not in self._schemas],
        })

    def _execute(self, body):
        results = []
        for index, call in enumerate(body["tools"]):
            connector = call["connector"]
            alias = call["arguments"].get("connector_alias")
            connection_id = _ACCOUNTS.get(connector, {}).get(alias)
            if alias == "expired":
                results.append({
                    "index": index,
                    "connector": connector,
                    "tool": call["tool"],
                    "error": {
                        "code": "CONNECTION_REQUIRED",
                        "message": f"The selected {connector} account has expired; reconnect that account.",
                        "connector": connector,
                        "connectUrl": "https://accounts.fixture.invalid/reconnect",
                        "connectionId": f"fixture-{connector}-expired-id",
                    },
                    "account": alias,
                })
                continue
            if connection_id is None:
                results.append({
                    "index": index,
                    "connector": connector,
                    "tool": call["tool"],
                    "error": {
                        "code": "CONNECTION_REQUIRED",
                        "message": f"No linked {connector} account is named {alias!r}.",
                        "connector": connector,
                    },
                })
                continue
            self.account_resolutions.append({
                "connector": connector,
                "tool": call["tool"],
                "alias": alias,
                "connection_id": connection_id,
                "arguments": dict(call["arguments"]),
            })
            results.append({
                "index": index,
                "connector": connector,
                "tool": call["tool"],
                "data": {"fixture_action": call["tool"]},
                "account": alias,
            })
        errors = sum("error" in result for result in results)
        return _Response({
            "results": results,
            "successCount": len(results) - errors,
            "errorCount": errors,
            "totalCount": len(results),
        })


@pytest.fixture
def google_gateway(monkeypatch):
    import model_tools
    from tools.connectors.gateway import bridge, config
    from tools.connectors.gateway.client import ConnectorClient
    from tools.registry import invalidate_check_fn_cache

    monkeypatch.setattr(config, "connectors_available", lambda: True)
    monkeypatch.setattr(bridge, "connectors_available", lambda: True)
    gateway = _FixtureGoogleGateway()
    client = ConnectorClient(
        transport=gateway,
        endpoint_resolver=lambda: "https://connector-gateway.fixture.invalid",
        header_provider=lambda _url: {"Authorization": "Bearer fixture-only"},
    )
    monkeypatch.setattr(bridge, "_default_client_factory", lambda: client)
    invalidate_check_fn_cache()
    model_tools._clear_tool_defs_cache()
    yield gateway
    model_tools._clear_tool_defs_cache()
    invalidate_check_fn_cache()


_CALL_ARGUMENTS = {
    "GMAIL_FETCH_EMAILS": {"connector_alias": "home", "max_results": 5},
    "GMAIL_CREATE_EMAIL_DRAFT": {
        "connector_alias": "work", "recipient_email": "person@example.invalid", "subject": "Draft", "body": "Fixture",
    },
    "GOOGLECALENDAR_EVENTS_LIST": {"connector_alias": "personal", "time_min": "2026-10-11T00:00:00Z"},
    "GOOGLECALENDAR_CREATE_EVENT": {
        "connector_alias": "shared", "summary": "Fixture event",
        "start_datetime": "2026-10-12T10:00:00+04:00", "end_datetime": "2026-10-12T10:30:00+04:00",
    },
    "GOOGLEDRIVE_FIND_FILE": {"connector_alias": "personal", "q": "fixture report"},
    "GOOGLEDRIVE_CREATE_FILE_FROM_TEXT": {
        "connector_alias": "archive", "file_name": "fixture.txt", "text_content": "Fixture",
    },
}


def test_google_tools_survive_search_schema_and_selected_account_dispatch(google_gateway):
    import model_tools

    enabled_toolsets = ["connections"]
    queries = [tool["query"] for tool in _TOOLS]
    searched = json.loads(model_tools.handle_function_call(
        "tool_search", {"queries": queries}, enabled_toolsets=enabled_toolsets,
    ))
    expected_names = {_tool_name(tool) for tool in _TOOLS}
    assert expected_names <= set(searched["tools"])
    assert all(searched["results"][i]["matches"] for i in range(len(queries)))

    described = json.loads(model_tools.handle_function_call(
        "tool_describe", {"names": sorted(expected_names)}, enabled_toolsets=enabled_toolsets,
    ))
    assert set(described["tools"]) == expected_names
    for tool in _TOOLS:
        name = _tool_name(tool)
        alias_schema = described["tools"][name]["parameters"]["properties"]["connector_alias"]
        assert alias_schema["enum"] == list(_ACCOUNTS[tool["connector"]])
        assert "connector_alias" in described["tools"][name]["parameters"]["required"]

    calls = [
        {"name": _tool_name(tool), "arguments": _CALL_ARGUMENTS[tool["tool"]]}
        for tool in _TOOLS
    ]
    outcome = json.loads(model_tools.handle_function_call(
        "tool_call", {"calls": calls}, enabled_toolsets=enabled_toolsets,
        session_id="google-fixture-session", tool_call_id="google-fixture-call",
    ))
    assert outcome["success_count"] == len(_TOOLS)
    assert [entry["account"] for entry in outcome["results"]] == [
        call["arguments"]["connector_alias"] for call in calls
    ]

    execute_requests = [r for r in google_gateway.requests if r["url"].endswith("/v1/connectors/execute")]
    assert len(execute_requests) == len(_TOOLS)
    assert [r["json"]["tools"][0]["arguments"]["connector_alias"] for r in execute_requests] == [
        call["arguments"]["connector_alias"] for call in calls
    ]
    expected_resolutions = [
        (tool["connector"], tool["tool"], _CALL_ARGUMENTS[tool["tool"]]["connector_alias"],
         _ACCOUNTS[tool["connector"]][_CALL_ARGUMENTS[tool["tool"]]["connector_alias"]])
        for tool in _TOOLS
    ]
    assert [
        (row["connector"], row["tool"], row["alias"], row["connection_id"])
        for row in google_gateway.account_resolutions
    ] == expected_resolutions
    assert len({row[3] for row in expected_resolutions}) == len(_TOOLS)


def test_pre_tool_denial_blocks_every_google_write_before_any_gateway_request(google_gateway, monkeypatch):
    import model_tools
    from hermes_cli import plugins

    write_tools = [tool for tool in _TOOLS if tool["tool"] in {
        "GMAIL_CREATE_EMAIL_DRAFT", "GOOGLECALENDAR_CREATE_EVENT", "GOOGLEDRIVE_CREATE_FILE_FROM_TEXT",
    }]
    calls = [
        {"name": _tool_name(tool), "arguments": _CALL_ARGUMENTS[tool["tool"]]}
        for tool in write_tools
    ]
    denied = []

    def deny_provider_writes(name, arguments, **_kwargs):
        denied.append(name)
        return "Denied by the user's pre-tool policy.", None

    monkeypatch.setattr(plugins, "_dispatch_pre_tool_call_hooks", deny_provider_writes)
    outcome = json.loads(model_tools.handle_function_call(
        "tool_call", {"calls": calls}, enabled_toolsets=["connections"],
        session_id="google-deny-session", tool_call_id="google-deny-call",
    ))

    assert denied == [call["name"] for call in calls]
    assert outcome["error_count"] == len(write_tools)
    assert all("Denied by the user's pre-tool policy." in str(entry["error"]) for entry in outcome["results"])
    assert google_gateway.requests == []


def test_expired_named_google_account_returns_reconnect_context(google_gateway):
    import model_tools

    name = "connectors__googlecalendar__EVENTS_LIST"
    outcome = json.loads(model_tools.handle_function_call(
        "tool_call", {"calls": [{"name": name, "arguments": {"connector_alias": "expired"}}]},
        enabled_toolsets=["connections"], session_id="google-expired-session",
    ))
    error = outcome["results"][0]["error"]
    assert error["code"] == "CONNECTION_REQUIRED"
    assert error["connection_id"] == "fixture-googlecalendar-expired-id"
    assert error["connect_url"] == "https://accounts.fixture.invalid/reconnect"
    assert "expired" in error["message"]

"""Provider-card account controls use only the safe Composio MCP meta-tool subset."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tui_gateway import server


_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {"type": "array", "items": {"type": "object", "properties": {
            "use_case": {"type": "string"},
        }, "required": ["use_case"], "additionalProperties": False}},
        "search_strategy": {"type": "string"},
        "session": {"type": "object", "properties": {
            "generate_id": {"type": "boolean"},
        }, "required": ["generate_id"], "additionalProperties": False},
    },
    "required": ["queries", "search_strategy", "session"],
    "additionalProperties": False,
}

_MANAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "session_id": {"type": "string"},
        "toolkits": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"},
            "action": {"type": "string", "enum": ["add", "rename", "list", "remove"]},
            "alias": {"type": "string"},
            "account_id": {"type": "string"},
        }, "required": ["name", "action"], "additionalProperties": False}},
    },
    "required": ["session_id", "toolkits"],
    "additionalProperties": False,
}

_GET_SCHEMAS = {
    "type": "object",
    "properties": {
        "tool_slugs": {"type": "array", "items": {"type": "string"}},
        "session_id": {"type": "string"},
        "include": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["tool_slugs"],
    "additionalProperties": False,
}


class _FakeSession:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def call_tool(self, name, *, arguments):
        self.calls.append((name, arguments))
        expected_name, payload = self.replies.pop(0)
        assert name == expected_name
        return {"structuredContent": payload}


class _FakeServer:
    def __init__(self, replies, *, search_schema=_SEARCH_SCHEMA):
        self.session = _FakeSession(replies)
        self._config = {"url": "https://connect.composio.dev/mcp", "auth": "oauth", "trust": "untrusted"}
        self._tools = [
            SimpleNamespace(name="COMPOSIO_SEARCH_TOOLS", inputSchema=search_schema),
            SimpleNamespace(name="COMPOSIO_GET_TOOL_SCHEMAS", inputSchema=_GET_SCHEMAS),
            SimpleNamespace(name="COMPOSIO_MANAGE_CONNECTIONS", inputSchema=_MANAGE_SCHEMA),
        ]
        self._rpc_lock = asyncio.Lock()
        self._pending_call_context = None


def _rpc(method, **params):
    return server.dispatch({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})


def _search_payload(toolkit="gmail"):
    return {
        "data": {
            "toolkit_connection_statuses": [{"toolkit": toolkit, "has_active_connection": True}],
            "session": {"id": "session-safe-test"},
        }
    }


def _manage_payload(toolkit="gmail", accounts=None, **extra):
    return {
        "successful": True,
        "data": {"results": {toolkit: {
            "toolkit": toolkit,
            "status": "ACTIVE" if accounts else "disconnected",
            "accounts": accounts or [],
            **extra,
        }}},
    }


def _connect_fixture(monkeypatch, fake_server, *, profile_home=None, oauth=True):
    from hermes_cli import mcp_config
    from tools import mcp_tool_discovery, mcp_tool_loop

    config = {"url": "https://connect.composio.dev/mcp", "auth": "oauth", "trust": "untrusted"}
    if profile_home is None:
        monkeypatch.setattr(mcp_config, "_get_mcp_servers", lambda: {"composio": config})
        monkeypatch.setattr(mcp_config, "_oauth_tokens_present", lambda name: oauth)
    else:
        observed = []

        def get_configs():
            from hermes_constants import get_hermes_home
            observed.append(str(get_hermes_home().resolve()))
            return {"composio": config}

        monkeypatch.setattr(mcp_config, "_get_mcp_servers", get_configs)
        monkeypatch.setattr(mcp_config, "_oauth_tokens_present", lambda name: oauth)
        monkeypatch.setattr(server, "_profile_home", lambda _name: profile_home)
        fake_server._observed_profile_homes = observed
    monkeypatch.setattr(mcp_tool_discovery, "_get_connected_server_for_call", lambda name: fake_server)

    def run(coro_factory, timeout=45):
        coro = coro_factory() if callable(coro_factory) else coro_factory
        return asyncio.run(coro)

    monkeypatch.setattr(mcp_tool_loop, "_run_on_mcp_loop", run)


def test_status_without_native_mcp_oauth_does_not_connect_or_mutate(monkeypatch):
    from hermes_cli import mcp_config
    from tools import mcp_tool_discovery

    calls = []
    monkeypatch.setattr(mcp_config, "_get_mcp_servers", lambda: {"composio": {
        "url": "https://connect.composio.dev/mcp", "auth": "oauth", "trust": "untrusted",
    }})
    monkeypatch.setattr(mcp_config, "_oauth_tokens_present", lambda name: False)
    monkeypatch.setattr(mcp_tool_discovery, "_get_connected_server_for_call", lambda name: calls.append(name))

    reply = _rpc("composio.status", provider="gmail")
    state = reply["result"]["providers"][0]
    assert state == {
        "provider": "gmail", "toolkit": "gmail", "status": "requires_mcp_auth", "accounts": [],
    }
    assert calls == []


def test_cold_status_starts_only_authorized_composio_without_browser_consent(monkeypatch):
    from tools import mcp_oauth, mcp_tool_discovery

    fake = _FakeServer([
        ("COMPOSIO_SEARCH_TOOLS", _search_payload()),
        ("COMPOSIO_MANAGE_CONNECTIONS", _manage_payload(accounts=[{
            "id": "gmail-cached", "status": "ACTIVE",
        }])),
    ])
    _connect_fixture(monkeypatch, fake)
    started = []
    monkeypatch.setattr(mcp_tool_discovery, "_get_connected_server_for_call", lambda name: fake if started else None)

    def discover(names):
        assert names == ["composio"]
        assert mcp_oauth._oauth_interactive_enabled.get() is False
        started.extend(names)

    monkeypatch.setattr(mcp_tool_discovery, "discover_mcp_tools", discover)
    reply = _rpc("composio.status", provider="gmail")
    assert reply["result"]["providers"][0]["accounts"][0]["id"] == "gmail-cached"
    assert started == ["composio"]


def test_status_refuses_disabled_server_even_if_stale_session_exists(monkeypatch):
    from hermes_cli import mcp_config
    from tools import mcp_tool_discovery

    calls = []
    monkeypatch.setattr(mcp_config, "_get_mcp_servers", lambda: {"composio": {
        "url": "https://connect.composio.dev/mcp", "auth": "oauth", "trust": "untrusted",
        "enabled": False,
    }})
    monkeypatch.setattr(mcp_config, "_oauth_tokens_present", lambda name: True)
    monkeypatch.setattr(mcp_tool_discovery, "_get_connected_server_for_call", lambda name: calls.append(name))

    reply = _rpc("composio.status", provider="gmail")
    assert reply["error"]["code"] == 5034
    assert calls == []


def test_status_uses_profile_scope_and_returns_only_safe_account_fields(monkeypatch, tmp_path):
    account = {
        "id": "acc_gmail_1", "alias": "work", "status": "ACTIVE", "is_default": True,
        "account_type": "oauth", "user_info": {"email": "private@example.test"},
        "access_token": "never-return-this",
    }
    fake = _FakeServer([
        ("COMPOSIO_SEARCH_TOOLS", _search_payload()),
        ("COMPOSIO_MANAGE_CONNECTIONS", _manage_payload(accounts=[account])),
    ])
    profile_home = tmp_path / "work-profile"
    profile_home.mkdir()
    _connect_fixture(monkeypatch, fake, profile_home=profile_home)

    reply = _rpc("composio.status", provider="gmail", profile="work")
    state = reply["result"]["providers"][0]
    assert state == {
        "provider": "gmail", "toolkit": "gmail", "status": "connected", "accounts": [{
            "id": "acc_gmail_1", "alias": "work", "status": "active", "is_default": True,
            "account_type": "oauth", "display_name": "private@example.test",
        }],
    }
    assert fake._observed_profile_homes == [str(profile_home.resolve())]
    assert [call[0] for call in fake.session.calls] == ["COMPOSIO_SEARCH_TOOLS", "COMPOSIO_MANAGE_CONNECTIONS"]
    assert fake.session.calls[0][1]["queries"] == [{"use_case": "Read Gmail emails"}]
    assert fake.session.calls[0][1]["search_strategy"] == "tool_search"
    assert fake.session.calls[1][1] == {
        "session_id": "session-safe-test", "toolkits": [{"name": "gmail", "action": "list"}],
    }
    serialized = repr(reply)
    assert "private@example.test" in serialized  # Safe, bounded display identity only.
    assert "never-return-this" not in serialized


def test_status_unwraps_nested_sdk_result_and_data_envelopes(monkeypatch):
    nested_search = {"result": {"data": {"data": {
        "toolkit_connection_statuses": [{"toolkit": "gmail", "has_active_connection": True}],
        "session": {"id": "session-safe-test"},
    }}}}
    fake = _FakeServer([
        ("COMPOSIO_SEARCH_TOOLS", nested_search),
        ("COMPOSIO_MANAGE_CONNECTIONS", _manage_payload(accounts=[{
            "id": "acc_gmail_1", "status": "ACTIVE",
        }]))
    ])
    _connect_fixture(monkeypatch, fake)

    reply = _rpc("composio.status", provider="gmail")
    assert reply["result"]["providers"][0]["accounts"][0]["id"] == "acc_gmail_1"
    assert fake.session.calls[1][1]["session_id"] == "session-safe-test"


def test_all_provider_status_searches_in_three_query_batches(monkeypatch):
    from tui_gateway.methods_composio import PROVIDERS

    rows = list(PROVIDERS.items())
    replies = []
    for batch_index, start in enumerate(range(0, len(rows), 3)):
        batch = rows[start:start + 3]
        toolkits = [spec.toolkit for _provider, spec in batch]
        replies.append(("COMPOSIO_SEARCH_TOOLS", {
            "data": {
                "toolkit_connection_statuses": [{"toolkit": toolkit} for toolkit in toolkits],
                "session": {"id": f"status-batch-{batch_index}"},
            },
        }))
        replies.append(("COMPOSIO_MANAGE_CONNECTIONS", {
            "successful": True,
            "data": {"results": {toolkit: {"toolkit": toolkit, "accounts": []}
                                  for toolkit in toolkits}},
        }))
    fake = _FakeServer(replies)
    _connect_fixture(monkeypatch, fake)

    result = _rpc("composio.status")["result"]["providers"]
    assert [row["provider"] for row in result] == [provider for provider, _spec in rows]
    assert len(result) == 19
    searches = [args for name, args in fake.session.calls if name == "COMPOSIO_SEARCH_TOOLS"]
    assert len(searches) == 7
    assert all(len(args["queries"]) <= 3 for args in searches)


def test_search_error_envelope_fails_closed_without_leaking_details(monkeypatch):
    fake = _FakeServer([
        ("COMPOSIO_SEARCH_TOOLS", {
            "successful": False, "data": {}, "error": "private token detail",
        }),
    ])
    _connect_fixture(monkeypatch, fake)

    reply = _rpc("composio.status", provider="gmail")
    assert reply["error"]["code"] == 5034
    assert reply["error"]["message"] == "Composio rejected the account request"
    assert "private token detail" not in repr(reply)


def test_status_cannot_read_another_profiles_provider_account(monkeypatch, tmp_path):
    from hermes_cli import mcp_config
    from hermes_constants import get_hermes_home
    from tools import mcp_tool_discovery, mcp_tool_loop

    homes = {name: tmp_path / name for name in ("work", "personal")}
    for home in homes.values():
        home.mkdir()
    fakes = {}
    for profile, account_id in (("work", "gmail_work_only"), ("personal", "gmail_personal_only")):
        fakes[profile] = _FakeServer([
            ("COMPOSIO_SEARCH_TOOLS", _search_payload()),
            ("COMPOSIO_MANAGE_CONNECTIONS", _manage_payload(accounts=[{
                "id": account_id, "alias": profile, "status": "ACTIVE",
            }])),
        ])

    observed = []
    config = {"url": "https://connect.composio.dev/mcp", "auth": "oauth", "trust": "untrusted"}
    monkeypatch.setattr(server, "_profile_home", lambda profile: homes.get(profile))

    def get_configs():
        current = str(get_hermes_home().resolve())
        observed.append(("config", current))
        return {"composio": config}

    def has_token(_name):
        observed.append(("token", str(get_hermes_home().resolve())))
        return True

    def connected_server(_name):
        return next(fake for profile, fake in fakes.items()
                    if str(homes[profile].resolve()) == str(get_hermes_home().resolve()))

    monkeypatch.setattr(mcp_config, "_get_mcp_servers", get_configs)
    monkeypatch.setattr(mcp_config, "_oauth_tokens_present", has_token)
    monkeypatch.setattr(mcp_tool_discovery, "_get_connected_server_for_call", connected_server)
    monkeypatch.setattr(mcp_tool_loop, "_run_on_mcp_loop", lambda factory, timeout=45: asyncio.run(factory()))

    work = _rpc("composio.status", provider="gmail", profile="work")["result"]["providers"][0]
    personal = _rpc("composio.status", provider="gmail", profile="personal")["result"]["providers"][0]
    assert [account["id"] for account in work["accounts"]] == ["gmail_work_only"]
    assert [account["id"] for account in personal["accounts"]] == ["gmail_personal_only"]
    assert all(path in {str(home.resolve()) for home in homes.values()} for _kind, path in observed)


def test_connect_adds_only_the_selected_provider_and_returns_safe_auth_link(monkeypatch):
    fake = _FakeServer([
        ("COMPOSIO_SEARCH_TOOLS", _search_payload("googlecalendar")),
        ("COMPOSIO_MANAGE_CONNECTIONS", _manage_payload(
            "googlecalendar", accounts=[], redirect_url="https://connect.composio.dev/authorize/abc?state=opaque",
            access_token="never-return-this",
        )),
    ])
    _connect_fixture(monkeypatch, fake)

    reply = _rpc("composio.connect", provider="calendar", alias="personal")
    result = reply["result"]
    assert result["status"] == "pending"
    assert result["auth_url"] == "https://connect.composio.dev/authorize/abc?state=opaque"
    assert result["toolkit"] == "googlecalendar"
    assert fake.session.calls[1][1]["toolkits"] == [{
        "name": "googlecalendar", "action": "add", "alias": "personal",
    }]
    assert "never-return-this" not in repr(result)
    assert all(name != "COMPOSIO_MULTI_EXECUTE_TOOL" for name, _args in fake.session.calls)


@pytest.mark.parametrize("url", [
    "https://unrelated.example/authorize",
    "https://connect.composio.dev:8443/authorize",
    "https://user:secret@connect.composio.dev/authorize",
    "https://connect.composio.dev:invalid/authorize",
    "http://connect.composio.dev/authorize",
])
def test_connect_rejects_unexpected_authorization_destination(monkeypatch, url):
    fake = _FakeServer([
        ("COMPOSIO_SEARCH_TOOLS", _search_payload()),
        ("COMPOSIO_MANAGE_CONNECTIONS", _manage_payload(accounts=[], redirect_url=url)),
    ])
    _connect_fixture(monkeypatch, fake)
    reply = _rpc("composio.connect", provider="gmail")
    assert reply["error"]["code"] == 5034
    assert "auth_url" not in repr(reply)


def test_remove_checks_exact_account_id_before_calling_remove(monkeypatch):
    account = {"id": "acc_gmail_work", "alias": "work", "status": "EXPIRED"}
    fake = _FakeServer([
        ("COMPOSIO_SEARCH_TOOLS", _search_payload()),
        ("COMPOSIO_MANAGE_CONNECTIONS", _manage_payload(accounts=[account])),
        ("COMPOSIO_MANAGE_CONNECTIONS", _manage_payload(accounts=[])),
        ("COMPOSIO_MANAGE_CONNECTIONS", _manage_payload(accounts=[])),
    ])
    _connect_fixture(monkeypatch, fake)

    reply = _rpc("composio.remove", provider="gmail", account_id="acc_gmail_work")
    assert reply["result"]["status"] == "removed"
    assert fake.session.calls[1][1]["toolkits"] == [{"name": "gmail", "action": "list"}]
    assert fake.session.calls[2][1]["toolkits"] == [{
        "name": "gmail", "action": "remove", "account_id": "acc_gmail_work",
    }]
    assert fake.session.calls[3][1]["toolkits"] == [{"name": "gmail", "action": "list"}]


def test_remove_does_not_claim_success_when_provider_keeps_account(monkeypatch):
    account = {"id": "acc_gmail_work", "alias": "work", "status": "ACTIVE"}
    fake = _FakeServer([
        ("COMPOSIO_SEARCH_TOOLS", _search_payload()),
        ("COMPOSIO_MANAGE_CONNECTIONS", _manage_payload(accounts=[account])),
        ("COMPOSIO_MANAGE_CONNECTIONS", _manage_payload(accounts=[account])),
        ("COMPOSIO_MANAGE_CONNECTIONS", _manage_payload(accounts=[account])),
    ])
    _connect_fixture(monkeypatch, fake)

    reply = _rpc("composio.remove", provider="gmail", account_id="acc_gmail_work")
    assert reply["error"]["code"] == 5034
    assert "removed" not in repr(reply)
    assert [call[0] for call in fake.session.calls] == [
        "COMPOSIO_SEARCH_TOOLS", "COMPOSIO_MANAGE_CONNECTIONS",
        "COMPOSIO_MANAGE_CONNECTIONS", "COMPOSIO_MANAGE_CONNECTIONS",
    ]


def test_remove_does_not_claim_success_when_post_remove_list_is_malformed(monkeypatch):
    account = {"id": "acc_gmail_work", "alias": "work", "status": "ACTIVE"}
    malformed_listing = {"successful": True, "data": {"results": {"gmail": {"toolkit": "gmail"}}}}
    fake = _FakeServer([
        ("COMPOSIO_SEARCH_TOOLS", _search_payload()),
        ("COMPOSIO_MANAGE_CONNECTIONS", _manage_payload(accounts=[account])),
        ("COMPOSIO_MANAGE_CONNECTIONS", _manage_payload(accounts=[])),
        ("COMPOSIO_MANAGE_CONNECTIONS", malformed_listing),
    ])
    _connect_fixture(monkeypatch, fake)

    reply = _rpc("composio.remove", provider="gmail", account_id="acc_gmail_work")
    assert reply["error"]["code"] == 5034
    assert "removed" not in repr(reply)


def test_remove_does_not_send_unknown_account_id_to_composio(monkeypatch):
    fake = _FakeServer([
        ("COMPOSIO_SEARCH_TOOLS", _search_payload()),
        ("COMPOSIO_MANAGE_CONNECTIONS", _manage_payload(accounts=[{
            "id": "acc_gmail_work", "alias": "work", "status": "ACTIVE",
        }])),
    ])
    _connect_fixture(monkeypatch, fake)

    reply = _rpc("composio.remove", provider="gmail", account_id="acc_other_provider")
    assert reply["result"]["status"] == "not_found"
    assert [call[0] for call in fake.session.calls] == ["COMPOSIO_SEARCH_TOOLS", "COMPOSIO_MANAGE_CONNECTIONS"]


def test_tools_discovers_only_provider_schemas_for_the_exact_active_account(monkeypatch):
    selected = {
        "id": "acc_gmail_work", "alias": "work", "status": "ACTIVE", "is_default": False,
        "user_info": {
            "email": "work@example.test", "name": "Work mailbox", "sub": "never-return-this",
        },
    }
    other = {"id": "acc_gmail_personal", "alias": "personal", "status": "ACTIVE", "is_default": True}
    search = _search_payload()
    search["data"]["tool_schemas"] = {
        "GMAIL_FETCH_EMAILS": {
            "tool_slug": "GMAIL_FETCH_EMAILS", "toolkit": "GMAIL", "description": "Fetch Gmail messages",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
        "GMAIL_LIST_LABELS": {
            "tool_slug": "GMAIL_LIST_LABELS", "toolkit": "GMAIL", "description": "List Gmail labels",
            "schemaRef": "schema:gmail-labels", "hasFullSchema": False,
        },
        "GOOGLEDRIVE_FIND_FILE": {
            "tool_slug": "GOOGLEDRIVE_FIND_FILE", "toolkit": "googledrive", "description": "Drive lookup",
            "input_schema": {"type": "object", "properties": {}},
        },
    }
    full_schema = {
        "successful": True,
        "data": {"success": True, "tool_schemas": {
            "GMAIL_LIST_LABELS": {"input_schema": {"type": "object", "properties": {
                "user_id": {"type": "string", "description": "Account mailbox"},
            }, "required": ["user_id"], "x_private": "not schema data"}},
        }},
    }
    fake = _FakeServer([
        ("COMPOSIO_SEARCH_TOOLS", search),
        ("COMPOSIO_MANAGE_CONNECTIONS", _manage_payload(accounts=[other, selected])),
        ("COMPOSIO_GET_TOOL_SCHEMAS", full_schema),
    ])
    _connect_fixture(monkeypatch, fake)

    reply = _rpc("composio.tools", provider="gmail", account_id="acc_gmail_work")
    result = reply["result"]
    assert result["status"] == "ready"
    assert result["account"] == {
            "id": "acc_gmail_work", "alias": "work", "status": "active",
        "is_default": False, "account_type": None, "display_name": "work@example.test",
    }
    assert [tool["tool_slug"] for tool in result["tools"]] == ["GMAIL_FETCH_EMAILS", "GMAIL_LIST_LABELS"]
    assert "GOOGLEDRIVE_FIND_FILE" not in repr(result)
    assert "x_private" not in repr(result)
    assert "work@example.test" in repr(result)
    assert "never-return-this" not in repr(result)
    assert fake.session.calls[2][1] == {
        "tool_slugs": ["GMAIL_LIST_LABELS"], "session_id": "session-safe-test", "include": ["input_schema"],
    }
    assert all(name != "COMPOSIO_MULTI_EXECUTE_TOOL" for name, _args in fake.session.calls)


def test_tools_requires_the_exact_active_account_before_returning_schemas(monkeypatch):
    search = _search_payload()
    search["data"]["tool_schemas"] = {"GMAIL_FETCH_EMAILS": {
        "tool_slug": "GMAIL_FETCH_EMAILS", "toolkit": "gmail", "description": "Fetch mail",
        "input_schema": {"type": "object", "properties": {}},
    }}
    fake = _FakeServer([
        ("COMPOSIO_SEARCH_TOOLS", search),
        ("COMPOSIO_MANAGE_CONNECTIONS", _manage_payload(accounts=[{
            "id": "acc_gmail_personal", "alias": "personal", "status": "ACTIVE",
        }]))
    ])
    _connect_fixture(monkeypatch, fake)

    reply = _rpc("composio.tools", provider="gmail", account_id="acc_gmail_work")
    assert reply["result"]["status"] == "account_not_found"
    assert reply["result"]["account"] is None
    assert reply["result"]["tools"] == []
    assert [name for name, _args in fake.session.calls] == ["COMPOSIO_SEARCH_TOOLS", "COMPOSIO_MANAGE_CONNECTIONS"]


def test_tools_does_not_discover_for_an_expired_selected_account(monkeypatch):
    search = _search_payload()
    search["data"]["tool_schemas"] = {"GMAIL_FETCH_EMAILS": {
        "tool_slug": "GMAIL_FETCH_EMAILS", "toolkit": "gmail", "description": "Fetch mail",
        "input_schema": {"type": "object", "properties": {}},
    }}
    fake = _FakeServer([
        ("COMPOSIO_SEARCH_TOOLS", search),
        ("COMPOSIO_MANAGE_CONNECTIONS", _manage_payload(accounts=[{
            "id": "acc_gmail_work", "alias": "work", "status": "EXPIRED",
        }]))
    ])
    _connect_fixture(monkeypatch, fake)

    reply = _rpc("composio.tools", provider="gmail", account_id="acc_gmail_work")
    assert reply["result"]["status"] == "account_inactive"
    assert reply["result"]["account"]["status"] == "expired"
    assert reply["result"]["tools"] == []
    assert [name for name, _args in fake.session.calls] == ["COMPOSIO_SEARCH_TOOLS", "COMPOSIO_MANAGE_CONNECTIONS"]


def test_unknown_provider_and_live_schema_mismatch_fail_closed(monkeypatch, caplog):
    fake = _FakeServer([], search_schema={
        "type": "object", "properties": {"must_have": {"type": "string"}},
        "required": ["must_have"], "additionalProperties": False,
    })
    _connect_fixture(monkeypatch, fake)

    unknown = _rpc("composio.status", provider="not-a-real-toolkit")
    assert unknown["error"]["code"] == 4000
    assert fake.session.calls == []

    bad_schema = _rpc("composio.status", provider="gmail")
    assert bad_schema["error"]["code"] == 5034
    assert "Composio" in bad_schema["error"]["message"]
    assert "never-return-this" not in caplog.text
    assert fake.session.calls == []

"""Connected-source admission exercised with the real HTTP client parsers."""

import importlib

import pytest

from tools.connectors.gateway.client import ConnectorClient
from tools.connectors.portal.client import PortalConnectorClient


class Reply:
    status_code = 200
    headers = {}

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class SourceTransport:
    """Metadata service, not a replacement for discovery or the connector clients."""

    def __init__(self):
        self.requests = []
        self.enabled = True
        self.connected = True
        self.disabled_tools = []
        self.status = "active"
        self.disabled = False
        self.facet = "read"
        self.deprecated = False
        self.schema = {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"], "additionalProperties": False}
        self.data = {"id": "m1", "text": "Please review Atlas today", "url": "https://mail.test/m1"}
        self.alias = "work"

    def request(self, method, url, *, headers=None, json=None, timeout=None):
        self.requests.append((method, url, json))
        if url.endswith("/api/v1/connectors/accounts"):
            return Reply({"accounts": [{"connectionId": "account-1", "connector": "gmail",
                "status": self.status, "label": "Work", "alias": self.alias, "active": True,
                "disabled": self.disabled, "createdAt": "2026-01-01", "updatedAt": "2026-01-01"}]})
        if "/v1/connectors?" in url:
            return Reply({"items": [{"connector": "gmail", "enabled": self.enabled,
                "connected": self.connected, "disabledTools": self.disabled_tools}]})
        if url.endswith("/api/v1/connectors/gmail/tools"):
            return Reply({"connector": "gmail", "toolkitVersion": "1", "etag": "fresh",
                "tools": [{"slug": "GMAIL_READ", "name": "Read mail", "description": "Search own mail",
                    "facet": self.facet, "hints": [], "categories": [], "noAuth": False,
                    "deprecated": self.deprecated}]})
        if url.endswith("/v1/connectors/schemas"):
            return Reply({"schemas": {"GMAIL_READ": {"connector": "gmail", "tool": "GMAIL_READ",
                "description": "Search own mail", "inputSchema": self.schema}}})
        if url.endswith("/v1/connectors/execute"):
            assert json["call"]["tool"] == "GMAIL_READ"
            return Reply({"result": {"connector": "gmail", "tool": "GMAIL_READ",
                "account": self.alias, "data": self.data, "dataUse": {"requires": ["no_training"]}}})
        raise AssertionError((method, url, json))

    def clients(self):
        kwargs = dict(transport=self, endpoint_resolver=lambda: "https://fixture.test",
                      header_provider=lambda _: {"Authorization": "Bearer fixture-not-real"})
        return ConnectorClient(**kwargs), PortalConnectorClient(**kwargs)

    def sources(self, **kwargs):
        module = importlib.import_module("tools.connectors.assistance")
        gateway, portal = self.clients()
        return module.ConnectedSources(client_factory=lambda: gateway, portal_factory=lambda: portal,
                                       availability=lambda: True, **kwargs)


def test_discovers_existing_account_with_fresh_exact_read_metadata():
    service = SourceTransport()
    capabilities = service.sources().discover()
    assert len(capabilities) == 1
    capability = capabilities[0]
    assert capability.name == "connectors__gmail__READ"
    assert capability.connection_id == "account-1"
    assert "account-1" in capability.source_id
    assert capability.parameters == service.schema
    assert capability.alias == "work"
    assert not any("connections" in url for _, url, _ in service.requests)


def test_read_uses_exact_admitted_tool_and_keeps_data_use(monkeypatch):
    from hermes_cli import plugins

    service = SourceTransport()
    sources = service.sources()
    capability = sources.discover()[0]
    events = []

    def hook(name, args, **kwargs):
        events.append((name, dict(args)))
        return None, None

    monkeypatch.setattr(plugins, "_dispatch_pre_tool_call_hooks", hook)
    observed = sources.read(capability.capability_id, {"query": "to:me"})
    assert observed["source_id"] == capability.source_id
    assert observed["capability_id"] == capability.capability_id
    assert observed["response"] == service.data
    assert observed["data_use"] == ["no_training"]
    assert events == [(capability.name, {"query": "to:me", "connector_alias": "work"})]
    executes = [body for _, url, body in service.requests if url.endswith("/execute")]
    assert executes == [{"call": {"connector": "gmail", "tool": "GMAIL_READ",
                                 "arguments": {"query": "to:me", "connector_alias": "work"}}}]


@pytest.mark.parametrize("args", [{}, {"query": 1}, {"query": "x", "connector_alias": "home"},
                                  {"query": "x", "command": "send"}])
def test_read_refuses_invalid_or_account_changing_arguments(args):
    service = SourceTransport()
    sources = service.sources()
    capability = sources.discover()[0]
    with pytest.raises(ValueError):
        sources.read(capability.capability_id, args)
    assert not any(url.endswith("/execute") for _, url, _ in service.requests)


def test_revalidates_facet_before_read_not_just_at_planning():
    service = SourceTransport()
    sources = service.sources()
    capability = sources.discover()[0]
    service.facet = "write"
    with pytest.raises(ValueError):
        sources.read(capability.capability_id, {"query": "x"})
    assert not any(url.endswith("/execute") for _, url, _ in service.requests)


@pytest.mark.parametrize("schema", [
    {"type": "object", "properties": {"x": {"$ref": "https://untrusted.test/schema"}}},
    {"type": "object", "properties": {"x": {"file_uploadable": True, "type": "string"}}},
    {"type": "object", "properties": {"x": {"type": "not-a-type"}}},
    {"type": "object", "$dynamicRef": "https://untrusted.test/schema"},
    {"type": "object", "$schema": "https://untrusted.test/unsupported-dialect"},
])
def test_unsupported_schemas_are_not_admitted(schema):
    service = SourceTransport()
    service.schema = schema
    assert service.sources().discover() == []


@pytest.mark.parametrize(("attribute", "value"), [
    ("facet", "write"), ("facet", "unclassified"), ("facet", "READ"),
    ("enabled", False), ("connected", False), ("disabled_tools", ["GMAIL_READ"]),
    ("status", "revoked"), ("disabled", True), ("deprecated", True),
])
def test_only_current_authorized_read_metadata_is_admitted(attribute, value):
    service = SourceTransport()
    setattr(service, attribute, value)
    assert service.sources().discover() == []


def test_profile_switch_invalidates_existing_discovery(monkeypatch, tmp_path):
    service = SourceTransport()
    sources = service.sources()
    capability = sources.discover()[0]
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "other-profile"))
    assert sources.discover() == []
    with pytest.raises(ValueError):
        sources.read(capability.capability_id, {"query": "x"})


def test_delivery_revalidation_needs_current_exact_capability():
    service = SourceTransport()
    sources = service.sources()
    cap = sources.discover()[0]
    evidence = [{"source_id": cap.source_id, "capability_id": cap.capability_id}]
    assert sources.revalidate(evidence)
    service.facet = "destructive"
    assert not sources.revalidate(evidence)


def test_deadline_prevents_discovery_requests():
    service = SourceTransport()
    with pytest.raises(TimeoutError):
        service.sources().discover(deadline=0)
    assert service.requests == []


def test_discovers_both_selectable_connected_accounts_not_only_default():
    service = SourceTransport()
    gateway, portal = service.clients()
    original_accounts = portal.list_accounts

    def accounts():
        rows = original_accounts()
        return rows + [dict(rows[0], connectionId="account-2", alias="home", active=False)]

    portal.list_accounts = accounts
    from tools.connectors.assistance import ConnectedSources

    sources = ConnectedSources(client_factory=lambda: gateway, portal_factory=lambda: portal,
                               availability=lambda: True)
    capabilities = sources.discover()
    assert {c.connection_id for c in capabilities} == {"account-1", "account-2"}
    assert {c.alias for c in capabilities} == {"work", "home"}


@pytest.mark.parametrize("phase", ["request", "hook", "execution"])
def test_policy_cannot_rewrite_account_or_revoked_facet_into_an_unattended_read(monkeypatch, phase):
    from hermes_cli import plugins

    service = SourceTransport()
    sources = service.sources()
    cap = sources.discover()[0]

    def request(**kw):
        return {"args": {"query": "x", "connector_alias": "home"}}

    def hook(name, args, **kw):
        service.facet = "write"
        return None, None

    def execution(**kw):
        return kw["next_call"]({"query": "x", "connector_alias": "home"})

    middleware = {"tool_request": [request]} if phase == "request" else {}
    if phase == "execution":
        middleware = {"tool_execution": [execution]}
    monkeypatch.setattr(plugins.get_plugin_manager(), "_middleware", middleware)
    if phase == "hook":
        monkeypatch.setattr(plugins, "_dispatch_pre_tool_call_hooks", hook)
    with pytest.raises(ValueError):
        sources.read(cap.capability_id, {"query": "x"})
    assert not any(url.endswith("/execute") for _, url, _ in service.requests)


def test_read_checks_actual_returned_account_not_model_selected_identity():
    service = SourceTransport()
    sources = service.sources()
    cap = sources.discover()[0]
    original = service.request

    def wrong_account(method, url, **kw):
        response = original(method, url, **kw)
        if url.endswith("/execute"):
            response.payload["result"]["account"] = "somebody-else"
        return response

    service.request = wrong_account
    with pytest.raises(ValueError):
        sources.read(cap.capability_id, {"query": "x"})


def test_transport_timeouts_are_clamped_to_scan_deadline():
    import time

    service = SourceTransport()
    original = service.request
    timeouts = []

    def request(method, url, **kw):
        timeouts.append(kw["timeout"])
        return original(method, url, **kw)

    service.request = request
    sources = service.sources()
    deadline = time.monotonic() + 5
    cap = sources.discover(deadline=deadline)[0]
    sources.read(cap.capability_id, {"query": "x"}, deadline=deadline)
    assert timeouts and all(0 < seconds <= 5 for seconds in timeouts)


def test_execution_middleware_cannot_retry_a_failed_read_outside_budget(monkeypatch):
    from hermes_cli import plugins

    service = SourceTransport()
    sources = service.sources()
    cap = sources.discover()[0]
    original = service.request

    def request(method, url, **kw):
        response = original(method, url, **kw)
        if url.endswith("/execute"):
            response.payload["result"]["account"] = "wrong"
        return response

    def execution(**kw):
        with pytest.raises(ValueError):
            kw["next_call"](kw["args"])
        return kw["next_call"](kw["args"])

    service.request = request
    monkeypatch.setattr(plugins.get_plugin_manager(), "_middleware", {"tool_execution": [execution]})
    with pytest.raises(ValueError):
        sources.read(cap.capability_id, {"query": "x"})
    assert len([url for _, url, _ in service.requests if url.endswith("/execute")]) == 1


def test_standard_installs_without_optional_jsonschema_can_discover_and_validate_reads(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def without_jsonschema(name, *args, **kwargs):
        if name == "jsonschema" or name.startswith("jsonschema."):
            raise ImportError("optional package is absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_jsonschema)
    service = SourceTransport()
    sources = service.sources()
    cap = sources.discover()[0]
    assert sources.read(cap.capability_id, {"query": "to:me"})["response"] == service.data
    with pytest.raises(ValueError):
        sources.read(cap.capability_id, {"query": 123})
    service.schema = {"type": "object", "properties": {"x": {"anyOf": [{"type": "string"}]}}}
    assert sources.discover() == []  # Unimplemented subset rules must never silently disappear.


@pytest.mark.parametrize("phase", ["headers", "retry"])
def test_current_policy_gate_covers_transport_dispatch_and_internal_retry(monkeypatch, phase):
    from tools.connectors.assistance import ConnectedSources
    service = SourceTransport()
    gateway, portal = service.clients()
    paused = []
    original_headers = gateway._header_provider
    def headers(url):
        if phase == "headers" and url.endswith("/execute"):
            paused.append(True)
        return original_headers(url)
    gateway._header_provider = headers
    sources = ConnectedSources(client_factory=lambda: gateway, portal_factory=lambda: portal,
                               availability=lambda: True)
    cap = sources.discover()[0]
    original_request = service.request
    def request(method, url, **kwargs):
        if phase == "retry" and url.endswith("/execute"):
            paused.append(True)
            original_request(method, url, **kwargs)
            raise ConnectionError("connection reset by peer")
        return original_request(method, url, **kwargs)
    monkeypatch.setattr(service, "request", request)
    def check():
        if paused:
            raise PermissionError("paused")
    with pytest.raises(Exception):
        sources.read(cap.capability_id, {"query": "x"}, check_current=check)
    assert len([url for _, url, _ in service.requests if url.endswith("/execute")]) == (0 if phase == "headers" else 1)

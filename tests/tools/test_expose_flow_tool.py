"""Unit tests for the expose_flow / list_flows native tools (mocked httpx)."""

import json

import tools.expose_flow_tool as eft


class _FakeResp:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json


class _FakeClient:
    """Minimal stand-in for httpx.Client used as a context manager."""

    def __init__(self, *, flows=None, flow_data=None, patch_status=200, patch_result=None):
        self._flows = flows or []
        self._flow_data = flow_data  # 单 flow GET 的完整数据(含 nodes);None = 取不到
        self._patch_status = patch_status
        self._patch_result = patch_result or {}
        self.patched = []  # list of (url, json_body)

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def get(self, url, **_kw):
        if url.endswith("/auto_login"):
            return _FakeResp({"access_token": "tok"})
        if url.endswith("/flows/"):  # 列表(带 query params)
            return _FakeResp(self._flows)
        if "/flows/" in url:  # 单 flow GET .../flows/{id}(校验对话流用)
            return _FakeResp(self._flow_data)
        raise AssertionError(f"unexpected GET {url}")

    def patch(self, url, **kw):
        self.patched.append((url, kw.get("json")))
        if self._patch_status >= 400:
            return _FakeResp({}, status_code=self._patch_status)
        result = dict(self._patch_result)
        result.update(kw.get("json") or {})
        result.setdefault("id", url.rsplit("/", 1)[-1])
        return _FakeResp(result)


def _patch_httpx(monkeypatch, client):
    class _FakeHttpx:
        @staticmethod
        def Client(*_a, **_k):  # noqa: N802 - mirrors httpx.Client
            return client

    monkeypatch.setattr(eft, "_load_httpx", lambda: _FakeHttpx)
    monkeypatch.setattr(eft, "_refresh_kari_flows", lambda: "skipped (test)")


def test_list_flows_filters_by_query_and_drops_user_id(monkeypatch):
    flows = [
        {"id": "1", "name": "Alpha", "mcp_enabled": True, "action_name": "alpha"},
        {"id": "2", "name": "Beta", "mcp_enabled": False, "action_name": None},
    ]
    _patch_httpx(monkeypatch, _FakeClient(flows=flows))

    out = json.loads(eft._handle_list_flows({"query": "alph"}))

    assert out["count"] == 1
    entry = out["flows"][0]
    assert entry["id"] == "1"
    # user_id is never returned by the list API → must not be surfaced/relied on.
    assert "user_id" not in entry
    assert "owned" not in entry


def test_expose_by_id_sets_mcp_enabled_and_action(monkeypatch):
    client = _FakeClient(patch_result={"name": "Alpha"})
    _patch_httpx(monkeypatch, client)

    out = json.loads(
        eft._handle_expose_flow(
            {"flow_id": "abc", "action_name": "alpha", "action_description": "does X"}
        )
    )

    assert out["success"] is True
    assert out["mcp_enabled"] is True
    url, body = client.patched[0]
    assert url.endswith("/api/v1/flows/abc")
    assert body == {"mcp_enabled": True, "action_name": "alpha", "action_description": "does X"}


def test_expose_by_name_resolves_id(monkeypatch):
    flows = [{"id": "xyz", "name": "My Flow", "mcp_enabled": False, "action_name": None}]
    client = _FakeClient(flows=flows, patch_result={"name": "My Flow"})
    _patch_httpx(monkeypatch, client)

    out = json.loads(eft._handle_expose_flow({"flow_name": "My Flow"}))

    assert out["success"] is True
    assert client.patched[0][0].endswith("/api/v1/flows/xyz")


def test_disable_clears_action_fields(monkeypatch):
    client = _FakeClient(patch_result={"name": "Alpha"})
    _patch_httpx(monkeypatch, client)

    eft._handle_expose_flow({"flow_id": "abc", "enabled": False})

    _, body = client.patched[0]
    assert body == {"mcp_enabled": False, "action_name": None, "action_description": None}


def test_enabled_string_false_is_treated_as_off(monkeypatch):
    client = _FakeClient(patch_result={"name": "Alpha"})
    _patch_httpx(monkeypatch, client)

    eft._handle_expose_flow({"flow_id": "abc", "enabled": "false"})

    assert client.patched[0][1]["mcp_enabled"] is False


def test_patch_404_reports_readonly(monkeypatch):
    client = _FakeClient(patch_status=404)
    _patch_httpx(monkeypatch, client)

    out = json.loads(eft._handle_expose_flow({"flow_id": "abc"}))

    assert "error" in out
    assert "404" in out["error"]
    assert not client.patched or client.patched[0][1]["mcp_enabled"] is True


def test_missing_flow_identifier_errors(monkeypatch):
    _patch_httpx(monkeypatch, _FakeClient())

    out = json.loads(eft._handle_expose_flow({}))

    assert "error" in out
    assert "flow_id" in out["error"]


def test_unknown_flow_name_errors(monkeypatch):
    _patch_httpx(monkeypatch, _FakeClient(flows=[{"id": "1", "name": "Other"}]))

    out = json.loads(eft._handle_expose_flow({"flow_name": "Nope"}))

    assert "error" in out
    assert "Nope" in out["error"]


def _flow_with(node_types):
    return {"id": "abc", "data": {"nodes": [{"data": {"type": t}} for t in node_types]}}


def test_expose_rejects_non_chat_flow(monkeypatch):
    # 硬约定门控:非「ChatInput + ChatOutput」对话流不能注册成工具 → 拒,且不 PATCH。
    client = _FakeClient(flow_data=_flow_with(["TextInput", "TextOutput"]), patch_result={"name": "Batch"})
    _patch_httpx(monkeypatch, client)

    out = json.loads(eft._handle_expose_flow({"flow_id": "abc"}))

    assert "error" in out
    assert "对话流" in out["error"]
    assert client.patched == []  # 没真去暴露


def test_expose_allows_chat_flow(monkeypatch):
    client = _FakeClient(flow_data=_flow_with(["ChatInput", "ChatOutput"]), patch_result={"name": "QA"})
    _patch_httpx(monkeypatch, client)

    out = json.loads(eft._handle_expose_flow({"flow_id": "abc"}))

    assert out["success"] is True
    assert client.patched[0][1]["mcp_enabled"] is True


def test_disable_skips_chat_flow_gate(monkeypatch):
    # 隐藏(enabled=False)不该被对话流门控拦住,即便 flow 不是对话流也能取消暴露。
    client = _FakeClient(flow_data=_flow_with(["TextInput"]), patch_result={"name": "X"})
    _patch_httpx(monkeypatch, client)

    out = json.loads(eft._handle_expose_flow({"flow_id": "abc", "enabled": False}))

    assert out["success"] is True
    assert client.patched[0][1]["mcp_enabled"] is False

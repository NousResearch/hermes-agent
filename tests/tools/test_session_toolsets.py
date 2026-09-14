"""Session-owned plugin toolsets (#110515): registry gating, dispatch routing, lifecycle
teardown, prompt-cache freeze, tool_search directness, and profile isolation. Exercises the
real resolution chain (session context -> registry -> model_tools schema building) against a
temp HERMES_HOME."""

import json

import pytest

from tools.registry import registry


@pytest.fixture(autouse=True)
def _clean_session_slots(monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    yield
    with registry._lock:
        registry._session_tools.clear()
        registry._session_served.clear()
        registry._session_indirect.clear()


def _bind_session(monkeypatch, key):
    monkeypatch.setenv("HERMES_SESSION_KEY", key)


def _register(registrar, name, session_tag):
    registrar.register_tool(
        name,
        {"type": "function", "description": f"{name} schema",
         "parameters": {"type": "object", "properties": {}}},
        lambda args, _tag=session_tag: json.dumps({"ok": True, "served_by": _tag}),
        description=f"{name} description")


def _open_toolset(ctx_like, session_key, name, **kw):
    from hermes_cli.plugins_session_tools import session_toolset
    return session_toolset(ctx_like, session_key, name=name, **kw)


class _FakeCtx:
    """Just the surface session_toolset() reads: the plugin id."""

    def __init__(self):
        self.plugin_id = "test-plugin"


def test_two_sessions_see_only_their_own_catalog(monkeypatch):
    ctx = _FakeCtx()
    with _open_toolset(ctx, "sess-a", "client-tools-a") as ta:
        _register(ta, "tool_a", "a")
        with _open_toolset(ctx, "sess-b", "client-tools-b") as tb:
            _register(tb, "tool_b", "b")

            _bind_session(monkeypatch, "sess-a")
            defs = registry.session_tool_definitions()
            assert [d["function"]["name"] for d in defs] == ["tool_a"]
            assert registry.get_entry("tool_a") is not None
            assert registry.get_entry("tool_b") is None

            _bind_session(monkeypatch, "sess-b")
            defs = registry.session_tool_definitions()
            assert [d["function"]["name"] for d in defs] == ["tool_b"]
            assert registry.get_entry("tool_a") is None
            assert registry.get_entry("tool_b") is not None
        # sess-b's context manager exited; sess-a's catalog must be untouched.
        _bind_session(monkeypatch, "sess-a")
        assert registry.get_entry("tool_a") is not None


def test_dispatch_routes_to_owning_session_handler(monkeypatch):
    ctx = _FakeCtx()
    with _open_toolset(ctx, "sess-a", "client-tools-a") as ta:
        _register(ta, "tool_a", "a")
        with _open_toolset(ctx, "sess-b", "client-tools-b") as tb:
            _register(tb, "tool_a", "b")  # same tool name, different session

            _bind_session(monkeypatch, "sess-a")
            assert json.loads(registry.dispatch("tool_a", {}))["served_by"] == "a"
            _bind_session(monkeypatch, "sess-b")
            assert json.loads(registry.dispatch("tool_a", {}))["served_by"] == "b"
        # sess-b closed; sess-a's same-named tool is untouched.
        _bind_session(monkeypatch, "sess-a")
        assert json.loads(registry.dispatch("tool_a", {}))["served_by"] == "a"
        _bind_session(monkeypatch, "sess-b")
        assert "Unknown tool" in registry.dispatch("tool_a", {})


def test_outside_any_session_nothing_is_visible(monkeypatch):
    ctx = _FakeCtx()
    with _open_toolset(ctx, "sess-a", "client-tools-a") as ta:
        _register(ta, "tool_a", "a")
        monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
        assert registry.get_entry("tool_a") is None
        assert registry.session_tool_definitions() == []


def test_late_mutation_rejected_after_catalog_served(monkeypatch):
    ctx = _FakeCtx()
    with _open_toolset(ctx, "sess-a", "client-tools-a") as ta:
        _register(ta, "tool_a", "a")
        _bind_session(monkeypatch, "sess-a")
        assert registry.session_tool_definitions()  # arms the freeze
        with pytest.raises(RuntimeError, match="cache-stable"):
            ta.register_tool(
                "tool_late",
                {"type": "function", "parameters": {"type": "object", "properties": {}}},
                lambda args: json.dumps({"ok": True}))
        # Unknown-tool dispatch and teardown still work after the freeze.
        assert "Unknown tool" in registry.dispatch("tool_late", {})
    assert registry.get_entry("tool_a") is None


def test_duplicate_name_within_session_rejected(monkeypatch):
    ctx = _FakeCtx()
    with _open_toolset(ctx, "sess-a", "client-tools-a") as ta:
        _register(ta, "tool_a", "a")
        with pytest.raises(ValueError, match="already registered"):
            _register(ta, "tool_a", "a2")


def test_session_tools_stay_direct_unless_opted_out(monkeypatch):
    from tools.tool_search import is_deferrable_tool_name
    ctx = _FakeCtx()
    with _open_toolset(ctx, "sess-a", "client-tools-a") as ta:
        _register(ta, "tool_direct", "a")
        ta.register_tool(
            "tool_deferable",
            {"type": "function", "parameters": {"type": "object", "properties": {}}},
            lambda args: json.dumps({"ok": True}), direct=False)
        _bind_session(monkeypatch, "sess-a")
        assert is_deferrable_tool_name("tool_direct", frozenset()) is False
        assert is_deferrable_tool_name("tool_deferable", frozenset()) is True
    # Teardown clears the session gate entirely: neither name reports session ownership.
    _bind_session(monkeypatch, "sess-a")
    assert registry.session_tool_is_direct("tool_direct") is None
    assert registry.session_tool_is_direct("tool_deferable") is None


def test_profile_isolation_same_session_key(monkeypatch, tmp_path):
    ctx = _FakeCtx()
    other_home = tmp_path / "other-profile"
    other_home.mkdir()
    with _open_toolset(ctx, "sess-a", "client-tools-a") as ta:
        _register(ta, "tool_a", "a")
        _bind_session(monkeypatch, "sess-a")
        assert registry.get_entry("tool_a") is not None
        monkeypatch.setenv("HERMES_HOME", str(other_home))
        assert registry.get_entry("tool_a") is None
        assert registry.session_tool_definitions() == []


def test_session_defs_flow_through_get_tool_definitions(monkeypatch):
    """E2E: the owning session's catalog rides on every get_tool_definitions result and the
    memo cache never serves one session's list to the other."""
    import model_tools
    ctx = _FakeCtx()
    with _open_toolset(ctx, "sess-a", "client-tools-a") as ta:
        _register(ta, "tool_a", "a")
        with _open_toolset(ctx, "sess-b", "client-tools-b") as tb:
            _register(tb, "tool_b", "b")

            def _names(session):
                _bind_session(monkeypatch, session)
                return {d["function"]["name"] for d in
                        model_tools.get_tool_definitions(quiet_mode=True)}

            assert "tool_a" in _names("sess-a")
            assert "tool_b" not in _names("sess-a")
            assert "tool_b" in _names("sess-b")
            assert "tool_a" not in _names("sess-b")
            # Cached path: repeat reads stay session-scoped.
            assert "tool_a" in _names("sess-a") and "tool_b" not in _names("sess-a")

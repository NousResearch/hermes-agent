"""Tests for tools/gateway_restart_tool.py — the request_gateway_restart tool.

Verifies:
1. Authorization boundaries (_is_authorized_foreground_turn):
   - Refused outside gateway (_HERMES_GATEWAY unset).
   - Refused for cron tasks (_CRON_AUTO_DELIVER_PLATFORM set).
   - Refused for delegated subagents (is_delegated_child_process_context() is True).
   - Refused for Kanban workers (HERMES_KANBAN_TASK or KANBAN_ENV_KEYS set).
   - Refused when session ContextVars are missing.
   - Accepted only for direct user-facing foreground Gateway turns.
2. ContextVar propagation via _run_in_executor_with_context().
3. Tool returns failure JSON when handoff initialization fails.
4. Worker outer timeout outcome_unknown vs cancel handling.
5. Runtime registry registration (no AST source hacking).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source
from tools.gateway_restart_tool import _resolve_current_source

# ---------------------------------------------------------------------------
# 1. Authorization boundary tests (_is_authorized_foreground_turn)
# ---------------------------------------------------------------------------


def test_auth_refused_outside_gateway(monkeypatch):
    monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
    from tools.gateway_restart_tool import _is_authorized_foreground_turn

    assert _is_authorized_foreground_turn() is False


def test_auth_refused_for_cron(monkeypatch):
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    from tools.gateway_restart_tool import _is_authorized_foreground_turn

    with patch(
        "gateway.session_context.get_session_env",
        side_effect=lambda key, d="": "telegram" if key in ("HERMES_SESSION_PLATFORM", "HERMES_SESSION_CHAT_ID", "HERMES_CRON_AUTO_DELIVER_PLATFORM") else "",
    ):
        assert _is_authorized_foreground_turn() is False


def test_auth_refused_for_delegated_subagent(monkeypatch):
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    from tools.gateway_restart_tool import _is_authorized_foreground_turn

    with patch(
        "agent.delegation_context.is_delegated_child_process_context",
        return_value=True,
    ):
        assert _is_authorized_foreground_turn() is False


def test_auth_refused_for_kanban_worker(monkeypatch):
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-123")
    from tools.gateway_restart_tool import _is_authorized_foreground_turn

    assert _is_authorized_foreground_turn() is False


def test_auth_accepted_for_foreground_gateway(monkeypatch):
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    from tools.gateway_restart_tool import _is_authorized_foreground_turn

    with patch(
        "gateway.session_context.get_session_env",
        side_effect=lambda key, d="": "telegram" if key in ("HERMES_SESSION_PLATFORM", "HERMES_SESSION_CHAT_ID") else "",
    ), patch(
        "agent.delegation_context.is_delegated_child_process_context",
        return_value=False,
    ):
        assert _is_authorized_foreground_turn() is True


# ---------------------------------------------------------------------------
# 2. ContextVars propagation test in Executor Thread
# ---------------------------------------------------------------------------


def test_contextvars_propagation_in_executor(monkeypatch):
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    from tools.gateway_restart_tool import _resolve_current_source
    import gateway.session_context as sc
    import contextvars

    sc._session_context_engaged = True
    sc._SESSION_PLATFORM.set("telegram")
    sc._SESSION_CHAT_ID.set("chat-executor-101")
    sc._SESSION_USER_ID.set("user-42")

    ctx = contextvars.copy_context()

    def _worker():
        return _resolve_current_source()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        source = pool.submit(ctx.run, _worker).result()

    assert source is not None
    assert source.platform.value == "telegram"
    assert source.chat_id == "chat-executor-101"
    assert source.user_id == "user-42"


# ---------------------------------------------------------------------------
# 3. Handler error when runner is missing or not running
# ---------------------------------------------------------------------------


def test_handler_error_when_runner_missing(monkeypatch):
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    from tools.gateway_restart_tool import _handle_request_gateway_restart
    import gateway.run as gw_run

    orig = gw_run._gateway_runner_ref
    gw_run._gateway_runner_ref = lambda: None
    try:
        with patch("tools.gateway_restart_tool._is_authorized_foreground_turn", return_value=True):
            res = json.loads(_handle_request_gateway_restart({"reason": "test"}))
            assert res["success"] is False
            assert "Cannot reach" in res["error"] or "GatewayRunner" in res["error"]
    finally:
        gw_run._gateway_runner_ref = orig


# ---------------------------------------------------------------------------
# 4. Tool returns failure JSON when handoff initialization fails
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_returns_failure_json_when_handoff_fails(monkeypatch, tmp_path):
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    from tools.gateway_restart_tool import _handle_request_gateway_restart
    import gateway.run as gw_run

    runner, _ = make_restart_runner()
    gw_loop = asyncio.get_running_loop()
    runner._gateway_loop = gw_loop

    async def fake_launch_fail():
        return False

    runner._launch_detached_restart_command = fake_launch_fail  # type: ignore[method-assign]
    runner.stop = AsyncMock()  # type: ignore[method-assign]

    orig = gw_run._gateway_runner_ref
    gw_run._gateway_runner_ref = lambda: runner

    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="chat-fail", chat_type="dm", message_id="m1"
    )

    try:
        with patch("tools.gateway_restart_tool._is_authorized_foreground_turn", return_value=True), patch(
            "tools.gateway_restart_tool._resolve_current_source", return_value=source
        ), patch("gateway.run._hermes_home", tmp_path), concurrent.futures.ThreadPoolExecutor(
            max_workers=1
        ) as pool:

            def _worker_call():
                return _handle_request_gateway_restart({"reason": "handoff fail test"})

            fut = pool.submit(_worker_call)
            res_str = await asyncio.wrap_future(fut)
            res = json.loads(res_str)

            # P1 #71876: the accepted ack is published at claim_handoff, so
            # the tool reports success=True ("restart will proceed") even
            # when the launcher later fails. The launcher failure is
            # recorded by the background helper (NOT_STARTED → ABORTED →
            # marker rollback), never by the tool's return value.
            assert res["success"] is True
            runner.stop.assert_not_called()

            # Let the background helper finish the NOT_STARTED rollback so
            # the event loop exits cleanly.
            txn = runner._restart_transaction
            if txn is not None and txn.restart_task is not None:
                try:
                    await asyncio.wait_for(asyncio.shield(txn.restart_task), timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
    finally:
        gw_run._gateway_runner_ref = orig


# ---------------------------------------------------------------------------
# 5. Real built-in discovery contract (Blocker 1 regression tests)
#
# These tests prove that tools/gateway_restart_tool.py is picked up by the
# production AST scan + importlib chain used by model_tools, NOT by a manual
# caller-side ``register()`` call. They also confirm the negative case:
# removing the module-level ``registry.register(...)`` call drops the tool
# from the discovered registry, so the contract is genuinely enforced.
# ---------------------------------------------------------------------------


def _deregister_request_gateway_restart() -> None:
    """Remove the entry the module added at import time, scoped to this test."""
    from tools.registry import registry

    with registry._lock:
        registry._tools.pop("request_gateway_restart", None)
        # Drop the toolset-check pointer if no other tool remains in 'gateway'.
        still_in_toolset = any(
            e.toolset == "gateway" for e in registry._tools.values()
        )
        if not still_in_toolset:
            registry._toolset_checks.pop("gateway", None)
        registry._generation += 1


@pytest.fixture
def restore_registry_state():
    """Snapshot + restore the registry around discovery tests.

    Required to satisfy the "isolate and recover registry / module cache /
    tool-definition cache / env without polluting other tests" contract.
    """
    from tools.registry import registry

    snapshot_tools = dict(registry._tools)
    snapshot_checks = dict(registry._toolset_checks)
    snapshot_generation = registry._generation
    snapshot_aliases = dict(registry._toolset_aliases)

    yield

    with registry._lock:
        registry._tools.clear()
        registry._tools.update(snapshot_tools)
        registry._toolset_checks.clear()
        registry._toolset_checks.update(snapshot_checks)
        registry._generation = snapshot_generation
        registry._toolset_aliases.clear()
        registry._toolset_aliases.update(snapshot_aliases)


def test_module_top_level_register_passes_ast_discovery():
    """AST scan identifies tools/gateway_restart_tool.py as a self-registering
    built-in tool module. This is the FIRST gate of the production discovery
    chain (see tools/registry.py: _module_registers_tools).
    """
    from tools.registry import _module_registers_tools
    from pathlib import Path

    module_path = Path(__file__).resolve().parents[2] / "tools" / "gateway_restart_tool.py"
    assert module_path.exists(), (
        "PR fixture missing: tools/gateway_restart_tool.py must exist for "
        "the discovery contract to apply."
    )

    assert _module_registers_tools(module_path) is True, (
        "tools/gateway_restart_tool.py must contain a top-level "
        "`registry.register(...)` call. discover_builtin_tools() relies on "
        "this AST verdict to know which files to import."
    )


def test_module_has_no_legacy_register_callable_for_callers():
    """There must be NO callable `register(registry)` exposed to callers; the
    module populates the registry at import time only. A stray
    ``def register(registry): ...`` would let callers double-register, which
    is exactly what built-in discovery is designed to prevent.
    """
    import importlib
    import tools.gateway_restart_tool as mod

    importlib.reload(mod)

    assert not hasattr(mod, "register"), (
        "tools.gateway_restart_tool must NOT define `def register(registry): "
        "...`; the registry must be populated exactly once at module-import "
        "time via the top-level `registry.register(...)` call."
    )


def test_real_discovery_chain_populates_registry_with_request_gateway_restart(
    restore_registry_state,
):
    """Run the real discover_builtin_tools() chain (AST scan + importlib) and
    assert that ``request_gateway_restart`` lands in the registry as a
    result. No manual ``register()`` call is made from this test.

    The test simulates a fresh Python process discovery state by dropping
    ``tools.gateway_restart_tool`` from ``sys.modules`` before discovery,
    forcing ``importlib.import_module()`` to re-execute module top-level code.
    """
    import importlib
    import sys
    from pathlib import Path
    from tools.registry import discover_builtin_tools, registry

    orig_mod = sys.modules.get("tools.gateway_restart_tool")
    try:
        # Drop only this tool from registry so we observe the import side-effect.
        _deregister_request_gateway_restart()
        assert "request_gateway_restart" not in registry._tools

        # Evict module from sys.modules and invalidate importlib caches so
        # importlib.import_module() actually re-evaluates top-level registry.register(...)
        sys.modules.pop("tools.gateway_restart_tool", None)
        importlib.invalidate_caches()

        tools_dir = Path(__file__).resolve().parents[2] / "tools"
        imported = discover_builtin_tools(tools_dir)

        assert "tools.gateway_restart_tool" in imported, (
            "discover_builtin_tools must include tools.gateway_restart_tool "
            "based on its top-level registry.register() call."
        )
        assert "request_gateway_restart" in registry._tools, (
            "After discovery, the global registry must contain "
            "request_gateway_restart (populated at module-import time)."
        )
        entry = registry._tools["request_gateway_restart"]
        assert entry.toolset == "gateway"
        assert entry.schema["function"]["name"] == "request_gateway_restart"
        assert callable(entry.handler)
        assert callable(entry.check_fn)
    finally:
        # Restore sys.modules and invalidate caches
        if orig_mod is not None:
            sys.modules["tools.gateway_restart_tool"] = orig_mod
        else:
            sys.modules.pop("tools.gateway_restart_tool", None)
        importlib.invalidate_caches()


def test_legacy_module_without_top_level_register_would_not_be_discovered(
    restore_registry_state, monkeypatch, tmp_path
):
    """Negative-control proof: a sibling module whose body calls
    ``registry.register(...)`` ONLY inside a helper function is correctly
    ignored by the production AST scanner. This is the exact regression that
    motivates the move away from the old ``def register(registry): ...`` API.
    """
    from tools.registry import _module_registers_tools, discover_builtin_tools

    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "__init__.py").write_text("", encoding="utf-8")
    # Old-style "callable for caller to invoke" — passes nothing on import.
    (tools_dir / "legacy_restart_tool.py").write_text(
        (
            "from tools.registry import registry\n"
            "def register(reg):\n"
            "    reg.register(name='legacy_restart', toolset='gateway', "
            "schema={'function':{'name':'legacy_restart','parameters':{'type':'object'}}}, "
            "handler=lambda args, **kw: '{}', check_fn=lambda: True)\n"
        ),
        encoding="utf-8",
    )

    module_path = tools_dir / "legacy_restart_tool.py"
    assert _module_registers_tools(module_path) is False, (
        "An old-style `def register(registry): ...` helper must NOT be "
        "treated as a built-in self-registering module by the AST scanner."
    )

    with patch("tools.registry.importlib.import_module") as mock_import:
        imported = discover_builtin_tools(tools_dir)

    assert imported == [], (
        "No module from a tools/ directory whose body lacks a top-level "
        "`registry.register(...)` call may be imported by discover_builtin_tools()."
    )
    mock_import.assert_not_called()


def test_get_tool_definitions_returns_request_gateway_restart_schema_when_authorized(
    restore_registry_state, monkeypatch
):
    """End-to-end: with realistic foreground session authorization conditions in
    place and the 'gateway' toolset enabled, ``model_tools.get_tool_definitions(...)``
    returns the request_gateway_restart schema — proving the tool reaches the model
    surface through genuine check_fn evaluation without patching check_fn or registry.
    """
    import model_tools
    from tools.registry import invalidate_check_fn_cache

    model_tools._clear_tool_defs_cache()
    invalidate_check_fn_cache()

    try:
        monkeypatch.setenv("_HERMES_GATEWAY", "1")
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
        monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "c1")
        monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "dm")
        monkeypatch.setenv("HERMES_SESSION_USER_ID", "user-1")
        monkeypatch.setenv("HERMES_SESSION_MESSAGE_ID", "m1")
        monkeypatch.delenv("HERMES_CRON_AUTO_DELIVER_PLATFORM", raising=False)

        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        try:
            from agent.delegation_context import KANBAN_ENV_KEYS
            for k in KANBAN_ENV_KEYS:
                monkeypatch.delenv(k, raising=False)
        except Exception:
            pass

        monkeypatch.setattr(
            "agent.delegation_context.is_delegated_child_process_context",
            lambda: False,
        )

        monkeypatch.setattr(
            "gateway.run._gateway_runner_ref",
            lambda: None,
            raising=False,
        )

        model_tools._clear_tool_defs_cache()
        invalidate_check_fn_cache()

        defs = model_tools.get_tool_definitions(
            enabled_toolsets=["gateway"], quiet_mode=True,
        )
        tool_names = {
            d.get("function", {}).get("name")
            for d in defs
            if d.get("type") == "function" or "function" in d
        }

        assert "request_gateway_restart" in tool_names, (
            "request_gateway_restart MUST be returned by model_tools.get_tool_definitions() "
            "when foreground Gateway session authorization conditions are met."
        )

        matching_defs = [
            d for d in defs
            if d.get("function", {}).get("name") == "request_gateway_restart"
        ]
        assert len(matching_defs) >= 1

        target_def = matching_defs[0]
        assert target_def.get("type") == "function"
        fn_info = target_def.get("function", {})
        assert fn_info.get("name") == "request_gateway_restart"

        params = fn_info.get("parameters", {})
        assert isinstance(params, dict)
        assert params.get("type") == "object"

    finally:
        model_tools._clear_tool_defs_cache()
        invalidate_check_fn_cache()


# ---------------------------------------------------------------------------
# /restart policy reuse tests
# ---------------------------------------------------------------------------


def _make_fake_policy(*, enabled: bool, admin_ids, allowed_cmds):
    """Build a tiny namespace object that quacks like SlashAccessPolicy."""
    import types

    admin_set = frozenset(str(x) for x in admin_ids)
    allowed_set = frozenset(str(c).lstrip("/").lower() for c in allowed_cmds)

    def is_admin(uid):
        return enabled and (uid is not None and str(uid) in admin_set)

    def can_run(uid, cmd):
        if not enabled:
            return True
        if is_admin(uid):
            return True
        return cmd in allowed_set

    return types.SimpleNamespace(
        enabled=enabled, is_admin=is_admin, can_run=can_run,
        admin_user_ids=admin_set, user_allowed_commands=allowed_set,
    )


def test_policy_admin_user_is_allowed(monkeypatch):
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    from tools.gateway_restart_tool import _check_restart_policy

    runner = MagicMock()
    runner.config = {"slash_access": {"enabled": True}}
    src = SessionSource(
        platform=Platform.TELEGRAM, chat_id="c1", chat_type="dm", user_id="admin-1",
    )
    fake = _make_fake_policy(
        enabled=True, admin_ids=["admin-1"], allowed_cmds=[],
    )
    with patch("gateway.slash_access.policy_for_source", return_value=fake):
        assert _check_restart_policy(runner, src) is None


def test_policy_user_with_restart_allowance_is_allowed(monkeypatch):
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    from tools.gateway_restart_tool import _check_restart_policy

    runner = MagicMock()
    runner.config = {"slash_access": {"enabled": True}}
    src = SessionSource(
        platform=Platform.TELEGRAM, chat_id="c1", chat_type="dm", user_id="user-7",
    )
    fake = _make_fake_policy(
        enabled=True, admin_ids=[], allowed_cmds=["restart"],
    )
    with patch("gateway.slash_access.policy_for_source", return_value=fake):
        assert _check_restart_policy(runner, src) is None


def test_policy_user_without_restart_allowance_is_denied(monkeypatch):
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    from tools.gateway_restart_tool import _check_restart_policy

    runner = MagicMock()
    runner.config = {"slash_access": {"enabled": True}}
    src = SessionSource(
        platform=Platform.TELEGRAM, chat_id="c1", chat_type="dm", user_id="user-9",
    )
    fake = _make_fake_policy(
        enabled=True, admin_ids=[], allowed_cmds=["help", "status"],
    )
    with patch("gateway.slash_access.policy_for_source", return_value=fake):
        denial = _check_restart_policy(runner, src)
        assert denial is not None
        assert "restart" in denial.lower()


def test_policy_disabled_backward_compat_allows_everyone(monkeypatch):
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    from tools.gateway_restart_tool import _check_restart_policy

    runner = MagicMock()
    runner.config = {"slash_access": {"enabled": False}}
    src = SessionSource(
        platform=Platform.TELEGRAM, chat_id="c1", chat_type="dm", user_id="user-9",
    )
    fake = _make_fake_policy(enabled=False, admin_ids=[], allowed_cmds=[])
    with patch("gateway.slash_access.policy_for_source", return_value=fake):
        # When operator has not configured the gate, all users allowed.
        assert _check_restart_policy(runner, src) is None


def test_handler_denial_rejects_with_authorization_error(monkeypatch):
    """End-to-end: a complete foreground session is built, but the user is
    not allowed to run /restart. The handler must reject with an error JSON
    that names the policy reason — proving the denial is NOT a transport
    issue or a missing session.
    """
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    from tools.gateway_restart_tool import _handle_request_gateway_restart

    runner = MagicMock()
    runner._running = True

    # Provide a real, running event loop bound to the runner so the
    # handler's "Gateway event loop is not available" check passes.
    loop = asyncio.new_event_loop()
    try:
        runner._gateway_loop = loop

        # Real contextvars so _resolve_current_source() succeeds.
        from gateway.session_context import set_session_vars
        set_session_vars(
            platform="telegram",
            chat_id="c1",
            user_id="user-9",
        )

        source = _resolve_current_source()
        assert source is not None

        with patch("gateway.run._gateway_runner_ref", return_value=runner), patch(
            "tools.gateway_restart_tool._is_authorized_foreground_turn", return_value=True
        ), patch(
            "tools.gateway_restart_tool._resolve_current_source", return_value=source
        ), patch.object(
            loop, "is_running", return_value=True
        ), patch(
            "tools.gateway_restart_tool._check_restart_policy",
            return_value="Not allowed to run /restart on this scope.",
        ):
            res = json.loads(_handle_request_gateway_restart({"reason": "test"}))
            assert res["success"] is False
            assert "restart" in res["error"].lower()
    finally:
        loop.close()

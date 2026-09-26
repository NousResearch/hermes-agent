"""Regression tests for backend hook-registration parity.

Every long-lived agent runtime must register user-configured shell hooks
and outbound webhooks at startup. The CLI (``_prepare_agent_startup``) and
the messaging gateway (``gateway/run.py``) always did; the TUI gateway
backends and the serve/dashboard backend historically did not, so a
webhook configured in config.yaml fired in ``hermes --cli`` but silently
never fired from ``hermes --tui``, the dashboard chat PTY, the desktop WS
sidecar, or ``hermes serve``.

Verifies:

* ``agent.hook_registration.ensure_hooks_registered`` registers both
  shell hooks and outbound webhooks from the loaded config.
* It is once-per-process (repeat calls are no-ops).
* A config/load failure never breaks backend startup.
* All three backend entry points call it: ``tui_gateway.entry.main``
  (stdio TUI + dashboard chat PTY), ``tui_gateway.ws.handle_ws``
  (dashboard / desktop WS sidecar), and ``web_server._lifespan``
  (serve / dashboard backend).
"""

from __future__ import annotations

import asyncio
import io

import pytest

import agent.hook_registration as hook_registration


@pytest.fixture(autouse=True)
def _fresh_hook_registration_guard():
    """Start every test with a clean once-per-process guard."""
    hook_registration.reset_for_tests()
    yield
    hook_registration.reset_for_tests()


class TestEnsureHooksRegistered:
    def test_registers_both_shell_hooks_and_outbound_webhooks(self, monkeypatch):
        import agent.outbound_webhooks as ow
        import agent.shell_hooks as sh

        calls: dict[str, object] = {}

        def _shell(cfg, *, accept_hooks):
            calls["shell"] = (cfg, accept_hooks)
            return []

        def _outbound(cfg):
            calls["outbound"] = cfg
            return []

        monkeypatch.setattr(sh, "register_from_config", _shell)
        monkeypatch.setattr(ow, "register_from_config", _outbound)

        cfg = {"hooks": {}}
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)

        hook_registration.ensure_hooks_registered()

        # Consent is delegated to the helper's own resolution (env/config),
        # never force-enabled from a backend.
        assert calls["shell"] == (cfg, False)
        assert calls["outbound"] is cfg

    def test_repeat_calls_are_noops(self, monkeypatch):
        import agent.outbound_webhooks as ow
        import agent.shell_hooks as sh

        n = {"shell": 0, "outbound": 0}
        monkeypatch.setattr(
            sh, "register_from_config",
            lambda cfg, *, accept_hooks: n.__setitem__("shell", n["shell"] + 1),
        )
        monkeypatch.setattr(
            ow, "register_from_config", lambda cfg: n.__setitem__("outbound", n["outbound"] + 1)
        )

        hook_registration.ensure_hooks_registered()
        hook_registration.ensure_hooks_registered()
        hook_registration.ensure_hooks_registered()

        assert n == {"shell": 1, "outbound": 1}

    def test_failure_does_not_raise(self, monkeypatch):
        """A broken config read must never take down backend startup."""

        def _boom():
            raise RuntimeError("malformed config")

        monkeypatch.setattr("hermes_cli.config.load_config", _boom)

        # Must not raise.
        hook_registration.ensure_hooks_registered()

    def test_explicit_cfg_skips_config_read(self, monkeypatch):
        """A caller-provided cfg is used as-is (no load_config round trip)."""
        import agent.outbound_webhooks as ow
        import agent.shell_hooks as sh

        seen: dict[str, object] = {}
        monkeypatch.setattr(sh, "register_from_config", lambda cfg, *, accept_hooks: seen.setdefault("shell", cfg))
        monkeypatch.setattr(ow, "register_from_config", lambda cfg: seen.setdefault("outbound", cfg))
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: (_ for _ in ()).throw(AssertionError("load_config must not be called")),
        )

        cfg = {"hooks": {}}
        hook_registration.ensure_hooks_registered(cfg)
        assert seen == {"shell": cfg, "outbound": cfg}


class TestEntryPointWiring:
    """All three backends must call the registration helper at startup —
    the same wiring pattern the heartbeat refresher and orphan sweep
    follow."""

    def _stub_entry_main_common(self, monkeypatch):
        from tui_gateway import entry, server

        monkeypatch.setattr(entry, "_install_sidecar_publisher", lambda: None)
        monkeypatch.setattr(entry, "ensure_mcp_discovery_started", lambda: None)
        monkeypatch.setattr(entry, "resolve_skin", lambda: "default")
        monkeypatch.setattr(entry.server, "_ensure_skin_watcher", lambda: None)
        monkeypatch.setattr(entry.server, "_schedule_startup_orphan_sweep", lambda: None)
        monkeypatch.setattr(entry, "_log_exit", lambda reason: None)
        monkeypatch.setattr(entry, "handle_spurious_eof", lambda *a: False)
        monkeypatch.setattr(entry, "write_json", lambda _payload: True)
        monkeypatch.setattr(entry.sys, "stdin", io.StringIO(""))

        import hermes_cli.model_switch as ms

        monkeypatch.setattr(ms, "prewarm_picker_cache_async", lambda: None)
        return entry, server

    def test_entry_main_registers_hooks(self, monkeypatch):
        entry, server = self._stub_entry_main_common(monkeypatch)

        started = {"n": 0}
        monkeypatch.setattr(
            server, "_register_hooks_from_config",
            lambda: started.__setitem__("n", started["n"] + 1),
        )

        entry.main()
        assert started["n"] == 1

    def test_handle_ws_registers_hooks(self, monkeypatch):
        from tui_gateway import server
        from tui_gateway import ws as ws_mod

        started = {"n": 0}
        monkeypatch.setattr(
            server, "_register_hooks_from_config",
            lambda: started.__setitem__("n", started["n"] + 1),
        )
        monkeypatch.setattr(server, "resolve_skin", lambda: "default")
        monkeypatch.setattr(server, "_ensure_skin_watcher", lambda: None)
        monkeypatch.setattr(server, "register_live_transport", lambda *_a, **_k: None)
        monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)

        class FakeWS:
            async def accept(self):
                pass

            async def send_text(self, line):
                pass

            async def receive_text(self):
                raise ws_mod._WebSocketDisconnect()

            async def close(self):
                pass

        asyncio.run(ws_mod.handle_ws(FakeWS()))
        assert started["n"] == 1

    def test_serve_lifespan_registers_hooks(self, monkeypatch):
        """The dashboard/serve backend registers hooks during startup."""
        import hermes_cli.web_server as web_server_mod

        started = {"n": 0}

        def _ensure(*_a, **_k):
            started["n"] += 1

        # The lifespan does a lazy `from agent.hook_registration import
        # ensure_hooks_registered` — patch the module attribute the import
        # resolves against.
        monkeypatch.setattr(hook_registration, "ensure_hooks_registered", _ensure)
        # Neutralize the other lifespan startup work we don't assert on.
        monkeypatch.setattr(web_server_mod, "_warm_gateway_module", lambda: None)

        from fastapi.testclient import TestClient

        with TestClient(web_server_mod.app, raise_server_exceptions=False):
            pass

        assert started["n"] == 1
"""The TUI/Desktop JSON-RPC model switch refuses when this backend runs pre-update code.

Sibling of the gateway ``/model`` guard and the dashboard Models-page guard (#86207): a
long-lived backend freezes ``sys.modules`` at boot, so after ``hermes update`` replaces the
checkout underneath it, the switch chain's first-time lazy imports resolve a freshly-pulled
module against a stale cached dependency. On the Desktop that surfaced raw in the chat toast
as ``cannot import name 'openrouter_variant_base' from 'hermes_constants'`` — ``hermes_constants``
in ``sys.modules`` came from the boot revision, which predated the symbol. The guard turns it
into the designed "Restart required" message, which the renderer already maps to a one-click
restart (``notifications.errors.codeSkewRestartRequired`` + ``RECOVERY_ACTIONS.restartHermes``).
"""

import builtins
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway import code_skew
from tui_gateway import server


class _Agent:
    def __init__(self):
        self.model, self.provider, self.base_url, self.api_key, self.api_mode = (
            "old", "nous", "", "", "")

    def switch_model(self, **_kw):
        self.model = _kw.get("new_model", self.model)


@pytest.fixture
def _switch_probe(monkeypatch):
    """Fake the switch chain so tests can assert whether it was ever entered."""
    entered: list[dict] = []
    result = SimpleNamespace(
        success=True, new_model="new/model", target_provider="nous", base_url="", api_key="key",
        api_mode="chat_completions", warning_message="", model_info=None, error_message="",
        runtime_capabilities=None)
    monkeypatch.setattr("hermes_cli.model_switch.switch_model",
                        lambda **kw: entered.append(kw) or result)
    monkeypatch.setattr("hermes_cli.model_switch.persist_model_selection", lambda _r: None)
    monkeypatch.setattr("hermes_cli.model_cost_guard.expensive_model_warning", lambda *a, **k: None)
    for name in ("_restart_slash_worker", "_persist_live_session_runtime",
                 "_persist_live_session_system_prompt", "_append_model_switch_marker",
                 "_emit_session_info"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    monkeypatch.setattr(server, "_write_config_key", lambda k, v: None)
    return entered


def _set_skew(monkeypatch, value):
    monkeypatch.setattr(code_skew, "detect_code_skew", lambda: value)


def test_refuses_and_never_enters_the_switch_chain(monkeypatch, _switch_probe):
    _set_skew(monkeypatch, ("abc1234567", "def4567890"))
    monkeypatch.delenv("HERMES_SERVE_HEADLESS", raising=False)

    with pytest.raises(ValueError) as err:
        server._apply_model_switch("sid", {"agent": _Agent()}, "new/model --provider nous")

    message = str(err.value)
    assert message.startswith("Restart required:")
    assert "abc1234567" in message and "def4567890" in message
    assert _switch_probe == [], "the risky lazy-import chain must not be entered"


def test_desktop_owned_backend_names_the_restart_affordance(monkeypatch, _switch_probe):
    _set_skew(monkeypatch, ("abc1234567", "def4567890"))
    monkeypatch.setenv("HERMES_SERVE_HEADLESS", "1")

    with pytest.raises(ValueError) as err:
        server._apply_model_switch("sid", {"agent": _Agent()}, "new/model --provider nous")

    assert "Restart backend in Hermes Desktop" in str(err.value)


def test_switch_runs_normally_without_skew(monkeypatch, _switch_probe):
    _set_skew(monkeypatch, None)

    out = server._apply_model_switch("sid", {"agent": _Agent()}, "new/model --provider nous")

    assert out["value"] == "new/model"
    assert len(_switch_probe) == 1


def test_mid_turn_pick_is_refused_before_it_is_stashed(monkeypatch):
    _set_skew(monkeypatch, ("abc1234567", "def4567890"))
    session = {"running": True}
    parsed = SimpleNamespace(model_input="new/model", explicit_provider="nous")

    with pytest.raises(ValueError, match="Restart required"):
        server._stash_pending_model_switch(42, "model", "new/model --provider nous", session, False, parsed)

    assert "pending_model_switch" not in session


@pytest.mark.parametrize("entry", ["idle", "running", "sessionless-once"])
def test_config_set_refuses_skew_before_lazy_import(monkeypatch, entry):
    _set_skew(monkeypatch, ("abc1234567", "def4567890"))
    agent = _Agent()
    session = None if entry == "sessionless-once" else {"agent": agent, "running": entry == "running"}
    original_session = dict(session) if session is not None else None
    value = "new/model --provider nous" + (" --once" if session is None else "")
    params = {"value": value, "session_id": "sid" if session is not None else ""}
    writes = {}
    for name in ("_stash_pending_model_switch", "_persist_live_session_runtime",
                 "_persist_live_session_system_prompt", "_write_config_key", "_save_cfg"):
        writes[name] = Mock(side_effect=AssertionError("selection must not be saved"))
        monkeypatch.setattr(server, name, writes[name])
    imports = []
    original_import = builtins.__import__

    def forbid_model_switch(name, *args, **kwargs):
        if name == "hermes_cli.model_switch":
            imports.append(name)
            raise ImportError("stale model_switch lazy import reached")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_model_switch)

    response = server._set_model(42, params, "model", value, session)

    assert response["error"]["code"] == 5001
    assert response["error"]["message"].startswith("Restart required:")
    assert "abc1234567" in response["error"]["message"]
    assert "def4567890" in response["error"]["message"]
    assert imports == [], "the config.set entry must guard before even importing model_switch"
    assert session == original_session
    assert agent.model == "old"
    for write in writes.values():
        write.assert_not_called()


def test_stdio_records_revision_before_starting_request_services(monkeypatch):
    from hermes_cli.observability import shared_metrics_process
    from tui_gateway import entry

    fingerprint = "git:main:abc1234567890"
    monkeypatch.setattr(code_skew, "_boot_fingerprint", None)
    monkeypatch.setattr(code_skew, "_fingerprint", lambda: fingerprint)
    monkeypatch.setattr(entry, "_install_sidecar_publisher", lambda: None)
    monkeypatch.setattr(entry, "_close_rpc_stdin_on_exec", lambda: None)
    monkeypatch.setattr(server, "install_tui_message_injector", lambda: None)
    monkeypatch.setattr(server, "_stdio_is_rpc_channel", False)

    def stop_after_boot(_surface):
        assert code_skew._boot_fingerprint == fingerprint
        raise RuntimeError("stop after boot fingerprint")

    monkeypatch.setattr(shared_metrics_process, "begin_process", stop_after_boot)
    with pytest.raises(RuntimeError, match="stop after boot fingerprint"):
        entry._serve()

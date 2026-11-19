"""The slash worker inherits the parent agent's provider and base URL, not the config default.

Regression for #57283 (provider) and #141953 (base URL; the child side is
``test_slash_worker_real_cli_route.py``, against a real ``HermesCLI``).

A Desktop/TUI MoA session pins the live agent to the virtual ``moa`` provider, but the
persistent slash-worker subprocess was spawned with only ``--model``. HermesCLI re-resolves
the provider from config.yaml, so a ``/moa`` one-shot dispatched the MoA PRESET NAME to the
configured real provider (openrouter 402 / "No available channel for model default"). The
worker argv must carry ``--provider`` and the child must hand it to ``HermesCLI``.
"""

from types import SimpleNamespace

import pytest

from tui_gateway import server


class _Popen:
    """Capture the spawn argv without a real child."""

    def __init__(self, argv, **kw):
        self.argv = list(argv)
        self.stdout = self.stderr = iter(())
        self.stdin = None

    def poll(self):
        return 0


def _spawn(monkeypatch, **worker_kwargs) -> list[str]:
    captured = {}

    def _popen(argv, **kw):
        captured["argv"] = list(argv)
        return _Popen(argv, **kw)

    monkeypatch.setattr(server, "spawn_owned", lambda argv, purpose, **kw: _popen(argv, **kw))
    server._SlashWorker("sess", "default", **worker_kwargs)
    return captured["argv"]


def test_slash_worker_argv_carries_provider(monkeypatch):
    argv = _spawn(monkeypatch, provider="moa")

    assert "--provider" in argv
    assert argv[argv.index("--provider") + 1] == "moa"
    # model still forwarded
    assert argv[argv.index("--model") + 1] == "default"


def test_slash_worker_argv_omits_empty_provider(monkeypatch):
    argv = _spawn(monkeypatch, provider=None)

    assert "--provider" not in argv


def test_slash_exec_spawns_with_the_live_agent_provider(monkeypatch):
    """First-use spawn in ``slash.exec`` passes the session agent's resolved provider — the MoA
    one-shot builds ``agent.provider == "moa"`` via ``_apply_model_switch`` before the worker runs."""
    worker_args = None

    class _RecordingWorker:
        def __init__(self, session_key, model, profile_home=None, provider=None, base_url=None):
            nonlocal worker_args
            worker_args = (session_key, model, profile_home, provider, base_url)
            self.stderr_tail = []

        def run(self, command):
            return "ok"

        def close(self):
            pass

    agent = SimpleNamespace(model="default", provider="moa", base_url="https://chatgpt.com/backend-api/codex")
    session = {"session_key": "s-key", "agent": agent, "profile_home": None, "running": False}

    monkeypatch.setattr(server, "_sessions", {"sid": session})
    monkeypatch.setattr(server, "_SlashWorker", _RecordingWorker)
    monkeypatch.setattr(server, "_sess_nowait", lambda params, rid: (session, None))
    monkeypatch.setattr(server, "_live_slash_command_output", lambda *a, **k: None)
    monkeypatch.setattr(server, "_is_snapshot_restore", lambda *a, **k: False)
    monkeypatch.setattr(server, "_mirror_slash_side_effects", lambda *a, **k: None)
    monkeypatch.setattr(server, "_resolve_model", lambda: "fallback-model")

    resp = server._methods["slash.exec"](1, {"session_id": "sid", "command": "/tools"})

    assert "error" not in resp, resp
    assert worker_args is not None
    assert worker_args[3:] == ("moa", "https://chatgpt.com/backend-api/codex")


def test_model_slash_confirms_before_switching_the_live_route(monkeypatch):
    """A mature Desktop chat cannot report success from the separate slash worker before its
    large-context confirmation commits the live agent's provider, model and endpoint."""
    agent = SimpleNamespace(model="qwen", provider="spark", base_url="http://spark/v1")
    session = {"session_key": "established", "agent": agent, "running": False}
    monkeypatch.setattr(server, "_sess_nowait", lambda params, rid: (session, None))
    monkeypatch.setattr(server, "_live_slash_command_output", lambda *a, **k: None)
    calls = []

    def _set_model(rid, params):
        calls.append(params)
        if not params.get("confirm_expensive_model"):
            return {"jsonrpc": "2.0", "id": rid, "result": {
                "key": "model", "value": "gpt-6.2-sol", "confirm_required": True,
                "confirm_message": "Large cached context: confirm the model switch."}}
        agent.model = "gpt-6.2-sol"
        agent.provider = "openai-codex"
        agent.base_url = "https://chatgpt.com/backend-api/codex"
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "key": "model", "value": agent.model, "confirm_required": False}}

    monkeypatch.setitem(server._methods, "config.set", _set_model)
    monkeypatch.setattr(server, "_slash_worker_for", lambda *_: (_ for _ in ()).throw(
        AssertionError("/model must not execute in the separate slash worker")))
    command = "/model gpt-6.2-sol --provider openai-codex --session"
    first = server._methods["slash.exec"](1, {"session_id": "sid", "command": command})
    assert first["result"]["confirm_required"] is True
    assert first["result"]["target_model"] == "gpt-6.2-sol"
    assert "switched" not in first["result"]["output"].lower()
    assert (agent.model, agent.provider, agent.base_url) == ("qwen", "spark", "http://spark/v1")

    confirmed = server._methods["slash.exec"](
        2, {"session_id": "sid", "command": command, "confirm_expensive_model": True})
    assert confirmed["result"]["confirm_required"] is False
    assert "switched" in confirmed["result"]["output"].lower()
    assert (agent.model, agent.provider, agent.base_url) == (
        "gpt-6.2-sol", "openai-codex", "https://chatgpt.com/backend-api/codex")
    assert [call["confirm_expensive_model"] for call in calls] == [False, True]
    assert [call["value"] for call in calls] == [
        "gpt-6.2-sol --provider openai-codex --session"] * 2


@pytest.mark.parametrize("arg", ["reset", "gpt-6.2-sol --global", "gpt-6.2-sol --reasoning high"])
def test_model_slash_passes_selection_flags_to_the_authoritative_setter(monkeypatch, arg):
    session = {"session_key": "established", "agent": object(), "running": False}
    monkeypatch.setattr(server, "_sess_nowait", lambda params, rid: (session, None))
    monkeypatch.setattr(server, "_live_slash_command_output", lambda *a, **k: None)
    seen = []

    def _set_model(rid, params):
        seen.append(params)
        return {"jsonrpc": "2.0", "id": rid, "result": {"value": "selected", "warning": ""}}

    monkeypatch.setitem(server._methods, "config.set", _set_model)
    result = server._methods["slash.exec"](1, {"session_id": "sid", "command": f"/model {arg}"})
    assert "error" not in result, result
    assert seen[0]["key"] == "model"
    assert seen[0]["value"] == arg


def test_slash_worker_child_forwards_provider_to_hermes_cli(monkeypatch):
    """The child's argparse accepts --provider and hands it to HermesCLI (mocked): the MoA
    virtual provider must reach ``HermesCLI(provider=...)`` instead of config re-resolution."""
    import sys as _sys
    import types

    import tui_gateway.slash_worker as sw

    built = {}

    class _FakeCLI:
        def __init__(self, **kwargs):
            built.update(kwargs)

    # main() resolves HermesCLI from ``cli`` at call time, so patch it there.
    monkeypatch.setitem(_sys.modules, "cli", types.SimpleNamespace(HermesCLI=_FakeCLI))
    monkeypatch.setattr(sw, "_prepare_slash_worker_runtime", lambda: None)
    monkeypatch.setattr(sw.sys, "stdin", type("S", (), {"readline": lambda self: ""})())
    monkeypatch.setattr(_sys, "argv", ["slash_worker", "--session-key", "s", "--model", "m",
                                       "--provider", "moa"])

    sw.main()

    assert built.get("provider") == "moa"
    assert built.get("model") == "m"

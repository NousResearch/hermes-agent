import types

import pytest

from tui_gateway import server


@pytest.fixture(autouse=True)
def _neuter_agent_prewarm_timer(request, monkeypatch):
    """Stub the deferred agent pre-warm timer for every test in this module.

    ``session.create`` and non-eager ``session.resume`` fire a 50 ms
    background ``threading.Timer`` (``_schedule_agent_build``) that calls
    whatever ``server._make_agent`` is patched in AT FIRE TIME. Left live,
    a timer armed by one test outlives it and lands in the NEXT test's
    ``_make_agent`` mock, racily corrupting its captured state (the
    ``'tip' == 'cont_tip'`` flakes in the session_resume tests). Tests that
    exercise the deferred build itself opt back in with
    ``@pytest.mark.real_agent_prewarm``.
    """
    if request.node.get_closest_marker("real_agent_prewarm"):
        yield
        return
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
    yield


def _sync_test_session(**extra):
    session = {
        "agent": types.SimpleNamespace(model="old/model"),
        "session_key": "session-key",
    }
    session.update(extra)
    return session


def _patch_config_model(monkeypatch, model, provider=""):
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
    cfg_model = {"default": model}
    if provider:
        cfg_model["provider"] = provider
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": cfg_model})


def test_config_sync_switches_when_only_provider_differs(monkeypatch):
    _patch_config_model(monkeypatch, "old/model", provider="nous")
    session = _sync_test_session(config_model_seen=("old/model", ""))
    calls = []
    monkeypatch.setattr(
        server,
        "_apply_model_switch",
        lambda sid, sess, raw, **kw: calls.append(raw),
    )

    server._sync_agent_model_with_config("sid", session)

    assert calls == ["old/model --provider nous"]


def test_config_sync_failure_emits_error_once_per_edit(monkeypatch):
    _patch_config_model(monkeypatch, "broken/model")
    session = _sync_test_session(config_model_seen=("old/model", ""))

    def boom(*a, **k):
        raise ValueError("no such model")

    monkeypatch.setattr(server, "_apply_model_switch", boom)
    emits = []
    monkeypatch.setattr(
        server, "_emit", lambda ev, sid, payload: emits.append((ev, payload))
    )

    server._sync_agent_model_with_config("sid", session)
    server._sync_agent_model_with_config("sid", session)

    assert len(emits) == 1
    assert emits[0][0] == "error"
    assert "broken/model" in emits[0][1]["message"]


def test_config_sync_config_wins_over_env_seed(monkeypatch):
    # Hosted instances set HERMES_INFERENCE_MODEL as a provision-time seed;
    # the per-turn sync must follow config.yaml edits, not stay pinned to it.
    monkeypatch.setenv("HERMES_INFERENCE_MODEL", "seed/model")
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": {"default": "new/model"}})
    session = _sync_test_session(config_model_seen=("seed/model", ""))
    calls = []
    monkeypatch.setattr(
        server,
        "_apply_model_switch",
        lambda sid, sess, raw, **kw: calls.append(raw),
    )

    server._sync_agent_model_with_config("sid", session)

    assert calls == ["new/model"]
    assert session["config_model_seen"] == ("new/model", "")


def test_config_sync_ignores_env_seed_without_config_model(monkeypatch):
    # `hermes --tui -m <model>` sets HERMES_MODEL/HERMES_INFERENCE_MODEL as a
    # launch-scoped seed. When config.yaml has NO model.default (typical
    # custom-provider-only setup), the sync must NOT adopt the env seed as a
    # config target — doing so replayed the -m flag as a /model switch and
    # (with persist_switch_by_default=True) wrote it into config.yaml
    # permanently.
    monkeypatch.setenv("HERMES_MODEL", "one-shot/model")
    monkeypatch.setenv("HERMES_INFERENCE_MODEL", "one-shot/model")
    monkeypatch.setattr(
        server, "_load_cfg", lambda: {"model": {"provider": "custom:mylocal"}}
    )
    session = _sync_test_session()
    monkeypatch.setattr(
        server,
        "_apply_model_switch",
        lambda *a, **k: pytest.fail("env seed must not trigger a config sync switch"),
    )

    server._sync_agent_model_with_config("sid", session)


def test_config_model_target_never_reads_env(monkeypatch):
    monkeypatch.setenv("HERMES_MODEL", "seed/model")
    monkeypatch.setenv("HERMES_INFERENCE_MODEL", "seed/model")
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": {"provider": "nous"}})

    assert server._config_model_target() == ("", "nous")


def test_apply_model_switch_persist_override_false_never_persists(monkeypatch):
    # Internal callers (config sync, /moa one-shot + restore) pass
    # persist_override=False; even with persist_switch_by_default=True the
    # switch must not write config.yaml.
    import types as _types

    result = _types.SimpleNamespace(
        success=True,
        new_model="new/model",
        target_provider="nous",
        base_url="",
        api_key="key",
        api_mode="chat_completions",
        warning_message="",
        model_info=None,
        error_message="",
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model", lambda **kw: result
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.resolve_persist_behavior",
        lambda *a: pytest.fail("persist_override must bypass resolve_persist_behavior"),
    )
    monkeypatch.setattr(
        server, "_persist_model_switch",
        lambda _r: pytest.fail("persist_override=False must not persist"),
    )
    monkeypatch.setattr(
        "hermes_cli.model_cost_guard.expensive_model_warning",
        lambda *a, **k: None,
    )
    session = {"agent": None}

    out = server._apply_model_switch(
        "sid", session, "new/model --provider nous", persist_override=False
    )

    assert out["value"] == "new/model"
    assert session["model_override"]["model"] == "new/model"

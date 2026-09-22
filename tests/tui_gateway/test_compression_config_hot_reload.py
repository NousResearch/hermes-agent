"""Desktop/TUI sessions must adopt live compression config on the next turn.

Regression for #95151: ``_sync_agent_model_with_config`` only compared the
model/provider. After ``hermes config set compression.threshold_tokens 100000``
the already-open session kept the computed threshold from agent creation.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent.context_compressor import ContextCompressor
from tui_gateway import server


def _session_with_compressor(**compression_ctor):
    compressor = ContextCompressor(
        model="gpt-5.6-sol",
        threshold_percent=0.85,
        config_context_length=272_000,
        quiet_mode=True,
        **compression_ctor,
    )
    agent = SimpleNamespace(
        model="gpt-5.6-sol",
        provider="openai-codex",
        context_compressor=compressor,
        compression_enabled=True,
        compression_idle_compact_after_seconds=0,
        codex_responses_native_compaction=False,
        codex_responses_compact_threshold=200_000,
    )
    return {
        "agent": agent,
        "session_key": "session-95151",
    }, compressor


def test_live_threshold_tokens_applies_on_next_turn_without_rebuild(monkeypatch):
    session, compressor = _session_with_compressor()
    stale = compressor.threshold_tokens
    assert stale > 100_000

    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {
            "model": {
                "default": "gpt-5.6-sol",
                "provider": "openai-codex",
                "context_length": 272_000,
            },
            "compression": {
                "threshold_tokens": 100_000,
                "proactive_prune_tokens": 48_000,
                "idle_compact_after_seconds": 1800,
                "tail_mode": "lean",
            },
        },
    )

    live_agent = session["agent"]
    server._sync_agent_compression_with_config("sid-95151", session)

    assert session["agent"] is live_agent
    assert compressor.threshold_tokens == 100_000
    assert compressor.proactive_prune_tokens == 48_000
    assert compressor.tail_mode == "lean"
    assert live_agent.compression_idle_compact_after_seconds == 1800


def test_live_codex_native_compaction_applies_on_next_turn(monkeypatch):
    session, _ = _session_with_compressor()

    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"compression": {"codex_responses_native": True}},
    )

    server._sync_agent_compression_with_config("sid-95151", session)

    assert session["agent"].codex_responses_native_compaction is True


def test_live_codex_native_threshold_applies_on_next_turn(monkeypatch):
    session, _ = _session_with_compressor()

    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {
            "compression": {"codex_responses_compact_threshold": 120_000}
        },
    )

    server._sync_agent_compression_with_config("sid-95151", session)

    assert session["agent"].codex_responses_compact_threshold == 120_000


def test_unchanged_compression_config_is_noop(monkeypatch):
    session, compressor = _session_with_compressor(threshold_tokens_cap=100_000)
    cfg = {
        "model": {"context_length": 272_000},
        "compression": {"threshold_tokens": 100_000},
    }
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    session["config_compression_seen"] = server._tui_compression_config_signature(cfg)

    compressor.threshold_tokens = 99_999
    server._sync_agent_compression_with_config("sid-95151", session)

    assert compressor.threshold_tokens == 99_999


def test_clearing_threshold_tokens_restores_default_cap(monkeypatch):
    session, compressor = _session_with_compressor(threshold_tokens_cap=100_000)
    assert compressor.threshold_tokens == 100_000

    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {
            "model": {"context_length": 272_000},
            "compression": {},
        },
    )
    server._sync_agent_compression_with_config("sid-95151", session)

    # Key removal restores what a fresh agent build installs (merged DEFAULT_CONFIG), not "no
    # cap": a None here re-derives the uncapped ratio trigger and the 256K default is lost.
    assert compressor.threshold_tokens_cap == 256_000
    assert compressor.threshold_tokens > 100_000


def test_absent_threshold_tokens_keeps_default_cap_on_1m_window(monkeypatch):
    """#117093: the live read is unmerged (missing key = unset), so a config.yaml without
    compression.threshold_tokens used to wipe the ctor-installed 256K cap at the first
    turn's sync — the trigger re-derived to the uncapped ratio value (500K on a 1M window)
    and compaction stopped firing at 256K while telemetry still reported the capped figure."""
    compressor = ContextCompressor(
        model="unset-test-model",
        threshold_percent=0.85,
        config_context_length=1_000_000,
        threshold_tokens_cap=256_000,
        quiet_mode=True,
    )
    agent = SimpleNamespace(
        model="unset-test-model",
        provider="",
        context_compressor=compressor,
        compression_enabled=True,
        compression_idle_compact_after_seconds=0,
        codex_responses_native_compaction=False,
        codex_responses_compact_threshold=200_000,
    )
    session = {"agent": agent, "session_key": "session-unset"}
    assert compressor.threshold_tokens == 256_000  # min(1M * 0.85, 256K cap)

    # The pin is scoped to the configured default route (#116467); that scoping has its own tests,
    # this one is about the cap, so keep the 1M window in scope for the bare test runtime.
    import agent.agent_init as agent_init

    monkeypatch.setattr(agent_init, "config_context_length_for_runtime", lambda _agent, _cfg=None: 1_000_000)
    _sync_with_cfg(monkeypatch, session, {"model": {"context_length": 1_000_000}, "compression": {}})

    assert compressor.threshold_tokens_cap == 256_000
    # threshold is absent too, so the derived 0.50 ratio applies — the 256K cap must still win.
    assert compressor.threshold_tokens == 256_000


def test_live_model_threshold_outranks_only_implicit_default_cap(monkeypatch):
    session, compressor = _neutral_session(threshold_tokens_cap=100_000)
    import agent.agent_init as agent_init

    monkeypatch.setattr(agent_init, "config_context_length_for_runtime", lambda _agent, _cfg=None: 600_000)
    cfg = {
        "model": {"context_length": 600_000},
        "compression": {"model_thresholds": {"unset-test-model": 0.6}},
    }

    _sync_with_cfg(monkeypatch, session, cfg)

    assert compressor.threshold_tokens_cap == 256_000
    assert compressor.threshold_tokens_cap_is_default is True
    assert compressor.threshold_tokens == 360_000

    cfg["compression"]["threshold_tokens"] = 256_000
    _sync_with_cfg(monkeypatch, session, cfg)
    assert compressor.threshold_tokens_cap_is_default is False
    assert compressor.threshold_tokens == 256_000



def test_prompt_submit_calls_compression_sync_after_model_sync():
    # Read the module that actually defines the turn (it moved out of server.py).
    source = open(server._run_prompt_submit.__code__.co_filename, encoding="utf-8").read()
    model_idx = source.find("_sync_agent_model_with_config(sid, session)")
    compression_idx = source.find("_sync_agent_compression_with_config(sid, session)")
    assert model_idx != -1
    assert compression_idx != -1
    assert model_idx < compression_idx


# ── Unset semantics (#94724 review finding on #95980) ────────────────────
# ``_apply_live_compression_config`` used to act only on PRESENT keys, so
# removing tail_mode / context_length / target_ratio / model_thresholds /
# proactive_prune_* / protect_last_n / min_tail_user_messages / threshold /
# idle_compact_after_seconds from config.yaml left stale values active in
# live sessions forever. Absence must restore the normalized default (or the
# model-derived value) through the same derivation the agent-construction
# path uses.


def _neutral_session(**compression_ctor):
    """Session on a model with no per-model threshold override in play."""
    compressor = ContextCompressor(
        model="unset-test-model",
        config_context_length=600_000,  # >=512K: no small-context floor
        quiet_mode=True,
        **compression_ctor,
    )
    agent = SimpleNamespace(
        model="unset-test-model",
        provider="",
        base_url="",
        context_compressor=compressor,
        compression_enabled=True,
        compression_idle_compact_after_seconds=0,
        codex_responses_native_compaction=False,
        codex_responses_compact_threshold=200_000,
    )
    return {"agent": agent, "session_key": "session-unset"}, compressor


def _sync_with_cfg(monkeypatch, session, cfg):
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    server._sync_agent_compression_with_config("sid-unset", session)


def test_removing_tail_mode_restores_lean_default(monkeypatch):
    session, compressor = _neutral_session(tail_mode="legacy")
    assert compressor.tail_mode == "legacy"
    _sync_with_cfg(monkeypatch, session, {"compression": {}})
    assert compressor.tail_mode == "lean"


def test_removing_target_ratio_restores_default(monkeypatch):
    session, compressor = _neutral_session(summary_target_ratio=0.60)
    assert compressor.summary_target_ratio == 0.60
    _sync_with_cfg(monkeypatch, session, {"compression": {}})
    assert compressor.summary_target_ratio == 0.20


def test_removing_protect_last_n_restores_default(monkeypatch):
    session, compressor = _neutral_session(protect_last_n=5)
    _sync_with_cfg(monkeypatch, session, {"compression": {}})
    assert compressor.protect_last_n == 20


def test_removing_proactive_prune_keys_restores_defaults(monkeypatch):
    session, compressor = _neutral_session(
        proactive_prune_tokens=48_000,
        proactive_prune_min_result_chars=30_000,
        proactive_prune_min_reclaim_tokens=1,
    )
    _sync_with_cfg(monkeypatch, session, {"compression": {}})
    assert compressor.proactive_prune_tokens == 0
    assert compressor.proactive_prune_min_result_chars == 8000
    assert compressor.proactive_prune_min_reclaim_tokens == 4096


def test_removing_min_tail_user_messages_restores_default(monkeypatch):
    session, compressor = _neutral_session(min_tail_user_messages=4)
    _sync_with_cfg(monkeypatch, session, {"compression": {}})
    assert compressor.min_tail_user_messages == 1


def test_removing_model_thresholds_restores_empty_map(monkeypatch):
    session, compressor = _neutral_session(
        model_thresholds={"unset-test-model": 0.95}
    )
    assert compressor.threshold_percent == 0.95
    _sync_with_cfg(monkeypatch, session, {"compression": {}})
    assert compressor.model_thresholds == {}
    # The stale per-model override must stop steering the live threshold too.
    assert compressor.threshold_percent == 0.50


def test_removing_threshold_restores_derived_default(monkeypatch):
    session, compressor = _neutral_session(threshold_percent=0.85)
    assert compressor.threshold_percent == 0.85
    _sync_with_cfg(
        monkeypatch,
        session,
        {"model": {"context_length": 600_000}, "compression": {}},
    )
    assert compressor._config_threshold_percent == 0.50
    assert compressor.threshold_percent == 0.50
    # The absent cap key restores the 256K default, which binds below the 300K ratio value.
    assert compressor.threshold_tokens == min(int(600_000 * 0.50), 256_000)


def test_removing_context_length_reinfers_from_model_metadata(monkeypatch):
    import agent.context_compressor as cc_mod

    session, compressor = _neutral_session()
    assert compressor.context_length == 600_000

    monkeypatch.setattr(
        cc_mod,
        "get_model_context_length",
        lambda *a, **k: 1_000_000,
    )
    _sync_with_cfg(monkeypatch, session, {"model": {}, "compression": {}})
    assert compressor._config_context_length is None
    assert compressor.context_length == 1_000_000
    assert compressor.threshold_tokens == min(int(1_000_000 * 0.50), 256_000)


def test_removing_idle_compact_after_seconds_restores_zero(monkeypatch):
    session, _ = _neutral_session()
    session["agent"].compression_idle_compact_after_seconds = 1800
    _sync_with_cfg(monkeypatch, session, {"compression": {}})
    assert session["agent"].compression_idle_compact_after_seconds == 0


def test_removing_enabled_restores_true(monkeypatch):
    session, _ = _neutral_session()
    session["agent"].compression_enabled = False
    _sync_with_cfg(monkeypatch, session, {"compression": {}})
    assert session["agent"].compression_enabled is True


def test_removing_codex_native_compaction_restores_false(monkeypatch):
    session, _ = _neutral_session()
    session["agent"].codex_responses_native_compaction = True
    _sync_with_cfg(monkeypatch, session, {"compression": {}})
    assert session["agent"].codex_responses_native_compaction is False


def test_removing_codex_native_threshold_restores_default(monkeypatch):
    session, _ = _neutral_session()
    session["agent"].codex_responses_compact_threshold = 120_000
    _sync_with_cfg(monkeypatch, session, {"compression": {}})
    assert session["agent"].codex_responses_compact_threshold == 200_000


def test_apply_live_compression_config_is_self_contained():
    # Regression for #115572: _apply_live_compression_config referenced
    # is_truthy_value without importing it, so a direct (non-rebound) call
    # raised NameError. The module must not depend on server.py injecting the
    # name via method_ctx.bind_module.
    from tui_gateway.session_compression import _apply_live_compression_config

    agent = SimpleNamespace(
        model="unset-test-model",
        provider="",
        context_compressor=None,
        compression_enabled=True,
        compression_idle_compact_after_seconds=0,
        codex_responses_native_compaction=False,
        codex_responses_compact_threshold=200_000,
    )
    _apply_live_compression_config(agent, {"compression": {"enabled": True}})
    assert agent.compression_enabled is True
    assert agent.codex_responses_native_compaction is False

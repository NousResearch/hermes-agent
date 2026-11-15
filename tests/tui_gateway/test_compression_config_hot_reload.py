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


def _default_cap():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    return DEFAULT_CONFIG["compression"]["threshold_tokens"]


def _with_default_cap(trigger: int) -> int:
    cap = _default_cap()
    return min(trigger, cap) if cap else trigger


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

    # Key removal restores what a fresh agent build installs (merged DEFAULT_CONFIG).
    assert compressor.threshold_tokens_cap == _default_cap()
    assert compressor.threshold_tokens > 100_000


def test_absent_threshold_tokens_keeps_default_cap_on_1m_window(monkeypatch):
    """#117093: the live read is unmerged (missing key = unset), while agent construction reads the
    merged config. A config.yaml without compression.threshold_tokens must leave the live session on
    the same cap a fresh build installs, not re-derive a different trigger at the first turn's sync."""
    compressor = ContextCompressor(
        model="unset-test-model",
        threshold_percent=0.85,
        config_context_length=1_000_000,
        threshold_tokens_cap=_default_cap(),
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
    assert compressor.threshold_tokens == _with_default_cap(850_000)

    # The pin is scoped to the configured default route (#116467); that scoping has its own tests,
    # this one is about the cap, so keep the 1M window in scope for the bare test runtime.
    from agent import agent_init

    monkeypatch.setattr(agent_init, "config_context_length_for_runtime", lambda _agent, _cfg=None: 1_000_000)
    _sync_with_cfg(monkeypatch, session, {"model": {"context_length": 1_000_000}, "compression": {}})

    assert compressor.threshold_tokens_cap == _default_cap()
    # threshold is absent too, so the derived 0.50 ratio applies under the default cap.
    assert compressor.threshold_tokens == _with_default_cap(500_000)





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


def test_removing_compressor_keys_restores_fresh_build_values(monkeypatch):
    """Absent keys must land on exactly what a fresh ContextCompressor installs, not stale values."""
    session, compressor = _neutral_session(
        tail_mode="legacy",
        summary_target_ratio=0.60,
        protect_last_n=5,
        proactive_prune_tokens=48_000,
        proactive_prune_min_result_chars=30_000,
        proactive_prune_min_reclaim_tokens=1,
        min_tail_user_messages=4,
        model_thresholds={"unset-test-model": 0.95},
    )
    assert compressor.threshold_percent == 0.95
    _sync_with_cfg(monkeypatch, session, {"compression": {}})

    _, fresh = _neutral_session()
    for attr in (
        "tail_mode",
        "summary_target_ratio",
        "protect_last_n",
        "proactive_prune_tokens",
        "proactive_prune_min_result_chars",
        "proactive_prune_min_reclaim_tokens",
        "min_tail_user_messages",
        "model_thresholds",
        "threshold_percent",  # the stale per-model override must stop steering the live threshold
    ):
        assert getattr(compressor, attr) == getattr(fresh, attr), attr


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
    # The absent cap key restores DEFAULT_CONFIG's cap on top of the derived ratio trigger.
    assert compressor.threshold_tokens == _with_default_cap(int(600_000 * 0.50))


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
    assert compressor.threshold_tokens == _with_default_cap(int(1_000_000 * 0.50))


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


def test_live_in_place_false_applies_and_arms_rotation_for_bloated_legacy_session(monkeypatch):
    session, compressor = _session_with_compressor()
    agent = session["agent"]
    agent.session_id = "20291231_235959_future"
    agent.compression_in_place = True
    compressor.last_prompt_tokens = 120_000

    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {
            "model": {
                "default": "gpt-5.6-sol",
                "provider": "openai-codex",
                "context_length": 1_048_576,
            },
            "compression": {
                "in_place": False,
                "proactive_prune_tokens": 0,
                "threshold": 0.185,
            },
        },
    )
    server._sync_agent_compression_with_config("20291231_235959_future", session)  # type: ignore[attr-defined]
    assert agent.compression_in_place is False
    assert compressor.proactive_prune_tokens == 0
    assert compressor.threshold_tokens == 75_000
    assert getattr(compressor, "_force_rotate_restore_threshold", 0) > 100_000


def test_rotated_child_session_inherits_europe_west2_affinity_and_restores_threshold():
    from typing import Any, cast

    from agent.conversation_compression import _publish_rotated_compaction
    from agent.vertex_adapter import (
        EUROPE_WEST2_VERTEX_REGION,
        _record_session_vertex_region_affinity,
        get_session_vertex_region_affinity,
    )

    compressor = ContextCompressor(
        model="google/gemini-4-argon",
        threshold_percent=0.185,
        config_context_length=1_048_576,
        quiet_mode=True,
    )
    compressor._force_rotate_restore_threshold = 193_986  # type: ignore[attr-defined]
    compressor.threshold_tokens = 75_000

    class _StubDB:
        def get_active_message_watermark(self, _sid):
            return 10

        def get_session_title(self, _sid):
            return "test"

        def publish_compression_child(self, **_kwargs):
            return None

        def update_session_title(self, _sid, _title):
            return None

    _record_session_vertex_region_affinity("20261114_010000_old", EUROPE_WEST2_VERTEX_REGION)
    agent = SimpleNamespace(
        session_id="20261114_010000_old",
        model="google/gemini-4-argon",
        context_compressor=compressor,
        _session_db=_StubDB(),
        _session_init_model_config={},
        _flush_messages_to_session_db=lambda *_a, **_k: None,
    )
    _publish_rotated_compaction(
        agent,
        [{"role": "user", "content": "hi"}],
        [{"role": "user", "content": "hi"}],
        new_system_prompt="system",
        lease=cast(Any, SimpleNamespace(holder=None, ttl=0.0, watermark=None)),
        old_session_id="20261114_010000_old",
        compressed_user_turn_outcome="none",
    )
    new_sid = agent.session_id
    assert new_sid != "20261114_010000_old"
    assert getattr(agent, "_rotated_to_child_session", False) is True
    assert getattr(compressor, "_force_rotate_restore_threshold", 0) == 0
    assert compressor.threshold_tokens == 193_986
    assert get_session_vertex_region_affinity(new_sid) == EUROPE_WEST2_VERTEX_REGION
    assert get_session_vertex_region_affinity("20261114_010000_old") is None


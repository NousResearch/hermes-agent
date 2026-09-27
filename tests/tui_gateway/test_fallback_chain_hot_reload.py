"""Desktop/TUI sessions must adopt a ``fallback_providers`` chain added after the chat was opened.

Regression for #95066: ``_make_agent`` read the chain once, so a session born before ``hermes fallback
add`` kept an empty ``_fallback_chain`` forever and a Codex ``usage_limit_reached`` 429 ended in a
provider error instead of switching to the configured fallback.

Both tests drive ``_prepare_turn_input`` (turn admission, the production entry) up to the sync's
successor so the wiring — not just the helper — is pinned.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from tui_gateway import server

FALLBACK = [{"provider": "xai-oauth", "model": "grok-4.6"}]


class _StopAfterSync(Exception):
    pass


def _session(chain=None):
    agent = SimpleNamespace(
        _fallback_chain=list(chain or []), _fallback_model=(chain or [None])[0], _fallback_index=0,
        _fallback_activated=False, _rate_limited_until=0, _unavailable_fallback_keys=set(),
    )
    return {"agent": agent, "session_key": "session-95066"}, agent


def _admit_turn(monkeypatch, tmp_path, session, config_text: str) -> None:
    """Run turn admission against ``config_text`` as the live config.yaml; stop right after the sync."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(config_text, encoding="utf-8")
    monkeypatch.setattr(server, "_active_config_path", lambda: cfg_path)
    monkeypatch.setattr(server, "_profile_runtime_scope_tokens", lambda profile_home: None)
    monkeypatch.setattr(server, "_set_session_context", lambda *a, **k: [])
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    for name in ("_apply_pending_model_switch", "_sync_agent_model_with_config", "_sync_agent_compression_with_config"):
        monkeypatch.setattr(server, name, lambda sid, session: None)

    def stop(sid, session):
        raise _StopAfterSync()

    monkeypatch.setattr(server, "_sync_bot_capabilities", stop)
    st = server._TurnRun(agent=None, one_turn_restore=None, terminal_callback=None, receipt_committed=False)
    with pytest.raises(_StopAfterSync):
        server._prepare_turn_input("sid", session, st, "hello", [])


def test_chain_added_after_open_reaches_the_live_agent_at_turn_admission(monkeypatch, tmp_path):
    session, agent = _session()

    _admit_turn(monkeypatch, tmp_path, session,
                "fallback_providers:\n  - provider: xai-oauth\n    model: grok-4.6\n")

    assert agent._fallback_chain == FALLBACK
    assert agent._fallback_model == FALLBACK[0]
    assert agent._fallback_index == 0


def test_torn_config_keeps_the_last_known_good_chain_but_removal_still_applies(monkeypatch, tmp_path):
    session, agent = _session(FALLBACK)

    # Torn mid-edit write: an unparsable config.yaml must NOT read as "chain removed".
    _admit_turn(monkeypatch, tmp_path, session, "fallback_providers: [\n  - provider: {{{\n")
    assert agent._fallback_chain == FALLBACK
    assert agent._fallback_model == FALLBACK[0]

    # Control: a valid config with the chain removed clears it on the next turn.
    _admit_turn(monkeypatch, tmp_path, session, "model:\n  provider: openai\n")
    assert agent._fallback_chain == []
    assert agent._fallback_model is None

    # While a cooldown holds the agent on an activated fallback, the sync leaves the chain alone.
    agent._fallback_chain, agent._fallback_activated = list(FALLBACK), True
    agent._rate_limited_until = time.monotonic() + 600
    _admit_turn(monkeypatch, tmp_path, session, "model:\n  provider: openai\n")
    assert agent._fallback_chain == FALLBACK

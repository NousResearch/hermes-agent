"""The AC-4 synthetic heavy-turn agent (``tui_gateway/synthetic_turn.py``) is a
test seam: it must stay dead unless ``HERMES_ISO_CERTIFY_SYNTH_TURN=1``.
"""

from __future__ import annotations

from tui_gateway.synthetic_turn import (
    maybe_build_synthetic_agent,
    synth_turn_armed,
)


def test_synth_seam_dead_when_env_unset(monkeypatch):
    monkeypatch.delenv("HERMES_ISO_CERTIFY_SYNTH_TURN", raising=False)
    assert synth_turn_armed() is False
    assert maybe_build_synthetic_agent("sid") is None

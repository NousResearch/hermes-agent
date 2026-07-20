"""Tests for the AC-4 isolation certify seam + harness helpers.

The synthetic heavy-turn agent (``tui_gateway/synthetic_turn.py``) is a test
seam: dead unless ``HERMES_ISO_CERTIFY_SYNTH_TURN=1``. These tests pin (a) the
dead-when-unset contract, (b) that an armed turn holds for the requested wall
duration and streams deltas, (c) that interrupt aborts it promptly, and (d) the
harness percentile math.
"""

from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path

import pytest

from tui_gateway.synthetic_turn import (
    SyntheticHeavyAgent,
    maybe_build_synthetic_agent,
    synth_turn_armed,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_iso_certify():
    path = REPO_ROOT / "scripts" / "iso-certify.py"
    spec = importlib.util.spec_from_file_location("iso_certify", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_synth_seam_dead_when_env_unset(monkeypatch):
    monkeypatch.delenv("HERMES_ISO_CERTIFY_SYNTH_TURN", raising=False)
    assert synth_turn_armed() is False
    assert maybe_build_synthetic_agent("sid") is None


def test_synth_seam_armed_builds_agent(monkeypatch):
    monkeypatch.setenv("HERMES_ISO_CERTIFY_SYNTH_TURN", "1")
    assert synth_turn_armed() is True
    agent = maybe_build_synthetic_agent("sid", {"model": "custom-x"})
    assert isinstance(agent, SyntheticHeavyAgent)
    assert agent.model == "custom-x"


def test_synth_turn_holds_duration_and_streams():
    agent = SyntheticHeavyAgent("s1")
    deltas: list[str] = []
    spec = '{"duration_s": 0.4, "delta_interval_s": 0.05, "tokens_per_delta": 100}'
    t0 = time.monotonic()
    result = agent.run_conversation(spec, stream_callback=deltas.append)
    elapsed = time.monotonic() - t0
    # Held for ~the requested wall time (allow generous upper bound under load).
    assert 0.35 <= elapsed <= 5.0, elapsed
    assert result["interrupted"] is False
    assert len(deltas) >= 3
    # Token accounting advanced (the 100K-token heavy-turn proxy).
    assert agent.session_output_tokens == 100 * len(deltas)
    assert agent.session_api_calls == 1
    # Role alternation preserved in the produced messages.
    roles = [m["role"] for m in result["messages"]]
    assert roles[-2:] == ["user", "assistant"]


def test_synth_turn_interrupt_aborts_promptly():
    agent = SyntheticHeavyAgent("s2")

    def _stop():
        time.sleep(0.2)
        agent.interrupt()

    threading.Thread(target=_stop, daemon=True).start()
    t0 = time.monotonic()
    result = agent.run_conversation('{"duration_s": 10.0}')
    elapsed = time.monotonic() - t0
    assert result["interrupted"] is True
    assert elapsed < 5.0, elapsed


def test_synth_turn_non_json_prompt_uses_defaults(monkeypatch):
    monkeypatch.setenv("HERMES_ISO_CERTIFY_DURATION_S", "0.2")
    agent = SyntheticHeavyAgent("s3")
    t0 = time.monotonic()
    agent.run_conversation("just a plain prompt, not json")
    assert time.monotonic() - t0 >= 0.15


def test_harness_percentile_and_guard():
    iso = _load_iso_certify()
    assert iso.percentile([], 99) == 0.0
    assert iso.percentile([5.0], 99) == 5.0
    vals = [float(i) for i in range(1, 101)]  # 1..100
    assert 98.0 <= iso.percentile(vals, 99) <= 100.0
    assert iso.percentile(vals, 50) == pytest.approx(50.5, abs=0.6)
    # The empty-timeline INCONCLUSIVE floor: too few probe samples never PASSes.
    assert iso.probe_thread_samples_ok([1.0, 2.0], [1.0, 2.0, 3.0]) is False
    assert iso.probe_thread_samples_ok([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) is True


def test_harness_summarize_shape():
    iso = _load_iso_certify()
    s = iso.summarize([10.0, 20.0, 30.0])
    assert s["count"] == 3
    assert s["max_ms"] == 30.0
    assert set(s) == {"count", "p50_ms", "p95_ms", "p99_ms", "max_ms"}

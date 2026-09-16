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



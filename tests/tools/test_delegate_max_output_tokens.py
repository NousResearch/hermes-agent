"""Tests for delegation.max_output_tokens resolution precedence.

Priority: runtime > cfg > legacy max_tokens. Explicit 0 = uncapped.
"""
from unittest.mock import MagicMock, patch
from tools.delegate_tool import _resolve_max_output_tokens


def _cfg(**overrides):
    base = {
        "delegation": {
            "max_output_tokens": 0,
            "max_tokens": 0,
        }
    }
    base["delegation"].update(overrides)
    return base


def _runtime(**overrides):
    return overrides


def test_runtime_overrides_cfg():
    """Runtime max_output_tokens beats cfg."""
    runtime = _runtime(max_output_tokens=4096)
    cfg = _cfg(max_output_tokens=2048, max_tokens=1024)
    assert _resolve_max_output_tokens(runtime, cfg) == 4096


def test_cfg_overrides_legacy():
    """cfg max_output_tokens beats legacy max_tokens."""
    runtime = _runtime()
    cfg = _cfg(max_output_tokens=2048, max_tokens=1024)
    assert _resolve_max_output_tokens(runtime, cfg) == 2048


def test_legacy_used_when_no_modern():
    """Legacy max_tokens used when modern keys absent."""
    runtime = _runtime()
    cfg = {"delegation": {"max_tokens": 1024}}
    assert _resolve_max_output_tokens(runtime, cfg) == 1024


def test_explicit_zero_is_preserved():
    """Explicit 0 = uncapped, not treated as missing."""
    runtime = _runtime(max_output_tokens=0)
    cfg = _cfg(max_output_tokens=4096)
    assert _resolve_max_output_tokens(runtime, cfg) == 0


def test_all_unset_returns_none():
    """No cap configured returns None."""
    runtime = _runtime()
    cfg = {"delegation": {}}
    assert _resolve_max_output_tokens(runtime, cfg) is None


def test_negative_clamped_to_zero():
    """Negative values clamped to 0."""
    runtime = _runtime(max_output_tokens=-1)
    cfg = _cfg()
    assert _resolve_max_output_tokens(runtime, cfg) == 0


def test_string_coerced_to_int():
    """String values coerced to int."""
    runtime = _runtime(max_output_tokens="4096")
    cfg = _cfg()
    assert _resolve_max_output_tokens(runtime, cfg) == 4096


def test_legacy_zero_is_preserved():
    """Legacy max_tokens=0 is preserved, not treated as missing."""
    runtime = _runtime()
    cfg = {"delegation": {"max_tokens": 0}}
    assert _resolve_max_output_tokens(runtime, cfg) == 0

"""Subagents spawned by delegate_task compress at an absolute context cap, not threshold x window.

In a 1,393-agent run on a 1M-window model the children's trigger was 0.85 x 1M = 850K; 1,373 of 1,375
never compressed and calls above 200K context carried ~55% of the bill.
"""
from types import SimpleNamespace

from agent.context_compressor import ContextCompressor
from tools.delegate_tool import _apply_child_compression_cap


def _child(window=1_000_000, threshold=0.85, cap=None):
    cc = ContextCompressor(model="anthropic/claude-fable-5.1", threshold_percent=threshold,
                           config_context_length=window, threshold_tokens_cap=cap)
    return SimpleNamespace(context_compressor=cc)


def test_default_cap_bounds_a_big_window_child():
    child = _child()
    _apply_child_compression_cap(child, {})
    assert child.context_compressor.threshold_tokens == 200_000


def test_cap_is_the_lower_of_delegation_and_global_and_never_raises():
    child = _child(cap=150_000)
    _apply_child_compression_cap(child, {"compression_threshold_tokens": 200_000})
    assert child.context_compressor.threshold_tokens == 150_000
    small = _child(window=128_000)
    before = small.context_compressor.threshold_tokens
    _apply_child_compression_cap(small, {"compression_threshold_tokens": 200_000})
    assert small.context_compressor.threshold_tokens == before  # cap above the ratio trigger: no effect


def test_zero_disables_and_an_already_resolved_trigger_is_reclamped():
    child = _child()
    assert child.context_compressor.threshold_tokens == 850_000  # resolve first
    _apply_child_compression_cap(child, {"compression_threshold_tokens": 0})
    assert child.context_compressor.threshold_tokens == 850_000
    _apply_child_compression_cap(child, {"compression_threshold_tokens": 300_000})
    assert child.context_compressor.threshold_tokens == 300_000


def test_config_values_are_validated_not_coerced():
    """Independent-review witnesses: YAML `true` coerced to int 1 (a one-token trigger) and "200k"
    silently disabled the cap. Both fall back to the default with a warning; 0/false/null disable."""
    from tools.delegate_tool import _child_compression_cap_tokens as cap
    assert cap(True) == 200_000 and cap("200k") == 200_000 and cap(5) == 200_000
    assert cap(0) is None and cap(False) is None and cap(None) is None
    assert cap(150_000) == 150_000 and cap(300_000.0) == 300_000

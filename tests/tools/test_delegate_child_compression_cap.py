"""``delegation.compression_threshold_tokens`` is an OPTIONAL absolute cap on a subagent's compaction
trigger, off by default, and its value is validated rather than coerced.

Default off: a 1M-window child compacts at the same 0.50 x window as its parent (500K). A replay of a
1,393-agent run put 200K-400K caps within 5% of each other once cache prefixes are intact, and every
compaction is a chance to lose detail, so the cap is opt-in. The validation matters because YAML
``true`` coerces to int 1 (a one-token trigger) and ``"200k"`` would silently read as no cap.
"""
from types import SimpleNamespace

from agent.context_compressor import ContextCompressor
from tools.delegate_tool import _apply_child_compression_cap, _child_compression_cap_tokens


def _child(window=1_000_000, threshold=0.50, cap=None):
    cc = ContextCompressor(model="anthropic/claude-fable-5.1", threshold_percent=threshold,
                           config_context_length=window, threshold_tokens_cap=cap)
    return SimpleNamespace(context_compressor=cc)


def test_default_is_no_cap_child_keeps_the_ratio_trigger():
    child = _child()
    _apply_child_compression_cap(child, {})
    assert child.context_compressor.threshold_tokens == 500_000
    _apply_child_compression_cap(child, {"compression_threshold_tokens": 0})
    assert child.context_compressor.threshold_tokens == 500_000


def test_explicit_cap_is_the_lower_of_delegation_and_global_and_never_raises():
    child = _child()
    _apply_child_compression_cap(child, {"compression_threshold_tokens": 300_000})
    assert child.context_compressor.threshold_tokens == 300_000
    child = _child(cap=150_000)
    _apply_child_compression_cap(child, {"compression_threshold_tokens": 200_000})
    assert child.context_compressor.threshold_tokens == 150_000
    small = _child(window=128_000)
    before = small.context_compressor.threshold_tokens
    _apply_child_compression_cap(small, {"compression_threshold_tokens": 200_000})
    assert small.context_compressor.threshold_tokens == before  # cap above the ratio trigger: no effect


def test_config_values_are_validated_not_coerced():
    """Independent-review witnesses: YAML ``true`` -> int 1 (a one-token trigger) and ``"200k"`` -> silently
    disabled. Both are ignored with a warning; the child keeps the ratio trigger."""
    for bad in (True, "200k", 5, 15_999, -1, 1.5):
        assert _child_compression_cap_tokens(bad) is None, bad
    for off in (None, 0, False):
        assert _child_compression_cap_tokens(off) is None
    assert _child_compression_cap_tokens(16_000) == 16_000
    assert _child_compression_cap_tokens(300_000.0) == 300_000
    child = _child()
    _apply_child_compression_cap(child, {"compression_threshold_tokens": True})
    assert child.context_compressor.threshold_tokens == 500_000

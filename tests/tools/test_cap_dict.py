"""Regression test for _cap_dict — must not mutate dict during iteration.

Bug: the old implementation called ``d.pop(next(it))`` inside a loop over
``iter(d)``, which is undefined behaviour in Python and raises RuntimeError
on some builds when the dict is resized during iteration.

Fix: snapshot the oldest keys via ``list(d)[:over]`` before popping.
See PR #XXXXX.
"""

from tools.file_state import _cap_dict


def test_cap_dict_basic():
    """Removes oldest entries to enforce the limit."""
    d = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    _cap_dict(d, 3)
    assert len(d) == 3
    # Oldest keys ("a", "b") should be gone
    assert "a" not in d
    assert "b" not in d
    assert set(d.keys()) == {"c", "d", "e"}


def test_cap_dict_noop_when_under_limit():
    """No-op when dict is already at or under the limit."""
    d = {"a": 1, "b": 2}
    _cap_dict(d, 5)
    assert len(d) == 2
    assert set(d.keys()) == {"a", "b"}


def test_cap_dict_exact_limit():
    """No-op when dict size equals limit."""
    d = {"a": 1, "b": 2, "c": 3}
    _cap_dict(d, 3)
    assert len(d) == 3


def test_cap_dict_empty():
    """No-op on empty dict."""
    d = {}
    _cap_dict(d, 5)
    assert len(d) == 0


def test_cap_dict_zero_limit():
    """Clears dict when limit is 0."""
    d = {"a": 1, "b": 2}
    _cap_dict(d, 0)
    assert len(d) == 0


def test_cap_dict_preserves_insertion_order():
    """Newest entries (most recently inserted) survive."""
    d = {f"k{i}": i for i in range(100)}
    _cap_dict(d, 10)
    assert len(d) == 10
    # The last 10 keys should remain
    assert set(d.keys()) == {f"k{i}" for i in range(90, 100)}


def test_cap_dict_no_runtime_error():
    """Regression: must not raise RuntimeError from mutation during iteration.

    The old code did ``d.pop(next(iter(d)))`` which mutates the dict while
    iterating.  On dicts with >~ 600 entries CPython triggers a full table
    resize on pop, invalidating the iterator and raising RuntimeError.
    """
    d = {f"key_{i:04d}": i for i in range(1000)}
    # Should complete without any exception
    _cap_dict(d, 50)
    assert len(d) == 50

"""Regression test for #105238.

``rebind`` memoizes every rebound object in ``_seen`` keyed by ``id(fn)`` so
repeated references to the same source function resolve to one canonical
object. The ``@contextlib.contextmanager`` branch was the one path that
returned early WITHOUT writing to ``_seen``, so the same contextmanager-wrapped
helper referenced from more than one closure produced a distinct rebound object
per reference — silently violating the identity/sharing invariant that the
memoization exists to guarantee (and redundantly re-binding the wrapped
generator on every reference).
"""

import contextlib

from tui_gateway.method_ctx import rebind


def _make_cm_helper():
    @contextlib.contextmanager
    def helper():
        yield "ok"

    return helper


def test_rebind_contextmanager_memoizes_same_source():
    helper = _make_cm_helper()
    seen = {}

    first = rebind(helper, {}, seen)
    second = rebind(helper, {}, seen)

    # The identity invariant every other branch guarantees: one canonical
    # rebound object per source function within a single _seen pass.
    assert first is second


def test_rebind_plain_function_still_memoizes():
    # Guard against a regression of the non-contextmanager path while we are
    # here: a plain function referenced twice must also resolve identically.
    def plain():
        return "x"

    seen = {}
    assert rebind(plain, {}, seen) is rebind(plain, {}, seen)

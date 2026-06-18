#!/usr/bin/env python3
"""Behavior-contract tests for the `@`-file picker frecency feature.

These assert INVARIANTS (how the algorithm must behave), not frozen numbers:
  * O(1) exponential-decay accumulator == full visit-history sum (fre identity)
  * exponential decay does NOT exhibit the zoxide/z/fasd re-visit spike
  * frequency dominates at equal recency; recency dominates at equal frequency
  * the gateway tiered blend never lets frecency cross a textual rank tier
  * disabling frecency is a true no-op (prior static ordering preserved)
  * the universal `@`-ref resolver records a hit for real files/folders

HERMES_HOME is redirected to a per-test tempdir by the autouse fixture in
tests/conftest.py, so the frecency sidecar is isolated per test.

Run with:  python -m pytest tests/tools/test_file_frecency.py -v
"""

import asyncio
import math
import os
import random

import pytest

from tools import file_frecency as ff


SECONDS_PER_DAY = 86400.0


def _lam(half_life_days: float) -> float:
    return math.log(2) / (half_life_days * SECONDS_PER_DAY)


@pytest.fixture(autouse=True)
def _enable_frecency():
    """Force a known config (enabled, 1d half-life) and clear the cache so the
    test doesn't depend on whatever config.yaml the environment carries."""
    ff.reset_config_cache()
    ff._config_cached = {
        "enabled": True,
        "half_life_days": 1.0,
        "weight": 40.0,
        "max_entries": 100000,  # high so per-test small-N cases never trip aging
        "max_total": 10000.0,
    }
    ff._config_loaded = True
    yield
    ff.reset_config_cache()


# ---------------------------------------------------------------------------
# Algorithm invariants
# ---------------------------------------------------------------------------
def _score_full_history(visit_times, now, half_life_days):
    l = _lam(half_life_days)
    return sum(math.exp(-l * (now - t)) for t in visit_times)


def _accumulate(visit_times, half_life_days):
    """Mirror file_frecency's O(1) update over a visit list, returning (w, t)."""
    l = _lam(half_life_days)
    w, t_last = 0.0, 0.0
    for t in visit_times:
        if w == 0.0:
            w, t_last = 1.0, t
        else:
            dt = t - t_last
            w = (w * math.exp(-l * dt) if dt > 0 else w) + 1.0
            t_last = max(t, t_last)
    return w, t_last


def test_accumulator_equals_full_history():
    """The compressed single-number form must equal the full visit-history sum."""
    random.seed(1)
    hl = 1.0
    for _ in range(200):
        n = random.randint(1, 40)
        base = 1_000_000.0
        ts = sorted(base + random.uniform(0, 60 * SECONDS_PER_DAY) for _ in range(n))
        now = ts[-1] + random.uniform(0, 10 * SECONDS_PER_DAY)
        ref = _score_full_history(ts, now, hl)
        w, t_last = _accumulate(ts, hl)
        got = w * math.exp(-_lam(hl) * (now - t_last))
        assert math.isclose(ref, got, rel_tol=1e-6, abs_tol=1e-9)


def test_no_revisit_spike():
    """A single re-visit to a stale file must NOT outrank a workhorse file
    (the documented zoxide/z/fasd bucket bug)."""
    now = 100 * SECONDS_PER_DAY
    hl = 1.0
    random.seed(2)
    work = [now - random.uniform(0, 2 * SECONDS_PER_DAY) for _ in range(30)]
    stale = [now - 80 * SECONDS_PER_DAY - random.uniform(0, 5 * SECONDS_PER_DAY)
             for _ in range(40)]
    stale.append(now - 60)  # one stray re-visit

    ww, wt = _accumulate(sorted(work), hl)
    sw, st = _accumulate(sorted(stale), hl)
    work_score = ww * math.exp(-_lam(hl) * (now - wt))
    stale_score = sw * math.exp(-_lam(hl) * (now - st))
    assert work_score > stale_score


def test_frequency_and_recency_ordering(tmp_path):
    """Through the real store: more-frequent wins at equal recency; more-recent
    wins at equal frequency."""
    now = 50 * SECONDS_PER_DAY
    a = str(tmp_path / "a.py")
    b = str(tmp_path / "b.py")
    for p in (a, b):
        open(p, "w").close()

    # equal recency, different frequency
    for _ in range(10):
        ff.record(a, now=now - 3600)
    for _ in range(3):
        ff.record(b, now=now - 3600)
    assert ff.score(a, now=now) > ff.score(b, now=now)

    # equal frequency, different recency
    c = str(tmp_path / "c.py")
    d = str(tmp_path / "d.py")
    for p in (c, d):
        open(p, "w").close()
    for _ in range(5):
        ff.record(c, now=now - 1 * SECONDS_PER_DAY)
    for _ in range(5):
        ff.record(d, now=now - 20 * SECONDS_PER_DAY)
    assert ff.score(c, now=now) > ff.score(d, now=now)


def test_half_life_decay(tmp_path):
    """With a 1-day half-life, weight halves after exactly one day."""
    now = 1_000_000.0
    p = str(tmp_path / "x.py")
    open(p, "w").close()
    for _ in range(4):
        ff.record(p, now=now)
    s0 = ff.score(p, now=now)
    s1 = ff.score(p, now=now + SECONDS_PER_DAY)
    assert math.isclose(s1, s0 / 2.0, rel_tol=1e-6)


def test_disabled_is_noop(tmp_path):
    """When disabled, record() writes nothing and score() returns 0."""
    ff._config_cached = {"enabled": False, "half_life_days": 1.0, "weight": 40.0,
                         "max_entries": 100000, "max_total": 10000.0}
    ff._config_loaded = True
    p = str(tmp_path / "x.py")
    open(p, "w").close()
    ff.record(p)
    assert ff.score(p) == 0.0
    assert ff.load_store() == {}


def test_weight_cap_rescales_without_dropping(tmp_path):
    """The weight cap rescales summed weight under max_total but never drops
    entries (so a flood can't collapse the store to empty)."""
    ff._config_cached = {"enabled": True, "half_life_days": 1.0, "weight": 40.0,
                         "max_entries": 100000, "max_total": 50.0}
    ff._config_loaded = True
    now = 1_000_000.0
    for i in range(200):
        p = str(tmp_path / f"f{i}.py")
        open(p, "w").close()
        ff.record(p, now=now)
    store = ff.load_store()
    assert len(store) == 200  # nothing dropped — under the count cap
    assert sum(v["w"] for v in store.values()) <= 50.0 + 1e-6  # weight rescaled


def test_count_cap_is_hard_bound_no_cliff():
    """The count cap keeps exactly max_entries even under a uniform flood that
    the old scale-then-threshold aging would have wiped to zero."""
    now = 1_000_000.0
    lam = _lam(1.0)
    data = {f"/p/{i:07d}.py": {"w": 1.0, "t": now} for i in range(200_000)}
    ff._age(data, max_entries=4000, max_total=10000.0, now=now, lam=lam)
    assert len(data) == 4000  # not 0 — no cliff


def test_count_cap_evicts_lowest_frecency():
    """When over the count cap, the lowest-frecency entries are dropped and the
    frequent/recent ones are kept."""
    now = 1_000_000.0
    lam = _lam(1.0)
    data = {}
    for i in range(4000):  # hot: high weight, recent
        data[f"/hot/{i}.py"] = {"w": 10.0, "t": now - 3600}
    for i in range(4000):  # cold: low weight, old
        data[f"/cold/{i}.py"] = {"w": 1.0, "t": now - 30 * SECONDS_PER_DAY}
    ff._age(data, max_entries=4000, max_total=1e9, now=now, lam=lam)
    assert len(data) == 4000
    assert all(k.startswith("/hot/") for k in data)  # cold ones evicted


def test_aging_caps_total(tmp_path):
    """End-to-end: with a small count cap, the live store never exceeds it."""
    ff._config_cached = {"enabled": True, "half_life_days": 1.0, "weight": 40.0,
                         "max_entries": 50, "max_total": 1e9}
    ff._config_loaded = True
    now = 1_000_000.0
    for i in range(200):
        p = str(tmp_path / f"f{i}.py")
        open(p, "w").close()
        ff.record(p, now=now)
    assert len(ff.load_store()) <= 50


def test_prune_missing(tmp_path):
    p = str(tmp_path / "gone.py")
    open(p, "w").close()
    ff.record(p)
    assert ff.score(p) > 0
    os.remove(p)
    removed = ff.prune_missing()
    assert removed == 1
    assert ff.load_store() == {}


def test_score_many_uses_cached_store(tmp_path, monkeypatch):
    """Per-keystroke scoring must not re-read+parse the store every call.

    score_many() reads through load_store_cached(), so N consecutive calls hit
    the disk via load_store() exactly once (within the TTL) — this is the
    hot-path efficiency contract for the picker."""
    p = str(tmp_path / "f.py")
    open(p, "w").close()
    ff.record(p)

    ff._invalidate_store_cache()
    calls = {"n": 0}
    real = ff.load_store

    def counting():
        calls["n"] += 1
        return real()

    monkeypatch.setattr(ff, "load_store", counting)
    for _ in range(25):
        ff.score_many([p])
    assert calls["n"] == 1  # 25 keystrokes, one disk read


def test_record_invalidates_cache(tmp_path):
    """A fresh record() must be visible to the very next score_many() despite
    the read-through cache (record invalidates it)."""
    p = str(tmp_path / "f.py")
    open(p, "w").close()
    # Prime the cache with an empty store.
    assert ff.score_many([p])[p] == 0.0
    # Record, then score again — must reflect the write, not the cached empty.
    ff.record(p)
    assert ff.score_many([p])[p] > 0.0


# ---------------------------------------------------------------------------
# Gateway tiered-blend invariant (TUI + Desktop ranker)
# ---------------------------------------------------------------------------
def _gateway_key(kind_tuple, rel, frec):
    """Mirror the gateway sort key: (rank_tuple, -frecency, len, rel)."""
    return (kind_tuple, -frec, len(rel), rel)


def test_gateway_frecency_never_crosses_tier():
    """An exact-name match (tier 0) must beat a hot substring match (tier 3)
    no matter how frecent the substring match is."""
    exact = ((0, 9), "deep/config.py", 0.0)
    hot_substr = ((3, 16), "config_loader.py", 9999.0)
    winner = sorted([exact, hot_substr], key=lambda c: _gateway_key(*c))[0]
    assert winner[0][0] == 0  # tier 0 wins


def test_gateway_frecency_breaks_within_tier():
    """Among equal-tier matches, the more frecent file leads."""
    cands = [
        ((3, 16), "alpha_handler.py", 0.5),
        ((3, 16), "bravo_handler.py", 12.0),
        ((3, 16), "carol_handler.py", 1.0),
    ]
    order = [c[1] for c in sorted(cands, key=lambda c: _gateway_key(*c))]
    assert order[0] == "bravo_handler.py"
    assert order[-1] == "alpha_handler.py"


# ---------------------------------------------------------------------------
# Universal record hook (covers CLI / TUI / Desktop via context_references)
# ---------------------------------------------------------------------------
def test_at_reference_records_file(tmp_path):
    from agent.context_references import preprocess_context_references_async

    work = tmp_path / "work"
    work.mkdir()
    target = work / "main.py"
    target.write_text("print('hi')\n", encoding="utf-8")

    asyncio.run(
        preprocess_context_references_async(
            "look at @file:main.py", cwd=str(work), context_length=100000
        )
    )
    store = ff.load_store()
    assert any(k.endswith("/main.py") for k in store)
    assert ff.score(str(target)) > 0


def test_at_reference_records_folder(tmp_path):
    from agent.context_references import preprocess_context_references_async

    work = tmp_path / "work"
    work.mkdir()
    (work / "f.py").write_text("x\n", encoding="utf-8")

    asyncio.run(
        preprocess_context_references_async(
            "see @folder:work", cwd=str(tmp_path), context_length=100000
        )
    )
    store = ff.load_store()
    assert any(k.endswith("/work") for k in store)


def test_at_reference_missing_file_not_recorded(tmp_path):
    from agent.context_references import preprocess_context_references_async

    asyncio.run(
        preprocess_context_references_async(
            "look at @file:nope.py", cwd=str(tmp_path), context_length=100000
        )
    )
    assert ff.load_store() == {}

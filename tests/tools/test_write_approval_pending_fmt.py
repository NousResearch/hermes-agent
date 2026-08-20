"""Regression tests for the /skills (and /memory) pending-list rendering.

Covers the unbounded-queue UX fix: the shared ``_fmt_pending_list`` in
hermes_cli/write_approval_commands.py used to render every pending record
oldest-first with no cap. When the background curator stages many skills, the
queue grew without bound, the just-staged (newest) records were pushed to the
bottom, and on non-chunking platforms the tail (the newest, most-actionable
records) was truncated away entirely.

The fix renders newest-first and caps the visible rows with a "… and N more"
footer, so the list stays actionable however large the queue grows.
"""
from __future__ import annotations

import json
import os
import tempfile
import shutil
import time

import pytest


@pytest.fixture
def hermes_home(monkeypatch):
    d = tempfile.mkdtemp(prefix="hermes_wa_fmt_")
    home = os.path.join(d, ".hermes")
    os.makedirs(home)
    monkeypatch.setenv("HERMES_HOME", home)
    yield home
    shutil.rmtree(d, ignore_errors=True)


def _write_record(subsystem, pending_id, *, created_at, summary, origin="foreground"):
    """Directly write a pending record with a controlled created_at, so the
    sort order is deterministic regardless of wall-clock timing."""
    from tools import write_approval as wa

    d = wa._pending_dir(subsystem)
    d.mkdir(parents=True, exist_ok=True)
    record = {
        "id": pending_id,
        "subsystem": subsystem,
        "action": "add",
        "summary": summary,
        "origin": origin,
        "created_at": created_at,
        "payload": {},
    }
    (d / f"{pending_id}.json").write_text(
        json.dumps(record), encoding="utf-8"
    )


def _fmt(subsystem):
    from hermes_cli import write_approval_commands as wac

    return wac._fmt_pending_list(subsystem)


def test_newest_first(hermes_home):
    """Newest (last-staged) record renders at the top, not the bottom."""
    _write_record("skills", "first", created_at=100.0, summary="oldest skill")
    _write_record("skills", "second", created_at=200.0, summary="newest skill")

    out = _fmt("skills")
    lines = out.splitlines()
    first_old = next(i for i, l in enumerate(lines) if "first" in l)
    first_new = next(i for i, l in enumerate(lines) if "second" in l)
    assert first_new < first_old, "newest record must render above oldest"


def test_below_cap_renders_every_record(hermes_home):
    """With a small queue nothing is dropped — behavior unchanged under cap."""
    for i in range(3):
        _write_record("memory", f"m{i}", created_at=float(i), summary=f"memory {i}")

    out = _fmt("memory")
    for i in range(3):
        assert f"m{i}" in out
    assert "more" not in out.lower()


def test_over_cap_renders_first_20_plus_footer(hermes_home):
    """Past the cap, only the first 20 are shown and a remainder footer is added.

    Regression: an uncapped dump let the newest records fall out of the visible
    tail on non-chunking platforms, or grew without bound on full-renderers.
    """
    n = 25
    for i in range(n):
        # stage i=0 first -> created_at smallest; i=24 newest
        _write_record("skills", f"s{i:02d}", created_at=float(i), summary=f"skill {i}")

    out = _fmt("skills")
    lines = out.splitlines()
    rows = [l for l in lines if l.strip().startswith(("s0", "s1", "s2"))]

    # cap is 20 visible rows
    assert len(rows) == 20, f"expected 20 visible rows, got {len(rows)}"
    # the 5 oldest (beyond the 20 newest) are hidden
    assert f"s{24:02d}" in out  # newest shown
    assert f"s{5:02d}" in out   # 20th-newest boundary shown
    assert "5 more" in out      # 25 total - 20 shown = 5 hidden
    # the oldest 5 are NOT rendered
    assert f"s{0:02d}" not in out
    assert f"s{4:02d}" not in out


def test_auto_tag_and_footer_coexist(hermes_home):
    """The [auto] origin tag is still applied, and the footer sits before the
    action epilogue (not appended after 'Apply:' / 'Reject:')."""
    for i in range(21):
        origin = "background_review" if i % 2 == 0 else "foreground"
        _write_record("skills", f"t{i:02d}", created_at=float(i),
                      summary=f"thing {i}", origin=origin)

    out = _fmt("skills")
    lines = out.splitlines()
    footer_idx = next(i for i, l in enumerate(lines) if "more" in l.lower())
    apply_idx = next(i for i, l in enumerate(lines) if "Apply:" in l)
    assert footer_idx < apply_idx, "footer must render before the action epilogue"
    assert any("[auto]" in l for l in lines), "background_review tag must render"

"""Tests: roster previews never end up with unbalanced markdown markers.

Why: ``profiles.list`` serves each bot's latest message as an 80-char excerpt
(``last_session.preview`` / ``canonical_session.preview``). Clients render it as
one plain-text line and strip markdown first, but stripping only works on
*paired* markers. Assistant replies often open with a bolded sentence longer
than 80 chars (``**Yes. Your existing credits can be used ...**``), so the
excerpt cut the closing ``**`` off and every client showed a dangling ``**`` at
the start of the row. The gateway now flattens markdown on the whole message
first and truncates the plain text afterwards.
"""

from __future__ import annotations

import pytest

import tui_gateway.server as srv


LONG_BOLD = (
    "**Yes. Your existing credits can be used for the new plan, and they carry over "
    "automatically when you upgrade.** Nothing else to do."
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    h.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


def _seed(home, *, title, text):
    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    db.create_session("s1", "cli")
    db.append_message("s1", "user", "do I lose my credits?", timestamp=1000)
    db.append_message("s1", "assistant", text, timestamp=1001)
    with db._lock:
        db._conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, "s1"))
    db.close()


def _default_row():
    profiles = srv._methods["profiles.list"](1, {})["result"]["profiles"]
    return next(p for p in profiles if p["name"] == "default")


@pytest.mark.parametrize("field,title", [("canonical_session", "Bot Chat"),
                                         ("last_session", "Scratch")])
def test_long_bold_opening_does_not_leak_markers(home, field, title):
    _seed(home, title=title, text=LONG_BOLD)

    preview = _default_row()[field]["preview"]

    assert "**" not in preview
    assert preview.startswith("Yes. Your existing credits can be used")
    # Still an excerpt: the 80-char budget applies to the plain text.
    assert preview.endswith("...")
    assert len(preview) <= 83


def test_inline_markdown_is_flattened_in_preview(home):
    _seed(home, title="Bot Chat",
          text="This isn't a `port` conflict. See [the docs](https://example.com/x).")

    preview = _default_row()["canonical_session"]["preview"]

    assert preview == "This isn't a port conflict. See the docs."


def test_plain_short_preview_is_unchanged(home):
    _seed(home, title="Bot Chat", text="just a plain reply")

    assert _default_row()["canonical_session"]["preview"] == "just a plain reply"

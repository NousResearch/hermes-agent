"""Stage-7 post-write read-back tests for tools/write_gate.py (issue #140848).

The 7-stage pre-write acceptance pipeline that also lives in write_gate.py
belongs to the write-gate feature's own PR; only the Stage-7 read-back
assertion is in scope for #140848.  These tests therefore call
``verify_post_write`` directly against main's ``MemoryStore`` (no ``doc:``
targets, no pre-write pipeline) — so they have no dependency on the
chain-doc / failure-class-registry surfaces that the write-gate feature's full
suite exercises.

Invariant per the Enough1122 review: a patch/replace whose new text contains
the old span lands correctly and must NOT be flagged.  The pre-fix bare
``old_text in entries`` assertion produced false read-back failures (and on
retry, ``patch`` re-spliced with ``replace(old_text, new_text, 1)`` producing
duplicated extensions like "pytest -x -q -x -q").
"""
import json

from tools.write_gate import verify_post_write
from tools.memory_tool_store import MemoryStore


def _make_store(limit: int = 5000, entries: list[str] | None = None):
    """In-memory MemoryStore preset with anchor entries (no disk I/O).

    ``MemoryStore._entries_for`` resolves ``memory`` to ``memory_entries``
    in memory, so ``verify_post_write`` reads these directly.
    """
    store = MemoryStore(memory_char_limit=limit, user_char_limit=3000)
    store.memory_entries = list(entries or [])
    store.user_entries = []
    return store


def _redirect_audit(monkeypatch, tmp_path):
    """Redirect the JSONL audit log to a temp file."""
    monkeypatch.setattr("tools.write_gate._audit_log_path",
                        lambda: tmp_path / "write_gate_audit.jsonl")


class TestPostWriteReadbackEradication:
    """Stage-7 read-back: verify_post_write asserts old_text is gone."""

    def test_old_text_eradicated_after_replace(self, tmp_path, monkeypatch):
        """A landed replace removes the old span → read-back admits (None)."""
        _redirect_audit(monkeypatch, tmp_path)
        store = _make_store(entries=["The deploy target is Fly.io with manual approval."])
        # Simulate the landed write: old span replaced, old_text no longer present.
        store.memory_entries = ["The deploy target is Fly.io with automated canary."]
        assert verify_post_write(store, "memory", "replace",
                                 "The deploy target is Fly.io with manual approval.") is None

    def test_old_text_eradicated_after_remove(self, tmp_path, monkeypatch):
        """A landed remove drops the entry → read-back admits (None)."""
        _redirect_audit(monkeypatch, tmp_path)
        store = _make_store(entries=["Temporary staging note that should be removed."])
        store.memory_entries = []  # removed
        assert verify_post_write(store, "memory", "remove",
                                 "Temporary staging note that should be removed.") is None

    def test_surviving_old_text_is_flagged(self, tmp_path, monkeypatch):
        """If old_text persists with no content explanation, read-back refuses."""
        _redirect_audit(monkeypatch, tmp_path)
        store = _make_store(entries=["An entry that was NOT actually removed."])
        result = verify_post_write(store, "memory", "remove",
                                   "An entry that was NOT actually removed.")
        assert result is not None
        parsed = json.loads(result)
        assert parsed["failure_class"] == "readback_failed"

    def test_span_extension_with_overlapping_text_not_flagged(self, tmp_path, monkeypatch):
        """Enough1122 #140848: a patch extending a span (new text CONTAINS the
        old span) lands correctly and must NOT be flagged as a failed read-back.

        Pre-fix (bare ``old_text in entries``) this was a false positive because
        ``pytest`` survives inside ``pytest -x -q``; the ``content`` param makes
        the read-back ignore a span the new text legitimately carries.
        """
        _redirect_audit(monkeypatch, tmp_path)
        new_text = "Run pytest -x -q to verify the suite."
        store = _make_store(entries=[new_text])  # landed patch: "pytest" -> "pytest -x -q"
        assert verify_post_write(store, "memory", "patch", "pytest",
                                 content=new_text) is None

    def test_surviving_span_without_content_explanation_is_flagged(self, tmp_path, monkeypatch):
        """Same surviving span, but the new text does NOT carry it → flagged."""
        _redirect_audit(monkeypatch, tmp_path)
        store = _make_store(entries=["pytest still here, unrelated text"])
        result = verify_post_write(store, "memory", "patch", "pytest",
                                   content="totally different change")
        assert result is not None
        assert json.loads(result)["failure_class"] == "readback_failed"

    def test_non_readback_actions_return_none(self, tmp_path, monkeypatch):
        """``add`` and empty ``old_text`` never trip the read-back (None)."""
        _redirect_audit(monkeypatch, tmp_path)
        store = _make_store(entries=["hello world"])
        assert verify_post_write(store, "memory", "add", "hello world",
                                 content="hello world") is None
        assert verify_post_write(store, "memory", "replace", None,
                                 content="x") is None
        assert verify_post_write(store, "memory", "patch", "",
                                 content="x") is None

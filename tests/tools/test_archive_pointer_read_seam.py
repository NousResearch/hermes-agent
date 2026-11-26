"""Regression for #156164 item 5 at the read_file seam: a missing archive is re-resolved by name.

The helper level is covered in ``test_archive_pointer_portable.py``; this drives the seam
``read_file_tool`` calls, so a refactor that stops wiring the re-resolution in fails here.
"""
import json

from tools.file_tools import _resolve_missing_archive


class _ReadResult:
    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return dict(self._payload)


class _Ops:
    """A host-path backend whose reads all miss, like a stub whose home has moved."""

    def __init__(self, found_at=None, body="recovered bytes"):
        self.found_at = found_at
        self.body = body
        self.reads = []

    def read_file(self, path, offset, limit):
        self.reads.append(path)
        if self.found_at is not None and path == self.found_at:
            return _ReadResult({"content": self.body, "path": path})
        return _ReadResult({"error": f"File not found: {path}"})


def test_a_missing_stub_is_re_read_from_this_home_and_its_error_clears(tmp_path, monkeypatch):
    """The archive lives here under the same name: the read succeeds, no 'expired' is attached."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    target = tmp_path / "cache" / "pruned" / "s_1_terminal_9f8e7d6c.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"body": 1}', encoding="utf-8")
    ops = _Ops(found_at=str(target))
    original = _ReadResult({"error": "File not found: ~/.hermes/cache/pruned/s_1_terminal_9f8e7d6c.json"})

    _result, payload, error = _resolve_missing_archive(
        "~/.hermes/cache/pruned/s_1_terminal_9f8e7d6c.json", original, dict(original.to_dict()), ops,
        1, 100, "/some/where", "default")

    assert error == ""
    assert payload.get("expired") is None, "a recovered archive must not be reported expired"
    assert payload.get("content") == "recovered bytes"
    assert ops.reads == [str(target)], "exactly the rehomed path is retried, once"


def test_an_archive_absent_here_still_reports_expired(tmp_path, monkeypatch):
    """No copy in this home: the model keeps the expired notice that tells it to re-run the tool."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ops = _Ops(found_at=None)
    original = _ReadResult({"error": "File not found: ~/.hermes/cache/pruned/gone.json"})
    payload = dict(original.to_dict())

    _result, payload, error = _resolve_missing_archive(
        "~/.hermes/cache/pruned/gone.json", original, payload, ops, 1, 100, "/x", "default")

    assert error.startswith("File not found:")
    assert payload.get("expired") is True
    assert "no longer recoverable" in payload.get("hint", "")
    assert ops.reads == [], "nothing to retry when this home has no such archive"


def test_a_path_that_is_not_an_archive_is_never_retried(tmp_path, monkeypatch):
    """Ordinary missing files keep their plain not-found answer, with no archive wording."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ops = _Ops(found_at=None)
    original = _ReadResult({"error": "File not found: ~/notes/missing.md"})
    payload = dict(original.to_dict())

    _result, payload, error = _resolve_missing_archive(
        "~/notes/missing.md", original, payload, ops, 1, 100, "/x", "default")

    assert error.startswith("File not found:")
    assert payload.get("expired") is None
    assert ops.reads == []
    assert "no longer recoverable" not in json.dumps(payload)
"""Shared assertions for desktop pane-read tools (terminal / preview / window)."""

import json

import tools.read_terminal_tool as rt

PASSWORD = "hunter2-fake-password"
TOKEN = "tok_fake_0123456789abcdef"
PAYLOAD = {
    "text": f"$ curl https://alice:{PASSWORD}@example.test/a?x=1",
    "url": f"https://example.test/app?token={TOKEN}&page=2",
}


def assert_redacted(out: str) -> None:
    assert PASSWORD not in out
    assert TOKEN not in out
    data = json.loads(out)
    assert "example.test" in data["text"]
    assert "page=2" in data["url"]


def assert_fail_closed(monkeypatch, run) -> None:
    def boom(*_a, **_kw):
        raise RuntimeError("redactor broke")

    monkeypatch.setattr(rt, "redact_sensitive_text", boom)
    out = run(lambda **_: json.dumps(PAYLOAD))
    assert PASSWORD not in out and TOKEN not in out
    data = json.loads(out)
    assert data["text"] == rt.REDACT_FAILED
    assert data["url"] == rt.REDACT_FAILED


def assert_non_string_leaves_kept(run) -> None:
    payload = {"n": 3, "ok": True, "none": None, "f": 1.5, "rows": [1, {"t": "plain", "k": False}]}
    assert json.loads(run(lambda **_: json.dumps(payload))) == payload

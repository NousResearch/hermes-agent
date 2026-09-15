import json

import tools.approval as approval


def test_description_is_redacted_too(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.redact.redact_sensitive_text", lambda text, **kwargs: text.replace("SENSITIVE_SENTINEL", "[REDACTED]"))
    approval._log_approval_event("test", "plugin description SENSITIVE_SENTINEL", "command SENSITIVE_SENTINEL", {"approved": True})
    raw = (tmp_path / "logs/approvals.jsonl").read_text()
    assert "SENSITIVE_SENTINEL" not in raw
    assert json.loads(raw)["approved"] is True


def test_default_home_and_redaction_failure_are_safe(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    def fail_redaction(*args, **kwargs):
        raise ValueError("redactor unavailable")

    monkeypatch.setattr("agent.redact.redact_sensitive_text", fail_redaction)
    approval._log_approval_event("test", "SENSITIVE_SENTINEL", "SENSITIVE_SENTINEL", {"approved": False})
    path = approval.get_hermes_home() / "logs/approvals.jsonl"
    assert path == tmp_path / ".hermes/logs/approvals.jsonl"
    raw = path.read_text()
    assert "SENSITIVE_SENTINEL" not in raw
    assert json.loads(raw)["description"] == "[redaction unavailable]"
    assert not (tmp_path / "logs/approvals.jsonl").exists()

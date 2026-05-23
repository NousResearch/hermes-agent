import json
import stat

import pytest

from tools import slash_confirm


def _events(audit_dir):
    rows = []
    for path in sorted(audit_dir.glob("*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    return rows


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture(autouse=True)
def _clear_slash_confirm_state():
    slash_confirm._pending.clear()
    yield
    slash_confirm._pending.clear()


@pytest.mark.asyncio
async def test_invalid_confirmation_choice_is_blocked_and_audited(tmp_path, monkeypatch):
    audit_dir = tmp_path / "audit"
    monkeypatch.setenv("HERMES_AUDIT_DIR", str(audit_dir))
    calls = []

    async def handler(choice):
        calls.append(choice)
        return "ran"

    slash_confirm.register("session-1", "confirm-1", "reload-mcp", handler)
    result = await slash_confirm.resolve("session-1", "confirm-1", "approve")

    assert result is None
    assert calls == []
    assert slash_confirm.get_pending("session-1") is not None
    assert _mode(audit_dir) == 0o700
    audit_file = next(audit_dir.glob("*.jsonl"))
    assert _mode(audit_file) == 0o600
    events = _events(audit_dir)
    assert events[-1]["event_type"] == "slash_confirmation.decision"
    assert events[-1]["decision"] == "blocked"
    assert events[-1]["status"] == "blocked_invalid_choice"
    assert "session-1" not in audit_file.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_cancel_confirmation_is_denied_and_audited(tmp_path, monkeypatch):
    audit_dir = tmp_path / "audit"
    monkeypatch.setenv("HERMES_AUDIT_DIR", str(audit_dir))
    calls = []

    async def handler(choice):
        calls.append(choice)
        return "cancelled"

    slash_confirm.register("session-2", "confirm-2", "new", handler)
    result = await slash_confirm.resolve("session-2", "confirm-2", "cancel")

    assert result == "cancelled"
    assert calls == ["cancel"]
    events = _events(audit_dir)
    assert events[-1]["decision"] == "denied"
    assert events[-1]["approval_scope"] == "cancel"
    assert events[-1]["status"] == "cancelled_by_user"


def test_confirmation_choice_requires_exact_choice():
    assert slash_confirm.normalize_confirmation_choice("once") == "once"
    assert slash_confirm.normalize_confirmation_choice(" always ") == "always"
    assert slash_confirm.normalize_confirmation_choice("approve") is None
    assert slash_confirm.normalize_confirmation_choice("yes") is None
    assert slash_confirm.normalize_confirmation_choice("always approve") is None

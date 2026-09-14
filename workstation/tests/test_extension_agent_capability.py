from __future__ import annotations

import json

from tools import workstation_extensions as extension_tools


class _Manager:
    def extract_extension_id(self, value: str) -> str:
        assert value == "https://chromewebstore.google.com/detail/example/cjpalhdlnbpafiamejdnhcphjbkeiagm"
        return "cjpalhdlnbpafiamejdnhcphjbkeiagm"

    def download_crx(self, extension_id: str) -> bytes:
        assert extension_id == "cjpalhdlnbpafiamejdnhcphjbkeiagm"
        return b"fixture"

    def inspect_crx(self, extension_id: str, crx: bytes):
        assert crx == b"fixture"
        return {"name": "Fixture extension", "version": "1.0"}, {
            "risk_level": "high", "reasons": ["privileged permissions: cookies"],
            "permissions": ["cookies"], "host_permissions": [],
        }

    def install_from_bytes(self, extension_id: str, crx: bytes):
        return {"id": extension_id, "version": "1.0", "path": f"C:/extensions/{extension_id}"}

    def uninstall_extension(self, extension_id: str) -> bool:
        raise AssertionError("verified runtime load must not roll back")


def test_agent_extension_install_runs_policy_approval_load_verify_and_journal(monkeypatch, tmp_path):
    events = []

    class Journal:
        task_id = "kanban-card-1"

        def __init__(self, *_args, **_kwargs):
            pass

        def record(self, kind, message, **kwargs):
            events.append((kind.value, message, kwargs.get("metadata", {}).get("event")))

    monkeypatch.setattr(extension_tools, "ChromeExtensionManager", _Manager)
    monkeypatch.setattr(extension_tools, "ExecutionJournal", Journal)
    monkeypatch.setattr(extension_tools, "_desktop_session", lambda: True)
    monkeypatch.setattr(extension_tools, "_approval", lambda *_args: (True, "approved"))
    monkeypatch.setattr(
        extension_tools,
        "_controller",
        lambda action, args, **_kwargs: {"extension_id": args["extension_id"], "loaded": True, "version": "1.0"},
    )

    result = json.loads(extension_tools._install(
        {"extension": "https://chromewebstore.google.com/detail/example/cjpalhdlnbpafiamejdnhcphjbkeiagm"},
        task_id="kanban-card-1", session_id="desktop-session-1",
    ))

    assert result["success"] is True
    assert result["runtime"]["loaded"] is True
    assert [event for _, _, event in events] == [
        "EXTENSION_DOWNLOAD_STARTED", "EXTENSION_POLICY_CHECK", "EXTENSION_APPROVAL_REQUESTED",
        "EXTENSION_APPROVED", "EXTENSION_INSTALLED", "EXTENSION_VERIFIED",
    ]


def test_agent_extension_install_fails_closed_when_runtime_does_not_verify(monkeypatch):
    class FailingManager(_Manager):
        rolled_back = False

        def uninstall_extension(self, extension_id: str) -> bool:
            self.rolled_back = True
            return True

    manager = FailingManager()
    monkeypatch.setattr(extension_tools, "ChromeExtensionManager", lambda: manager)
    monkeypatch.setattr(extension_tools, "_desktop_session", lambda: True)
    monkeypatch.setattr(extension_tools, "_approval", lambda *_args: (True, "approved"))
    monkeypatch.setattr(extension_tools, "_controller", lambda *_args, **_kwargs: {"loaded": False})
    result = json.loads(extension_tools._install(
        {"extension": "https://chromewebstore.google.com/detail/example/cjpalhdlnbpafiamejdnhcphjbkeiagm"},
        task_id="task-1", session_id="desktop-session-1",
    ))
    assert result["success"] is False
    assert result["code"] == "extension_load_failed"
    assert manager.rolled_back is True

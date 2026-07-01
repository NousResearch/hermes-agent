import json
from types import SimpleNamespace

from tools import browser_dialog_tool as mod


class FakeSupervisor:
    def __init__(self, snapshot):
        self._snapshot = snapshot
        self.responded = False

    def snapshot(self):
        return self._snapshot

    def respond_to_dialog(self, **kwargs):
        self.responded = True
        return {
            "ok": True,
            "dialog": {"id": kwargs.get("dialog_id") or "dlg-1"},
        }


def _snapshot(url: str, *, frame_id: str = "frame-1"):
    dialog = SimpleNamespace(id="dlg-1", frame_id=frame_id)
    return SimpleNamespace(
        pending_dialogs=(dialog,),
        frame_tree={
            "top": {
                "frame_id": frame_id,
                "url": url,
            },
            "children": [],
        },
    )


def test_browser_dialog_blocks_private_page(monkeypatch):
    supervisor = FakeSupervisor(_snapshot("http://127.0.0.1:8080/admin"))
    monkeypatch.setattr(mod, "SUPERVISOR_REGISTRY", {"task-1": supervisor})
    monkeypatch.setattr(
        "tools.browser_tool._eval_ssrf_guard_active",
        lambda task_id: True,
    )

    result = json.loads(
        mod.browser_dialog("accept", dialog_id="dlg-1", task_id="task-1")
    )

    assert result["success"] is False
    assert "private/internal page" in result["error"]
    assert "127.0.0.1" in result["error"]
    assert supervisor.responded is False


def test_browser_dialog_allows_public_page(monkeypatch):
    supervisor = FakeSupervisor(_snapshot("https://example.com/confirm"))
    monkeypatch.setattr(mod, "SUPERVISOR_REGISTRY", {"task-1": supervisor})
    monkeypatch.setattr(
        "tools.browser_tool._eval_ssrf_guard_active",
        lambda task_id: True,
    )

    result = json.loads(
        mod.browser_dialog("accept", dialog_id="dlg-1", task_id="task-1")
    )

    assert result["success"] is True
    assert supervisor.responded is True


def test_browser_dialog_skips_probe_when_guard_inactive(monkeypatch):
    supervisor = FakeSupervisor(_snapshot("http://127.0.0.1:8080/admin"))
    monkeypatch.setattr(mod, "SUPERVISOR_REGISTRY", {"task-1": supervisor})
    monkeypatch.setattr(
        "tools.browser_tool._eval_ssrf_guard_active",
        lambda task_id: False,
    )

    def fail_if_called(url):
        raise AssertionError("URL safety check should not run when guard is inactive")

    monkeypatch.setattr("tools.browser_tool._is_safe_url", fail_if_called)

    result = json.loads(
        mod.browser_dialog("accept", dialog_id="dlg-1", task_id="task-1")
    )

    assert result["success"] is True
    assert supervisor.responded is True

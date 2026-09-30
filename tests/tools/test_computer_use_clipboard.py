"""computer_use clipboard actions (port of code-yeongyu/oh-my-openagent#9053 to the cua-driver backend).

cua-driver 0.21+ ships ``clipboard_read`` / ``clipboard_write``; hermes' action enum never exposed them, so the
model could only move text through ``type`` (slow, layout-dependent) and could not read what the user copied.
"""
import json
from types import SimpleNamespace

import pytest

from tools.computer_use.backend import ActionResult, ComputerUseBackend
from tools.computer_use.cua_backend_clipboard import _ClipboardMixin
from tools.computer_use.schema import COMPUTER_USE_SCHEMA
from tools.computer_use.tool import _ACTIONS, _dispatch


class _Backend:
    def __init__(self):
        self.calls = []

    def clipboard_read(self):
        self.calls.append(("clipboard_read", {}))
        return ActionResult(ok=True, action="clipboard_read", meta={"types": ["text/plain"], "text": "copied", "supported": True})

    def clipboard_write(self, **kw):
        self.calls.append(("clipboard_write", kw))
        return ActionResult(ok=True, action="clipboard_write", meta={"types": ["text/plain"]})


def test_clipboard_actions_are_exposed_and_approval_gated():
    # Both actions in the advertised enum, and both behind the approval prompt: write mutates the user's clipboard,
    # read discloses whatever the user last copied (routinely a password) — neither is a free read like capture.
    enum = COMPUTER_USE_SCHEMA["parameters"]["properties"]["action"]["enum"]
    assert {"clipboard_read", "clipboard_write"} <= set(enum)
    assert _ACTIONS["clipboard_read"].destructive and _ACTIONS["clipboard_write"].destructive
    assert not _ACTIONS["clipboard_read"].input  # no sticky-target / app= guard: the clipboard has no window


def test_clipboard_dispatch_returns_flat_payload_without_verdict():
    backend = _Backend()
    out = json.loads(_dispatch(backend, "clipboard_read", {}))
    assert out == {"ok": True, "action": "clipboard_read", "types": ["text/plain"], "text": "copied"}
    assert "verdict" not in out  # nothing on screen to re-capture

    out = json.loads(_dispatch(backend, "clipboard_write", {"text": "hello"}))
    assert out["ok"] is True and out["types"] == ["text/plain"]
    assert backend.calls[-1] == ("clipboard_write", {"text": "hello"})


@pytest.mark.parametrize("args", [{}, {"text": "a", "file_path": "/x"}])
def test_clipboard_write_requires_exactly_one_payload(args):
    backend = _Backend()
    out = json.loads(_dispatch(backend, "clipboard_write", args))
    assert "exactly one" in out["error"] and backend.calls == []


def test_base_backend_refuses_with_stable_code():
    class Minimal(ComputerUseBackend):  # a third-party backend that predates the clipboard hooks
        start = stop = lambda self: None
        is_available = lambda self: True
        capture = click = drag = scroll = type_text = key = list_apps = focus_app = set_value = lambda self, *a, **k: None

    res = Minimal().clipboard_read()
    assert res.ok is False and res.code == "clipboard_unsupported"
    assert Minimal().clipboard_write(text="x").code == "clipboard_unsupported"


def test_cua_mixin_refuses_only_after_discovery_proves_absence_and_forwards_include_text():
    class Cua(_ClipboardMixin):
        def __init__(self, tools, discovered=True):
            self._session = SimpleNamespace(capabilities_discovered=discovered, _has_tool=lambda n: n in tools)
            self.actions = []

        def _action(self, name, args):
            self.actions.append((name, args))
            return ActionResult(ok=True, action=name, meta={"types": []})

    old = Cua(tools={"click"})
    assert old.clipboard_read().code == "clipboard_unsupported" and old.actions == []

    undiscovered = Cua(tools=set(), discovered=False)  # tools/list not in yet: let the call report the real error
    assert undiscovered.clipboard_read().ok is True

    new = Cua(tools={"clipboard_read", "clipboard_write"})
    new.clipboard_read()
    new.clipboard_write(image_path="/home/u/shot.png")
    assert new.actions == [("clipboard_read", {"include_text": True}), ("clipboard_write", {"image_path": "/home/u/shot.png"})]

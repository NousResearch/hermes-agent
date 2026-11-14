"""Native quick-alias routing contracts (regression for #86508)."""

import json
import threading
import types

import pytest

from tui_gateway import server


class _Worker:
    def __init__(self):
        self.commands = []

    def run(self, command):
        self.commands.append(command)
        return "handled by worker"


@pytest.fixture(autouse=True)
def session(monkeypatch):
    record = {
        "agent": types.SimpleNamespace(),
        "session_key": "quick-alias-session",
        "history": [],
        "history_lock": threading.Lock(),
        "running": False,
        "slash_worker": _Worker(),
    }
    monkeypatch.setitem(server._sessions, "sid", record)
    return record


def _slash_exec(command: str) -> dict:
    return server.handle_request(
        {
            "id": command,
            "method": "slash.exec",
            "params": {"session_id": "sid", "command": command},
        }
    )


def test_slash_exec_routes_only_native_quick_alias_to_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {
            "quick_commands": {
                "capture": {"type": "alias", "target": "/image latest.png"}
            }
        },
    )

    def _dispatch(rid, params):
        calls.append(params)
        return {"id": rid, "result": {"type": "alias", "target": "/image latest.png"}}

    monkeypatch.setitem(server._methods, "command.dispatch", _dispatch)
    assert _slash_exec("capture describe this image")["result"] == {
        "type": "alias",
        "target": "/image latest.png",
    }
    assert calls == [
        {"name": "capture", "arg": "describe this image", "session_id": "sid"}
    ]


@pytest.mark.parametrize(
    "quick_commands",
    [
        {"capture": {"type": "alias", "target": "/not-native arg"}},
        {
            "capture": {"type": "alias", "target": "other-quick"},
            "other-quick": {"type": "exec", "command": "printf chained"},
        },
        {"capture": {"type": "alias", "target": "/research"}},
        {"capture": {"type": "exec", "command": "printf shell"}},
        {},
    ],
    ids=("non-native-alias", "chained-alias", "skill-alias", "shell", "unknown"),
)
def test_slash_exec_keeps_extensions_in_worker(monkeypatch, quick_commands, session):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"quick_commands": quick_commands})
    assert _slash_exec("capture keep worker routing")["result"] == {
        "output": "handled by worker"
    }
    assert session["slash_worker"].commands == ["capture keep worker routing"]


def test_slash_exec_keeps_worker_when_alias_config_read_fails(monkeypatch, session):
    def _unavailable():
        raise ValueError("invalid quick-command configuration")

    monkeypatch.setattr(server, "_load_cfg", _unavailable)
    assert _slash_exec("capture keep worker routing")["result"] == {"output": "handled by worker"}
    assert session["slash_worker"].commands == ["capture keep worker routing"]


def test_slash_exec_loads_quick_config_only_for_extension_names(monkeypatch):
    loads = 0

    def _load():
        nonlocal loads
        loads += 1
        return {"quick_commands": {"model": {"type": "alias", "target": "/image shadow.png"}}}

    monkeypatch.setattr(server, "_load_cfg", _load)
    monkeypatch.setattr(server, "_mirror_slash_side_effects", lambda *_args: "")
    assert _slash_exec("model")["result"]["output"] == "Current model: (unknown)"
    assert loads == 0
    assert _slash_exec("unknown-one")["result"]["output"] == "handled by worker"
    assert _slash_exec("unknown-two")["result"]["output"] == "handled by worker"
    assert loads == 2


@pytest.mark.parametrize("name", ["CaptureLatest", "capturelatest", "CAPTURELATEST"])
def test_slash_exec_uses_catalog_case_canonicalization_for_quick_alias(monkeypatch, name):
    calls = []
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {
            "quick_commands": {
                "CaptureLatest": {"type": "alias", "target": "/IMAGE latest.png"}
            }
        },
    )
    monkeypatch.setitem(
        server._methods,
        "command.dispatch",
        lambda rid, params: calls.append(params)
        or {"id": rid, "result": {"type": "alias", "target": "/IMAGE latest.png"}},
    )
    assert _slash_exec(f"{name} describe")["result"]["type"] == "alias"
    assert calls[0]["name"] == "CaptureLatest"


@pytest.mark.parametrize(
    ("name", "target"),
    [("Capture", "/image exact.png"), ("capture", "image folded.png"), ("CAPTURE", "/image exact.png")],
)
def test_native_alias_reads_session_config_and_real_dispatch(tmp_path, session, name, target):
    home = tmp_path / "served-profile"
    home.mkdir()
    (home / "config.yaml").write_text(json.dumps({
        "quick_commands": {
            "Capture": {"type": "alias", "target": "/image exact.png"},
            "capture": {"type": "alias", "target": "image folded.png"},
        }
    }))
    session["profile_home"] = str(home)
    response = _slash_exec(f"{name} describe this image")
    assert response["result"] == {"type": "alias", "target": target}
    assert session["slash_worker"].commands == []

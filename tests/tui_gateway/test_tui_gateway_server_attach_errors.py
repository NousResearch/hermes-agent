"""An attach payload over its cap is refused in one sentence naming its size and the cap.

``image.attach_bytes`` (native Desktop's remote pastes), ``pdf.attach`` and ``file.attach`` decode
base64 payloads in ``tui_gateway/prompt_attachments.py``, which refuses a payload over its cap
either before decoding (the base64 text is too long) or after. Either way the client reads main's
wording, ``<what> too large (<n> bytes; cap is <m> MB)``.
"""

import base64
import sys
import threading
import types

import pytest

from tui_gateway import server

_MIB = 1024 * 1024


@pytest.fixture
def session(tmp_path, monkeypatch):
    fake_cli = types.ModuleType("cli")
    fake_cli._IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
    monkeypatch.setitem(sys.modules, "cli", fake_cli)
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    record = {
        "profile_home": str(tmp_path / "profile"),
        "cwd": str(workspace),
        "attached_images": [],
        "image_counter": 0,
        "running": False,
        "history_lock": threading.Lock(),
    }
    monkeypatch.setattr(server, "_sessions", {"oversize": record})
    return "oversize"


# One byte over a 1 MiB cap is refused after decoding; 2 MiB is refused from the base64 length.
@pytest.mark.parametrize("size", [_MIB + 1, 2 * _MIB + 5])
def test_an_oversized_image_names_its_size_and_the_cap(session, monkeypatch, size):
    monkeypatch.setattr(server, "_ATTACH_BYTES_MAX_BYTES", _MIB)
    payload = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * (size - 8)).decode("ascii")

    reply = server._methods["image.attach_bytes"](1, {"session_id": session, "content_base64": payload})

    assert reply["error"]["code"] == 4018
    assert reply["error"]["message"] == f"image too large ({size} bytes; cap is 1 MB)"


@pytest.mark.parametrize("size", [_MIB + 1, 2 * _MIB + 5])
def test_an_oversized_inline_file_names_its_size_and_the_cap(session, monkeypatch, size):
    monkeypatch.setattr(server, "_ATTACHMENT_MAX_BYTES", _MIB)
    data_url = "data:application/octet-stream;base64," + base64.b64encode(b"0" * size).decode("ascii")

    reply = server._methods["file.attach"](1, {"session_id": session, "name": "big.bin", "data_url": data_url})

    assert reply["error"]["message"] == f"attachment too large ({size} bytes; cap is 1 MB)"

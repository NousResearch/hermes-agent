"""Images staged by same-profile RPC sessions retain their owner and bytes (#75761)."""

import base64
from datetime import datetime
from pathlib import Path

import pytest

from tui_gateway import server


@pytest.mark.parametrize("method", ["image.attach_bytes", "clipboard.paste"])
def test_same_second_images_keep_session_payload_and_detach_ownership(
    tmp_path, monkeypatch, method,
):
    sessions = {
        sid: {"profile_home": str(tmp_path), "attached_images": []}
        for sid in ("first", "second")
    }
    monkeypatch.setattr(server, "_sessions", sessions)

    class FrozenDateTime:
        @staticmethod
        def now():
            return datetime(2026, 9, 26, 12, 0, 0)

    monkeypatch.setattr(server, "datetime", FrozenDateTime)
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9ZphwAAAAASUVORK5CYII="
    )
    payloads = (png + b"first image", png + b"second image")
    captured = iter(payloads)

    def save_clipboard_image(path):
        # The OS capture boundary supplies pixels; staging still uses real file I/O.
        path.write_bytes(next(captured))
        return True

    monkeypatch.setattr("hermes_cli.clipboard.save_clipboard_image", save_clipboard_image)
    paths = []
    for sid, payload in zip(sessions, payloads):
        params = {"session_id": sid}
        if method == "image.attach_bytes":
            params.update(filename="image.png", content_base64=base64.b64encode(payload).decode())
        response = server._methods[method](1, params)
        assert "error" not in response, response
        paths.append(response["result"]["path"])

    assert len(set(paths)) == len(payloads)
    for sid, path, payload in zip(sessions, paths, payloads):
        assert Path(path).read_bytes() == payload
        assert Path(path).parent == tmp_path / "images"
        assert sessions[sid]["attached_images"] == [path]
    detached = server._methods["image.detach"](2, {"session_id": "first", "path": paths[0]})
    assert detached["result"] == {"detached": True, "count": 0}
    assert sessions["second"]["attached_images"] == [paths[1]]
    assert Path(paths[1]).read_bytes() == payloads[1]

import base64
import importlib
import threading
from pathlib import Path


def _png_data_url() -> str:
    # 1x1 transparent PNG
    raw = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def _install_ready_session(server, sid="remote-image-session"):
    ready = threading.Event()
    ready.set()
    server._sessions[sid] = {
        "session_key": f"test/{sid}",
        "agent_ready": ready,
        "agent_error": None,
        "attached_images": [],
    }
    return sid


def test_image_attach_accepts_remote_data_url_even_when_client_path_is_windows(tmp_path, monkeypatch):
    server = importlib.import_module("tui_gateway.server")
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    sid = _install_ready_session(server)

    response = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": "img-1",
            "method": "image.attach",
            "params": {
                "session_id": sid,
                "path": r"C:\\Users\\eddy\\AppData\\Roaming\\Hermes\\composer-images\\clip.png",
                "name": "clip.png",
                "data_url": _png_data_url(),
            },
        }
    )

    assert response["result"]["attached"] is True
    stored_path = Path(response["result"]["path"])
    assert stored_path.exists()
    assert stored_path.parent == tmp_path / "images"
    assert stored_path.suffix == ".png"
    assert str(stored_path) in server._sessions[sid]["attached_images"]
    assert response["result"]["text"] == "[User attached image: clip.png]"


def test_image_attach_rejects_non_image_data_url(tmp_path, monkeypatch):
    server = importlib.import_module("tui_gateway.server")
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    sid = _install_ready_session(server, "remote-image-reject")

    bad = "data:text/plain;base64," + base64.b64encode(b"not an image").decode("ascii")
    response = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": "img-2",
            "method": "image.attach",
            "params": {"session_id": sid, "path": r"C:\\tmp\\note.txt", "data_url": bad},
        }
    )

    assert response["error"]["code"] == 4016
    assert "unsupported image data" in response["error"]["message"]

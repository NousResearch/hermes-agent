"""Image attachment diagnostics must preserve paths while retaining path-plus-text input."""

import base64

import pytest

from tui_gateway import server


@pytest.fixture
def attachment_session(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    session = {
        "session_key": "image-paths",
        "profile_home": str(home),
        "cwd": str(tmp_path),
        "attached_images": [],
        "transport": None,
    }
    monkeypatch.setitem(server._sessions, "image-paths", session)
    return session


def attach(path):
    return server.handle_request({
        "id": "attach",
        "method": "image.attach",
        "params": {"session_id": "image-paths", "path": path},
    })


def test_missing_image_reports_the_complete_input(attachment_session, tmp_path):
    missing = tmp_path / "Application Support" / "Hermes" / "Screenshot with spaces.png"

    response = attach(str(missing))

    assert response["error"]["code"] == 4016
    assert str(missing) in response["error"]["message"]
    assert attachment_session["attached_images"] == []


@pytest.mark.parametrize("remainder", ["", "describe this screenshot"])
def test_spaced_image_path_preserves_trailing_text(attachment_session, tmp_path, remainder):
    image_path = tmp_path / "Application Support" / "Screenshot with spaces.png"
    image_path.parent.mkdir()
    image_path.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aRZkAAAAASUVORK5CYII="
    ))

    response = attach(f"{image_path} {remainder}".rstrip())

    assert "error" not in response, response
    result = response["result"]
    assert result["attached"] is True
    assert result["path"] == str(image_path.resolve())
    assert result["remainder"] == remainder
    assert attachment_session["attached_images"] == [result["path"]]

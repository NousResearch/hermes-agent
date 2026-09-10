"""Drop classification may opt out of queuing an image for the next turn."""

import pytest
from PIL import Image

from tui_gateway import server


@pytest.fixture
def session(tmp_path, monkeypatch):
    record = {
        "session_key": "detect-drop-test",
        "cwd": str(tmp_path),
        "attached_images": [],
    }
    monkeypatch.setitem(server._sessions, record["session_key"], record)
    return record


def detect_drop(session, text, **options):
    response = server.dispatch({
        "jsonrpc": "2.0", "id": 1, "method": "input.detect_drop",
        "params": {"session_id": session["session_key"], "text": text, **options},
    })
    assert isinstance(response, dict)
    assert "error" not in response, response
    return response["result"]


@pytest.mark.parametrize("caption", ["", "describe this image"])
@pytest.mark.parametrize("already_attached", [False, True])
def test_detection_only_preserves_image_result_without_queuing(
    session, tmp_path, caption, already_attached,
):
    image = Image.new("RGB", (37, 19), "orange")
    path = tmp_path / "image with spaces.png"
    image.save(path)
    if already_attached:
        queued = tmp_path / "previous.png"
        image.save(queued)
        session["attached_images"].append(str(queued))
    before = list(session["attached_images"])
    text = f"{path} {caption}".rstrip()

    detected = detect_drop(session, text, attach_image=False)

    assert session["attached_images"] == before
    assert detected["matched"] is True
    assert detected["is_image"] is True
    assert detected["path"] == str(path)
    assert detected["name"] == path.name
    assert (detected["width"], detected["height"]) == image.size
    assert detected["token_estimate"] > 0
    assert detected["count"] == len(before)
    assert detected["text"] == (caption or f"[User attached image: {path.name}]")
    assert detected["remainder"] == caption

    # Omitted and explicit true retain the legacy attach behavior; only count differs.
    expected_images = list(before)
    for options in ({}, {"attach_image": True}):
        attached = detect_drop(session, text, **options)
        expected_images.append(str(path))
        assert session["attached_images"] == expected_images
        assert attached == {**detected, "count": len(expected_images)}


@pytest.mark.parametrize("kind", ["file", "missing", "text", "empty"])
def test_attach_image_option_does_not_change_non_image_detection(session, tmp_path, kind):
    path = tmp_path / "notes with spaces.txt"
    path.write_text("actual file contents", encoding="utf-8")
    caption = "summarize these notes"
    text = {
        "file": f"{path} {caption}",
        "missing": str(tmp_path / "missing.png"),
        "text": "ordinary chat text",
        "empty": "",
    }[kind]
    expected = {"matched": False}
    if kind == "file":
        expected = {
            "matched": True, "is_image": False, "path": str(path), "name": path.name,
            "text": f"[User attached file: {path}]\n{caption}",
        }

    for options in ({}, {"attach_image": False}, {"attach_image": True}):
        assert detect_drop(session, text, **options) == expected
        assert session["attached_images"] == []

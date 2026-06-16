import json
from pathlib import Path

from tools.whatsapp_profile_tool import whatsapp_profile_picture_tool


def _payload(result: str) -> dict:
    return json.loads(result)


def test_whatsapp_profile_picture_requires_confirmation(tmp_path):
    image = tmp_path / "avatar.jpg"
    image.write_bytes(b"fake image")

    result = _payload(whatsapp_profile_picture_tool({
        "action": "set",
        "image_path": str(image),
        "confirmed": False,
    }))

    assert "error" in result
    assert "confirmation" in result["error"].lower()


def test_whatsapp_profile_picture_set_posts_file_path(monkeypatch, tmp_path):
    image = tmp_path / "avatar.jpg"
    image.write_bytes(b"fake image")
    posted = {}

    def fake_post(payload):
        posted.update(payload)
        return {"status": 200, "data": {"success": True, "action": "updated"}}

    monkeypatch.setattr("tools.whatsapp_profile_tool._post_to_bridge", fake_post)

    result = _payload(whatsapp_profile_picture_tool({
        "action": "set",
        "image_path": str(image),
        "width": 1024,
        "height": 1024,
        "confirmed": True,
    }))

    assert result == {"success": True, "action": "updated"}
    assert posted == {"filePath": str(image), "width": 640, "height": 640}


def test_whatsapp_profile_picture_remove_posts_remove(monkeypatch):
    posted = {}

    def fake_post(payload):
        posted.update(payload)
        return {"status": 200, "data": {"success": True, "action": "removed"}}

    monkeypatch.setattr("tools.whatsapp_profile_tool._post_to_bridge", fake_post)

    result = _payload(whatsapp_profile_picture_tool({
        "action": "remove",
        "confirmed": True,
    }))

    assert result == {"success": True, "action": "removed"}
    assert posted == {"remove": True}


def test_whatsapp_bridge_exposes_profile_picture_endpoint():
    bridge = Path("scripts/whatsapp-bridge/bridge.js").read_text()

    assert "app.post('/profile-picture'" in bridge
    assert "sock.updateProfilePicture" in bridge
    assert "sock.removeProfilePicture" in bridge
    assert "sock.user?.id" in bridge

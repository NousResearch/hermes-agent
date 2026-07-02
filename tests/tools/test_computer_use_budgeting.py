import base64
import json

from tools.computer_use.backend import CaptureResult, UIElement
from tools.computer_use import tool as cu_tool


_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAADUlEQVR4nG"
    "NgGAUgAAABCAABgukLHQAAAABJRU5ErkJggg=="
)


def test_compact_capture_omits_inline_image_and_writes_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr("hermes_constants.get_hermes_dir", lambda: tmp_path)
    cap = CaptureResult(
        mode="som",
        width=800,
        height=600,
        png_b64=_PNG_B64,
        elements=[UIElement(index=0, role="AXButton", label="Send", bounds=(1, 2, 3, 4))],
        app="Telegram",
        window_title="Telegram",
        png_bytes_len=len(base64.b64decode(_PNG_B64)),
    )

    resp = cu_tool._capture_response(cap, inspect_full=False, max_inline_chars=8000)
    payload = json.loads(resp)

    assert "data:image" not in resp
    assert payload["image_inline_omitted"] is True
    assert payload["screenshot_path"].startswith(str(tmp_path))
    assert (tmp_path / "cache" / "computer-use").exists()


def test_action_text_response_is_capped():
    result = cu_tool.ActionResult(
        ok=True,
        action="type",
        meta={"visible_text": "x" * 5000},
    )

    resp = cu_tool._text_response(result, max_inline_chars=1000)

    assert len(resp) <= 1100
    assert "inline_truncated" in resp


from __future__ import annotations

import base64
from pathlib import Path

from tools.browser_cdp_tool import _materialize_cdp_image_result


_TINY_PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def test_capture_screenshot_materializes_base64_to_file(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "tools.browser_cdp_tool.get_hermes_dir",
        lambda *parts: tmp_path.joinpath(*parts),
    )
    result = {
        "data": base64.b64encode(_TINY_PNG).decode("ascii"),
        "format": "png",
    }

    materialized = _materialize_cdp_image_result("Page.captureScreenshot", result)

    assert materialized["data_redacted"] is True
    assert materialized["screenshot_bytes"] == len(_TINY_PNG)
    assert "data" not in materialized
    screenshot_path = Path(materialized["screenshot_path"])
    assert screenshot_path.exists()
    assert screenshot_path.read_bytes() == _TINY_PNG


def test_non_screenshot_method_is_unchanged():
    result = {"data": "abc", "format": "png"}
    assert _materialize_cdp_image_result("Runtime.evaluate", result) == result


def test_invalid_base64_keeps_original_result(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "tools.browser_cdp_tool.get_hermes_dir",
        lambda *parts: tmp_path.joinpath(*parts),
    )
    result = {"data": "not-valid-base64", "format": "png"}

    materialized = _materialize_cdp_image_result("Page.captureScreenshot", result)

    assert materialized == result

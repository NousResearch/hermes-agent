"""Per-task image forwarding on delegate_task.

Port of RooCodeInc/Roomote#1796 / #1767 (Fast → coding-task attachment
forwarding): a task may carry an ``images`` list (local paths or http(s)
URLs). Vision-capable children receive the pixels as native ``image_url``
content parts on their goal turn; non-vision children get
``[Image attached at: …]`` hint lines they can feed to ``vision_analyze``.
Forwarding is best-effort — any failure degrades to the text-only goal.
"""

import base64
from unittest.mock import patch

import pytest

from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _MAX_TASK_IMAGES,
    _build_child_goal_message,
    _normalize_task_images,
)


class _FakeChild:
    provider = "openrouter"
    model = "some/vision-model"


# ---------------------------------------------------------------------------
# _normalize_task_images
# ---------------------------------------------------------------------------


class TestNormalizeTaskImages:
    def test_absent_is_none(self):
        cleaned, err = _normalize_task_images({"goal": "g"}, 0)
        assert cleaned is None
        assert err is None

    def test_list_passes_through_stripped(self):
        cleaned, err = _normalize_task_images(
            {"images": [" /tmp/a.png ", "https://x.test/b.jpg"]}, 0
        )
        assert err is None
        assert cleaned == ["/tmp/a.png", "https://x.test/b.jpg"]

    def test_bare_string_wrapped(self):
        cleaned, err = _normalize_task_images({"images": "/tmp/a.png"}, 0)
        assert err is None
        assert cleaned == ["/tmp/a.png"]

    def test_non_list_rejected(self):
        cleaned, err = _normalize_task_images({"images": {"path": "x"}}, 2)
        assert cleaned is None
        assert err and "Task 2" in err

    def test_non_string_entry_rejected(self):
        cleaned, err = _normalize_task_images({"images": ["/tmp/a.png", 42]}, 0)
        assert cleaned is None
        assert err

    def test_empty_string_entry_rejected(self):
        cleaned, err = _normalize_task_images({"images": ["  "]}, 0)
        assert cleaned is None
        assert err

    def test_over_limit_rejected(self):
        many = [f"/tmp/img{i}.png" for i in range(_MAX_TASK_IMAGES + 1)]
        cleaned, err = _normalize_task_images({"images": many}, 1)
        assert cleaned is None
        assert err and str(_MAX_TASK_IMAGES) in err

    def test_empty_list_normalizes_to_none(self):
        cleaned, err = _normalize_task_images({"images": []}, 0)
        assert cleaned is None
        assert err is None


# ---------------------------------------------------------------------------
# _build_child_goal_message
# ---------------------------------------------------------------------------


def _make_png(tmp_path, name="shot.png"):
    # 1x1 transparent PNG
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBg"
        "AAAABQABh6FO1AAAAABJRU5ErkJggg=="
    )
    p = tmp_path / name
    p.write_bytes(png)
    return str(p)


class TestBuildChildGoalMessage:
    def test_native_mode_builds_content_parts(self, tmp_path):
        path = _make_png(tmp_path)
        with patch(
            "agent.image_routing.decide_image_input_mode", return_value="native"
        ):
            msg = _build_child_goal_message("Inspect the mock", [path], _FakeChild())
        assert isinstance(msg, list)
        types = [p.get("type") for p in msg]
        assert types[0] == "text"
        assert "image_url" in types
        assert "Inspect the mock" in msg[0]["text"]
        # local file embedded as data URL
        img = next(p for p in msg if p.get("type") == "image_url")
        assert img["image_url"]["url"].startswith("data:image/")

    def test_native_mode_passes_urls_verbatim(self):
        url = "https://example.test/mock.png"
        with patch(
            "agent.image_routing.decide_image_input_mode", return_value="native"
        ):
            msg = _build_child_goal_message("Look", [url], _FakeChild())
        assert isinstance(msg, list)
        img = next(p for p in msg if p.get("type") == "image_url")
        assert img["image_url"]["url"] == url

    def test_native_mode_all_unreadable_falls_back_to_text(self, tmp_path):
        missing = str(tmp_path / "nope.png")
        with patch(
            "agent.image_routing.decide_image_input_mode", return_value="native"
        ):
            msg = _build_child_goal_message("Goal text", [missing], _FakeChild())
        assert msg == "Goal text"

    def test_text_mode_appends_hints(self, tmp_path):
        path = _make_png(tmp_path)
        url = "https://example.test/a.png"
        with patch(
            "agent.image_routing.decide_image_input_mode", return_value="text"
        ):
            msg = _build_child_goal_message("Goal", [path, url], _FakeChild())
        assert isinstance(msg, str)
        assert f"[Image attached at: {path}]" in msg
        assert f"[Image attached: {url}]" in msg
        assert "vision_analyze" in msg

    def test_text_mode_missing_path_dropped(self, tmp_path):
        missing = str(tmp_path / "gone.png")
        with patch(
            "agent.image_routing.decide_image_input_mode", return_value="text"
        ):
            msg = _build_child_goal_message("Goal", [missing], _FakeChild())
        assert msg == "Goal"

    def test_any_exception_degrades_to_plain_goal(self):
        with patch(
            "agent.image_routing.decide_image_input_mode",
            side_effect=RuntimeError("boom"),
        ):
            msg = _build_child_goal_message("Plain goal", ["/tmp/x.png"], _FakeChild())
        assert msg == "Plain goal"


# ---------------------------------------------------------------------------
# Tool-schema surface
# ---------------------------------------------------------------------------


class TestToolSchemaSurface:
    def test_images_on_task_items(self):
        item = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]
        assert "images" in item["properties"]
        assert item["properties"]["images"]["type"] == "array"
        assert "images" not in item["required"]

    def test_images_not_a_top_level_schema_param(self):
        # Handler accepts top-level `images` for the legacy single-goal shape,
        # but only the per-task field is advertised.
        assert "images" not in DELEGATE_TASK_SCHEMA["parameters"]["properties"]

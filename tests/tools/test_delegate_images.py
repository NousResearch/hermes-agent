"""Per-task image forwarding on delegate_task: a task's ``images`` (local paths, http(s) or data: URLs) reach a
vision-capable child as native ``image_url`` content parts on its goal turn; non-vision children get
``[Image attached …]`` hints for ``vision_analyze``; any failure degrades to the text-only goal."""

import base64
from unittest.mock import patch

from tools.delegate_tool import DELEGATE_TASK_SCHEMA, _MAX_TASK_IMAGES, _build_child_goal_message, _normalize_task_images
from tools.delegate_tool_child_run import _ChildRun

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg=="
)


class _FakeChild:
    provider = "openrouter"
    model = "some/vision-model"


def _png(tmp_path, name="shot.png"):
    p = tmp_path / name
    p.write_bytes(_PNG)
    return str(p)


def test_normalize_task_images_shapes_and_limit():
    assert _normalize_task_images({"goal": "g"}, 0) == (None, None)
    assert _normalize_task_images({"images": []}, 0) == (None, None)
    assert _normalize_task_images({"images": "/tmp/a.png"}, 0) == (["/tmp/a.png"], None)
    assert _normalize_task_images({"images": [" /tmp/a.png ", "https://x.test/b.jpg"]}, 0) == (
        ["/tmp/a.png", "https://x.test/b.jpg"], None,
    )
    for bad in ({"path": "x"}, ["/tmp/a.png", 42], ["  "]):
        cleaned, err = _normalize_task_images({"images": bad}, 2)
        assert cleaned is None and "Task 2" in err
    cleaned, err = _normalize_task_images({"images": [f"/tmp/{i}.png" for i in range(_MAX_TASK_IMAGES + 1)]}, 1)
    assert cleaned is None and str(_MAX_TASK_IMAGES) in err


def test_native_mode_builds_image_parts_for_every_source_kind(tmp_path):
    path = _png(tmp_path)
    url = "https://example.test/mock.png"
    data_url = "data:image/png;base64," + base64.b64encode(_PNG).decode()
    with patch("agent.image_routing.decide_image_input_mode", return_value="native"):
        msg = _build_child_goal_message("Inspect the mock", [path, url, data_url], _FakeChild())
    assert msg[0]["type"] == "text" and "Inspect the mock" in msg[0]["text"]
    images = [p["image_url"]["url"] for p in msg if p.get("type") == "image_url"]
    assert len(images) == 3 and url in images
    assert images.count(data_url) == 2  # local file embedded as a data URL + the inline data URL passed verbatim
    assert data_url not in msg[0]["text"]  # inline base64 never leaks into the text part
    # every source unreadable → plain goal, no empty multimodal envelope
    with patch("agent.image_routing.decide_image_input_mode", return_value="native"):
        assert _build_child_goal_message("Goal text", [str(tmp_path / "nope.png")], _FakeChild()) == "Goal text"


def test_text_mode_hints_and_failure_degrade(tmp_path):
    path = _png(tmp_path)
    url = "https://example.test/a.png"
    with patch("agent.image_routing.decide_image_input_mode", return_value="text"):
        msg = _build_child_goal_message("Goal", [path, url, str(tmp_path / "gone.png")], _FakeChild())
    assert isinstance(msg, str)
    assert f"[Image attached at: {path}]" in msg and f"[Image attached: {url}]" in msg and "vision_analyze" in msg
    assert "gone.png" not in msg
    with patch("agent.image_routing.decide_image_input_mode", side_effect=RuntimeError("boom")):
        assert _build_child_goal_message("Plain goal", ["/tmp/x.png"], _FakeChild()) == "Plain goal"


def test_child_run_sends_multimodal_goal_turn(tmp_path):
    """The attached images reach ``run_conversation`` as the FIRST user turn's content list."""
    seen = {}

    class _Child(_FakeChild):
        _delegate_images = [_png(tmp_path)]

        def run_conversation(self, user_message, **kw):
            seen["user_message"] = user_message
            return {"final_response": "ok", "completed": True}

    class _Parent:
        _current_task_id = None

    run = _ChildRun(_Child(), _Parent(), 0, "Describe it", None, None)
    with patch("tools.delegate_tool_child_run._create_isolated_worktree", return_value=None), \
            patch("agent.image_routing.decide_image_input_mode", return_value="native"):
        run.seed_workspace()
        result, failure, _ = run.await_child()
    assert failure is None and result["final_response"] == "ok"
    content = seen["user_message"]
    assert isinstance(content, list) and content[0]["type"] == "text"
    assert [p["type"] for p in content].count("image_url") == 1
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_images_advertised_per_task_only():
    item = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]
    assert item["properties"]["images"]["type"] == "array" and "images" not in item["required"]
    assert "images" not in DELEGATE_TASK_SCHEMA["parameters"]["properties"]

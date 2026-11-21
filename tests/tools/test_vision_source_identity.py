"""Vision reuse checks must identify the same source that the image loader reads."""

from __future__ import annotations

import base64
import copy
import json
from io import BytesIO

import pytest

from agent.image_routing import build_native_content_parts
from hermes_cli.config import get_config_path
from tests.e2e.core.history._helpers import NO_BACKGROUND_REVIEW, OFFLINE_CONFIG
from tests.fakes.fake_llm_provider import FakeLLMServer, Response, Text, ToolCall
from tools import vision_tools_history_budget as budget

Image = pytest.importorskip("PIL.Image")


@pytest.fixture(autouse=True)
def isolated_vision_policy():
    config = get_config_path()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        "model:\n  provider: openai\n  name: gpt-4o\n  supports_vision: true\n  streaming: false\n"
        "agent:\n  image_input_mode: native\nvision:\n  max_calls_per_image: 1\n"
        + OFFLINE_CONFIG + NO_BACKGROUND_REVIEW,
        encoding="utf-8",
    )
    with budget._repeat_lock:
        budget._repeat_counts.clear()
    yield
    with budget._repeat_lock:
        budget._repeat_counts.clear()


def _image_pair(root, layout):
    names = {
        "filename": ("capture#before.png", "capture#after.png"),
        "directory": ("run#before/capture.png", "run#after/capture.png"),
        "plain": ("before.png", "after.png"),
    }[layout]
    paths = [root / name for name in names]
    for path, color in zip(paths, ("red", "blue")):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), color).save(path)
    return paths


@pytest.mark.parametrize("layout", ["filename", "directory", "plain"])
@pytest.mark.parametrize("file_prefix", ["", "file://", "FILE://"])
def test_distinct_images_reach_the_next_model_request(tmp_path, monkeypatch, layout, file_prefix):
    from run_agent import AIAgent

    monkeypatch.chdir(tmp_path)
    before, after = _image_pair(tmp_path, layout)
    parts, skipped = build_native_content_parts("Compare before and after.", [str(before)])
    assert not skipped
    original = copy.deepcopy(parts)
    script: list[Response] = [
        ToolCall("vision_analyze", {"image_url": file_prefix + str(before), "question": "Read before."}),
        ToolCall("vision_analyze", {"image_url": file_prefix + str(after), "question": "Read after."}),
        Text("Comparison complete."),
    ]
    with FakeLLMServer(script) as server:
        agent = AIAgent(
            model="gpt-4o", provider="openai", api_key="test-key", base_url=server.base_url,
            enabled_toolsets=["vision"], platform="cli", max_iterations=4,
            quiet_mode=True, skip_memory=True, skip_context_files=True,
        )
        try:
            result = agent.run_conversation(parts)
            requests = server.main_requests()
        finally:
            agent.close()

    assert result["completed"] and result["final_response"] == "Comparison complete."
    assert len(requests) == 3
    tool_messages = [m for m in requests[-1]["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 2
    assert "already_in_context" in tool_messages[0]["content"]
    content = tool_messages[1]["content"]
    assert isinstance(content, list), content
    image_urls = [p["image_url"]["url"] for p in content if p.get("type") == "image_url"]
    assert len(image_urls) == 1
    with Image.open(BytesIO(base64.b64decode(image_urls[0].split(",", 1)[1]))) as image:
        assert image.size == (16, 16)
        assert image.convert("RGB").tobytes() == Image.new("RGB", (16, 16), "blue").tobytes()
    assert parts == original
    assert before.read_bytes() != after.read_bytes()


@pytest.mark.parametrize("case, same_source", [
    ("filename", False), ("directory", False), ("plain", False),
    ("file_prefix", True), ("uppercase_file_prefix", True), ("relative", True), ("home_relative", True),
    ("url_fragment", True), ("url_query", False), ("url_encoded_hash", False),
    ("same_data", True), ("different_data", False),
])
def test_repeat_budget_respects_source_identity(tmp_path, monkeypatch, case, same_source):
    monkeypatch.chdir(tmp_path)
    before, after = _image_pair(tmp_path, case if case in {"filename", "directory", "plain"} else "filename")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    data_before, data_after = ["data:image/png;base64," + base64.b64encode(p.read_bytes()).decode() for p in (before, after)]
    pairs = {
        "file_prefix": (str(before), "file://" + str(before)),
        "uppercase_file_prefix": (str(before), "FILE://" + str(before)),
        "relative": (str(before), before.name),
        "home_relative": (str(before), "~/" + before.name),
        "url_fragment": ("https://example.test/image.png#one", "https://example.test/image.png#two"),
        "url_query": ("https://example.test/image.png?v=1", "https://example.test/image.png?v=2"),
        "url_encoded_hash": ("https://example.test/image%231.png", "https://example.test/image%232.png"),
        "same_data": (data_before, data_before),
        "different_data": (data_before, data_after),
    }
    before_ref, after_ref = pairs.get(case, (str(before), str(after)))

    assert budget.repeat_refusal(before_ref) is None
    second = budget.repeat_refusal(after_ref)
    if same_source:
        assert isinstance(second, str)
        assert json.loads(second)["success"] is False
    else:
        assert second is None, second
    for ref in (before_ref, after_ref):
        refused = budget.repeat_refusal(ref)
        assert isinstance(refused, str)
        assert json.loads(refused)["success"] is False

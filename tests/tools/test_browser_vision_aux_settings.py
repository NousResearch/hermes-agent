"""browser_vision and Camofox vision read ``auxiliary.vision`` timeout/temperature the same way."""

import json
from unittest.mock import MagicMock, patch

import pytest

from hermes_constants import get_hermes_home
from tools.browser_camofox import camofox_navigate, camofox_vision
from tools.browser_tool_vision import _analyze_screenshot_with_aux_llm

# (auxiliary.vision config, expected (timeout, temperature))
CASES = [
    ({"timeout": 45, "temperature": 1}, (45.0, 1.0)),
    ({}, (120.0, 0.1)),
    ({"timeout": 45, "temperature": "hot"}, (45.0, 0.1)),
    ({"timeout": None, "temperature": 0.5}, (120.0, 0.5)),
]


def _reply():
    reply = MagicMock()
    reply.choices = [MagicMock()]
    reply.choices[0].message.content = "ok"
    return reply


def _via_browser_vision(tmp_path):
    shot = tmp_path / "shot.png"
    shot.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    return _analyze_screenshot_with_aux_llm(shot, "what is shown?")


def _via_camofox(tmp_path):
    navigated = MagicMock()
    navigated.json.return_value = {"tabId": "tab-vision", "url": "https://example.com"}
    shot = MagicMock()
    shot.content = b"\x89PNG\r\n\x1a\nfake"
    with (
        patch("tools.browser_camofox.requests.post", return_value=navigated),
        patch("tools.browser_camofox._get_raw", return_value=shot),
    ):
        camofox_navigate("https://example.com", task_id="vision-settings")
        return camofox_vision("what is shown?", task_id="vision-settings")


@pytest.mark.parametrize("run", [_via_browser_vision, _via_camofox], ids=["browser_vision", "camofox"])
@pytest.mark.parametrize(("cfg", "expected"), CASES)
def test_vision_settings_resolve_identically(run, cfg, expected, tmp_path, monkeypatch):
    monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
    (get_hermes_home() / "config.yaml").write_text(json.dumps({"auxiliary": {"vision": cfg}}), encoding="utf-8")
    with patch("agent.auxiliary_client.call_llm", return_value=_reply()) as llm:
        run(tmp_path)

    assert (llm.call_args.kwargs["timeout"], llm.call_args.kwargs["temperature"]) == expected

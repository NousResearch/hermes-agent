"""Known Maps test data must carry a warning through browser_exec (#153121)."""

import json
import os
import sys

import pytest

from tools import browser_use_cli
from tools.registry import registry


@pytest.fixture
def run_browser_output(tmp_path, monkeypatch):
    """Exercise real dispatch/subprocess/result handling without a logged-in browser."""
    monkeypatch.setattr(browser_use_cli, "_find_cli", lambda: [
        sys.executable, "-c", "import sys; exec(sys.stdin.read())",
    ])
    monkeypatch.setattr(browser_use_cli, "_base_subprocess_env", lambda: os.environ.copy())
    monkeypatch.setattr(browser_use_cli, "_route_exec_current",
                        lambda env, *args: (env, None, False, 0))
    monkeypatch.setattr(browser_use_cli, "_workspace_dir", lambda task_id: str(tmp_path))
    monkeypatch.setattr(browser_use_cli, "_attach_vault_supervisor", lambda *args: None)
    monkeypatch.setattr(browser_use_cli, "_own_harness_daemon", lambda env: None)
    monkeypatch.setattr(browser_use_cli, "_reload_if_republished", lambda *args: None)
    monkeypatch.setattr("tools.browser_use_shot.harness_command", lambda cmd, **kw: cmd)

    def run(stdout, stderr="", exit_code=0, screenshot=False):
        code = f"import sys\nprint({stdout!r})\nprint({stderr!r}, file=sys.stderr)\n"
        if screenshot:
            shot = tmp_path / "shot.png"
            code += f"from pathlib import Path\nPath({str(shot)!r}).write_bytes(b'PNG')\nprint({str(shot)!r})\n"
            monkeypatch.setattr("tools.vision_tools._should_use_native_vision_fast_path", lambda: True)
            monkeypatch.setattr("tools.vision_tools._resize_image_for_vision",
                                lambda *args, **kw: "data:image/png;base64,QUJD")
        code += f"raise SystemExit({exit_code})\n"
        return registry.dispatch("browser_exec", {"code": code}, task_id="maps-reliability")

    return run


@pytest.mark.parametrize("channel,marker,exit_code,screenshot", [
    ("stdout", "webdriver - 0 - austin", 0, False),
    ("stdout", "WEBDRIVER - 12 - AUSTIN", 0, True),
    ("stderr", "webdriver\t- 2 -\naustin", 1, False),
])
def test_maps_test_data_warning_survives_result_delivery(
    run_browser_output, channel, marker, exit_code, screenshot,
):
    # stderr's marker is deliberately beyond the display cap; truncation must not hide the warning.
    stdout = marker if channel == "stdout" else "location lookup failed"
    stderr = "x" * 4100 + "\n" + marker if channel == "stderr" else ""
    result = run_browser_output(stdout, stderr, exit_code, screenshot)
    if screenshot:
        assert result["_multimodal"] is True
        payload = json.loads(result["text_summary"])
        assert payload["warnings"][0]["message"] in result["content"][0]["text"]
    else:
        payload = json.loads(result)
    assert payload["success"] is (exit_code == 0)
    assert payload["exit_code"] == exit_code
    assert stdout in payload["output"]
    warning = payload["warnings"][0]
    assert warning["code"] == "google_maps_webdriver_test_data"
    assert "Do not report" in warning["message"]
    assert "independent" in warning["message"]


@pytest.mark.parametrize("output", [
    "Austin, TX", "30.4587205, -97.7504158", "navigator.webdriver = true",
    "webdriver - 0 - austinville", "webdriver - example - austin", "",
])
def test_ordinary_output_is_not_classified_as_maps_test_data(run_browser_output, output):
    payload = json.loads(run_browser_output(output))
    assert payload["success"] is True
    assert payload["output"] == output + "\n"
    assert "warnings" not in payload

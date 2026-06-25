"""AI-Scientist nc_kan smoke + harness /scientist/run Redis E2E (hermetic)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.vendor.fake_redis import FakeRedis

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_AI = REPO_ROOT / "vendor" / "openclaw-mirror" / "AI-Scientist"
HARNESS_SCRIPTS = REPO_ROOT / "vendor" / "openclaw-mirror" / "extensions" / "hypura-harness" / "scripts"
NC_KAN = VENDOR_AI / "templates" / "nc_kan"


def _require_nc_kan_template() -> Path:
    if not NC_KAN.is_dir():
        pytest.skip("AI-Scientist submodule is not checked out")
    pytest.importorskip("numpy")
    return NC_KAN


def _require_pythonosc() -> None:
    pytest.importorskip("pythonosc")


@pytest.fixture()
def harness_scripts_path():
    path = str(HARNESS_SCRIPTS)
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if path not in sys.path:
        sys.path.insert(0, path)
    return HARNESS_SCRIPTS


def test_nc_kan_template_experiment_smoke() -> None:
    nc_kan = _require_nc_kan_template()
    proc = subprocess.run(
        [sys.executable, "experiment.py", "--out_dir=run_smoke"],
        cwd=str(nc_kan),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads((nc_kan / "run_smoke" / "final_info.json").read_text(encoding="utf-8"))
    assert "nc_kan" in payload
    assert "accuracy" in payload["nc_kan"]["means"]


def test_ai_scientist_research_nc_kan_command_smoke(monkeypatch, tmp_path) -> None:
    from tools import ai_scientist_tool as mod

    monkeypatch.setattr(mod, "AI_SCIENTIST_DIR", tmp_path)
    entry = tmp_path / "launch_scientist.py"
    launcher = tmp_path / "ai_scientist_launcher.py"
    entry.write_text("# stub\n", encoding="utf-8")
    launcher.write_text("# stub\n", encoding="utf-8")
    monkeypatch.setattr(mod, "AI_SCIENTIST_ENTRYPOINT", entry)
    monkeypatch.setattr(mod, "AI_SCIENTIST_LAUNCHER", launcher)
    monkeypatch.setattr(
        mod,
        "resolve_ai_scientist_run_config",
        lambda model=None: {
            "sakana_model": "gpt-4o-mini",
            "overlay": {"OPENAI_API_KEY": "test"},
            "has_credentials": True,
            "provider_id": "openai-codex",
            "routing": "openai_shim",
        },
    )
    monkeypatch.setattr(mod, "ensure_ai_scientist_deps", lambda **kwargs: None)

    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    result = json.loads(
        mod.ai_scientist_research(
            experiment="nc_kan",
            num_ideas=1,
            model="gpt-4o-mini",
            task_id="smoke_nc_kan",
            skip_novelty_check=True,
        )
    )
    assert result["success"] is True
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "nc_kan" in cmd
    assert "--skip-novelty-check" in cmd
    assert str(tmp_path) in str(captured["cwd"])


def test_runner_stores_findings_in_fake_redis(harness_scripts_path) -> None:
    import ai_scientist_runner as runner_mod

    fake = FakeRedis()
    fake.rpush(
        "atlas:failures",
        json.dumps({"error": "tool timeout", "stop_reason": "max_retries"}),
    )

    sample_ideas = [
        {
            "Name": "orthonormal_last_layer",
            "Title": "Orthonormal constraints",
            "Experiment": "Project weights each step.",
            "fitness_hint": "stabilize NC ratio",
        }
    ]

    with patch.object(runner_mod, "_get_redis", return_value=fake), patch.object(
        runner_mod.AiScientistRunner,
        "run_ideas",
        return_value=sample_ideas,
    ):
        result = runner_mod.AiScientistRunner().run_from_failures()

    assert result["success"] is True
    assert fake.llen("ai_scientist:findings") == 1
    assert fake.llen("shinka:fitness_hints") == 1
    finding = json.loads(fake.lrange("ai_scientist:findings", 0, -1)[0])
    assert finding["idea"]["Name"] == "orthonormal_last_layer"


def test_harness_scientist_run_nc_kan_e2e(harness_scripts_path) -> None:
    _require_pythonosc()
    from fastapi.testclient import TestClient

    import harness_daemon as hd

    fake = FakeRedis()
    mock_runner = MagicMock()
    mock_runner.run_ideas.return_value = [
        {
            "Name": "kan_depth_sweep",
            "Title": "Depth sweep",
            "Experiment": "Sweep hidden depth.",
            "fitness_hint": "lower nc_ratio",
        }
    ]

    with patch.object(hd, "_get_scientist", return_value=mock_runner), patch.object(
        hd.redis_loop, "_get_redis", return_value=fake
    ), patch.object(hd.redis_loop, "_redis", None):
        client = TestClient(hd.app)
        resp = client.post(
            "/scientist/run",
            json={
                "topic": "improve nc_kan representation",
                "template": "nc_kan",
                "num_ideas": 1,
                "run_experiment": False,
                "model": "ollama/qwen-hakua-core:latest",
            },
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert body["ideas_generated"] == 1
    assert body["findings_stored"] == 1
    assert fake.llen("ai_scientist:findings") == 1
    mock_runner.run_ideas.assert_called_once()


def test_harness_scientist_status_reads_fake_redis(harness_scripts_path) -> None:
    _require_pythonosc()
    from fastapi.testclient import TestClient

    import harness_daemon as hd

    fake = FakeRedis()
    fake.rpush("ai_scientist:findings", json.dumps({"topic": "t"}))
    fake.rpush("ai_scientist:tasks", json.dumps({"task": "x"}))

    with patch.object(hd.redis_loop, "_get_redis", return_value=fake), patch.object(
        hd.redis_loop, "_redis", None
    ):
        client = TestClient(hd.app)
        resp = client.get("/scientist/status")

    assert resp.status_code == 200
    data = resp.json()
    assert data["findings"] == 1
    assert data["tasks"] == 1

"""Tests for the hermes-supervised-run helper script."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "hermes-supervised-run.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("hermes_supervised_run", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run_helper(workdir: Path, *command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--name",
            "pytest-supervised-run",
            "--workdir",
            str(workdir),
            "--log",
            "run.log",
            "--status",
            "status.json",
            "--manifest",
            "manifest.json",
            "--",
            *command,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_supervised_run_success_writes_status_manifest_and_log(tmp_path):
    result = _run_helper(
        tmp_path,
        sys.executable,
        "-c",
        "from pathlib import Path; print('STAGE: start'); Path('artifact.txt').write_text('ok')",
    )

    assert result.returncode == 0
    log = (tmp_path / "run.log").read_text()
    status = json.loads((tmp_path / "status.json").read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())

    assert "STAGE: start" in log
    assert status["status"] == "completed"
    assert status["exit_code"] == 0
    assert status["current_stage"] == "start"
    assert manifest["run_name"] == "pytest-supervised-run"
    assert manifest["exit_code"] == 0
    assert "artifact.txt" in manifest["files_created_or_updated"]
    assert "artifact.txt" in manifest["sha256"]


def test_supervised_run_failure_exits_with_underlying_code(tmp_path):
    result = _run_helper(tmp_path, sys.executable, "-c", "print('about to fail'); raise SystemExit(7)")

    assert result.returncode == 7
    status = json.loads((tmp_path / "status.json").read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())

    assert status["status"] == "failed"
    assert status["exit_code"] == 7
    assert manifest["exit_code"] == 7


def test_stage_blocker_and_redaction_helpers(tmp_path):
    module = _load_module()

    state = module.RunState(name="unit", started_at="2026-01-01T00:00:00Z")
    redacted = module.redact_text("SHOPIFY_API_KEY=shpat_123456 token=abc123456789 Authorization: Bearer secret-token")
    module.update_markers("STAGE: crawl\nBLOCKED: login required", state)

    assert "shpat_123456" not in redacted
    assert "abc123456789" not in redacted
    assert "secret-token" not in redacted
    assert state.current_stage == "crawl"
    assert "login required" in state.blockers

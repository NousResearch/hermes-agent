"""Portable cross-implementation proof through a real MCP stdio child."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


def _run_independent_stdio_fixture(tmp_path, legacy):
    pytest.importorskip("mcp")
    root = Path(__file__).resolve().parents[2]
    runner = root / "tests" / "fixtures" / "mcp_skills_interop" / "run_interop.py"
    output = tmp_path / "interop.json"
    result = subprocess.run(
        [sys.executable, "-B", str(runner), "--hermes", str(root), "--output", str(output),
         *(["--legacy"] if legacy else [])],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"
    checks = receipt["checks"]
    assert checks["cache_hint_envelopes"] == ("legacy" if legacy else "modern")
    assert checks["shutdown_returned"] is True
    assert checks["alternate_uri_scheme"] is True
    assert checks["real_stdio_discovery"]["catalog_entries"] == 7
    assert checks["native_skill_view"]["binary_materialized"] is True
    assert checks["collisions"]["ambiguous_bare_name_rejected"] is True
    assert checks["uri_only"] == {
        "approved_load": True, "denial_body_reads": 0, "registered_after_get": True}
    assert checks["nested_overlap"]["separate_activation_prompts"] == 2
    assert checks["nested_overlap"]["support_read_inert"] is True
    assert checks["nested_overlap"]["execution_separately_denied"] is True


@pytest.mark.linux_only
@pytest.mark.parametrize("legacy", [False, True], ids=["modern", "legacy"])
def test_independent_stdio_skills_fixture_linux(tmp_path, legacy):
    _run_independent_stdio_fixture(tmp_path, legacy)


@pytest.mark.macos_only
@pytest.mark.parametrize("legacy", [False, True], ids=["modern", "legacy"])
def test_independent_stdio_skills_fixture_macos(tmp_path, legacy):
    _run_independent_stdio_fixture(tmp_path, legacy)

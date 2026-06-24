import json
import os
from pathlib import Path
from types import SimpleNamespace

from tools import gbrain_tool


def _make_fake_gbrain(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import sys

if len(sys.argv) < 2:
    raise SystemExit(2)

if sys.argv[1] == "search":
    print("[0.9999] systems/hermes-runtime -- Hermes Runtime Notes")
elif sys.argv[1] == "get":
    print("# Hermes Runtime Notes\\nGBrain output is context.")
else:
    raise SystemExit(2)
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_gbrain_search_uses_user_bun_bin_when_path_is_minimal(tmp_path, monkeypatch):
    home = tmp_path / "home"
    bin_dir = home / ".bun" / "bin"
    bin_dir.mkdir(parents=True)
    _make_fake_gbrain(bin_dir / "gbrain")

    monkeypatch.delenv("HERMES_GBRAIN_BIN", raising=False)
    monkeypatch.delenv("GBRAIN_BIN", raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", os.pathsep.join(["/usr/bin", "/bin"]))

    result = json.loads(gbrain_tool.gbrain_search("Hermes Runtime", limit=3))

    assert result["ok"] is True
    assert "systems/hermes-runtime" in result["output"]
    assert "source-of-truth" in result["note"]


def test_gbrain_get_supports_env_binary(tmp_path, monkeypatch):
    fake = tmp_path / "gbrain"
    _make_fake_gbrain(fake)
    monkeypatch.setenv("GBRAIN_BIN", str(fake))

    result = json.loads(gbrain_tool.gbrain_get("systems/hermes-runtime"))

    assert result["ok"] is True
    assert "# Hermes Runtime Notes" in result["output"]


def test_gbrain_search_reports_pglite_lock(monkeypatch):
    monkeypatch.setattr(gbrain_tool, "_resolve_gbrain_bin", lambda: "/tmp/gbrain")

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="GBrain: Timed out waiting for PGLite lock.",
        )

    monkeypatch.setattr(gbrain_tool.subprocess, "run", fake_run)

    result = json.loads(gbrain_tool.gbrain_search("anything"))

    assert result["ok"] is False
    assert result["error"] == "gbrain_failed"
    assert "PGLite lock" in result["fix"]


def test_gbrain_tools_are_in_core_hermes_toolsets():
    from toolsets import resolve_toolset

    assert "gbrain_search" in resolve_toolset("gbrain")
    assert "gbrain_get" in resolve_toolset("gbrain")
    assert "gbrain_search" in resolve_toolset("hermes-cli")
    assert "gbrain_get" in resolve_toolset("hermes-slack")

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _retained_logs(log_dir: Path):
    live = log_dir / "mcp-stderr.log"
    backups = [p for p in log_dir.glob("mcp-stderr.log.*") if p.suffix[1:].isdigit()]
    return ([live] if live.exists() else []) + sorted(backups)


def test_mcp_stderr_log_redacts_secrets_and_bounds_retention(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    monkeypatch.setenv("HERMES_HOME", str(home))

    from tools import mcp_tool_config as config

    config._close_mcp_stderr_logs()
    canary = "sk-proj-" + "canarycredential" * 4
    try:
        for _ in range(config._MCP_STDERR_BACKUP_COUNT + 3):
            tee = config._StderrTee(config._get_mcp_stderr_log())
            tee.sink.write((f"credential={canary} " + "x" * config._MCP_STDERR_MAX_BYTES + "\n").encode())
            tee.close()
    finally:
        config._close_mcp_stderr_logs()

    logs = _retained_logs(home / "logs")
    assert 1 <= len(logs) <= config._MCP_STDERR_BACKUP_COUNT + 1
    retained = b"".join(path.read_bytes() for path in logs)
    assert canary.encode() not in retained
    assert b"credential=" in retained
    assert sum(path.stat().st_size for path in logs) <= (
        config._MCP_STDERR_MAX_BYTES * (config._MCP_STDERR_BACKUP_COUNT + 1)
    )


_WRITER = textwrap.dedent("""
    import os, sys, time
    from pathlib import Path
    from tools import mcp_tool_config as config
    config._MCP_STDERR_MAX_BYTES = 4096
    config._MCP_STDERR_BACKUP_COUNT = 400  # keep every generation: lost lines become visible
    tag, lines, go = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    log = config._get_mcp_stderr_log()
    while not go.exists():
        time.sleep(0.005)
    for i in range(lines):
        log.write(f"{tag}-{i:05d} " + "y" * 150 + "\\n")
    config._close_mcp_stderr_logs()
""")


def test_two_processes_rotating_one_profile_log_lose_nothing_and_stay_bounded(tmp_path):
    """gateway + Desktop backend share one HERMES_HOME: concurrent rotations at the size
    boundary must neither drop lines nor let any generation grow past the bound."""
    home = tmp_path / "profile"
    go = tmp_path / "go"
    lines = 600
    env = {**os.environ, "HERMES_HOME": str(home),
           "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), os.environ.get("PYTHONPATH", "")])}
    procs = [subprocess.Popen([sys.executable, "-c", _WRITER, tag, str(lines), str(go)],
                              cwd=REPO_ROOT, env=env)
             for tag in ("a", "b")]
    time.sleep(1.0)  # let both children import and open the shared log
    go.touch()
    for proc in procs:
        assert proc.wait(timeout=120) == 0

    logs = _retained_logs(home / "logs")
    assert len(logs) > 2, "the scenario must actually cross the rotation boundary repeatedly"
    for path in logs:
        assert path.stat().st_size <= 4096, f"{path.name} grew past the bound"
    seen = [line.split(" ", 1)[0] for path in logs
            for line in path.read_text().splitlines() if line]
    expected = {f"{tag}-{i:05d}" for tag in ("a", "b") for i in range(lines)}
    assert sorted(seen) == sorted(expected), "lines were lost or duplicated across rotation"

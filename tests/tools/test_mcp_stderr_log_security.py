import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.waiting import wait_until

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
    (go.parent / f"ready-{tag}").touch()
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
    wait_until(lambda: all((tmp_path / f"ready-{tag}").exists() for tag in ("a", "b")), timeout=60,
               what="both writers opened the shared log")
    go.touch()
    for proc in procs:
        assert proc.wait(timeout=120) == 0

    logs = _retained_logs(home / "logs")
    assert len(logs) > 2, "the scenario must actually cross the rotation boundary repeatedly"
    for path in logs:
        assert path.stat().st_size <= 4096, f"{path.name} grew past the bound"
    seen = [line.split(" ", 1)[0] for path in logs
            for line in path.read_text(encoding="utf-8-sig").splitlines() if line]
    expected = {f"{tag}-{i:05d}" for tag in ("a", "b") for i in range(lines)}
    assert sorted(seen) == sorted(expected), "lines were lost or duplicated across rotation"


_CHILD = r"""
import sys
v = sys.argv[1]
sys.stderr.write('{\n  "api_key":\n    "' + 'jsonSplitOpaque4471qq' + '",\n  "n": 1\n}\n')
sys.stderr.write('{\n"api_key":\n"' + 'jsonUnindented8823zz' + '"\n}\n')
sys.stderr.write('banner ' + v[:6] + '\x1b[0m' + v[6:] + '\n')
sys.stderr.flush()
"""


def test_records_are_redacted_whole_with_the_spawning_profiles_values(tmp_path, monkeypatch):
    """A pretty-printed dump whose key and value sit on separate lines, and a configured value split
    by a colour code, never reach the log; the reader thread redacts with the spawning profile's
    values even though it is a fresh thread (no inherited profile binding)."""
    from agent.redact_known_values import MARKER, clear_known_secret_values
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools import mcp_tool_config as config

    secret = "q7Wz2pLx9Rv4Kd8Nf3Ty"
    launch, served = tmp_path / "launch", tmp_path / "served"
    launch.mkdir()
    served.mkdir()
    (served / ".env").write_text(f"ACME_SERVICE_TOKEN={secret}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    clear_known_secret_values()
    token = set_hermes_home_override(served)
    try:
        config._close_mcp_stderr_logs()
        tee = config._StderrTee(config._get_mcp_stderr_log())
        subprocess.run([sys.executable, "-c", _CHILD, secret], stderr=tee.sink, check=True, timeout=60)
        tee.close(timeout=10)
    finally:
        config._close_mcp_stderr_logs()
        reset_hermes_home_override(token)
        clear_known_secret_values()
    log = (served / "logs" / "mcp-stderr.log").read_text(encoding="utf-8-sig")
    assert "jsonSplitOpaque4471qq" not in log
    assert "jsonUnindented8823zz" not in log
    assert secret not in log
    assert '"n": 1' in log and "banner " + MARKER in log


@pytest.mark.platforms("windows")
def test_log_bytes_on_disk_match_the_size_check(tmp_path):
    """Multiline writes land byte-for-byte (no ``\\r\\n`` growth), so rotation's size check bounds the file."""
    from tools import mcp_tool_config as config

    log = config._McpStderrLog(tmp_path / "mcp-stderr.log")
    try:
        log.write("first line\nsecond line\nthird line\n")
    finally:
        log.close()
    assert (tmp_path / "mcp-stderr.log").read_bytes() == b"first line\nsecond line\nthird line\n"

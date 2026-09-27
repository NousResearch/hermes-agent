"""Regression: the compose liveness probe must not crash on undecodable child output.

``_compose_live_state_reason`` ran ``docker compose ps`` with ``text=True`` and no
``encoding``/``errors``: under a non-UTF-8 locale (Windows ANSI code page), a stray
byte in docker's stderr raised ``UnicodeDecodeError`` — a ``ValueError`` that escapes
the probe's ``except (FileNotFoundError, subprocess.TimeoutExpired)`` and kills the
whole ``hermes verify`` run instead of being reported as a probe failure. Same class
as the cron script runner (#53744) and the CLI sweep (b00b4bb7f2); the verify runner's
own ``_SUBPROCESS_KW`` profile already pins ``errors="replace"`` — this probe bypassed it.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent.verify.runner import _compose_live_state_reason


def _fake_docker(root: Path, *, stdout_bytes: bytes, stderr_bytes: bytes = b"") -> None:
    script = (
        "#!%s\n"
        "import sys\n"
        "sys.stdout.buffer.write(%r)\n"
        "sys.stderr.buffer.write(%r)\n"
        "sys.exit(1)\n"
    ) % (sys.executable, stdout_bytes, stderr_bytes)
    (root / "docker").write_text(script, encoding="utf-8")
    (root / "docker").chmod(0o755)
    if sys.platform == "win32":
        # Windows resolves bare "docker" via PATHEXT, so provide docker.exe too.
        (root / "docker.exe").write_bytes((root / "docker").read_bytes())


def test_compose_probe_tolerates_undecodable_stderr(tmp_path, monkeypatch):
    # With text=True + strict locale decoding this raises UnicodeDecodeError inside
    # subprocess.run; the probe must instead report the failed probe (str), never raise.
    _fake_docker(tmp_path, stdout_bytes=b"ok\n", stderr_bytes=b"\xff\xfe bad name\n")
    monkeypatch.setenv("PATH", str(tmp_path) + ":")
    reason = _compose_live_state_reason(tmp_path)
    assert isinstance(reason, str)
    assert reason  # non-zero exit + stderr present -> probe failure is reported


def test_compose_probe_never_raises_on_undecodable_stdout(tmp_path, monkeypatch):
    # Contract: the probe returns str | None for ANY child output bytes.
    _fake_docker(tmp_path, stdout_bytes=b"\xff\xfe\xfd container_1\n", stderr_bytes=b"")
    monkeypatch.setenv("PATH", str(tmp_path) + ":")
    reason = _compose_live_state_reason(tmp_path)
    assert reason is None or isinstance(reason, str)

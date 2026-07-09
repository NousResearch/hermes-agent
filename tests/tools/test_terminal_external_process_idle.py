from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest

psutil = pytest.importorskip("psutil")

from tools.process_registry import process_registry


def _touch(path: Path, *, mtime: int, content: bytes = b"partial media") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    os.utime(path, (mtime, mtime))


def _python_for_shell() -> str:
    return Path(sys.executable).as_posix()


def _wait_for_media_evidence(session_id: str, *, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        last = process_registry.poll(session_id)
        if last.get("external_media_processes"):
            return last
        time.sleep(0.1)
    return last


def test_terminal_background_media_process_reports_idle_part_file(tmp_path):
    now = int(time.time())
    final_output = tmp_path / "clip.mp4"
    part_output = Path(str(final_output) + ".part")
    _touch(part_output, mtime=now - 300)
    command = (
        f"{shlex.quote(_python_for_shell())} -c "
        f"{shlex.quote('import time; time.sleep(60)')} "
        f"yt-dlp -o {shlex.quote(final_output.as_posix())}"
    )

    session = process_registry.spawn_local(
        command=command,
        cwd=str(tmp_path),
        task_id="task_terminal_external_idle",
        session_key="session_terminal_external_idle",
    )
    try:
        result = _wait_for_media_evidence(session.id)

        media = result["external_media_processes"][0]
        assert media["command_name"] == "yt-dlp"
        assert media["pid"]
        assert media["output_path"] == str(part_output)
        assert media["output_size_bytes"] == part_output.stat().st_size
        assert media["output_mtime_age_seconds"] >= 300
        assert media["elapsed_seconds"] >= 0
        assert media["is_idle"] is True
        assert "cpu_percent" in media
        assert process_registry.poll(session.id)["status"] == "running"
    finally:
        process_registry.kill_process(session.id, source="test.cleanup")


def test_external_media_evidence_helper_does_not_kill_owned_process(tmp_path):
    from tools.process_registry import external_media_process_evidence

    now = int(time.time())
    final_output = tmp_path / "audio.wav"
    part_output = Path(str(final_output) + ".part")
    _touch(part_output, mtime=now - 240)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            "ffmpeg",
            "-y",
            "-i",
            "input.wav",
            str(final_output),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        evidence = external_media_process_evidence(
            proc.pid,
            command="ffmpeg -y -i input.wav audio.wav",
            cwd=str(tmp_path),
            now=now,
            idle_threshold_seconds=60,
        )

        assert evidence[0]["command_name"] == "ffmpeg"
        assert evidence[0]["output_path"] == str(part_output)
        assert evidence[0]["is_idle"] is True
        assert proc.poll() is None
    finally:
        try:
            psutil.Process(proc.pid).terminate()
        except psutil.NoSuchProcess:
            pass
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)

"""Test-only adapter for verify runner tests that use tmp_path.

Production tests of the sandbox helper are separate. This fixture makes the
runner unit tests explicit about using local subprocesses rather than silently
falling back to host execution in production code.
"""

import os
import signal
import subprocess
import urllib.error
import urllib.request

import pytest

import agent.verify.runner as runner


@pytest.fixture(autouse=True)
def _inject_local_test_adapter(monkeypatch):
    processes = {}

    def local_helper(*args, timeout):
        action = args[0]
        if action == "run":
            _, workspace, _phase, timeout_seconds, command = args
            try:
                result = subprocess.run(
                    command,
                    cwd=workspace,
                    timeout=float(timeout_seconds),
                    **runner._SUBPROCESS_KW,
                )
                return subprocess.CompletedProcess(args, result.returncode, result.stdout or "")
            except subprocess.TimeoutExpired as exc:
                raw = exc.stdout
                output = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else (raw or "")
                return subprocess.CompletedProcess(args, 124, output)

        if action == "start":
            _, workspace, _runtime, token, command = args
            proc = subprocess.Popen(
                command,
                cwd=workspace,
                start_new_session=True,
                **runner._SUBPROCESS_KW,
            )
            processes[token] = proc
            return subprocess.CompletedProcess(args, 0, f"STARTED {token}\n")

        if action == "probe":
            _, token, url = args
            proc = processes.get(token)
            if proc is None or proc.poll() is not None:
                return subprocess.CompletedProcess(args, 1, "NOT_READY: process inactive\n")
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    return subprocess.CompletedProcess(args, 0, f"READY {response.status}\n")
            except urllib.error.HTTPError as exc:
                return subprocess.CompletedProcess(args, 0, f"READY {exc.code}\n")
            except Exception as exc:
                return subprocess.CompletedProcess(
                    args, 1, f"NOT_READY {type(exc).__name__}: {exc}\n"
                )

        if action == "stop":
            _, token = args
            proc = processes.pop(token, None)
            if proc is None:
                return subprocess.CompletedProcess(args, 0, "")
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        pass
            try:
                output = proc.stdout.read() if proc.stdout is not None else ""
            except (OSError, ValueError):
                output = ""
            return subprocess.CompletedProcess(args, 0, output or "")

        return subprocess.CompletedProcess(args, 2, f"Unknown test adapter action: {action}\n")

    monkeypatch.setattr(runner, "_run_sandbox_helper", local_helper)

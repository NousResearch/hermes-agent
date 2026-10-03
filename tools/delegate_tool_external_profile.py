"""External Hermes-profile runner owned by the async-delegation lifecycle.

The external process is a Docker leaf: its stdout is the only result channel and
its parent owns durable async-delegation delivery. It receives a disposable
copy of the selected profile state, never the parent gateway process, socket or
profile directory.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

_LEAF_IMAGE = "hermes-agent-external-leaf:local"
_LEAF_HOME = "/home/hermes/.hermes"
_LEAF_WORKSPACE = "/workspace"
_PROFILE_STATE_NAMES = ("config.yaml", ".env", "auth.json", "skills", "SOUL.md", "USER.md", "MEMORY.md")


def validate_external_profile(raw: str) -> tuple[str, Path]:
    """Resolve only a live canonical profile; paths and aliases never reach argv."""
    from hermes_cli.profiles import get_profile_dir, normalize_profile_name, profile_exists, validate_profile_name

    profile = normalize_profile_name(raw)
    validate_profile_name(profile)
    if not profile_exists(profile):
        raise ValueError(f"Profile '{profile}' does not exist.")
    return profile, Path(get_profile_dir(profile))


def _leaf_stage_dir() -> Path:
    from hermes_constants import get_scratch_dir

    return Path(tempfile.mkdtemp(prefix="external-profile-leaf-", dir=str(get_scratch_dir())))


def stage_profile_state(profile: str, profile_home: Path) -> Path:
    """Create the leaf's short-lived writable profile copy using an explicit allowlist."""
    stage = _leaf_stage_dir()
    try:
        target_home = stage / "profiles" / profile
        target_home.mkdir(parents=True)
        for name in _PROFILE_STATE_NAMES:
            source = profile_home / name
            target = target_home / name
            if source.is_dir():
                shutil.copytree(source, target, symlinks=True)
            elif source.is_file():
                shutil.copy2(source, target, follow_symlinks=False)
        os.chmod(stage, 0o700)
        return stage
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def external_profile_is_authorized(profile: str) -> bool:
    """Fail closed unless this origin profile explicitly names the target profile."""
    from hermes_cli.config import load_config

    configured = ((load_config() or {}).get("delegation") or {}).get("external_profile_allowlist") or []
    return profile in {str(item).strip() for item in configured if str(item).strip()}


def _docker_leaf_argv(*, profile: str, stage: Path, cwd: str) -> list[str]:
    """Build a host-isolated Docker invocation; no host sockets or parent state mount."""
    profile_stage = stage / "profiles" / profile
    argv = [
        "docker", "run", "--rm", "--init", "--network", "bridge", "--read-only",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "128",
        "--memory", "4g", "--cpus", "2",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=512m",
        "--tmpfs", "/run:rw,nosuid,nodev,size=64m",
        "--mount", f"type=bind,src={cwd},dst={_LEAF_WORKSPACE},readonly",
        "--mount", f"type=bind,src={stage},dst={_LEAF_HOME}",
        "--workdir", _LEAF_WORKSPACE,
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--env", "HOME=/tmp",
        "--env", f"HERMES_HOME={_LEAF_HOME}",
        "--env", "HERMES_DELEGATED_CHILD_CONTEXT=1",
        "--entrypoint", "/opt/hermes/.venv/bin/hermes",
        _LEAF_IMAGE,
        "--profile", profile, "--in", _LEAF_WORKSPACE, "-t", "terminal",
    ]
    # OAuth refresh tokens and API-key files must never be mutated in the
    # disposable profile copy: a rotation saved only in the leaf can invalidate
    # the original profile's credential. A file bind over the writable stage is
    # a separate read-only mount that an unprivileged leaf cannot unlink.
    for name in ("auth.json", ".env"):
        source = profile_stage / name
        if source.is_file():
            argv[argv.index(_LEAF_IMAGE):argv.index(_LEAF_IMAGE)] = [
                "--mount", f"type=bind,src={source},dst={_LEAF_HOME}/profiles/{profile}/{name},readonly",
            ]
    return argv


def _docker_client_env() -> dict[str, str]:
    """Keep Docker-client launch state minimal; profile credentials live only in the leaf stage."""
    kept = ("PATH", "LANG", "LC_ALL", "TZ", "SSL_CERT_FILE", "SSL_CERT_DIR")
    return {key: os.environ[key] for key in kept if os.environ.get(key)}


def make_external_profile_runner(*, profile: str, profile_home: Path, goal: str, context: str | None,
                                 cwd: str) -> tuple[Callable[[], dict[str, Any]], Callable[[], None]]:
    """Return a Docker-leaf runner plus an idempotent interrupt hook for async_delegation."""
    lock = threading.Lock()
    proc: subprocess.Popen[str] | None = None
    cancelled = False
    prompt = goal if not context else f"{goal}\n\nContext:\n{context}"

    def interrupt() -> None:
        nonlocal cancelled, proc
        with lock:
            cancelled = True
            current = proc
        if current is None or current.poll() is not None:
            return
        try:
            if os.name != "nt":
                os.killpg(current.pid, signal.SIGTERM)
            else:
                current.terminate()
        except OSError:
            pass

    def run() -> dict[str, Any]:
        nonlocal proc
        from agent.redact import redact_terminal_output

        started = time.monotonic()
        stage: Path | None = None
        try:
            with lock:
                if cancelled:
                    return {"status": "interrupted", "summary": None, "api_calls": 0,
                            "duration_seconds": round(time.monotonic() - started, 2),
                            "exit_reason": "interrupted_before_launch", "external_profile": profile}
                stage = stage_profile_state(profile, profile_home)
                argv = _docker_leaf_argv(profile=profile, stage=stage, cwd=cwd)
                proc = subprocess.Popen(argv + ["-z", prompt], cwd=cwd, env=_docker_client_env(), text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        start_new_session=(os.name != "nt"))
                interrupted_after_launch = cancelled
            if interrupted_after_launch:
                interrupt()
            output, _ = proc.communicate()
            duration = round(time.monotonic() - started, 2)
            safe = redact_terminal_output(output or "", "hermes external profile Docker leaf", force=True)[-12000:]
            with lock:
                was_cancelled = cancelled
            if was_cancelled:
                return {"status": "interrupted", "summary": safe or None, "api_calls": 0,
                        "duration_seconds": duration, "exit_reason": "interrupted", "external_profile": profile}
            if proc.returncode == 0:
                return {"status": "completed", "summary": safe, "api_calls": 0,
                        "duration_seconds": duration, "model": None, "exit_reason": "completed",
                        "external_profile": profile}
            return {"status": "error", "summary": None,
                    "error": safe or f"profile Docker leaf exited {proc.returncode}",
                    "api_calls": 0, "duration_seconds": duration, "exit_reason": "process_error",
                    "external_profile": profile}
        except (OSError, shutil.Error) as exc:
            return {"status": "error", "summary": None, "error": str(exc), "api_calls": 0,
                    "duration_seconds": round(time.monotonic() - started, 2), "exit_reason": "launch_error",
                    "external_profile": profile}
        finally:
            if stage is not None:
                shutil.rmtree(stage, ignore_errors=True)

    return run, interrupt

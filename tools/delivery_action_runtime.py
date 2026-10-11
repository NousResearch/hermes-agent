"""Bounded subprocess and disk helpers for structured delivery verification."""
from __future__ import annotations

import os
from pathlib import Path
import selectors
import signal
import shutil
import subprocess
import time
from typing import Optional

import psutil


class DeliveryRuntimeError(RuntimeError):
    """A bounded delivery subprocess or quota failed."""


def _stop_process(proc: subprocess.Popen[bytes]) -> None:
    """Signal the parent first, then terminate and reap every surviving child."""
    parent_live = proc.poll() is None
    try:
        parent = psutil.Process(proc.pid)
        descendants = parent.children(recursive=True)
    except (psutil.Error, OSError):
        descendants = []
    if parent_live:
        try:
            proc.terminate()
        except OSError:
            pass
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
    else:
        proc.wait(timeout=0.1)

    # start_new_session gives this subprocess a private process group on POSIX.
    # The group signal catches a child that inherited a pipe after its parent
    # exited, while the psutil snapshot also catches descendants that escaped
    # into their own group. Both happen only after the parent was signalled.
    if os.name == "posix":
        try:
            getattr(os, "killpg")(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for child in descendants:
        try:
            child.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(descendants, timeout=1)
    for child in alive:
        try:
            child.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=1)
    if os.name == "posix":
        try:
            getattr(os, "killpg")(proc.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        except ProcessLookupError:
            pass


def _bounded_popen(
    argv: list[str], *, cwd: Optional[str], env: Optional[dict[str, str]], timeout: int,
    output_limit: int, stdout_path: Optional[Path] = None, disk_reserve: int = 0,
) -> subprocess.CompletedProcess[str]:
    """Stream both pipes under one aggregate cap; never buffer attacker output unbounded."""
    if output_limit <= 0:
        raise DeliveryRuntimeError("subprocess output limit is invalid")
    try:
        proc = subprocess.Popen(
            argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=(os.name == "posix"),
        )
    except OSError as exc:
        raise DeliveryRuntimeError(f"structured operation failed to start: {exc}") from exc
    assert proc.stdout is not None and proc.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
    selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    total = 0
    started = time.monotonic()
    output_file = None
    try:
        if stdout_path is not None:
            try:
                output_file = stdout_path.open("xb")
            except OSError as exc:
                raise DeliveryRuntimeError("verification output file could not be created") from exc
        while selector.get_map():
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                _stop_process(proc)
                raise DeliveryRuntimeError("structured operation timed out")
            events = selector.select(min(remaining, 0.25))
            for key, _mask in events:
                chunk = os.read(key.fd, 64 * 1024)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                total += len(chunk)
                if total > output_limit:
                    _stop_process(proc)
                    raise DeliveryRuntimeError(
                        f"structured operation exceeded the {output_limit}-byte output limit"
                    )
                if key.data == "stdout" and output_file is not None:
                    assert stdout_path is not None
                    require_free_disk(stdout_path.parent, len(chunk), reserve=disk_reserve)
                    output_file.write(chunk)
                else:
                    captured[key.data].extend(chunk)
        try:
            returncode = proc.wait(timeout=max(0.1, timeout - (time.monotonic() - started)))
        except subprocess.TimeoutExpired as exc:
            _stop_process(proc)
            raise DeliveryRuntimeError("structured operation timed out") from exc
    finally:
        selector.close()
        if output_file is not None:
            output_file.close()
        if proc.poll() is None:
            _stop_process(proc)
        proc.stdout.close()
        proc.stderr.close()
    return subprocess.CompletedProcess(
        argv, returncode,
        captured["stdout"].decode("utf-8", "replace"),
        captured["stderr"].decode("utf-8", "replace"),
    )


def run_bounded(
    argv: list[str], *, cwd: Optional[str] = None, env: Optional[dict[str, str]] = None,
    timeout: int, output_limit: int,
) -> subprocess.CompletedProcess[str]:
    return _bounded_popen(
        argv, cwd=cwd, env=env, timeout=timeout, output_limit=output_limit,
    )


def stream_to_file_bounded(
    argv: list[str], destination: Path, *, cwd: Optional[str] = None,
    env: Optional[dict[str, str]] = None, timeout: int, byte_limit: int,
    disk_reserve: int = 0,
) -> subprocess.CompletedProcess[str]:
    return _bounded_popen(
        argv, cwd=cwd, env=env, timeout=timeout, output_limit=byte_limit,
        stdout_path=destination, disk_reserve=disk_reserve,
    )


def require_free_disk(path: Path, required: int, *, reserve: int) -> None:
    try:
        free = shutil.disk_usage(path).free
    except OSError as exc:
        raise DeliveryRuntimeError("verification free-disk capacity could not be determined") from exc
    if free < required + reserve:
        raise DeliveryRuntimeError(
            "verification has insufficient free disk for the configured materialization quota"
        )

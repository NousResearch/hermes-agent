"""Singularity/Apptainer persistent container environment.

Security-hardened with --containall, --no-home, capability dropping.
Supports configurable resource limits and optional filesystem persistence
via writable overlay directories that survive across sessions.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from tools.environments.base import BaseEnvironment
from tools.interrupt import is_interrupted

logger = logging.getLogger(__name__)

_SNAPSHOT_STORE = Path.home() / ".hermes" / "singularity_snapshots.json"


def _load_snapshots() -> Dict[str, str]:
    if _SNAPSHOT_STORE.exists():
        try:
            return json.loads(_SNAPSHOT_STORE.read_text())
        except Exception:
            pass
    return {}


def _save_snapshots(data: Dict[str, str]) -> None:
    _SNAPSHOT_STORE.parent.mkdir(parents=True, exist_ok=True)
    _SNAPSHOT_STORE.write_text(json.dumps(data, indent=2))


# -------------------------------------------------------------------------
# Singularity helpers (scratch dir, SIF cache, SIF building)
# -------------------------------------------------------------------------


def _get_scratch_dir() -> Path:
    """Get the best directory for Singularity sandboxes.

    Resolution order:
      1. TERMINAL_SCRATCH_DIR (explicit override)
      2. TERMINAL_SANDBOX_DIR / singularity (shared sandbox root)
      3. /scratch (common on HPC clusters)
      4. ~/.hermes/sandboxes/singularity (fallback)
    """
    custom_scratch = os.getenv("TERMINAL_SCRATCH_DIR")
    if custom_scratch:
        scratch_path = Path(custom_scratch)
        scratch_path.mkdir(parents=True, exist_ok=True)
        return scratch_path

    from tools.environments.base import get_sandbox_dir

    sandbox = get_sandbox_dir() / "singularity"

    scratch = Path("/scratch")
    if scratch.exists() and os.access(scratch, os.W_OK):
        user_scratch = scratch / os.getenv("USER", "hermes") / "hermes-agent"
        user_scratch.mkdir(parents=True, exist_ok=True)
        logger.info("Using /scratch for sandboxes: %s", user_scratch)
        return user_scratch

    sandbox.mkdir(parents=True, exist_ok=True)
    return sandbox


def _get_apptainer_cache_dir() -> Path:
    """Get the Apptainer cache directory for SIF images."""
    cache_dir = os.getenv("APPTAINER_CACHEDIR")
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path
    scratch = _get_scratch_dir()
    cache_path = scratch / ".apptainer"
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


_sif_build_lock = threading.Lock()


def _get_or_build_sif(image: str, executable: str = "apptainer") -> str:
    """Get or build a SIF image from a docker:// URL.

    Returns the path unchanged if it's already a .sif file.
    For docker:// URLs, checks the cache and builds if needed.
    """
    if image.endswith(".sif") and Path(image).exists():
        return image
    if not image.startswith("docker://"):
        return image

    image_name = image.replace("docker://", "").replace("/", "-").replace(":", "-")
    cache_dir = _get_apptainer_cache_dir()
    sif_path = cache_dir / f"{image_name}.sif"

    if sif_path.exists():
        return str(sif_path)

    with _sif_build_lock:
        if sif_path.exists():
            return str(sif_path)

        logger.info("Building SIF image (one-time setup)...")
        logger.info("  Source: %s", image)
        logger.info("  Target: %s", sif_path)

        tmp_dir = cache_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["APPTAINER_TMPDIR"] = str(tmp_dir)
        env["APPTAINER_CACHEDIR"] = str(cache_dir)

        try:
            result = subprocess.run(
                [executable, "build", str(sif_path), image],
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
            )
            if result.returncode != 0:
                logger.warning("SIF build failed, falling back to docker:// URL")
                logger.warning("  Error: %s", result.stderr[:500])
                return image
            logger.info("SIF image built successfully")
            return str(sif_path)
        except subprocess.TimeoutExpired:
            logger.warning("SIF build timed out, falling back to docker:// URL")
            if sif_path.exists():
                sif_path.unlink()
            return image
        except Exception as e:
            logger.warning("SIF build error: %s, falling back to docker:// URL", e)
            return image


# -------------------------------------------------------------------------
# SingularityEnvironment
# -------------------------------------------------------------------------


class SingularityEnvironment(BaseEnvironment):
    """Hardened Singularity/Apptainer container with resource limits and persistence.

    Security: --containall (isolated PID/IPC/mount namespaces, no host home mount),
    --no-home, writable-tmpfs for scratch space. The container cannot see or modify
    the host filesystem outside of explicitly bound paths.

    Persistence: when enabled, the writable overlay directory is preserved across
    sessions so installed packages and files survive cleanup/restore.
    """

    def __init__(
        self,
        image: str,
        cwd: str = "~",
        timeout: int = 60,
        cpu: float = 0,
        memory: int = 0,
        disk: int = 0,
        persistent_filesystem: bool = False,
        task_id: str = "default",
    ):
        super().__init__(cwd=cwd, timeout=timeout)
        self.executable = "apptainer" if shutil.which("apptainer") else "singularity"
        self.image = _get_or_build_sif(image, self.executable)
        self.instance_id = f"hermes_{uuid.uuid4().hex[:12]}"
        self._instance_started = False
        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._overlay_dir: Optional[Path] = None

        # Resource limits
        self._cpu = cpu
        self._memory = memory

        # Persistent overlay directory
        if self._persistent:
            overlay_base = _get_scratch_dir() / "hermes-overlays"
            overlay_base.mkdir(parents=True, exist_ok=True)
            self._overlay_dir = overlay_base / f"overlay-{task_id}"
            self._overlay_dir.mkdir(parents=True, exist_ok=True)

        self._start_instance()

    def _start_instance(self):
        cmd = [self.executable, "instance", "start"]

        # Security: full isolation from host
        cmd.extend(["--containall", "--no-home"])

        # Writable layer
        if self._persistent and self._overlay_dir:
            # Persistent writable overlay -- survives across restarts
            cmd.extend(["--overlay", str(self._overlay_dir)])
        else:
            cmd.append("--writable-tmpfs")

        # Resource limits (cgroup-based, may require root or appropriate config)
        if self._memory > 0:
            cmd.extend(["--memory", f"{self._memory}M"])
        if self._cpu > 0:
            cmd.extend(["--cpus", str(self._cpu)])

        cmd.extend([str(self.image), self.instance_id])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to start instance: {result.stderr}")
            self._instance_started = True
            logger.info("Singularity instance %s started (persistent=%s)", self.instance_id, self._persistent)
        except subprocess.TimeoutExpired:
            raise RuntimeError("Instance start timed out")

    def execute(
        self, command: str, cwd: str = "", *, timeout: int | None = None, stdin_data: str | None = None
    ) -> dict:
        if not self._instance_started:
            return {"output": "Instance not started", "returncode": -1}

        effective_timeout = timeout or self.timeout
        work_dir = cwd or self.cwd
        exec_command = self._prepare_command(command)

        # apptainer exec --pwd doesn't expand ~, so prepend a cd into the command
        if work_dir == "~" or work_dir.startswith("~/"):
            exec_command = f"cd {work_dir} && {exec_command}"
            work_dir = "/tmp"

        cmd = [self.executable, "exec", "--pwd", work_dir, f"instance://{self.instance_id}", "bash", "-c", exec_command]

        try:
            import time as _time

            _output_chunks = []
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE if stdin_data else subprocess.DEVNULL,
                text=True,
            )
            if stdin_data:
                try:
                    proc.stdin.write(stdin_data)
                    proc.stdin.close()
                except Exception:
                    pass

            def _drain():
                try:
                    if proc.stdout:
                        for line in proc.stdout:
                            _output_chunks.append(line)
                except Exception:
                    pass

            reader = threading.Thread(target=_drain, daemon=True)
            reader.start()
            deadline = _time.monotonic() + effective_timeout

            while proc.poll() is None:
                if is_interrupted():
                    proc.terminate()
                    try:
                        proc.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    reader.join(timeout=2)
                    return {
                        "output": "".join(_output_chunks) + "\n[Command interrupted]",
                        "returncode": 130,
                    }
                if _time.monotonic() > deadline:
                    proc.kill()
                    reader.join(timeout=2)
                    return self._timeout_result(effective_timeout)
                _time.sleep(0.2)

            reader.join(timeout=5)
            return {"output": "".join(_output_chunks), "returncode": proc.returncode}
        except Exception as e:
            return {"output": f"Singularity execution error: {e}", "returncode": 1}

    def cleanup(self):
        """Stop the instance. If persistent, the overlay dir survives for next creation."""
        if self._instance_started:
            try:
                subprocess.run(
                    [self.executable, "instance", "stop", self.instance_id],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                logger.info("Singularity instance %s stopped", self.instance_id)
            except Exception as e:
                logger.warning("Failed to stop Singularity instance %s: %s", self.instance_id, e)
            self._instance_started = False

        # Record overlay path for persistence restoration
        if self._persistent and self._overlay_dir:
            snapshots = _load_snapshots()
            snapshots[self._task_id] = str(self._overlay_dir)
            _save_snapshots(snapshots)

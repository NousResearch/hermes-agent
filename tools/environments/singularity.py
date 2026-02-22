"""Singularity/Apptainer persistent container environment.

Also contains the Singularity-specific helpers: scratch dir management,
Apptainer cache, and SIF image building.
"""

import logging
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

from tools.environments.base import BaseEnvironment

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Singularity helpers (scratch dir, SIF cache, SIF building)
# -------------------------------------------------------------------------

def _get_scratch_dir() -> Path:
    """Get the best directory for Singularity sandboxes -- prefers /scratch on HPC."""
    custom_scratch = os.getenv("TERMINAL_SCRATCH_DIR")
    if custom_scratch:
        scratch_path = Path(custom_scratch)
        scratch_path.mkdir(parents=True, exist_ok=True)
        return scratch_path

    scratch = Path("/scratch")
    if scratch.exists() and os.access(scratch, os.W_OK):
        user_scratch = scratch / os.getenv("USER", "hermes") / "hermes-agent"
        user_scratch.mkdir(parents=True, exist_ok=True)
        logger.info("Using /scratch for sandboxes: %s", user_scratch)
        return user_scratch

    logger.debug("/scratch not available, using /tmp for sandboxes")
    return Path(tempfile.gettempdir())


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
    if image.endswith('.sif') and Path(image).exists():
        return image
    if not image.startswith('docker://'):
        return image

    image_name = image.replace('docker://', '').replace('/', '-').replace(':', '-')
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
                capture_output=True, text=True, timeout=600, env=env,
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
    """Persistent Singularity/Apptainer container environment.

    Uses ``apptainer instance`` to create a long-running container that persists
    state across all commands within a task.
    """

    def __init__(self, image: str, cwd: str = "/root", timeout: int = 60):
        super().__init__(cwd=cwd, timeout=timeout)
        self.executable = "apptainer" if shutil.which("apptainer") else "singularity"
        self.image = _get_or_build_sif(image, self.executable)
        self.instance_id = f"hermes_{uuid.uuid4().hex[:12]}"
        self._instance_started = False
        self._start_instance()

    def _start_instance(self):
        cmd = [
            self.executable, "instance", "start",
            "--writable-tmpfs", "--containall",
            str(self.image), self.instance_id,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to start instance: {result.stderr}")
            self._instance_started = True
            logger.info("Singularity instance %s started", self.instance_id)
        except subprocess.TimeoutExpired:
            raise RuntimeError("Instance start timed out")

    def execute(self, command: str, cwd: str = "", *,
                timeout: int | None = None,
                stdin_data: str | None = None) -> dict:
        if not self._instance_started:
            return {"output": "Instance not started", "returncode": -1}

        cmd = [self.executable, "exec", "--pwd", cwd or self.cwd,
               f"instance://{self.instance_id}",
               "bash", "-c", self._prepare_command(command)]

        try:
            result = subprocess.run(cmd, **self._build_run_kwargs(timeout, stdin_data))
            return {"output": result.stdout, "returncode": result.returncode}
        except subprocess.TimeoutExpired:
            return self._timeout_result(timeout)

    def cleanup(self):
        if self._instance_started:
            try:
                subprocess.run(
                    [self.executable, "instance", "stop", self.instance_id],
                    capture_output=True, text=True, timeout=30,
                )
                logger.info("Singularity instance %s stopped", self.instance_id)
            except Exception as e:
                logger.warning("Failed to stop Singularity instance %s: %s", self.instance_id, e)
            self._instance_started = False

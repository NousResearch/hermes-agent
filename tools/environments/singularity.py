"""Singularity/Apptainer persistent container environment.

Security-hardened with --containall, --no-home, capability dropping. Supports
resource limits and optional persistence via writable overlay dirs that survive sessions.
"""

import logging
import hashlib
import os
import shutil
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Mapping, Optional

from hermes_constants import get_hermes_home
from tools.environments.base import BaseEnvironment, _load_json_store, _save_json_store
from tools.environments.base_output import _popen_bash
from tools.environments.path_utils import sanitize_task_id_for_path
from tools.environments.remote_common import bash_argv, run_capture

logger = logging.getLogger(__name__)


def _snapshot_store_path(owner_home: str | os.PathLike | None = None) -> Path:
    """Return the active profile's Singularity snapshot registry path."""
    home = Path(owner_home).resolve() if owner_home is not None else get_hermes_home()
    return home / "singularity_snapshots.json"

# Apptainer accepts these exact variables for private Docker-registry pulls.
# Image construction gets this narrow capability set back after the common
# sanitizer runs; it never receives the rest of the trusted Hermes env.
_REGISTRY_AUTH_ENV_VARS = (
    "APPTAINER_DOCKER_USERNAME",
    "APPTAINER_DOCKER_PASSWORD",
    "SINGULARITY_DOCKER_USERNAME",
    "SINGULARITY_DOCKER_PASSWORD",
    "DOCKER_USERNAME",
    "DOCKER_PASSWORD",
)


def _registry_auth_source(
    owner_home: str | os.PathLike | None = None,
) -> Mapping[str, str]:
    """Return registry grants from the exact target profile authority.

    A multiplexed process must bind the grants to the captured artifact owner.
    Falling back to process-global ``os.environ`` there would reintroduce the
    launch profile's credentials after the generic child boundary sanitized
    them. Non-multiplex callers retain the historical ambient fallback.
    """
    from agent.secret_scope import (
        build_profile_secret_scope,
        current_secret_scope,
        is_multiplex_active,
    )

    scope = current_secret_scope()
    multiplex_active = is_multiplex_active()
    owner = Path(owner_home).resolve() if owner_home is not None else None
    if scope is not None:
        if multiplex_active:
            if owner is not None and scope.profile_home != owner:
                raise RuntimeError(
                    "Singularity registry authority does not match artifact owner"
                )
            return scope
        source = dict(os.environ)
        source.update({str(key): str(value) for key, value in scope.items()})
        return source
    if multiplex_active:
        if owner is None:
            raise RuntimeError(
                "Singularity image build requested without an active target-profile "
                "secret scope or explicit artifact owner while multiplexing is enabled"
            )
        return build_profile_secret_scope(owner, fail_closed_external=True)
    return os.environ


def _singularity_subprocess_env(
    *,
    include_registry_auth: bool = False,
    owner_home: str | os.PathLike | None = None,
    source_home: str | os.PathLike | None = None,
) -> dict[str, str]:
    from agent.secret_scope import is_multiplex_active

    enforce_profile_boundary = owner_home is not None and (
        is_multiplex_active()
        or (
            source_home is not None
            and Path(source_home).resolve() != Path(owner_home).resolve()
        )
    )
    env = build_subprocess_env(
        profile_home=owner_home,
        source_profile_home=source_home,
        enforce_profile_boundary=enforce_profile_boundary,
    )
    if include_registry_auth:
        auth_names = {name.upper() for name in _REGISTRY_AUTH_ENV_VARS}
        for key in list(env):
            if key.upper() in auth_names:
                env.pop(key, None)
        source = _registry_auth_source(owner_home)
        for key in _REGISTRY_AUTH_ENV_VARS:
            value = source.get(key)
            if value is not None:
                env[key] = str(value)
    return env


def _find_singularity_executable() -> str:
    """Locate the apptainer or singularity CLI binary."""
    for exe in ("apptainer", "singularity"):
        if shutil.which(exe):
            return exe
    raise RuntimeError(
        "Neither 'apptainer' nor 'singularity' was found in PATH. "
        "Install Apptainer (https://apptainer.org/docs/admin/main/installation.html) "
        "or Singularity and ensure the CLI is available.")


def _ensure_singularity_available(
    *,
    owner_home: str | os.PathLike | None = None,
    source_home: str | os.PathLike | None = None,
) -> str:
    """Preflight check: resolve the executable and verify it responds."""
    exe = _find_singularity_executable()
    try:
        result = run_capture([exe, "version"], timeout=10)
    except FileNotFoundError:
        raise RuntimeError(f"Singularity backend selected but '{exe}' could not be executed.")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"'{exe} version' timed out.")
    if result.returncode != 0:
        stderr = result.stderr.strip()[:200]
        raise RuntimeError(f"'{exe} version' failed (exit code {result.returncode}): {stderr}")
    return exe


def _load_snapshots(owner_home: str | os.PathLike | None = None) -> dict:
    return _load_json_store(_snapshot_store_path(owner_home))


def _save_snapshots(
    data: dict,
    owner_home: str | os.PathLike | None = None,
) -> None:
    _save_json_store(_snapshot_store_path(owner_home), data)


def _get_scratch_dir() -> Path:
    """``TERMINAL_SCRATCH_DIR`` override, else a writable ``/scratch`` (HPC), else the sandbox dir."""
    custom_scratch = os.getenv("TERMINAL_SCRATCH_DIR")
    if custom_scratch:
        scratch_path = Path(custom_scratch)
    else:
        from tools.environments.base import get_sandbox_dir
        scratch_path = get_sandbox_dir() / "singularity"
        scratch = Path("/scratch")
        if scratch.exists() and os.access(scratch, os.W_OK):
            scratch_path = scratch / os.getenv("USER", "hermes") / "hermes-agent"
            scratch_path.mkdir(parents=True, exist_ok=True)
            logger.info("Using /scratch for sandboxes: %s", scratch_path)
    scratch_path.mkdir(parents=True, exist_ok=True)
    return scratch_path


def _get_apptainer_cache_dir() -> Path:
    cache_dir = os.getenv("APPTAINER_CACHEDIR")
    cache_path = Path(cache_dir) if cache_dir else _get_scratch_dir() / ".apptainer"
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


_sif_build_lock = threading.Lock()


def _get_or_build_sif(image: str, executable: str = "apptainer") -> str:
    """Build (once, cached) a SIF from a ``docker://`` URL; falls back to the URL on failure."""
    if (image.endswith('.sif') and Path(image).exists()) or not image.startswith('docker://'):
        return image

    explicit_owner = Path(owner_home).resolve() if owner_home is not None else None
    owner = explicit_owner or get_hermes_home().resolve()
    owner_id = _profile_artifact_id(owner)
    image_id = _image_identity(image)
    cache_dir = _get_apptainer_cache_dir()
    sif_path = cache_dir / f"{image_name}.sif"
    if sif_path.exists():
        return str(sif_path)

    with _sif_build_lock:
        if cacheable and sif_path.exists():
            return str(sif_path)

        logger.info("Building SIF image (one-time setup)...")
        logger.info("  Source: %s", image)
        logger.info("  Target: %s", sif_path)

        tmp_dir = profile_cache / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        build_path = tmp_dir / f"{image_id}-{uuid.uuid4().hex}.building.sif"

        # External build tool may need registry credentials from the user env — exact preservation.
        from tools.environments.local import build_subprocess_env
        env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=False)
        env["APPTAINER_TMPDIR"] = str(tmp_dir)
        env["APPTAINER_CACHEDIR"] = str(cache_dir)

        try:
            result = run_capture([executable, "build", str(sif_path), image], timeout=600, env=env)
            if result.returncode != 0:
                logger.warning("SIF build failed, falling back to docker:// URL")
                logger.warning("  Error: %s", result.stderr[:500])
                return image
            if not build_path.exists():
                raise RuntimeError("Apptainer reported success without publishing a SIF")
            os.replace(build_path, sif_path)
            logger.info("SIF image built successfully")
            return str(sif_path)
        except subprocess.TimeoutExpired:
            logger.warning("SIF build timed out, falling back to docker:// URL")
            if build_path.exists():
                build_path.unlink()
            return image
        except Exception as e:
            if build_path.exists():
                build_path.unlink()
            logger.warning("SIF build error: %s, falling back to docker:// URL", e)
            return image


class SingularityEnvironment(BaseEnvironment):
    """Hardened Singularity/Apptainer container with resource limits and persistence.

    Spawn-per-call: every execute() spawns a fresh ``apptainer exec ... bash -c`` process.
    Session snapshot preserves env vars across calls; CWD persists via in-band stdout markers.
    """

    def __init__(self, image: str, cwd: str = "~", timeout: int = 60, cpu: float = 0,
                 memory: int = 0, disk: int = 0, persistent_filesystem: bool = False,
                 task_id: str = "default"):
        super().__init__(cwd=cwd, timeout=timeout)
        boundary = getattr(self, "_profile_env_boundary", None)
        from hermes_constants import get_process_hermes_home

        self._owner_home = (
            boundary.target_home
            if boundary is not None
            else get_hermes_home().resolve()
        )
        self._source_home = (
            boundary.source_home
            if boundary is not None
            else Path(get_process_hermes_home()).resolve()
        )
        self._owner_generation = (
            boundary.target_generation if boundary is not None else ""
        )
        self._artifact_epoch = (
            hashlib.sha256(self._owner_generation.encode()).hexdigest()
            if self._owner_generation
            else "single-profile"
        )
        self.executable = _ensure_singularity_available(
            owner_home=self._owner_home,
            source_home=self._source_home,
        )
        self.image = _get_or_build_sif(
            image,
            self.executable,
            owner_home=self._owner_home,
            source_home=self._source_home,
            policy_generation=self._owner_generation,
        )
        self._image_authority_id = _resolved_image_identity(
            image,
            self.image,
            require_immutable=boundary is not None,
        )
        self.instance_id = f"hermes_{uuid.uuid4().hex[:12]}"
        self._instance_started = False
        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._overlay_dir: Optional[Path] = None
        self._cpu = cpu
        self._memory = memory

        if self._persistent:
            # A raw session-key task_id carries colons etc. unsafe in host path components;
            # the shared sanitizer keeps all backends agreeing on the mapping.
            self._overlay_dir = (
                _get_scratch_dir() / "hermes-overlays" / f"overlay-{sanitize_task_id_for_path(task_id)}")
            self._overlay_dir.mkdir(parents=True, exist_ok=True)

        self._start_instance()
        self.init_session()

    def _start_instance(self):
        cmd = [self.executable, "instance", "start", "--containall", "--no-home"]
        if self._persistent and self._overlay_dir:
            cmd.extend(["--overlay", str(self._overlay_dir)])
        else:
            cmd.append("--writable-tmpfs")

        try:
            from tools.credential_files import get_credential_file_mounts, get_skills_directory_mount
            for entry in (*get_credential_file_mounts(), *get_skills_directory_mount()):
                cmd.extend(["--bind", f"{entry['host_path']}:{entry['container_path']}:ro"])
        except Exception as e:
            logger.debug("Singularity: could not load credential/skills mounts: %s", e)

        if self._memory > 0:
            cmd.extend(["--memory", f"{self._memory}M"])
        if self._cpu > 0:
            cmd.extend(["--cpus", str(self._cpu)])
        cmd.extend([str(self.image), self.instance_id])

        try:
            result = run_capture(cmd, timeout=120)
        except subprocess.TimeoutExpired:
            raise RuntimeError("Instance start timed out")
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start instance: {result.stderr}")
        self._instance_started = True
        logger.info("Singularity instance %s started (persistent=%s)", self.instance_id, self._persistent)

    def _run_bash(self, cmd_string: str, *, login: bool = False, timeout: int = 120,
                  stdin_data: str | None = None) -> subprocess.Popen:
        """Spawn a bash process inside the Singularity instance."""
        if not self._instance_started:
            raise RuntimeError("Singularity instance not started")
        cmd = [self.executable, "exec", f"instance://{self.instance_id}", *bash_argv(cmd_string, login)]
        return _popen_bash(cmd, stdin_data)

    def cleanup(self):
        """Stop the instance. If persistent, the overlay dir survives."""
        owner_home = self._owner_home
        source_home = self._source_home
        if self._instance_started:
            try:
                run_capture([self.executable, "instance", "stop", self.instance_id], timeout=30)
                logger.info("Singularity instance %s stopped", self.instance_id)
            except Exception as e:
                logger.warning("Failed to stop Singularity instance %s: %s", self.instance_id, e)
            self._instance_started = False

        if self._persistent and self._overlay_dir:
            snapshots = _load_snapshots(owner_home)
            snapshot_key = (
                f"{self._image_authority_id}:{self._artifact_epoch}:{self._task_id}"
            )
            snapshots[snapshot_key] = str(self._overlay_dir)
            _save_snapshots(snapshots, owner_home)

"""Independent runtime supervision and out-of-band recovery contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
from enum import Enum
from typing import Any, Callable, Mapping
from uuid import uuid4

from hermes_constants import get_hermes_home


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SupervisorState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    RESTARTING = "restarting"
    DEGRADED = "degraded"
    FAILED = "failed"
    ROLLED_BACK = "rolled-back"


@dataclass(slots=True)
class RuntimeHealth:
    healthy: bool
    state: str
    checked_at: str = field(default_factory=_utc_now)
    detail: str = ""


@dataclass(slots=True)
class RuntimeCheckpoint:
    checkpoint_id: str
    label: str
    artifact_path: str
    created_at: str = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


class SubprocessRuntime:
    """Small child-process wrapper used by the independent supervisor."""

    def __init__(self, command: list[str], *, cwd: Path | None = None, env: Mapping[str, str] | None = None) -> None:
        if not command:
            raise ValueError("runtime command is required")
        self.command = list(command)
        self.cwd = str(cwd) if cwd is not None else None
        self.env = dict(env) if env is not None else None
        self.process: subprocess.Popen[bytes] | None = None

    def start(self) -> "SubprocessRuntime":
        if self.is_alive():
            return self
        self.process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=self.env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return self

    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self, timeout: float = 5.0) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=max(0.0, timeout))
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=max(0.0, timeout))


class RuntimeSupervisor:
    """Supervise a runtime through an object boundary independent of the runtime.

    The factory/health callbacks are intentionally small so a real service
    process and a test double use the same lifecycle contract.
    """

    def __init__(
        self,
        *,
        runtime_factory: Callable[[], Any],
        health_check: Callable[[Any], bool],
        state_path: Path | None = None,
        checkpoint_dir: Path | None = None,
        max_restart_attempts: int = 3,
    ) -> None:
        root = get_hermes_home() / "workstation"
        self.state_path = Path(state_path or root / "supervisor.json")
        self.checkpoint_dir = Path(checkpoint_dir or root / "checkpoints")
        self.runtime_factory = runtime_factory
        self.health_check = health_check
        self.runtime: Any | None = None
        self.state = SupervisorState.STOPPED
        self.last_error: str | None = None
        self._checkpoints: list[RuntimeCheckpoint] = []
        self.max_restart_attempts = max(1, max_restart_attempts)
        self._restart_failures = 0
        self._load_state()

    @classmethod
    def from_command(
        cls,
        command: list[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        state_path: Path | None = None,
        checkpoint_dir: Path | None = None,
        max_restart_attempts: int = 3,
    ) -> "RuntimeSupervisor":
        return cls(
            runtime_factory=lambda: SubprocessRuntime(command, cwd=cwd, env=env).start(),
            health_check=lambda runtime: runtime.is_alive(),
            state_path=state_path,
            checkpoint_dir=checkpoint_dir,
            max_restart_attempts=max_restart_attempts,
        )

    def _load_state(self) -> None:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            return
        self._checkpoints = [
            RuntimeCheckpoint(**item)
            for item in data.get("checkpoints", [])
            if isinstance(item, dict) and item.get("artifact_path")
        ]

    def _persist_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state": self.state,
            "last_error": self.last_error,
            "checkpoints": [asdict(checkpoint) for checkpoint in self._checkpoints[-10:]],
        }
        temp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp.replace(self.state_path)

    def start(self) -> bool:
        if self.runtime is not None and self.health().healthy:
            return True
        self.state = SupervisorState.STARTING
        self.last_error = None
        try:
            self.runtime = self.runtime_factory()
            if not self.health_check(self.runtime):
                raise RuntimeError("runtime health check failed during start")
            self.state = SupervisorState.RUNNING
            self._restart_failures = 0
            self._persist_state()
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self.state = SupervisorState.FAILED
            self._persist_state()
            return False

    def health(self) -> RuntimeHealth:
        if self.runtime is None:
            return RuntimeHealth(False, self.state, detail="runtime is not started")
        previous_state = self.state
        try:
            healthy = bool(self.health_check(self.runtime))
            if not healthy and self.state == SupervisorState.RUNNING:
                self.state = SupervisorState.DEGRADED
            if self.state != previous_state:
                self._persist_state()
            return RuntimeHealth(healthy, self.state, detail="health check")
        except Exception as exc:
            self.state = SupervisorState.DEGRADED
            self.last_error = str(exc)
            self._persist_state()
            return RuntimeHealth(False, self.state, detail=str(exc))

    def stop(self) -> None:
        if self.runtime is not None:
            stop = getattr(self.runtime, "stop", None)
            if callable(stop):
                stop()
        self.runtime = None
        self.state = SupervisorState.STOPPED
        self._persist_state()

    def restart(self) -> bool:
        self.state = SupervisorState.RESTARTING
        self._persist_state()
        self.stop()
        return self.start()

    def watchdog_step(self) -> RuntimeHealth:
        """Run one independent health/recovery cycle without agent-runtime help."""
        current = self.health()
        if current.healthy:
            return current
        if self._restart_failures >= self.max_restart_attempts:
            self.state = SupervisorState.FAILED
            self.last_error = "runtime crash loop detected"
            self._persist_state()
            return RuntimeHealth(False, self.state, detail=self.last_error)
        self._restart_failures += 1
        if self.restart():
            return self.health()
        if self._restart_failures >= self.max_restart_attempts:
            self.state = SupervisorState.FAILED
            self.last_error = self.last_error or "runtime restart failed"
            self._persist_state()
        return RuntimeHealth(False, self.state, detail=self.last_error or "runtime restart failed")

    def checkpoint_artifact(
        self,
        artifact_path: Path,
        *,
        label: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> RuntimeCheckpoint:
        source = Path(artifact_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_id = f"checkpoint-{uuid4().hex}"
        destination = self.checkpoint_dir / f"{checkpoint_id}-{source.name}"
        temp = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copy2(source, temp)
        temp.replace(destination)
        checkpoint = RuntimeCheckpoint(
            checkpoint_id=checkpoint_id,
            label=label,
            artifact_path=str(destination),
            metadata=dict(metadata or {}),
        )
        self._checkpoints.append(checkpoint)
        self._persist_state()
        return checkpoint

    def restore_last_good(self, target_path: Path) -> bool:
        for checkpoint in reversed(self._checkpoints):
            source = Path(checkpoint.artifact_path)
            if not source.is_file():
                continue
            target = Path(target_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(target.suffix + ".restore.tmp")
            shutil.copy2(source, temp)
            temp.replace(target)
            self.state = SupervisorState.ROLLED_BACK
            self._persist_state()
            return True
        return False

    def diagnostics(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "healthy": self.health().healthy if self.runtime is not None else False,
            "last_error": self.last_error,
            "checkpoints": [asdict(checkpoint) for checkpoint in self._checkpoints[-10:]],
        }


class RecoveryPlane:
    """Minimal control-plane record for safe-mode diagnostics/quarantine."""

    def __init__(self, state_path: Path | None = None) -> None:
        self.state_path = Path(
            state_path or get_hermes_home() / "workstation" / "recovery-plane.json"
        )
        self._components: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            return
        if isinstance(data, dict):
            self._components = {
                str(name): value
                for name, value in data.get("components", {}).items()
                if isinstance(value, dict)
            }

    def _persist(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temp.write_text(
            json.dumps({"components": self._components}, indent=2),
            encoding="utf-8",
        )
        temp.replace(self.state_path)

    def quarantine(self, component: str, reason: str) -> bool:
        name = component.strip()
        if not name:
            raise ValueError("component name is required")
        self._components[name] = {
            "status": "quarantined",
            "reason": reason,
            "updated_at": _utc_now(),
        }
        self._persist()
        return True

    def restore(self, component: str) -> bool:
        if component not in self._components:
            return False
        del self._components[component]
        self._persist()
        return True

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {name: dict(value) for name, value in self._components.items()}

    def is_quarantined(self, component: str) -> bool:
        return self._components.get(component, {}).get("status") == "quarantined"

    def diagnose(self, checks: Mapping[str, Callable[[], Any]]) -> dict[str, dict[str, Any]]:
        report: dict[str, dict[str, Any]] = {}
        for name, check in checks.items():
            if self.is_quarantined(name):
                record = dict(self._components[name])
                report[name] = record
                continue
            try:
                result = check()
                healthy = result is True or (isinstance(result, str) and result.lower() in {"ok", "healthy"})
                report[name] = {
                    "status": "healthy" if healthy else "unhealthy",
                    "detail": result if isinstance(result, str) else "check completed",
                }
            except Exception as exc:
                report[name] = {"status": "unhealthy", "detail": str(exc)}
        return report

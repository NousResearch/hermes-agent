"""Modal-backed desktop environment with a task-scoped CUA service."""

from __future__ import annotations

import base64
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tools.computer_use.backend import ActionResult, CaptureResult, ComputerUseBackend, UIElement
from tools.computer_use.transports.base import CuaToolTransport
from tools.computer_use.transports.stdio import StdioMcpTransport
from tools.environments.compute_provider import ComputeLease, EnvironmentCapabilities
from tools.environments.modal import ModalEnvironment


@dataclass(frozen=True)
class ModalDesktopConfig:
    image: str = "trycua/cua:latest"
    cwd: str = "/root"
    timeout: int = 60
    persistent_filesystem: bool = True
    cpu: float = 2
    memory: int = 8192
    cua_driver_command: tuple[str, ...] = ("cua-driver", "mcp")
    sandbox_kwargs: Mapping[str, Any] = field(default_factory=dict)


class _TransportComputerBackend(ComputerUseBackend):
    """Adapt a CUA MCP transport to Hermes' synchronous backend contract."""

    def __init__(self, transport: CuaToolTransport):
        self.transport = transport

    def start(self) -> None:
        self.transport.start()

    def stop(self) -> None:
        self.transport.stop()

    def is_available(self) -> bool:
        return self.transport.is_alive()

    def capture(self, mode: str = "som", app: Optional[str] = None, pid: Optional[int] = None,
                window_id: Optional[int] = None) -> CaptureResult:
        result = self.transport.call_tool("capture", {
            "mode": mode, "app": app, "pid": pid, "window_id": window_id,
        })
        data = _result_data(result)
        elements = [_element_from(item) for item in data.get("elements", []) if isinstance(item, Mapping)]
        png_b64 = data.get("png_b64") or data.get("image")
        if isinstance(png_b64, str) and png_b64.startswith("data:"):
            png_b64 = png_b64.split(",", 1)[-1]
        return CaptureResult(
            mode=mode, width=int(data.get("width", 0)), height=int(data.get("height", 0)),
            png_b64=png_b64 if isinstance(png_b64, str) else None, elements=elements,
            app=str(data.get("app", app or "")), window_title=str(data.get("window_title", "")),
        )

    def click(self, **kwargs: Any) -> ActionResult:
        return self._action("click", kwargs)

    def drag(self, **kwargs: Any) -> ActionResult:
        return self._action("drag", kwargs)

    def scroll(self, **kwargs: Any) -> ActionResult:
        return self._action("scroll", kwargs)

    def type_text(self, text: str, **kwargs: Any) -> ActionResult:
        return self._action("type_text", {"text": text, **kwargs})

    def key(self, keys: str, **kwargs: Any) -> ActionResult:
        return self._action("hotkey", {"keys": keys, **kwargs})

    def list_apps(self) -> List[Dict[str, Any]]:
        result = _result_data(self.transport.call_tool("list_apps", {}))
        apps = result.get("apps", result.get("result", []))
        return apps if isinstance(apps, list) else []

    def list_windows(self) -> List[Dict[str, Any]]:
        result = _result_data(self.transport.call_tool("list_windows", {}))
        windows = result.get("windows", result.get("result", []))
        return windows if isinstance(windows, list) else []

    def focus_app(self, app: str, raise_window: bool = False) -> ActionResult:
        return self._action("focus_app", {"app": app, "raise_window": raise_window})

    def set_value(self, value: str, element: Optional[int] = None) -> ActionResult:
        return self._action("set_value", {"value": value, "element": element})

    def _action(self, action: str, arguments: Mapping[str, Any]) -> ActionResult:
        data = _result_data(self.transport.call_tool(action, arguments))
        return ActionResult(
            ok=bool(data.get("ok", True)), action=action, message=str(data.get("message", "")),
            meta=dict(data.get("meta", {})) if isinstance(data.get("meta"), Mapping) else {},
        )


def _result_data(result: Mapping[str, Any]) -> Mapping[str, Any]:
    structured = result.get("structuredContent")
    if isinstance(structured, Mapping):
        return structured
    return result


def _element_from(value: Mapping[str, Any]) -> UIElement:
    bounds = value.get("bounds", (0, 0, 0, 0))
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        bounds = (0, 0, 0, 0)
    return UIElement(
        index=int(value.get("index", 0)), role=str(value.get("role", "")),
        label=str(value.get("label", "")), bounds=tuple(int(part) for part in bounds),
        app=str(value.get("app", "")), pid=int(value.get("pid", 0)),
        window_id=int(value.get("window_id", 0)),
    )


class ModalDesktopEnvironment(ModalEnvironment):
    """Modal terminal environment exposing the paired CUA driver backend."""

    def __init__(self, *, compute_lease: ComputeLease, config: ModalDesktopConfig):
        self._compute_lease = compute_lease
        self._desktop_config = config
        self._computer_backend: ComputerUseBackend | None = None
        sandbox_kwargs = {"cpu": config.cpu, "memory": config.memory, **dict(config.sandbox_kwargs)}
        super().__init__(
            image=config.image, cwd=config.cwd, timeout=config.timeout,
            modal_sandbox_kwargs=sandbox_kwargs,
            persistent_filesystem=config.persistent_filesystem, task_id=compute_lease.task_id,
        )

    @property
    def compute_lease(self) -> ComputeLease:
        return self._compute_lease

    def get_computer_backend(self) -> ComputerUseBackend:
        if self._computer_backend is None:
            # The image owns cua-driver. The default command preserves the
            # existing stdio MCP contract while the lease owns its lifecycle.
            self._computer_backend = _TransportComputerBackend(
                StdioMcpTransport(self._desktop_config.cua_driver_command)
            )
            self._computer_backend.start()
        return self._computer_backend

    def cleanup(self):
        if self._computer_backend is not None:
            self._computer_backend.stop()
        super().cleanup()


class ModalDesktopProvider:
    """Provision Modal desktop sandboxes and expose their shared lease."""

    name = "modal"

    def __init__(self, config: ModalDesktopConfig | None = None):
        self.config = config or ModalDesktopConfig()

    def acquire(self, task_id: str, *, image: str | None = None,
                capabilities: Sequence[str] | None = None) -> ComputeLease:
        enabled = EnvironmentCapabilities(computer_use=True)
        requested = frozenset(capabilities or enabled.to_capabilities())
        missing = requested - enabled.to_capabilities()
        if missing:
            raise ValueError(f"Modal desktop image lacks requested capabilities: {sorted(missing)}")
        return ComputeLease(
            task_id=task_id, lease_id=uuid.uuid4().hex, provider=self.name,
            image=image or self.config.image, capabilities=enabled,
        )

    def create_environment(self, lease: ComputeLease) -> ModalDesktopEnvironment:
        config = self.config if lease.image == self.config.image else ModalDesktopConfig(
            image=lease.image, cwd=self.config.cwd, timeout=self.config.timeout,
            persistent_filesystem=self.config.persistent_filesystem, cpu=self.config.cpu,
            memory=self.config.memory, cua_driver_command=self.config.cua_driver_command,
            sandbox_kwargs=self.config.sandbox_kwargs,
        )
        return ModalDesktopEnvironment(compute_lease=lease, config=config)

    def release(self, lease: ComputeLease) -> None:
        return None

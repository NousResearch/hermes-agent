"""Host-owned terminal backend definitions and factories.

The registry stores every user-selectable built-in backend.  Factories live
here rather than in the terminal tool so both legacy and registry runtimes use
the same construction path while lifecycle ownership remains with the host.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from tools.environments.definitions import (
    BackendCapabilities,
    BackendDefinition,
    BackendFactoryRequest,
    ExecutionLocation,
    FilesystemSemantics,
)

if TYPE_CHECKING:
    from tools.environments.base import BaseEnvironment
    from tools.environments.registry import TerminalBackendRegistry


def _nested_config(request: BackendFactoryRequest, name: str) -> dict[str, Any]:
    value = request.terminal_config.get(name)
    return dict(value) if isinstance(value, dict) else {}


def _require_backend(request: BackendFactoryRequest, expected: str) -> None:
    if request.backend_name != expected:
        raise ValueError(
            f"Factory for terminal backend {expected!r} received request for "
            f"{request.backend_name!r}"
        )


def _create_local_environment(request: BackendFactoryRequest) -> "BaseEnvironment":
    _require_backend(request, "local")
    from tools.environments.local import LocalEnvironment

    # local_config.persistent is intentionally ignored: this matches the
    # pre-registry constructor path, where local persistence is handled by the
    # host lifecycle rather than the environment constructor.
    return LocalEnvironment(cwd=request.cwd, timeout=request.timeout)


def _create_docker_environment(request: BackendFactoryRequest) -> "BaseEnvironment":
    _require_backend(request, "docker")
    from tools.environments.docker import DockerEnvironment
    from tools.terminal_tool import (
        _docker_session_isolation_enabled,
        _has_isolation_overrides,
        _maybe_reap_docker_orphans,
    )

    config = _nested_config(request, "container_config")
    _maybe_reap_docker_orphans(config)
    session_scoped = (
        _docker_session_isolation_enabled()
        and request.task_id != "default"
        and not _has_isolation_overrides(request.task_id)
    )
    environment = DockerEnvironment(
        image=request.image,
        cwd=request.cwd,
        timeout=request.timeout,
        cpu=config.get("container_cpu", 1),
        memory=config.get("container_memory", 5120),
        disk=config.get("container_disk", 51200),
        persistent_filesystem=config.get("container_persistent", True),
        task_id=request.task_id,
        volumes=config.get("docker_volumes", []),
        host_cwd=request.host_cwd,
        auto_mount_cwd=config.get("docker_mount_cwd_to_workspace", False),
        forward_env=config.get("docker_forward_env", []),
        env=config.get("docker_env", {}),
        run_as_host_user=config.get("docker_run_as_host_user", False),
        network=config.get("docker_network", True),
        extra_args=config.get("docker_extra_args", []),
        persist_across_processes=(
            False
            if session_scoped
            else config.get("docker_persist_across_processes", True)
        ),
        shm_size=config.get("docker_shm_size", "1g"),
    )
    environment._session_scoped = session_scoped
    return environment


def _create_singularity_environment(
    request: BackendFactoryRequest,
) -> "BaseEnvironment":
    _require_backend(request, "singularity")
    from tools.environments.singularity import SingularityEnvironment

    config = _nested_config(request, "container_config")
    return SingularityEnvironment(
        image=request.image,
        cwd=request.cwd,
        timeout=request.timeout,
        cpu=config.get("container_cpu", 1),
        memory=config.get("container_memory", 5120),
        disk=config.get("container_disk", 51200),
        persistent_filesystem=config.get("container_persistent", True),
        task_id=request.task_id,
    )


def _modal_sandbox_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    cpu = config.get("container_cpu", 1)
    memory = config.get("container_memory", 5120)
    disk = config.get("container_disk", 51200)
    if cpu > 0:
        kwargs["cpu"] = cpu
    if memory > 0:
        kwargs["memory"] = memory
    if disk > 0:
        try:
            import modal

            if "ephemeral_disk" in inspect.signature(
                modal.Sandbox.create
            ).parameters:
                kwargs["ephemeral_disk"] = disk
        except Exception:
            pass
    return kwargs


def _create_modal_environment(request: BackendFactoryRequest) -> "BaseEnvironment":
    _require_backend(request, "modal")
    from tools.environments.managed_modal import ManagedModalEnvironment
    from tools.environments.modal import ModalEnvironment
    from tools.managed_tool_gateway import is_managed_tool_gateway_ready
    from tools.tool_backend_helpers import (
        has_direct_modal_credentials,
        managed_nous_tools_enabled,
        nous_tool_gateway_unavailable_message,
        resolve_modal_backend_state,
    )

    config = _nested_config(request, "container_config")
    sandbox_kwargs = _modal_sandbox_kwargs(config)
    modal_state = resolve_modal_backend_state(
        config.get("modal_mode"),
        has_direct=has_direct_modal_credentials(),
        managed_ready=is_managed_tool_gateway_ready("modal"),
    )

    common = {
        "image": request.image,
        "cwd": request.cwd,
        "timeout": request.timeout,
        "modal_sandbox_kwargs": sandbox_kwargs,
        "persistent_filesystem": config.get("container_persistent", True),
        "task_id": request.task_id,
    }
    if modal_state["selected_backend"] == "managed":
        return ManagedModalEnvironment(**common)

    if modal_state["selected_backend"] != "direct":
        if modal_state["managed_mode_blocked"]:
            raise ValueError(
                "Modal backend is configured for managed mode, but "
                "Nous Tool Gateway access is not currently available and no direct "
                "Modal credentials/config were found. "
                + nous_tool_gateway_unavailable_message("managed Modal execution")
                + " Choose TERMINAL_MODAL_MODE=direct/auto to use direct Modal credentials."
            )
        if modal_state["mode"] == "managed":
            raise ValueError(
                "Modal backend is configured for managed mode, but the managed "
                "tool gateway is unavailable. "
                + nous_tool_gateway_unavailable_message("managed Modal execution")
            )
        if modal_state["mode"] == "direct":
            raise ValueError(
                "Modal backend is configured for direct mode, but no direct "
                "Modal credentials/config were found."
            )
        message = (
            "Modal backend selected but no direct Modal credentials/config was found."
        )
        if managed_nous_tools_enabled():
            message = (
                "Modal backend selected but no direct Modal credentials/config or "
                "managed tool gateway was found."
            )
        raise ValueError(message)

    return ModalEnvironment(**common)


def _create_daytona_environment(request: BackendFactoryRequest) -> "BaseEnvironment":
    _require_backend(request, "daytona")
    from tools.environments.daytona import DaytonaEnvironment

    config = _nested_config(request, "container_config")
    return DaytonaEnvironment(
        image=request.image,
        cwd=request.cwd,
        timeout=request.timeout,
        cpu=int(config.get("container_cpu", 1)),
        memory=config.get("container_memory", 5120),
        disk=config.get("container_disk", 51200),
        persistent_filesystem=config.get("container_persistent", True),
        task_id=request.task_id,
    )


def _create_vercel_sandbox_environment(
    request: BackendFactoryRequest,
) -> "BaseEnvironment":
    _require_backend(request, "vercel_sandbox")
    from tools.environments.vercel_sandbox import VercelSandboxEnvironment

    config = _nested_config(request, "container_config")
    return VercelSandboxEnvironment(
        runtime=config.get("vercel_runtime") or None,
        cwd=request.cwd,
        timeout=request.timeout,
        cpu=config.get("container_cpu", 1),
        memory=config.get("container_memory", 5120),
        disk=config.get("container_disk", 51200),
        persistent_filesystem=config.get("container_persistent", True),
        task_id=request.task_id,
    )


def _create_ssh_environment(request: BackendFactoryRequest) -> "BaseEnvironment":
    _require_backend(request, "ssh")
    from tools.environments.ssh import SSHEnvironment

    config = _nested_config(request, "ssh_config")
    if not config.get("host") or not config.get("user"):
        raise ValueError(
            "SSH environment requires ssh_host and ssh_user to be configured"
        )
    # ssh_config.persistent is intentionally ignored for legacy parity.
    return SSHEnvironment(
        host=config["host"],
        user=config["user"],
        port=config.get("port", 22),
        key_path=config.get("key", ""),
        cwd=request.cwd,
        timeout=request.timeout,
    )


def raise_unknown_builtin_environment(name: str) -> None:
    """Raise the historical terminal error for an unknown environment name."""
    names = tuple(definition.name for definition in _builtin_backend_definitions())
    choices = ", ".join(repr(candidate) for candidate in names[:-1])
    choices += f", or {names[-1]!r}"
    raise ValueError(f"Unknown environment type: {name}. Use {choices}")


def _builtin_available(name: str) -> bool:
    # Import lazily to avoid a terminal_tool ↔ builtin_backends import cycle.
    from tools.terminal_tool import _check_terminal_backend_requirements

    return _check_terminal_backend_requirements(
        backend_name=name,
        skip_registry=True,
    )


def _local_available() -> bool:
    return _builtin_available("local")


def _docker_available() -> bool:
    return _builtin_available("docker")


def _singularity_available() -> bool:
    return _builtin_available("singularity")


def _modal_available() -> bool:
    return _builtin_available("modal")


def _daytona_available() -> bool:
    return _builtin_available("daytona")


def _vercel_sandbox_available() -> bool:
    return _builtin_available("vercel_sandbox")


def _ssh_available() -> bool:
    return _builtin_available("ssh")


def _remote_capabilities(
    *,
    accepts_host_cwd: bool = False,
    supports_image: bool = False,
    supports_resource_limits: bool = False,
) -> BackendCapabilities:
    return BackendCapabilities(
        execution_location=ExecutionLocation.REMOTE,
        filesystem_semantics=FilesystemSemantics.ISOLATED,
        accepts_host_cwd=accepts_host_cwd,
        requires_sandbox_cwd=True,
        supports_image=supports_image,
        supports_resource_limits=supports_resource_limits,
        supports_pty=False,
        supports_background_processes=True,
        supports_file_transfer=True,
        supports_persistence=True,
    )


def _builtin_backend_definitions() -> tuple[BackendDefinition, ...]:
    return (
        BackendDefinition(
            name="local",
            label="Local",
            description="Execute commands directly on the Hermes host.",
            factory=_create_local_environment,
            availability_check=_local_available,
            capabilities=BackendCapabilities(
                execution_location=ExecutionLocation.LOCAL,
                filesystem_semantics=FilesystemSemantics.HOST,
                accepts_host_cwd=True,
                supports_pty=True,
                supports_background_processes=True,
                supports_persistence=True,
            ),
            source="builtin",
        ),
        BackendDefinition(
            name="docker",
            label="Docker",
            description="Execute commands in a host-managed Docker container.",
            factory=_create_docker_environment,
            availability_check=_docker_available,
            capabilities=_remote_capabilities(
                accepts_host_cwd=True,
                supports_image=True,
                supports_resource_limits=True,
            ),
            default_cwd="/root",
            source="builtin",
        ),
        BackendDefinition(
            name="singularity",
            label="Singularity",
            description="Execute commands in a Singularity or Apptainer container.",
            factory=_create_singularity_environment,
            availability_check=_singularity_available,
            capabilities=_remote_capabilities(
                supports_image=True, supports_resource_limits=True
            ),
            default_cwd="/root",
            source="builtin",
        ),
        BackendDefinition(
            name="modal",
            label="Modal",
            description="Execute commands in direct or managed Modal sandboxes.",
            factory=_create_modal_environment,
            availability_check=_modal_available,
            capabilities=_remote_capabilities(
                supports_image=True, supports_resource_limits=True
            ),
            default_cwd="/root",
            config_schema={
                "modal_mode": {
                    "type": "select",
                    "description": "Modal sandbox mode",
                    "options": ["sandbox", "function"],
                    "config_key": "terminal.modal_mode",
                },
            },
            source="builtin",
        ),
        BackendDefinition(
            name="daytona",
            label="Daytona",
            description="Execute commands in a Daytona cloud sandbox.",
            factory=_create_daytona_environment,
            availability_check=_daytona_available,
            capabilities=_remote_capabilities(
                supports_image=True, supports_resource_limits=True
            ),
            default_cwd="/root",
            source="builtin",
        ),
        BackendDefinition(
            name="vercel_sandbox",
            label="Vercel Sandbox",
            description="Execute commands in a Vercel Sandbox environment.",
            factory=_create_vercel_sandbox_environment,
            availability_check=_vercel_sandbox_available,
            capabilities=_remote_capabilities(supports_resource_limits=True),
            default_cwd="/vercel/sandbox",
            config_schema={
                "vercel_runtime": {
                    "type": "select",
                    "description": "Vercel Sandbox runtime",
                    "options": ["node24", "node22", "python3.13"],
                    "config_key": "terminal.vercel_runtime",
                },
            },
            source="builtin",
        ),
        BackendDefinition(
            name="ssh",
            label="SSH",
            description="Execute commands on a configured SSH host.",
            factory=_create_ssh_environment,
            availability_check=_ssh_available,
            capabilities=_remote_capabilities(),
            default_cwd="~",
            source="builtin",
        ),
    )


def _canonical_builtin_definition(name: str) -> BackendDefinition | None:
    """Return the canonical selectable definition for *name*, if any."""
    return next(
        (
            definition
            for definition in _builtin_backend_definitions()
            if definition.name == name
        ),
        None,
    )


def is_canonical_builtin_definition(definition: BackendDefinition) -> bool:
    """Return whether a definition exactly matches its host-owned canonical form."""
    canonical = _canonical_builtin_definition(definition.name)
    return canonical is not None and definition == canonical


SELECTABLE_BUILTIN_BACKEND_NAMES = frozenset(
    definition.name for definition in _builtin_backend_definitions()
)

# ``managed_modal`` is selected internally by the public ``modal`` backend's
# modal_mode setting. Reserve the implementation identity without exposing it
# as a selectable TERMINAL_ENV value.
RESERVED_BUILTIN_BACKEND_NAMES = (
    SELECTABLE_BUILTIN_BACKEND_NAMES | frozenset({"managed_modal"})
)


def register_builtin_terminal_backends(registry: "TerminalBackendRegistry") -> None:
    """Register canonical built-ins and reject altered reserved definitions."""
    from tools.environments.registry import BackendAlreadyRegisteredError

    for canonical in _builtin_backend_definitions():
        try:
            registry.register_or_verify(canonical)
        except BackendAlreadyRegisteredError:
            raise BackendAlreadyRegisteredError(
                f"Terminal backend {canonical.name!r} conflicts with the canonical "
                "built-in definition"
            ) from None

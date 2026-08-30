"""Apple Container terminal backend, expressed as a registry provider.

Upstream moved terminal-backend classification off hardcoded frozensets onto
:mod:`agent.terminal_env_registry`. This module registers ``apple_container``
through that registry instead of adding an eighth built-in special case at
every classification site.

``apple_container`` is deliberately NOT in
:data:`agent.terminal_env_registry.BUILTIN_BACKEND_NAMES`, so the registry
accepts it. The execution class itself stays in-tree at
:mod:`tools.environments.apple_container`; this file is only the interface
adapter plus the setup, probe and doctor UX the CLI surfaces read.

Registration happens through :func:`register_builtin_providers`, which the
registry calls lazily on its first read. The backend must be classified
correctly even when no plugin discovery ran (a bare ``run_agent`` process
never calls ``discover_plugins``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from agent.terminal_env_provider import TerminalEnvironmentProvider

#: Same default image the wizard and config_defaults.py write.
DEFAULT_IMAGE = "python:3.11-slim-bookworm"


class AppleContainerProvider(TerminalEnvironmentProvider):
    """VM-isolated Linux containers via Apple's ``container`` CLI (macOS 26+)."""

    @property
    def name(self) -> str:
        return "apple_container"

    @property
    def display_name(self) -> str:
        return "Apple Container"

    @property
    def description(self) -> str:
        return "Run commands in a Linux VM using Apple's container CLI (macOS 26+, Apple Silicon)."

    is_remote = True
    is_container = True

    @property
    def skip_container_guards(self) -> bool:
        """Never skip approval prompts: volumes bind host paths into the VM.

        ``tools.approval._should_skip_container_guards`` still decides this
        itself (it reads no provider flag), and it treats apple_container the
        way it treats docker: guards apply once host paths are mounted. This
        flag agrees with that so the two cannot drift apart.
        """
        return False

    @property
    def cache_path_base(self) -> Optional[str]:
        """Cache files are bind-mounted at the container root home."""
        return "/root/.hermes"

    @property
    def env_description(self) -> str:
        return "a Linux VM via Apple Container"

    def is_available(self) -> bool:
        from tools.environments.apple_container import (
            find_container_cli,
            is_apple_container_supported_host,
        )

        return bool(is_apple_container_supported_host() and find_container_cli())

    def check_requirements(self, config: Dict[str, Any]) -> bool:
        from tools.environments.apple_container import _ensure_container_available

        _ensure_container_available()
        return True

    def probe(self) -> Tuple[str, str]:
        from tools.environments.apple_container import (
            container_system_status,
            find_container_cli,
            is_apple_container_supported_host,
        )

        if not is_apple_container_supported_host():
            return (
                "unavailable",
                "Apple Container requires macOS 26 or later on Apple Silicon (arm64).",
            )
        executable = find_container_cli()
        if not executable:
            return (
                "needs_setup",
                "Apple Container CLI not found, install it manually.",
            )
        running, _detail = container_system_status(executable)
        if not running:
            return (
                "needs_setup",
                "Apple Container system is stopped, run `container system start` manually.",
            )
        return ("ready", "")

    def setup_instructions(self) -> List[str]:
        """Static guidance only. The wizard already probed this backend to
        decide whether to offer it; probing again would spawn a second
        `container system status` call per setup run."""
        return [
            "Runs commands in a Linux VM using Apple's container CLI.",
            "If the system is stopped, start it manually: container system start",
            "Tune terminal.apple_container_image / apple_container_volumes / "
            "apple_container_extra_args in config.yaml.",
        ]

    def doctor_checks(self) -> List[Tuple[bool, str, str]]:
        from tools.environments.apple_container import (
            container_system_status,
            find_container_cli,
            is_apple_container_supported_host,
        )

        if not is_apple_container_supported_host():
            return [(
                False,
                "Apple Container requires macOS 26 or later on Apple Silicon (arm64)",
                "(unsupported host)",
            )]
        executable = find_container_cli()
        if not executable:
            return [(
                False,
                "container CLI not found",
                "(required for TERMINAL_ENV=apple_container)",
            )]
        running, _detail = container_system_status(executable)
        if running:
            return [(True, "Apple Container", "(system running)")]
        return [(
            False,
            "Apple Container system not running",
            "(start manually with: container system start)",
        )]

    def apply_config_defaults(self, terminal: Dict[str, Any]) -> None:
        """Seed the terminal.* keys `hermes setup` used to write inline.

        Duck-typed hook, not part of the upstream ABC: the wizard calls it
        when a provider defines it. The image and volume keys only fill a
        gap, but the resource floor is an override: the shared default of 1
        CPU makes an Apple Container VM unusable, and it is free on the
        Apple Silicon hosts this backend requires.
        """
        terminal.setdefault("apple_container_image", DEFAULT_IMAGE)
        terminal.setdefault("apple_container_volumes", [])
        terminal["container_cpu"] = 4
        terminal["container_memory"] = 5120
        terminal["container_persistent"] = True

    def create_environment(
        self,
        *,
        cwd: str,
        timeout: int,
        task_id: str = "default",
        image: Optional[str] = None,
        container_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        from tools.environments.apple_container import AppleContainerEnvironment

        cc = container_config or {}
        return AppleContainerEnvironment(
            image=cc.get("apple_container_image") or image or DEFAULT_IMAGE,
            cwd=cwd,
            timeout=timeout,
            cpu=int(cc.get("container_cpu", 0) or 0),
            memory=int(cc.get("container_memory", 0) or 0),
            persistent_filesystem=bool(cc.get("container_persistent", False)),
            task_id=task_id,
            volumes=cc.get("apple_container_volumes", []),
            extra_args=cc.get("apple_container_extra_args", []),
        )


def register_builtin_providers() -> None:
    """Register the in-tree providers that ride the plugin registry.

    Idempotent: ``register_provider`` overwrites a same-named entry, and the
    registry only calls this once per generation.
    """
    from agent.terminal_env_registry import register_provider

    register_provider(AppleContainerProvider())

"""Apple Container provider for the existing terminal registry.

The registry exposes this in-tree provider as an immutable fallback and reserves
its name against plugin replacement. No plugin discovery or bootstrap is needed.
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

        Approval routing makes a per-configuration decision through
        apple_container_has_host_access. Static-only consumers must not skip.
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
            image=image or cc.get("apple_container_image") or DEFAULT_IMAGE,
            cwd=cwd,
            timeout=timeout,
            cpu=int(cc.get("container_cpu", 0) or 0),
            memory=int(cc.get("container_memory", 0) or 0),
            persistent_filesystem=bool(cc.get("container_persistent", False)),
            task_id=task_id,
            volumes=cc.get("apple_container_volumes", []),
            extra_args=cc.get("apple_container_extra_args", []),
        )


def read_apple_container_config() -> Dict[str, Any]:
    """Read the active profile through the same scope-aware bridge as terminal."""
    import json
    from tools.terminal_tool_config import _parse_env_var, _tenv
    result = {"apple_container_image": _tenv("TERMINAL_APPLE_CONTAINER_IMAGE", DEFAULT_IMAGE)}
    for key in ("apple_container_volumes", "apple_container_extra_args"):
        value = _parse_env_var("TERMINAL_" + key.upper(), "[]", json.loads, "valid JSON")
        if not isinstance(value, list) or any(not isinstance(arg, str) for arg in value):
            raise ValueError(f"{key} must be a list of strings")
        if any(any(c in arg for c in "\x00\r\n") for arg in value):
            raise ValueError(f"{key} contains control characters")
        result[key] = value
    return result


def apple_container_has_host_access(config: Dict[str, Any]) -> bool:
    """Enable guards for user mounts and potential mount options in raw arguments.

    Apple uses Swift ArgumentParser, not Docker's pflag. Deliberately count all
    mount-like options, even named volumes or another option's value. This can
    require approval or block unattended deny-mode execution. Managed workspace
    persistence and automatic read-only skill/cache mounts retain existing policy.
    """
    if config.get("apple_container_volumes"):
        return True
    args = config.get("apple_container_extra_args") or []
    if not isinstance(args, list) or any(not isinstance(a, str) for a in args):
        return True
    for arg in args:
        if arg in ("--mount", "--volume", "--ssh") or arg.startswith(("--mount=", "--volume=", "--ssh=")):
            return True
        if arg.startswith("-") and not arg.startswith("--") and "v" in arg[1:]:
            return True
    return False

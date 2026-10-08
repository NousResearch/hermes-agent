"""Generic plugin policy bridge for file and process write boundaries.

Core calls this bridge at each write/execute entry point. The provider decides
which names to protect and how to wrap a process; those rules live in a plugin.
An absent provider leaves ordinary profiles unchanged. A provider that is
present but fails or returns an invalid result never permits an unguarded write.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_CONTRACT = "hermes.write-boundary/v1"
_MISSING = object()


def _require_loaded(tool_name: str) -> None:
    from tools.required_plugin_gate import required_tool_block
    blocked = required_tool_block(tool_name)
    if blocked:
        raise PermissionError(blocked)


def _home(home: str | None) -> str:
    if home is None:
        from hermes_constants import get_hermes_home
        home = str(get_hermes_home())
    return str(Path(home).expanduser().resolve())


def _decision(operation: str, home: str | None = None, **payload: Any) -> Any:
    from hermes_cli.plugins import has_hook, invoke_hook

    if not has_hook("write_boundary_provider"):
        return _MISSING
    results = invoke_hook("write_boundary_provider", operation=operation,
                          hermes_home=_home(home), **payload)
    if not results:
        raise PermissionError("Write-boundary provider did not decide for this profile")
    if len(results) != 1:
        raise PermissionError("Multiple write-boundary providers claimed this profile")
    result = results[0]
    if isinstance(result, dict) and result.get("action") == "block":
        raise PermissionError(str(result.get("message") or "Write boundary unavailable"))
    if not isinstance(result, dict) or result.get("contract") != _CONTRACT or "value" not in result:
        raise PermissionError("Write-boundary provider returned an invalid decision")
    return result["value"]


def protected_basenames(home: str | None = None) -> frozenset[str]:
    value = _decision("protected_basenames", home)
    if value is _MISSING:
        _require_loaded("execute_code")
        return frozenset()
    if not isinstance(value, list) or any(not isinstance(name, str) for name in value):
        raise PermissionError("Write-boundary provider returned invalid protected names")
    return frozenset(value)


def refuse_paths(paths, home: str | None = None) -> str | None:
    try:
        value = _decision("refuse_paths", home, paths=[str(path) for path in paths or ()])
    except PermissionError as exc:
        return str(exc)
    if value is _MISSING:
        _require_loaded("write_file")
        return None
    if value is not None and not isinstance(value, str):
        return "Write-boundary provider returned an invalid path decision"
    return value


def authority_write_block(tool_name: str, paths, home: str | None = None) -> str | None:
    try:
        from tools.required_plugin_gate import required_tool_block
        blocked = required_tool_block(tool_name)
    except Exception:
        blocked = f"Required plugin check failed; refusing {tool_name}."
    return blocked or refuse_paths(paths, home)


def refuse_command(command: str, home: str | None = None) -> str | None:
    try:
        value = _decision("refuse_command", home, command=command)
    except PermissionError as exc:
        return str(exc)
    if value is _MISSING:
        try:
            _require_loaded("terminal")
        except PermissionError as exc:
            return str(exc)
        return None
    if value is not None and not isinstance(value, str):
        return "Write-boundary provider returned an invalid command decision"
    return value


def guard_command(command: str, *, env_type: str, home: str | None = None) -> str | None:
    try:
        value = _decision("guard_command", home, command=command, env_type=env_type)
    except PermissionError:
        return None
    if value is _MISSING:
        try:
            _require_loaded("terminal")
        except PermissionError:
            return None
        return command
    return value if isinstance(value, str) else None


def wrap_code(code: str, home: str | None = None) -> str:
    value = _decision("wrap_code", home, code=code)
    if value is _MISSING:
        _require_loaded("execute_code")
        return code
    if not isinstance(value, str):
        raise PermissionError("Write-boundary provider returned invalid Python code")
    return value


def guard_process_argv(argv: list[str], *, home: str | None = None) -> list[str] | None:
    value = _decision("guard_process_argv", home, argv=list(argv))
    if value is _MISSING:
        _require_loaded("execute_code")
        return argv
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise PermissionError("Write-boundary provider returned invalid process argv")
    return value


def guard_remote_process_command(argv: list[str], *, home: str | None = None) -> str | None:
    value = _decision("guard_remote_process_command", home, argv=list(argv))
    if value is _MISSING:
        _require_loaded("execute_code")
        import shlex
        return shlex.join(argv)
    if value is not None and not isinstance(value, str):
        raise PermissionError("Write-boundary provider returned invalid remote command")
    return value


# Compatibility for deployments that still register the earlier Core boundary.
# New profiles use the plugin provider; both layers apply when both are present.
from tools import write_boundary_legacy as _legacy

register_protected_basenames = _legacy.register_protected_basenames
clear_protected_basenames = _legacy.clear_protected_basenames
register_official_writer = _legacy.register_official_writer
clear_official_writers = _legacy.clear_official_writers
official_exec_argv = _legacy.official_exec_argv
official_writer_command = _legacy.official_writer_command
command_write_targets = _legacy.command_write_targets
prelude = _legacy.prelude

_provider_protected_basenames = protected_basenames
_provider_refuse_paths = refuse_paths
_provider_refuse_command = refuse_command
_provider_guard_command = guard_command
_provider_wrap_code = wrap_code
_provider_guard_process_argv = guard_process_argv
_provider_guard_remote_process_command = guard_remote_process_command


def protected_basenames(home: str | None = None) -> frozenset[str]:
    return _provider_protected_basenames(home) | _legacy.protected_basenames(home)


def refuse_paths(paths, home: str | None = None) -> str | None:
    return _provider_refuse_paths(paths, home) or _legacy.refuse_paths(paths, home)


def refuse_command(command: str, home: str | None = None) -> str | None:
    return _provider_refuse_command(command, home) or _legacy.refuse_command(command, home)


def guard_command(command: str, *, env_type: str, home: str | None = None) -> str | None:
    guarded = _provider_guard_command(command, env_type=env_type, home=home)
    if guarded is None:
        return None
    return _legacy.guard_command(guarded, env_type=env_type, home=home)


def wrap_code(code: str, home: str | None = None) -> str:
    return _legacy.wrap_code(_provider_wrap_code(code, home), home)


def guard_process_argv(argv: list[str], *, home: str | None = None) -> list[str] | None:
    guarded = _provider_guard_process_argv(argv, home=home)
    if guarded is None:
        return None
    names = _legacy.protected_basenames(home)
    if not names:
        return guarded
    from tools.authority_os_guard import posix_spawn_argv
    return posix_spawn_argv(guarded, names)


def guard_remote_process_command(argv: list[str], *, home: str | None = None) -> str | None:
    guarded = _provider_guard_remote_process_command(argv, home=home)
    if guarded is None:
        return None
    names = _legacy.protected_basenames(home)
    if not names:
        return guarded
    from tools.authority_os_guard import linux_argv_command
    return linux_argv_command(["sh", "-c", guarded], names)

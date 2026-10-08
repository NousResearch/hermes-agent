"""Fail closed when a profile declares a required plugin that did not load.

Profiles without ``plugins.required`` are unchanged. A check that cannot be
completed blocks the tools named by that section instead of continuing.
"""

from __future__ import annotations

from pathlib import Path


_FALLBACK_TOOLS = ("delegate_task", "write_file", "patch", "terminal", "execute_code")
GUARDED_TOOLS = frozenset(_FALLBACK_TOOLS) | {"submit_delivery"}


def dispatcher_failure_block(tool_name: str) -> str | None:
    """Block a mutating tool when the pre-tool dispatcher itself raises."""
    if tool_name in GUARDED_TOOLS:
        return f"Required plugin check failed; refusing {tool_name}."
    return None


def _config_path() -> Path:
    from hermes_constants import get_hermes_home
    return Path(get_hermes_home()) / "config.yaml"


def _plugins_section() -> dict:
    path = _config_path()
    if not path.is_file():
        return {}
    import hermes_yaml as yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    plugins = data.get("plugins") if isinstance(data, dict) else None
    return plugins if isinstance(plugins, dict) else {}


def _loaded_names() -> set[str]:
    from hermes_cli.plugins import get_plugin_manager
    return set(get_plugin_manager().loaded_plugin_names())


def required_tool_block(tool_name: str) -> str | None:
    """Return a block message when ``tool_name`` needs a plugin that is not active."""
    path = _config_path()
    try:
        section = _plugins_section()
    except Exception:
        if tool_name in _FALLBACK_TOOLS and path.is_file() and "required:" in path.read_text(encoding="utf-8"):
            return f"Required plugin check failed; refusing {tool_name}."
        return None
    required = section.get("required") or {}
    if not isinstance(required, dict) or not required:
        return None
    try:
        loaded = _loaded_names()
        failed = False
    except Exception:
        loaded = set()
        failed = True
    for name, spec in required.items():
        tools = spec.get("tools") if isinstance(spec, dict) else list(_FALLBACK_TOOLS)
        if not isinstance(tools, (list, tuple)) or tool_name not in tools:
            continue
        if failed or name not in loaded:
            return f"Required plugin {name} is not loaded; refusing {tool_name}."
    return None

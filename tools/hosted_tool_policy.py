"""Tools a hosted scope refuses because they would run or read on this host.

In a hosted scope nothing tenant-supplied may execute on this host or touch its filesystem: the
terminal, code, background-process and file tools reach the agent's own sandbox
(``tools.environments.sandbox_scope``). The tool families below have no sandbox route yet. They
run here (a browser harness executing the model's Python, a desktop driver, local TTS engines),
write their result as a host file that is only deliverable by host path, or act for the profile
rather than the agent (skills, cron, MCP servers from the profile's config, child agents). Until each
gets a route through the agent's environment, a hosted scope refuses it before any setup, file or
process: :func:`hosted_refusal` is checked by ``ToolRegistry.dispatch``, and the paths that skip the
registry (the browser harness entry, checkpoints, inline skill shell, ``@`` references, workspace
probes, cron scripts, MCP stdio spawns, media delivery, transcription, delegation) call
:func:`refuse_on_host` themselves.

A local install (no scope, or a local one) is untouched: every function here answers "allowed".
"""

from __future__ import annotations

from typing import Optional

# toolset -> what the family would do on this host.
_REFUSED_TOOLSETS = {
    "browser-use": "browser_exec runs its Python on this host",
    "browser-cdp": "it drives a browser this host launched",
    "browser": "it fills the operator's vault into a browser this host launched",
    "computer_use": "it drives this host's desktop",
    "desktop_ui": "it drives this host's desktop app",
    "project": "it creates workspaces on this host",
    "image_gen": "its output is a host file",
    "video_gen": "its output is a host file",
    "tts": "it writes host audio files and runs local speech engines",
    "skills": "skills live in the profile's host directory and may run inline shell on this host",
    "cronjob": "cron jobs and their scripts run on this host for the profile",
    "delegation": "a child agent would run outside this agent's sandbox routing",
    "kanban": "the board dispatches workers on this host for the profile",
    "catalog": "it installs plugins and skills into this host's profile",
    "connections": "it can install and start local MCP servers on this host",
    "onboarding": "it changes this host profile's setup",
}
_MCP_TOOLSET_PREFIX = "mcp-"


class HostedToolRefused(PermissionError):
    """A hosted scope asked for something that would run or read on this host."""


def _message(what: str, why: str) -> str:
    return (f"{what} is not available to a hosted agent: {why}. Hosted tools run only in the "
            "agent's own sandbox (terminal, execute_code, background processes, file tools).")


def hosted_refusal(tool_name: str, toolset: str) -> Optional[str]:
    """Why a hosted scope may not dispatch *tool_name* (of *toolset*), or ``None``."""
    from tools.environments.sandbox_scope import hosted_scope
    if hosted_scope() is None:
        return None
    if toolset.startswith(_MCP_TOOLSET_PREFIX):
        return _message(tool_name, "MCP servers come from the profile's config and run for it, not for "
                                   "this agent")
    why = _REFUSED_TOOLSETS.get(toolset)
    return _message(tool_name, why) if why else None


def refuse_on_host(what: str, why: str) -> None:
    """Raise :class:`HostedToolRefused` in a hosted scope; a no-op otherwise."""
    from tools.environments.sandbox_scope import hosted_scope
    if hosted_scope() is not None:
        raise HostedToolRefused(_message(what, why))


def host_refusal(what: str, why: str) -> Optional[str]:
    """:func:`refuse_on_host` as a value: the refusal message in a hosted scope, else ``None``."""
    try:
        refuse_on_host(what, why)
    except HostedToolRefused as exc:
        return str(exc)
    return None


def refused_toolsets() -> frozenset[str]:
    """The toolsets :func:`hosted_refusal` refuses by name (MCP toolsets are refused by prefix)."""
    return frozenset(_REFUSED_TOOLSETS)

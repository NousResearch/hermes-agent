"""ToolRegistry V2 — Lifecycle hooks layer that coexists with ToolRegistry V1.

This module provides a thin wrapper around the existing ToolRegistry that adds
lifecycle event hooks for tool load/unload events. It does NOT replace the
existing registry; V1 remains the canonical registry for dispatch.

Usage:
    from tools.registry_v2 import tool_registry_v2
    tool_registry_v2.set_base_registry(registry)  # Pass existing registry

    # Register lifecycle hooks
    tool_registry_v2.register_lifecycle_hook('on_tool_loaded', my_callback)
    tool_registry_v2.register_lifecycle_hook('on_tool_unloaded', my_callback)

    # After importing a tool module (in model_tools._discover_tools):
    tool_registry_v2.on_tool_registered('my_tool', 'my_toolset')
"""

import logging
import threading
from typing import Callable, Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class ToolRegistryV2:
    """Lifecycle-aware wrapper for ToolRegistry.

    Does NOT own tool schemas or handlers — those live in the base registry.
    This class only adds lifecycle hook support for observability and
    dynamic tool management (MCP discovery, hot-reload, etc.).

    Thread-safe: all public methods use the instance lock.
    Fire-and-forget: hook exceptions are logged but not raised.
    """

    def __init__(self):
        self._base_registry: Optional[Any] = None
        self._lifecycle_hooks: Dict[str, List[Callable]] = {
            "on_tool_loaded": [],
            "on_tool_unloaded": [],
        }
        self._lock = threading.Lock()

    def set_base_registry(self, base_registry) -> None:
        """Set the reference to the existing ToolRegistry instance.

        This registry is NOT copied or modified — V2 simply holds a reference
        for use by hook callbacks that need to query the base registry.
        """
        with self._lock:
            self._base_registry = base_registry

    @property
    def base_registry(self):
        """Return the base registry reference (read-only)."""
        with self._lock:
            return self._base_registry

    def register_lifecycle_hook(self, event: str, callback: Callable) -> None:
        """Register a lifecycle hook callback.

        Args:
            event: One of 'on_tool_loaded' or 'on_tool_unloaded'.
            callback: Callable that receives (tool_name, **kwargs).
                      For 'on_tool_loaded': kwargs includes 'toolset'.
                      For 'on_tool_unloaded': kwargs is empty.

        Raises:
            ValueError: If event is not a valid lifecycle event name.
        """
        with self._lock:
            if event not in self._lifecycle_hooks:
                raise ValueError(
                    f"Unknown lifecycle event: {event}. "
                    f"Valid events: {list(self._lifecycle_hooks.keys())}"
                )
            self._lifecycle_hooks.setdefault(event, []).append(callback)

    def unregister_lifecycle_hook(self, event: str, callback: Callable) -> bool:
        """Remove a lifecycle hook callback.

        Returns True if the callback was found and removed, False otherwise.
        """
        with self._lock:
            hooks = self._lifecycle_hooks.get(event, [])
            if callback in hooks:
                hooks.remove(callback)
                return True
            return False

    def emit_lifecycle(self, event: str, tool_name: str, **kwargs) -> None:
        """Emit a lifecycle event to all registered hooks.

        This is fire-and-forget: exceptions raised by hooks are logged
        as warnings but do not propagate.

        Args:
            event: Lifecycle event name ('on_tool_loaded' or 'on_tool_unloaded').
            tool_name: Name of the tool being loaded/unloaded.
            **kwargs: Additional context passed to hook callbacks.
        """
        with self._lock:
            hooks = list(self._lifecycle_hooks.get(event, []))

        for hook in hooks:
            try:
                hook(tool_name, **kwargs)
            except Exception as e:
                logger.warning(
                    "Lifecycle hook '%s' for tool '%s' raised: %s",
                    event,
                    tool_name,
                    e,
                )

    def on_tool_registered(self, tool_name: str, toolset: str = "") -> None:
        """Fire the on_tool_loaded lifecycle hook.

        Called by model_tools._discover_tools() after successfully importing
        a tool module. MCP discovery also calls this when registering new
        tools from external servers.

        Args:
            tool_name: Name of the tool that was registered.
            toolset: Name of the toolset the tool belongs to.
        """
        self.emit_lifecycle("on_tool_loaded", tool_name, toolset=toolset)

    def on_tool_deregistered(self, tool_name: str) -> None:
        """Fire the on_tool_unloaded lifecycle hook.

        Called when a tool is removed from the registry, such as during
        MCP notifications/tools/list_changed events.

        Args:
            tool_name: Name of the tool that was deregistered.
        """
        self.emit_lifecycle("on_tool_unloaded", tool_name)


# Module-level singleton for use across the codebase
tool_registry_v2 = ToolRegistryV2()

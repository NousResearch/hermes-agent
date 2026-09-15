"""Runtime Capability Registry for Hermes Workstation.

Discovers and verifies available Python libraries, Node runtimes, and system tools
BEFORE execution starts, preventing mid-run failures from missing dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib.util
import logging
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RuntimeCapabilityReport:
    python_version: str
    python_executable: str
    os_name: str
    installed_modules: Dict[str, bool] = field(default_factory=dict)
    system_tools: Dict[str, Optional[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RuntimeCapabilityRegistry:
    """Discovers and inspects runtime capabilities and installed tools."""

    COMMON_MODULES = (
        "openpyxl",
        "pandas",
        "numpy",
        "playwright",
        "requests",
        "urllib3",
        "aiohttp",
        "websockets",
        "sqlite3",
        "psutil",
    )

    COMMON_SYSTEM_TOOLS = (
        "git",
        "node",
        "npm",
        "python",
        "curl",
    )

    @classmethod
    def has_python_module(cls, module_name: str) -> bool:
        """Check whether a python module is importable without importing heavy dependencies."""
        try:
            return importlib.util.find_spec(module_name) is not None
        except Exception:
            return False

    @classmethod
    def has_system_tool(cls, tool_name: str) -> bool:
        return shutil.which(tool_name) is not None

    @classmethod
    def get_tool_path(cls, tool_name: str) -> Optional[str]:
        return shutil.which(tool_name)

    @classmethod
    def inspect_environment(
        cls,
        *,
        check_modules: Optional[List[str]] = None,
        check_tools: Optional[List[str]] = None,
    ) -> RuntimeCapabilityReport:
        modules = check_modules or list(cls.COMMON_MODULES)
        tools = check_tools or list(cls.COMMON_SYSTEM_TOOLS)

        module_results = {mod: cls.has_python_module(mod) for mod in modules}
        tool_results = {tool: cls.get_tool_path(tool) for tool in tools}

        return RuntimeCapabilityReport(
            python_version=sys.version.split()[0],
            python_executable=sys.executable,
            os_name=os.name,
            installed_modules=module_results,
            system_tools=tool_results,
        )

    @classmethod
    def audit_environment(
        cls,
        *,
        check_modules: Optional[List[str]] = None,
        check_tools: Optional[List[str]] = None,
    ) -> RuntimeCapabilityReport:
        """Alias for inspect_environment for compatibility with pre-flight audit callers."""
        return cls.inspect_environment(check_modules=check_modules, check_tools=check_tools)

    @classmethod
    def audit_environment(
        cls,
        *,
        check_modules: Optional[List[str]] = None,
        check_tools: Optional[List[str]] = None,
    ) -> RuntimeCapabilityReport:
        """Alias for inspect_environment conforming to operational guidelines."""
        return cls.inspect_environment(
            check_modules=check_modules,
            check_tools=check_tools,
        )

    @classmethod
    def assert_capability(cls, capability: str) -> None:
        """Raise RuntimeError if a required capability is missing."""
        if capability.startswith("python:"):
            mod = capability[len("python:"):]
            if not cls.has_python_module(mod):
                raise RuntimeError(
                    f"Required Python capability '{mod}' is missing. Install with 'pip install {mod}'."
                )
        elif capability.startswith("bin:"):
            tool = capability[len("bin:"):]
            if not cls.has_system_tool(tool):
                raise RuntimeError(
                    f"Required system tool '{tool}' is not found in PATH."
                )

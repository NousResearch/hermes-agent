"""Hermes execution environment backends.

Each backend provides the same interface (BaseEnvironment ABC) for running
shell commands in a specific execution context: local, Docker, SSH,
Singularity, Modal, Daytona, or Vercel Sandbox. (Modal additionally has
direct and Nous-managed modes, selected via terminal.modal_mode.)

The migration facade in terminal_tool.py selects the legacy or experimental
runtime. Third-party plugins import only the backend definitions re-exported by
this package; host lifecycle services remain in their dedicated modules.
"""

from tools.environments.base import BaseEnvironment, _ThreadedProcessHandle
from tools.environments.definitions import (
    AvailabilityCheck,
    BackendCapabilities,
    BackendDefinition,
    BackendFactory,
    BackendFactoryRequest,
    ExecutionLocation,
    FilesystemSemantics,
)

from tools.environments.forward_env import (
    collect_forwarded_env_values,
    normalize_forward_env_names,
)

ThreadedProcessHandle = _ThreadedProcessHandle

__all__ = [
    "AvailabilityCheck",
    "BackendCapabilities",
    "BackendDefinition",
    "BackendFactory",
    "BackendFactoryRequest",
    "BaseEnvironment",
    "ExecutionLocation",
    "FilesystemSemantics",
    "ThreadedProcessHandle",
    "collect_forwarded_env_values",
    "normalize_forward_env_names",
]

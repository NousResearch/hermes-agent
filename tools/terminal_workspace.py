"""Validated workspace authority for dispatcher-owned kanban workers."""
import os
from pathlib import Path


def kanban_workspace(env=None):
    """Return the worker's host workspace, or None outside a kanban worker.

    A present but unusable assignment must not fall back to a profile directory.
    """
    env = os.environ if env is None else env
    if not env.get('HERMES_KANBAN_TASK'):
        return None
    value = env.get('HERMES_KANBAN_WORKSPACE')
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute() or not path.is_dir():
        raise ValueError('Kanban workspace must be an existing absolute directory')
    return str(path.resolve())


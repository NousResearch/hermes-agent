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


def apple_workspace_path(path, task_id):
    """Translate a path under an opted-in Apple host workspace to its guest path."""
    from tools.terminal_scope import terminal_env
    if terminal_env("TERMINAL_ENV", "local") != "apple_container":
        return path
    from tools.terminal_tool import _get_env_config, _resolve_task_host_cwd
    config = _get_env_config()
    if config['env_type'] != 'apple_container' or not config.get('apple_container_mount_cwd_to_workspace'):
        return path
    host = _resolve_task_host_cwd(config, task_id)
    if not host:
        return path
    from tools.environments.apple_container import _parse_user_mount
    for value in config.get('apple_container_volumes', []):
        source, target, _ = _parse_user_mount(value)
        if target == '/workspace':
            host = source
            break
    from pathlib import PurePosixPath
    try:
        relative = PurePosixPath(str(path)).relative_to(str(Path(host).resolve()))
    except ValueError:
        return path
    return PurePosixPath('/workspace') / relative

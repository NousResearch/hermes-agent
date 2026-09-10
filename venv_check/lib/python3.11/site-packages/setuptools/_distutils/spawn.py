"""distutils.spawn

Provides the 'spawn()' function, a front-end to various platform-
specific functions for launching another program in a sub-process.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import warnings
from collections.abc import Mapping, MutableSequence
from typing import TYPE_CHECKING, TypeVar, overload

from ._log import log
from .debug import DEBUG
from .errors import DistutilsExecError

if TYPE_CHECKING:
    from subprocess import _ENV


_MappingT = TypeVar("_MappingT", bound=Mapping)


def _debug(cmd):
    """
    Render a subprocess command differently depending on DEBUG.
    """
    return cmd if DEBUG else cmd[0]


def _inject_macos_ver(env: _MappingT | None) -> _MappingT | dict[str, str | int] | None:
    if platform.system() != 'Darwin':
        return env

    from .util import MACOSX_VERSION_VAR, get_macosx_target_ver

    target_ver = get_macosx_target_ver()
    update = {MACOSX_VERSION_VAR: target_ver} if target_ver else {}
    return {**_resolve(env), **update}


@overload
def _resolve(env: None) -> os._Environ[str]: ...
@overload
def _resolve(env: _MappingT) -> _MappingT: ...
def _resolve(env: _MappingT | None) -> _MappingT | os._Environ[str]:
    return os.environ if env is None else env


def spawn(
    cmd: MutableSequence[bytes | str | os.PathLike[str]],
    search_path: bool = True,
    verbose: bool = False,
    dry_run: bool = False,
    env: _ENV | None = None,
) -> None:
    """Run another program, specified as a command list 'cmd', in a new process.

    'cmd' is just the argument list for the new process, ie.
    cmd[0] is the program to run and cmd[1:] are the rest of its arguments.
    There is no way to run a program with a name different from that of its
    executable.

    If 'search_path' is true (the default), the system's executable
    search path will be used to find the program; otherwise, cmd[0]
    must be the exact path to the executable.  If 'dry_run' is true,
    the command will not actually be run.

    Raise DistutilsExecError if running the program fails in any way; just
    return on success.
    """
    log.info(subprocess.list2cmdline(cmd))
    if dry_run:
        return

    if search_path:
        executable = shutil.which(cmd[0])
        if executable is not None:
            cmd[0] = executable

    try:
        subprocess.check_call(cmd, env=_inject_macos_ver(env))
    except OSError as exc:
        raise DistutilsExecError(
            f"command {_debug(cmd)!r} failed: {exc.args[-1]}"
        ) from exc
    except subprocess.CalledProcessError as err:
        raise DistutilsExecError(
            f"command {_debug(cmd)!r} failed with exit code {err.returncode}"
        ) from err


def find_executable(executable: str, path: str | None = None) -> str | None:
    """Tries to find 'executable' in the directories listed in 'path'.

    A string listing directories separated by 'os.pathsep'; defaults to
    os.environ['PATH'].  Returns the complete filename or None if not found.
    """
    warnings.warn(
        'Use shutil.which instead of find_executable', DeprecationWarning, stacklevel=2
    )
    _, ext = os.path.splitext(executable)
    if (sys.platform == 'win32') and (ext != '.exe'):
        executable = executable + '.exe'

    if os.path.isfile(executable):
        return executable

    if path is None:
        path = os.environ.get('PATH', None)
        # bpo-35755: Don't fall through if PATH is the empty string
        if path is None:
            try:
                path = os.confstr("CS_PATH")
            except (AttributeError, ValueError):
                # os.confstr() or CS_PATH is not available
                path = os.defpath

    # PATH='' doesn't match, whereas PATH=':' looks in the current directory
    if not path:
        return None

    paths = path.split(os.pathsep)
    for p in paths:
        f = os.path.join(p, executable)
        if os.path.isfile(f):
            # the file exists, we have a shot at spawn working
            return f
    return None

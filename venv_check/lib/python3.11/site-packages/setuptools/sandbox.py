from __future__ import annotations

import builtins
import contextlib
import functools
import itertools
import operator
import os
import pickle
import re
import sys
import tempfile
import textwrap
from types import TracebackType
from typing import TYPE_CHECKING, Any, ClassVar

import pkg_resources
from pkg_resources import working_set

from distutils.errors import DistutilsError

if TYPE_CHECKING:
    import os as _os
elif sys.platform.startswith('java'):
    import org.python.modules.posix.PosixModule as _os  # pyright: ignore[reportMissingImports]
else:
    _os = sys.modules[os.name]
_open = open


if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = [
    "AbstractSandbox",
    "DirectorySandbox",
    "SandboxViolation",
    "run_setup",
]


def _execfile(filename, globals, locals=None):
    """
    Python 3 implementation of execfile.
    """
    mode = 'rb'
    with open(filename, mode) as stream:
        script = stream.read()
    if locals is None:
        locals = globals
    code = compile(script, filename, 'exec')
    exec(code, globals, locals)


@contextlib.contextmanager
def save_argv(repl=None):
    saved = sys.argv[:]
    if repl is not None:
        sys.argv[:] = repl
    try:
        yield saved
    finally:
        sys.argv[:] = saved


@contextlib.contextmanager
def save_path():
    saved = sys.path[:]
    try:
        yield saved
    finally:
        sys.path[:] = saved


@contextlib.contextmanager
def override_temp(replacement):
    """
    Monkey-patch tempfile.tempdir with replacement, ensuring it exists
    """
    os.makedirs(replacement, exist_ok=True)

    saved = tempfile.tempdir

    tempfile.tempdir = replacement

    try:
        yield
    finally:
        tempfile.tempdir = saved


@contextlib.contextmanager
def pushd(target):
    saved = os.getcwd()
    os.chdir(target)
    try:
        yield saved
    finally:
        os.chdir(saved)


class UnpickleableException(Exception):
    """
    An exception representing another Exception that could not be pickled.
    """

    @staticmethod
    def dump(type, exc):
        """
        Always return a dumped (pickled) type and exc. If exc can't be pickled,
        wrap it in UnpickleableException first.
        """
        try:
            return pickle.dumps(type), pickle.dumps(exc)
        except Exception:
            # get UnpickleableException inside the sandbox
            from setuptools.sandbox import UnpickleableException as cls

            return cls.dump(cls, cls(repr(exc)))


class ExceptionSaver:
    """
    A Context Manager that will save an exception, serialize, and restore it
    later.
    """

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if not exc:
            return False

        # dump the exception
        self._saved = UnpickleableException.dump(type, exc)
        self._tb = tb

        # suppress the exception
        return True

    def resume(self):
        "restore and re-raise any exception"

        if '_saved' not in vars(self):
            return

        _type, exc = map(pickle.loads, self._saved)
        raise exc.with_traceback(self._tb)


@contextlib.contextmanager
def save_modules():
    """
    Context in which imported modules are saved.

    Translates exceptions internal to the context into the equivalent exception
    outside the context.
    """
    saved = sys.modules.copy()
    with ExceptionSaver() as saved_exc:
        yield saved

    sys.modules.update(saved)
    # remove any modules imported since
    del_modules = (
        mod_name
        for mod_name in sys.modules
        if mod_name not in saved
        # exclude any encodings modules. See #285
        and not mod_name.startswith('encodings.')
    )
    _clear_modules(del_modules)

    saved_exc.resume()


def _clear_modules(module_names):
    for mod_name in list(module_names):
        del sys.modules[mod_name]


@contextlib.contextmanager
def save_pkg_resources_state():
    saved = pkg_resources.__getstate__()
    try:
        yield saved
    finally:
        pkg_resources.__setstate__(saved)


@contextlib.contextmanager
def setup_context(setup_dir):
    temp_dir = os.path.join(setup_dir, 'temp')
    with save_pkg_resources_state():
        with save_modules():
            with save_path():
                hide_setuptools()
                with save_argv():
                    with override_temp(temp_dir):
                        with pushd(setup_dir):
                            # ensure setuptools commands are available
                            __import__('setuptools')
                            yield


_MODULES_TO_HIDE = {
    'setuptools',
    'distutils',
    'pkg_resources',
    'Cython',
    '_distutils_hack',
}


def _needs_hiding(mod_name):
    """
    >>> _needs_hiding('setuptools')
    True
    >>> _needs_hiding('pkg_resources')
    True
    >>> _needs_hiding('setuptools_plugin')
    False
    >>> _needs_hiding('setuptools.__init__')
    True
    >>> _needs_hiding('distutils')
    True
    >>> _needs_hiding('os')
    False
    >>> _needs_hiding('Cython')
    True
    """
    base_module = mod_name.split('.', 1)[0]
    return base_module in _MODULES_TO_HIDE


def hide_setuptools():
    """
    Remove references to setuptools' modules from sys.modules to allow the
    invocation to import the most appropriate setuptools. This technique is
    necessary to avoid issues such as #315 where setuptools upgrading itself
    would fail to find a function declared in the metadata.
    """
    _distutils_hack = sys.modules.get('_distutils_hack', None)
    if _distutils_hack is not None:
        _distutils_hack._remove_shim()

    modules = filter(_needs_hiding, sys.modules)
    _clear_modules(modules)


def run_setup(setup_script, args):
    """Run a distutils setup script, sandboxed in its directory"""
    setup_dir = os.path.abspath(os.path.dirname(setup_script))
    with setup_context(setup_dir):
        try:
            sys.argv[:] = [setup_script] + list(args)
            sys.path.insert(0, setup_dir)
            # reset to include setup dir, w/clean callback list
            working_set.__init__()
            working_set.callbacks.append(lambda dist: dist.activate())

            with DirectorySandbox(setup_dir):
                ns = dict(__file__=setup_script, __name__='__main__')
                _execfile(setup_script, ns)
        except SystemExit as v:
            if v.args and v.args[0]:
                raise
            # Normal exit, just return


class AbstractSandbox:
    """Wrap 'os' module and 'open()' builtin for virtualizing setup scripts"""

    _active = False

    def __init__(self) -> None:
        self._attrs = [
            name
            for name in dir(_os)
            if not name.startswith('_') and hasattr(self, name)
        ]

    def _copy(self, source):
        for name in self._attrs:
            setattr(os, name, getattr(source, name))

    def __enter__(self) -> None:
        self._copy(self)
        builtins.open = self._open
        self._active = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ):
        self._active = False
        builtins.open = _open
        self._copy(_os)

    def run(self, func):
        """Run 'func' under os sandboxing"""
        with self:
            return func()

    def _mk_dual_path_wrapper(name: str):  # type: ignore[misc] # https://github.com/pypa/setuptools/pull/4099
        original = getattr(_os, name)

        def wrap(self, src, dst, *args, **kw):
            if self._active:
                src, dst = self._remap_pair(name, src, dst, *args, **kw)
            return original(src, dst, *args, **kw)

        return wrap

    for __name in ["rename", "link", "symlink"]:
        if hasattr(_os, __name):
            locals()[__name] = _mk_dual_path_wrapper(__name)

    def _mk_single_path_wrapper(name: str, original=None):  # type: ignore[misc] # https://github.com/pypa/setuptools/pull/4099
        original = original or getattr(_os, name)

        def wrap(self, path, *args, **kw):
            if self._active:
                path = self._remap_input(name, path, *args, **kw)
            return original(path, *args, **kw)

        return wrap

    _open = _mk_single_path_wrapper('open', _open)
    for __name in [
        "stat",
        "listdir",
        "chdir",
        "open",
        "chmod",
        "chown",
        "mkdir",
        "remove",
        "unlink",
        "rmdir",
        "utime",
        "lchown",
        "chroot",
        "lstat",
        "startfile",
        "mkfifo",
        "mknod",
        "pathconf",
        "access",
    ]:
        if hasattr(_os, __name):
            locals()[__name] = _mk_single_path_wrapper(__name)

    def _mk_single_with_return(name: str):  # type: ignore[misc] # https://github.com/pypa/setuptools/pull/4099
        original = getattr(_os, name)

        def wrap(self, path, *args, **kw):
            if self._active:
                path = self._remap_input(name, path, *args, **kw)
                return self._remap_output(name, original(path, *args, **kw))
            return original(path, *args, **kw)

        return wrap

    for __name in ['readlink', 'tempnam']:
        if hasattr(_os, __name):
            locals()[__name] = _mk_single_with_return(__name)

    def _mk_query(name: str):  # type: ignore[misc] # https://github.com/pypa/setuptools/pull/4099
        original = getattr(_os, name)

        def wrap(self, *args, **kw):
            retval = original(*args, **kw)
            if self._active:
                return self._remap_output(name, retval)
            return retval

        return wrap

    for __name in ['getcwd', 'tmpnam']:
        if hasattr(_os, __name):
            locals()[__name] = _mk_query(__name)

    def _validate_path(self, path):
        """Called to remap or validate any path, whether input or output"""
        return path

    def _remap_input(self, operation, path, *args, **kw):
        """Called for path inputs"""
        return self._validate_path(path)

    def _remap_output(self, operation, path):
        """Called for path outputs"""
        return self._validate_path(path)

    def _remap_pair(self, operation, src, dst, *args, **kw):
        """Called for path pairs like rename, link, and symlink operations"""
        return (
            self._remap_input(operation + '-from', src, *args, **kw),
            self._remap_input(operation + '-to', dst, *args, **kw),
        )

    if TYPE_CHECKING:
        # This is a catch-all for all the dynamically created attributes.
        # This isn't public API anyway
        def __getattribute__(self, name: str) -> Any: ...


if hasattr(os, 'devnull'):
    _EXCEPTIONS = [os.devnull]
else:
    _EXCEPTIONS = []


class DirectorySandbox(AbstractSandbox):
    """Restrict operations to a single subdirectory - pseudo-chroot"""

    write_ops: ClassVar[dict[str, None]] = dict.fromkeys([
        "open",
        "chmod",
        "chown",
        "mkdir",
        "remove",
        "unlink",
        "rmdir",
        "utime",
        "lchown",
        "chroot",
        "mkfifo",
        "mknod",
        "tempnam",
    ])

    _exception_patterns: list[str | re.Pattern] = []
    "exempt writing to paths that match the pattern"

    def __init__(self, sandbox, exceptions=_EXCEPTIONS) -> None:
        self._sandbox = os.path.normcase(os.path.realpath(sandbox))
        self._prefix = os.path.join(self._sandbox, '')
        self._exceptions = [
            os.path.normcase(os.path.realpath(path)) for path in exceptions
        ]
        AbstractSandbox.__init__(self)

    def _violation(self, operation, *args, **kw):
        from setuptools.sandbox import SandboxViolation

        raise SandboxViolation(operation, args, kw)

    def _open(self, path, mode='r', *args, **kw):
        if mode not in ('r', 'rt', 'rb', 'rU', 'U') and not self._ok(path):
            self._violation("open", path, mode, *args, **kw)
        return _open(path, mode, *args, **kw)

    def tmpnam(self) -> None:
        self._violation("tmpnam")

    def _ok(self, path):
        active = self._active
        try:
            self._active = False
            realpath = os.path.normcase(os.path.realpath(path))
            return (
                self._exempted(realpath)
                or realpath == self._sandbox
                or realpath.startswith(self._prefix)
            )
        finally:
            self._active = active

    def _exempted(self, filepath):
        start_matches = (
            filepath.startswith(exception) for exception in self._exceptions
        )
        pattern_matches = (
            re.match(pattern, filepath) for pattern in self._exception_patterns
        )
        candidates = itertools.chain(start_matches, pattern_matches)
        return any(candidates)

    def _remap_input(self, operation, path, *args, **kw):
        """Called for path inputs"""
        if operation in self.write_ops and not self._ok(path):
            self._violation(operation, os.path.realpath(path), *args, **kw)
        return path

    def _remap_pair(self, operation, src, dst, *args, **kw):
        """Called for path pairs like rename, link, and symlink operations"""
        if not self._ok(src) or not self._ok(dst):
            self._violation(operation, src, dst, *args, **kw)
        return (src, dst)

    def open(self, file, flags, mode: int = 0o777, *args, **kw) -> int:
        """Called for low-level os.open()"""
        if flags & WRITE_FLAGS and not self._ok(file):
            self._violation("os.open", file, flags, mode, *args, **kw)
        return _os.open(file, flags, mode, *args, **kw)


WRITE_FLAGS = functools.reduce(
    operator.or_,
    [
        getattr(_os, a, 0)
        for a in "O_WRONLY O_RDWR O_APPEND O_CREAT O_TRUNC O_TEMPORARY".split()
    ],
)


class SandboxViolation(DistutilsError):
    """A setup script attempted to modify the filesystem outside the sandbox"""

    tmpl = textwrap.dedent(
        """
        SandboxViolation: {cmd}{args!r} {kwargs}

        The package setup script has attempted to modify files on your system
        that are not within the EasyInstall build area, and has been aborted.

        This package cannot be safely installed by EasyInstall, and may not
        support alternate installation locations even if you run its setup
        script by hand.  Please inform the package's author and the EasyInstall
        maintainers to find out if a fix or workaround is available.
        """
    ).lstrip()

    def __str__(self) -> str:
        cmd, args, kwargs = self.args
        return self.tmpl.format(**locals())

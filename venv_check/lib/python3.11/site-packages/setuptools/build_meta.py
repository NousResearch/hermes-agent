"""A PEP 517 interface to setuptools

Previously, when a user or a command line tool (let's call it a "frontend")
needed to make a request of setuptools to take a certain action, for
example, generating a list of installation requirements, the frontend
would call "setup.py egg_info" or "setup.py bdist_wheel" on the command line.

PEP 517 defines a different method of interfacing with setuptools. Rather
than calling "setup.py" directly, the frontend should:

  1. Set the current directory to the directory with a setup.py file
  2. Import this module into a safe python interpreter (one in which
     setuptools can potentially set global variables or crash hard).
  3. Call one of the functions defined in PEP 517.

What each function does is defined in PEP 517. However, here is a "casual"
definition of the functions (this definition should not be relied on for
bug reports or API stability):

  - `build_wheel`: build a wheel in the folder and return the basename
  - `get_requires_for_build_wheel`: get the `setup_requires` to build
  - `prepare_metadata_for_build_wheel`: get the `install_requires`
  - `build_sdist`: build an sdist in the folder and return the basename
  - `get_requires_for_build_sdist`: get the `setup_requires` to build

Again, this is not a formal definition! Just a "taste" of the module.
"""

from __future__ import annotations

import contextlib
import io
import os
import shlex
import shutil
import sys
import tempfile
import tokenize
import warnings
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Union

import setuptools

from . import errors
from ._path import StrPath, same_path
from ._reqs import parse_strings
from .warnings import SetuptoolsDeprecationWarning

import distutils
from distutils.util import strtobool

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

__all__ = [
    'get_requires_for_build_sdist',
    'get_requires_for_build_wheel',
    'prepare_metadata_for_build_wheel',
    'build_wheel',
    'build_sdist',
    'get_requires_for_build_editable',
    'prepare_metadata_for_build_editable',
    'build_editable',
    '__legacy__',
    'SetupRequirementsError',
]


class SetupRequirementsError(BaseException):
    def __init__(self, specifiers) -> None:
        self.specifiers = specifiers


class Distribution(setuptools.dist.Distribution):
    def fetch_build_eggs(self, specifiers):
        specifier_list = list(parse_strings(specifiers))

        raise SetupRequirementsError(specifier_list)

    @classmethod
    @contextlib.contextmanager
    def patch(cls):
        """
        Replace
        distutils.dist.Distribution with this class
        for the duration of this context.
        """
        orig = distutils.core.Distribution
        distutils.core.Distribution = cls  # type: ignore[misc] # monkeypatching
        try:
            yield
        finally:
            distutils.core.Distribution = orig  # type: ignore[misc] # monkeypatching


@contextlib.contextmanager
def no_install_setup_requires():
    """Temporarily disable installing setup_requires

    Under PEP 517, the backend reports build dependencies to the frontend,
    and the frontend is responsible for ensuring they're installed.
    So setuptools (acting as a backend) should not try to install them.
    """
    orig = setuptools._install_setup_requires
    setuptools._install_setup_requires = lambda attrs: None
    try:
        yield
    finally:
        setuptools._install_setup_requires = orig


def _get_immediate_subdirectories(a_dir):
    return [
        name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))
    ]


def _file_with_extension(directory: StrPath, extension: str | tuple[str, ...]):
    matching = (f for f in os.listdir(directory) if f.endswith(extension))
    try:
        (file,) = matching
    except ValueError:
        raise ValueError(
            'No distribution was found. Ensure that `setup.py` '
            'is not empty and that it calls `setup()`.'
        ) from None
    return file


def _open_setup_script(setup_script):
    if not os.path.exists(setup_script):
        # Supply a default setup.py
        return io.StringIO("from setuptools import setup; setup()")

    return tokenize.open(setup_script)


@contextlib.contextmanager
def suppress_known_deprecation():
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'setup.py install is deprecated')
        yield


_ConfigSettings: TypeAlias = Union[Mapping[str, Union[str, list[str], None]], None]
"""
Currently the user can run::

    pip install -e . --config-settings key=value
    python -m build -C--key=value -C key=value

- pip will pass both key and value as strings and overwriting repeated keys
  (pypa/pip#11059).
- build will accumulate values associated with repeated keys in a list.
  It will also accept keys with no associated value.
  This means that an option passed by build can be ``str | list[str] | None``.
- PEP 517 specifies that ``config_settings`` is an optional dict.
"""


class _ConfigSettingsTranslator:
    """Translate ``config_settings`` into distutils-style command arguments.
    Only a limited number of options is currently supported.
    """

    # See pypa/setuptools#1928 pypa/setuptools#2491

    def _get_config(self, key: str, config_settings: _ConfigSettings) -> list[str]:
        """
        Get the value of a specific key in ``config_settings`` as a list of strings.

        >>> fn = _ConfigSettingsTranslator()._get_config
        >>> fn("--global-option", None)
        []
        >>> fn("--global-option", {})
        []
        >>> fn("--global-option", {'--global-option': 'foo'})
        ['foo']
        >>> fn("--global-option", {'--global-option': ['foo']})
        ['foo']
        >>> fn("--global-option", {'--global-option': 'foo'})
        ['foo']
        >>> fn("--global-option", {'--global-option': 'foo bar'})
        ['foo', 'bar']
        """
        cfg = config_settings or {}
        opts = cfg.get(key) or []
        return shlex.split(opts) if isinstance(opts, str) else opts

    def _global_args(self, config_settings: _ConfigSettings) -> Iterator[str]:
        """
        Let the user specify ``verbose`` or ``quiet`` + escape hatch via
        ``--global-option``.
        Note: ``-v``, ``-vv``, ``-vvv`` have similar effects in setuptools,
        so we just have to cover the basic scenario ``-v``.

        >>> fn = _ConfigSettingsTranslator()._global_args
        >>> list(fn(None))
        []
        >>> list(fn({"verbose": "False"}))
        ['-q']
        >>> list(fn({"verbose": "1"}))
        ['-v']
        >>> list(fn({"--verbose": None}))
        ['-v']
        >>> list(fn({"verbose": "true", "--global-option": "-q --no-user-cfg"}))
        ['-v', '-q', '--no-user-cfg']
        >>> list(fn({"--quiet": None}))
        ['-q']
        """
        cfg = config_settings or {}
        falsey = {"false", "no", "0", "off"}
        if "verbose" in cfg or "--verbose" in cfg:
            level = str(cfg.get("verbose") or cfg.get("--verbose") or "1")
            yield ("-q" if level.lower() in falsey else "-v")
        if "quiet" in cfg or "--quiet" in cfg:
            level = str(cfg.get("quiet") or cfg.get("--quiet") or "1")
            yield ("-v" if level.lower() in falsey else "-q")

        yield from self._get_config("--global-option", config_settings)

    def __dist_info_args(self, config_settings: _ConfigSettings) -> Iterator[str]:
        """
        The ``dist_info`` command accepts ``tag-date`` and ``tag-build``.

        .. warning::
           We cannot use this yet as it requires the ``sdist`` and ``bdist_wheel``
           commands run in ``build_sdist`` and ``build_wheel`` to reuse the egg-info
           directory created in ``prepare_metadata_for_build_wheel``.

        >>> fn = _ConfigSettingsTranslator()._ConfigSettingsTranslator__dist_info_args
        >>> list(fn(None))
        []
        >>> list(fn({"tag-date": "False"}))
        ['--no-date']
        >>> list(fn({"tag-date": None}))
        ['--no-date']
        >>> list(fn({"tag-date": "true", "tag-build": ".a"}))
        ['--tag-date', '--tag-build', '.a']
        """
        cfg = config_settings or {}
        if "tag-date" in cfg:
            val = strtobool(str(cfg["tag-date"] or "false"))
            yield ("--tag-date" if val else "--no-date")
        if "tag-build" in cfg:
            yield from ["--tag-build", str(cfg["tag-build"])]

    def _editable_args(self, config_settings: _ConfigSettings) -> Iterator[str]:
        """
        The ``editable_wheel`` command accepts ``editable-mode=strict``.

        >>> fn = _ConfigSettingsTranslator()._editable_args
        >>> list(fn(None))
        []
        >>> list(fn({"editable-mode": "strict"}))
        ['--mode', 'strict']
        """
        cfg = config_settings or {}
        mode = cfg.get("editable-mode") or cfg.get("editable_mode")
        if not mode:
            return
        yield from ["--mode", str(mode)]

    def _arbitrary_args(self, config_settings: _ConfigSettings) -> Iterator[str]:
        """
        Users may expect to pass arbitrary lists of arguments to a command
        via "--global-option" (example provided in PEP 517 of a "escape hatch").

        >>> fn = _ConfigSettingsTranslator()._arbitrary_args
        >>> list(fn(None))
        []
        >>> list(fn({}))
        []
        >>> list(fn({'--build-option': 'foo'}))
        ['foo']
        >>> list(fn({'--build-option': ['foo']}))
        ['foo']
        >>> list(fn({'--build-option': 'foo'}))
        ['foo']
        >>> list(fn({'--build-option': 'foo bar'}))
        ['foo', 'bar']
        >>> list(fn({'--global-option': 'foo'}))
        []
        """
        yield from self._get_config("--build-option", config_settings)


class _BuildMetaBackend(_ConfigSettingsTranslator):
    def _get_build_requires(
        self, config_settings: _ConfigSettings, requirements: list[str]
    ):
        sys.argv = [
            *sys.argv[:1],
            *self._global_args(config_settings),
            "egg_info",
        ]
        try:
            with Distribution.patch():
                self.run_setup()
        except SetupRequirementsError as e:
            requirements += e.specifiers

        return requirements

    def run_setup(self, setup_script: str = 'setup.py'):
        # Note that we can reuse our build directory between calls
        # Correctness comes first, then optimization later
        __file__ = os.path.abspath(setup_script)
        __name__ = '__main__'

        with _open_setup_script(__file__) as f:
            code = f.read().replace(r'\r\n', r'\n')

        try:
            exec(code, locals())
        except SystemExit as e:
            if e.code:
                raise
            # We ignore exit code indicating success
            SetuptoolsDeprecationWarning.emit(
                "Running `setup.py` directly as CLI tool is deprecated.",
                "Please avoid using `sys.exit(0)` or similar statements "
                "that don't fit in the paradigm of a configuration file.",
                see_url="https://blog.ganssle.io/articles/2021/10/"
                "setup-py-deprecated.html",
            )

    def get_requires_for_build_wheel(self, config_settings: _ConfigSettings = None):
        return self._get_build_requires(config_settings, requirements=[])

    def get_requires_for_build_sdist(self, config_settings: _ConfigSettings = None):
        return self._get_build_requires(config_settings, requirements=[])

    def _bubble_up_info_directory(
        self, metadata_directory: StrPath, suffix: str
    ) -> str:
        """
        PEP 517 requires that the .dist-info directory be placed in the
        metadata_directory. To comply, we MUST copy the directory to the root.

        Returns the basename of the info directory, e.g. `proj-0.0.0.dist-info`.
        """
        info_dir = self._find_info_directory(metadata_directory, suffix)
        if not same_path(info_dir.parent, metadata_directory):
            shutil.move(str(info_dir), metadata_directory)
            # PEP 517 allow other files and dirs to exist in metadata_directory
        return info_dir.name

    def _find_info_directory(self, metadata_directory: StrPath, suffix: str) -> Path:
        for parent, dirs, _ in os.walk(metadata_directory):
            candidates = [f for f in dirs if f.endswith(suffix)]

            if len(candidates) != 0 or len(dirs) != 1:
                assert len(candidates) == 1, f"Multiple {suffix} directories found"
                return Path(parent, candidates[0])

        msg = f"No {suffix} directory found in {metadata_directory}"
        raise errors.InternalError(msg)

    def prepare_metadata_for_build_wheel(
        self, metadata_directory: StrPath, config_settings: _ConfigSettings = None
    ):
        sys.argv = [
            *sys.argv[:1],
            *self._global_args(config_settings),
            "dist_info",
            "--output-dir",
            str(metadata_directory),
            "--keep-egg-info",
        ]
        with no_install_setup_requires():
            self.run_setup()

        self._bubble_up_info_directory(metadata_directory, ".egg-info")
        return self._bubble_up_info_directory(metadata_directory, ".dist-info")

    def _build_with_temp_dir(
        self,
        setup_command: Iterable[str],
        result_extension: str | tuple[str, ...],
        result_directory: StrPath,
        config_settings: _ConfigSettings,
        arbitrary_args: Iterable[str] = (),
    ):
        result_directory = os.path.abspath(result_directory)

        # Build in a temporary directory, then copy to the target.
        os.makedirs(result_directory, exist_ok=True)

        with tempfile.TemporaryDirectory(
            prefix=".tmp-", dir=result_directory
        ) as tmp_dist_dir:
            sys.argv = [
                *sys.argv[:1],
                *self._global_args(config_settings),
                *setup_command,
                "--dist-dir",
                tmp_dist_dir,
                *arbitrary_args,
            ]
            with no_install_setup_requires():
                self.run_setup()

            result_basename = _file_with_extension(tmp_dist_dir, result_extension)
            result_path = os.path.join(result_directory, result_basename)
            if os.path.exists(result_path):
                # os.rename will fail overwriting on non-Unix.
                os.remove(result_path)
            os.rename(os.path.join(tmp_dist_dir, result_basename), result_path)

        return result_basename

    def build_wheel(
        self,
        wheel_directory: StrPath,
        config_settings: _ConfigSettings = None,
        metadata_directory: StrPath | None = None,
    ):
        def _build(cmd: list[str]):
            with suppress_known_deprecation():
                return self._build_with_temp_dir(
                    cmd,
                    '.whl',
                    wheel_directory,
                    config_settings,
                    self._arbitrary_args(config_settings),
                )

        if metadata_directory is None:
            return _build(['bdist_wheel'])

        try:
            return _build(['bdist_wheel', '--dist-info-dir', str(metadata_directory)])
        except SystemExit as ex:  # pragma: nocover
            # pypa/setuptools#4683
            if "--dist-info-dir not recognized" not in str(ex):
                raise
            _IncompatibleBdistWheel.emit()
            return _build(['bdist_wheel'])

    def build_sdist(
        self, sdist_directory: StrPath, config_settings: _ConfigSettings = None
    ):
        return self._build_with_temp_dir(
            ['sdist', '--formats', 'gztar'], '.tar.gz', sdist_directory, config_settings
        )

    def _get_dist_info_dir(self, metadata_directory: StrPath | None) -> str | None:
        if not metadata_directory:
            return None
        dist_info_candidates = list(Path(metadata_directory).glob("*.dist-info"))
        assert len(dist_info_candidates) <= 1
        return str(dist_info_candidates[0]) if dist_info_candidates else None

    def build_editable(
        self,
        wheel_directory: StrPath,
        config_settings: _ConfigSettings = None,
        metadata_directory: StrPath | None = None,
    ):
        # XXX can or should we hide our editable_wheel command normally?
        info_dir = self._get_dist_info_dir(metadata_directory)
        opts = ["--dist-info-dir", info_dir] if info_dir else []
        cmd = ["editable_wheel", *opts, *self._editable_args(config_settings)]
        with suppress_known_deprecation():
            return self._build_with_temp_dir(
                cmd, ".whl", wheel_directory, config_settings
            )

    def get_requires_for_build_editable(self, config_settings: _ConfigSettings = None):
        return self.get_requires_for_build_wheel(config_settings)

    def prepare_metadata_for_build_editable(
        self, metadata_directory: StrPath, config_settings: _ConfigSettings = None
    ):
        return self.prepare_metadata_for_build_wheel(
            metadata_directory, config_settings
        )


class _BuildMetaLegacyBackend(_BuildMetaBackend):
    """Compatibility backend for setuptools

    This is a version of setuptools.build_meta that endeavors
    to maintain backwards
    compatibility with pre-PEP 517 modes of invocation. It
    exists as a temporary
    bridge between the old packaging mechanism and the new
    packaging mechanism,
    and will eventually be removed.
    """

    def run_setup(self, setup_script: str = 'setup.py'):
        # In order to maintain compatibility with scripts assuming that
        # the setup.py script is in a directory on the PYTHONPATH, inject
        # '' into sys.path. (pypa/setuptools#1642)
        sys_path = list(sys.path)  # Save the original path

        script_dir = os.path.dirname(os.path.abspath(setup_script))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        # Some setup.py scripts (e.g. in pygame and numpy) use sys.argv[0] to
        # get the directory of the source code. They expect it to refer to the
        # setup.py script.
        sys_argv_0 = sys.argv[0]
        sys.argv[0] = setup_script

        try:
            super().run_setup(setup_script=setup_script)
        finally:
            # While PEP 517 frontends should be calling each hook in a fresh
            # subprocess according to the standard (and thus it should not be
            # strictly necessary to restore the old sys.path), we'll restore
            # the original path so that the path manipulation does not persist
            # within the hook after run_setup is called.
            sys.path[:] = sys_path
            sys.argv[0] = sys_argv_0


class _IncompatibleBdistWheel(SetuptoolsDeprecationWarning):
    _SUMMARY = "wheel.bdist_wheel is deprecated, please import it from setuptools"
    _DETAILS = """
    Ensure that any custom bdist_wheel implementation is a subclass of
    setuptools.command.bdist_wheel.bdist_wheel.
    """
    _DUE_DATE = (2025, 10, 15)
    # Initially introduced in 2024/10/15, but maybe too disruptive to be enforced?
    _SEE_URL = "https://github.com/pypa/wheel/pull/631"


# The primary backend
_BACKEND = _BuildMetaBackend()

get_requires_for_build_wheel = _BACKEND.get_requires_for_build_wheel
get_requires_for_build_sdist = _BACKEND.get_requires_for_build_sdist
prepare_metadata_for_build_wheel = _BACKEND.prepare_metadata_for_build_wheel
build_wheel = _BACKEND.build_wheel
build_sdist = _BACKEND.build_sdist
get_requires_for_build_editable = _BACKEND.get_requires_for_build_editable
prepare_metadata_for_build_editable = _BACKEND.prepare_metadata_for_build_editable
build_editable = _BACKEND.build_editable


# The legacy backend
__legacy__ = _BuildMetaLegacyBackend()

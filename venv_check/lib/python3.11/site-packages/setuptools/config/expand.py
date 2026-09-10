"""Utility functions to expand configuration directives or special values
(such glob patterns).

We can split the process of interpreting configuration files into 2 steps:

1. The parsing the file contents from strings to value objects
   that can be understand by Python (for example a string with a comma
   separated list of keywords into an actual Python list of strings).

2. The expansion (or post-processing) of these values according to the
   semantics ``setuptools`` assign to them (for example a configuration field
   with the ``file:`` directive should be expanded from a list of file paths to
   a single string with the contents of those files concatenated)

This module focus on the second step, and therefore allow sharing the expansion
functions among several configuration file formats.

**PRIVATE MODULE**: API reserved for setuptools internal usage only.
"""

from __future__ import annotations

import ast
import importlib
import os
import pathlib
import sys
from collections.abc import Iterable, Iterator, Mapping
from configparser import ConfigParser
from glob import iglob
from importlib.machinery import ModuleSpec, all_suffixes
from itertools import chain
from pathlib import Path
from types import ModuleType, TracebackType
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from .. import _static
from .._path import StrPath, same_path as _same_path
from ..discovery import find_package_path
from ..warnings import SetuptoolsWarning

from distutils.errors import DistutilsOptionError

if TYPE_CHECKING:
    from typing_extensions import Self

    from setuptools.dist import Distribution

_K = TypeVar("_K")
_V_co = TypeVar("_V_co", covariant=True)


class StaticModule:
    """Proxy to a module object that avoids executing arbitrary code."""

    def __init__(self, name: str, spec: ModuleSpec) -> None:
        module = ast.parse(pathlib.Path(spec.origin).read_bytes())  # type: ignore[arg-type] # Let it raise an error on None
        vars(self).update(locals())
        del self.self

    def _find_assignments(self) -> Iterator[tuple[ast.AST, ast.AST]]:
        for statement in self.module.body:
            if isinstance(statement, ast.Assign):
                yield from ((target, statement.value) for target in statement.targets)
            elif isinstance(statement, ast.AnnAssign) and statement.value:
                yield (statement.target, statement.value)

    def __getattr__(self, attr: str):
        """Attempt to load an attribute "statically", via :func:`ast.literal_eval`."""
        try:
            return next(
                ast.literal_eval(value)
                for target, value in self._find_assignments()
                if isinstance(target, ast.Name) and target.id == attr
            )
        except Exception as e:
            raise AttributeError(f"{self.name} has no attribute {attr}") from e


def glob_relative(
    patterns: Iterable[str], root_dir: StrPath | None = None
) -> list[str]:
    """Expand the list of glob patterns, but preserving relative paths.

    :param list[str] patterns: List of glob patterns
    :param str root_dir: Path to which globs should be relative
                         (current directory by default)
    :rtype: list
    """
    glob_characters = {'*', '?', '[', ']', '{', '}'}
    expanded_values = []
    root_dir = root_dir or os.getcwd()
    for value in patterns:
        # Has globby characters?
        if any(char in value for char in glob_characters):
            # then expand the glob pattern while keeping paths *relative*:
            glob_path = os.path.abspath(os.path.join(root_dir, value))
            expanded_values.extend(
                sorted(
                    os.path.relpath(path, root_dir).replace(os.sep, "/")
                    for path in iglob(glob_path, recursive=True)
                )
            )

        else:
            # take the value as-is
            path = os.path.relpath(value, root_dir).replace(os.sep, "/")
            expanded_values.append(path)

    return expanded_values


def read_files(
    filepaths: StrPath | Iterable[StrPath], root_dir: StrPath | None = None
) -> str:
    """Return the content of the files concatenated using ``\n`` as str

    This function is sandboxed and won't reach anything outside ``root_dir``

    (By default ``root_dir`` is the current directory).
    """
    from more_itertools import always_iterable

    root_dir = os.path.abspath(root_dir or os.getcwd())
    _filepaths = (os.path.join(root_dir, path) for path in always_iterable(filepaths))
    return '\n'.join(
        _read_file(path)
        for path in _filter_existing_files(_filepaths)
        if _assert_local(path, root_dir)
    )


def _filter_existing_files(filepaths: Iterable[StrPath]) -> Iterator[StrPath]:
    for path in filepaths:
        if os.path.isfile(path):
            yield path
        else:
            SetuptoolsWarning.emit(f"File {path!r} cannot be found")


def _read_file(filepath: bytes | StrPath) -> str:
    with open(filepath, encoding='utf-8') as f:
        return f.read()


def _assert_local(filepath: StrPath, root_dir: str):
    if Path(os.path.abspath(root_dir)) not in Path(os.path.abspath(filepath)).parents:
        msg = f"Cannot access {filepath!r} (or anything outside {root_dir!r})"
        raise DistutilsOptionError(msg)

    return True


def read_attr(
    attr_desc: str,
    package_dir: Mapping[str, str] | None = None,
    root_dir: StrPath | None = None,
) -> Any:
    """Reads the value of an attribute from a module.

    This function will try to read the attributed statically first
    (via :func:`ast.literal_eval`), and only evaluate the module if it fails.

    Examples:
        read_attr("package.attr")
        read_attr("package.module.attr")

    :param str attr_desc: Dot-separated string describing how to reach the
        attribute (see examples above)
    :param dict[str, str] package_dir: Mapping of package names to their
        location in disk (represented by paths relative to ``root_dir``).
    :param str root_dir: Path to directory containing all the packages in
        ``package_dir`` (current directory by default).
    :rtype: str
    """
    root_dir = root_dir or os.getcwd()
    attrs_path = attr_desc.strip().split('.')
    attr_name = attrs_path.pop()
    module_name = '.'.join(attrs_path)
    module_name = module_name or '__init__'
    path = _find_module(module_name, package_dir, root_dir)
    spec = _find_spec(module_name, path)

    try:
        value = getattr(StaticModule(module_name, spec), attr_name)
        # XXX: Is marking as static contents coming from modules too optimistic?
        return _static.attempt_conversion(value)
    except Exception:
        # fallback to evaluate module
        module = _load_spec(spec, module_name)
        return getattr(module, attr_name)


def _find_spec(module_name: str, module_path: StrPath | None) -> ModuleSpec:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    spec = spec or importlib.util.find_spec(module_name)

    if spec is None:
        raise ModuleNotFoundError(module_name)

    return spec


def _load_spec(spec: ModuleSpec, module_name: str) -> ModuleType:
    name = getattr(spec, "__name__", module_name)
    if name in sys.modules:
        return sys.modules[name]
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # cache (it also ensures `==` works on loaded items)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _find_module(
    module_name: str, package_dir: Mapping[str, str] | None, root_dir: StrPath
) -> str | None:
    """Find the path to the module named ``module_name``,
    considering the ``package_dir`` in the build configuration and ``root_dir``.

    >>> tmp = getfixture('tmpdir')
    >>> _ = tmp.ensure("a/b/c.py")
    >>> _ = tmp.ensure("a/b/d/__init__.py")
    >>> r = lambda x: x.replace(str(tmp), "tmp").replace(os.sep, "/")
    >>> r(_find_module("a.b.c", None, tmp))
    'tmp/a/b/c.py'
    >>> r(_find_module("f.g.h", {"": "1", "f": "2", "f.g": "3", "f.g.h": "a/b/d"}, tmp))
    'tmp/a/b/d/__init__.py'
    """
    path_start = find_package_path(module_name, package_dir or {}, root_dir)
    candidates = chain.from_iterable(
        (f"{path_start}{ext}", os.path.join(path_start, f"__init__{ext}"))
        for ext in all_suffixes()
    )
    return next((x for x in candidates if os.path.isfile(x)), None)


def resolve_class(
    qualified_class_name: str,
    package_dir: Mapping[str, str] | None = None,
    root_dir: StrPath | None = None,
) -> Callable:
    """Given a qualified class name, return the associated class object"""
    root_dir = root_dir or os.getcwd()
    idx = qualified_class_name.rfind('.')
    class_name = qualified_class_name[idx + 1 :]
    pkg_name = qualified_class_name[:idx]

    path = _find_module(pkg_name, package_dir, root_dir)
    module = _load_spec(_find_spec(pkg_name, path), pkg_name)
    return getattr(module, class_name)


def cmdclass(
    values: dict[str, str],
    package_dir: Mapping[str, str] | None = None,
    root_dir: StrPath | None = None,
) -> dict[str, Callable]:
    """Given a dictionary mapping command names to strings for qualified class
    names, apply :func:`resolve_class` to the dict values.
    """
    return {k: resolve_class(v, package_dir, root_dir) for k, v in values.items()}


def find_packages(
    *,
    namespaces=True,
    fill_package_dir: dict[str, str] | None = None,
    root_dir: StrPath | None = None,
    **kwargs,
) -> list[str]:
    """Works similarly to :func:`setuptools.find_packages`, but with all
    arguments given as keyword arguments. Moreover, ``where`` can be given
    as a list (the results will be simply concatenated).

    When the additional keyword argument ``namespaces`` is ``True``, it will
    behave like :func:`setuptools.find_namespace_packages`` (i.e. include
    implicit namespaces as per :pep:`420`).

    The ``where`` argument will be considered relative to ``root_dir`` (or the current
    working directory when ``root_dir`` is not given).

    If the ``fill_package_dir`` argument is passed, this function will consider it as a
    similar data structure to the ``package_dir`` configuration parameter add fill-in
    any missing package location.

    :rtype: list
    """
    from more_itertools import always_iterable, unique_everseen

    from setuptools.discovery import construct_package_dir

    # check "not namespaces" first due to python/mypy#6232
    if not namespaces:
        from setuptools.discovery import PackageFinder
    else:
        from setuptools.discovery import PEP420PackageFinder as PackageFinder

    root_dir = root_dir or os.curdir
    where = kwargs.pop('where', ['.'])
    packages: list[str] = []
    fill_package_dir = {} if fill_package_dir is None else fill_package_dir
    search = list(unique_everseen(always_iterable(where)))

    if len(search) == 1 and all(not _same_path(search[0], x) for x in (".", root_dir)):
        fill_package_dir.setdefault("", search[0])

    for path in search:
        package_path = _nest_path(root_dir, path)
        pkgs = PackageFinder.find(package_path, **kwargs)
        packages.extend(pkgs)
        if pkgs and not (
            fill_package_dir.get("") == path or os.path.samefile(package_path, root_dir)
        ):
            fill_package_dir.update(construct_package_dir(pkgs, path))

    return packages


def _nest_path(parent: StrPath, path: StrPath) -> str:
    path = parent if path in {".", ""} else os.path.join(parent, path)
    return os.path.normpath(path)


def version(value: Callable | Iterable[str | int] | str) -> str:
    """When getting the version directly from an attribute,
    it should be normalised to string.
    """
    _value = value() if callable(value) else value

    if isinstance(_value, str):
        return _value
    if hasattr(_value, '__iter__'):
        return '.'.join(map(str, _value))
    return f'{_value}'


def canonic_package_data(package_data: dict) -> dict:
    if "*" in package_data:
        package_data[""] = package_data.pop("*")
    return package_data


def canonic_data_files(
    data_files: list | dict, root_dir: StrPath | None = None
) -> list[tuple[str, list[str]]]:
    """For compatibility with ``setup.py``, ``data_files`` should be a list
    of pairs instead of a dict.

    This function also expands glob patterns.
    """
    if isinstance(data_files, list):
        return data_files

    return [
        (dest, glob_relative(patterns, root_dir))
        for dest, patterns in data_files.items()
    ]


def entry_points(
    text: str, text_source: str = "entry-points"
) -> dict[str, dict[str, str]]:
    """Given the contents of entry-points file,
    process it into a 2-level dictionary (``dict[str, dict[str, str]]``).
    The first level keys are entry-point groups, the second level keys are
    entry-point names, and the second level values are references to objects
    (that correspond to the entry-point value).
    """
    # Using undocumented behaviour, see python/typeshed#12700
    parser = ConfigParser(default_section=None, delimiters=("=",))  # type: ignore[call-overload]
    parser.optionxform = str  # case sensitive
    parser.read_string(text, text_source)
    groups = {k: dict(v.items()) for k, v in parser.items()}
    groups.pop(parser.default_section, None)
    return groups


class EnsurePackagesDiscovered:
    """Some expand functions require all the packages to already be discovered before
    they run, e.g. :func:`read_attr`, :func:`resolve_class`, :func:`cmdclass`.

    Therefore in some cases we will need to run autodiscovery during the evaluation of
    the configuration. However, it is better to postpone calling package discovery as
    much as possible, because some parameters can influence it (e.g. ``package_dir``),
    and those might not have been processed yet.
    """

    def __init__(self, distribution: Distribution) -> None:
        self._dist = distribution
        self._called = False

    def __call__(self):
        """Trigger the automatic package discovery, if it is still necessary."""
        if not self._called:
            self._called = True
            self._dist.set_defaults(name=False)  # Skip name, we can still be parsing

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ):
        if self._called:
            self._dist.set_defaults.analyse_name()  # Now we can set a default name

    def _get_package_dir(self) -> Mapping[str, str]:
        self()
        pkg_dir = self._dist.package_dir
        return {} if pkg_dir is None else pkg_dir

    @property
    def package_dir(self) -> Mapping[str, str]:
        """Proxy to ``package_dir`` that may trigger auto-discovery when used."""
        return LazyMappingProxy(self._get_package_dir)


class LazyMappingProxy(Mapping[_K, _V_co]):
    """Mapping proxy that delays resolving the target object, until really needed.

    >>> def obtain_mapping():
    ...     print("Running expensive function!")
    ...     return {"key": "value", "other key": "other value"}
    >>> mapping = LazyMappingProxy(obtain_mapping)
    >>> mapping["key"]
    Running expensive function!
    'value'
    >>> mapping["other key"]
    'other value'
    """

    def __init__(self, obtain_mapping_value: Callable[[], Mapping[_K, _V_co]]) -> None:
        self._obtain = obtain_mapping_value
        self._value: Mapping[_K, _V_co] | None = None

    def _target(self) -> Mapping[_K, _V_co]:
        if self._value is None:
            self._value = self._obtain()
        return self._value

    def __getitem__(self, key: _K) -> _V_co:
        return self._target()[key]

    def __len__(self) -> int:
        return len(self._target())

    def __iter__(self) -> Iterator[_K]:
        return iter(self._target())

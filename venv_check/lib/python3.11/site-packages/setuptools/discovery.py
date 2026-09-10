"""Automatic discovery of Python modules and packages (for inclusion in the
distribution) and other config values.

For the purposes of this module, the following nomenclature is used:

- "src-layout": a directory representing a Python project that contains a "src"
  folder. Everything under the "src" folder is meant to be included in the
  distribution when packaging the project. Example::

    .
    ├── tox.ini
    ├── pyproject.toml
    └── src/
        └── mypkg/
            ├── __init__.py
            ├── mymodule.py
            └── my_data_file.txt

- "flat-layout": a Python project that does not use "src-layout" but instead
  have a directory under the project root for each package::

    .
    ├── tox.ini
    ├── pyproject.toml
    └── mypkg/
        ├── __init__.py
        ├── mymodule.py
        └── my_data_file.txt

- "single-module": a project that contains a single Python script direct under
  the project root (no directory used)::

    .
    ├── tox.ini
    ├── pyproject.toml
    └── mymodule.py

"""

from __future__ import annotations

import itertools
import os
from collections.abc import Iterable, Iterator, Mapping
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import _distutils_hack.override  # noqa: F401

from ._path import StrPath

from distutils import log
from distutils.util import convert_path

if TYPE_CHECKING:
    from setuptools import Distribution

chain_iter = itertools.chain.from_iterable


def _valid_name(path: StrPath) -> bool:
    # Ignore invalid names that cannot be imported directly
    return os.path.basename(path).isidentifier()


class _Filter:
    """
    Given a list of patterns, create a callable that will be true only if
    the input matches at least one of the patterns.
    """

    def __init__(self, *patterns: str) -> None:
        self._patterns = dict.fromkeys(patterns)

    def __call__(self, item: str) -> bool:
        return any(fnmatchcase(item, pat) for pat in self._patterns)

    def __contains__(self, item: str) -> bool:
        return item in self._patterns


class _Finder:
    """Base class that exposes functionality for module/package finders"""

    ALWAYS_EXCLUDE: ClassVar[tuple[str, ...]] = ()
    DEFAULT_EXCLUDE: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def find(
        cls,
        where: StrPath = '.',
        exclude: Iterable[str] = (),
        include: Iterable[str] = ('*',),
    ) -> list[str]:
        """Return a list of all Python items (packages or modules, depending on
        the finder implementation) found within directory 'where'.

        'where' is the root directory which will be searched.
        It should be supplied as a "cross-platform" (i.e. URL-style) path;
        it will be converted to the appropriate local path syntax.

        'exclude' is a sequence of names to exclude; '*' can be used
        as a wildcard in the names.
        When finding packages, 'foo.*' will exclude all subpackages of 'foo'
        (but not 'foo' itself).

        'include' is a sequence of names to include.
        If it's specified, only the named items will be included.
        If it's not specified, all found items will be included.
        'include' can contain shell style wildcard patterns just like
        'exclude'.
        """

        exclude = exclude or cls.DEFAULT_EXCLUDE
        return list(
            cls._find_iter(
                convert_path(str(where)),
                _Filter(*cls.ALWAYS_EXCLUDE, *exclude),
                _Filter(*include),
            )
        )

    @classmethod
    def _find_iter(
        cls, where: StrPath, exclude: _Filter, include: _Filter
    ) -> Iterator[str]:
        raise NotImplementedError


class PackageFinder(_Finder):
    """
    Generate a list of all Python packages found within a directory
    """

    ALWAYS_EXCLUDE = ("ez_setup", "*__pycache__")

    @classmethod
    def _find_iter(
        cls, where: StrPath, exclude: _Filter, include: _Filter
    ) -> Iterator[str]:
        """
        All the packages found in 'where' that pass the 'include' filter, but
        not the 'exclude' filter.
        """
        for root, dirs, files in os.walk(str(where), followlinks=True):
            # Copy dirs to iterate over it, then empty dirs.
            all_dirs = dirs[:]
            dirs[:] = []

            for dir in all_dirs:
                full_path = os.path.join(root, dir)
                rel_path = os.path.relpath(full_path, where)
                package = rel_path.replace(os.path.sep, '.')

                # Skip directory trees that are not valid packages
                if '.' in dir or not cls._looks_like_package(full_path, package):
                    continue

                # Should this package be included?
                if include(package) and not exclude(package):
                    yield package

                # Early pruning if there is nothing else to be scanned
                if f"{package}*" in exclude or f"{package}.*" in exclude:
                    continue

                # Keep searching subdirectories, as there may be more packages
                # down there, even if the parent was excluded.
                dirs.append(dir)

    @staticmethod
    def _looks_like_package(path: StrPath, _package_name: str) -> bool:
        """Does a directory look like a package?"""
        return os.path.isfile(os.path.join(path, '__init__.py'))


class PEP420PackageFinder(PackageFinder):
    @staticmethod
    def _looks_like_package(_path: StrPath, _package_name: str) -> bool:
        return True


class ModuleFinder(_Finder):
    """Find isolated Python modules.
    This function will **not** recurse subdirectories.
    """

    @classmethod
    def _find_iter(
        cls, where: StrPath, exclude: _Filter, include: _Filter
    ) -> Iterator[str]:
        for file in glob(os.path.join(where, "*.py")):
            module, _ext = os.path.splitext(os.path.basename(file))

            if not cls._looks_like_module(module):
                continue

            if include(module) and not exclude(module):
                yield module

    _looks_like_module = staticmethod(_valid_name)


# We have to be extra careful in the case of flat layout to not include files
# and directories not meant for distribution (e.g. tool-related)


class FlatLayoutPackageFinder(PEP420PackageFinder):
    _EXCLUDE = (
        "ci",
        "bin",
        "debian",
        "doc",
        "docs",
        "documentation",
        "manpages",
        "news",
        "newsfragments",
        "changelog",
        "test",
        "tests",
        "unit_test",
        "unit_tests",
        "example",
        "examples",
        "scripts",
        "tools",
        "util",
        "utils",
        "python",
        "build",
        "dist",
        "venv",
        "env",
        "requirements",
        # ---- Task runners / Build tools ----
        "tasks",  # invoke
        "fabfile",  # fabric
        "site_scons",  # SCons
        # ---- Other tools ----
        "benchmark",
        "benchmarks",
        "exercise",
        "exercises",
        "htmlcov",  # Coverage.py
        # ---- Hidden directories/Private packages ----
        "[._]*",
    )

    DEFAULT_EXCLUDE = tuple(chain_iter((p, f"{p}.*") for p in _EXCLUDE))
    """Reserved package names"""

    @staticmethod
    def _looks_like_package(_path: StrPath, package_name: str) -> bool:
        names = package_name.split('.')
        # Consider PEP 561
        root_pkg_is_valid = names[0].isidentifier() or names[0].endswith("-stubs")
        return root_pkg_is_valid and all(name.isidentifier() for name in names[1:])


class FlatLayoutModuleFinder(ModuleFinder):
    DEFAULT_EXCLUDE = (
        "setup",
        "conftest",
        "test",
        "tests",
        "example",
        "examples",
        "build",
        # ---- Task runners ----
        "toxfile",
        "noxfile",
        "pavement",
        "dodo",
        "tasks",
        "fabfile",
        # ---- Other tools ----
        "[Ss][Cc]onstruct",  # SCons
        "conanfile",  # Connan: C/C++ build tool
        "manage",  # Django
        "benchmark",
        "benchmarks",
        "exercise",
        "exercises",
        # ---- Hidden files/Private modules ----
        "[._]*",
    )
    """Reserved top-level module names"""


def _find_packages_within(root_pkg: str, pkg_dir: StrPath) -> list[str]:
    nested = PEP420PackageFinder.find(pkg_dir)
    return [root_pkg] + [".".join((root_pkg, n)) for n in nested]


class ConfigDiscovery:
    """Fill-in metadata and options that can be automatically derived
    (from other metadata/options, the file system or conventions)
    """

    def __init__(self, distribution: Distribution) -> None:
        self.dist = distribution
        self._called = False
        self._disabled = False
        self._skip_ext_modules = False

    def _disable(self):
        """Internal API to disable automatic discovery"""
        self._disabled = True

    def _ignore_ext_modules(self):
        """Internal API to disregard ext_modules.

        Normally auto-discovery would not be triggered if ``ext_modules`` are set
        (this is done for backward compatibility with existing packages relying on
        ``setup.py`` or ``setup.cfg``). However, ``setuptools`` can call this function
        to ignore given ``ext_modules`` and proceed with the auto-discovery if
        ``packages`` and ``py_modules`` are not given (e.g. when using pyproject.toml
        metadata).
        """
        self._skip_ext_modules = True

    @property
    def _root_dir(self) -> StrPath:
        # The best is to wait until `src_root` is set in dist, before using _root_dir.
        return self.dist.src_root or os.curdir

    @property
    def _package_dir(self) -> dict[str, str]:
        if self.dist.package_dir is None:
            return {}
        return self.dist.package_dir

    def __call__(
        self, force: bool = False, name: bool = True, ignore_ext_modules: bool = False
    ):
        """Automatically discover missing configuration fields
        and modifies the given ``distribution`` object in-place.

        Note that by default this will only have an effect the first time the
        ``ConfigDiscovery`` object is called.

        To repeatedly invoke automatic discovery (e.g. when the project
        directory changes), please use ``force=True`` (or create a new
        ``ConfigDiscovery`` instance).
        """
        if force is False and (self._called or self._disabled):
            # Avoid overhead of multiple calls
            return

        self._analyse_package_layout(ignore_ext_modules)
        if name:
            self.analyse_name()  # depends on ``packages`` and ``py_modules``

        self._called = True

    def _explicitly_specified(self, ignore_ext_modules: bool) -> bool:
        """``True`` if the user has specified some form of package/module listing"""
        ignore_ext_modules = ignore_ext_modules or self._skip_ext_modules
        ext_modules = not (self.dist.ext_modules is None or ignore_ext_modules)
        return (
            self.dist.packages is not None
            or self.dist.py_modules is not None
            or ext_modules
            or hasattr(self.dist, "configuration")
            and self.dist.configuration
            # ^ Some projects use numpy.distutils.misc_util.Configuration
        )

    def _analyse_package_layout(self, ignore_ext_modules: bool) -> bool:
        if self._explicitly_specified(ignore_ext_modules):
            # For backward compatibility, just try to find modules/packages
            # when nothing is given
            return True

        log.debug(
            "No `packages` or `py_modules` configuration, performing "
            "automatic discovery."
        )

        return (
            self._analyse_explicit_layout()
            or self._analyse_src_layout()
            # flat-layout is the trickiest for discovery so it should be last
            or self._analyse_flat_layout()
        )

    def _analyse_explicit_layout(self) -> bool:
        """The user can explicitly give a package layout via ``package_dir``"""
        package_dir = self._package_dir.copy()  # don't modify directly
        package_dir.pop("", None)  # This falls under the "src-layout" umbrella
        root_dir = self._root_dir

        if not package_dir:
            return False

        log.debug(f"`explicit-layout` detected -- analysing {package_dir}")
        pkgs = chain_iter(
            _find_packages_within(pkg, os.path.join(root_dir, parent_dir))
            for pkg, parent_dir in package_dir.items()
        )
        self.dist.packages = list(pkgs)
        log.debug(f"discovered packages -- {self.dist.packages}")
        return True

    def _analyse_src_layout(self) -> bool:
        """Try to find all packages or modules under the ``src`` directory
        (or anything pointed by ``package_dir[""]``).

        The "src-layout" is relatively safe for automatic discovery.
        We assume that everything within is meant to be included in the
        distribution.

        If ``package_dir[""]`` is not given, but the ``src`` directory exists,
        this function will set ``package_dir[""] = "src"``.
        """
        package_dir = self._package_dir
        src_dir = os.path.join(self._root_dir, package_dir.get("", "src"))
        if not os.path.isdir(src_dir):
            return False

        log.debug(f"`src-layout` detected -- analysing {src_dir}")
        package_dir.setdefault("", os.path.basename(src_dir))
        self.dist.package_dir = package_dir  # persist eventual modifications
        self.dist.packages = PEP420PackageFinder.find(src_dir)
        self.dist.py_modules = ModuleFinder.find(src_dir)
        log.debug(f"discovered packages -- {self.dist.packages}")
        log.debug(f"discovered py_modules -- {self.dist.py_modules}")
        return True

    def _analyse_flat_layout(self) -> bool:
        """Try to find all packages and modules under the project root.

        Since the ``flat-layout`` is more dangerous in terms of accidentally including
        extra files/directories, this function is more conservative and will raise an
        error if multiple packages or modules are found.

        This assumes that multi-package dists are uncommon and refuse to support that
        use case in order to be able to prevent unintended errors.
        """
        log.debug(f"`flat-layout` detected -- analysing {self._root_dir}")
        return self._analyse_flat_packages() or self._analyse_flat_modules()

    def _analyse_flat_packages(self) -> bool:
        self.dist.packages = FlatLayoutPackageFinder.find(self._root_dir)
        top_level = remove_nested_packages(remove_stubs(self.dist.packages))
        log.debug(f"discovered packages -- {self.dist.packages}")
        self._ensure_no_accidental_inclusion(top_level, "packages")
        return bool(top_level)

    def _analyse_flat_modules(self) -> bool:
        self.dist.py_modules = FlatLayoutModuleFinder.find(self._root_dir)
        log.debug(f"discovered py_modules -- {self.dist.py_modules}")
        self._ensure_no_accidental_inclusion(self.dist.py_modules, "modules")
        return bool(self.dist.py_modules)

    def _ensure_no_accidental_inclusion(self, detected: list[str], kind: str):
        if len(detected) > 1:
            from inspect import cleandoc

            from setuptools.errors import PackageDiscoveryError

            msg = f"""Multiple top-level {kind} discovered in a flat-layout: {detected}.

            To avoid accidental inclusion of unwanted files or directories,
            setuptools will not proceed with this build.

            If you are trying to create a single distribution with multiple {kind}
            on purpose, you should not rely on automatic discovery.
            Instead, consider the following options:

            1. set up custom discovery (`find` directive with `include` or `exclude`)
            2. use a `src-layout`
            3. explicitly set `py_modules` or `packages` with a list of names

            To find more information, look for "package discovery" on setuptools docs.
            """
            raise PackageDiscoveryError(cleandoc(msg))

    def analyse_name(self) -> None:
        """The packages/modules are the essential contribution of the author.
        Therefore the name of the distribution can be derived from them.
        """
        if self.dist.metadata.name or self.dist.name:
            # get_name() is not reliable (can return "UNKNOWN")
            return

        log.debug("No `name` configuration, performing automatic discovery")

        name = (
            self._find_name_single_package_or_module()
            or self._find_name_from_packages()
        )
        if name:
            self.dist.metadata.name = name

    def _find_name_single_package_or_module(self) -> str | None:
        """Exactly one module or package"""
        for field in ('packages', 'py_modules'):
            items = getattr(self.dist, field, None) or []
            if items and len(items) == 1:
                log.debug(f"Single module/package detected, name: {items[0]}")
                return items[0]

        return None

    def _find_name_from_packages(self) -> str | None:
        """Try to find the root package that is not a PEP 420 namespace"""
        if not self.dist.packages:
            return None

        packages = remove_stubs(sorted(self.dist.packages, key=len))
        package_dir = self.dist.package_dir or {}

        parent_pkg = find_parent_package(packages, package_dir, self._root_dir)
        if parent_pkg:
            log.debug(f"Common parent package detected, name: {parent_pkg}")
            return parent_pkg

        log.warn("No parent package detected, impossible to derive `name`")
        return None


def remove_nested_packages(packages: list[str]) -> list[str]:
    """Remove nested packages from a list of packages.

    >>> remove_nested_packages(["a", "a.b1", "a.b2", "a.b1.c1"])
    ['a']
    >>> remove_nested_packages(["a", "b", "c.d", "c.d.e.f", "g.h", "a.a1"])
    ['a', 'b', 'c.d', 'g.h']
    """
    pkgs = sorted(packages, key=len)
    top_level = pkgs[:]
    size = len(pkgs)
    for i, name in enumerate(reversed(pkgs)):
        if any(name.startswith(f"{other}.") for other in top_level):
            top_level.pop(size - i - 1)

    return top_level


def remove_stubs(packages: list[str]) -> list[str]:
    """Remove type stubs (:pep:`561`) from a list of packages.

    >>> remove_stubs(["a", "a.b", "a-stubs", "a-stubs.b.c", "b", "c-stubs"])
    ['a', 'a.b', 'b']
    """
    return [pkg for pkg in packages if not pkg.split(".")[0].endswith("-stubs")]


def find_parent_package(
    packages: list[str], package_dir: Mapping[str, str], root_dir: StrPath
) -> str | None:
    """Find the parent package that is not a namespace."""
    packages = sorted(packages, key=len)
    common_ancestors = []
    for i, name in enumerate(packages):
        if not all(n.startswith(f"{name}.") for n in packages[i + 1 :]):
            # Since packages are sorted by length, this condition is able
            # to find a list of all common ancestors.
            # When there is divergence (e.g. multiple root packages)
            # the list will be empty
            break
        common_ancestors.append(name)

    for name in common_ancestors:
        pkg_path = find_package_path(name, package_dir, root_dir)
        init = os.path.join(pkg_path, "__init__.py")
        if os.path.isfile(init):
            return name

    return None


def find_package_path(
    name: str, package_dir: Mapping[str, str], root_dir: StrPath
) -> str:
    """Given a package name, return the path where it should be found on
    disk, considering the ``package_dir`` option.

    >>> path = find_package_path("my.pkg", {"": "root/is/nested"}, ".")
    >>> path.replace(os.sep, "/")
    './root/is/nested/my/pkg'

    >>> path = find_package_path("my.pkg", {"my": "root/is/nested"}, ".")
    >>> path.replace(os.sep, "/")
    './root/is/nested/pkg'

    >>> path = find_package_path("my.pkg", {"my.pkg": "root/is/nested"}, ".")
    >>> path.replace(os.sep, "/")
    './root/is/nested'

    >>> path = find_package_path("other.pkg", {"my.pkg": "root/is/nested"}, ".")
    >>> path.replace(os.sep, "/")
    './other/pkg'
    """
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        # Look backwards, the most specific package_dir first
        partial_name = ".".join(parts[:i])
        if partial_name in package_dir:
            parent = package_dir[partial_name]
            return os.path.join(root_dir, parent, *parts[i:])

    parent = package_dir.get("") or ""
    return os.path.join(root_dir, *parent.split("/"), *parts)


def construct_package_dir(packages: list[str], package_path: StrPath) -> dict[str, str]:
    parent_pkgs = remove_nested_packages(packages)
    prefix = Path(package_path).parts
    return {pkg: "/".join([*prefix, *pkg.split(".")]) for pkg in parent_pkgs}

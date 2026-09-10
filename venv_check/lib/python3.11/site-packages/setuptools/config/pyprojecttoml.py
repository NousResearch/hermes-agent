"""
Load setuptools configuration from ``pyproject.toml`` files.

**PRIVATE MODULE**: API reserved for setuptools internal usage only.

To read project metadata, consider using
``build.util.project_wheel_metadata`` (https://pypi.org/project/build/).
For simple scenarios, you can also try parsing the file directly
with the help of ``tomllib`` or ``tomli``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from contextlib import contextmanager
from functools import partial
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable

from .._path import StrPath
from ..errors import FileError, InvalidConfigError
from ..warnings import SetuptoolsWarning
from . import expand as _expand
from ._apply_pyprojecttoml import _PREVIOUSLY_DEFINED, _MissingDynamic, apply as _apply

if TYPE_CHECKING:
    from typing_extensions import Self

    from setuptools.dist import Distribution

_logger = logging.getLogger(__name__)


def load_file(filepath: StrPath) -> dict:
    from ..compat.py310 import tomllib

    with open(filepath, "rb") as file:
        return tomllib.load(file)


def validate(config: dict, filepath: StrPath) -> bool:
    from . import _validate_pyproject as validator

    trove_classifier = validator.FORMAT_FUNCTIONS.get("trove-classifier")
    if hasattr(trove_classifier, "_disable_download"):
        # Improve reproducibility by default. See abravalheri/validate-pyproject#31
        trove_classifier._disable_download()  # type: ignore[union-attr]

    try:
        return validator.validate(config)
    except validator.ValidationError as ex:
        summary = f"configuration error: {ex.summary}"
        if ex.name.strip("`") != "project":
            # Probably it is just a field missing/misnamed, not worthy the verbosity...
            _logger.debug(summary)
            _logger.debug(ex.details)

        error = f"invalid pyproject.toml config: {ex.name}."
        raise ValueError(f"{error}\n{summary}") from None


def apply_configuration(
    dist: Distribution,
    filepath: StrPath,
    ignore_option_errors: bool = False,
) -> Distribution:
    """Apply the configuration from a ``pyproject.toml`` file into an existing
    distribution object.
    """
    config = read_configuration(filepath, True, ignore_option_errors, dist)
    return _apply(dist, config, filepath)


def read_configuration(
    filepath: StrPath,
    expand: bool = True,
    ignore_option_errors: bool = False,
    dist: Distribution | None = None,
) -> dict[str, Any]:
    """Read given configuration file and returns options from it as a dict.

    :param str|unicode filepath: Path to configuration file in the ``pyproject.toml``
        format.

    :param bool expand: Whether to expand directives and other computed values
        (i.e. post-process the given configuration)

    :param bool ignore_option_errors: Whether to silently ignore
        options, values of which could not be resolved (e.g. due to exceptions
        in directives such as file:, attr:, etc.).
        If False exceptions are propagated as expected.

    :param Distribution|None: Distribution object to which the configuration refers.
        If not given a dummy object will be created and discarded after the
        configuration is read. This is used for auto-discovery of packages and in the
        case a dynamic configuration (e.g. ``attr`` or ``cmdclass``) is expanded.
        When ``expand=False`` this object is simply ignored.

    :rtype: dict
    """
    filepath = os.path.abspath(filepath)

    if not os.path.isfile(filepath):
        raise FileError(f"Configuration file {filepath!r} does not exist.")

    asdict = load_file(filepath) or {}
    project_table = asdict.get("project", {})
    tool_table = asdict.get("tool", {})
    setuptools_table = tool_table.get("setuptools", {})
    if not asdict or not (project_table or setuptools_table):
        return {}  # User is not using pyproject to configure setuptools

    if "setuptools" in asdict.get("tools", {}):
        # let the user know they probably have a typo in their metadata
        _ToolsTypoInMetadata.emit()

    if "distutils" in tool_table:
        _ExperimentalConfiguration.emit(subject="[tool.distutils]")

    # There is an overall sense in the community that making include_package_data=True
    # the default would be an improvement.
    # `ini2toml` backfills include_package_data=False when nothing is explicitly given,
    # therefore setting a default here is backwards compatible.
    if dist and dist.include_package_data is not None:
        setuptools_table.setdefault("include-package-data", dist.include_package_data)
    else:
        setuptools_table.setdefault("include-package-data", True)
    # Persist changes:
    asdict["tool"] = tool_table
    tool_table["setuptools"] = setuptools_table

    if "ext-modules" in setuptools_table:
        _ExperimentalConfiguration.emit(subject="[tool.setuptools.ext-modules]")

    with _ignore_errors(ignore_option_errors):
        # Don't complain about unrelated errors (e.g. tools not using the "tool" table)
        subset = {"project": project_table, "tool": {"setuptools": setuptools_table}}
        validate(subset, filepath)

    if expand:
        root_dir = os.path.dirname(filepath)
        return expand_configuration(asdict, root_dir, ignore_option_errors, dist)

    return asdict


def expand_configuration(
    config: dict,
    root_dir: StrPath | None = None,
    ignore_option_errors: bool = False,
    dist: Distribution | None = None,
) -> dict:
    """Given a configuration with unresolved fields (e.g. dynamic, cmdclass, ...)
    find their final values.

    :param dict config: Dict containing the configuration for the distribution
    :param str root_dir: Top-level directory for the distribution/project
        (the same directory where ``pyproject.toml`` is place)
    :param bool ignore_option_errors: see :func:`read_configuration`
    :param Distribution|None: Distribution object to which the configuration refers.
        If not given a dummy object will be created and discarded after the
        configuration is read. Used in the case a dynamic configuration
        (e.g. ``attr`` or ``cmdclass``).

    :rtype: dict
    """
    return _ConfigExpander(config, root_dir, ignore_option_errors, dist).expand()


class _ConfigExpander:
    def __init__(
        self,
        config: dict,
        root_dir: StrPath | None = None,
        ignore_option_errors: bool = False,
        dist: Distribution | None = None,
    ) -> None:
        self.config = config
        self.root_dir = root_dir or os.getcwd()
        self.project_cfg = config.get("project", {})
        self.dynamic = self.project_cfg.get("dynamic", [])
        self.setuptools_cfg = config.get("tool", {}).get("setuptools", {})
        self.dynamic_cfg = self.setuptools_cfg.get("dynamic", {})
        self.ignore_option_errors = ignore_option_errors
        self._dist = dist
        self._referenced_files = set[str]()

    def _ensure_dist(self) -> Distribution:
        from setuptools.dist import Distribution

        attrs = {"src_root": self.root_dir, "name": self.project_cfg.get("name", None)}
        return self._dist or Distribution(attrs)

    def _process_field(self, container: dict, field: str, fn: Callable):
        if field in container:
            with _ignore_errors(self.ignore_option_errors):
                container[field] = fn(container[field])

    def _canonic_package_data(self, field="package-data"):
        package_data = self.setuptools_cfg.get(field, {})
        return _expand.canonic_package_data(package_data)

    def expand(self):
        self._expand_packages()
        self._canonic_package_data()
        self._canonic_package_data("exclude-package-data")

        # A distribution object is required for discovering the correct package_dir
        dist = self._ensure_dist()
        ctx = _EnsurePackagesDiscovered(dist, self.project_cfg, self.setuptools_cfg)
        with ctx as ensure_discovered:
            package_dir = ensure_discovered.package_dir
            self._expand_data_files()
            self._expand_cmdclass(package_dir)
            self._expand_all_dynamic(dist, package_dir)

        dist._referenced_files.update(self._referenced_files)
        return self.config

    def _expand_packages(self):
        packages = self.setuptools_cfg.get("packages")
        if packages is None or isinstance(packages, (list, tuple)):
            return

        find = packages.get("find")
        if isinstance(find, dict):
            find["root_dir"] = self.root_dir
            find["fill_package_dir"] = self.setuptools_cfg.setdefault("package-dir", {})
            with _ignore_errors(self.ignore_option_errors):
                self.setuptools_cfg["packages"] = _expand.find_packages(**find)

    def _expand_data_files(self):
        data_files = partial(_expand.canonic_data_files, root_dir=self.root_dir)
        self._process_field(self.setuptools_cfg, "data-files", data_files)

    def _expand_cmdclass(self, package_dir: Mapping[str, str]):
        root_dir = self.root_dir
        cmdclass = partial(_expand.cmdclass, package_dir=package_dir, root_dir=root_dir)
        self._process_field(self.setuptools_cfg, "cmdclass", cmdclass)

    def _expand_all_dynamic(self, dist: Distribution, package_dir: Mapping[str, str]):
        special = (  # need special handling
            "version",
            "readme",
            "entry-points",
            "scripts",
            "gui-scripts",
            "classifiers",
            "dependencies",
            "optional-dependencies",
        )
        # `_obtain` functions are assumed to raise appropriate exceptions/warnings.
        obtained_dynamic = {
            field: self._obtain(dist, field, package_dir)
            for field in self.dynamic
            if field not in special
        }
        obtained_dynamic.update(
            self._obtain_entry_points(dist, package_dir) or {},
            version=self._obtain_version(dist, package_dir),
            readme=self._obtain_readme(dist),
            classifiers=self._obtain_classifiers(dist),
            dependencies=self._obtain_dependencies(dist),
            optional_dependencies=self._obtain_optional_dependencies(dist),
        )
        # `None` indicates there is nothing in `tool.setuptools.dynamic` but the value
        # might have already been set by setup.py/extensions, so avoid overwriting.
        updates = {k: v for k, v in obtained_dynamic.items() if v is not None}
        self.project_cfg.update(updates)

    def _ensure_previously_set(self, dist: Distribution, field: str):
        previous = _PREVIOUSLY_DEFINED[field](dist)
        if previous is None and not self.ignore_option_errors:
            msg = (
                f"No configuration found for dynamic {field!r}.\n"
                "Some dynamic fields need to be specified via `tool.setuptools.dynamic`"
                "\nothers must be specified via the equivalent attribute in `setup.py`."
            )
            raise InvalidConfigError(msg)

    def _expand_directive(
        self, specifier: str, directive, package_dir: Mapping[str, str]
    ):
        from more_itertools import always_iterable

        with _ignore_errors(self.ignore_option_errors):
            root_dir = self.root_dir
            if "file" in directive:
                self._referenced_files.update(always_iterable(directive["file"]))
                return _expand.read_files(directive["file"], root_dir)
            if "attr" in directive:
                return _expand.read_attr(directive["attr"], package_dir, root_dir)
            raise ValueError(f"invalid `{specifier}`: {directive!r}")
        return None

    def _obtain(self, dist: Distribution, field: str, package_dir: Mapping[str, str]):
        if field in self.dynamic_cfg:
            return self._expand_directive(
                f"tool.setuptools.dynamic.{field}",
                self.dynamic_cfg[field],
                package_dir,
            )
        self._ensure_previously_set(dist, field)
        return None

    def _obtain_version(self, dist: Distribution, package_dir: Mapping[str, str]):
        # Since plugins can set version, let's silently skip if it cannot be obtained
        if "version" in self.dynamic and "version" in self.dynamic_cfg:
            return _expand.version(
                # We already do an early check for the presence of "version"
                self._obtain(dist, "version", package_dir)  # pyright: ignore[reportArgumentType]
            )
        return None

    def _obtain_readme(self, dist: Distribution) -> dict[str, str] | None:
        if "readme" not in self.dynamic:
            return None

        dynamic_cfg = self.dynamic_cfg
        if "readme" in dynamic_cfg:
            return {
                # We already do an early check for the presence of "readme"
                "text": self._obtain(dist, "readme", {}),
                "content-type": dynamic_cfg["readme"].get("content-type", "text/x-rst"),
            }  # pyright: ignore[reportReturnType]

        self._ensure_previously_set(dist, "readme")
        return None

    def _obtain_entry_points(
        self, dist: Distribution, package_dir: Mapping[str, str]
    ) -> dict[str, dict[str, Any]] | None:
        fields = ("entry-points", "scripts", "gui-scripts")
        if not any(field in self.dynamic for field in fields):
            return None

        text = self._obtain(dist, "entry-points", package_dir)
        if text is None:
            return None

        groups = _expand.entry_points(text)
        # Any is str | dict[str, str], but causes variance issues
        expanded: dict[str, dict[str, Any]] = {"entry-points": groups}

        def _set_scripts(field: str, group: str):
            if group in groups:
                value = groups.pop(group)
                if field not in self.dynamic:
                    raise InvalidConfigError(_MissingDynamic.details(field, value))
                expanded[field] = value

        _set_scripts("scripts", "console_scripts")
        _set_scripts("gui-scripts", "gui_scripts")

        return expanded

    def _obtain_classifiers(self, dist: Distribution):
        if "classifiers" in self.dynamic:
            value = self._obtain(dist, "classifiers", {})
            if value:
                return value.splitlines()
        return None

    def _obtain_dependencies(self, dist: Distribution):
        if "dependencies" in self.dynamic:
            value = self._obtain(dist, "dependencies", {})
            if value:
                return _parse_requirements_list(value)
        return None

    def _obtain_optional_dependencies(self, dist: Distribution):
        if "optional-dependencies" not in self.dynamic:
            return None
        if "optional-dependencies" in self.dynamic_cfg:
            optional_dependencies_map = self.dynamic_cfg["optional-dependencies"]
            assert isinstance(optional_dependencies_map, dict)
            return {
                group: _parse_requirements_list(
                    self._expand_directive(
                        f"tool.setuptools.dynamic.optional-dependencies.{group}",
                        directive,
                        {},
                    )
                )
                for group, directive in optional_dependencies_map.items()
            }
        self._ensure_previously_set(dist, "optional-dependencies")
        return None


def _parse_requirements_list(value):
    return [
        line
        for line in value.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


@contextmanager
def _ignore_errors(ignore_option_errors: bool):
    if not ignore_option_errors:
        yield
        return

    try:
        yield
    except Exception as ex:
        _logger.debug(f"ignored error: {ex.__class__.__name__} - {ex}")


class _EnsurePackagesDiscovered(_expand.EnsurePackagesDiscovered):
    def __init__(
        self, distribution: Distribution, project_cfg: dict, setuptools_cfg: dict
    ) -> None:
        super().__init__(distribution)
        self._project_cfg = project_cfg
        self._setuptools_cfg = setuptools_cfg

    def __enter__(self) -> Self:
        """When entering the context, the values of ``packages``, ``py_modules`` and
        ``package_dir`` that are missing in ``dist`` are copied from ``setuptools_cfg``.
        """
        dist, cfg = self._dist, self._setuptools_cfg
        package_dir: dict[str, str] = cfg.setdefault("package-dir", {})
        package_dir.update(dist.package_dir or {})
        dist.package_dir = package_dir  # needs to be the same object

        dist.set_defaults._ignore_ext_modules()  # pyproject.toml-specific behaviour

        # Set `name`, `py_modules` and `packages` in dist to short-circuit
        # auto-discovery, but avoid overwriting empty lists purposefully set by users.
        if dist.metadata.name is None:
            dist.metadata.name = self._project_cfg.get("name")
        if dist.py_modules is None:
            dist.py_modules = cfg.get("py-modules")
        if dist.packages is None:
            dist.packages = cfg.get("packages")

        return super().__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """When exiting the context, if values of ``packages``, ``py_modules`` and
        ``package_dir`` are missing in ``setuptools_cfg``, copy from ``dist``.
        """
        # If anything was discovered set them back, so they count in the final config.
        self._setuptools_cfg.setdefault("packages", self._dist.packages)
        self._setuptools_cfg.setdefault("py-modules", self._dist.py_modules)
        return super().__exit__(exc_type, exc_value, traceback)


class _ExperimentalConfiguration(SetuptoolsWarning):
    _SUMMARY = (
        "`{subject}` in `pyproject.toml` is still *experimental* "
        "and likely to change in future releases."
    )


class _ToolsTypoInMetadata(SetuptoolsWarning):
    _SUMMARY = (
        "Ignoring [tools.setuptools] in pyproject.toml, did you mean [tool.setuptools]?"
    )

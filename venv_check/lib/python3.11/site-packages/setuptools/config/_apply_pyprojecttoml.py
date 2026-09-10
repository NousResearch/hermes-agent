"""Translation layer between pyproject config and setuptools distribution and
metadata objects.

The distribution and metadata objects are modeled after (an old version of)
core metadata, therefore configs in the format specified for ``pyproject.toml``
need to be processed before being applied.

**PRIVATE MODULE**: API reserved for setuptools internal usage only.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from email.headerregistry import Address
from functools import partial, reduce
from inspect import cleandoc
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union

from .. import _static
from .._path import StrPath
from ..errors import InvalidConfigError, RemovedConfigError
from ..extension import Extension
from ..warnings import SetuptoolsDeprecationWarning, SetuptoolsWarning

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from setuptools._importlib import metadata
    from setuptools.dist import Distribution

    from distutils.dist import _OptionsList  # Comes from typeshed


EMPTY: Mapping = MappingProxyType({})  # Immutable dict-like
_ProjectReadmeValue: TypeAlias = Union[str, dict[str, str]]
_Correspondence: TypeAlias = Callable[["Distribution", Any, Union[StrPath, None]], None]
_T = TypeVar("_T")

_logger = logging.getLogger(__name__)


def apply(dist: Distribution, config: dict, filename: StrPath) -> Distribution:
    """Apply configuration dict read with :func:`read_configuration`"""

    if not config:
        return dist  # short-circuit unrelated pyproject.toml file

    root_dir = os.path.dirname(filename) or "."

    _apply_project_table(dist, config, root_dir)
    _apply_tool_table(dist, config, filename)

    current_directory = os.getcwd()
    os.chdir(root_dir)
    try:
        dist._finalize_requires()
        dist._finalize_license_expression()
        dist._finalize_license_files()
    finally:
        os.chdir(current_directory)

    return dist


def _apply_project_table(dist: Distribution, config: dict, root_dir: StrPath):
    orig_config = config.get("project", {})
    if not orig_config:
        return  # short-circuit

    project_table = {k: _static.attempt_conversion(v) for k, v in orig_config.items()}
    _handle_missing_dynamic(dist, project_table)
    _unify_entry_points(project_table)

    for field, value in project_table.items():
        norm_key = json_compatible_key(field)
        corresp = PYPROJECT_CORRESPONDENCE.get(norm_key, norm_key)
        if callable(corresp):
            corresp(dist, value, root_dir)
        else:
            _set_config(dist, corresp, value)


def _apply_tool_table(dist: Distribution, config: dict, filename: StrPath):
    tool_table = config.get("tool", {}).get("setuptools", {})
    if not tool_table:
        return  # short-circuit

    if "license-files" in tool_table:
        if "license-files" in config.get("project", {}):
            # https://github.com/pypa/setuptools/pull/4837#discussion_r2004983349
            raise InvalidConfigError(
                "'project.license-files' is defined already. "
                "Remove 'tool.setuptools.license-files'."
            )

        pypa_guides = "guides/writing-pyproject-toml/#license-files"
        SetuptoolsDeprecationWarning.emit(
            "'tool.setuptools.license-files' is deprecated in favor of "
            "'project.license-files' (available on setuptools>=77.0.0).",
            see_url=f"https://packaging.python.org/en/latest/{pypa_guides}",
            due_date=(2026, 2, 18),  # Warning introduced on 2025-02-18
        )

    for field, value in tool_table.items():
        norm_key = json_compatible_key(field)

        if norm_key in TOOL_TABLE_REMOVALS:
            suggestion = cleandoc(TOOL_TABLE_REMOVALS[norm_key])
            msg = f"""
            The parameter `tool.setuptools.{field}` was long deprecated
            and has been removed from `pyproject.toml`.
            """
            raise RemovedConfigError("\n".join([cleandoc(msg), suggestion]))

        norm_key = TOOL_TABLE_RENAMES.get(norm_key, norm_key)
        corresp = TOOL_TABLE_CORRESPONDENCE.get(norm_key, norm_key)
        if callable(corresp):
            corresp(dist, value)
        else:
            _set_config(dist, corresp, value)

    _copy_command_options(config, dist, filename)


def _handle_missing_dynamic(dist: Distribution, project_table: dict):
    """Be temporarily forgiving with ``dynamic`` fields not listed in ``dynamic``"""
    dynamic = set(project_table.get("dynamic", []))
    for field, getter in _PREVIOUSLY_DEFINED.items():
        if not (field in project_table or field in dynamic):
            value = getter(dist)
            if value:
                _MissingDynamic.emit(field=field, value=value)
                project_table[field] = _RESET_PREVIOUSLY_DEFINED.get(field)


def json_compatible_key(key: str) -> str:
    """As defined in :pep:`566#json-compatible-metadata`"""
    return key.lower().replace("-", "_")


def _set_config(dist: Distribution, field: str, value: Any):
    val = _PREPROCESS.get(field, _noop)(dist, value)
    setter = getattr(dist.metadata, f"set_{field}", None)
    if setter:
        setter(val)
    elif hasattr(dist.metadata, field) or field in SETUPTOOLS_PATCHES:
        setattr(dist.metadata, field, val)
    else:
        setattr(dist, field, val)


_CONTENT_TYPES = {
    ".md": "text/markdown",
    ".rst": "text/x-rst",
    ".txt": "text/plain",
}


def _guess_content_type(file: str) -> str | None:
    _, ext = os.path.splitext(file.lower())
    if not ext:
        return None

    if ext in _CONTENT_TYPES:
        return _static.Str(_CONTENT_TYPES[ext])

    valid = ", ".join(f"{k} ({v})" for k, v in _CONTENT_TYPES.items())
    msg = f"only the following file extensions are recognized: {valid}."
    raise ValueError(f"Undefined content type for {file}, {msg}")


def _long_description(
    dist: Distribution, val: _ProjectReadmeValue, root_dir: StrPath | None
):
    from setuptools.config import expand

    file: str | tuple[()]
    if isinstance(val, str):
        file = val
        text = expand.read_files(file, root_dir)
        ctype = _guess_content_type(file)
    else:
        file = val.get("file") or ()
        text = val.get("text") or expand.read_files(file, root_dir)
        ctype = val["content-type"]

    # XXX: Is it completely safe to assume static?
    _set_config(dist, "long_description", _static.Str(text))

    if ctype:
        _set_config(dist, "long_description_content_type", _static.Str(ctype))

    if file:
        dist._referenced_files.add(file)


def _license(dist: Distribution, val: str | dict, root_dir: StrPath | None):
    from setuptools.config import expand

    if isinstance(val, str):
        if getattr(dist.metadata, "license", None):
            SetuptoolsWarning.emit("`license` overwritten by `pyproject.toml`")
            dist.metadata.license = None
        _set_config(dist, "license_expression", _static.Str(val))
    else:
        pypa_guides = "guides/writing-pyproject-toml/#license"
        SetuptoolsDeprecationWarning.emit(
            "`project.license` as a TOML table is deprecated",
            "Please use a simple string containing a SPDX expression for "
            "`project.license`. You can also use `project.license-files`. "
            "(Both options available on setuptools>=77.0.0).",
            see_url=f"https://packaging.python.org/en/latest/{pypa_guides}",
            due_date=(2026, 2, 18),  # Introduced on 2025-02-18
        )
        if "file" in val:
            # XXX: Is it completely safe to assume static?
            value = expand.read_files([val["file"]], root_dir)
            _set_config(dist, "license", _static.Str(value))
            dist._referenced_files.add(val["file"])
        else:
            _set_config(dist, "license", _static.Str(val["text"]))


def _people(dist: Distribution, val: list[dict], _root_dir: StrPath | None, kind: str):
    field = []
    email_field = []
    for person in val:
        if "name" not in person:
            email_field.append(person["email"])
        elif "email" not in person:
            field.append(person["name"])
        else:
            addr = Address(display_name=person["name"], addr_spec=person["email"])
            email_field.append(str(addr))

    if field:
        _set_config(dist, kind, _static.Str(", ".join(field)))
    if email_field:
        _set_config(dist, f"{kind}_email", _static.Str(", ".join(email_field)))


def _project_urls(dist: Distribution, val: dict, _root_dir: StrPath | None):
    _set_config(dist, "project_urls", val)


def _python_requires(dist: Distribution, val: str, _root_dir: StrPath | None):
    _set_config(dist, "python_requires", _static.SpecifierSet(val))


def _dependencies(dist: Distribution, val: list, _root_dir: StrPath | None):
    if getattr(dist, "install_requires", []):
        msg = "`install_requires` overwritten in `pyproject.toml` (dependencies)"
        SetuptoolsWarning.emit(msg)
    dist.install_requires = val


def _optional_dependencies(dist: Distribution, val: dict, _root_dir: StrPath | None):
    if getattr(dist, "extras_require", None):
        msg = "`extras_require` overwritten in `pyproject.toml` (optional-dependencies)"
        SetuptoolsWarning.emit(msg)
    dist.extras_require = val


def _ext_modules(dist: Distribution, val: list[dict]) -> list[Extension]:
    existing = dist.ext_modules or []
    args = ({k.replace("-", "_"): v for k, v in x.items()} for x in val)
    new = [Extension(**kw) for kw in args]
    return [*existing, *new]


def _noop(_dist: Distribution, val: _T) -> _T:
    return val


def _identity(val: _T) -> _T:
    return val


def _unify_entry_points(project_table: dict):
    project = project_table
    given = project.pop("entry-points", project.pop("entry_points", {}))
    entry_points = dict(given)  # Avoid problems with static
    renaming = {"scripts": "console_scripts", "gui_scripts": "gui_scripts"}
    for key, value in list(project.items()):  # eager to allow modifications
        norm_key = json_compatible_key(key)
        if norm_key in renaming:
            # Don't skip even if value is empty (reason: reset missing `dynamic`)
            entry_points[renaming[norm_key]] = project.pop(key)

    if entry_points:
        project["entry-points"] = {
            name: [f"{k} = {v}" for k, v in group.items()]
            for name, group in entry_points.items()
            if group  # now we can skip empty groups
        }
        # Sometimes this will set `project["entry-points"] = {}`, and that is
        # intentional (for resetting configurations that are missing `dynamic`).


def _copy_command_options(pyproject: dict, dist: Distribution, filename: StrPath):
    tool_table = pyproject.get("tool", {})
    cmdclass = tool_table.get("setuptools", {}).get("cmdclass", {})
    valid_options = _valid_command_options(cmdclass)

    cmd_opts = dist.command_options
    for cmd, config in pyproject.get("tool", {}).get("distutils", {}).items():
        cmd = json_compatible_key(cmd)
        valid = valid_options.get(cmd, set())
        cmd_opts.setdefault(cmd, {})
        for key, value in config.items():
            key = json_compatible_key(key)
            cmd_opts[cmd][key] = (str(filename), value)
            if key not in valid:
                # To avoid removing options that are specified dynamically we
                # just log a warn...
                _logger.warning(f"Command option {cmd}.{key} is not defined")


def _valid_command_options(cmdclass: Mapping = EMPTY) -> dict[str, set[str]]:
    from setuptools.dist import Distribution

    from .._importlib import metadata

    valid_options = {"global": _normalise_cmd_options(Distribution.global_options)}

    unloaded_entry_points = metadata.entry_points(group='distutils.commands')
    loaded_entry_points = (_load_ep(ep) for ep in unloaded_entry_points)
    entry_points = (ep for ep in loaded_entry_points if ep)
    for cmd, cmd_class in chain(entry_points, cmdclass.items()):
        opts = valid_options.get(cmd, set())
        opts = opts | _normalise_cmd_options(getattr(cmd_class, "user_options", []))
        valid_options[cmd] = opts

    return valid_options


def _load_ep(ep: metadata.EntryPoint) -> tuple[str, type] | None:
    if ep.value.startswith("wheel.bdist_wheel"):
        # Ignore deprecated entrypoint from wheel and avoid warning pypa/wheel#631
        # TODO: remove check when `bdist_wheel` has been fully removed from pypa/wheel
        return None

    # Ignore all the errors
    try:
        return (ep.name, ep.load())
    except Exception as ex:
        msg = f"{ex.__class__.__name__} while trying to load entry-point {ep.name}"
        _logger.warning(f"{msg}: {ex}")
        return None


def _normalise_cmd_option_key(name: str) -> str:
    return json_compatible_key(name).strip("_=")


def _normalise_cmd_options(desc: _OptionsList) -> set[str]:
    return {_normalise_cmd_option_key(fancy_option[0]) for fancy_option in desc}


def _get_previous_entrypoints(dist: Distribution) -> dict[str, list]:
    ignore = ("console_scripts", "gui_scripts")
    value = getattr(dist, "entry_points", None) or {}
    return {k: v for k, v in value.items() if k not in ignore}


def _get_previous_scripts(dist: Distribution) -> list | None:
    value = getattr(dist, "entry_points", None) or {}
    return value.get("console_scripts")


def _get_previous_gui_scripts(dist: Distribution) -> list | None:
    value = getattr(dist, "entry_points", None) or {}
    return value.get("gui_scripts")


def _set_static_list_metadata(attr: str, dist: Distribution, val: list) -> None:
    """Apply distutils metadata validation but preserve "static" behaviour"""
    meta = dist.metadata
    setter, getter = getattr(meta, f"set_{attr}"), getattr(meta, f"get_{attr}")
    setter(val)
    setattr(meta, attr, _static.List(getter()))


def _attrgetter(attr):
    """
    Similar to ``operator.attrgetter`` but returns None if ``attr`` is not found
    >>> from types import SimpleNamespace
    >>> obj = SimpleNamespace(a=42, b=SimpleNamespace(c=13))
    >>> _attrgetter("a")(obj)
    42
    >>> _attrgetter("b.c")(obj)
    13
    >>> _attrgetter("d")(obj) is None
    True
    """
    return partial(reduce, lambda acc, x: getattr(acc, x, None), attr.split("."))


def _some_attrgetter(*items):
    """
    Return the first "truth-y" attribute or None
    >>> from types import SimpleNamespace
    >>> obj = SimpleNamespace(a=42, b=SimpleNamespace(c=13))
    >>> _some_attrgetter("d", "a", "b.c")(obj)
    42
    >>> _some_attrgetter("d", "e", "b.c", "a")(obj)
    13
    >>> _some_attrgetter("d", "e", "f")(obj) is None
    True
    """

    def _acessor(obj):
        values = (_attrgetter(i)(obj) for i in items)
        return next((i for i in values if i is not None), None)

    return _acessor


PYPROJECT_CORRESPONDENCE: dict[str, _Correspondence] = {
    "readme": _long_description,
    "license": _license,
    "authors": partial(_people, kind="author"),
    "maintainers": partial(_people, kind="maintainer"),
    "urls": _project_urls,
    "dependencies": _dependencies,
    "optional_dependencies": _optional_dependencies,
    "requires_python": _python_requires,
}

TOOL_TABLE_RENAMES = {"script_files": "scripts"}
TOOL_TABLE_REMOVALS = {
    "namespace_packages": """
        Please migrate to implicit native namespaces instead.
        See https://packaging.python.org/en/latest/guides/packaging-namespace-packages/.
        """,
}
TOOL_TABLE_CORRESPONDENCE = {
    # Fields with corresponding core metadata need to be marked as static:
    "obsoletes": partial(_set_static_list_metadata, "obsoletes"),
    "provides": partial(_set_static_list_metadata, "provides"),
    "platforms": partial(_set_static_list_metadata, "platforms"),
}

SETUPTOOLS_PATCHES = {
    "long_description_content_type",
    "project_urls",
    "provides_extras",
    "license_file",
    "license_files",
    "license_expression",
}

_PREPROCESS = {
    "ext_modules": _ext_modules,
}

_PREVIOUSLY_DEFINED = {
    "name": _attrgetter("metadata.name"),
    "version": _attrgetter("metadata.version"),
    "description": _attrgetter("metadata.description"),
    "readme": _attrgetter("metadata.long_description"),
    "requires-python": _some_attrgetter("python_requires", "metadata.python_requires"),
    "license": _some_attrgetter("metadata.license_expression", "metadata.license"),
    # XXX: `license-file` is currently not considered in the context of `dynamic`.
    #      See TestPresetField.test_license_files_exempt_from_dynamic
    "authors": _some_attrgetter("metadata.author", "metadata.author_email"),
    "maintainers": _some_attrgetter("metadata.maintainer", "metadata.maintainer_email"),
    "keywords": _attrgetter("metadata.keywords"),
    "classifiers": _attrgetter("metadata.classifiers"),
    "urls": _attrgetter("metadata.project_urls"),
    "entry-points": _get_previous_entrypoints,
    "scripts": _get_previous_scripts,
    "gui-scripts": _get_previous_gui_scripts,
    "dependencies": _attrgetter("install_requires"),
    "optional-dependencies": _attrgetter("extras_require"),
}


_RESET_PREVIOUSLY_DEFINED: dict = {
    # Fix improper setting: given in `setup.py`, but not listed in `dynamic`
    # Use "immutable" data structures to avoid in-place modification.
    # dict: pyproject name => value to which reset
    "license": "",
    # XXX: `license-file` is currently not considered in the context of `dynamic`.
    #      See TestPresetField.test_license_files_exempt_from_dynamic
    "authors": _static.EMPTY_LIST,
    "maintainers": _static.EMPTY_LIST,
    "keywords": _static.EMPTY_LIST,
    "classifiers": _static.EMPTY_LIST,
    "urls": _static.EMPTY_DICT,
    "entry-points": _static.EMPTY_DICT,
    "scripts": _static.EMPTY_DICT,
    "gui-scripts": _static.EMPTY_DICT,
    "dependencies": _static.EMPTY_LIST,
    "optional-dependencies": _static.EMPTY_DICT,
}


class _MissingDynamic(SetuptoolsWarning):
    _SUMMARY = "`{field}` defined outside of `pyproject.toml` is ignored."

    _DETAILS = """
    The following seems to be defined outside of `pyproject.toml`:

    `{field} = {value!r}`

    According to the spec (see the link below), however, setuptools CANNOT
    consider this value unless `{field}` is listed as `dynamic`.

    https://packaging.python.org/en/latest/specifications/pyproject-toml/#declaring-project-metadata-the-project-table

    To prevent this problem, you can list `{field}` under `dynamic` or alternatively
    remove the `[project]` table from your file and rely entirely on other means of
    configuration.
    """
    # TODO: Consider removing this check in the future?
    #       There is a trade-off here between improving "debug-ability" and the cost
    #       of running/testing/maintaining these unnecessary checks...

    @classmethod
    def details(cls, field: str, value: Any) -> str:
        return cls._DETAILS.format(field=field, value=value)

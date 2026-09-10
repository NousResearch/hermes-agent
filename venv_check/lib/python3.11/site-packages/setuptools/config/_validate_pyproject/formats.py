"""
The functions in this module are used to validate schemas with the
`format JSON Schema keyword
<https://json-schema.org/understanding-json-schema/reference/string#format>`_.

The correspondence is given by replacing the ``_`` character in the name of the
function with a ``-`` to obtain the format name and vice versa.
"""

import builtins
import logging
import os
import re
import string
import typing
from itertools import chain as _chain

if typing.TYPE_CHECKING:
    from typing_extensions import Literal

_logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------
# PEP 440

VERSION_PATTERN = r"""
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?
            (?P<pre_l>alpha|a|beta|b|preview|pre|c|rc)
            [-_\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)
                [-_\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          # dev release
            [-_\.]?
            (?P<dev_l>dev)
            [-_\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
"""

VERSION_REGEX = re.compile(r"^\s*" + VERSION_PATTERN + r"\s*$", re.X | re.I)


def pep440(version: str) -> bool:
    """See :ref:`PyPA's version specification <pypa:version-specifiers>`
    (initially introduced in :pep:`440`).
    """
    return VERSION_REGEX.match(version) is not None


# -------------------------------------------------------------------------------------
# PEP 508

PEP508_IDENTIFIER_PATTERN = r"([A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9])"
PEP508_IDENTIFIER_REGEX = re.compile(f"^{PEP508_IDENTIFIER_PATTERN}$", re.I)


def pep508_identifier(name: str) -> bool:
    """See :ref:`PyPA's name specification <pypa:name-format>`
    (initially introduced in :pep:`508#names`).
    """
    return PEP508_IDENTIFIER_REGEX.match(name) is not None


try:
    try:
        from packaging import requirements as _req
    except ImportError:  # pragma: no cover
        # let's try setuptools vendored version
        from setuptools._vendor.packaging import (  # type: ignore[no-redef]
            requirements as _req,
        )

    def pep508(value: str) -> bool:
        """See :ref:`PyPA's dependency specifiers <pypa:dependency-specifiers>`
        (initially introduced in :pep:`508`).
        """
        try:
            _req.Requirement(value)
            return True
        except _req.InvalidRequirement:
            return False

except ImportError:  # pragma: no cover
    _logger.warning(
        "Could not find an installation of `packaging`. Requirements, dependencies and "
        "versions might not be validated. "
        "To enforce validation, please install `packaging`."
    )

    def pep508(value: str) -> bool:
        return True


def pep508_versionspec(value: str) -> bool:
    """Expression that can be used to specify/lock versions (including ranges)
    See ``versionspec`` in :ref:`PyPA's dependency specifiers
    <pypa:dependency-specifiers>` (initially introduced in :pep:`508`).
    """
    if any(c in value for c in (";", "]", "@")):
        # In PEP 508:
        # conditional markers, extras and URL specs are not included in the
        # versionspec
        return False
    # Let's pretend we have a dependency called `requirement` with the given
    # version spec, then we can reuse the pep508 function for validation:
    return pep508(f"requirement{value}")


# -------------------------------------------------------------------------------------
# PEP 517


def pep517_backend_reference(value: str) -> bool:
    """See PyPA's specification for defining build-backend references
    introduced in :pep:`517#source-trees`.

    This is similar to an entry-point reference (e.g., ``package.module:object``).
    """
    module, _, obj = value.partition(":")
    identifiers = (i.strip() for i in _chain(module.split("."), obj.split(".")))
    return all(python_identifier(i) for i in identifiers if i)


# -------------------------------------------------------------------------------------
# Classifiers - PEP 301


def _download_classifiers() -> str:
    import ssl
    from email.message import Message
    from urllib.request import urlopen

    url = "https://pypi.org/pypi?:action=list_classifiers"
    context = ssl.create_default_context()
    with urlopen(url, context=context) as response:  # noqa: S310 (audit URLs)
        headers = Message()
        headers["content_type"] = response.getheader("content-type", "text/plain")
        return response.read().decode(headers.get_param("charset", "utf-8"))  # type: ignore[no-any-return]


class _TroveClassifier:
    """The ``trove_classifiers`` package is the official way of validating classifiers,
    however this package might not be always available.
    As a workaround we can still download a list from PyPI.
    We also don't want to be over strict about it, so simply skipping silently is an
    option (classifiers will be validated anyway during the upload to PyPI).
    """

    downloaded: typing.Union[None, "Literal[False]", typing.Set[str]]
    """
    None => not cached yet
    False => unavailable
    set => cached values
    """

    def __init__(self) -> None:
        self.downloaded = None
        self._skip_download = False
        self.__name__ = "trove_classifier"  # Emulate a public function

    def _disable_download(self) -> None:
        # This is a private API. Only setuptools has the consent of using it.
        self._skip_download = True

    def __call__(self, value: str) -> bool:
        if self.downloaded is False or self._skip_download is True:
            return True

        if os.getenv("NO_NETWORK") or os.getenv("VALIDATE_PYPROJECT_NO_NETWORK"):
            self.downloaded = False
            msg = (
                "Install ``trove-classifiers`` to ensure proper validation. "
                "Skipping download of classifiers list from PyPI (NO_NETWORK)."
            )
            _logger.debug(msg)
            return True

        if self.downloaded is None:
            msg = (
                "Install ``trove-classifiers`` to ensure proper validation. "
                "Meanwhile a list of classifiers will be downloaded from PyPI."
            )
            _logger.debug(msg)
            try:
                self.downloaded = set(_download_classifiers().splitlines())
            except Exception:
                self.downloaded = False
                _logger.debug("Problem with download, skipping validation")
                return True

        return value in self.downloaded or value.lower().startswith("private ::")


try:
    from trove_classifiers import classifiers as _trove_classifiers

    def trove_classifier(value: str) -> bool:
        """See https://pypi.org/classifiers/"""
        return value in _trove_classifiers or value.lower().startswith("private ::")

except ImportError:  # pragma: no cover
    trove_classifier = _TroveClassifier()


# -------------------------------------------------------------------------------------
# Stub packages - PEP 561


def pep561_stub_name(value: str) -> bool:
    """Name of a directory containing type stubs.
    It must follow the name scheme ``<package>-stubs`` as defined in
    :pep:`561#stub-only-packages`.
    """
    top, *children = value.split(".")
    if not top.endswith("-stubs"):
        return False
    return python_module_name(".".join([top[: -len("-stubs")], *children]))


# -------------------------------------------------------------------------------------
# Non-PEP related


def url(value: str) -> bool:
    """Valid URL (validation uses :obj:`urllib.parse`).
    For maximum compatibility please make sure to include a ``scheme`` prefix
    in your URL (e.g. ``http://``).
    """
    from urllib.parse import urlparse

    try:
        parts = urlparse(value)
        if not parts.scheme:
            _logger.warning(
                "For maximum compatibility please make sure to include a "
                "`scheme` prefix in your URL (e.g. 'http://'). "
                f"Given value: {value}"
            )
            if not (value.startswith("/") or value.startswith("\\") or "@" in value):
                parts = urlparse(f"http://{value}")

        return bool(parts.scheme and parts.netloc)
    except Exception:
        return False


# https://packaging.python.org/specifications/entry-points/
ENTRYPOINT_PATTERN = r"[^\[\s=]([^=]*[^\s=])?"
ENTRYPOINT_REGEX = re.compile(f"^{ENTRYPOINT_PATTERN}$", re.I)
RECOMMEDED_ENTRYPOINT_PATTERN = r"[\w.-]+"
RECOMMEDED_ENTRYPOINT_REGEX = re.compile(f"^{RECOMMEDED_ENTRYPOINT_PATTERN}$", re.I)
ENTRYPOINT_GROUP_PATTERN = r"\w+(\.\w+)*"
ENTRYPOINT_GROUP_REGEX = re.compile(f"^{ENTRYPOINT_GROUP_PATTERN}$", re.I)


def python_identifier(value: str) -> bool:
    """Can be used as identifier in Python.
    (Validation uses :obj:`str.isidentifier`).
    """
    return value.isidentifier()


def python_qualified_identifier(value: str) -> bool:
    """
    Python "dotted identifier", i.e. a sequence of :obj:`python_identifier`
    concatenated with ``"."`` (e.g.: ``package.module.submodule``).
    """
    if value.startswith(".") or value.endswith("."):
        return False
    return all(python_identifier(m) for m in value.split("."))


def python_module_name(value: str) -> bool:
    """Module name that can be used in an ``import``-statement in Python.
    See :obj:`python_qualified_identifier`.
    """
    return python_qualified_identifier(value)


def python_module_name_relaxed(value: str) -> bool:
    """Similar to :obj:`python_module_name`, but relaxed to also accept
    dash characters (``-``) and cover special cases like ``pip-run``.

    It is recommended, however, that beginners avoid dash characters,
    as they require advanced knowledge about Python internals.

    The following are disallowed:

    * names starting/ending in dashes,
    * names ending in ``-stubs`` (potentially collide with :obj:`pep561_stub_name`).
    """
    if value.startswith("-") or value.endswith("-"):
        return False
    if value.endswith("-stubs"):
        return False  # Avoid collision with PEP 561
    return python_module_name(value.replace("-", "_"))


def python_entrypoint_group(value: str) -> bool:
    """See ``Data model > group`` in the :ref:`PyPA's entry-points specification
    <pypa:entry-points>`.
    """
    return ENTRYPOINT_GROUP_REGEX.match(value) is not None


def python_entrypoint_name(value: str) -> bool:
    """See ``Data model > name`` in the :ref:`PyPA's entry-points specification
    <pypa:entry-points>`.
    """
    if not ENTRYPOINT_REGEX.match(value):
        return False
    if not RECOMMEDED_ENTRYPOINT_REGEX.match(value):
        msg = f"Entry point `{value}` does not follow recommended pattern: "
        msg += RECOMMEDED_ENTRYPOINT_PATTERN
        _logger.warning(msg)
    return True


def python_entrypoint_reference(value: str) -> bool:
    """Reference to a Python object using in the format::

        importable.module:object.attr

    See ``Data model >object reference`` in the :ref:`PyPA's entry-points specification
    <pypa:entry-points>`.
    """
    module, _, rest = value.partition(":")
    if "[" in rest:
        obj, _, extras_ = rest.partition("[")
        if extras_.strip()[-1] != "]":
            return False
        extras = (x.strip() for x in extras_.strip(string.whitespace + "[]").split(","))
        if not all(pep508_identifier(e) for e in extras):
            return False
        _logger.warning(f"`{value}` - using extras for entry points is not recommended")
    else:
        obj = rest

    module_parts = module.split(".")
    identifiers = _chain(module_parts, obj.split(".")) if rest else iter(module_parts)
    return all(python_identifier(i.strip()) for i in identifiers)


def uint8(value: builtins.int) -> bool:
    r"""Unsigned 8-bit integer (:math:`0 \leq x < 2^8`)"""
    return 0 <= value < 2**8


def uint16(value: builtins.int) -> bool:
    r"""Unsigned 16-bit integer (:math:`0 \leq x < 2^{16}`)"""
    return 0 <= value < 2**16


def uint(value: builtins.int) -> bool:
    r"""Unsigned 64-bit integer (:math:`0 \leq x < 2^{64}`)"""
    return 0 <= value < 2**64


def int(value: builtins.int) -> bool:
    r"""Signed 64-bit integer (:math:`-2^{63} \leq x < 2^{63}`)"""
    return -(2**63) <= value < 2**63


try:
    from packaging import licenses as _licenses

    def SPDX(value: str) -> bool:
        """See :ref:`PyPA's License-Expression specification
        <pypa:core-metadata-license-expression>` (added in :pep:`639`).
        """
        try:
            _licenses.canonicalize_license_expression(value)
            return True
        except _licenses.InvalidLicenseExpression:
            return False

except ImportError:  # pragma: no cover
    _logger.warning(
        "Could not find an up-to-date installation of `packaging`. "
        "License expressions might not be validated. "
        "To enforce validation, please install `packaging>=24.2`."
    )

    def SPDX(value: str) -> bool:
        return True

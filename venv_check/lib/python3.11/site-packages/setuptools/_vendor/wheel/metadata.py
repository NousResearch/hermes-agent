"""
Tools for converting old- to new-style metadata.
"""

from __future__ import annotations

import functools
import itertools
import os.path
import re
import textwrap
from email.message import Message
from email.parser import Parser
from typing import Generator, Iterable, Iterator, Literal

from .vendored.packaging.requirements import Requirement


def _nonblank(str: str) -> bool | Literal[""]:
    return str and not str.startswith("#")


@functools.singledispatch
def yield_lines(iterable: Iterable[str]) -> Iterator[str]:
    r"""
    Yield valid lines of a string or iterable.
    >>> list(yield_lines(''))
    []
    >>> list(yield_lines(['foo', 'bar']))
    ['foo', 'bar']
    >>> list(yield_lines('foo\nbar'))
    ['foo', 'bar']
    >>> list(yield_lines('\nfoo\n#bar\nbaz #comment'))
    ['foo', 'baz #comment']
    >>> list(yield_lines(['foo\nbar', 'baz', 'bing\n\n\n']))
    ['foo', 'bar', 'baz', 'bing']
    """
    return itertools.chain.from_iterable(map(yield_lines, iterable))


@yield_lines.register(str)
def _(text: str) -> Iterator[str]:
    return filter(_nonblank, map(str.strip, text.splitlines()))


def split_sections(
    s: str | Iterator[str],
) -> Generator[tuple[str | None, list[str]], None, None]:
    """Split a string or iterable thereof into (section, content) pairs
    Each ``section`` is a stripped version of the section header ("[section]")
    and each ``content`` is a list of stripped lines excluding blank lines and
    comment-only lines.  If there are any such lines before the first section
    header, they're returned in a first ``section`` of ``None``.
    """
    section = None
    content: list[str] = []
    for line in yield_lines(s):
        if line.startswith("["):
            if line.endswith("]"):
                if section or content:
                    yield section, content
                section = line[1:-1].strip()
                content = []
            else:
                raise ValueError("Invalid section heading", line)
        else:
            content.append(line)

    # wrap up last segment
    yield section, content


def safe_extra(extra: str) -> str:
    """Convert an arbitrary string to a standard 'extra' name
    Any runs of non-alphanumeric characters are replaced with a single '_',
    and the result is always lowercased.
    """
    return re.sub("[^A-Za-z0-9.-]+", "_", extra).lower()


def safe_name(name: str) -> str:
    """Convert an arbitrary string to a standard distribution name
    Any runs of non-alphanumeric/. characters are replaced with a single '-'.
    """
    return re.sub("[^A-Za-z0-9.]+", "-", name)


def requires_to_requires_dist(requirement: Requirement) -> str:
    """Return the version specifier for a requirement in PEP 345/566 fashion."""
    if requirement.url:
        return " @ " + requirement.url

    requires_dist: list[str] = []
    for spec in requirement.specifier:
        requires_dist.append(spec.operator + spec.version)

    if requires_dist:
        return " " + ",".join(sorted(requires_dist))
    else:
        return ""


def convert_requirements(requirements: list[str]) -> Iterator[str]:
    """Yield Requires-Dist: strings for parsed requirements strings."""
    for req in requirements:
        parsed_requirement = Requirement(req)
        spec = requires_to_requires_dist(parsed_requirement)
        extras = ",".join(sorted(safe_extra(e) for e in parsed_requirement.extras))
        if extras:
            extras = f"[{extras}]"

        yield safe_name(parsed_requirement.name) + extras + spec


def generate_requirements(
    extras_require: dict[str | None, list[str]],
) -> Iterator[tuple[str, str]]:
    """
    Convert requirements from a setup()-style dictionary to
    ('Requires-Dist', 'requirement') and ('Provides-Extra', 'extra') tuples.

    extras_require is a dictionary of {extra: [requirements]} as passed to setup(),
    using the empty extra {'': [requirements]} to hold install_requires.
    """
    for extra, depends in extras_require.items():
        condition = ""
        extra = extra or ""
        if ":" in extra:  # setuptools extra:condition syntax
            extra, condition = extra.split(":", 1)

        extra = safe_extra(extra)
        if extra:
            yield "Provides-Extra", extra
            if condition:
                condition = "(" + condition + ") and "
            condition += f"extra == '{extra}'"

        if condition:
            condition = " ; " + condition

        for new_req in convert_requirements(depends):
            canonical_req = str(Requirement(new_req + condition))
            yield "Requires-Dist", canonical_req


def pkginfo_to_metadata(egg_info_path: str, pkginfo_path: str) -> Message:
    """
    Convert .egg-info directory with PKG-INFO to the Metadata 2.1 format
    """
    with open(pkginfo_path, encoding="utf-8") as headers:
        pkg_info = Parser().parse(headers)

    pkg_info.replace_header("Metadata-Version", "2.1")
    # Those will be regenerated from `requires.txt`.
    del pkg_info["Provides-Extra"]
    del pkg_info["Requires-Dist"]
    requires_path = os.path.join(egg_info_path, "requires.txt")
    if os.path.exists(requires_path):
        with open(requires_path, encoding="utf-8") as requires_file:
            requires = requires_file.read()

        parsed_requirements = sorted(split_sections(requires), key=lambda x: x[0] or "")
        for extra, reqs in parsed_requirements:
            for key, value in generate_requirements({extra: reqs}):
                if (key, value) not in pkg_info.items():
                    pkg_info[key] = value

    description = pkg_info["Description"]
    if description:
        description_lines = pkg_info["Description"].splitlines()
        dedented_description = "\n".join(
            # if the first line of long_description is blank,
            # the first line here will be indented.
            (
                description_lines[0].lstrip(),
                textwrap.dedent("\n".join(description_lines[1:])),
                "\n",
            )
        )
        pkg_info.set_payload(dedented_description)
        del pkg_info["Description"]

    return pkg_info

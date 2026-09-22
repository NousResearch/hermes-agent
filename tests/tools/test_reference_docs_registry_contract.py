"""Contract: the shipped tool reference docs name only tools the shipped code registers.

``website/docs/reference/tools-reference.md`` and ``toolsets-reference.md`` are the
authoritative built-in-tool surface published with the package. This pins them to the
live registry: every tool name the docs claim must (a) be a registered tool and (b)
resolve inside the toolset that claims it — the registry/queryable half of the
"documented surface == registered surface" contract. A rename in ``tools/`` or
``toolsets.py`` that skips these pages fails here instead of shipping stale tool
names to readers (they shipped stale once: the project/preview/tour/tip
consolidations left ``project_create``/``open_preview``/``tour``/``tip`` behind, and
the browser row still claimed the ``web_search`` membership #64503 removed).

Reads only the two shipped .md docs (data, like cli-config.yaml.example contracts)
and the live registry/toolset resolver — never source text.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_REF = REPO_ROOT / "website" / "docs" / "reference" / "tools-reference.md"
TOOLSETS_REF = REPO_ROOT / "website" / "docs" / "reference" / "toolsets-reference.md"

_SECTION_RE = re.compile(r"^## `([a-z0-9_-]+)` toolset", re.MULTILINE)
_ROW_RE = re.compile(r"^\| `([a-z0-9_]+)` \|", re.MULTILINE)


@pytest.fixture(scope="module")
def registered_names():
    """Every tool name the shipped surface registers: built-ins plus bundled
    backend plugins (the spotify plugin owns the doc's spotify rows). The same
    discovery path production runs at startup."""
    import model_tools  # noqa: F401 — triggers built-in tool discovery
    from hermes_cli.plugins import discover_plugins

    discover_plugins()
    from tools.registry import registry

    return {entry.name for entry in registry.get_all_entries()}


def _tool_sections(path) -> list[tuple[str, list[str]]]:
    """(toolset, claimed tool names) per ``## ``toolset`` toolset`` section."""
    text = path.read_text(encoding="utf-8")
    headers = [(m.start(), m.group(1)) for m in _SECTION_RE.finditer(text)]
    return [
        (toolset, _ROW_RE.findall(text[start:next_start]))
        for (start, toolset), (next_start, _) in zip(headers, [*headers[1:], (len(text), None)])
    ]


def _core_toolset_rows(path) -> list[tuple[str, str]]:
    """(toolset, Tools cell) rows of the Core Toolsets table."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^## Core Toolsets$.*?^(?=\| `)", text, re.MULTILINE | re.DOTALL)
    assert match, "Core Toolsets table vanished from toolsets-reference.md"
    section = text[match.start() : text.index("## Platform Toolsets", match.start())]
    return [
        (cells[0].strip("` "), cells[1])
        for line in section.splitlines()
        if line.startswith("| `") and len(cells := [c.strip() for c in line.strip().strip("|").split("|")]) > 1
    ]


def test_tools_reference_rows_resolve_in_their_toolset(registered_names):
    from toolsets import resolve_toolset

    stale = [
        (toolset, name)
        for toolset, rows in _tool_sections(TOOLS_REF)
        for name in rows
        if name not in registered_names or name not in set(resolve_toolset(toolset))
    ]
    assert not stale, (
        "tools-reference.md documents tool names the registry does not register "
        "(or that the claimed toolset does not resolve): "
        + ", ".join(f"{ts}:{n}" for ts, n in stale)
    )


def test_toolsets_reference_core_table_resolves(registered_names):
    """Core Toolsets table: every claimed member resolves within that toolset;
    composite rows name real toolsets."""
    from toolsets import TOOLSETS, resolve_toolset, validate_toolset

    stale, unknown = [], []
    for toolset, tools_cell in _core_toolset_rows(TOOLSETS_REF):
        if "composite" in tools_cell:
            unknown += [(toolset, n) for n in re.findall(r"`([a-z0-9_]+)`", tools_cell) if not validate_toolset(n)]
            continue
        cell = re.sub(r"\([^)]*\)", " ", tools_cell)  # drop "(via `includes`)"
        resolved = set(resolve_toolset(toolset))
        for name in re.findall(r"`([a-z0-9_]+)`", cell):
            if name not in registered_names or name not in resolved:
                stale.append((toolset, name))
    assert not stale, (
        "toolsets-reference.md claims tool membership the toolsets.py/registry resolver "
        "does not deliver: " + ", ".join(f"{ts}:{n}" for ts, n in stale)
    )
    assert not unknown, (
        "toolsets-reference.md composite rows name unknown toolsets: "
        + ", ".join(f"{ts}:{n}" for ts, n in unknown)
    )

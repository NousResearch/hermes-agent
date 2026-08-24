"""Doc-fact contract: slash-commands.md must match the command registry.

Ported from nearai/ironclaw#7378 (doc-fact contract tests: parse the real
surface, cross-check the published doc in both directions). Their CLI
reference test caught real drift — commands with no doc row and doc rows
teaching commands the binary doesn't have. Same class of drift existed
here: ``/loop`` shipped with a feature page (user-guide/features/loops.md)
but never got a row in the slash-commands reference, and the zh-Hans doc
carried retired ``/credits``/``/billing`` rows for months (PR #69639).

Two directions, both driven by ``hermes_cli.commands.COMMAND_REGISTRY``
(the single source every surface derives from):

1. Every registered command must be documented under at least one visible
   form — its canonical name or any declared alias. (IronClaw rule: "any
   visible alias form counts".) This keeps new CommandDefs from shipping
   undocumented.
2. Every command token that appears as a table row in the doc must resolve
   to a registered name or alias. This keeps retired commands from
   lingering in the doc after they leave the registry.

Only the English doc is gated: i18n copies lag by design and are synced
in dedicated passes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = REPO_ROOT / "website" / "docs" / "reference" / "slash-commands.md"

# Table rows look like:  | `/name ...` | description |
# Only the leading command token of a row is a doc-fact claim; command
# mentions in prose or descriptions are not rows.
_ROW_CMD_RE = re.compile(r"^\|\s*`/([a-z0-9_-]+)", re.MULTILINE)

# Doc rows that are deliberately not CommandDef entries.
_NON_REGISTRY_ROWS = {
    # Dynamic skill invocation — every installed skill becomes /<skill-name>.
    "skill-name",
}


@pytest.fixture(scope="module")
def registry():
    from hermes_cli.commands import COMMAND_REGISTRY

    return COMMAND_REGISTRY


@pytest.fixture(scope="module")
def doc_text() -> str:
    assert DOC_PATH.is_file(), f"missing doc: {DOC_PATH}"
    return DOC_PATH.read_text(encoding="utf-8")


def test_every_registered_command_is_documented(registry, doc_text):
    """Each command must appear in the doc as /name or any /alias."""
    missing = []
    for cmd in registry:
        forms = [cmd.name, *(cmd.aliases or ())]
        if not any(f"/{form}" in doc_text for form in forms):
            missing.append(cmd.name)
    assert not missing, (
        "Commands registered in hermes_cli/commands.py but absent from "
        f"website/docs/reference/slash-commands.md: {missing}. "
        "Add a table row (Session/Configuration/Tools & Skills/Info for the "
        "CLI table, and the messaging table if the command works on the "
        "gateway), or document one of its aliases."
    )


def test_every_documented_row_is_registered(registry, doc_text):
    """Each doc table row's command token must exist in the registry."""
    known = {c.name for c in registry}
    for c in registry:
        known.update(c.aliases or ())
    known |= _NON_REGISTRY_ROWS

    unknown = sorted(
        {tok for tok in _ROW_CMD_RE.findall(doc_text) if tok not in known}
    )
    assert not unknown, (
        "slash-commands.md documents commands that no longer exist in "
        f"COMMAND_REGISTRY: {unknown}. Remove the stale rows (or register "
        "the command)."
    )


def test_doc_parses_at_least_the_known_surface(doc_text):
    """Sanity: the row regex actually sees the tables (guards against a
    format change silently turning both contracts vacuous)."""
    rows = set(_ROW_CMD_RE.findall(doc_text))
    assert len(rows) >= 60, (
        f"only {len(rows)} command rows parsed from slash-commands.md — "
        "table format may have changed; update _ROW_CMD_RE."
    )

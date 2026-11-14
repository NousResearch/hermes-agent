"""execute_code's lifecycle guard blocks running a gateway restart, not mentioning one (#146582)."""

from __future__ import annotations

import json

import pytest

import tools.code_execution_tool as cet

_CMD = "sudo systemctl restart hermes-gateway.service"


@pytest.fixture
def in_gateway(monkeypatch):
    from tools import process_registry

    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)


def _blocked(code: str) -> str | None:
    from tools.code_execution_lifecycle import lifecycle_guard_rejection

    return lifecycle_guard_rejection(code)


@pytest.mark.parametrize("code", [
    f'print("this script only prints; the command is: {_CMD}")',
    f'doc = "Runbook:\\n    {_CMD}\\n"\nopen("runbook.md", "w", encoding="utf-8").write(doc)',
    f'import json, pathlib\npathlib.Path("notes.md").write_text(json.dumps({{"fix": "{_CMD}"}}))',
    f'with open("runbook.md", "w") as fh:\n    fh.write("{_CMD}")',
])
def test_documentation_only_cell_is_not_refused(code):
    assert _blocked(code) is None


@pytest.mark.parametrize("code", [
    f'import os\nos.system("{_CMD}")',
    f'import subprocess\nsubprocess.run("{_CMD}", shell=True)',
    # the kernel persists: os / helpers from an earlier cell are reachable without an import
    f'os.system("{_CMD}")',
    f'sh("{_CMD}")',
    f'sh = sh\nsh("{_CMD}")',  # the right-hand side still reads the earlier cell's helper
    f'from hermes_tools import terminal\nterminal("{_CMD}")',
    f'exec("import os; os.system(\'{_CMD}\')")',
    f'import pathlib\npathlib.os.system("{_CMD}")',
    f'print("{_CMD}"',  # unparsable -> scanned as before
])
def test_cell_that_can_reach_a_process_is_still_refused(code):
    assert _blocked(code) is not None


def test_rejection_names_the_matched_command(in_gateway):
    result = json.loads(cet.execute_code(f'import os\nos.system("{_CMD}")'))
    assert "cannot restart or stop the gateway" in result["error"]
    assert "systemctl restart hermes-gateway" in result["error"]


# Only straight-line top-level code is judged inert: a binding inside a nested scope, a branch or a
# loop may never be the one a module-level read sees, and an earlier cell's helper would answer.
@pytest.mark.parametrize("code", [
    f'def g(sh): pass\nsh("{_CMD}")',
    f'g = lambda sh: 0\nsh("{_CMD}")',
    f'[0 for sh in []]\nsh("{_CMD}")',
    f'def g():\n    sh = 1\nsh("{_CMD}")',
    f'class K:\n    sh = 1\nsh("{_CMD}")',
    f'if 0:\n    sh = None\nsh("{_CMD}")',
    f'for sh in []:\n    pass\nsh("{_CMD}")',
    f'g = (x for x in [0])\ng.gi_frame.f_globals["sh"]("{_CMD}")',
    f'import typing\ndef f(x: "sh()"): pass\ntyping.get_type_hints(f)\nprint("{_CMD}")',
    # conservative on purpose: helpers and loops keep the normal scan even when harmless
    f'def fmt(line):\n    return "- " + line\nprint(fmt("{_CMD}"))',
])
def test_scoped_or_conditional_bindings_never_count_as_inert(code):
    assert _blocked(code) is not None


def test_parser_exhaustion_is_scanned_as_before():
    assert _blocked("(" * 5000 + f'"{_CMD}"' + ")" * 5000) is not None

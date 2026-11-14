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
    f'def fmt(line):\n    return "- " + line\nprint(fmt("{_CMD}"))',
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

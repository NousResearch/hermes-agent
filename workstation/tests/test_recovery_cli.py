from __future__ import annotations

import json

from workstation.recovery_cli import main


def test_recovery_cli_quarantines_and_restores_without_manual_state_edit(tmp_path, capsys):
    path = tmp_path / "recovery.json"
    assert main(["--recovery-state", str(path), "quarantine", "plugin-x", "import failed"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["components"]["plugin-x"]["status"] == "quarantined"

    assert main(["--recovery-state", str(path), "restore", "plugin-x"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["components"] == {}

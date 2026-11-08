"""Exercise recoverable metadata writes at real process interruption boundaries."""
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

import hermes_yaml as yaml
from tui_gateway.group_chat_storage import GROUP_CHAT_KEY as KEY, GROUP_CHAT_STATE as STATE


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (root / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    (root / "profile.yaml").write_text(yaml.safe_dump({
        "display_name": "Example", "ui_meta": {
            KEY: {"rooms": {}, "label": "old"}, "example-settings": {"color": "blue"}},
        "_ui_meta_revisions": {KEY: 3, "example-settings": 1},
    }), encoding="utf-8")
    return root


def child(home, script, *args):
    return subprocess.run([sys.executable, "-c", script, *args], capture_output=True,
                          text=True, timeout=30, check=False,
                          env=os.environ | {"HOME": str(home.parent), "HERMES_HOME": str(home)})


WRITE = """
from pathlib import Path
import os
import sys
import utils
import tui_gateway.server as srv
home = Path(os.environ['HERMES_HOME'])
real = utils.atomic_yaml_write
phase = sys.argv[1]
def interrupted(path, data, **kwargs):
    path = Path(path)
    if path == home / 'state/group-chat.yaml' and phase == 'before-state':
        os._exit(23)
    if path == home / 'profile.yaml' and phase == 'before-profile':
        os._exit(23)
    real(path, data, **kwargs)
    if path == home / 'profile.yaml' and phase == 'after-profile':
        os._exit(23)
    if path == home / 'state/group-chat.yaml' and phase == 'after-clear' and 'pending_ui_meta' not in data:
        os._exit(23)
utils.atomic_yaml_write = interrupted
srv._methods['profiles.configure']('write', {
    'name': 'default', 'ui_meta': {
        'hermes-bots-groups': {'rooms': {}, 'label': 'new'},
        'example-settings': {'color': 'green'}},
    'ui_meta_expected_revisions': {'hermes-bots-groups': 3, 'example-settings': 1},
})
raise RuntimeError('interruption point was not reached')
"""

READ = """
import json
import sys
import tui_gateway.server as srv
result = srv._methods['profiles.list']('read', {'include_sessions': False})
print(json.dumps(next(p for p in result['result']['profiles'] if p['name'] == 'default')), file=sys.__stdout__)
"""

RECOVER = """
from pathlib import Path
import os
from tui_gateway.group_chat_storage import migrate_group_chat_state
migrate_group_chat_state(Path(os.environ['HERMES_HOME']))
"""


@pytest.mark.parametrize("phase", ["before-state", "before-profile", "after-profile", "after-clear"])
def test_new_process_reads_and_recovers_the_entire_interrupted_update(home, phase):
    result = child(home, WRITE, phase)
    assert result.returncode == 23, result.stderr
    observed = child(home, READ)
    assert observed.returncode == 0, observed.stderr
    row = json.loads(observed.stdout.strip().splitlines()[-1])
    committed = phase != "before-state"
    assert row["ui_meta"][KEY]["label"] == ("new" if committed else "old")
    assert row["ui_meta"]["example-settings"]["color"] == ("green" if committed else "blue")
    assert row["ui_meta_revisions"] == {KEY: 4 if committed else 3, "example-settings": 2 if committed else 1}

    recovered = child(home, RECOVER)
    assert recovered.returncode == 0, recovered.stderr
    state = yaml.safe_load((home / STATE).read_text())
    metadata = yaml.safe_load((home / "profile.yaml").read_text())
    assert "pending_ui_meta" not in state
    assert KEY not in metadata["ui_meta"]
    assert metadata["ui_meta"]["example-settings"]["color"] == ("green" if committed else "blue")
    before = (home / STATE).read_bytes(), (home / "profile.yaml").read_bytes()
    assert child(home, RECOVER).returncode == 0
    assert ((home / STATE).read_bytes(), (home / "profile.yaml").read_bytes()) == before


def test_recovery_refuses_to_clobber_an_intervening_edit_to_its_namespace(home):
    assert child(home, WRITE, "before-profile").returncode == 23
    metadata = yaml.safe_load((home / "profile.yaml").read_text())
    metadata["ui_meta"]["example-settings"] = {"color": "orange"}
    (home / "profile.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
    before = (home / STATE).read_bytes(), (home / "profile.yaml").read_bytes()
    assert child(home, RECOVER).returncode != 0
    assert ((home / STATE).read_bytes(), (home / "profile.yaml").read_bytes()) == before


@pytest.mark.parametrize("pending", [None, {}, {"example": {}}, {KEY: {"before": {}, "after": {"value": {}}}}])
def test_malformed_journal_cannot_rewrite_profile_metadata(home, pending):
    (home / STATE).parent.mkdir()
    (home / STATE).write_text(yaml.safe_dump({
        "revision": 4, "snapshot": {"rooms": {}}, "pending_ui_meta": pending,
    }), encoding="utf-8")
    before = (home / STATE).read_bytes(), (home / "profile.yaml").read_bytes()
    assert child(home, RECOVER).returncode != 0
    observed = child(home, READ)
    assert observed.returncode == 0, observed.stderr
    row = json.loads(observed.stdout.strip().splitlines()[-1])
    assert row["ui_meta"]["example-settings"] == {"color": "blue"}
    assert KEY not in row["ui_meta"]
    assert ((home / STATE).read_bytes(), (home / "profile.yaml").read_bytes()) == before

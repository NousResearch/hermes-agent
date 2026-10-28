"""Gateway-owned revisions for concurrent profile UI metadata updates.

Clients use ``profiles.list`` to read a shared key and its revision, merge a
local mutation, then pass that revision back to ``profiles.configure``.  The
gateway must let exactly one concurrent writer advance the key and make every
stale writer retry instead of silently replacing somebody else's state.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

import tui_gateway.server as srv


@pytest.fixture
def home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def _configure(ui_meta, expected=None):
    params = {"name": "default", "ui_meta": ui_meta}
    if expected is not None:
        params["ui_meta_expected_revisions"] = expected
    return srv._methods["profiles.configure"]("configure", params)["result"]["applied"]


def _default_profile():
    rows = srv._methods["profiles.list"](
        "list", {"include_sessions": False}
    )["result"]["profiles"]
    return next(row for row in rows if row["name"] == "default")


def test_ui_meta_revision_advances_and_stale_compare_and_swap_is_rejected(home):
    first = _configure({"shared-room": {"messages": ["one"]}})
    assert first["ui_meta"] is True
    assert first["ui_meta_revisions"] == {"shared-room": 1}

    second = _configure(
        {"shared-room": {"messages": ["one", "two"]}},
        {"shared-room": 1},
    )
    assert second["ui_meta"] is True
    assert second["ui_meta_revisions"] == {"shared-room": 2}

    stale = _configure(
        {"shared-room": {"messages": ["stale replacement"]}},
        {"shared-room": 1},
    )
    assert stale["ui_meta"] is False
    assert stale["ui_meta_conflicts"] == {
        "shared-room": {"expected": 1, "actual": 2}
    }

    row = _default_profile()
    assert row["ui_meta"]["shared-room"] == {"messages": ["one", "two"]}
    assert row["ui_meta_revisions"]["shared-room"] == 2


def test_ui_meta_revision_survives_key_deletion(home):
    _configure({"shared-room": {"messages": ["one"]}})
    deleted = _configure({"shared-room": None}, {"shared-room": 1})

    assert deleted["ui_meta"] is True
    assert deleted["ui_meta_revisions"] == {"shared-room": 2}
    row = _default_profile()
    assert "shared-room" not in row.get("ui_meta", {})
    assert row["ui_meta_revisions"]["shared-room"] == 2

    stale_recreate = _configure(
        {"shared-room": {"messages": ["resurrected"]}},
        {"shared-room": 1},
    )
    assert stale_recreate["ui_meta"] is False
    assert stale_recreate["ui_meta_conflicts"]["shared-room"]["actual"] == 2


def test_profiles_list_normalizes_yaml_timestamps_in_ui_meta(home):
    """#92506: an unquoted ISO timestamp under ui_meta is parsed by YAML as datetime; the
    profiles.list row must still be JSON-serializable (the value becomes its ISO string)."""
    (home / "profile.yaml").write_text(
        "ui_meta:\n  hermes-bots:\n    created: 2026-08-22T00:00:00Z\n",
        encoding="utf-8",
    )

    row = _default_profile()

    assert row["ui_meta"]["hermes-bots"]["created"] == "2026-08-22T00:00:00+00:00"
    json.dumps(row)


def test_two_concurrent_writers_cannot_both_replace_the_same_revision(home):
    def write(label):
        return _configure(
            {"shared-room": {"messages": [label]}},
            {"shared-room": 0},
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(write, ("alpha", "beta")))

    assert sum(result["ui_meta"] is True for result in results) == 1
    assert sum(result["ui_meta"] is False for result in results) == 1
    loser = next(result for result in results if result["ui_meta"] is False)
    assert loser["ui_meta_conflicts"]["shared-room"]["actual"] == 1
    assert _default_profile()["ui_meta_revisions"]["shared-room"] == 1


def test_bot_title_write_is_attributed_only_after_successful_cas(home, caplog):
    import logging

    caplog.set_level(logging.INFO)
    first = _configure({"hermes-bots": {"title": "Fixture Bot"}}, {"hermes-bots": 0})
    assert first["ui_meta"] is True
    traces = [r.getMessage() for r in caplog.records if "ui_meta hermes-bots title" in r.getMessage()]
    assert len(traces) == 1
    assert "default" in traces[0] and "Fixture Bot" in traces[0] and "revision 1" in traces[0]
    caplog.clear()
    rejected = _configure({"hermes-bots": {"title": "Stale Bot"}}, {"hermes-bots": 0})
    assert rejected["ui_meta"] is False
    assert not [r for r in caplog.records if "ui_meta hermes-bots title" in r.getMessage()]
    assert _default_profile()["ui_meta"]["hermes-bots"]["title"] == "Fixture Bot"

def test_bot_title_removal_is_attributed_without_claiming_previous_title(home, caplog):
    import logging

    _configure({"hermes-bots": {"title": "Fixture Bot"}})
    caplog.set_level(logging.INFO)
    caplog.clear()
    deleted = _configure({"hermes-bots": None}, {"hermes-bots": 1})
    assert deleted["ui_meta"] is True
    traces = [r.getMessage() for r in caplog.records if "ui_meta hermes-bots title" in r.getMessage()]
    assert len(traces) == 1
    assert "None" in traces[0] and "revision 2" in traces[0]
    assert "Fixture Bot" not in traces[0]
    assert "hermes-bots" not in _default_profile().get("ui_meta", {})


_RACE_CHILD = '''
import json
import sys
from pathlib import Path
if sys.argv[1] == 'gateway':
    import tui_gateway.server as srv
else:
    from hermes_cli.profiles_ui_meta import apply_ui_meta_cas
print('ready', file=sys.__stdout__, flush=True)
sys.stdin.readline()
if sys.argv[1] == 'gateway':
    result = srv._methods['profiles.configure']('cas-race', {
        'name': 'default', 'ui_meta': {'shared-room': {'writer': 'gateway'}},
        'ui_meta_expected_revisions': {'shared-room': 0}})['result']['applied']
else:
    result = apply_ui_meta_cas(Path(sys.argv[2]), {'shared-room': {'writer': 'helper'}},
                               expected_revisions={'shared-room': 0})
    result = {'ui_meta': result['applied'], 'ui_meta_revisions': result['revisions'],
              'ui_meta_conflicts': result.get('conflicts')}
print(json.dumps(result), file=sys.__stdout__, flush=True)
'''


def test_real_gateway_rpc_and_helper_processes_share_cas(home):
    import subprocess
    import sys
    from pathlib import Path
    import hermes_yaml as yaml

    children = []
    try:
        for mode in ('gateway', 'helper'):
            child = subprocess.Popen([sys.executable, '-c', _RACE_CHILD, mode, str(home)],
                                     cwd=Path(__file__).resolve().parents[2],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
            children.append(child)
        for child in children:
            assert child.stdout.readline().strip() == 'ready'
        for child in children:
            child.stdin.write('\n')
            child.stdin.flush()
        results = []
        for child in children:
            out, err = child.communicate(timeout=60)
            assert child.returncode == 0, err
            results.append(json.loads(out.strip().splitlines()[-1]))
        assert sum(r['ui_meta'] is True for r in results) == 1
        loser = next(r for r in results if r['ui_meta'] is False)
        assert loser['ui_meta_conflicts'] == {'shared-room': {'expected': 0, 'actual': 1}}
        winner_index = next(i for i, r in enumerate(results) if r['ui_meta'] is True)
        document = yaml.safe_load((home / 'profile.yaml').read_text())
        assert document['ui_meta']['shared-room']['writer'] == ('gateway', 'helper')[winner_index]
        assert document['_ui_meta_revisions']['shared-room'] == 1
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                _, err = child.communicate(timeout=30)
                print(err)


_GATEWAY_TIMEOUT_CHILD = '''
import json
import logging
import sys
import tui_gateway.server as srv
import hermes_cli.profiles_ui_meta as profiles
profiles._PROFILE_UI_META_LOCK_TIMEOUT_S = 0.1
logging.basicConfig(level=logging.INFO)
print('ready', file=sys.__stdout__, flush=True)
sys.stdin.readline()
result = srv._methods['profiles.configure']('lock-timeout', {
    'name': 'default', 'ui_meta': {'hermes-bots': {'title': 'Rejected'}},
    'ui_meta_expected_revisions': {'hermes-bots': 0}})['result']['applied']
print(json.dumps(result), file=sys.__stdout__, flush=True)
'''


def test_gateway_refuses_while_canonical_helper_lock_is_held(home):
    import subprocess
    import sys
    from pathlib import Path
    from hermes_cli.profiles_ui_meta import _profile_ui_meta_file_lock

    path = home / 'profile.yaml'
    path.write_text('description: keep\n')
    before = path.read_bytes()
    child = None
    try:
        with _profile_ui_meta_file_lock(home):
            child = subprocess.Popen([sys.executable, '-c', _GATEWAY_TIMEOUT_CHILD],
                                     cwd=Path(__file__).resolve().parents[2],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
            assert child.stdout.readline().strip() == 'ready'
            out, err = child.communicate('\n', timeout=60)
            assert child.returncode == 0, err
            assert json.loads(out.strip().splitlines()[-1]) == {'ui_meta': False}
            assert 'ui_meta hermes-bots title' not in err
            assert path.read_bytes() == before
        assert _configure({'hermes-bots': {'title': 'Accepted'}}, {'hermes-bots': 0})['ui_meta'] is True
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.communicate(timeout=30)

"""Offline failure contract on top of the runner bootstrap in PR #123070.

The child may fail dependency activation AFTER the parent has durably admitted the DM.
No transport, live owner, external profile, or network is used here.
"""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from tools import bot_mode_dm as dm


# Execute the actual script entrypoint; only the external activation boundary is sabotaged.
# An in-process runpy call would inherit the parent's already-activated package state.
_CHILD = """import builtins, runpy, sys
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'hermes_bootstrap':
        raise RuntimeError('test: dependency generation unavailable')
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""


def _failed_child(tmp_path, dm_file, profile_home, *, waiter=False, command=None):
    argv = (["--wait-reply", str(tmp_path / "reply.json"), "@peer", "1"] if waiter else
            ["--run-delivery", "query-file", str(dm_file), "--profile-home", str(profile_home),
             "/nonexistent/transport"])
    runner = shlex.split(command) if command else [sys.executable, str(Path(dm.__file__).resolve()), *argv]
    assert runner[1] == str(Path(dm.__file__).resolve())
    env = {k: v for k, v in os.environ.items() if k not in
           ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "HERMES_RUNTIME_DIR") and
           not (k.endswith("_KEY") or k.endswith("_TOKEN"))}
    env["HERMES_HOME"] = str(tmp_path / "isolated-hermes")
    env["HOME"] = str(tmp_path)
    env["TMPDIR"] = str(tmp_path)
    return subprocess.run([runner[0], "-c", _CHILD, *runner[1:]],
                          cwd=str(Path(dm.__file__).resolve().parents[1]), env=env,
                          capture_output=True, text=True, timeout=20)


def test_parent_admission_then_child_activation_failure_remains_unknown(tmp_path, monkeypatch):
    profile_home = tmp_path / "recipient"
    profile_home.mkdir()
    dm_file = tmp_path / "dm.txt"
    monkeypatch.setattr(dm, "_write_dm_file", lambda content: (dm_file.write_text(content), str(dm_file))[1])
    observed = {}

    def admitted(home, path, author):
        assert home == profile_home and path == str(dm_file)
        delivery_id = dm._dm_delivery_id(path)
        Path(path + ".live.json").write_text(json.dumps({"delivery_id": delivery_id}))
        receipt = profile_home / "runtime" / "bot_live_delivery" / f"{delivery_id}.json"
        receipt.parent.mkdir(parents=True)
        receipt.write_text(json.dumps({"status": "queued"}))
        return {"status": "queued", "delivery_id": delivery_id}

    def spawn(command, label, *, task_id, agent):
        result = _failed_child(tmp_path, dm_file, profile_home, command=command)
        observed["child"] = result
        return json.dumps({"process_id": "offline-only"})

    monkeypatch.setattr(dm, "_admit_live_dm", admitted)
    monkeypatch.setattr(dm, "_spawn_delivery", spawn)
    parent = json.loads(dm._start_delivery(["/nonexistent/transport"], "offline-message", "@peer",
                                           stdin_file=False, task_id=None, agent=None,
                                           profile_home=profile_home))
    child = observed["child"]
    result = json.loads(child.stdout.strip())
    assert parent["status"] == "queued" and child.returncode == 1
    assert result["status"] == "ambiguous" and result["outcome"] == "UNKNOWN"
    assert result["delivery_id"] == parent["delivery_id"]
    assert "Do not resend" in result["detail"]
    assert dm_file.exists()


def test_activation_fails_before_admission_only_with_strong_absence_evidence(tmp_path):
    profile_home = tmp_path / "recipient"
    profile_home.mkdir()
    dm_file = tmp_path / "dm.txt"
    dm_file.write_text("offline-message")
    child = _failed_child(tmp_path, dm_file, profile_home)
    result = json.loads(child.stdout.strip())
    assert child.returncode == 1 and result["status"] == "not_delivered"
    assert result["delivery_id"] == dm._dm_delivery_id(str(dm_file))
    assert dm_file.exists()

    # A receipt with no intent, or a truncated intent, still forbids a resend.
    receipt = profile_home / "runtime" / "bot_live_delivery" / f"{dm._dm_delivery_id(str(dm_file))}.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"status": "queued"}))
    child = _failed_child(tmp_path, dm_file, profile_home)
    result = json.loads(child.stdout.strip())
    assert child.returncode == 1 and result["status"] == "ambiguous"
    receipt.unlink()
    Path(str(dm_file) + ".live.json").write_text("")
    child = _failed_child(tmp_path, dm_file, profile_home)
    result = json.loads(child.stdout.strip())
    assert child.returncode == 1 and result["status"] == "ambiguous"

    Path(str(dm_file) + ".live.json").unlink()
    child = _failed_child(tmp_path, dm_file, profile_home,
                          command=shlex.join([sys.executable, str(Path(dm.__file__).resolve()),
                                              "--run-delivery", "query-file", str(dm_file),
                                              "/nonexistent/transport"]))
    assert json.loads(child.stdout)["outcome"] == "UNKNOWN"  # no pinned receipt home
    dm_file.unlink()
    child = _failed_child(tmp_path, dm_file, profile_home)
    assert json.loads(child.stdout)["outcome"] == "UNKNOWN"  # payload vanished

    (tmp_path / "reply.json").write_text(json.dumps({"reply": "offline-pong"}))
    waiter = _failed_child(tmp_path, dm_file, profile_home, waiter=True)
    assert waiter.returncode == 0 and "offline-pong" in waiter.stdout

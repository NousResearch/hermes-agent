"""The frozen owner terminal policy controls the HOME of terminal children."""
import json
import os
from dataclasses import asdict

import pytest


@pytest.mark.parametrize("configured,expected_mode", [("real", "real"), ("profile", "profile"), (None, "auto")])
def test_owner_policy_home_reaches_subprocess(tmp_path, monkeypatch, configured, expected_mode):
    from gateway.session_policy import build_policy, policy_scope, restore_policy
    from hermes_constants import get_subprocess_home
    from tools.environments.local import build_subprocess_env

    real = tmp_path / "real"
    profile = tmp_path / "profile"
    real.mkdir()
    (profile / "home").mkdir(parents=True)
    terminal = {} if configured is None else {"home_mode": configured}
    (profile / "config.yaml").write_text(json.dumps({"terminal": terminal}), encoding="utf-8")
    monkeypatch.setenv("HOME", str(real))
    monkeypatch.setenv("HERMES_REAL_HOME", str(real))
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setenv("TERMINAL_HOME_MODE", "profile" if configured != "profile" else "real")
    monkeypatch.setattr("hermes_constants.is_container", lambda: True)
    policy = restore_policy(asdict(build_policy({"source": "tui", "cwd": str(tmp_path), "model": "m"}, {})))
    expected = str(real if expected_mode == "real" else profile / "home")
    # Once serialized/restored, a later YAML edit must not replace the turn's policy.
    (profile / "config.yaml").write_text('{"terminal":{"home_mode":"' + (
        "profile" if configured == "real" else "real") + '"}}', encoding="utf-8")
    before = dict(os.environ)
    with policy_scope(policy):
        assert (get_subprocess_home() or str(real)) == expected
        assert build_subprocess_env()["HOME"] == expected
    assert dict(os.environ) == before

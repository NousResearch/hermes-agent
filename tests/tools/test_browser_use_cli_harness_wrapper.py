"""``browser.harness_wrapper``: every browser_exec CLI call of a profile runs under that profile's command.

A deployment that shares one browser between several agents (``browser.cdp_url``) uses it to queue the
calls or to account for the tabs each call opens. The wrapper gets the harness command as its arguments,
the model's code on stdin and the call's environment. A wrapper that is invalid or cannot launch fails
the call with an error naming the setting: the harness never runs unwrapped.
"""
import json
import sys

import pytest

from gateway.run import _profile_runtime_scope
from tools import browser_session_cdp as bs_cdp
from tools import browser_use_cli as bu_cli

_FAKE_CLI = "import os, sys\nsys.stdout.write(sys.stdin.read() + '|' + os.environ.get('BU_NAME', ''))\n"
# Records the command it was given, then runs it, or only records it when told to stop there.
_WRAPPER = """import json, subprocess, sys
with open(sys.argv[1], 'a', encoding='utf-8') as log:
    log.write(json.dumps(sys.argv[2:]) + '\\n')
raise SystemExit(0 if sys.argv[2:3] == ['--record-only'] else subprocess.call(sys.argv[2:]))
"""


def _write_config(home, value):
    (home / "config.yaml").write_text(f"browser:\n  harness_wrapper: {json.dumps(value)}\n", encoding="utf-8")


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """A fake harness CLI echoing its stdin and BU_NAME, and two profiles: A wrapped, B not."""
    cli = tmp_path / "fake_cli.py"
    cli.write_text(_FAKE_CLI, encoding="utf-8")
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(_WRAPPER, encoding="utf-8")
    calls = tmp_path / "wrapper-calls.jsonl"
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _write_config(a, [sys.executable, str(wrapper), str(calls)])
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    monkeypatch.setattr(bs_cdp, "_live_overrides", {})
    monkeypatch.setattr(bs_cdp, "resolve_cdp_override", lambda url: url)
    monkeypatch.setattr(bu_cli, "_find_cli", lambda: [sys.executable, str(cli)])
    monkeypatch.setattr(bu_cli, "_driven_daemons", {})
    monkeypatch.setattr(bu_cli, "_attach_vault_supervisor", lambda env, task_id: None)
    return a, b, cli, wrapper, calls


def _exec(home, code, session=""):
    with _profile_runtime_scope(home, hydrate_secrets=False):
        bs_cdp.set_live_cdp_override("http://127.0.0.1:9301")
        try:
            return json.loads(bu_cli.browser_exec(code, session=session, task_id="t"))
        finally:
            bs_cdp.set_live_cdp_override(None)


def _recorded(calls):
    return [json.loads(line) for line in calls.read_text(encoding="utf-8-sig").splitlines()]


def test_a_wrapped_profile_runs_the_harness_through_its_command(harness):
    a, _b, cli, _wrapper, calls = harness
    result = _exec(a, "print('hi')", session="s1")
    # stdin (a named session on a shared browser may get a preamble first) and the env reached the harness
    assert result["output"].endswith("print('hi')|s1")
    assert _recorded(calls) == [[sys.executable, str(cli)]]  # the wrapper received the harness command


def test_profiles_are_islands_an_unwrapped_profile_runs_the_harness_directly(harness):
    a, b, _cli, _wrapper, calls = harness
    assert _exec(b, "print(2)")["output"].startswith("print(2)|")
    assert not calls.exists()
    _exec(a, "print(3)")
    assert len(_recorded(calls)) == 1


def test_the_wrapper_wraps_the_launcher_harness_command_builds(harness, monkeypatch):
    """The real module command is rewritten into the helper launcher first; the wrapper must get that."""
    a, _b, _cli, wrapper, calls = harness
    monkeypatch.setattr(bu_cli, "_find_cli", lambda: [sys.executable, "-m", "browser_harness.run"])
    _write_config(a, [sys.executable, str(wrapper), str(calls), "--record-only"])
    _exec(a, "print(5)")
    [[marker, interpreter, flag, launcher]] = _recorded(calls)
    assert (marker, interpreter, flag) == ("--record-only", sys.executable, "-c")
    assert "browser_use_shot" in launcher


@pytest.mark.parametrize("value", [["flock", "-w", 30, "lock"], ["/bin/wrap", ""], {"cmd": "/bin/wrap"}, 7])
def test_an_invalid_wrapper_fails_the_call_and_names_the_setting(harness, value):
    a, _b, _cli, _wrapper, calls = harness
    _write_config(a, value)
    result = _exec(a, "print(6)")
    assert "browser.harness_wrapper" in result["error"]
    assert not calls.exists()


def test_a_wrapper_that_cannot_launch_fails_the_call_and_names_the_setting(harness, tmp_path):
    a, _b, _cli, _wrapper, _calls = harness
    _write_config(a, str(tmp_path / "gone"))
    result = _exec(a, "print(4)")
    assert "browser.harness_wrapper" in result["error"] and "gone" in result["error"]

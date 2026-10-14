"""A hand edit to ``command_allowlist`` must reach a long-lived process's approval check.

The unified gateway daemon serves every local session, so the import-time load of
``command_allowlist`` lived as long as the daemon: an entry the operator removed by hand
(the documented way to withdraw a standing approval) stayed auto-approved until restart.
"""

import pytest

from hermes_cli.config import get_config_path
from tools import approval
from tools.approval_floors import _command_matches_permanent_allowlist


@pytest.fixture
def allowlist_file():
    """Write ``command_allowlist`` into the isolated test home's config.yaml."""
    saved_approved = set(approval._permanent_approved)
    saved_baseline = dict(approval._permanent_baseline_by_home)
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    def _write(entries):
        lines = ["command_allowlist:"] + [f"- '{e}'" for e in entries] if entries else ["command_allowlist: []"]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        yield _write
    finally:
        approval._permanent_approved.clear()
        approval._permanent_approved.update(saved_approved)
        approval._permanent_baseline_by_home.clear()
        approval._permanent_baseline_by_home.update(saved_baseline)


def test_entry_removed_by_hand_prompts_again(allowlist_file):
    allowlist_file(["rm -rf build", "ls *"])
    approval.load_permanent_allowlist()
    assert _command_matches_permanent_allowlist("rm -rf build")

    allowlist_file(["ls *"])                     # operator withdraws the standing approval

    assert not _command_matches_permanent_allowlist("rm -rf build")
    assert not approval.is_approved("s1", "rm -rf build")
    assert _command_matches_permanent_allowlist("ls -la")


def test_entry_added_by_hand_takes_effect(allowlist_file):
    allowlist_file(["ls *"])
    approval.load_permanent_allowlist()
    assert not _command_matches_permanent_allowlist("podman ps")

    allowlist_file(["ls *", "podman *"])

    assert _command_matches_permanent_allowlist("podman ps")


def test_approval_made_in_process_survives_an_unrelated_hand_edit(allowlist_file):
    """An [a]lways whose save failed lives only in memory; a later edit must not drop it."""
    allowlist_file(["ls *", "git status"])
    approval.load_permanent_allowlist()
    approval.approve_permanent("recursive delete")

    allowlist_file(["ls *"])

    assert approval._is_permanently_approved("recursive delete")
    assert not _command_matches_permanent_allowlist("git status")

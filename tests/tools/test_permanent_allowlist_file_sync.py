"""A hand edit to ``command_allowlist`` must reach the approval hot path.

Removing an entry from ``command_allowlist`` in config.yaml is the documented
way to withdraw a standing approval (``save_permanent_allowlist`` docstring).
The permanent set was loaded once per profile home and cached in
``_permanent_approved`` / ``_permanent_approved_by_home``; nothing re-read the
file on the approval hot path, so a revoked entry kept auto-approving commands
until the process restarted (or an unrelated ``[a]lways`` happened to save).

These tests use the real config.yaml under the per-test HERMES_HOME that the
autouse ``_hermetic_environment`` fixture provides — no monkeypatched loaders —
because the bug is precisely that the file is never consulted again.
"""

import pytest

import tools.approval as approval
from hermes_constants import get_hermes_home


@pytest.fixture
def clean_approval_state():
    """Clean module approval state; restore it afterwards."""
    saved_approved = set(approval._permanent_approved)
    saved_baseline = dict(approval._permanent_baseline_by_home)
    sig_by_home = getattr(approval, "_permanent_sig_by_home", None)
    saved_sig = dict(sig_by_home) if sig_by_home is not None else None
    approval._permanent_approved.clear()
    approval._permanent_baseline_by_home.clear()
    if sig_by_home is not None:
        sig_by_home.clear()
    try:
        yield
    finally:
        approval._permanent_approved.clear()
        approval._permanent_approved.update(saved_approved)
        approval._permanent_baseline_by_home.clear()
        approval._permanent_baseline_by_home.update(saved_baseline)
        if sig_by_home is not None:
            sig_by_home.clear()
            sig_by_home.update(saved_sig)


def _write_allowlist(entries):
    """The operator's edit path: write config.yaml by hand."""
    (get_hermes_home() / "config.yaml").write_text(
        "command_allowlist:\n" + "".join(f"  - {e}\n" for e in entries),
        encoding="utf-8",
    )


def test_revoking_an_entry_on_disk_withdraws_the_approval(clean_approval_state):
    """Grant a standing approval, delete the line from config.yaml, and the
    very next approval check must stop honouring it — no restart, no save."""
    _write_allowlist(["git status", "ls *"])
    approval.load_permanent_allowlist()          # what import does at startup
    assert approval.is_approved("s", "git status") is True

    _write_allowlist(["ls *"])                   # operator revokes it

    assert approval.is_approved("s", "git status") is False, (
        "a revoked standing approval still auto-approves until restart"
    )
    assert approval.is_approved("s", "ls *") is True


def test_resync_keeps_pending_approvals_and_revoked_stays_revoked(
        clean_approval_state):
    """The re-read must not union the stale set back in (the #101741 shape):
    an entry approved in memory but not yet saved survives the sync, while a
    revoked baseline entry stays revoked even after another approval lands."""
    _write_allowlist(["git status", "ls *"])
    approval.load_permanent_allowlist()

    approval.approve_permanent("docker *")       # in memory, not yet on disk
    _write_allowlist(["ls *", "npm test"])       # revoke one, add one by hand

    assert approval.is_approved("s", "git status") is False
    assert approval.is_approved("s", "npm test") is True, (
        "a hand-added entry must be picked up, not just removals"
    )
    assert approval.is_approved("s", "docker *") is True, (
        "an approval this process made but has not saved yet must survive the sync"
    )

    # ... and the revoked entry must still be gone after a real save lands.
    approval.save_permanent_allowlist(set(approval._permanent_approved))
    assert approval.is_approved("s", "git status") is False
    assert approval.is_approved("s", "docker *") is True

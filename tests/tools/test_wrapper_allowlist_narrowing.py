"""Permanent allowlist matches may be withdrawn, but never created, by wrapper projection."""

import pytest

from tools import approval
from tools.approval_floors import _command_matches_permanent_allowlist


@pytest.fixture
def permanent_patterns(monkeypatch):
    patterns: set[str] = set()
    monkeypatch.setattr(approval, "_permanent_set", lambda: patterns)

    def replace(values):
        patterns.clear()
        patterns.update(values)

    return replace


@pytest.mark.parametrize(
    "command",
    [
        "su -c 'rm -rf /'",
        "doas rm -rf /",
        "watch -n1 rm -rf /tmp/x",
    ],
)
def test_dangerous_wrapper_projection_withdraws_glob_match(permanent_patterns, command):
    permanent_patterns(["su *", "doas *", "watch *", "ls"])
    assert not _command_matches_permanent_allowlist(command)


@pytest.mark.parametrize(
    "command",
    [
        "watch ls",
        "doas ls",
        "su - alice",
    ],
)
def test_benign_wrapper_glob_match_remains_allowlisted(permanent_patterns, command):
    permanent_patterns(["su *", "doas *", "watch *", "ls"])
    assert _command_matches_permanent_allowlist(command)


def test_projection_never_widens_allowlist(permanent_patterns):
    permanent_patterns(["ls"])
    assert not _command_matches_permanent_allowlist("watch ls")


def test_exact_match_is_not_withdrawn(permanent_patterns):
    command = "doas rm -rf /"
    permanent_patterns([command])
    assert _command_matches_permanent_allowlist(command)


def test_unwrapped_dangerous_glob_match_is_not_withdrawn(permanent_patterns):
    permanent_patterns(["rm *"])
    assert _command_matches_permanent_allowlist("rm -rf /tmp/x")


@pytest.mark.parametrize(
    "command",
    [
        "su -c '/sbin/reboot'",
        "su root -c '/sbin/reboot'",
    ],
)
def test_absolute_path_payload_remains_allowlisted_until_path_projection_lands(
    permanent_patterns,
    command,
):
    permanent_patterns(["su *"])
    assert _command_matches_permanent_allowlist(command)

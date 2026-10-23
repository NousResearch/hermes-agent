"""``~user`` cwds must reach the SSH shell unexpanded, like ``~`` already does.

`_is_ssh_remote_tilde_cwd` preserved ``~`` and ``~/...`` but not ``~other``.
The named form fails the same way and more quietly:
``os.path.expanduser("~alice/work")`` resolves against the HOST's account
database, so it silently becomes a local absolute path whenever a local user
shares the remote name, and passes through untouched otherwise. The remote
therefore receives a path that was resolved on the wrong machine.
"""

from __future__ import annotations

import getpass
import os

import pytest

from hermes_cli.config import _is_ssh_remote_tilde_cwd, apply_terminal_config_to_env


@pytest.mark.parametrize(
    "cwd",
    ["~", "~/work", "~alice", "~alice/work", "~alice/deep/path"],
)
def test_every_tilde_form_is_left_for_the_remote_shell(cwd):
    assert _is_ssh_remote_tilde_cwd("ssh", cwd) is True


@pytest.mark.parametrize("cwd", ["/abs/path", "relative/path", ".", ""])
def test_non_tilde_cwds_are_unaffected(cwd):
    assert _is_ssh_remote_tilde_cwd("ssh", cwd) is False


@pytest.mark.parametrize("backend", ["local", "docker", "", "SSH-ish"])
def test_only_the_ssh_backend_defers_expansion(backend):
    """Guard: every other backend keeps expanding on the Hermes host."""
    assert _is_ssh_remote_tilde_cwd(backend, "~/work") is False


def test_ssh_backend_is_matched_case_insensitively():
    assert _is_ssh_remote_tilde_cwd("SSH", "~alice/work") is True
    assert _is_ssh_remote_tilde_cwd("  ssh  ", "~alice/work") is True


def test_local_account_name_does_not_rewrite_the_remote_path():
    """The failure this fixes, using a name that really exists on this host.

    Expanding here produces the HOST's home directory for that account, which
    is then handed to the remote shell as an absolute path.
    """
    local_user = getpass.getuser()
    cwd = f"~{local_user}/work"
    assert os.path.expanduser(cwd) != cwd, (
        "test needs a locally resolvable account name to be meaningful"
    )

    env = apply_terminal_config_to_env(
        env={}, config={"terminal": {"backend": "ssh", "cwd": cwd}}
    )

    assert env.get("TERMINAL_CWD") == cwd, (
        "the cwd was expanded against the local account database before SSH saw it"
    )


def test_plain_tilde_still_preserved_end_to_end():
    """Guard: the case the original fix covered keeps working."""
    env = apply_terminal_config_to_env(
        env={}, config={"terminal": {"backend": "ssh", "cwd": "~/work"}}
    )

    assert env.get("TERMINAL_CWD") == "~/work"


def test_non_ssh_backend_still_expands_end_to_end():
    """Guard: a local backend must still resolve the path on this host."""
    env = apply_terminal_config_to_env(
        env={}, config={"terminal": {"backend": "local", "cwd": "~/work"}}
    )

    assert env.get("TERMINAL_CWD") == os.path.expanduser("~/work")

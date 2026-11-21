"""``execute_code`` keeps git's commit identity, and nothing else, through the env scrub.

``_SECRET_SUBSTRINGS`` matches ``AUTH`` as a plain substring, so ``GIT_AUTHOR_*`` was blocked,
while ``GIT_COMMITTER_*`` (no secret substring) still failed the allow gate — every ``git commit``
a sandbox script made fell back to the repository's ``user.name``/``user.email`` (#155010). The
terminal tool is unaffected because it blocks by name list, which is why the same commit
attributed differently depending on which tool ran it.
"""

import os
import subprocess
import sys
from unittest.mock import patch

import pytest

from tools.code_execution_env import _build_child_env, _scrub_child_env, is_secret_env_name

# git refuses a commit without both roles; DATE completes the identity git composes.
IDENTITY = {
    "GIT_AUTHOR_NAME": "Agent", "GIT_AUTHOR_EMAIL": "agent@example.com",
    "GIT_AUTHOR_DATE": "2026-11-13T20:05:00+08:00",
    "GIT_COMMITTER_NAME": "Agent", "GIT_COMMITTER_EMAIL": "agent@example.com",
    "GIT_COMMITTER_DATE": "2026-11-13T20:05:00+08:00",
}
# Real credentials that the AUTH rule exists for: an exemption keyed on the exact identity names
# must not become "AUTHOR is allowed" or "GIT_ is allowed".
CREDENTIALS = {
    "HTTP_AUTHORIZATION": "Bearer fake", "GITHUB_TOKEN": "fake", "OPENAI_API_KEY": "fake",
    "NODE_AUTH_TOKEN": "fake", "SSH_AUTH_SOCK": "/run/ssh-agent",
}
ELSEWHERE_GIT_VARS = {"GIT_SSH_COMMAND": "ssh -i /home/x/.ssh/id", "GIT_TERMINAL_PROMPT": "0"}


def _no_passthrough(_name):
    return False


def _scrub(source):
    return _scrub_child_env(source, is_passthrough=_no_passthrough, is_windows=False)


def test_scrub_keeps_the_git_identity_names_verbatim():
    result = _scrub({"PATH": "/usr/bin", "HOME": "/home/x", **IDENTITY})
    assert {name: result.get(name) for name in IDENTITY} == IDENTITY


def test_the_allowlist_stays_exact_name_not_auth_or_git_prefix():
    source = {"PATH": "/usr/bin", **CREDENTIALS, **ELSEWHERE_GIT_VARS, **IDENTITY}
    result = _scrub(source)
    assert not set(CREDENTIALS) & set(result)
    assert not set(ELSEWHERE_GIT_VARS) & set(result)
    assert set(IDENTITY) <= set(result)


@pytest.mark.parametrize("name", sorted(IDENTITY))
def test_identity_names_are_not_classified_as_secrets(name):
    assert is_secret_env_name(name) is False


@pytest.mark.parametrize("name", sorted(CREDENTIALS))
def test_credential_names_keep_their_classification(name):
    assert is_secret_env_name(name) is True


def _git(repository, *args, env=None):
    result = subprocess.run(["git", "-C", str(repository), *args], capture_output=True,
                            text=True, check=False, env=env, stdin=subprocess.DEVNULL)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_a_sandbox_child_commits_as_the_configured_identity(tmp_path):
    """The reported symptom: the child's ``git var`` must not fall back to the repo's own identity."""
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Repo Default")
    _git(repository, "config", "user.email", "repo@example.com")

    with patch.dict(os.environ, {"PATH": os.environ.get("PATH", "/usr/bin"),
                                 "HOME": str(tmp_path), **IDENTITY}, clear=True):
        child_env = _build_child_env(rpc_endpoint="unix:///nonexistent.sock", rpc_token="token",
                                     tmpdir=str(tmp_path), child_python=sys.executable)

    ident = _git(repository, "var", "GIT_AUTHOR_IDENT", env=child_env).strip()
    assert ident.startswith("Agent <agent@example.com>"), ident

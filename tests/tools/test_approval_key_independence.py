"""Independent approval keys for shell-rc vs SSH-config writes (#85321)."""


class TestApprovalKeyIndependence:
    """Regression: a session approval for SSH client config must NOT
    authorize a shell startup file write (and vice versa). The two
    classes carry independent execution risks and are gated under
    independent approval keys. See issue #85321."""

    def test_ssh_config_approval_does_not_authorize_shell_rc(self, tmp_path, monkeypatch):
        from agent.file_safety import is_write_approval_required
        import tools.approval as _approval
        import tools.file_tools as ft

        fake_home = tmp_path / "home"
        fake_home.mkdir()
        bashrc = fake_home / ".bashrc"
        ssh_config = fake_home / ".ssh" / "config"
        ssh_config.parent.mkdir(parents=True)
        bashrc.touch()
        ssh_config.touch()

        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setenv("HERMES_HOME", str(fake_home / ".hermes"))
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.setenv("HERMES_SESSION_KEY", "test-key-independence")
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.delenv("HERMES_SINGLE_QUERY", raising=False)

        _approval.clear_session("test-key-independence")

        # Sanity: both targets are approval-gated.
        assert is_write_approval_required(str(bashrc)) is True
        assert is_write_approval_required(str(ssh_config)) is True

        # Approve ONLY the ssh_config_write key for this session.
        _approval.approve_session("test-key-independence", "ssh_config_write")

        # A shell-rc write must still be blocked: the gate sees
        # is_approved("...","shell_rc_write") is False and, with no
        # interactive callback wired, returns a blocked result.
        err = ft._check_approval_required_write([str(bashrc)])
        assert err is not None, (
            "shell rc write passed after an SSH-config-only session approval"
        )
        assert "BLOCKED" in err

        _approval.clear_session("test-key-independence")

    def test_shell_rc_approval_does_not_authorize_ssh_config(self, tmp_path, monkeypatch):
        import tools.approval as _approval
        import tools.file_tools as ft

        fake_home = tmp_path / "home"
        fake_home.mkdir()
        bashrc = fake_home / ".bashrc"
        ssh_config = fake_home / ".ssh" / "config"
        ssh_config.parent.mkdir(parents=True)
        bashrc.touch()
        ssh_config.touch()

        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setenv("HERMES_HOME", str(fake_home / ".hermes"))
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.setenv("HERMES_SESSION_KEY", "test-key-independence-2")
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.delenv("HERMES_SINGLE_QUERY", raising=False)

        _approval.clear_session("test-key-independence-2")

        # Approve ONLY the shell_rc_write key for this session.
        _approval.approve_session("test-key-independence-2", "shell_rc_write")

        # An SSH-config write must still be blocked.
        err = ft._check_approval_required_write([str(ssh_config)])
        assert err is not None, (
            "SSH config write passed after a shell-rc-only session approval"
        )
        assert "BLOCKED" in err

        _approval.clear_session("test-key-independence-2")

    def test_mixed_batch_requires_both_keys(self, tmp_path, monkeypatch):
        import tools.approval as _approval
        import tools.file_tools as ft

        fake_home = tmp_path / "home"
        fake_home.mkdir()
        bashrc = fake_home / ".bashrc"
        ssh_config = fake_home / ".ssh" / "config"
        ssh_config.parent.mkdir(parents=True)
        bashrc.touch()
        ssh_config.touch()

        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setenv("HERMES_HOME", str(fake_home / ".hermes"))
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.setenv("HERMES_SESSION_KEY", "test-key-mixed")
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.delenv("HERMES_SINGLE_QUERY", raising=False)

        _approval.clear_session("test-key-mixed")

        # A patch can carry several targets at once. Approving only the
        # shell-rc class must NOT leak into the SSH-config target of the
        # same batch: each class present needs its own approval.
        _approval.approve_session("test-key-mixed", "shell_rc_write")
        err = ft._check_approval_required_write([str(bashrc), str(ssh_config)])
        assert err is not None, (
            "mixed batch passed with only the shell-rc key approved"
        )
        assert "SSH client config" in err
        assert "BLOCKED" in err

        _approval.clear_session("test-key-mixed")

    def test_mixed_batch_passes_when_both_keys_approved(self, tmp_path, monkeypatch):
        import tools.approval as _approval
        import tools.file_tools as ft

        fake_home = tmp_path / "home"
        fake_home.mkdir()
        bashrc = fake_home / ".bashrc"
        ssh_config = fake_home / ".ssh" / "config"
        ssh_config.parent.mkdir(parents=True)
        bashrc.touch()
        ssh_config.touch()

        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setenv("HERMES_HOME", str(fake_home / ".hermes"))
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.setenv("HERMES_SESSION_KEY", "test-key-mixed-pass")
        monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
        monkeypatch.delenv("HERMES_SINGLE_QUERY", raising=False)

        _approval.clear_session("test-key-mixed-pass")

        _approval.approve_session("test-key-mixed-pass", "shell_rc_write")
        _approval.approve_session("test-key-mixed-pass", "ssh_config_write")
        err = ft._check_approval_required_write([str(bashrc), str(ssh_config)])
        assert err is None, f"mixed batch blocked with both keys approved: {err}"

        _approval.clear_session("test-key-mixed-pass")

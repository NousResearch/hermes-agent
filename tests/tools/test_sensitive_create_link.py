"""touch/mkdir/ln into credential, SSH, or shell-rc paths (#85321)."""

from tools.approval import detect_dangerous_command


class TestSensitiveCreateLinkPattern:
    """touch/mkdir/ln into credential, SSH, or shell-rc paths (#85321)."""

    def test_create_or_link_into_sensitive_path(self):
        for command in (
            "touch ~/.bashrc",
            "touch ~/.bash_login",
            "touch ~/.bash_aliases",
            "touch ~/.zshenv",
            "touch ~/.zlogin",
            "touch ~/.config/fish/config.fish",
            "touch ~/.ssh/authorized_keys",
            "mkdir ~/.ssh",
            "mkdir -p ~/.ssh/foo",
            "ln -s /tmp/evil ~/.bashrc",
            "ln /tmp/e ~/.ssh/authorized_keys",
            "touch $ZDOTDIR/.zshrc",
            "touch $XDG_CONFIG_HOME/fish/config.fish",
        ):
            dangerous, key, desc = detect_dangerous_command(command)
            assert dangerous is True, command
            assert key is not None, command

    def test_custom_rc_under_home_keeps_exact_filename_boundary(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_REAL_HOME", str(tmp_path))
        for variable in ("BASH_ENV", "ENV"):
            target = tmp_path / f"{variable.lower()}-startup"
            monkeypatch.setenv(variable, str(target))
            assert detect_dangerous_command(f"touch {target}")[0] is True
            assert detect_dangerous_command(f"touch {target}.bak")[0] is False

    def test_unrelated_touch_mkdir_ln_safe(self):
        for cmd in (
            "touch notes.txt",
            "mkdir src",
            "ln -s a.txt b.txt",
            "touch ~/.bashrc_dir",
            "touch ~/.zshrc.bak",
            "touch ~/.netrc-extra",
            "touch ~/.hermes/config.yaml.bak",
            "touch ~/.ssh-notes.txt",
        ):
            dangerous, key, desc = detect_dangerous_command(cmd)
            assert dangerous is False, cmd

    def test_relocated_shell_rc_absolute_paths_are_flagged(self, tmp_path, monkeypatch):
        zsh_dir = tmp_path / "zsh"
        xdg_dir = tmp_path / "xdg"
        monkeypatch.setenv("ZDOTDIR", str(zsh_dir))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_dir))

        for command in (
            f"touch {zsh_dir}/.zshrc",
            f"touch {xdg_dir}/fish/config.fish",
        ):
            dangerous, key, desc = detect_dangerous_command(command)
            assert dangerous is True, command
            assert key is not None, command

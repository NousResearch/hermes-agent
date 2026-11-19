"""Terminal approval hardening: classify strings only, never execute them."""

import pytest

from tools.approval import detect_dangerous_command


@pytest.fixture(autouse=True)
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home" / ".hermes"))


@pytest.mark.parametrize("command", [
    "echo x > app/.envrc", "echo x | tee .envrc", "cp input .envrc",
])
def test_envrc_writes_require_approval(command):
    assert detect_dangerous_command(command)[0]
    assert not detect_dangerous_command(command.replace('.envrc', '.envrc.example'))[0]
    assert not detect_dangerous_command('cat app/.envrc')[0]


@pytest.mark.parametrize("name", [".zshenv", ".zlogin", ".zlogout", ".bash_aliases", ".bash_login", ".bash_logout", ".kshrc", ".cshrc", ".tcshrc", ".login"])
def test_shell_startup_shared_inventory(name, tmp_path):
    from agent.file_safety import _approval_paths_by_key

    home = str(tmp_path / "home")
    assert _approval_paths_by_key(home).get(f"{home}/{name}") == "shell_rc_write"
    assert detect_dangerous_command(f"echo x > ~/{name}")[0]
    assert not detect_dangerous_command(f"cat ~/{name}")[0]
    assert not detect_dangerous_command(f"echo x > ~/{name}x")[0]


@pytest.mark.parametrize("command", ["echo x > ~/.git-credentials", "echo x | tee ~/.git-credentials", "cp input ~/.git-credentials", "sed -i 's/a/b/' ~/.git-credentials"])
def test_git_credentials_writes(command):
    assert detect_dangerous_command(command)[0]
    assert not detect_dangerous_command("cat ~/.git-credentials")[0]
    assert not detect_dangerous_command("cp ~/.git-credentials backup")[0]


@pytest.mark.parametrize("directory", [".aws", ".gnupg", ".kube", ".docker", ".azure", ".config/gh", ".config/gcloud"])
@pytest.mark.parametrize("template", ["echo x > ~/{}/token", "cp input ~/{}/token", "sed -i 's/a/b/' ~/{}/token"])
def test_home_credential_directory_writes(directory, template):
    assert detect_dangerous_command(template.format(directory))[0]
    assert not detect_dangerous_command(template.format(directory + '-backup'))[0]
    assert not detect_dangerous_command(f"cat ~/{directory}/token")[0]
    assert not detect_dangerous_command(f"cp ~/{directory}/token backup")[0]


def test_sed_posix_is_read_only():
    assert not detect_dangerous_command("sed --posix s/a/b/ ~/.bashrc")[0]
    assert detect_dangerous_command("sed -n -i s/a/b/ ~/.bashrc")[0]


def test_dot_component_home_write():
    assert detect_dangerous_command("echo x > ~/./.bashrc")[0]
    assert not detect_dangerous_command("cat ~/./.bashrc")[0]


def test_anthropic_oauth_write():
    assert detect_dangerous_command("echo x > ~/.hermes/.anthropic_oauth.json")[0]
    assert not detect_dangerous_command("cat ~/.hermes/.anthropic_oauth.json")[0]

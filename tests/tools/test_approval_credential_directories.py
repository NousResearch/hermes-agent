"""Credential-directory terminal write contracts (#99201).

Classification only: command strings are never executed. Read/copy-out safety here
is a detector contract, not a claim that file-read guards cover these directories.

Absolute /root spellings depend on the shared home normalizer (#98868); reordered
sed options and read-only --posix depend on the shared sed grammar (#99228).
These are asserted normally, not skipped, so integration must close both holes.
"""

import pytest

from tools.approval import detect_dangerous_command


DIRECTORIES = (".aws", ".gnupg", ".kube", ".docker", ".azure", ".config/gh", ".config/gcloud")


@pytest.mark.parametrize("home", ["/home/credential-test-user", "/root"])
@pytest.mark.parametrize("directory", DIRECTORIES)
@pytest.mark.parametrize("spelling", ["~", "$HOME", "${HOME}", "absolute"])
@pytest.mark.parametrize("vector", [
    "echo x >> {p}/arbitrary/nested/token", "echo x > {p}/token",
    "echo x | tee -a {p}/token", "cp input {p} && true",
    'mv input "{p}"; true', 'install -m600 input "{p}" || true',
    "sed -i 's/a/b/' {p}/token", "sed -n -i 's/a/b/' {p}/token",
    "perl -pi 's/a/b/' {p}/token", "ruby -i -pe 'gsub(/a/, \"b\")' {p}/token",
])
def test_credential_directory_mutations_require_approval(monkeypatch, tmp_path, home, directory, spelling, vector):
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    prefix = home if spelling == "absolute" else spelling
    command = vector.format(p=f"{prefix}/{directory}")
    dangerous, key, reason = detect_dangerous_command(command)
    assert dangerous and key and reason, command


@pytest.mark.parametrize("home", ["/home/credential-test-user", "/root"])
@pytest.mark.parametrize("directory", DIRECTORIES)
@pytest.mark.parametrize("control", ["read-copy-out-near-miss", "sed-read-only"])
def test_credential_directory_controls_stay_safe(monkeypatch, tmp_path, home, directory, control):
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    commands = (
        f"echo x >> ~/{directory}-backup/token",
        f"cp input ~/{directory}-backup && true",
        f"mv input ~/{directory}-backup; true",
        f"install input ~/{directory}-backup || true",
        f"echo x > ~/{directory}#backup",
        f"cp ~/{directory}/token /tmp/token-copy",
        f"cat ~/{directory}/token",
        f"grep -r token ~/{directory}/token",
        "echo x > ~/.config/nvim/init.lua",
        "cp input ~/.config/Code/User/settings.json",
        "sed -i 's/a/b/' ~/.config/systemd/user/app.service",
    ) if control == "read-copy-out-near-miss" else (
        f"sed --posix 's/a/b/' ~/{directory}/token",
        f"sed -n 's/a/b/p' ~/{directory}/token",
    )
    for command in commands:
        assert detect_dangerous_command(command) == (False, None, None), command

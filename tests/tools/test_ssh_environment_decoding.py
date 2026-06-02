from types import SimpleNamespace

from tools.environments import ssh as ssh_env


def test_ensure_remote_dirs_decodes_subprocess_output_tolerantly(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    env = object.__new__(ssh_env.SSHEnvironment)
    env._remote_home = "/home/testuser"
    env._build_ssh_command = lambda: ["ssh", "example.com"]

    monkeypatch.setattr(ssh_env.subprocess, "run", fake_run)

    env._ensure_remote_dirs()

    assert calls
    _, kwargs = calls[0]
    assert kwargs["text"] is True
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "replace"


def test_ssh_subprocess_text_captures_all_use_tolerant_decoding():
    source = ssh_env.Path(ssh_env.__file__).read_text(encoding="utf-8")
    snippets = [
        line for line in source.splitlines()
        if "subprocess.run(" in line and "capture_output=True" in line and "text=True" in line
    ]

    assert snippets
    for line in snippets:
        assert 'encoding="utf-8"' in line
        assert 'errors="replace"' in line

from hermes_cli.search_backends import SearchBackendMatch, SearchBackendResult, register_search_backend
from tools.file_operations import ShellFileOperations


class RecordingLocalEnvironment:
    is_local = True
    def __init__(self, root):
        self.root = str(root)
        self.cwd = self.root
        self.config = type("Config", (), {"cwd": self.root})()
        self.commands = []

    def execute(self, command, **kwargs):
        self.commands.append(command)
        if command.startswith("test -e "):
            return {"output": "exists\n", "returncode": 0}
        if command.startswith("command -v rg"):
            return {"output": "/usr/bin/rg\n", "returncode": 0}
        if "--line-number" in command:
            return {"output": f"{self.root}/native.py:2:Needle\n", "returncode": 0}
        return {"output": "", "returncode": 1}


def test_content_search_uses_backend_then_falls_back_on_backend_error(monkeypatch, tmp_path):
    from hermes_constants import hermes_home_key

    scope = hermes_home_key()
    repo = tmp_path / "repo"
    repo.mkdir()
    fast = repo / "fast.py"
    native = repo / "native.py"
    fast.write_text("\n" * 6 + "Needle\n")
    native.write_text("\nNeedle\n")
    env = RecordingLocalEnvironment(repo)
    ops = ShellFileOperations(env)
    selected = register_search_backend(
        "fast",
        lambda _request: SearchBackendResult(
            backend="fast",
            route_reason="eligible",
            matches=[SearchBackendMatch(path=str(fast), line_number=7, content="Needle")],
            total_count=1,
        ),
        scope=scope,
    )
    result = ops.search("Needle", path=str(repo), target="content")
    assert [(m.path, m.line_number) for m in result.matches] == [(str(fast), 7)]
    assert result.backend == "fast"
    assert result.route_reason == "eligible"
    assert not any("--line-number" in command for command in env.commands)
    selected.dispose()

    failed = register_search_backend(
        "broken", lambda _request: (_ for _ in ()).throw(RuntimeError("boom")), scope=scope)
    result = ops.search("Needle", path=str(repo), target="content")
    assert [(m.path, m.line_number) for m in result.matches] == [(str(native), 2)]
    assert result.backend == "rg"
    assert result.route_reason == "backend_error:broken"
    failed.dispose()

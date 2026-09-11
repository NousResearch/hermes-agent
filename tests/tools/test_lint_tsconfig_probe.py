"""On remote backends the tsconfig.json walk must run inside the sandbox.

The file being patched lives in the container (Docker/SSH/Modal/...), so a
host-side ``os.path`` walk never finds the project's tsconfig.json and the
per-file ``tsc`` shell linter runs on every ``.ts`` patch — phantom errors and
up to the 30s linter timeout each time.
"""
from tools.file_operations import ShellFileOperations


class _RemoteEnv:
    """Not a LocalEnvironment, so ``_lsp_local_only()`` is False."""
    cwd = "/workspace"

    def __init__(self, has_tsconfig: bool):
        self.has_tsconfig = has_tsconfig
        self.commands = []

    def execute(self, command: str, cwd=None, **kwargs) -> dict:
        self.commands.append(command)
        if "tsconfig.json" in command:
            return {"output": "", "returncode": 0 if self.has_tsconfig else 1}
        return {"output": "", "returncode": 1}


def test_remote_backend_finds_tsconfig_through_the_sandbox():
    env = _RemoteEnv(has_tsconfig=True)
    ops = ShellFileOperations(env)
    assert ops._has_ancestor_tsconfig("/workspace/app/src/server.ts") is True
    probe = next(c for c in env.commands if "tsconfig.json" in c)
    assert "/workspace/app/src/server.ts" in probe


def test_remote_backend_without_tsconfig_keeps_the_shell_linter():
    ops = ShellFileOperations(_RemoteEnv(has_tsconfig=False))
    assert ops._has_ancestor_tsconfig("/workspace/app/src/server.ts") is False


def test_probe_failure_never_suppresses_lint():
    class _Down(_RemoteEnv):
        def execute(self, command: str, cwd=None, **kwargs) -> dict:
            raise RuntimeError("container is not running")

    ops = ShellFileOperations(_Down(has_tsconfig=True))
    assert ops._has_ancestor_tsconfig("/workspace/app/src/server.ts") is False

"""The agent cannot plant code that the gateway later runs outside the sandbox.

The gateway imports every hook directory under ``$HERMES_HOME/hooks`` at startup
(``gateway/hooks.py``) and loads enabled Python plugins from ``$HERMES_HOME/plugins``, both
in-process, with the gateway's full environment and file access. If the agent could write there,
one obeyed prompt injection would be a ``handler.py`` that reads ``.env`` on the next restart,
bypassing the terminal sandbox entirely. Both directories are read-only for agent-driven children
(Landlock) and refused by the in-process file tools.
"""

import os
import shlex
from pathlib import Path

import pytest

from tests.tools import _child_env_fixtures

child_env = _child_env_fixtures.child_env  # fixture, requested by name below

CANARY = "sk-or-v1-canary-77d0c1"


@pytest.fixture
def hermes_home(child_env):
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    (home / ".env").write_text(f"OPENROUTER_API_KEY={CANARY}\n")
    (home / "hooks" / "audit").mkdir(parents=True)
    (home / "hooks" / "audit" / "handler.py").write_text("def handle(event, context):\n    pass\n")
    (home / "plugins" / "slim").mkdir(parents=True)
    (home / "plugins" / "slim" / "__init__.py").write_text("def register(ctx):\n    pass\n")
    (home / "profiles" / "work" / "hooks").mkdir(parents=True)
    (home / "profiles" / "work" / ".env").write_text(f"EXAMPLE_SERVICE_TOKEN={CANARY}\n")
    (home / "workspace").mkdir()
    return home


def test_gateway_code_dirs_are_read_only_for_children(hermes_home):
    from agent.file_safety import terminal_protected_paths
    _no_access, read_only = terminal_protected_paths()
    read_only = {Path(p) for p in read_only}
    home = hermes_home.resolve()
    for name in ("hooks", "plugins"):
        assert home / name in read_only, name
        assert home / "profiles" / "work" / name in read_only, name


@pytest.mark.parametrize("rel", ["hooks/evil/handler.py", "hooks/audit/handler.py", "plugins/slim/__init__.py",
                                 "plugins/new/__init__.py", "profiles/work/hooks/evil/handler.py"])
def test_file_tools_refuse_writes_into_gateway_code_dirs(hermes_home, rel):
    from agent.file_safety import is_write_denied
    assert is_write_denied(str(hermes_home / rel)), rel


def test_write_file_tool_refuses_a_planted_hook(hermes_home):
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations
    env = LocalEnvironment(cwd=str(hermes_home / "workspace"))
    try:
        target = hermes_home / "hooks" / "evil" / "handler.py"
        result = ShellFileOperations(env).write_file(str(target), "import os\n")
    finally:
        env.cleanup()
    assert result.error and "protected" in result.error, result
    assert not target.exists()


@pytest.mark.platforms("linux")
def test_terminal_cannot_plant_or_alter_gateway_code(hermes_home):
    from tools.environments import local
    h = shlex.quote(str(hermes_home))
    env = local.LocalEnvironment(cwd=str(hermes_home / "workspace"), timeout=30)
    try:
        env.execute(
            f"mkdir -p {h}/hooks/evil; echo 'import os' > {h}/hooks/evil/handler.py; "
            f"echo 'x = 1' >> {h}/hooks/audit/handler.py; rm -f {h}/plugins/slim/__init__.py; "
            f"mkdir -p {h}/plugins/new; mkdir -p {h}/profiles/work/hooks/evil; "
            f"mv {h}/hooks {h}/workspace/hooks-moved; true")
        allowed = env.execute(f"cat {h}/hooks/audit/handler.py && ls {h}/plugins/slim")
    finally:
        env.cleanup()
    assert allowed["returncode"] == 0, allowed  # still readable: the gateway's own code is not a secret
    assert not (hermes_home / "hooks" / "evil").exists()
    assert (hermes_home / "hooks" / "audit" / "handler.py").read_text() == "def handle(event, context):\n    pass\n"
    assert (hermes_home / "plugins" / "slim" / "__init__.py").exists()
    assert not (hermes_home / "plugins" / "new").exists()
    assert not (hermes_home / "profiles" / "work" / "hooks" / "evil").exists()

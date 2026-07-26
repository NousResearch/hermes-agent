"""The compound-background rewriter is retired; commands must reach bash verbatim.

``_rewrite_compound_background`` textually rewrote ``A && B &`` into
``A && { B & }`` so a backgrounded compound couldn't leak a subshell stuck in
``wait4`` on a long-running B (the vela/sal/combiagent fleet leaks; #68915).
The worker hang that made the leak urgent was fixed at the process layer in
#71008 (orphan-held stdout pipes), and review of the rewriter (#68948) kept
finding inputs where the textual scan turned valid bash into invalid bash or
silently changed program data:

- ``echo `A && B` &``        -> unmatched-backtick syntax error
- ``echo ${x:-A&&B} &``      -> broken expansion
- ``[[ -n x && -n y ]] &``   -> broken conditional
- ``echo $[1&&2] &``         -> broken legacy arithmetic
- ``a[1&&2]=x &``            -> broken array subscript
- a heredoc payload containing ``A && B &``  -> payload data changed
- ``$'...'`` ANSI-C strings with ``\\'``      -> string data changed
- ``false && echo B &``      -> observable ``$?`` changed (0 -> 1)

Every scanner marker added for one of these surfaced the next; syntax created
at runtime (alias expansion, ``eval``) is out of reach of ANY pre-execution
textual check.  So the rewrite is removed instead of patched again.  These
tests pin the retirement at two depths: a seam probe (nothing transforms the
command before ``_wrap_command``), and a ``subprocess.Popen`` capture on the
concrete local backends (the exact argv bash receives) so a rewrite hidden
inside ``_wrap_command`` or ``_run_bash`` cannot slip past either.
"""

import inspect
import os

import pytest

import tools.process_registry as process_registry
import tools.terminal_tool as terminal_tool
from tools.environments import base as env_base
from tools.environments import local as env_local

# Inputs the retired rewriter provably corrupted (syntax or data), plus the
# ``A && B &`` shape it was built to transform.  If any transformation
# reappears on the execute path, at least one of these identity assertions
# fails and points here.
CORRUPTION_CLASS = [
    "A && B &",
    "A || B &",
    "echo `A && B` &",
    "echo ${x:-A&&B} &",
    "[[ -n x && -n y ]] &",
    "echo $[1&&2] &",
    "a[1&&2]=x &",
    'echo "x`printf "%s && %s" A B`y" &',
    "read -r x <<'EOF'\nA && B &\nEOF\nprintf '<%s>\\n' \"$x\"",
    "printf '%s\\n' $'prefix\\' A && B &\nsuffix'",
    "false && echo B &\nprintf 'status=%s\\n' \"$?\"\nwait",
]


def test_rewriter_is_gone():
    assert not hasattr(terminal_tool, "_rewrite_compound_background")


def test_execute_has_no_rewrite_parameter():
    sig = inspect.signature(env_base.BaseEnvironment.execute)
    assert "rewrite_compound_background" not in sig.parameters


class _ProbeEnv(env_base.BaseEnvironment):
    """Concrete environment that records what reaches ``_wrap_command`` --
    the exact seam the retired rewriter used to sit in front of."""

    def __init__(self):
        self.timeout = 5
        self.cwd = ""
        self._stdin_mode = "none"
        self._snapshot_ready = True
        self._prefer_nonlogin = False
        self.seen = []

    def _before_execute(self):
        pass

    def _prepare_command(self, command):
        return command, None

    def _wrap_command(self, command, cwd):
        self.seen.append(command)
        return command

    def _run_bash(self, command, *, login=False, timeout=None, stdin_data=None):
        return None

    def _wait_for_process(self, proc, *, timeout=None, bounded_capture=False):
        return {"output": "", "returncode": 0}

    def _update_cwd(self, result):
        pass

    def cleanup(self):
        pass


@pytest.mark.parametrize("command", CORRUPTION_CLASS)
def test_execute_passes_command_verbatim(command):
    """execute() must hand the prepared command to _wrap_command
    byte-identical: nothing may transform it on the way."""
    env = _ProbeEnv()
    env.execute(command)
    assert env.seen == [command]


class _ArgvRecordingProc:
    """Popen stand-in: satisfies the minimal lifecycle execute()/spawn_local()
    drive after spawning (poll/wait/reader), so the test can assert on the
    captured argv without running a real shell."""

    def __init__(self):
        self.pid = 4242
        self.stdout = None
        self.returncode = 0

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


@pytest.fixture
def _argv_capture(monkeypatch):
    """Capture the final subprocess.Popen argv on both local backends.

    Only the two shell-invocation shapes under test are intercepted;
    everything else (Windows shell/ASLR probes, _find_bash checks) is
    delegated to the real Popen so their module-level caches stay truthful."""
    seen = []
    real_popen = env_local.subprocess.Popen

    def _fake_popen(args, **kwargs):
        argv = list(args) if isinstance(args, (list, tuple)) else [args]
        if len(argv) == 3 and argv[1] in ("-c", "-lic"):
            seen.append(argv)
            return _ArgvRecordingProc()
        return real_popen(args, **kwargs)

    monkeypatch.setattr(env_local.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(process_registry.subprocess, "Popen", _fake_popen)
    return seen


@pytest.fixture
def _local_env(monkeypatch):
    """A real LocalEnvironment minus the login-shell snapshot bootstrap.

    init_session is stubbed out (it spawns a real login bash); with
    ``_prefer_nonlogin`` set, execute() takes the plain ``bash -c`` path with
    no init-file prepend, so the wrapped script is fully deterministic."""
    monkeypatch.setattr(env_local.LocalEnvironment, "init_session", lambda self: None)
    env = env_local.LocalEnvironment(cwd=os.getcwd())
    env._snapshot_ready = False
    env._prefer_nonlogin = True
    return env


@pytest.mark.parametrize("command", CORRUPTION_CLASS)
def test_local_execute_final_bash_argv_is_verbatim(command, _argv_capture, _local_env):
    """The argv LocalEnvironment hands to Popen is what bash receives — the
    boundary the retired rewriter can no longer sit in front of.  The wrapper
    embeds the user command as ``eval '<escaped>'`` where the only permitted
    transformation is the documented single-quote escape; asserting that exact
    payload pins the command body byte-identical through _prepare_command,
    _wrap_command, and _run_bash at once."""
    _local_env.execute(command)
    assert len(_argv_capture) == 1
    args = _argv_capture[0]
    assert len(args) == 3 and args[1] == "-c"  # plain non-login foreground shape
    escaped = command.replace("'", "'\\''")
    assert f"eval '{escaped}'" in args[2]


@pytest.mark.parametrize("command", CORRUPTION_CLASS)
def test_spawn_local_final_shell_argv_is_verbatim(command, _argv_capture, monkeypatch, tmp_path):
    """spawn_local's contract is ``[shell, -lic, "set +m; <command>"]`` with the
    command verbatim — full argv equality, so ANY reintroduced transformation
    (including substring-preserving wrappers) fails here."""
    # CHECKPOINT_PATH is resolved at import time, before conftest's per-test
    # HERMES_HOME redirect — repoint it so the test never touches the real one.
    monkeypatch.setattr(process_registry, "CHECKPOINT_PATH", tmp_path / "processes.json")
    reg = process_registry.ProcessRegistry()
    session = reg.spawn_local(command)
    assert len(_argv_capture) == 1
    args = _argv_capture[0]
    assert args[1:] == ["-lic", f"set +m; {command}"]
    assert session.command == command

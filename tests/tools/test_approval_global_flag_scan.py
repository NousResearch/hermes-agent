"""Global-flag and command-position rules must terminate on long flag and whitespace runs without
dropping long commands (#129281).

Subprocess timeout bounds regressions without hanging pytest on the GIL.
"""
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run(code):
    # Pin the checkout under test: a bare `python -c` resolves `tools` through the venv's editable
    # install (the primary clone), not through the worktree pytest is running in.
    env = {**os.environ, "PYTHONPATH": REPO_ROOT}
    subprocess.run([sys.executable, '-c', code], check=True, timeout=10, cwd=REPO_ROOT, env=env)


def test_long_flag_runs_still_reach_the_target():
    _run('''
from tools.approval_detection import detect_dangerous_command
run = "--opt val --flag=x -q - " * 58
cases = {
    f"hermes {run}gateway restart": "stop/restart hermes gateway (kills running agents)",
    f"docker {run}-H ssh://prod ps": "docker with remote daemon redirect (-H/--host)",
    f"docker {run}--context=prod ps": "docker with daemon redirect (--context: alternate daemon)",
    f"podman {run}--url tcp://prod ps": "podman with remote daemon redirect (--url/--connection/--identity)",
    f"podman {run}--remote ps": "podman remote mode (-r/--remote: remote daemon)",
    f"docker compose {run}down": "docker compose restart/stop/kill/down (container lifecycle)",
    f"docker {run}kill app": "docker restart/stop/kill (container lifecycle)",
}
for command, description in cases.items():
    assert detect_dangerous_command(command) == (True, description, description), command[:40]
''')


def test_nonmatching_flag_runs_finish():
    _run('''
from tools.approval_detection import detect_dangerous_command
for prefix in ("rclone lsf $HOME/.hermes --recursive --files-only", "hermes", "docker",
               "docker compose", "podman"):
    for run in ("--exclude " * 58, "--opt val " * 58, "--opt=val " * 58, "--opt" + " " * 100 + "val "):
        assert detect_dangerous_command(f"{prefix} {run}ps 2>/dev/null")[0] is False, prefix
''')


def test_value_whitespace_decisions_are_unchanged():
    # The fix must not move any approval decision: docker/podman values follow exactly one
    # whitespace character, as before, while hermes values may follow a whitespace run.
    _run('''
from tools.approval_detection import detect_dangerous_command
cases = {
    "docker --log-level debug stop app": True,
    "docker --log-level\\tdebug stop app": True,
    "docker --log-level  debug stop app": False,
    "docker --log-level  debug -H ssh://prod ps": False,
    "docker --log-level  debug --context prod ps": False,
    "podman --log-level  debug --remote ps": False,
    "podman --log-level  debug --url tcp://prod ps": False,
    "docker compose --project-name  demo down": False,
    "docker compose --project-name demo  down": True,
    "hermes --config  x.yaml gateway restart": True,
}
for command, dangerous in cases.items():
    assert detect_dangerous_command(command)[0] is dangerous, command
''')


def test_whitespace_runs_at_a_command_position_finish():
    # Every _CMDPOS rule re-split a blank-line run between its leading and trailing whitespace from
    # each newline of the run: cubic in the run length, inside one GIL-holding re.search.
    _run('''
from tools.approval_detection import DANGEROUS_PATTERNS_COMPILED, HARDLINE_PATTERNS_COMPILED, _CMDPOS
rules = [entry[0] for entry in HARDLINE_PATTERNS_COMPILED + DANGEROUS_PATTERNS_COMPILED
         if entry[0].pattern.startswith(_CMDPOS)]
mkfs = [rx for rx in rules if "mkfs" in rx.pattern]
assert len(rules) > 10 and mkfs
for run in ("\\n" * 24000, " \\n" * 12000, "\\t\\n" * 12000, "sudo" + " " * 24000, "env" + " " * 24000,
            "exec" + " " * 24000):
    assert not any(rx.search(run + "x") for rx in rules), run[:6]
    assert all(rx.search(run + "\\nmkfs.ext4 /dev/sda1") for rx in mkfs), run[:6]
''')


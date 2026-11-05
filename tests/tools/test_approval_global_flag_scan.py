"""Global-flag rules must terminate on long flag runs without dropping long commands (#129281).

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


def test_value_whitespace_runs_reach_the_container_verb():
    # docker/podman values may follow a whitespace run, like hermes values always could: the shell
    # collapses the run before docker sees the command, so a value behind two spaces used to end the
    # flag group and hide the verb (#130511). Single-space commands keep their exact decisions.
    _run('''
from tools.approval_detection import detect_dangerous_command
cases = {
    "docker --log-level debug stop app": True,
    "docker --log-level\\tdebug stop app": True,
    "docker --log-level  debug stop app": True,
    "docker --log-level  debug -H ssh://prod ps": True,
    "docker --log-level  debug --context prod ps": True,
    "podman --log-level  debug --remote ps": True,
    "podman --log-level  debug --url tcp://prod ps": True,
    "docker compose --project-name  demo down": True,
    "docker compose --project-name demo  down": True,
    "hermes --config  x.yaml gateway restart": True,
}
for command, dangerous in cases.items():
    assert detect_dangerous_command(command)[0] is dangerous, command
''')


def test_whitespace_run_before_a_value_cannot_hide_the_verb():
    # The measured #130511 spellings: two spaces (or tabs) between a container flag and its value
    # used to leave the lifecycle verb past the flag group's reach. Harmless verbs stay benign.
    _run('''
from tools.approval_detection import detect_dangerous_command
dangerous = {
    "docker --log-level  warn stop web": True,
    "docker --log-level  warn restart web": True,
    "docker --log-level  warn kill web": True,
    "docker --log-level\\t\\twarn stop web": True,
    "docker compose --profile  prod down": True,
}
benign = {
    "docker --log-level  warn ps": False,
    "docker --log-level  warn logs web": False,
    "docker compose --profile  prod config": False,
    # compose must directly follow docker; a flag before the subcommand stays out with one space
    # too, so the whitespace alignment does not reach it either.
    "docker --log-level warn compose down": False,
    "docker --log-level  warn compose down": False,
    # podman has no lifecycle rule (its grammar feeds the remote-daemon rules, covered above);
    # the verb stays out with one space too.
    "podman --log-level warn stop web": False,
    "podman --log-level  warn stop web": False,
}
for command, expected in {**dangerous, **benign}.items():
    assert detect_dangerous_command(command)[0] is expected, command
''')

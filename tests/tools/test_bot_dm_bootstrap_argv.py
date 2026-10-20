"""Regression guard: the Bot-Mode DM runner must carry its own import path.

Measured 2026-09-27 (live failure): ``_delivery_command`` spawned
``[sys.executable, bot_mode_dm.py, "--run-delivery", ...]``. When ``sys.executable`` is the bare
store python — exactly the case for a tool running inside a bootstrapped agent process, whose
dependency activation is in-process only and inherited by nobody — the child died at its first
Hermes helper import with ``ModuleNotFoundError: No module named 'ruamel'``
(``tools.bot_live_delivery`` -> ``utils`` -> ``hermes_yaml`` -> ruamel), and the sender saw
``Live admission outcome unknown: No module named 'ruamel'``. The same shape already fixed the
kanban worker argv (``hermes_cli.kanban_db_dispatch._module_hermes_argv``).

The end-to-end proof lives OUTSIDE this file: the emitted argv was run with the real store python,
off-cwd with a stripped environment, and the child reached ``tools.bot_live_delivery`` (rc 0). A
functional copy of that check cannot live here — this suite runs under an isolated ``HERMES_HOME``
(``tests/home_io_guard.py``), where ``hermes_bootstrap`` has no dependency environment to lease, so
the child would fail for a reason unrelated to this fix. These tests therefore pin the argv
CONTRACT: dependency activation before the entry point, and never a bare launch.
"""

from __future__ import annotations

import pathlib
import shlex
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import bot_mode_dm  # noqa: E402

SCRIPT = str(REPO / "tools" / "bot_mode_dm.py")


def test_runner_argv_is_self_importing():
    argv = bot_mode_dm._runner_bootstrap_argv(SCRIPT)

    assert len(argv) == 5, argv
    assert argv[1:3] == ["-I", "-c"], argv
    assert argv[4] == SCRIPT, argv


def test_runner_argv_is_not_a_bare_launch():
    """The pre-fix shape was ``[sys.executable, script]`` — the script must not be argv[1]."""
    argv = bot_mode_dm._runner_bootstrap_argv(SCRIPT)

    assert argv[1] != SCRIPT, argv
    assert "-c" in argv[1:3], argv


def test_bootstrap_activates_dependencies_before_the_entry_point():
    """Order is the fix: env scrub -> checkout root -> hermes_bootstrap -> run the script."""
    code = bot_mode_dm._runner_bootstrap_argv(SCRIPT)[3]

    assert str(REPO) in code, code
    for step in ("import hermes_bootstrap", "runpy.run_path", "sys.argv = sys.argv[1:]"):
        assert step in code, (step, code)
    assert code.index("import hermes_bootstrap") < code.index("runpy.run_path"), code
    assert code.index(f"sys.path.insert(0, {str(REPO)!r})") < code.index("import hermes_bootstrap"), code
    # A dependency-less child must not inherit a hostile interpreter environment.
    for scrubbed in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        assert f"pop('{scrubbed}'" in code, (scrubbed, code)


def test_delivery_command_keeps_target_argv_after_the_bootstrap(tmp_path):
    dm_file = str(tmp_path / "dm.txt")
    target = ["hermes", "-p", "ops", "chat", "--query-file", dm_file]
    cmd = bot_mode_dm._delivery_command(target, dm_file, stdin_file=False)
    parts = shlex.split(cmd)

    assert parts[1:3] == ["-I", "-c"], parts
    assert "import hermes_bootstrap" in parts[3], parts
    assert parts[4] == SCRIPT, parts
    # Runner args stay intact and in order, so _delivery_main parses them exactly as before.
    assert parts[5:7] == ["--run-delivery", "query-file"], parts
    assert parts[7] == dm_file, parts
    assert parts[-len(target):] == target, parts
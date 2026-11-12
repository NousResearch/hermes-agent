"""The protected agent-instruction gate must cover execute_code and terminal, not just the
file tools (#143266).

Two layers are covered here:

* the child-side audit hook, exercised by running the generated guard module in a real
  subprocess and attempting every write spelling against a protected name;
* the host-side terminal check, exercised through ``terminal_write_targets`` and the
  ``_run_approval_guards`` wiring that makes ``force``/``--yolo`` unable to skip it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tools import protected_instruction_guard as pig


# --- child-side guard, in a real subprocess ------------------------------------------------


def _run_guarded(
    tmp_path: Path,
    body: str,
    *,
    spec: dict | None = None,
    env_override: str | None = None,
) -> subprocess.CompletedProcess:
    """Run *body* under the generated guard module in a fresh interpreter."""
    guard = tmp_path / pig.CHILD_GUARD_FILENAME
    guard.write_text(
        pig.child_guard_source(spec or {"basenames": list(pig.BASENAMES)}),
        encoding="utf-8",
    )
    script = tmp_path / "probe.py"
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    env = dict(os.environ)
    if env_override is not None:
        env[pig.SPEC_ENV] = env_override
    else:
        env[pig.SPEC_ENV] = json.dumps(
            spec
            or {
                "enabled": True,
                "basenames": list(pig.BASENAMES),
                "extra": [],
                "exempt": [],
            }
        )
    env["PYTHONPATH"] = str(tmp_path)
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.parametrize(
    "spelling",
    [
        "open(p, 'w').write('x')",
        "open(p, 'a').write('x')",
        "open(p, 'r+')",
        "os.close(os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC))",
        "pathlib.Path(p).write_text('x')",
        "pathlib.Path(p).touch()",
        "shutil.copyfile(src, p)",
        "shutil.move(src, p)",
        "os.replace(src, p)",
        "os.rename(src, p)",
        "os.remove(p)",
        "os.truncate(p, 0)",
        "os.chmod(p, 0o600)",
    ],
)
def test_child_guard_blocks_every_in_process_write_spelling(tmp_path, spelling):
    result = _run_guarded(
        tmp_path,
        f"""
        import os, pathlib, shutil
        import hermes_protected_instruction_guard  # noqa: F401
        p = "AGENTS.md"
        src = "plain.txt"
        open(src, "w").write("payload")
        open(p, "w").write("original")
        {spelling}
        print("WROTE")
    """,
    )
    assert "WROTE" not in result.stdout, result.stdout + result.stderr
    assert "BLOCKED" in result.stderr, result.stderr
    assert "hermes_tools.write_file" in result.stderr


def test_child_guard_blocks_the_spawned_shell_spelling(tmp_path):
    """A spawned process's writes never reach the hook, so the shell text is matched."""
    result = _run_guarded(
        tmp_path,
        """
        import subprocess
        import hermes_protected_instruction_guard  # noqa: F401
        subprocess.run("echo changed > AGENTS.md", shell=True)
        print("WROTE")
    """,
    )
    assert "WROTE" not in result.stdout, result.stdout + result.stderr
    assert "BLOCKED" in result.stderr


def test_child_guard_leaves_reads_and_unrelated_writes_alone(tmp_path):
    result = _run_guarded(
        tmp_path,
        """
        import os, pathlib
        import hermes_protected_instruction_guard  # noqa: F401
        open("AGENTS.md", "w").write("seed")
    """,
    )
    # The seed write itself is blocked; verify the read path and an unrelated name are free.
    result = _run_guarded(
        tmp_path,
        """
        import os, pathlib
        import hermes_protected_instruction_guard  # noqa: F401
        open("AGENTS.md", "r").read() if os.path.exists("AGENTS.md") else None
        open("notes.txt", "w").write("fine")
        pathlib.Path("build").mkdir(exist_ok=True)
        open("build/AGENTS.txt", "w").write("different file")
        print("OK")
    """,
    )
    assert "OK" in result.stdout, result.stdout + result.stderr


def test_child_guard_honours_the_hermes_home_exemption(tmp_path):
    home = tmp_path / "hermes-home"
    home.mkdir()
    target = home / "AGENTS.md"
    result = _run_guarded(
        tmp_path,
        f"""
        import hermes_protected_instruction_guard  # noqa: F401
        open({str(target)!r}, "w").write("agent's own store")
        print("OK")
    """,
        spec={
            "enabled": True,
            "basenames": list(pig.BASENAMES),
            "extra": [],
            "exempt": [str(home)],
        },
    )
    assert "OK" in result.stdout, result.stdout + result.stderr
    assert target.read_text(encoding="utf-8") == "agent's own store"


def test_child_guard_honours_extra_patterns(tmp_path):
    result = _run_guarded(
        tmp_path,
        """
        import hermes_protected_instruction_guard  # noqa: F401
        open("GEMINI.md", "w").write("x")
        print("WROTE")
    """,
        spec={
            "enabled": True,
            "basenames": list(pig.BASENAMES),
            "extra": ["gemini.md"],
            "exempt": [],
        },
    )
    assert "WROTE" not in result.stdout
    assert "BLOCKED" in result.stderr


def test_child_guard_fails_closed_without_a_spec(tmp_path):
    """A child spawned outside the guarded path must not run at all."""
    result = _run_guarded(
        tmp_path,
        """
        import hermes_protected_instruction_guard  # noqa: F401
        print("RAN")
    """,
        env_override="",
    )
    assert "RAN" not in result.stdout
    assert "no guard spec" in result.stderr


def test_child_guard_off_marker_disables_it(tmp_path):
    result = _run_guarded(
        tmp_path,
        """
        import hermes_protected_instruction_guard  # noqa: F401
        open("AGENTS.md", "w").write("x")
        print("OK")
    """,
        env_override="off",
    )
    assert "OK" in result.stdout, result.stdout + result.stderr


def test_child_guard_survives_a_cell_that_clears_its_globals(tmp_path):
    """The spec is bound into the hook's closure, so rebinding module globals can't retune it."""
    result = _run_guarded(
        tmp_path,
        """
        import hermes_protected_instruction_guard as g  # noqa: F401
        import sys
        import hermes_protected_instruction_guard  # noqa: F401
        open("AGENTS.md", "w").write("x")
        print("WROTE")
    """,
    )
    assert "WROTE" not in result.stdout
    assert "BLOCKED" in result.stderr


# --- host-side terminal check --------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "echo changed > AGENTS.md",
        "echo changed >> AGENTS.md",
        "printf x 2> SOUL.md",
        'sed -i "s/a/b/" AGENTS.md',
        "sed -i.bak -e 's/a/b/' CLAUDE.md",
        "tee AGENTS.md < payload",
        "cat payload | tee -a .cursorrules",
        "cp other.md SOUL.md",
        "mv /tmp/x CLAUDE.md",
        "install -m 644 other.md AGENTS.md",
        "truncate -s0 AGENTS.md",
        "dd if=/dev/zero of=AGENTS.md",
        'sh -c "echo changed > AGENTS.md"',
        """python -c "open('AGENTS.md','w').write('x')\"""",
        """python -c "import pathlib; pathlib.Path('AGENTS.md').write_text('x')\"""",
        # PowerShell / cmd.exe, where `terminal` on Windows spawns a different shell.
        "Copy-Item -Force backup.md SOUL.md",
        "copy-item backup.md C:/repo/AGENTS.md",
        "Copy-Item -Path backup.md -Destination SOUL.md",
        "Copy-Item -Destination:AGENTS.md -Path backup.md",
        "copy backup.md SOUL.md",
        "move backup.md CLAUDE.md",
        "xcopy backup.md .cursorrules",
        "Set-Content -Path AGENTS.md -Value 'x'",
        "Set-Content AGENTS.md 'x'",
        "Set-Content AGENTS.md -Value 'see backup.md'",
        "Add-Content -LiteralPath SOUL.md -Value 'x'",
        "Out-File AGENTS.md",
        "New-Item -ItemType File -Path CLAUDE.md",
    ],
)
def test_terminal_write_targets_sees_mechanical_shell_writes(command):
    targets = pig.terminal_write_targets(command, "/work/proj", env_type="local")
    assert targets, command
    assert all(os.path.basename(t).lower() in pig.BASENAMES for t in targets), targets


def test_terminal_write_targets_detects_powershell_copy_item(tmp_path, monkeypatch):
    """PowerShell ``Copy-Item -Force`` must not bypass the protected-file approval seam."""
    monkeypatch.setattr(
        pig,
        "guard_spec",
        lambda task_id="default": {
            "enabled": True,
            "basenames": list(pig.BASENAMES),
            "extra": [],
            "exempt": [],
            "task_id": task_id,
        },
    )
    targets = pig.terminal_write_targets(
        "Copy-Item -Force backup.md SOUL.md", str(tmp_path), env_type="local"
    )
    assert targets == [os.path.normpath(str(tmp_path / "SOUL.md"))]


@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "git status",
        "cat AGENTS.md",
        "grep -n TODO AGENTS.md",
        'echo "see AGENTS.md"',
        'echo "AGENTS.md" > notes.txt',
        "cp AGENTS.md backup/",
        "cat notes.txt > AGENTS.md.bak",
        "sed -n 1p AGENTS.md",
        """python -c "print(open('AGENTS.md').read())\"""",
        'python -c "print(1)"',
        # A PowerShell payload that merely names a protected file is not a write to it.
        "Set-Content -Path notes.txt -Value 'AGENTS.md'",
        "Copy-Item AGENTS.md backup/",
    ],
)
def test_terminal_write_targets_ignores_reads_and_mentions(command):
    assert pig.terminal_write_targets(command, "/work/proj", env_type="local") == []


def test_terminal_write_targets_skips_remote_backends():
    """The path belongs to the remote host, so resolving it here would compare wrong trees."""
    assert (
        pig.terminal_write_targets("echo x > AGENTS.md", "/work", env_type="ssh") == []
    )


def test_terminal_write_targets_skips_the_hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setattr(
        pig,
        "guard_spec",
        lambda task_id="default": {
            "enabled": True,
            "basenames": list(pig.BASENAMES),
            "extra": [],
            "exempt": [os.path.realpath(str(home))],
            "task_id": task_id,
        },
    )
    assert (
        pig.terminal_write_targets(
            f'echo x > "{home}/AGENTS.md"', str(tmp_path), env_type="local"
        )
        == []
    )


def test_terminal_write_targets_respects_the_config_switch(monkeypatch):
    monkeypatch.setattr(
        pig,
        "guard_spec",
        lambda task_id="default": {
            "enabled": False,
            "basenames": list(pig.BASENAMES),
            "extra": [],
            "exempt": [],
            "task_id": task_id,
        },
    )
    assert (
        pig.terminal_write_targets("echo x > AGENTS.md", "/work", env_type="local")
        == []
    )


def test_terminal_write_targets_resolves_relative_to_workdir(tmp_path):
    targets = pig.terminal_write_targets(
        "echo x > AGENTS.md",
        str(tmp_path),
        env_type="local",
        workdir=str(tmp_path / "sub"),
    )
    assert targets == [os.path.normpath(str(tmp_path / "sub" / "AGENTS.md"))]


def test_terminal_gate_runs_before_force(monkeypatch):
    """``force`` (the user already confirmed a DANGEROUS command) must not skip this gate."""
    from tools import terminal_tool as tt

    calls = []
    monkeypatch.setattr(
        tt,
        "_protected_instruction_block",
        lambda *a, **k: calls.append(k) or '{"error": "BLOCKED: test"}',
    )
    with pytest.raises(tt._Rejected):
        tt._run_approval_guards(
            "echo x > AGENTS.md",
            "local",
            {},
            force=True,
            cwd="/work",
            workdir=None,
            task_id="t1",
        )
    assert calls, "the protected-instruction gate did not run under force=True"


def test_terminal_gate_passes_through_when_nothing_is_protected(monkeypatch):
    from tools import terminal_tool as tt

    monkeypatch.setattr(tt, "_protected_instruction_block", lambda *a, **k: None)
    monkeypatch.setattr(tt, "_check_all_guards", lambda *a, **k: {"approved": True})
    verdict = tt._run_approval_guards(
        "ls -la", "local", {}, force=False, cwd="/work", workdir=None, task_id="t1"
    )
    assert verdict.note is None


def test_terminal_gate_fails_closed_when_the_check_raises(monkeypatch):
    from tools import terminal_tool as tt

    def boom(*_a, **_k):
        raise RuntimeError("guard exploded")

    monkeypatch.setattr(pig, "terminal_write_block", boom)
    with pytest.raises(tt._Rejected) as excinfo:
        tt._run_approval_guards("ls -la", "local", {}, force=False, cwd="/work")
    assert "could not run" in excinfo.value.result_json


# --- basename list stays in sync -----------------------------------------------------------


def test_basename_list_matches_the_file_tool_gate():
    from tools.file_tools_write_guards import _PROTECTED_INSTRUCTION_BASENAMES

    assert set(pig.BASENAMES) == {n.lower() for n in _PROTECTED_INSTRUCTION_BASENAMES}

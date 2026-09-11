"""Adversarial acceptance for interpreter-backed live-checkout mutations."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.mutation_effect_guard import MutationEffect, MutationEffectGuard


@pytest.fixture
def repos(tmp_path: Path) -> tuple[Path, Path]:
    live = tmp_path / "hermes-agent"
    other = tmp_path / "other-project"
    for root in (live, other):
        root.mkdir()
        subprocess.run(["git", "init", "-q", str(root)], check=True)
    return live.resolve(), other.resolve()


def _script(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def test_blocks_exact_write_file_then_interpreter_incident(
    repos: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    live, _ = repos
    script = _script(
        tmp_path,
        "upgrade.py",
        (
            "import subprocess\n"
            "subprocess.run("
            "['git', 'reset', '--hard', 'origin/main'], "
            f"cwd={str(live)!r}, check=True)\n"
            "subprocess.run("
            "['python', '-m', 'pip', 'install', '-e', '.'], "
            "check=True)\n"
        ),
    )

    effect = MutationEffectGuard(live).detect(f'python "{script}"', tmp_path)

    assert effect is not None
    assert effect.operation == "git reset"
    assert "subprocess.run" in effect.origin
    assert str(live) in effect.message


def test_blocks_script_outside_repo_when_subprocess_cwd_targets_live_checkout(
    repos: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    live, _ = repos
    script = _script(
        tmp_path,
        "outside.py",
        (
            "from subprocess import check_call as run\n"
            f"run(['git', 'checkout', 'main'], cwd={str(live)!r})\n"
        ),
    )

    effect = MutationEffectGuard(live).detect(f'python "{script}"', tmp_path)

    assert effect is not None
    assert effect.operation == "git checkout"
    assert "subprocess.check_call" in effect.origin


def test_blocks_static_aliases_variables_and_path_composition(
    repos: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    live, _ = repos
    script = _script(
        tmp_path,
        "variables.py",
        (
            "import subprocess as sp\n"
            "from pathlib import Path\n"
            f"root = Path({str(live)!r})\n"
            "command = ['git', '-C', str(root), 'reset', '--hard']\n"
            "sp.Popen(command)\n"
        ),
    )

    effect = MutationEffectGuard(live).detect(f'py -3.12 "{script}"', tmp_path)

    assert effect is not None
    assert effect.operation == "git reset"
    assert "subprocess.Popen" in effect.origin


@pytest.mark.parametrize(
    "command",
    [
        "python -c \"import os; os.system('git reset --hard')\"",
        (
            "bash -c \"python -c 'import subprocess; "
            "subprocess.run([\\\"git\\\", \\\"checkout\\\", \\\"main\\\"])'\""
        ),
    ],
)
def test_blocks_inline_and_nested_interpreter_payloads(
    repos: tuple[Path, Path],
    command: str,
) -> None:
    live, _ = repos

    effect = MutationEffectGuard(live).detect(command, live)

    assert effect is not None
    assert "Python -c" in effect.origin


def test_blocks_nested_python_script_spawn(
    repos: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    live, _ = repos
    inner = _script(
        tmp_path,
        "inner.py",
        "import os\nos.system('git reset --hard')\n",
    )
    outer = _script(
        tmp_path,
        "outer.py",
        (
            "import subprocess\n"
            f"subprocess.check_call(['python', {str(inner)!r}])\n"
        ),
    )

    effect = MutationEffectGuard(live).detect(f'python "{outer}"', live)

    assert effect is not None
    assert str(inner) in effect.origin
    assert effect.operation == "git reset"


def test_unscannable_oversized_script_fails_closed(
    repos: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    live, _ = repos
    script = _script(tmp_path, "large.py", "print('safe')\n" * 20)

    effect = MutationEffectGuard(live, max_script_bytes=32).detect(
        f'python "{script}"',
        live,
    )

    assert effect is not None
    assert effect.operation == "oversized interpreter input"
    assert "cannot prove" in effect.message


@pytest.mark.parametrize(
    ("name", "source"),
    [
        ("safe.py", "print('safe')\n"),
        ("syntax_error.py", "def broken(:\n"),
    ],
)
def test_safe_or_non_executable_source_is_left_to_terminal_owner(
    repos: tuple[Path, Path],
    tmp_path: Path,
    name: str,
    source: str,
) -> None:
    live, _ = repos
    script = _script(tmp_path, name, source)

    assert MutationEffectGuard(live).detect(f'python "{script}"', live) is None


def test_missing_script_is_left_to_terminal_owner(
    repos: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    live, _ = repos

    assert MutationEffectGuard(live).detect(
        f"python {tmp_path / 'missing.py'}",
        live,
    ) is None


def test_mutation_in_unrelated_repo_remains_allowed(
    repos: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    live, other = repos
    script = _script(
        tmp_path,
        "other.py",
        (
            "import subprocess\n"
            f"subprocess.run(['git', 'reset', '--hard'], cwd={str(other)!r})\n"
        ),
    )

    assert MutationEffectGuard(live).detect(f'python "{script}"', live) is None


def test_uv_run_python_is_inspected(
    repos: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    live, _ = repos
    script = _script(
        tmp_path,
        "uv_child.py",
        "import subprocess\nsubprocess.run(['git', 'reset', '--hard'])\n",
    )

    effect = MutationEffectGuard(live).detect(
        f'uv run --python 3.12 python "{script}"',
        live,
    )

    assert effect is not None
    assert effect.operation == "git reset"


def test_called_main_function_is_scanned_but_dormant_helper_is_not(
    repos: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    live, _ = repos
    dormant = _script(
        tmp_path,
        "dormant.py",
        (
            "import subprocess\n"
            "def dangerous():\n"
            "    subprocess.run(['git', 'reset', '--hard'])\n"
            "print('safe')\n"
        ),
    )
    called = _script(
        tmp_path,
        "called.py",
        (
            "import subprocess\n"
            "def main():\n"
            "    subprocess.run(['git', 'reset', '--hard'])\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
    )

    assert MutationEffectGuard(live).detect(f"python {dormant}", live) is None
    effect = MutationEffectGuard(live).detect(f"python {called}", live)

    assert effect is not None
    assert effect.operation == "git reset"


def test_static_exec_and_subprocess_alias_are_scanned(
    repos: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    live, _ = repos
    aliased = _script(
        tmp_path,
        "aliased.py",
        (
            "import subprocess\n"
            "runner = subprocess.run\n"
            "runner(['git', 'checkout', 'main'])\n"
        ),
    )

    alias_effect = MutationEffectGuard(live).detect(f"python {aliased}", live)
    exec_effect = MutationEffectGuard(live).detect(
        (
            "python -c \"exec(\\\"import os; "
            "os.system('git reset --hard')\\\")\""
        ),
        live,
    )

    assert alias_effect is not None
    assert alias_effect.operation == "git checkout"
    assert exec_effect is not None
    assert exec_effect.operation == "git reset"


def test_wrapper_blocks_only_interpreter_owned_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import terminal_mutation_guard as wrapper

    nested = MutationEffect(
        operation="git reset",
        message="blocked",
        origin="terminal command via script.py via subprocess.run",
    )
    monkeypatch.setattr(wrapper, "_terminal_effect", lambda *args, **kwargs: nested)
    monkeypatch.setattr(
        wrapper._terminal_owner,
        "_handle_terminal",
        lambda *args, **kwargs: "delegated",
    )

    payload = json.loads(
        wrapper._handle_terminal_with_effect_guard({"command": "python x.py"})
    )

    assert payload["status"] == "blocked"
    assert payload["exit_code"] == 1
    assert "indirection" in payload["error"]


def test_wrapper_preserves_existing_direct_command_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import terminal_mutation_guard as wrapper

    direct = MutationEffect(
        operation="git reset",
        message="blocked",
        origin="terminal command",
    )
    monkeypatch.setattr(wrapper, "_terminal_effect", lambda *args, **kwargs: direct)
    monkeypatch.setattr(
        wrapper._terminal_owner,
        "_handle_terminal",
        lambda *args, **kwargs: "delegated",
    )

    assert wrapper._handle_terminal_with_effect_guard(
        {"command": "git reset --hard"},
    ) == "delegated"


def test_terminal_effect_uses_the_terminal_owner_cwd_pipeline(
    repos: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import approval
    from tools import terminal_mutation_guard as wrapper

    live, other = repos
    script = live / "upgrade.py"
    script.write_text(
        "import subprocess\n"
        "subprocess.run(['git', 'reset', '--hard'], check=True)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        wrapper._terminal_owner,
        "_get_env_config",
        lambda: {"env_type": "local", "cwd": str(other)},
    )
    monkeypatch.setattr(
        wrapper._terminal_owner,
        "resolve_task_overrides",
        lambda task_id: {"cwd": str(live)} if task_id == "session-a" else {},
    )
    monkeypatch.setattr(wrapper._terminal_owner, "get_session_cwd", lambda key: None)
    monkeypatch.setattr(approval, "get_current_session_key", lambda default="": "")

    effect = wrapper._terminal_effect(
        {"command": "python upgrade.py"},
        {"task_id": "session-a"},
        source_root=live,
        active=True,
    )

    assert effect is not None
    assert effect.operation == "git reset"


def test_terminal_effect_skips_nonlocal_execution_owner(
    repos: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import terminal_mutation_guard as wrapper

    live, _ = repos
    monkeypatch.setattr(
        wrapper._terminal_owner,
        "_get_env_config",
        lambda: {"env_type": "ssh", "cwd": str(live)},
    )

    assert (
        wrapper._terminal_effect(
            {"command": "python upgrade.py"},
            {},
            source_root=live,
            active=True,
        )
        is None
    )

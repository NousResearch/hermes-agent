"""HERMES_WRITE_SAFE_ROOT permutation matrix (#70688 coverage extension).

Models routed desktop/multiplex turn scope (same path as test_write_safe_root_profile_isolation).
Every case prints setup / expected / actual / why.
"""

from __future__ import annotations

import os
from pathlib import Path

from agent.file_safety import get_safe_write_roots
from hermes_cli.env_loader import load_hermes_dotenv
from tests.tools.test_write_safe_root_profile_isolation import (
    _desktop_turn_scope,
    _file_ops,
    _log_case,
    _profile_home,
    _write_probe,
)
from tools.write_safe_root_scope import get_write_safe_root_scope


def _load_launch(tmp_path, monkeypatch, *, launch_dotenv: str, secondary_dotenv: str):
    """Launch profile dotenv → process env; secondary profile optional .env."""
    launch_home = _profile_home(tmp_path, "default", dotenv=launch_dotenv)
    secondary_home = _profile_home(tmp_path, "app", dotenv=secondary_dotenv)

    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.delenv("HERMES_WRITE_SAFE_ROOT", raising=False)
    load_hermes_dotenv(hermes_home=launch_home)

    return {
        "launch_home": launch_home,
        "secondary_home": secondary_home,
        "process_wsr": os.environ.get("HERMES_WRITE_SAFE_ROOT"),
    }


def _assert_write_matrix(
    *,
    label: str,
    env: dict,
    own_target: Path,
    foreign_target: Path,
    expected_own_allowed: bool,
    expected_foreign_allowed: bool,
    expected_roots: set[str] | None = None,
):
    with _desktop_turn_scope(env["secondary_home"]):
        roots = get_safe_write_roots()
        own_allowed, own_err = _write_probe(_file_ops(own_target.parent), own_target)
        foreign_allowed, foreign_err = _write_probe(
            _file_ops(foreign_target.parent), foreign_target
        )
        scoped = get_write_safe_root_scope()

    if expected_roots is not None:
        _log_case(
            label=f"{label} roots",
            setup=f"scoped={scoped!r} process={env['process_wsr']!r}",
            expected=f"roots == {sorted(expected_roots)}",
            actual=f"roots == {sorted(roots)}",
            passed=roots == expected_roots,
        )
        assert roots == expected_roots

    _log_case(
        label=f"{label} own vault",
        setup=f"target={own_target}",
        expected="allowed" if expected_own_allowed else "denied",
        actual=f"allowed={own_allowed} error={own_err!r}",
        passed=own_allowed == expected_own_allowed,
    )
    _log_case(
        label=f"{label} foreign/launch vault",
        setup=f"target={foreign_target}",
        expected="allowed" if expected_foreign_allowed else "denied",
        actual=f"allowed={foreign_allowed} error={foreign_err!r}",
        passed=foreign_allowed == expected_foreign_allowed,
    )

    assert own_allowed == expected_own_allowed, own_err
    assert foreign_allowed == expected_foreign_allowed, foreign_err


def test_m1_launch_unset_secondary_set(tmp_path, monkeypatch):
    """M1: launch WSR unset; secondary WSR set → own allow, foreign deny."""
    foreign_vault = tmp_path / "secondary-only-vault"
    foreign_vault.mkdir()
    env = _load_launch(
        tmp_path,
        monkeypatch,
        launch_dotenv="OPENAI_API_KEY=placeholder\n",
        secondary_dotenv=f"HERMES_WRITE_SAFE_ROOT={foreign_vault}\n",
    )
    own = foreign_vault / "own.txt"
    foreign = tmp_path / "outside.txt"
    foreign.parent.mkdir(exist_ok=True)

    _assert_write_matrix(
        label="M1",
        env=env,
        own_target=own,
        foreign_target=foreign,
        expected_own_allowed=True,
        expected_foreign_allowed=False,
        expected_roots={os.path.realpath(str(foreign_vault))},
    )


def test_m2_launch_set_secondary_unset(tmp_path, monkeypatch):
    """M2 hard gate: launch WSR set, secondary unset → launch vault must stay denied."""
    launch_vault = tmp_path / "launch-vault"
    launch_vault.mkdir()
    env = _load_launch(
        tmp_path,
        monkeypatch,
        launch_dotenv=f"HERMES_WRITE_SAFE_ROOT={launch_vault}\n",
        secondary_dotenv="OPENAI_API_KEY=placeholder\n",
    )
    own_outside = tmp_path / "neutral.txt"
    launch_target = launch_vault / "leak.txt"

    _assert_write_matrix(
        label="M2",
        env=env,
        own_target=own_outside,
        foreign_target=launch_target,
        expected_own_allowed=True,
        expected_foreign_allowed=False,
        expected_roots=set(),
    )


def test_m3_both_unset_no_process_leak(tmp_path, monkeypatch):
    """M3: both unset; no leftover process WSR opens a foreign vault."""
    env = _load_launch(
        tmp_path,
        monkeypatch,
        launch_dotenv="OPENAI_API_KEY=placeholder\n",
        secondary_dotenv="OPENAI_API_KEY=other\n",
    )
    monkeypatch.delenv("HERMES_WRITE_SAFE_ROOT", raising=False)
    assert env["process_wsr"] in (None, "")

    neutral = tmp_path / "neutral.txt"
    foreign = tmp_path / "other.txt"

    _assert_write_matrix(
        label="M3",
        env=env,
        own_target=neutral,
        foreign_target=foreign,
        expected_own_allowed=True,
        expected_foreign_allowed=True,
        expected_roots=set(),
    )


def test_m4_nested_parent_child_vaults(tmp_path, monkeypatch):
    """M4: nested parent vs child WSR → child allow; parent-outside-child deny."""
    parent = tmp_path / "Developer"
    child = parent / "app"
    parent.mkdir()
    child.mkdir(parents=True)
    launch_home = _profile_home(
        tmp_path, "default", dotenv=f"HERMES_WRITE_SAFE_ROOT={parent}\n",
    )
    secondary_home = _profile_home(
        tmp_path, "app", dotenv=f"HERMES_WRITE_SAFE_ROOT={child}\n",
    )
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.delenv("HERMES_WRITE_SAFE_ROOT", raising=False)
    load_hermes_dotenv(hermes_home=launch_home)
    env = {
        "launch_home": launch_home,
        "secondary_home": secondary_home,
        "parent_vault": parent,
        "child_vault": child,
        "process_wsr": os.environ.get("HERMES_WRITE_SAFE_ROOT"),
    }

    child_target = child / "own.txt"
    parent_outside = parent / "sibling.txt"

    _assert_write_matrix(
        label="M4 child",
        env=env,
        own_target=child_target,
        foreign_target=parent_outside,
        expected_own_allowed=True,
        expected_foreign_allowed=False,
        expected_roots={os.path.realpath(str(child))},
    )


def test_m5_unscoped_reads_process_env(tmp_path, monkeypatch):
    """M5: scope not bound → get_safe_write_roots follows os.environ."""
    vault = tmp_path / "env-vault"
    vault.mkdir()
    monkeypatch.delenv("HERMES_WRITE_SAFE_ROOT", raising=False)

    roots_unset = get_safe_write_roots()
    _log_case(
        label="M5 unset",
        setup="scope unbound, HERMES_WRITE_SAFE_ROOT unset",
        expected="roots == set()",
        actual=f"roots == {sorted(roots_unset)}",
        passed=roots_unset == set(),
    )

    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(vault))
    roots_set = get_safe_write_roots()
    expected = {os.path.realpath(str(vault))}
    _log_case(
        label="M5 set",
        setup=f"scope unbound, env={vault}",
        expected=f"roots == {sorted(expected)}",
        actual=f"roots == {sorted(roots_set)}",
        passed=roots_set == expected,
    )
    assert roots_unset == set()
    assert roots_set == expected


def test_m6_empty_string_vs_absent_key_same_when_scoped(tmp_path, monkeypatch):
    """M6: HERMES_WRITE_SAFE_ROOT= vs key absent → same scoped build (empty)."""
    from tools.write_safe_root_scope import build_profile_write_safe_root

    absent = _profile_home(tmp_path, "absent", dotenv="FOO=bar\n")
    empty = _profile_home(tmp_path, "empty", dotenv="HERMES_WRITE_SAFE_ROOT=\nFOO=bar\n")

    absent_val = build_profile_write_safe_root(absent)
    empty_val = build_profile_write_safe_root(empty)
    _log_case(
        label="M6 build",
        setup="profile .env absent key vs empty value",
        expected="both == ''",
        actual=f"absent={absent_val!r} empty={empty_val!r}",
        passed=absent_val == "" == empty_val,
    )
    assert absent_val == empty_val == ""

"""#70688 (fluxkapacitor): HERMES_WRITE_SAFE_ROOT must not inherit the launch profile.

When a secondary-profile session runs inside a long-lived host (desktop/TUI gateway,
multiplex gateway), the process ``os.environ`` carries the launch profile's
``HERMES_WRITE_SAFE_ROOT``. Profile turn scopes install secret + terminal policy
but ``agent.file_safety.get_safe_write_roots()`` still reads the ambient env, so
the routed profile's own vault is refused while the launch profile's vault is
permitted.

These tests model the real desktop turn boundary (``prompt_turn._prepare_turn_input``)
and the multiplex gateway boundary (``gateway.run._profile_runtime_scope`` +
multiplex dotenv skip). They log setup / expected / actual / pass-fail — not just
a pytest name.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import pytest

from agent.file_safety import get_safe_write_roots, get_write_denied_error
from agent.secret_scope import (
    build_profile_secret_scope,
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from hermes_cli.env_loader import load_hermes_dotenv
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.terminal_scope import install_profile_terminal_scope, reset_terminal_scope
from tools.write_safe_root_scope import (
    install_profile_write_safe_root_scope,
    reset_write_safe_root_scope,
)


def _profile_home(tmp_path: Path, name: str, *, dotenv: str = "") -> Path:
    if name == "default":
        home = tmp_path / ".hermes"
    else:
        home = tmp_path / ".hermes" / "profiles" / name
    home.mkdir(parents=True, exist_ok=True)
    if dotenv:
        (home / ".env").write_text(dotenv, encoding="utf-8")
    return home


def _file_ops(cwd: Path) -> ShellFileOperations:
    env = LocalEnvironment(cwd=str(cwd))
    return ShellFileOperations(env, cwd=str(cwd))


def _write_probe(ops: ShellFileOperations, target: Path) -> tuple[bool, str | None]:
    result = ops.write_file(str(target), "probe")
    return result.error is None, result.error


def _log_case(
    *,
    label: str,
    setup: str,
    expected: str,
    actual: str,
    passed: bool,
) -> None:
    status = "PASS" if passed else "FAIL"
    why = "matches expectation" if passed else "BUG: launch-profile safe root leaked"
    print(
        f"\n[{label}] {status}\n"
        f"  setup: {setup}\n"
        f"  expected: {expected}\n"
        f"  actual: {actual}\n"
        f"  why: {why}"
    )


@contextlib.contextmanager
def _desktop_turn_scope(profile_home: str | Path):
    """Mirror ``tui_gateway/prompt_turn.py::_prepare_turn_input`` scope binding."""
    home_token = set_hermes_home_override(str(profile_home))
    secret_token = set_secret_scope(build_profile_secret_scope(Path(profile_home)))
    terminal_token = install_profile_terminal_scope(Path(profile_home))
    wsr_token = install_profile_write_safe_root_scope(Path(profile_home))
    try:
        yield
    finally:
        reset_write_safe_root_scope(wsr_token)
        reset_terminal_scope(terminal_token)
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


@contextlib.contextmanager
def _multiplex_turn_scope(profile_home: str | Path):
    """Mirror ``gateway/run.py::_profile_runtime_scope`` (home + secrets + terminal)."""
    with _desktop_turn_scope(profile_home):
        yield


@pytest.fixture
def launch_env(tmp_path, monkeypatch):
    """Launch profile loaded into process env; secondary profile only in its .env."""
    default_vault = tmp_path / "default-vault"
    secondary_vault = tmp_path / "secondary-vault"
    default_vault.mkdir()
    secondary_vault.mkdir()

    default_home = _profile_home(
        tmp_path,
        "default",
        dotenv=f"HERMES_WRITE_SAFE_ROOT={default_vault}\n",
    )
    secondary_home = _profile_home(
        tmp_path,
        "app",
        dotenv=f"HERMES_WRITE_SAFE_ROOT={secondary_vault}\n",
    )

    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.delenv("HERMES_WRITE_SAFE_ROOT", raising=False)

    # Desktop/TUI gateway startup: launch profile dotenv becomes process-global.
    load_hermes_dotenv(hermes_home=default_home)

    return {
        "default_home": default_home,
        "secondary_home": secondary_home,
        "default_vault": default_vault,
        "secondary_vault": secondary_vault,
    }


def test_desktop_turn_scope_uses_launch_safe_root_not_profile_vault(launch_env):
    """Models a desktop-hosted turn for a secondary profile after cli -> desktop."""
    secondary = launch_env["secondary_home"]
    own_target = launch_env["secondary_vault"] / "own.txt"
    default_target = launch_env["default_vault"] / "private.txt"

    with _desktop_turn_scope(secondary):
        roots = get_safe_write_roots()
        own_allowed, own_err = _write_probe(_file_ops(launch_env["secondary_vault"]), own_target)
        default_allowed, default_err = _write_probe(
            _file_ops(launch_env["default_vault"]), default_target
        )

    expected_roots = {os.path.realpath(str(launch_env["secondary_vault"]))}
    actual_roots = roots

    _log_case(
        label="safe-root resolution",
        setup=(
            f"process env from {launch_env['default_home']}/.env; "
            f"desktop turn scope for {secondary}"
        ),
        expected=f"roots == {sorted(expected_roots)}",
        actual=f"roots == {sorted(actual_roots)}",
        passed=actual_roots == expected_roots,
    )

    _log_case(
        label="write own vault",
        setup=f"target={own_target}",
        expected="allowed (inside profile vault)",
        actual=f"allowed={own_allowed} error={own_err!r}",
        passed=own_allowed,
    )

    _log_case(
        label="write launch vault",
        setup=f"target={default_target}",
        expected="denied (outside profile vault)",
        actual=f"allowed={default_allowed} error={default_err!r}",
        passed=not default_allowed,
    )

    assert actual_roots == expected_roots
    assert own_allowed, own_err
    assert not default_allowed, "launch vault must not be writable for secondary profile turn"


def test_multiplex_gateway_scope_skips_dotenv_and_keeps_launch_safe_root(launch_env):
    """Models multiplex gateway: routed profile .env never reloads into os.environ."""
    secondary = launch_env["secondary_home"]
    own_target = launch_env["secondary_vault"] / "own.txt"
    default_target = launch_env["default_vault"] / "private.txt"

    set_multiplex_active(True)
    home_token = set_hermes_home_override(str(secondary))
    try:
        loaded = load_hermes_dotenv(hermes_home=secondary)
        assert loaded == [], "multiplex must skip process-global dotenv for routed home"

        with _multiplex_turn_scope(secondary):
            roots = get_safe_write_roots()
            own_err = get_write_denied_error(str(own_target))
            default_err = get_write_denied_error(str(default_target))
    finally:
        reset_hermes_home_override(home_token)
        set_multiplex_active(False)

    expected_roots = {os.path.realpath(str(launch_env["secondary_vault"]))}
    actual_roots = roots

    _log_case(
        label="multiplex dotenv skip",
        setup=f"load_hermes_dotenv({secondary}) under multiplex",
        expected="[] (no process env mutation)",
        actual=f"loaded={loaded}",
        passed=loaded == [],
    )

    _log_case(
        label="multiplex safe-root resolution",
        setup=f"launch env + multiplex turn scope for {secondary}",
        expected=f"roots == {sorted(expected_roots)}",
        actual=f"roots == {sorted(actual_roots)}",
        passed=actual_roots == expected_roots,
    )

    _log_case(
        label="multiplex write own vault",
        setup=f"target={own_target}",
        expected="no denial",
        actual=f"error={own_err!r}",
        passed=own_err is None,
    )

    _log_case(
        label="multiplex write launch vault",
        setup=f"target={default_target}",
        expected="safe_root denial",
        actual=f"error={default_err!r}",
        passed=default_err is not None and "outside HERMES_WRITE_SAFE_ROOT" in default_err,
    )

    assert actual_roots == expected_roots
    assert own_err is None
    assert default_err is not None


def test_surface_switch_does_not_rebind_write_safe_root(launch_env):
    """cli -> desktop surface switch keeps stored prompt; it must not fix safe roots either."""
    from agent.surface_switch import stage_surface_switch_note

    secondary = launch_env["secondary_home"]
    own_target = launch_env["secondary_vault"] / "after-switch.txt"

    class _FakeAgent:
        platform = "desktop"
        provider = "openai"
        api_mode = "chat_completions"
        session_id = "sess-70688"
        _surface_switch_note = ""

    agent = _FakeAgent()
    prompt = "...\n\nPlatform: cli\n\n[runtime tail]"
    history = [{"role": "user", "content": "prior turn on cli"}]

    with _desktop_turn_scope(secondary):
        switched = stage_surface_switch_note(agent, prompt, history)
        roots_after_switch = get_safe_write_roots()
        allowed, err = _write_probe(_file_ops(launch_env["secondary_vault"]), own_target)

    expected_roots = {os.path.realpath(str(launch_env["secondary_vault"]))}

    _log_case(
        label="surface switch note",
        setup="platform cli -> desktop under secondary profile scope",
        expected="note staged",
        actual=f"switched={switched} note={getattr(agent, '_surface_switch_note', '')[:80]!r}...",
        passed=switched,
    )

    _log_case(
        label="safe root after surface switch",
        setup="same turn scope as desktop prompt submit",
        expected=f"roots == {sorted(expected_roots)}",
        actual=f"roots == {sorted(roots_after_switch)} allowed={allowed} error={err!r}",
        passed=roots_after_switch == expected_roots and allowed,
    )

    assert switched
    assert roots_after_switch == expected_roots
    assert allowed, err

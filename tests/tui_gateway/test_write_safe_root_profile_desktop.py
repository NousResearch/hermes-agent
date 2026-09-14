"""Regression: desktop/TUI routed profile must not inherit launch HERMES_WRITE_SAFE_ROOT (#70688).

fluxkapacitor (Sep 11 2026): a desktop-hosted session for a non-default profile ran in a
process that still carried the default profile's ``HERMES_WRITE_SAFE_ROOT``. Own-vault writes
were refused; the default profile's vault was writable — inverted permissions.

This models the real host path: ``tui_gateway/server.py`` loads the launch profile dotenv once at
startup; a routed session binds ``_session_profile_runtime_scope`` (home + secret + terminal
scopes) per turn — but ``agent/file_safety.get_safe_write_roots()`` reads ``os.environ`` only.

Every check prints expected vs actual (brief requirement); silent asserts alone are insufficient.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.file_safety import get_safe_write_roots, get_write_denied_error
from agent.secret_scope import set_multiplex_active
from hermes_cli.env_loader import load_hermes_dotenv


def _report(check: str, expected, actual) -> bool:
    ok = expected == actual
    print(f"\n  CHECK: {check}")
    print(f"    expected: {expected!r}")
    print(f"    actual:   {actual!r}")
    print(f"    result:   {'PASS' if ok else 'FAIL'}")
    return ok


def _profile_home(tmp_path: Path, name: str, *, dotenv: str) -> Path:
    home = tmp_path / ".hermes" / "profiles" / name
    home.mkdir(parents=True)
    (home / ".env").write_text(dotenv, encoding="utf-8")
    (home / "config.yaml").write_text("toolsets: []\n", encoding="utf-8")
    return home


def test_routed_profile_write_safe_root_matches_profile_dotenv_not_launch_process(
    tmp_path, monkeypatch,
):
    """After launch dotenv + desktop session scope, file writes use the routed profile vault.

    fluxkapacitor used nested ``~/Developer`` vs ``~/Developer/<app>`` on macOS; disjoint
    sibling vaults here assert post-fix behavior: own vault allowed, foreign vault denied.
    """
    default_vault = tmp_path / "default-vault"
    profile_vault = tmp_path / "profile-vault"
    default_vault.mkdir()
    profile_vault.mkdir()

    launch_home = tmp_path / ".hermes"
    launch_home.mkdir()
    (launch_home / ".env").write_text(
        f"HERMES_WRITE_SAFE_ROOT={default_vault}\n", encoding="utf-8"
    )
    (launch_home / "config.yaml").write_text("toolsets: []\n", encoding="utf-8")

    profile_home = _profile_home(
        tmp_path,
        "bee",
        dotenv=f"HERMES_WRITE_SAFE_ROOT={profile_vault}\n",
    )

    print("\n=== SETUP ===")
    print(f"  launch_home:        {launch_home}")
    print(f"  profile_home:       {profile_home}")
    print(f"  default_vault:      {default_vault}")
    print(f"  profile_vault:      {profile_vault}")
    print("  host path:          load_hermes_dotenv(launch) then _session_profile_runtime_scope")

    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    set_multiplex_active(True)

    # Mirror tui_gateway/server.py module import: launch profile dotenv → process env.
    load_hermes_dotenv(hermes_home=launch_home)
    launch_wsr = os.environ.get("HERMES_WRITE_SAFE_ROOT")
    _report(
        "process env after launch dotenv (default vault)",
        str(default_vault),
        launch_wsr,
    )

    import tui_gateway.server as server

    own_target = profile_vault / "own.txt"
    foreign_target = default_vault / "foreign.txt"
    profile_root = os.path.realpath(profile_vault)

    with server._session_profile_runtime_scope({"profile_home": str(profile_home)}):
        active_roots = get_safe_write_roots()
        own_error = get_write_denied_error(str(own_target))
        foreign_error = get_write_denied_error(str(foreign_target))

        print("\n=== INSIDE ROUTED DESKTOP SESSION SCOPE ===")
        checks = [
            _report(
                "get_safe_write_roots()",
                {profile_root},
                active_roots,
            ),
            _report(
                "write own profile vault (denial error)",
                None,
                own_error,
            ),
            _report(
                "write default profile vault blocked",
                True,
                foreign_error is not None
                and "outside HERMES_WRITE_SAFE_ROOT" in foreign_error,
            ),
        ]
        if foreign_error is not None:
            print(f"    foreign_error text: {foreign_error!r}")

    print("\n=== SUMMARY ===")
    failed = [i for i, ok in enumerate(checks, 1) if not ok]
    if failed:
        print(f"  FAILED checks: {failed}")
    else:
        print("  all checks passed")

    assert checks[0], "active safe roots must be the routed profile vault, not launch process env"
    assert checks[1], f"own-vault write must be allowed; got {own_error!r}"
    assert checks[2], f"default-vault write must be denied; got {foreign_error!r}"

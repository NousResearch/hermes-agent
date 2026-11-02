"""Profile-aware approval behavior for exact sibling gateway restarts."""
from __future__ import annotations

from pathlib import Path

import pytest


def _reset_profile_home_cache(monkeypatch):
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)


def _profile(root: Path, name: str) -> Path:
    home = root if name == "default" else root / "profiles" / name
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("model:\n  provider: test\n", encoding="utf-8")
    return home


def _install_user_unit(xdg_home: Path, service: str, hermes_home: Path) -> None:
    unit_dir = xdg_home / "systemd" / "user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    (unit_dir / f"{service}.service").write_text(
        "[Service]\n"
        f'Environment="HERMES_HOME={hermes_home}"\n'
        "ExecStart=/usr/bin/python -m hermes_cli.main gateway run\n",
        encoding="utf-8",
    )


@pytest.fixture
def profile_gateway_fleet(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    forge_home = _profile(root, "forge")
    jarvis_home = _profile(root, "jarvis")
    default_home = _profile(root, "default")
    xdg_home = tmp_path / "xdg"
    monkeypatch.setenv("HERMES_HOME", str(forge_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_home))
    _reset_profile_home_cache(monkeypatch)
    _install_user_unit(xdg_home, "hermes-gateway-forge", forge_home)
    _install_user_unit(xdg_home, "hermes-gateway-jarvis", jarvis_home)
    _install_user_unit(xdg_home, "hermes-gateway", default_home)
    return {
        "root": root,
        "current": forge_home,
        "jarvis": jarvis_home,
        "default": default_home,
        "xdg": xdg_home,
    }


def test_exact_sibling_gateway_systemd_restart_is_not_flagged(profile_gateway_fleet):
    from tools.approval import detect_dangerous_command

    is_dangerous, _key, desc = detect_dangerous_command(
        "systemctl --user restart hermes-gateway-jarvis.service"
    )

    assert is_dangerous is False, desc


@pytest.mark.parametrize(
    "command",
    [
        "systemctl --user restart hermes-gateway-forge.service",
        "systemctl --user restart hermes-gateway-forge",
    ],
)
def test_current_gateway_restart_stays_flagged(profile_gateway_fleet, command):
    from tools.approval import detect_dangerous_command

    is_dangerous, _key, desc = detect_dangerous_command(command)

    assert is_dangerous is True
    assert desc == "stop/restart system service"


@pytest.mark.parametrize(
    "command",
    [
        "systemctl --user restart 'hermes-gateway*'",
        "systemctl --user restart hermes-gateway-jarvis.service hermes-gateway-forge.service",
        "systemctl --user restart hermes-gateway@jarvis.service",
    ],
)
def test_wildcard_multi_target_and_template_gateway_restarts_stay_flagged(profile_gateway_fleet, command):
    from tools.approval import detect_dangerous_command

    is_dangerous, _key, desc = detect_dangerous_command(command)

    assert is_dangerous is True
    assert desc == "stop/restart system service"


def test_missing_target_unit_fails_closed(profile_gateway_fleet):
    from tools.approval import detect_dangerous_command

    is_dangerous, _key, desc = detect_dangerous_command(
        "systemctl --user restart hermes-gateway-missing.service"
    )

    assert is_dangerous is True
    assert desc == "stop/restart system service"


def test_unit_without_matching_profile_home_fails_closed(profile_gateway_fleet):
    from tools.approval import detect_dangerous_command

    xdg_home = profile_gateway_fleet["xdg"]
    unit = xdg_home / "systemd" / "user" / "hermes-gateway-jarvis.service"
    unit.write_text(
        "[Service]\nEnvironment=\"HERMES_HOME=/var/lib/not-jarvis\"\n",
        encoding="utf-8",
    )

    is_dangerous, _key, desc = detect_dangerous_command(
        "systemctl --user restart hermes-gateway-jarvis.service"
    )

    assert is_dangerous is True
    assert desc == "stop/restart system service"


def test_multiplexed_gateway_host_fails_closed(profile_gateway_fleet, monkeypatch):
    from agent import secret_scope
    from tools.approval import detect_dangerous_command

    secret_scope.set_multiplex_active(True)
    try:
        is_dangerous, _key, desc = detect_dangerous_command(
            "systemctl --user restart hermes-gateway-jarvis.service"
        )
    finally:
        secret_scope.set_multiplex_active(False)

    assert is_dangerous is True
    assert desc == "stop/restart system service"


def test_indirect_shell_restart_of_sibling_stays_flagged(profile_gateway_fleet):
    from tools.approval import detect_dangerous_command

    is_dangerous, _key, desc = detect_dangerous_command(
        "bash -c 'systemctl --user restart hermes-gateway-jarvis.service'"
    )

    assert is_dangerous is True
    assert desc in {
        "stop/restart system service",
        "script execution via -c flag",
    }

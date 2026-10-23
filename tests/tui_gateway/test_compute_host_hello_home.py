"""The compute-host hello check must compare HERMES_HOME as a path.

The child hello reports the raw ``HERMES_HOME`` env value while the parent
holds ``str(get_hermes_home())``. On Windows a forward-slash env value
(``D:/x``) became ``D:\\x`` on the parent side, so every isolated turn fell
back inline with "compute host HERMES_HOME mismatch". See #135365.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tui_gateway.host_supervisor import HostSupervisor


def _supervisor(tmp_path: Path, expected: str) -> HostSupervisor:
    return HostSupervisor(
        registry_path=tmp_path / "registry.json",
        expected_build_sha="unknown",
        expected_hermes_home=expected,
        autostart=False,
    )


def test_hello_accepts_forward_slash_form_of_same_home(tmp_path):
    home = tmp_path / "home"
    sup = _supervisor(tmp_path, str(home))
    sup._hello = {"hermes_home": home.as_posix(), "build_sha": "x"}
    sup._validate_hello()


def test_hello_accepts_trailing_separator(tmp_path):
    home = tmp_path / "home"
    sup = _supervisor(tmp_path, str(home))
    sup._hello = {"hermes_home": home.as_posix() + "/", "build_sha": "x"}
    sup._validate_hello()


def test_hello_still_rejects_a_different_home(tmp_path):
    sup = _supervisor(tmp_path, str(tmp_path / "home"))
    sup._hello = {"hermes_home": (tmp_path / "other").as_posix(), "build_sha": "x"}
    with pytest.raises(RuntimeError, match="HERMES_HOME mismatch"):
        sup._validate_hello()

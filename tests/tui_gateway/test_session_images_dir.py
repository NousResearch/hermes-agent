"""Write-side scoping for desktop/clipboard image uploads (#69575).

Attach RPCs (``image.attach_bytes``, ``clipboard.paste``, ``pdf.attach``) run
before ``prompt.submit`` installs the session's profile HERMES_HOME override, so
the upload must be written under the session's *stored* ``profile_home`` — the
same scope the Docker mount and the vision host-read allowlist resolve at run
time. Otherwise, in a multi-profile / root-gateway deployment, the file is
written to the launch home while the sandbox mounts (and vision reads) the
profile home, and the agent can never see the upload it was handed.
"""

from pathlib import Path
from unittest.mock import patch

from tui_gateway.server import _session_images_dir


def test_profile_home_session_writes_under_profile(tmp_path):
    """A session pinned to a profile writes uploads under that profile's home."""
    profile_home = tmp_path / ".hermes" / "profiles" / "coder"
    session = {"profile_home": str(profile_home)}

    assert _session_images_dir(session) == profile_home / "images"


def test_launch_home_fallback_when_no_profile(tmp_path):
    """No ``profile_home`` on the session → the gateway launch home is used."""
    launch_home = tmp_path / ".hermes"
    session = {}

    with patch("tui_gateway.server._hermes_home", launch_home):
        assert _session_images_dir(session) == launch_home / "images"


def test_empty_profile_home_falls_back_to_launch_home(tmp_path):
    """An empty-string ``profile_home`` is treated as absent, not as ``/images``."""
    launch_home = tmp_path / ".hermes"
    session = {"profile_home": ""}

    with patch("tui_gateway.server._hermes_home", launch_home):
        assert _session_images_dir(session) == launch_home / "images"


def test_two_profiles_are_isolated(tmp_path):
    """Uploads from different profile sessions never share an images dir."""
    home_a = tmp_path / ".hermes" / "profiles" / "a"
    home_b = tmp_path / ".hermes" / "profiles" / "b"

    dir_a = _session_images_dir({"profile_home": str(home_a)})
    dir_b = _session_images_dir({"profile_home": str(home_b)})

    assert dir_a == home_a / "images"
    assert dir_b == home_b / "images"
    assert dir_a != dir_b

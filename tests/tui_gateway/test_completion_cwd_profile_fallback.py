"""A pooled profile's sessions land in THAT profile's workspace (#87584).

The Desktop spawns every pooled backend with the app-global
``TERMINAL_CWD: hermesCwd`` (the launch profile's workspace). When a named
profile's own ``terminal.cwd`` is a placeholder (``"."``) or unset,
``_completion_cwd`` fell through the profile-aware branch to
``_launch_configured_cwd()``/``TERMINAL_CWD`` — the launch (or another)
profile's workspace won. This pins both halves: the gateway tail (a named
local profile with no configured workspace falls back to its own home) and the
per-session override (``cwd_explicit`` still wins for a deliberate pick).
"""

from pathlib import Path

from tui_gateway import server
from tests.tui_gateway.test_tui_gateway_server import _write_profile_cfg


def test_named_profile_placeholder_cwd_falls_back_to_own_home(monkeypatch, tmp_path):
    """#87584: a NAMED profile with a placeholder/unset terminal.cwd must NOT
    inherit the launch profile's workspace; it lands in its own home."""
    home = _write_profile_cfg(tmp_path / "home-beta", ".")  # placeholder cwd
    launch_ws = tmp_path / "launch-repo"
    launch_ws.mkdir()
    launch_home = _write_profile_cfg(tmp_path / "launch-home", str(launch_ws))
    stale_env = tmp_path / "stale-env"
    stale_env.mkdir()

    monkeypatch.setenv("TERMINAL_CWD", str(stale_env))
    monkeypatch.setattr(server, "_hermes_home", launch_home)
    monkeypatch.setattr(server, "_profile_home", lambda name: home if name else None)

    assert server._completion_cwd({"profile": "beta"}) == str(home)
    # The launch profile itself still resolves its configured workspace.
    assert server._completion_cwd({}) == str(launch_ws)


def test_named_profile_unset_cwd_falls_back_to_own_home(monkeypatch, tmp_path):
    """No terminal.cwd key at all — same contract as the placeholder case."""
    home = tmp_path / "home-plain"
    home.mkdir()
    (home / "config.yaml").write_text("{}", encoding="utf-8")
    launch_ws = tmp_path / "launch-repo"
    launch_ws.mkdir()

    monkeypatch.setenv("TERMINAL_CWD", str(launch_ws))
    monkeypatch.setattr(server, "_profile_home", lambda name: home if name else None)

    assert server._completion_cwd({"profile": "plain"}) == str(home)


def test_named_profile_explicit_pick_still_wins(monkeypatch, tmp_path):
    """A deliberate per-session workspace pick (cwd_explicit) keeps winning over
    the new profile-home fallback — #52589's contract is unchanged."""
    home = _write_profile_cfg(tmp_path / "home-beta", ".")
    explicit = tmp_path / "explicit"
    explicit.mkdir()

    monkeypatch.setattr(server, "_profile_home", lambda name: home if name else None)
    assert (
        server._completion_cwd(
            {"profile": "beta", "cwd": str(explicit), "cwd_explicit": True}
        )
        == str(explicit)
    )


def test_named_profile_configured_cwd_still_wins(monkeypatch, tmp_path):
    """A named profile WITH a configured workspace is unaffected: the
    profile-aware branch still resolves it before any fallback."""
    profile_ws = tmp_path / "products"
    profile_ws.mkdir()
    home = _write_profile_cfg(tmp_path / "home-dev", str(profile_ws))
    launch_ws = tmp_path / "launch-repo"
    launch_ws.mkdir()

    monkeypatch.setenv("TERMINAL_CWD", str(launch_ws))
    monkeypatch.setattr(server, "_profile_home", lambda name: home if name else None)

    assert server._completion_cwd({"profile": "dev"}) == str(profile_ws)


def test_terminal_task_cwd_named_profile_placeholder_uses_own_home(monkeypatch, tmp_path):
    """The sibling consumer: ``_terminal_task_cwd_with_source``'s TERMINAL_CWD
    fallback must not leak the launch profile's workspace into a named
    profile's placeholder session either (its cwd comes from _completion_cwd's
    tail when the session has none)."""
    home = _write_profile_cfg(tmp_path / "home-beta", ".")
    launch_ws = tmp_path / "launch-repo"
    launch_ws.mkdir()

    monkeypatch.setenv("TERMINAL_CWD", str(launch_ws))
    monkeypatch.setattr(server, "_profile_home", lambda name: home if name else None)

    # A session with no cwd of its own resolves its workspace via the tail.
    cwd = server._completion_cwd({"profile": "beta"})
    assert cwd == str(home)
    # And a session dict carrying that cwd is what terminal tools use.
    task_cwd = server._terminal_task_cwd({"cwd": cwd, "profile_home": home, "explicit_cwd": False})
    assert task_cwd == str(home)

"""The launch home the TUI server binds launch-profile turns to must follow the process's launch
home: the live process home, or the home pinned for routing decisions when one is set."""
import os
from pathlib import Path

import pytest

# Imported at collection on purpose: that is when real test modules import the server,
# before any per-test fixture redirects HERMES_HOME, so ``_hermes_home`` freezes to the
# pre-fixture home exactly as it does for the rest of the suite.
from tui_gateway import server
from tui_gateway.launch_profile_policy import launch_profile_scope_if_multiplexed, launch_secret_scope


def test_launch_home_follows_the_process_home_redirected_after_import():
    """``server._hermes_home`` is get_hermes_home() at import — under a developer shell the
    honored custom HERMES_HOME, a guarded root. Launch-profile turns read ``<launch home>/.env``
    (``launch_secret_scope``), so the home they bind must be resolved at call time from the
    process env, like the launch ``state.db`` handle (#112692), never the import-time value."""
    sandbox = Path(os.environ["HERMES_HOME"])
    assert server._launch_home() == sandbox
    (sandbox / ".env").write_text("HERMES_LAUNCH_HOME_PROBE=from-sandbox\n", encoding="utf-8")
    assert launch_secret_scope(server._launch_home()).get("HERMES_LAUNCH_HOME_PROBE") == "from-sandbox"


def _tui_launch_session_scope():
    return server._session_profile_runtime_scope({"profile_home": None})


@pytest.mark.parametrize("enter_launch_scope", [launch_profile_scope_if_multiplexed, _tui_launch_session_scope],
                         ids=["keepalive-and-standalone-gateway", "tui-launch-session"])
def test_launch_scope_binds_the_pinned_launch_home_not_a_mirrored_served_home(
        tmp_path, monkeypatch, enter_launch_scope):
    """An embedding host (Hermes WebUI) pins its own home with ``pin_process_hermes_home()`` and
    mirrors each turn's profile into ``HERMES_HOME``; a multiplexed Desktop/dashboard backend does
    the same while ``DELETE /api/profiles/b`` removes B's service unit. The launch profile's own
    scope (bound by the Nous keepalive thread, every standalone gateway path, and every
    launch-profile TUI/Desktop RPC) must still bind the pinned home: resolving the live env var
    bound the launch env frozen at activation under profile B's home."""
    from agent.secret_scope import get_secret, set_multiplex_active
    from hermes_constants import get_hermes_home, pin_process_hermes_home
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting

    launch = tmp_path / "launch"
    served = launch / "profiles" / "b"
    served.mkdir(parents=True)
    (launch / ".env").write_text("LAUNCH_FILE_KEY=launch\n", encoding="utf-8")
    (served / ".env").write_text("SERVED_FILE_KEY=served\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    pin_process_hermes_home(launch)
    activate_multi_profile_hosting()
    try:
        monkeypatch.setenv("HERMES_HOME", str(served))  # the host's per-turn mirror
        with enter_launch_scope():
            bound = (get_hermes_home(), get_secret("LAUNCH_FILE_KEY"), get_secret("SERVED_FILE_KEY"))
    finally:
        set_multiplex_active(False)
        pin_process_hermes_home(None)
    assert bound == (launch, "launch", None)

"""A user's ``AGENT_BROWSER_PROFILE`` must reach agent-browser as a path, not as a bare name (#132755).

``profile_dir()`` has always resolved the pin — ``~`` expands, a relative value anchors at this
profile's HERMES_HOME — but the value actually on its way to the browser came from
``_build_browser_env``, which inherits the raw string. ``env_for_agent`` set-defaulted and moved
on, so a relative pin reached agent-browser verbatim, where a bare ``pin`` is a Chrome profile
*name* to go looking for rather than ``<HERMES_HOME>/pin``.

What this file deliberately does NOT assert: that every host-lane browser shares one persistent
jar. It must not. Chromium's singleton means the first session to start owns a shared
``--user-data-dir`` and the rest die with "Chrome exited early (exit code 21)"; the screen path
can share one jar only because a human takes the lease there. See the note on
``_agent_browser_command_env``.
"""

import pytest

import tools.bot_desktop.browser as bd_browser
from tools import browser_tool_session as session


@pytest.fixture
def hermes_home(monkeypatch, tmp_path):
    """Pin the home both ``profile_dir`` and ``resolved_profile_pin`` resolve against."""
    monkeypatch.setattr(bd_browser.runtime, "get_hermes_home", lambda: tmp_path / "home")
    return tmp_path / "home"


@pytest.fixture(autouse=True)
def _no_inherited_pin(monkeypatch):
    monkeypatch.delenv("AGENT_BROWSER_PROFILE", raising=False)


def test_relative_pin_becomes_a_path(hermes_home, monkeypatch):
    # Documented (bot-screen.md): ``pin`` means ``<HERMES_HOME>/pin``.
    monkeypatch.setenv("AGENT_BROWSER_PROFILE", "pin")
    assert bd_browser.env_for_agent({})["AGENT_BROWSER_PROFILE"] == str(hermes_home / "pin")


def test_relative_pin_is_resolved_even_when_it_arrives_through_the_child_env(hermes_home):
    # The real shape: _build_browser_env has already carried the raw string into the dict.
    env = bd_browser.env_for_agent({"AGENT_BROWSER_PROFILE": "pin"})
    assert env["AGENT_BROWSER_PROFILE"] == str(hermes_home / "pin")


def test_tilde_pin_expands(hermes_home, monkeypatch):
    monkeypatch.setenv("AGENT_BROWSER_PROFILE", "~/somewhere")
    env = bd_browser.env_for_agent({})
    assert env["AGENT_BROWSER_PROFILE"] == str(bd_browser.os.path.expanduser("~/somewhere"))


def test_absolute_pin_is_left_exactly_where_the_user_put_it(hermes_home, tmp_path):
    pin = str(tmp_path / "elsewhere")
    assert bd_browser.env_for_agent({"AGENT_BROWSER_PROFILE": pin})["AGENT_BROWSER_PROFILE"] == pin


def test_absent_pin_still_gets_the_profile_default(hermes_home):
    # Unchanged behaviour: no pin anywhere means the per-profile jar.
    assert bd_browser.env_for_agent({})["AGENT_BROWSER_PROFILE"] == str(
        hermes_home / "bot-desktop" / "browser-profile"
    )


def test_resolver_agrees_with_profile_dir(hermes_home, monkeypatch):
    # One set of rules, two entry points — a pin resolved one way must not differ from the other.
    monkeypatch.setenv("AGENT_BROWSER_PROFILE", "pin")
    assert bd_browser.resolved_profile_pin() == str(bd_browser.profile_dir())


def test_host_lane_does_not_pin_one_jar_for_every_session(hermes_home, monkeypatch, tmp_path):
    """Guard rail, not a wish: the host lane must leave the pin alone.

    Two browser tasks in one run each get their own agent-browser session. Pin them to one
    user-data-dir and the first to start owns it while the others fail to launch (measured:
    "Chrome exited early (exit code 21)"), which trades a lesser annoyance — state that does not
    outlive a run — for a lane that cannot run two browsers.
    """
    import tools.environments.local as local

    monkeypatch.setattr(local, "hermes_subprocess_env", lambda inherit_credentials=False: {})
    env = session._agent_browser_command_env(str(tmp_path / "socket"))
    assert "AGENT_BROWSER_PROFILE" not in env


def test_lightpanda_never_receives_a_chromium_profile(hermes_home, monkeypatch, tmp_path):
    """An inherited pin must not be handed to the text-only engine.

    Lightpanda rejects Chromium-only launch config and has no Chromium jar to keep state in, so
    the key is stripped for that engine exactly like the other Chromium knobs.
    """
    import tools.environments.local as local

    monkeypatch.setattr(local, "hermes_subprocess_env", lambda inherit_credentials=False: {})
    monkeypatch.setenv("AGENT_BROWSER_PROFILE", str(tmp_path / "pin"))

    captured = {}

    def _capture(cmd_parts, browser_env, task_socket_dir, *a, **kw):
        captured.update(browser_env)
        raise RuntimeError("stop after env capture")

    monkeypatch.setattr(session, "_popen_agent_browser", _capture)
    monkeypatch.setattr(session, "_ensure_screen_for_headed_chromium", lambda: None)
    monkeypatch.setattr(session, "_sandbox_wrap", lambda c, e, s: (c, e))

    with pytest.raises(RuntimeError):
        session._spawn_and_collect(
            "task1", {"session_name": "s"}, ["agent-browser", "snapshot"],
            "snapshot", "lightpanda", timeout=1,
        )
    assert "AGENT_BROWSER_PROFILE" not in captured
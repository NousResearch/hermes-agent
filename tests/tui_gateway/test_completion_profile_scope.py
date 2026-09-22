"""``complete.slash`` / ``commands.catalog`` must honor ``params.profile`` when the chat's session
is live in ANOTHER process.

The Desktop answers a chat's ``/`` popover on the window's ambient socket, not on the backend that
serves that chat: opening a Bot Chat deliberately keeps the gateway on the launch profile (the
sidebar must not look like every session disappeared), so the completion request carries the
profile owning the chat while its session is live elsewhere. Both handlers resolved the home from
the live-session record alone and silently fell back to the answering process's LAUNCH home, so a
secondary profile's local skills vanished from the palette — the popover said "No matches. Try
/help." while typing the command anyway still ran it (dispatch routes to the owning backend).

The ladder mirrors ``server._profile_scoped``: the explicit ``profile`` wins, the live session is
the fallback, and a sessionless call without one still means the launch profile. An unknown profile
fails closed (``ProfileUnavailableError`` → ``4064``), never a silent launch-home scan.
"""

from __future__ import annotations

from pathlib import Path

import tui_gateway.server as server
from tui_gateway.contracts.registry import METHODS, validate_params

SECOND_PROFILE = "worker"
FOREIGN_SESSION = "served-by-another-backend"


def _skill(home: Path, name: str, description: str) -> None:
    d = home / "skills" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n", encoding="utf-8")


def _homes(tmp_path: Path) -> tuple[Path, Path]:
    launch = tmp_path / "launch"
    worker = tmp_path / "profiles" / SECOND_PROFILE
    for home, name, desc in (
        (launch, "launch-skill", "from the launch profile"),
        (worker, "worker-skill", "from the worker profile"),
    ):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("skills:\n  external_dirs: []\n", encoding="utf-8")
        _skill(home, name, desc)
    return launch, worker


def _bind(monkeypatch, launch: Path, worker: Path) -> None:
    """Launch shape: this process answers as ``launch``; ``SECOND_PROFILE`` is another home on disk.

    ``_profile_home`` is pinned (the ``test_config_profile_scope`` pattern) so the test exercises the
    completion handlers' ladder rather than the profile-dir lookup, and so a correct override can't
    flip this process into multi-profile hosting.
    """
    import agent.skill_commands as skill_commands
    import agent.skill_utils as skill_utils

    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(
        server, "_profile_home", lambda name: worker if (name or "").strip() == SECOND_PROFILE else None)
    # The chat's session is NOT live here — that is the cross-profile Desktop case.
    monkeypatch.setattr(server, "_sessions", {})
    # Neutral workspace: neither the process cwd nor TERMINAL_CWD may contribute a project skill.
    elsewhere = launch.parent / "workspace"
    elsewhere.mkdir(exist_ok=True)
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("TERMINAL_CWD", str(elsewhere))
    monkeypatch.setattr(skill_commands, "_skill_commands", {})
    monkeypatch.setattr(skill_commands, "_skill_commands_platform", None)
    monkeypatch.setattr(skill_commands, "_skill_commands_home", None)
    monkeypatch.setattr(skill_commands, "_skill_commands_project", None)
    skill_utils._external_dirs_cache_clear()


def _skill_texts(response: dict) -> list[str]:
    return [item["text"] for item in response["result"]["items"] if item.get("kind") == "skill"]


def test_complete_slash_resolves_the_params_profile_for_a_foreign_session(tmp_path, monkeypatch):
    launch, worker = _homes(tmp_path)
    _bind(monkeypatch, launch, worker)

    response = server._methods["complete.slash"](
        "s", {"text": "/worker", "session_id": FOREIGN_SESSION, "profile": SECOND_PROFILE})

    assert _skill_texts(response) == ["worker-skill"]
    # The explicit profile REPLACED the home (not unioned with it): the launch profile's own skill
    # is not on offer for a chat that belongs to ``worker``.
    other = server._methods["complete.slash"](
        "s2", {"text": "/launch", "session_id": FOREIGN_SESSION, "profile": SECOND_PROFILE})
    assert _skill_texts(other) == []


def test_commands_catalog_resolves_the_params_profile_for_a_foreign_session(tmp_path, monkeypatch):
    launch, worker = _homes(tmp_path)
    _bind(monkeypatch, launch, worker)

    catalog = server._methods["commands.catalog"](
        "c", {"session_id": FOREIGN_SESSION, "profile": SECOND_PROFILE})["result"]
    assert "/worker-skill" in catalog["skills"]
    assert "/launch-skill" not in catalog["skills"]

    # Sessionless without a profile still means the launch profile (a draft in the launch profile).
    launch_catalog = server._methods["commands.catalog"]("c2", {})["result"]
    assert "/launch-skill" in launch_catalog["skills"]
    assert "/worker-skill" not in launch_catalog["skills"]


def test_complete_slash_live_session_wins_only_when_no_profile_is_sent(tmp_path, monkeypatch):
    launch, worker = _homes(tmp_path)
    _bind(monkeypatch, launch, worker)

    # The pre-existing ladder: a session that IS live here binds its own profile_home.
    server._sessions["live-here"] = {"session_key": "live-here", "profile_home": str(worker)}
    assert _skill_texts(server._methods["complete.slash"](
        "s3", {"text": "/worker", "session_id": "live-here"})) == ["worker-skill"]
    # …and a launch-profile session stays on the launch home.
    server._sessions["live-launch"] = {"session_key": "live-launch", "profile_home": None}
    assert _skill_texts(server._methods["complete.slash"](
        "s4", {"text": "/launch", "session_id": "live-launch"})) == ["launch-skill"]


def test_complete_slash_accepts_a_routed_profile_param(tmp_path):
    # The routing design sends ``profile`` (``requestGatewayForProfile``); the params model must accept
    # it, or a routed request dies with 4000 before the handler ever resolves the home.
    _params, problem = validate_params(
        METHODS["complete.slash"], {"text": "/openevent", "session_id": FOREIGN_SESSION, "profile": SECOND_PROFILE})

    assert problem is None
    # Unknown keys are still rejected: this added one declared field, it did not loosen the contract.
    _typo, typo_problem = validate_params(METHODS["complete.slash"], {"text": "/x", "ssession_id": "oops"})
    assert _typo is None and typo_problem is not None and "ssession_id" in typo_problem


def test_complete_slash_fails_closed_on_an_unknown_profile(tmp_path, monkeypatch):
    # A profile name that does not resolve must answer an ERROR (the handler's own 5020 catcher;
    # dispatch maps it to 4064), never fall back to scanning the launch home: the silent fallback is
    # the bug, not a safety net.
    launch = tmp_path / ".hermes"
    (launch / "skills").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setattr(server, "_sessions", {})

    response = server._methods["complete.slash"]("s", {"text": "/x", "profile": "renamed-away"})

    assert "result" not in response
    assert response["error"]["code"] == 5020
    assert "renamed-away" in response["error"]["message"]

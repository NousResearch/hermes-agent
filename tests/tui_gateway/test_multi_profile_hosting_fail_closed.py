"""Multi-profile hosting in the TUI gateway is fail-closed and every profile-scoped RPC runs under
the FULL runtime scope of the requested profile (home + secrets + terminal), the launch profile
included once the process multiplexes.

Regression for the silent cross-profile secret leak class: ``hermes serve`` hosted many profile
homes but never called ``set_multiplex_active(True)``, so every unscoped ``get_secret`` read for a
secondary silently returned the LAUNCH profile's ``os.environ`` value; ``@_profile_scoped`` bound
only HERMES_HOME. And the launch-profile asymmetry: a default-member hosted-room turn in a
``multiplex_profiles: true`` gateway died at agent build with ``UnscopedSecretError``.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import tui_gateway.server as server
from tui_gateway import launch_profile_policy as lpp

A_VAL = "a-only-secret-0001"
B_VAL = "b-only-secret-0002"
ENV_VAL = "systemd-injected-0003"
A_API_KEY = "launch-api-key-0004"
B_API_KEY = "secondary-api-key-0005"
A_BASE_URL = "https://launch.example.invalid/v1"
B_BASE_URL = "https://secondary.example.invalid/v1"


@pytest.fixture
def two_homes(tmp_path, monkeypatch):
    """Launch home (root) + secondary ``profiles/b``; B's config references both tokens."""
    root = tmp_path / "hermes_home"
    b = root / "profiles" / "b"
    b.mkdir(parents=True)
    (root / ".env").write_text(
        f"A_ONLY_TOKEN={A_VAL}\nHERMES_API_KEY={A_API_KEY}\nHERMES_BASE_URL={A_BASE_URL}\n",
        encoding="utf-8")
    (b / ".env").write_text(
        f"B_ONLY_TOKEN={B_VAL}\nHERMES_API_KEY={B_API_KEY}\nHERMES_BASE_URL={B_BASE_URL}\n",
        encoding="utf-8")
    for home in (root, b):
        (home / "config.yaml").write_text(
            "probe:\n  a_ref: ${A_ONLY_TOKEN}\n  b_ref: ${B_ONLY_TOKEN}\n  env_ref: ${INJECTED_TOKEN}\n",
            encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("A_ONLY_TOKEN", A_VAL)  # the launch process loaded its own .env
    monkeypatch.setenv("HERMES_API_KEY", A_API_KEY)
    monkeypatch.setenv("HERMES_BASE_URL", A_BASE_URL)
    monkeypatch.setenv("INJECTED_TOKEN", ENV_VAL)  # systemd / op run credential injection
    monkeypatch.setattr(server, "_hermes_home", root)
    monkeypatch.setattr(server, "_served_profile_homes", set())
    monkeypatch.setattr(lpp, "_snapshot", None)
    from agent import secret_scope
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    server._cfg_cache = server._cfg_mtime = server._cfg_path = None
    return root, b


def _probe(profile: str | None) -> dict:
    params = {"key": "full"}
    if profile:
        params["profile"] = profile
    server._cfg_cache = server._cfg_mtime = server._cfg_path = None
    resp = server._methods["config.get"]("rid", params)
    assert "error" not in resp, resp
    return resp["result"]["config"]["probe"]


def test_config_get_for_secondary_resolves_only_its_own_secrets_and_flips_fail_closed(two_homes):
    from agent.secret_scope import UnscopedSecretError, get_secret, is_multiplex_active

    root, _b = two_homes
    assert not is_multiplex_active()  # single-profile so far

    probe_b = _probe("b")
    assert probe_b["b_ref"] == B_VAL
    assert probe_b["a_ref"] == "${A_ONLY_TOKEN}"  # never the launch profile's value
    assert probe_b["env_ref"] == "${INJECTED_TOKEN}"  # never the launch process env
    # Hosting a second home flipped the process: an unscoped read now raises instead of borrowing.
    assert is_multiplex_active()
    with pytest.raises(UnscopedSecretError):
        get_secret("A_ONLY_TOKEN")
    assert os.environ["A_ONLY_TOKEN"] == A_VAL  # never mutated

    # The launch profile is a profile too: its RPC keeps its own .env AND its injected env.
    probe_a = _probe(None)
    assert probe_a["a_ref"] == A_VAL
    assert probe_a["env_ref"] == ENV_VAL
    assert probe_a["b_ref"] == "${B_ONLY_TOKEN}"


def test_single_profile_serve_keeps_environ_fallthrough(two_homes):
    """Control: with no secondary ever requested the launch profile stays unscoped, so credentials
    injected only via the process env (systemd, ``op run``) keep resolving."""
    from agent.secret_scope import get_secret, is_multiplex_active

    probe = _probe(None)
    assert probe["env_ref"] == ENV_VAL
    assert not is_multiplex_active()
    assert get_secret("INJECTED_TOKEN") == ENV_VAL


def test_rpc_scope_reaches_llm_oneshot_and_model_options(two_homes, monkeypatch):
    """The scope must wrap the body of every credential-reading RPC, not only config.get."""
    from agent.secret_scope import get_secret

    root, b = two_homes
    seen = {}

    def fake_oneshot(**kwargs):
        seen["oneshot"] = (Path(os.environ.get("HERMES_HOME", "")), get_secret("B_ONLY_TOKEN"), get_secret("A_ONLY_TOKEN"))
        from hermes_constants import get_hermes_home
        seen["oneshot_home"] = Path(get_hermes_home())
        return "t"

    monkeypatch.setattr("agent.oneshot.run_oneshot", fake_oneshot)
    monkeypatch.setattr(server, "_model_picker_context", lambda agent: object())

    def build_payload(ctx, **kwargs):
        from hermes_constants import get_hermes_home
        seen["options"] = (Path(get_hermes_home()), get_secret("B_ONLY_TOKEN"), get_secret("A_ONLY_TOKEN"))
        return {"providers": []}

    monkeypatch.setattr("hermes_cli.inventory.build_model_options_payload", build_payload)

    r = server._methods["llm.oneshot"]("r1", {"profile": "b", "instructions": "x", "input": "y"})
    assert r["result"]["text"] == "t"
    assert seen["oneshot_home"] == b and seen["oneshot"][1:] == (B_VAL, None)
    r = server._methods["model.options"]("r2", {"profile": "b"})
    assert r["result"] == {"providers": []}
    assert seen["options"] == (b, B_VAL, None)


def test_live_review_binds_the_sessions_full_runtime_scope(two_homes, monkeypatch):
    """Live /review must resolve the reviewer's provider credentials from its session's profile
    (``start_review`` → ``delegate_task`` → profile-scoped ``get_secret``); unscoped it fails
    closed with UnscopedSecretError on a multiplexed gateway (#117544)."""
    from agent.secret_scope import get_secret
    from hermes_constants import get_hermes_home

    root, b = two_homes
    seen = []

    def fake_start_review(agent, snapshot, arg):
        seen.append((Path(get_hermes_home()), get_secret("A_ONLY_TOKEN"), get_secret("B_ONLY_TOKEN")))
        return object()

    monkeypatch.setattr("agent.review_engine.start_review", fake_start_review)
    monkeypatch.setattr("agent.review_engine.format_dispatch_note", lambda result, arg: "dispatched")
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda value: False)

    def invoke(profile_home):
        sid = f"review-{len(seen)}"
        agent = SimpleNamespace(_cached_system_prompt="", tools=None)
        session = {
            "agent": agent,
            "profile_home": str(profile_home) if profile_home else None,
            "history": [{"role": "user", "content": "hello"}],
            "history_lock": threading.Lock(),
            "history_version": 0,
            "running": False,
            "session_key": sid,
        }
        server._sessions[sid] = session
        try:
            assert server._format_live_review_output(sid, session, "") == "dispatched"
        finally:
            server._sessions.pop(sid, None)

    invoke(None)
    _probe("b")  # activate multiplexing and freeze the launch profile's own secret scope
    invoke(b)
    invoke(None)

    assert seen == [
        (root, A_VAL, None),
        (b, None, B_VAL),
        (root, A_VAL, None),
    ]
    assert os.environ["A_ONLY_TOKEN"] == A_VAL
    assert "B_ONLY_TOKEN" not in os.environ


@pytest.mark.parametrize("route", ["session.compress", "slash.compress"])
def test_manual_compress_routes_bind_the_sessions_full_runtime_scope(two_homes, monkeypatch, route):
    """Manual compression must resolve secrets from its session across an A→B→A sequence."""
    from agent.secret_scope import get_secret
    from hermes_constants import get_hermes_home

    root, b = two_homes
    seen = []

    def observe_scope():
        seen.append((Path(get_hermes_home()), get_secret("A_ONLY_TOKEN"), get_secret("B_ONLY_TOKEN")))

    def invoke(profile_home):
        sid = f"compress-{len(seen)}"
        agent = SimpleNamespace(_cached_system_prompt="", tools=None)
        session = {
            "agent": agent,
            "profile_home": str(profile_home) if profile_home else None,
            "history": [{"role": "user", "content": "hello"}],
            "history_lock": threading.Lock(),
            "history_version": 0,
            "running": False,
            "session_key": sid,
        }
        server._sessions[sid] = session
        try:
            if route == "session.compress":
                monkeypatch.setattr(server, "_sess_nowait", lambda params, rid: (session, None))
                monkeypatch.setattr(server, "_sess", lambda params, rid: (session, None))
                monkeypatch.setattr(server, "_session_uses_compute_host", lambda value: False)

                def compress_live(*args, **kwargs):
                    observe_scope()
                    return server._ok("rid", {"status": "compressed"})

                monkeypatch.setattr(server, "_compress_live", compress_live)
                response = server._methods["session.compress"]("rid", {"session_id": sid})
                assert "error" not in response
            else:
                def compress_history(*args, **kwargs):
                    observe_scope()
                    raise server.CompressionLockHeld("test holder")

                monkeypatch.setattr(server, "_compress_session_history", compress_history)
                monkeypatch.setattr(
                    "agent.model_metadata.estimate_request_tokens_rough", lambda *args, **kwargs: 1)
                server._compress_live_with_feedback(sid, session, agent, "", snapshot_kwargs=True)
        finally:
            server._sessions.pop(sid, None)

    invoke(None)
    _probe("b")  # activate multiplexing and freeze the launch profile's own secret scope
    invoke(b)
    invoke(None)

    assert seen == [
        (root, A_VAL, None),
        (b, None, B_VAL),
        (root, A_VAL, None),
    ]
    assert os.environ["A_ONLY_TOKEN"] == A_VAL
    assert "B_ONLY_TOKEN" not in os.environ


def test_config_show_keeps_each_profiles_values_after_multiplex_activation(two_homes):
    """A→B→A config.show calls resolve the requested profile instead of running unscoped."""
    root, b = two_homes

    def displayed_values(profile=None):
        params = {"profile": profile} if profile else {}
        response = server._methods["config.show"]("rid", params)
        assert "error" not in response, response
        sections = {
            section["title"]: dict(section["rows"])
            for section in response["result"]["sections"]
        }
        return sections["Model"], sections["Environment"]

    for profile, home, api_key, base_url in (
        (None, root, A_API_KEY, A_BASE_URL),
        ("b", b, B_API_KEY, B_BASE_URL),
        (None, root, A_API_KEY, A_BASE_URL),
    ):
        model, environment = displayed_values(profile)
        assert model["API Key"] == f"****{api_key[-4:]}"
        assert model["Base URL"] == base_url
        assert environment["Config File"] == str(home / "config.yaml")


def test_launch_profile_agent_build_is_scoped_once_multiplexing(two_homes, monkeypatch):
    """The C6 asymmetry: a default-profile session (``profile_home`` None) in a multiplexing process
    must bind the launch profile's own scope for its agent build instead of running unscoped."""
    from agent.secret_scope import current_secret_scope, set_multiplex_active
    from hermes_constants import get_hermes_home

    root, _b = two_homes
    set_multiplex_active(True)  # the messaging gateway's flip (GatewayRunner.__init__)
    scopes = server._bind_build_profile_scopes(None)
    try:
        scope = current_secret_scope()
        assert scope is not None and scope["A_ONLY_TOKEN"] == A_VAL and scope["INJECTED_TOKEN"] == ENV_VAL
        assert "B_ONLY_TOKEN" not in scope
        assert Path(get_hermes_home()) == root
    finally:
        server._release_build_profile_scopes(scopes)
    assert current_secret_scope() is None


class _MemoryManager:
    """Stands in for an external memory provider: ``system_prompt_block()`` reads its credential via get_secret."""
    def build_system_prompt(self):
        from agent.secret_scope import get_secret
        from hermes_constants import get_hermes_home
        return f"{get_hermes_home()}|{get_secret('MEM_PROVIDER_KEY')}"


def _prompt_building_session(profile_home, key):
    import threading
    from types import SimpleNamespace
    agent = SimpleNamespace(
        _memory_manager=_MemoryManager(), _cached_system_prompt="", session_id=key, model="m", tools=[],
        _session_db=SimpleNamespace(update_system_prompt=lambda sid, prompt: None))
    agent._build_system_prompt = lambda system_message=None: agent._memory_manager.build_system_prompt()
    return {"agent": agent, "history": [], "history_lock": threading.Lock(), "history_version": 0,
            "running": False, "session_key": key, "profile_home": profile_home, "cwd": os.getcwd()}


def test_off_turn_prompt_rebuilds_run_under_the_sessions_profile_scope(two_homes, monkeypatch):
    """Regression for #112927: ``session.context_breakdown`` (Desktop refetches it after every turn) and the
    model-switch prompt re-persist rebuilt the system prompt with no secret scope, so the external memory
    provider's ``system_prompt_block()`` hit ``UnscopedSecretError`` on the LAUNCH profile once the process
    hosted a second home — and for a secondary they resolved the launch profile's credential/home."""
    import agent.system_prompt as system_prompt

    root, b = two_homes
    (root / ".env").write_text((root / ".env").read_text() + "MEM_PROVIDER_KEY=launch-mem-key\n")
    (b / ".env").write_text((b / ".env").read_text() + "MEM_PROVIDER_KEY=b-mem-key\n")
    monkeypatch.setattr(system_prompt, "build_system_prompt_parts",
                        lambda agent, system_message=None: {"stable": "", "context": "",
                                                            "volatile": agent._memory_manager.build_system_prompt()})
    monkeypatch.setattr("agent.context_file_sources.context_file_sources_for_agent", lambda agent: [])
    sessions = {"sa": _prompt_building_session(None, "sess-a"), "sb": _prompt_building_session(str(b), "sess-b")}
    monkeypatch.setattr(server, "_sessions", sessions)
    assert _probe("b")["b_ref"] == B_VAL  # flips the process to fail-closed multi-profile hosting

    for sid, home, key in (("sa", root, "launch-mem-key"), ("sb", b, "b-mem-key"), ("sa", root, "launch-mem-key")):
        resp = server._methods["session.context_breakdown"]("rid", {"session_id": sid})
        assert "error" not in resp, resp
        server._persist_live_session_system_prompt(sessions[sid])
        assert sessions[sid]["agent"]._cached_system_prompt == f"{home}|{key}"
    assert os.environ.get("MEM_PROVIDER_KEY") is None

"""Bot-chain in-process sessions use safe unattended approval semantics."""

import os

from gateway.session_context import (
    _UNSET,
    _VAR_MAP,
    clear_session_vars,
    get_session_env,
)
from tui_gateway import server


def _reset_context() -> None:
    for variable in _VAR_MAP.values():
        variable.set(_UNSET)


def test_session_create_requires_in_process_proof_for_single_query(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    for name in (
        "HERMES_GATEWAY_SESSION",
        "HERMES_EXEC_ASK",
        "HERMES_INTERACTIVE",
    ):
        monkeypatch.delenv(name, raising=False)

    created_ids = []
    try:
        trusted = server._methods["session.create"](
            "trusted",
            {
                "cwd": str(tmp_path),
                "source": "cli",
                "_single_query_proof": server._IN_PROCESS_SINGLE_QUERY_PROOF,
            },
        )["result"]
        created_ids.append(trusted["session_id"])
        trusted_session = server._sessions[trusted["session_id"]]
        assert trusted_session["single_query"] is True
        assert "HERMES_GATEWAY_SESSION" not in os.environ
        assert "HERMES_EXEC_ASK" not in os.environ
        assert "HERMES_INTERACTIVE" not in os.environ

        tokens = server._set_session_context(trusted["stored_session_id"])
        try:
            assert get_session_env("HERMES_SINGLE_QUERY_SESSION") == "1"
        finally:
            clear_session_vars(tokens)

        untrusted = server._methods["session.create"](
            "untrusted",
            {
                "cwd": str(tmp_path),
                "source": "cli",
                "_single_query_proof": "serializable-lookalike",
            },
        )["result"]
        created_ids.append(untrusted["session_id"])
        assert server._sessions[untrusted["session_id"]]["single_query"] is False
        assert os.environ["HERMES_GATEWAY_SESSION"] == "1"
        assert os.environ["HERMES_EXEC_ASK"] == "1"
        assert os.environ["HERMES_INTERACTIVE"] == "1"
    finally:
        for session_id in created_ids:
            server._close_session_by_id(session_id, end_reason="test_cleanup")
        _reset_context()


def test_bot_chain_executor_round_trips_through_real_session_handlers(
    tmp_path, monkeypatch
):
    from agent.bot_chain import (
        BotChainControl,
        HermesSessionRpcTurnExecutor,
        _SessionRPCClient,
    )
    from hermes_cli.active_sessions import try_acquire_active_session
    from hermes_cli.bot_profiles import BotProfile

    sessions_before = set(server._sessions)
    profile_path = tmp_path / "worker"
    profile_path.mkdir()

    def mark_agent_ready(session_id):
        server._sessions[session_id]["agent_ready"].set()

    def finish_turn(_rid, session_id, session, text, **_kwargs):
        with session["history_lock"]:
            session["running"] = False
        server._emit(
            "message.complete",
            session_id,
            {"status": "complete", "text": f"handled: {text}"},
        )

    monkeypatch.setattr(server, "_schedule_agent_build", mark_agent_ready)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: True)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(server, "_run_prompt_submit", finish_turn)
    monkeypatch.setattr(
        server,
        "_profile_home",
        lambda profile: profile_path if profile == "worker" else None,
    )

    def teardown(session, *, end_reason):
        server._release_active_session_slot(session)
        return session is not None

    monkeypatch.setattr(
        server,
        "_teardown_popped_session",
        teardown,
    )

    profile = BotProfile(
        name="worker",
        path=profile_path,
        model="test/model",
        provider="test",
        system_prompt="Work",
    )
    executor = HermesSessionRpcTurnExecutor(_SessionRPCClient(server))
    desktop_lease, refusal = try_acquire_active_session(
        session_id="existing-bot-chat",
        surface="desktop",
        config={},
        metadata={"live_session_id": "desktop-owner"},
        registry_home=profile_path,
    )
    assert desktop_lease is not None and refusal is None
    try:
        assert executor(
            profile,
            "real protocol",
            BotChainControl(),
            conversation_name="Bot Chain real-handlers",
        ) == "handled: real protocol"
    finally:
        desktop_lease.release()
    assert set(server._sessions) == sessions_before

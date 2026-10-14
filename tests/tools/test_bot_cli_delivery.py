"""Local CLI fallback: real mailbox/flock, public dispatch, and actual target turn seam.

Only the external model/transport boundary is faked; no live profiles or services.
"""
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import bot_cli_delivery as delivery, bot_live_delivery as mailbox, bot_mode_dm as dm, bot_relay


@pytest.fixture
def install(tmp_path, monkeypatch):
    root = tmp_path / "install"
    target = root / "profiles" / "ops"
    target.mkdir(parents=True)
    (root / "profile.yaml").write_text("ui_meta:\n  hermes-bots: {}\n", encoding="utf-8")
    (target / "profile.yaml").write_text("description: teammate\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(dm, "_dm_dir", lambda: tmp_path)
    monkeypatch.setattr(bot_relay, "_hermes_cli", lambda: "hermes")
    monkeypatch.setattr(delivery, "RUNNER_WAIT_SECONDS", 0)
    agent = SimpleNamespace(_session_title_hint="Bot Chat", session_id="sender-chat",
                            _session_db=SimpleNamespace(db_path=root / "state.db"))
    calls = []
    monkeypatch.setattr("tools.terminal_tool.terminal_tool",
                        lambda command, **kwargs: calls.append(command) or json.dumps({"session_id": "worker", "notify_on_complete": True}))
    return root, target, agent, calls


def send(install, text="héllo 世界", key="stable"):
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
    _, _, agent, calls = install
    result = json.loads(INLINE_TOOL_EXECUTORS["message_agent"](
        agent, dict(target="ops", message=text, idempotency_key=key), InlineToolContext(effective_task_id="")))
    return result, shlex.split(calls[-1])[2:] if calls else []


def invoke(command):
    return dm._delivery_main(command)


def persisted(install, ack):
    return mailbox.read_delivery_result(install[1], ack["delivery_id"])


def child_turn(monkeypatch, result=None, observed=None):
    """Exercise the actual quiet entrypoint; fake only the external model call."""
    from hermes_cli import quiet_single_query as quiet
    import cli
    from agent.turn_author import TURN_AUTHOR_ENV

    result = result if result is not None else {"final_response": "pong"}
    observed = observed if observed is not None else []
    monkeypatch.setattr(quiet, "continue_quiet_notify_completions", lambda *a, **k: None)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)

    def transport(argv, *, env, report_path, **kwargs):
        monkeypatch.setenv("HERMES_HOME", env["HERMES_HOME"])
        monkeypatch.setenv(TURN_AUTHOR_ENV, env[TURN_AUTHOR_ENV])
        monkeypatch.setenv(delivery.TICKET_ENV, env[delivery.TICKET_ENV])
        monkeypatch.setenv(quiet.TURN_REPORT_FILE_ENV, report_path)
        query = Path(argv[-1]).read_text(encoding="utf-8")
        from hermes_state import SessionDB

        db = SessionDB(db_path=Path(env["HERMES_HOME"]) / "state.db")
        if db.get_session("target-chat") is None:
            db.create_session(session_id="target-chat", source="cli")
            db.set_session_title("target-chat", "Bot Chat")

        def model(**kwargs):
            assert delivery.TICKET_ENV not in os.environ
            observed.append(kwargs)
            db.append_message("target-chat", "user", kwargs["user_message"])
            if result.get("final_response"):
                db.append_message("target-chat", "assistant", result["final_response"])
            return result

        agent = SimpleNamespace(session_id="target-chat", run_conversation=model)
        the_cli = SimpleNamespace(agent=agent, session_id="target-chat", conversation_history=[])
        with pytest.raises(SystemExit) as exited:
            cli._run_quiet_single_query(the_cli, query)
        db.close()
        return subprocess.CompletedProcess(argv, exited.value.code, "UNTRUSTED STDOUT", "")

    monkeypatch.setattr(quiet, "run_reported_turn", transport)
    return observed


@pytest.mark.platforms("posix")
def test_public_busy_handoff_is_durable_deduplicated_fifo_and_resumes(install, monkeypatch, capsys):
    root, target, _, _ = install
    first, command = send(install)
    second, command2 = send(install, "second", "second")
    assert first["status"] == second["status"] == "queued"
    with bot_relay.acquire_turn_lock(root, "ops", timeout_seconds=0):
        assert invoke(command) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["reason"] == "target_busy" and output["status"] == "queued"
    retained = persisted(install, first)
    assert retained["message"].endswith("héllo 世界")
    assert retained["retry_after"] > retained["created_at"] / 1e9
    retry, retry_command = send(install)
    assert retry["delivery_id"] == first["delivery_id"]
    assert retry["sequence"] == first["sequence"] < second["sequence"]
    mismatch, _ = send(install, "different")
    assert "error" in mismatch and persisted(install, first)["message"] == retained["message"]
    # No later turn may overtake the deferred head.
    calls = child_turn(monkeypatch)
    assert invoke(command2) == 0
    assert calls == [] and persisted(install, second)["status"] == "queued"
    monkeypatch.setattr(delivery.time, "time", lambda: retained["retry_after"] + 1)
    assert invoke(retry_command) == 0
    assert persisted(install, first)["status"] == "settled"
    assert invoke(command2) == 0
    assert persisted(install, second)["status"] == "settled"
    assert [c["user_message"] for c in calls] == [retained["message"], "Message from 🤖 hermes (@hermes): second"]
    assert all(c["turn_author"]["id"] == "bot:default" for c in calls)
    assert persisted(install, first)["reply"] == "pong"  # not stdout
    assert persisted(install, first)["message"] == ""  # terminal payload is scrubbed
    from hermes_state import SessionDB

    with SessionDB(db_path=target / "state.db", read_only=True) as db:
        assert db.get_session_by_title("Bot Chat")["id"] == "target-chat"
        assert [row["content"] for row in db.get_messages("target-chat") if row["role"] == "user"] == [
            retained["message"], "Message from 🤖 hermes (@hermes): second"]


@pytest.mark.parametrize("result, expected", [({"final_response": "pong"}, "settled"),
                                              ({"failed": True, "error": "private provider failure"}, "failed")])
def test_target_receipt_not_process_exit_proves_outcome_and_never_replays(install, monkeypatch, result, expected):
    ack, command = send(install)
    calls = child_turn(monkeypatch, result)
    invoke(command)
    record = persisted(install, ack)
    assert record["status"] == expected
    assert record["receipt"]["session_id"] == "target-chat"
    assert record["receipt"]["profile_home"] == str(install[1])
    assert record["receipt"]["payload_digest"] == record["payload_digest"]
    assert delivery.verified_result(install[1], record["delivery_id"], record) == record
    retry, _ = send(install)
    assert retry["status"] == expected
    invoke(command)
    assert len(calls) == 1
    assert "private provider" not in json.dumps(record)


@pytest.mark.parametrize("failure", ["no-receipt", "transport-error", "wrong-token", "wrong-target", "wrong-input", "wrong-author"])
def test_uncertain_outcome_blocks_replay_and_fifo_without_leaking_payload(install, monkeypatch, caplog, capsys, failure):
    from hermes_cli import quiet_single_query as quiet
    ack, command = send(install, "PRIVATE_PAYLOAD_SENTINEL")
    calls = []

    def transport(argv, *, env, **kwargs):
        calls.append(argv)
        if failure == "transport-error":
            raise RuntimeError("PRIVATE_PROVIDER_SECRET")
        if failure != "no-receipt":
            monkeypatch.setenv("HERMES_HOME", env["HERMES_HOME"] if failure != "wrong-target" else str(install[0]))
            ctx = json.loads(env[delivery.TICKET_ENV])
            if failure == "wrong-token":
                ctx["claim_token"] = "wrong"
            query = Path(argv[-1]).read_text()
            author = persisted(install, ack)["author"]
            if failure == "wrong-input":
                query = "other"
            if failure == "wrong-author":
                author = {**author, "id": "bot:forged"}
            with pytest.raises(ValueError):
                delivery.take_target_ticket(query, author, {delivery.TICKET_ENV: json.dumps(ctx)})
        return subprocess.CompletedProcess(argv, 0, "fake success", "")

    monkeypatch.setattr(quiet, "run_reported_turn", transport)
    assert invoke(command) == 1
    assert persisted(install, ack)["status"] == "ambiguous"
    capsys.readouterr()
    later, later_command = send(install, "later", "later")
    assert invoke(command) == 1
    assert invoke(later_command) == 0
    assert len(calls) == 1 and persisted(install, later)["status"] == "queued"
    output = capsys.readouterr().out + caplog.text
    assert "PRIVATE_PAYLOAD_SENTINEL" not in output and "PRIVATE_PROVIDER_SECRET" not in output


def test_only_typed_preturn_refusal_may_requeue(install, monkeypatch):
    from hermes_cli import quiet_single_query as quiet
    ack, command = send(install)
    monkeypatch.setattr(quiet, "run_reported_turn", lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "hermes-refusal-reason: SESSION_NOT_OWNED\nowner busy"))
    assert invoke(command) == 0
    record = persisted(install, ack)
    assert record["status"] == "queued" and record["attempts"] == 1
    assert "claim_token" not in record
    # Repeated transport deliveries during backoff do not consume another attempt.
    assert invoke(command) == 0
    assert persisted(install, ack)["attempts"] == 1


def test_expiry_capacity_and_clock_rollback_are_bounded(install, monkeypatch):
    ack, command = send(install)
    retained = persisted(install, ack)
    monkeypatch.setattr(delivery.time, "time", lambda: retained["expires_at"] + 1)
    assert invoke(command) == 1
    assert persisted(install, ack)["status"] == "cancelled"
    assert persisted(install, ack)["message"] == ""
    # Permanent digest prevents a late duplicate from becoming a fresh input.
    assert send(install)[0]["status"] == "cancelled"
    monkeypatch.setattr(delivery, "MAX_PENDING", 1)
    next_ack, _ = send(install, "second", "second")
    assert next_ack["status"] == "queued" and next_ack["sequence"] > retained["sequence"]
    refused, _ = send(install, "third", "third")
    assert "error" in refused


def test_claim_persists_across_process_death_without_reexecution(install, monkeypatch):
    ack, command = send(install)
    # Actual exec boundary: committed claim then abrupt exit. No live worker is touched.
    script = "import sys; from tools.bot_cli_delivery import claim; claim(sys.argv[1],sys.argv[2]); raise SystemExit(23)"
    proc = subprocess.run([sys.executable, "-c", script, str(install[1]), ack["delivery_id"]],
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 23, proc.stderr
    assert persisted(install, ack)["status"] == "claimed"
    assert invoke(command) == 0
    assert persisted(install, ack)["status"] == "claimed"
    assert send(install)[0]["status"] == "claimed"


def test_missing_ticket_never_falls_back_to_legacy_transport(install, monkeypatch):
    ack, command = send(install)
    path = install[1] / "runtime" / mailbox.DELIVERY_DIR_NAME / f"{ack['delivery_id']}.json"
    path.unlink()
    monkeypatch.setattr(dm, "_run_local_turn", lambda *a, **k: pytest.fail("must not replay a missing durable ticket"))
    assert invoke(command) == 1


def test_live_mailbox_and_peer_boundary_remain_separate(install):
    ack, _ = send(install)
    home = install[1]
    owner = dict(profile_home=str(home), session_id="chat", lease_id="lease", live_session_id="live")
    live = mailbox.deliver_to_live_owner(home, owner, "live input")
    assert mailbox.claim_pending_delivery(home, owner)["delivery_id"] == live["delivery_id"]
    assert persisted(install, ack)["status"] == "queued"
    refused = json.loads(dm.message_agent_tool(target="remote/ops", message="hello", idempotency_key="key", agent=install[2]))
    assert "error" in refused and "nothing sent" in refused["error"]


def test_target_capability_is_one_use_canonical_and_receipt_is_read_back(install, monkeypatch):
    from hermes_state import SessionDB

    ack, _ = send(install)
    home = install[1]
    db = SessionDB(db_path=home / "state.db")
    db.create_session(session_id="canonical", source="cli")
    db.set_session_title("canonical", "Bot Chat")
    db.create_session(session_id="other", source="cli")
    claimed, started = delivery.claim(home, ack["delivery_id"])
    assert started
    context = {"delivery_id": ack["delivery_id"], "claim_token": claimed["claim_token"]}
    monkeypatch.setenv("HERMES_HOME", str(home))

    def take(session):
        return delivery.take_target_ticket(claimed["message"], claimed["author"],
                                           {delivery.TICKET_ENV: json.dumps(context)}, session_id=session)

    with pytest.raises(ValueError, match="canonical"):
        take("other")
    ticket = take("canonical")
    with pytest.raises(ValueError, match="no replay"):
        take("canonical")
    with pytest.raises(ValueError, match="cannot be requeued"):
        delivery.defer(home, ack["delivery_id"], claimed["claim_token"])
    with pytest.raises(ValueError, match="persisted session"):
        delivery.complete_target(ticket, exit_code=0, session_id="other", reply="forged")
    delivery.complete_target(ticket, exit_code=0, session_id="canonical", reply="NO_REPLY")
    record = delivery.verified_result(home, ack["delivery_id"], claimed)
    assert record["reply"] == ""
    # A syntactically valid but mismatched terminal receipt cannot authorize replay/success.
    with mailbox._locked(home) as root:
        record["receipt"]["payload_digest"] = "forged"
        mailbox._write(root / f"{ack['delivery_id']}.json", record)
    assert delivery.verified_result(home, ack["delivery_id"], claimed) is None
    db.close()


def test_budget_clock_rollback_and_new_live_owner_do_not_bypass_pending_cli(install, monkeypatch):
    ack, command = send(install)
    first = persisted(install, ack)
    monkeypatch.setattr(delivery.time, "time", lambda: first["created_at"] / 1e9 - 10)
    monkeypatch.setattr(dm, "_admit_live_dm", lambda *a, **k: pytest.fail("must not bypass older CLI input"))
    second, _ = send(install, "second", "second")
    assert second["sequence"] > first["sequence"]
    monkeypatch.setattr(delivery.time, "time", lambda: first["retry_after"] + 1)
    monkeypatch.setattr(delivery, "MAX_BUSY_ATTEMPTS", 1)
    with bot_relay.acquire_turn_lock(install[0], "ops", timeout_seconds=0):
        assert invoke(command) == 1
    assert persisted(install, ack)["status"] == "cancelled"
    assert persisted(install, ack)["message"] == ""


def test_corrupt_mailbox_refuses_new_admission_without_payload_logging(install, caplog):
    ack, _ = send(install)
    root = install[1] / "runtime" / mailbox.DELIVERY_DIR_NAME
    (root / "corrupt.json").write_text("PRIVATE_CORRUPTION_PAYLOAD")
    # Public entrypoint refuses unreadable admission rather than allowing overtaking.
    refused, _ = send(install, "later", "later")
    assert "error" in refused
    assert "PRIVATE_CORRUPTION_PAYLOAD" not in caplog.text
    assert persisted(install, ack)["status"] == "queued"


@pytest.mark.platforms("posix")
def test_concurrent_duplicate_admission_and_runners_execute_at_most_once(install, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from hermes_cli import quiet_single_query as quiet

    ack, _ = send(install)
    home = install[1]
    initial = persisted(install, ack)
    argv = ["hermes", "-p", "ops", *bot_relay.BOT_CHAT_TURN_ARGS]

    def admit_duplicate(_):
        return delivery.admit(home, sender_home=initial["sender_home"], target_profile="ops",
                              message=initial["message"], author=initial["author"],
                              delivery_id=ack["delivery_id"], dm_file=initial["dm_file"])

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(admit_duplicate, range(8)))
    assert all(r["sequence"] == initial["sequence"] for r in results)
    calls = []

    def transport(*args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "not a receipt", "")

    monkeypatch.setattr(quiet, "run_reported_turn", transport)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: delivery.run_ticket(home, ack["delivery_id"], argv,
                                                   initial["dm_file"], initial["author"]), range(8)))
    # A losing flock contender can defer while the winner is still before claim.
    # In that valid ordering none ran yet; resume after the recorded backoff.
    retained = persisted(install, ack)
    monkeypatch.setattr(delivery.time, "time", lambda: retained["retry_after"] + 1)
    delivery.run_ticket(home, ack["delivery_id"], argv, initial["dm_file"], initial["author"])
    assert len(calls) == 1
    assert persisted(install, ack)["status"] == "ambiguous"

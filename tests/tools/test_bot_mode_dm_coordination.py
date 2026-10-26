"""``message_agent`` coordination: fan-out with one combined reply, notify-only, cancel, live
teammate state, and the single reply path (tools/bot_mode_dm.py + tools/bot_mode_dm_coordination.py)."""

import json
import os
import shlex
import sys

from tools import bot_live_delivery as live
from tools import bot_mode_dm, bot_mode_dm_coordination as coordination

# _fresh_probe_cache is the autouse probe-cache reset; importing it applies it here too.
from tests.tools.test_bot_mode_dm import (
    _FakeAgent, _canonical_target, _capture_spawn, _fresh_probe_cache, _managed_home,
)


def _spec(calls):
    parts = shlex.split(calls[0]["command"])
    with open(parts[parts.index("--fan-out") + 1], encoding="utf-8-sig") as stream:
        return json.load(stream)


def test_fan_out_spawns_one_runner_for_every_teammate(tmp_path, monkeypatch):
    home = _managed_home(tmp_path, teammates=("researcher", "scribe"))
    _canonical_target(monkeypatch, home / "profiles" / "researcher", home / "profiles" / "scribe")
    calls = _capture_spawn(monkeypatch)

    ack = json.loads(bot_mode_dm.message_agent_tool("researcher, @scribe", "status?", agent=_FakeAgent(home)))

    assert len(calls) == 1 and calls[0]["notify_on_complete"] is True  # one wake, not one per teammate
    assert [entry["label"] for entry in _spec(calls)] == ["@researcher", "@scribe"]
    assert [entry["delivery_id"] for entry in _spec(calls)] == [a["delivery_id"] for a in ack["fan_out"]]
    assert ack["process_id"] == "proc_test1234"


def test_fan_out_runner_prints_one_typed_outcome_per_teammate(tmp_path, monkeypatch, capsys):
    script = tmp_path / "child.py"
    script.write_text("import sys, time\nmode = sys.argv[1]\n"
                      "if mode == 'json': print('{\"status\": \"settled\", \"reply\": \"done\", \"target_state\": \"idle\"}')\n"
                      "elif mode == 'fail': print('Delivery to @x failed [reason: runtime_offline]: gone'); sys.exit(1)\n"
                      "else: time.sleep(30)\n", encoding="utf-8")
    entries = [{"command": shlex.join([sys.executable, str(script), mode]), "label": f"@{mode}", "delivery_id": mode,
                "output": "json" if mode == "json" else "text"} for mode in ("json", "fail", "slow")]
    spec = tmp_path / "fanout-1.json"
    spec.write_text(json.dumps(entries), encoding="utf-8")
    monkeypatch.setattr(coordination, "FAN_OUT_WAIT_SECONDS", 3)

    assert coordination.run_fan_out(str(spec)) == 1
    out = json.loads(capsys.readouterr().out)
    by_to = {entry["to"]: entry for entry in out["replies"]}
    assert by_to["@json"] == {"to": "@json", "delivery_id": "json", "status": "settled", "reply": "done",
                              "target_state": "idle"}
    assert by_to["@fail"]["status"] == "failed" and "runtime_offline" in by_to["@fail"]["error"]
    assert by_to["@slow"]["status"] == "pending"
    assert (out["settled"], out["of"]) == (1, 3) and not spec.exists()


def test_notify_only_never_starts_a_waiter_and_marks_the_message(tmp_path, monkeypatch):
    home = _managed_home(tmp_path)
    authority = _canonical_target(monkeypatch, home / "profiles" / "researcher")
    calls = _capture_spawn(monkeypatch)
    before = set(os.listdir(bot_mode_dm._dm_dir()))

    ack = json.loads(bot_mode_dm.message_agent_tool("researcher", "FYI: build is green", agent=_FakeAgent(home),
                                                    expects_reply=False))

    assert calls == [] and ack["expects_reply"] is False and ack["delivery_id"]
    assert authority.calls[0][1]["message"].endswith("): [no reply needed] FYI: build is green")
    assert set(os.listdir(bot_mode_dm._dm_dir())) == before  # no payload or intent left for a runner that never runs


def test_a_reply_to_the_turns_sender_goes_in_the_final_response(tmp_path, monkeypatch):
    home = _managed_home(tmp_path)
    authority = _canonical_target(monkeypatch, home / "profiles" / "researcher")
    calls = _capture_spawn(monkeypatch)
    agent = _FakeAgent(home)
    agent._turn_author = {"id": "bot:researcher", "name": "researcher", "is_bot": True}

    out = json.loads(bot_mode_dm.message_agent_tool("researcher", "here is my answer", agent=agent))

    assert "final response already goes back" in out["error"]
    assert authority.calls == [] and calls == []


def test_cancel_is_the_senders_own_and_reaches_the_owner(tmp_path, monkeypatch):
    home = _managed_home(tmp_path)
    target = home / "profiles" / "researcher"
    _canonical_target(monkeypatch, target)
    _capture_spawn(monkeypatch)
    ack = json.loads(bot_mode_dm.message_agent_tool("researcher", "long job", agent=_FakeAgent(home)))
    with live._locked(target) as root:  # the owner's receipt records who sent it
        path = root / f"{ack['delivery_id']}.json"
        record = json.loads(path.read_text(encoding="utf-8-sig"))
        live._write(path, {**record, "author": {"id": "bot:default", "name": "hermes", "is_bot": True}})
    rpcs = []
    monkeypatch.setattr(live, "_authority_rpc", lambda h, method, params: rpcs.append((method, params)) or
                        {"status": "terminal", "outcome": "cancelled"})

    stranger = _FakeAgent(home)
    stranger._session_db.db_path = str(target / "state.db")  # @researcher itself asks
    refused = json.loads(bot_mode_dm.message_agent_tool("researcher", cancel=ack["delivery_id"], agent=stranger))
    done = json.loads(bot_mode_dm.message_agent_tool("researcher", cancel=ack["delivery_id"], agent=_FakeAgent(home)))

    assert "Only the agent that sent" in refused["error"]
    assert done["status"] == "cancelled"
    assert rpcs == [("prompt.cancel", {"session_id": "local-bot", "admission_id": record["admission_id"]})]


def test_ack_reports_a_teammate_parked_on_its_user(tmp_path, monkeypatch):
    home = _managed_home(tmp_path)
    authority = _canonical_target(monkeypatch, home / "profiles" / "researcher")
    plain = authority.__call__
    monkeypatch.setattr(live, "authority_delivery", lambda h, p: {**plain(h, p), "target_state": "waiting_approval"})
    _capture_spawn(monkeypatch)

    ack = json.loads(bot_mode_dm.message_agent_tool("researcher", "deploy?", agent=_FakeAgent(home)))

    assert ack["target_state"] == "waiting_approval"
    assert "waiting for its user to approve" in ack["detail"]


def _run_children(tmp_path, monkeypatch, capsys, body: str) -> dict:
    script = tmp_path / "child.py"
    script.write_text(body, encoding="utf-8")
    spec = tmp_path / "fanout-2.json"
    spec.write_text(json.dumps([{"command": shlex.join([sys.executable, str(script)]), "label": "@peer",
                                 "delivery_id": "d1", "output": "text"}]), encoding="utf-8")
    coordination.run_fan_out(str(spec))
    return json.loads(capsys.readouterr().out)["replies"][0]


def test_a_text_reply_that_looks_like_json_stays_the_reply(tmp_path, monkeypatch, capsys):
    reply = _run_children(tmp_path, monkeypatch, capsys, "print('Reply from @peer:')\nprint('{\"status\": \"x\"}')\n")

    assert reply["status"] == "settled" and reply["reply"].endswith('{"status": "x"}')


def test_an_accepted_but_unfinished_peer_request_is_pending_with_its_warning(tmp_path, monkeypatch, capsys):
    reply = _run_children(tmp_path, monkeypatch, capsys,
                          "import sys\nprint(\"Peer 'p' accepted the message but its turn is still running. \"\n"
                          "      \"Do NOT resend.\", file=sys.stderr)\nsys.exit(1)\n")

    assert reply["status"] == "pending" and "accepted the message" in reply["detail"]


def test_one_name_for_one_bot_sends_once_and_a_comma_name_resolves_whole(tmp_path, monkeypatch):
    home = _managed_home(tmp_path, teammates=("researcher", "scribe"))
    (home / "profiles" / "scribe" / "profile.yaml").write_text("display_name: QA, Release\n", encoding="utf-8")
    authority = _canonical_target(monkeypatch, home / "profiles" / "researcher", home / "profiles" / "scribe")
    _capture_spawn(monkeypatch)

    bot_mode_dm.message_agent_tool("researcher, RESEARCHER", "status?", agent=_FakeAgent(home))
    bot_mode_dm.message_agent_tool("QA, Release", "ship?", agent=_FakeAgent(home))

    assert [target.name for target, _ in authority.calls] == ["researcher", "scribe"]


def test_one_targets_setup_failure_keeps_the_others_combined_reply(tmp_path, monkeypatch):
    home = _managed_home(tmp_path, teammates=("researcher", "scribe"))
    _canonical_target(monkeypatch, home / "profiles" / "researcher", home / "profiles" / "scribe")
    calls = _capture_spawn(monkeypatch)
    dispatch = bot_mode_dm._dispatch

    def flaky(target, *args, **kwargs):
        if target == "scribe":
            raise OSError("disk full")
        return dispatch(target, *args, **kwargs)

    monkeypatch.setattr(bot_mode_dm, "_dispatch", flaky)
    ack = json.loads(bot_mode_dm.message_agent_tool("researcher, scribe", "status?", agent=_FakeAgent(home)))

    assert [entry["label"] for entry in _spec(calls)] == ["@researcher"]
    assert "disk full" in ack["fan_out"][1]["error"] and not ack["fan_out"][0].get("error")


def test_cancel_follows_the_retry_the_owner_is_admitting(tmp_path, monkeypatch):
    home = _managed_home(tmp_path)
    target = home / "profiles" / "researcher"
    _canonical_target(monkeypatch, target)
    _capture_spawn(monkeypatch)
    ack = json.loads(bot_mode_dm.message_agent_tool("researcher", "long job", agent=_FakeAgent(home)))
    with live._locked(target) as root:
        path = root / f"{ack['delivery_id']}.json"
        record = {**json.loads(path.read_text(encoding="utf-8-sig")),
                  "author": {"id": "bot:default", "name": "hermes", "is_bot": True}}
        live._write(path, record)
    rpcs = []

    def owner(h, method, params):
        rpcs.append(params["admission_id"])
        if params["admission_id"] == record["admission_id"]:  # the first attempt failed; the owner admits its retry
            with live._locked(target) as root:
                live._write(root / f"{ack['delivery_id']}.json", {**record, "retry_admission_id": "retry-1"})
            return {"status": "terminal", "outcome": "failed"}
        return {"status": "terminal", "outcome": "cancelled"}

    monkeypatch.setattr(live, "_authority_rpc", owner)
    done = json.loads(bot_mode_dm.message_agent_tool("researcher", cancel=ack["delivery_id"], agent=_FakeAgent(home)))

    assert done["status"] == "cancelled" and rpcs == [record["admission_id"], "retry-1"]

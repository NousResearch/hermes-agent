"""Tests: bot_relay.* JSON-RPC handlers (tui_gateway/methods_bot_relay.py).

The Desktop's relay door on each connected gateway. Contracts:
- roster.sync persists validated rows and reports the accepted count;
- outbox.drain returns queued envelopes exactly once;
- deliver validates the target profile against THIS install and runs the
  one-turn Bot Chat transport (the turn runner is faked here — the argv
  contract is what's pinned; the runner itself is pinned below with real
  children: its cap bounds the turn, not the exit linger, #114980);
- reply writes the waiter's file and rejects malformed envelope ids.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from unittest import mock

import pytest

import tui_gateway.server as srv
from hermes_cli.dashboard_auth.ws_tickets import INTERNAL_PROVIDER, INTERNAL_USER_ID
from tools import bot_relay
from tui_gateway import methods_bot_relay


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    (h / "profiles" / "ops").mkdir(parents=True)
    (h / "profiles" / "ops" / "config.yaml").write_text("{}\n")  # identity marker: a bare dir is no target
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


def _result(envelope):
    assert "error" not in envelope, envelope
    return envelope["result"]


def test_roster_sync_persists_and_counts(home):
    out = _result(
        srv._methods["bot_relay.roster.sync"](
            1,
            {
                "agents": [
                    {"profile": "scout", "handle": "scout", "connection_id": "cloud-1"},
                    {"profile": "", "connection_id": "cloud-1"},  # dropped
                ]
            },
        )
    )
    assert out["count"] == 1
    assert [r["profile"] for r in bot_relay.read_remote_roster(home)] == ["scout"]


def test_outbox_drain_returns_each_envelope_once(home):
    target = {"profile": "scout", "handle": "scout", "connection_id": "cloud-1",
              "connection_label": "", "title": "", "description": ""}
    env = bot_relay.enqueue_envelope(
        home, target=target, message="m", sender_profile="default", sender_handle="hermes"
    )
    first = _result(srv._methods["bot_relay.outbox.drain"](1, {}))
    assert [e["id"] for e in first["envelopes"]] == [env["id"]]
    second = _result(srv._methods["bot_relay.outbox.drain"](2, {}))
    assert second["envelopes"] == []


def test_deliver_validates_profile_and_runs_transport(home, monkeypatch):
    calls = {}

    class _Proc:
        returncode = 0
        stdout = "pong from ops"
        stderr = ""

    def _fake_run(argv, **kwargs):
        calls["argv"] = argv
        calls["kwargs"] = kwargs
        return _Proc()

    monkeypatch.setattr("hermes_cli.quiet_single_query.run_reported_turn", _fake_run)
    out = _result(
        srv._methods["bot_relay.deliver"](1, {"profile": "ops", "message": "ping"})
    )
    assert out["reply"] == "pong from ops"
    # The cap bounds the turn; the relay wants the printed answer, so a reported child is
    # booked only at the cap (exit_grace=None), never early like the cron lane (#114980).
    assert calls["kwargs"]["timeout"] == bot_relay.TURN_ATTEMPT_TIMEOUT_SECONDS
    assert calls["kwargs"]["exit_grace"] is None
    assert calls["kwargs"]["report_path"].endswith(".turn.json")
    argv = calls["argv"]
    # argv[0] may be a resolved venv path (#93590) — match by basename.
    assert argv[1:3] == ["-p", "ops"]
    assert argv[0].rsplit("\\", 1)[-1].rsplit("/", 1)[-1] in ("hermes", "hermes.exe")
    assert "Bot Chat" in argv and "--query-file" in argv

    # 'hermes' alias resolves to default
    _result(srv._methods["bot_relay.deliver"](2, {"profile": "hermes", "message": "x"}))
    assert calls["argv"][1:3] == ["-p", "default"]

    # unknown profile refuses without spawning; so does a bare infra dir under profiles/ (#99392)
    calls.clear()
    (home / "profiles" / "sessions" / "cron").mkdir(parents=True)
    for target in ("ghost", "sessions"):
        err = srv._methods["bot_relay.deliver"](3, {"profile": target, "message": "x"})
        assert "error" in err and target in err["error"]["message"]
    assert not calls


def test_deliver_requires_params(home):
    err = srv._methods["bot_relay.deliver"](1, {"profile": "", "message": ""})
    assert "error" in err


def test_deliver_restamps_relayed_sender_with_a_reply_safe_handle(home, monkeypatch):
    """#103731: the sender signs with its bare @handle, which for another machine's ``default`` is
    ``@hermes`` — the recipient's OWN default. The delivered text names the sender by the form this
    gateway resolves back to it: its title slug when the local relay roster carries it, else
    ``handle@connection``. A stamp that is not the relay's is left alone."""
    seen = []

    class _Proc:
        returncode, stderr, stdout = 0, "", "ok"

    def _fake_run(argv, **_kwargs):
        seen.append(Path(argv[argv.index("--query-file") + 1]).read_text(encoding="utf-8"))
        return _Proc()

    # The deliver child's runner, whichever this tree has: subprocess.run today, and
    # quiet_single_query.run_reported_turn once the relay books turns from their report
    # (#114980) — patching only the first would spawn a real ``hermes chat -Q`` child there.
    monkeypatch.setattr("subprocess.run", _fake_run)
    monkeypatch.setattr("hermes_cli.quiet_single_query.run_reported_turn", _fake_run, raising=False)
    bot_relay.write_remote_roster(home, [
        {"profile": "default", "handle": "hermes", "connection_id": "vps-1", "title": "CoS Bot"},
    ])
    stamp = "Message from 🤖 CoS Bot (@hermes): are we done?"
    sender = {"from_profile": "default", "from_handle": "hermes", "from_connection": "vps-1"}
    _result(srv._methods["bot_relay.deliver"](1, {"profile": "ops", "message": stamp, **sender}))
    _result(srv._methods["bot_relay.deliver"](2, {"profile": "ops", "message": stamp, **sender, "from_connection": "lan-2"}))
    _result(srv._methods["bot_relay.deliver"](3, {"profile": "ops", "message": "plain text (@hermes): x", **sender}))
    assert seen == ["Message from 🤖 CoS Bot (@cos-bot): are we done?",
                    "Message from 🤖 CoS Bot (@hermes@lan-2): are we done?",
                    "plain text (@hermes): x"]


def test_deliver_relays_empty_reply_for_a_bare_silence_marker(home, monkeypatch):
    """#110782: the subprocess transport applies the gateway's silence rule — a bare marker
    relays as "", prose that merely mentions one is relayed verbatim."""
    class _Proc:
        returncode, stderr = 0, ""
        stdout = " *NO_REPLY* "

    monkeypatch.setattr("hermes_cli.quiet_single_query.run_reported_turn", lambda *_a, **_k: _Proc())
    assert _result(srv._methods["bot_relay.deliver"](1, {"profile": "ops", "message": "ping"}))["reply"] == ""

    _Proc.stdout = "The NO_REPLY marker means do not answer."
    out = _result(srv._methods["bot_relay.deliver"](1, {"profile": "ops", "message": "ping"}))
    assert out["reply"] == _Proc.stdout.strip()


def test_deliver_lands_in_live_bot_chat_instead_of_subprocess(home, monkeypatch):
    """#100523: a Desktop-owned Bot Chat receives the DM as a normal user turn.

    With the target's Bot Chat live in this gateway, the subprocess transport
    would be fenced out by the single-owner lease and drop the payload. The
    handler must route through prompt.submit (the composer's choke point) and
    never spawn the CLI.
    """
    spawned = []
    submitted = []

    class _Proc:
        returncode, stdout, stderr = 0, "pong", ""

    def _fake_run(argv, *a, **k):
        # The server module's import-time update prefetch runs `git ...` on a
        # daemon thread; only the relay's `hermes` CLI spawn is under test.
        if argv and argv[0] != "git":
            spawned.append(argv)
        return _Proc()

    monkeypatch.setattr("hermes_cli.quiet_single_query.run_reported_turn", _fake_run)
    monkeypatch.setitem(
        srv._methods, "prompt.submit", lambda rid, p: submitted.append(p) or srv._ok(rid, {"status": "streaming"})
    )
    monkeypatch.setattr(srv, "_profile_home", lambda name: home / "profiles" / name)
    monkeypatch.setitem(
        srv._sessions,
        "live-ops",
        {"profile_home": str(home / "profiles" / "ops"), "pending_title": "Bot Chat", "history": []},
    )
    out = _result(srv._methods["bot_relay.deliver"](1, {"profile": "ops", "message": "ping"}))
    # queued=True is the invariant: a DM never interrupts a turn in flight.
    assert submitted == [{"session_id": "live-ops", "text": "ping", "queued": True}]
    assert not spawned
    assert "reply" in out

    # A live session titled anything else for the same profile does not qualify:
    # the subprocess path runs exactly as before.
    srv._sessions["live-ops"]["pending_title"] = "Scratch"
    submitted.clear()

    out = _result(srv._methods["bot_relay.deliver"](2, {"profile": "ops", "message": "ping"}))
    assert out["reply"] == "pong" and spawned and not submitted


def test_deliver_hands_off_to_a_bot_chat_owned_by_another_process(home, monkeypatch):
    """#113753: the relay RPC lands in whichever backend the Desktop routes the target CONNECTION
    to, while the Desktop-opened Bot Chat can be live in a sibling process for that profile
    (per-(connection, profile) pool, per-profile SSH dashboards). That owner's lease refuses the
    subprocess transport with SESSION_NOT_OWNED, so the handler must hand the DM to the owner
    through the durable mailbox local DMs use, and never spawn the CLI.
    """
    from hermes_cli.active_sessions import try_acquire_active_session
    from hermes_state import SessionDB
    from tools import bot_live_delivery as mailbox

    ops_home = home / "profiles" / "ops"
    db = SessionDB(db_path=ops_home / "state.db")
    db.create_session(session_id="chat", source="desktop")
    db.set_session_title("chat", "Bot Chat")
    db.close()
    # The sibling process's lease: a mailbox-capable live owner registered in the target's home.
    lease, refusal = try_acquire_active_session(
        session_id="chat", surface="desktop", config={}, registry_home=ops_home,
        metadata={"live_session_id": "live-in-other-process", "bot_live_delivery_consumer": True})
    assert refusal is None
    spawned = []

    def _fake_run(argv, *a, **k):
        if argv and argv[0] != "git":
            spawned.append(argv)
        raise AssertionError("the CLI transport collides with the live owner")

    monkeypatch.setattr("hermes_cli.quiet_single_query.run_reported_turn", _fake_run)
    monkeypatch.setattr(srv, "_profile_home", lambda name: ops_home)
    monkeypatch.setattr(srv, "_sessions", {})  # THIS process hosts nothing for ops
    try:
        out = _result(srv._methods["bot_relay.deliver"](1, {
            "profile": "ops", "message": "ping", "from_profile": "cody", "from_handle": "cody",
            "from_connection": "conn-a"}))
        assert not spawned and "open Bot Chat" in out["reply"]
        (queued,) = [
            r for p in (ops_home / "runtime" / mailbox.DELIVERY_DIR_NAME).glob("*.json")
            if (r := json.loads(p.read_text(encoding="utf-8")))]
        assert queued["status"] == "queued" and queued["message"] == "ping"
        assert queued["owner"]["lease_id"] == lease.lease_id
        assert queued["author"]["name"] == "cody" and queued["author"]["is_bot"] is True
    finally:
        lease.release()


def test_reply_roundtrip_and_id_validation(home):
    envelope_id = "c" * 32
    _result(srv._methods["bot_relay.reply"](1, {"id": envelope_id, "reply": "hi"}))
    path = bot_relay.relay_root(home) / bot_relay.REPLIES_DIR / f"{envelope_id}.json"
    assert json.loads(path.read_text(encoding="utf-8"))["reply"] == "hi"

    err = srv._methods["bot_relay.reply"](2, {"id": "../evil"})
    assert "error" in err


def test_deliver_write_failure_still_removes_tempfile(home, monkeypatch, tmp_path):
    """A failed payload write must not leak the relay DM tempfile."""
    import glob
    import os
    import tempfile as _tempfile

    made = []
    real_mkstemp = _tempfile.mkstemp

    def _tracking_mkstemp(*args, **kwargs):
        kwargs["dir"] = str(tmp_path)
        fd, path = real_mkstemp(*args, **kwargs)
        made.append(path)
        return fd, path

    class _BrokenWriter:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def write(self, content):
            raise OSError("disk full")

    monkeypatch.setattr("tempfile.mkstemp", _tracking_mkstemp)
    monkeypatch.setattr("os.fdopen", lambda *a, **k: _BrokenWriter())
    err = srv._methods["bot_relay.deliver"](1, {"profile": "ops", "message": "x"})
    assert "error" in err
    assert made, "mkstemp was never reached"
    assert not glob.glob(str(tmp_path / "hermes-relay-dm-*")), "tempfile leaked"


@pytest.fixture
def fake_runs(monkeypatch):
    """Fake ``subprocess.run`` that records each call's kwargs; ``outcomes`` holds (returncode, stderr) per call."""
    calls, outcomes = [], []

    def _fake_run(argv, **kwargs):
        calls.append(kwargs)
        code, err = outcomes.pop(0) if outcomes else (0, "")

        class _Proc:
            returncode, stdout, stderr = code, "ok" if code == 0 else "", err

        return _Proc()

    monkeypatch.setattr("hermes_cli.quiet_single_query.run_reported_turn", _fake_run)
    return calls, outcomes


@pytest.mark.parametrize("sender, expected", [
    ({"from_profile": "scout", "from_handle": "scout"}, {"id": "bot:scout", "name": "scout", "is_bot": True}),
    ({"from_profile": "scout", "from_handle": "scout", "from_connection": "cloud-1"},
     {"id": "bot:cloud-1/scout", "name": "scout", "is_bot": True}),
    ({}, None),
], ids=["sender fields", "sender on another connection", "no sender fields"])
def test_deliver_child_env_carries_the_envelope_sender_on_every_attempt(home, monkeypatch, fake_runs, sender, expected):
    """HERMES_TURN_AUTHOR on the child comes from the envelope's sender fields alone: the retry gets the same
    author, and without sender fields a stale author on the gateway's own environment never reaches the child."""
    from agent.turn_author import TURN_AUTHOR_ENV

    calls, outcomes = fake_runs
    outcomes.extend([(1, "HTTP 429 rate limit"), (0, "")])
    monkeypatch.setenv("HERMES_RELAY_TEST_MARKER", "kept")
    monkeypatch.setenv(TURN_AUTHOR_ENV, json.dumps({"id": "bot:stale", "name": "stale", "is_bot": True}))

    _result(srv._methods["bot_relay.deliver"](1, {"profile": "ops", "message": "ping", **sender}))

    envs = [c["env"] for c in calls]
    assert len(envs) == 2
    assert [json.loads(e[TURN_AUTHOR_ENV]) if TURN_AUTHOR_ENV in e else None for e in envs] == [expected, expected]
    assert all(e["HERMES_RELAY_TEST_MARKER"] == "kept" for e in envs)


class _Client:
    def __init__(self, auth_identity=None):
        self.auth_identity = auth_identity

    def write(self, obj):
        return True

    def close(self):
        return None


@pytest.fixture
def bound_client(monkeypatch):
    """Bind a fake calling transport for the handler; yields a setter for its ``auth_identity``."""
    client = _Client()
    token = srv.bind_transport(client)
    try:
        yield client
    finally:
        srv.reset_transport(token)


SENDER = {"from_profile": "scout", "from_handle": "scout", "from_connection": "cloud-1"}
SENDER_AUTHOR = {"id": "bot:cloud-1/scout", "name": "scout", "is_bot": True}


@pytest.mark.parametrize("identity", [
    None,
    {"user_id": INTERNAL_USER_ID, "provider": INTERNAL_PROVIDER},
], ids=["no identity", "server-internal identity"])
def test_deliver_accepts_a_sender_from_an_admitted_non_login_client(home, fake_runs, bound_client, identity):
    """A caller with no identity, or one holding the ``?internal=`` credential, keeps its sender fields.

    NOT the Desktop: it mints a ws-ticket carrying the signed-in ``{user_id, provider}`` on every
    gateway that requires sign-in (``hermes_cli/dashboard_auth/routes.py``), so it is a login
    identity and takes the principal-author branch above.
    """
    from agent.turn_author import TURN_AUTHOR_ENV

    calls, _outcomes = fake_runs
    bound_client.auth_identity = identity

    _result(srv._methods["bot_relay.deliver"](1, {"profile": "ops", "message": "ping", **SENDER}))

    assert [json.loads(c["env"][TURN_AUTHOR_ENV]) for c in calls] == [SENDER_AUTHOR]


def test_deliver_from_a_logged_in_client_is_attributed_to_its_principal_never_to_the_claimed_sender(
        home, fake_runs, bound_client):
    """A logged-in client's sender fields are not trusted — but the dm is neither refused nor left unattributed.

    Refusing the CALL (the original guard) took cross-machine relay offline for every auth-gated gateway,
    because the Desktop is itself a logged-in client there. Dropping the AUTHOR instead made the turn the
    human's to the recipient's memory (Honcho routes an unattributed turn into the human session and allows
    conclusion / profile / mirror writes). So the author is derived from the caller's minted identity:
    stable, unspoofable, and still a bot — whether or not the client named a sender.
    """
    import json

    from agent.turn_author import TURN_AUTHOR_ENV

    calls, _outcomes = fake_runs
    bound_client.auth_identity = {"user_id": "alice", "provider": "google"}

    shapes = ({"from_profile": "scout"}, {"from_connection": "cloud-1"}, SENDER, {})
    for rid, sender in enumerate(shapes):
        _result(srv._methods["bot_relay.deliver"](rid, {"profile": "ops", "message": "ping", **sender}))

    assert len(calls) == len(shapes), "every relayed dm from a logged-in client must still run its turn"
    authors = [json.loads(c["env"][TURN_AUTHOR_ENV]) for c in calls]
    assert all(a["is_bot"] is True for a in authors), "a relayed dm stays bot-authored for the recipient's memory"
    assert all(a["id"].startswith("bot:principal:dashboard:") and a["id"].endswith("/relay") for a in authors)
    assert all(a["name"] == "relayed teammate" for a in authors)
    assert SENDER_AUTHOR not in authors, "the claimed sender must not become the author"
    assert len({a["id"] for a in authors}) == 1, "one signed-in principal, one author — with or without sender fields"

    bound_client.auth_identity = {"user_id": "bob", "provider": "google"}
    _result(srv._methods["bot_relay.deliver"](9, {"profile": "ops", "message": "ping", **SENDER}))
    assert json.loads(calls[-1]["env"][TURN_AUTHOR_ENV])["id"] != authors[0]["id"], "a different principal is a different author"


@pytest.mark.parametrize("subdir", ["profiles/ops", "dev"])
def test_gateway_drains_the_mailbox_the_tools_write_to(tmp_path, monkeypatch, subdir):
    """Both ends of the relay mailbox derive the install root from HERMES_HOME with ONE formula.
    The writer side (``message_agent``'s ``_hermes_root``) and the drain side
    (``methods_bot_relay._relay_root``) must agree for a ``profiles/<name>`` home AND for an
    arbitrary subdir of the native ``~/.hermes`` — a split here is silent non-delivery."""
    from tools.bot_mode_probe import _default_home, _hermes_root
    from tui_gateway import methods_bot_relay

    monkeypatch.setenv("HOME", str(tmp_path))
    home = tmp_path / ".hermes" / subdir
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    writer_root = _hermes_root(Path(_default_home()))
    target = {"profile": "scout", "handle": "scout", "connection_id": "cloud-1",
              "connection_label": "", "title": "", "description": ""}
    env = bot_relay.enqueue_envelope(
        writer_root, target=target, message="m", sender_profile="default", sender_handle="hermes")

    assert methods_bot_relay._relay_root() == writer_root
    drained = _result(srv._methods["bot_relay.outbox.drain"](1, {}))
    assert [e["id"] for e in drained["envelopes"]] == [env["id"]]


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _child_argv(monkeypatch, body: str) -> dict:
    """Stand in a Python child for the ``hermes`` transport; it imports ``hermes_cli`` from this checkout."""
    argv = [sys.executable, "-c", textwrap.dedent(body)]
    monkeypatch.setattr(bot_relay, "local_delivery_command", lambda prof, tmp: argv)
    return {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (_REPO_ROOT, os.environ.get("PYTHONPATH")) if p)}


def _spy_popen():
    procs, real_popen = [], subprocess.Popen

    def spy(*args, **kwargs):
        procs.append(real_popen(*args, **kwargs))
        return procs[-1]

    return procs, spy


def test_reported_turn_still_lingering_at_the_cap_is_booked_from_its_latest_report_not_killed(tmp_path, monkeypatch):
    """#114980: the cap bounds the TURN. A child that reported its turn and then lingers for a
    nested notify_on_complete reply (bounded by oneshot_completion_wait_seconds, default == the
    cap) is booked at the cap from its report — the answer a follow-up turn last wrote there,
    exit code 0, never delivery_timeout — and is NOT killed, so its own handoff survives."""
    env = _child_argv(monkeypatch, """
        import os, time
        from hermes_cli.quiet_single_query import TURN_REPORT_FILE_ENV, write_turn_report
        path = os.environ.pop(TURN_REPORT_FILE_ENV)
        write_turn_report(path, exit_code=0, reply="asking the teammate")
        time.sleep(0.5)
        write_turn_report(path, exit_code=0, reply="teammate says: done")
        time.sleep(30)
        """)
    procs, spy = _spy_popen()
    tmp = tmp_path / "dm.txt"
    tmp.write_text("hi", encoding="utf-8")
    started = time.monotonic()
    try:
        with mock.patch.object(subprocess, "Popen", side_effect=spy) as popen:
            result = methods_bot_relay._run_delivery("ops", str(tmp), env, timeout=2)
        elapsed = time.monotonic() - started
        assert (result.returncode, result.stdout, result.stderr) == (0, "teammate says: done", "")
        assert 2 <= elapsed < 8, elapsed
        assert procs[0].poll() is None, "the lingering child must survive the booking"
        assert not (tmp_path / "dm.txt.turn.json").exists(), "the report is the runner's to clean up"
        # Decoding stays pinned through the runner (#93590 sibling defect): without encoding= the
        # child's UTF-8 output is decoded with the locale codec — cp1252/GBK on Windows — mangling
        # non-ASCII replies; errors="replace" keeps a bad byte from raising instead of delivering.
        assert popen.call_args.kwargs["encoding"] == "utf-8" and popen.call_args.kwargs["errors"] == "replace"
    finally:
        for proc in procs:
            proc.kill()
            proc.wait(timeout=10)


def test_turn_that_never_ends_is_still_a_delivery_timeout(tmp_path, monkeypatch):
    """Control: with no turn report the cap stays the guard it always was."""
    env = _child_argv(monkeypatch, "import time; time.sleep(30)")
    tmp = tmp_path / "dm.txt"
    tmp.write_text("hi", encoding="utf-8")
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        methods_bot_relay._run_delivery("ops", str(tmp), env, timeout=1)
    assert time.monotonic() - started < 8

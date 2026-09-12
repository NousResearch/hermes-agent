"""Tests for tools/bot_mode_dm.py — the Bot-Chat-only ``message_agent`` tool.

The containment contract is the headline here: the tool must exist ONLY in a
canonical Bot Chat session on a Bot-Mode-managed install, and must refuse to
deliver from anywhere else even if a schema leaks.
"""

import json
import os
import shlex
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from tools import bot_mode_dm, bot_mode_probe


@pytest.fixture(autouse=True)
def _fresh_probe_cache():
    bot_mode_probe._reset_cache_for_tests()
    yield
    bot_mode_probe._reset_cache_for_tests()


def _managed_home(tmp_path, *, teammates=("researcher",), peers=()) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    for name in teammates:
        d = home / "profiles" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "profile.yaml").write_text(
            textwrap.dedent(
                """\
                description: teammate for tests
                ui_meta:
                  hermes-bots:
                    shape: cloud
                """
            ),
            encoding="utf-8",
        )
    if peers:
        lines = ["bot_peers:"]
        for peer in peers:
            lines += [f"  {peer}:", f"    url: http://{peer}.lan:8377"]
        (home / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return home


class _FakeDB:
    def __init__(self, home: Path, title: str):
        self.db_path = str(home / "state.db")
        self._title = title

    def get_session_title(self, _sid):
        return self._title


class _FakeAgent:
    def __init__(self, home: Path, title: str = "Bot Chat"):
        self._session_db = _FakeDB(home, title)
        self.session_id = "sess-1"
        self._session_title_hint = None
        self._bot_mode_protocol = True
        self.tools: list = []
        self.valid_tool_names: set = set()


# ── injection gate (leak containment) ────────────────────────────────────────


def test_injects_only_into_bot_chat_on_managed_install(tmp_path):
    home = _managed_home(tmp_path)
    agent = _FakeAgent(home, title="Bot Chat")
    assert bot_mode_dm.ensure_message_agent_tool(agent) is True
    names = [t["function"]["name"] for t in agent.tools]
    assert names == [bot_mode_dm.MESSAGE_AGENT_TOOL_NAME]
    assert bot_mode_dm.MESSAGE_AGENT_TOOL_NAME in agent.valid_tool_names

    # idempotent: second call adds nothing (byte-stable tool list per turn)
    assert bot_mode_dm.ensure_message_agent_tool(agent) is True
    assert len(agent.tools) == 1


@pytest.mark.parametrize(
    "title",
    ["", "My research chat", "Group: room-abc123", "handoff-12ab34cd"],
)
def test_never_injects_outside_bot_chat(tmp_path, title):
    """CLI sessions, ordinary chats, group-room member sessions: no tool."""
    home = _managed_home(tmp_path)
    agent = _FakeAgent(home, title=title)
    assert bot_mode_dm.ensure_message_agent_tool(agent) is False
    assert agent.tools == []
    assert agent.valid_tool_names == set()


def test_never_injects_on_unmanaged_install(tmp_path):
    """A 'Bot Chat'-titled session on a plain install stays tool-free."""
    home = tmp_path / ".hermes"
    home.mkdir()
    agent = _FakeAgent(home, title="Bot Chat")
    assert bot_mode_dm.ensure_message_agent_tool(agent) is False
    assert agent.tools == []


def test_config_toggle_disables_injection(tmp_path):
    home = _managed_home(tmp_path)
    agent = _FakeAgent(home, title="Bot Chat")
    agent._bot_mode_protocol = False
    assert bot_mode_dm.ensure_message_agent_tool(agent) is False
    assert agent.tools == []


def test_schema_never_in_global_registry():
    """message_agent must not be registered/toolset-reachable anywhere."""
    from tools.registry import registry

    assert bot_mode_dm.MESSAGE_AGENT_TOOL_NAME not in getattr(registry, "_tools", {})
    import toolsets

    for names in toolsets.TOOLSETS.values():
        assert bot_mode_dm.MESSAGE_AGENT_TOOL_NAME not in names


# ── dispatch gate (defense in depth) ─────────────────────────────────────────


def test_tool_refuses_outside_bot_chat(tmp_path):
    home = _managed_home(tmp_path)
    agent = _FakeAgent(home, title="Ordinary chat")
    result = json.loads(
        bot_mode_dm.message_agent_tool(target="researcher", message="hi", agent=agent)
    )
    assert "error" in result
    assert "Bot Chat" in result["error"]


def test_tool_refuses_on_unmanaged_install(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    agent = _FakeAgent(home, title="Bot Chat")
    result = json.loads(
        bot_mode_dm.message_agent_tool(target="researcher", message="hi", agent=agent)
    )
    assert "error" in result


# ── target validation ────────────────────────────────────────────────────────


def test_unknown_target_lists_roster(tmp_path):
    home = _managed_home(tmp_path, teammates=("researcher", "coder"))
    agent = _FakeAgent(home, title="Bot Chat")
    result = json.loads(
        bot_mode_dm.message_agent_tool(target="nosuchbot", message="hi", agent=agent)
    )
    assert "error" in result
    assert set(result["teammates"]) == {"researcher", "coder"}


def test_cannot_message_self(tmp_path):
    home = _managed_home(tmp_path)
    agent = _FakeAgent(home, title="Bot Chat")  # default profile
    result = json.loads(
        bot_mode_dm.message_agent_tool(target="hermes", message="hi", agent=agent)
    )
    assert "error" in result
    assert "yourself" in result["error"]


def test_empty_and_oversized_message_rejected(tmp_path):
    home = _managed_home(tmp_path)
    agent = _FakeAgent(home, title="Bot Chat")
    assert "error" in json.loads(
        bot_mode_dm.message_agent_tool(target="researcher", message="  ", agent=agent)
    )
    big = "x" * (bot_mode_dm.MESSAGE_MAX_CHARS + 1)
    assert "error" in json.loads(
        bot_mode_dm.message_agent_tool(target="researcher", message=big, agent=agent)
    )


def test_unregistered_peer_rejected(tmp_path):
    home = _managed_home(tmp_path, peers=("spark",))
    agent = _FakeAgent(home, title="Bot Chat")
    result = json.loads(
        bot_mode_dm.message_agent_tool(target="homelab/coder", message="hi", agent=agent)
    )
    assert "error" in result
    assert result["peers"] == ["spark"]


# ── delivery command shape ───────────────────────────────────────────────────


def _capture_spawn(monkeypatch):
    calls = []

    def fake_terminal_tool(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return json.dumps({"output": "Background process started", "session_id": "proc_test1234"})

    import tools.terminal_tool as terminal_tool_module

    monkeypatch.setattr(terminal_tool_module, "terminal_tool", fake_terminal_tool)
    return calls


def _runner_parts(command):
    """(mode, dm_file, transport argv) of a runner command. The optional ``--author <json>`` pair is skipped."""
    parts = shlex.split(command)
    marker = parts.index("--run-delivery")
    if parts[marker + 1] == "--author":
        marker += 2
    argv = parts[marker + 3 :]
    if argv[:1] == ["--profile-home"]:
        argv = argv[2:]
    return parts[marker + 1], parts[marker + 2], argv


def _runner_author(command):
    """The parsed ``--author`` payload of a runner command, or None when the pair is absent."""
    parts = shlex.split(command)
    marker = parts.index("--run-delivery")
    if parts[marker + 1] != "--author":
        return None
    return json.loads(parts[marker + 2])


def test_local_delivery_command_and_ack(tmp_path, monkeypatch):
    calls = _capture_spawn(monkeypatch)
    home = _managed_home(tmp_path, teammates=("researcher",))
    agent = _FakeAgent(home, title="Bot Chat")

    result = json.loads(
        bot_mode_dm.message_agent_tool(
            target="@researcher",
            message=(
                'status? give me the "PAYLOAD_SENTINEL_7A91" numbers '
                "$(and this is not shell)"
            ),
            agent=agent,
        )
    )
    assert result["status"] == "sent"
    assert result["to"] == "@researcher"
    assert result["process_id"] == "proc_test1234"
    assert "do NOT wait" in result["detail"]

    assert len(calls) == 1
    call = calls[0]
    assert call["background"] is True
    assert call["notify_on_complete"] is True
    assert call["_host_local"] is True
    assert Path(call["workdir"]) == Path(bot_mode_dm.__file__).resolve().parent.parent
    command = call["command"]
    mode, dm_file, transport_argv = _runner_parts(command)
    assert mode == "query-file"
    assert transport_argv == [
        "hermes",
        "-p",
        "researcher",
        "chat",
        "--in",
        "~",
        "-c",
        "Bot Chat",
        "--create-if-missing",
        "-Q",
    ]
    # message body rides the temp file, never the command line
    assert "PAYLOAD_SENTINEL_7A91" not in command
    assert "$(" not in command
    # the sender rides the runner argv as a stable id plus display handle
    assert _runner_author(command) == {"id": "bot:default", "name": "hermes", "is_bot": True}

    # attribution prefix applied server-side; body verbatim inside the file
    content = Path(dm_file).read_text(encoding="utf-8")
    assert content.startswith("Message from 🤖 hermes (@hermes): ")
    assert '$(and this is not shell)' in content


def test_peer_delivery_command_pins_registry_profile_for_secondary_bots(
    tmp_path, monkeypatch
):
    """A secondary-profile bot's peer DM must run in the registry-owning
    profile (#93935). `hermes peer` resolves bot_peers through
    profile-scoped load_config(); unpinned, the subprocess inherits the
    calling bot's profile and dies with "No peer named" even though the
    tool-side roster (read from the machine-root config) validated the
    target."""
    calls = _capture_spawn(monkeypatch)
    home = _managed_home(tmp_path, peers=("spark",))
    # A reviewer-profile gateway context: the agent's session db lives under
    # that profile's home, so _agent_home() resolves there while the
    # machine-root config (home/config.yaml) still holds the registry.
    reviewer_home = home / "profiles" / "reviewer"
    reviewer_home.mkdir(parents=True)
    agent = _FakeAgent(reviewer_home, title="Bot Chat")

    result = json.loads(
        bot_mode_dm.message_agent_tool(target="spark", message="ping", agent=agent)
    )
    assert result["status"] == "sent"
    mode, _dm_file, transport_argv = _runner_parts(calls[0]["command"])
    assert mode == "stdin"
    # The registry the tool validated against is the machine root's — the
    # default profile's home — so the CLI runs there, not in reviewer.
    assert transport_argv == ["hermes", "-p", "default", "peer", "dm", "spark"]


def test_peer_delivery_command(tmp_path, monkeypatch):
    calls = _capture_spawn(monkeypatch)
    monkeypatch.setattr("socket.gethostname", lambda: "eri-mac.local")
    home = _managed_home(tmp_path, peers=("spark",))
    agent = _FakeAgent(home, title="Bot Chat")

    result = json.loads(
        bot_mode_dm.message_agent_tool(target="spark/researcher", message="ping", agent=agent)
    )
    assert result["status"] == "sent"
    assert "spark" in result["to"]
    mode, _dm_file, transport_argv = _runner_parts(calls[0]["command"])
    assert mode == "stdin"
    assert transport_argv == ["hermes", "-p", "default", "peer", "dm", "spark/researcher"]
    # the peer child reads the author from its env and forwards it in the request body
    assert _runner_author(calls[0]["command"]) == {"id": "bot:eri-mac.local/default", "name": "hermes", "is_bot": True}

    # bare peer name targets the peer's main agent
    result2 = json.loads(
        bot_mode_dm.message_agent_tool(target="spark", message="ping", agent=agent)
    )
    assert result2["status"] == "sent"
    mode, _dm_file, transport_argv = _runner_parts(calls[1]["command"])
    assert mode == "stdin"
    assert transport_argv == ["hermes", "-p", "default", "peer", "dm", "spark"]


def test_peer_delivery_author_carries_the_sender_hostname_and_local_stays_bare(tmp_path, monkeypatch):
    """A peer dm crosses installs, so its author id is ``bot:<hostname>/<profile>``: the peer's own ``coder`` and a
    remote ``coder`` must not share one id. A teammate on this install still sees the bare ``bot:coder``."""
    calls = _capture_spawn(monkeypatch)
    monkeypatch.setattr("socket.gethostname", lambda: " eri/mac\x00.local ")
    home = _managed_home(tmp_path, teammates=("researcher", "coder"), peers=("spark",))
    agent = _FakeAgent(home / "profiles" / "coder", title="Bot Chat")

    assert json.loads(bot_mode_dm.message_agent_tool(target="spark", message="ping", agent=agent))["status"] == "sent"
    assert json.loads(bot_mode_dm.message_agent_tool(target="researcher", message="ping", agent=agent))["status"] == "sent"

    assert _runner_author(calls[0]["command"]) == {"id": "bot:erimac.local/coder", "name": "coder", "is_bot": True}
    assert _runner_author(calls[1]["command"]) == {"id": "bot:coder", "name": "coder", "is_bot": True}


def test_named_profile_sender_prefix(tmp_path, monkeypatch):
    """A named-profile bot signs with its own handle, not @hermes."""
    calls = _capture_spawn(monkeypatch)
    home = _managed_home(tmp_path, teammates=("researcher", "coder"))
    profile_home = home / "profiles" / "coder"
    agent = _FakeAgent(profile_home, title="Bot Chat")

    result = json.loads(
        bot_mode_dm.message_agent_tool(target="researcher", message="hi", agent=agent)
    )
    assert result["status"] == "sent"
    _mode, dm_file, _transport_argv = _runner_parts(calls[0]["command"])
    assert Path(dm_file).read_text(encoding="utf-8").startswith(
        "Message from 🤖 coder (@coder): "
    )
    assert _runner_author(calls[0]["command"]) == {"id": "bot:coder", "name": "coder", "is_bot": True}


def test_delivery_command_author_json_survives_quoting_and_windows_slash_rewrite(tmp_path, monkeypatch):
    """The author JSON sits between ``--run-delivery`` and the mode, survives shlex, and the Windows slash rewrite skips it."""
    author = {"id": "bot:default", "name": "hermes", "is_bot": True}
    command = bot_mode_dm._delivery_command(["hermes", "-p", "x"], str(tmp_path / "dm.txt"),
                                            stdin_file=False, author=author)
    parts = shlex.split(command)
    assert parts[2:4] == ["--run-delivery", "--author"]
    assert json.loads(parts[4]) == author
    assert parts[5] == "query-file"
    assert _runner_parts(command) == ("query-file", str(tmp_path / "dm.txt"), ["hermes", "-p", "x"])

    monkeypatch.setattr(sys, "platform", "win32")
    command = bot_mode_dm._delivery_command(["hermes", "-p", "x"], "C:\\Users\\me\\dm.txt",
                                            stdin_file=False, author={"id": "bot:default", "name": 'q"q', "is_bot": True})
    parts = shlex.split(command)
    assert json.loads(parts[4]) == {"id": "bot:default", "name": 'q"q', "is_bot": True}
    assert parts[6] == "C:/Users/me/dm.txt"

    # without an author the legacy shape is produced unchanged
    command = bot_mode_dm._delivery_command(["hermes"], "dm.txt", stdin_file=True)
    assert shlex.split(command)[2:4] == ["--run-delivery", "stdin"]
    assert _runner_author(command) is None


def test_spawn_failure_reports_error(tmp_path, monkeypatch):
    home = _managed_home(tmp_path)
    agent = _FakeAgent(home, title="Bot Chat")

    import tools.terminal_tool as terminal_tool_module

    def boom(command, **kwargs):
        raise RuntimeError("spawn failed")

    monkeypatch.setattr(terminal_tool_module, "terminal_tool", boom)
    result = json.loads(
        bot_mode_dm.message_agent_tool(target="researcher", message="hi", agent=agent)
    )
    assert "error" in result
    assert "could not be started" in result["error"]


def test_live_dm_admitted_before_waiter_failure(tmp_path, monkeypatch):
    from tools import bot_live_delivery as live

    home = _managed_home(tmp_path)
    target = home / "profiles" / "researcher"
    owner = dict(profile_home=str(target), session_id="bot", lease_id="lease", live_session_id="live")
    monkeypatch.setattr(live, "find_canonical_live_owner", lambda h: owner if Path(h) == target else None)
    monkeypatch.setattr(bot_mode_dm, "_dm_dir", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "wrong-home"))
    import tools.terminal_tool as terminal
    monkeypatch.setattr(terminal, "terminal_tool", lambda *a, **k: json.dumps({"error": "spawn failed"}))

    result = json.loads(bot_mode_dm.message_agent_tool("researcher", "hello", agent=_FakeAgent(home)))
    assert result["status"] == "queued"
    record = live.read_delivery_result(target, result["delivery_id"])
    assert record is not None
    assert record["owner"] == owner
    assert record["message"] == "Message from 🤖 hermes (@hermes): hello"
    assert record["author"] == {"id": "bot:default", "name": "hermes", "is_bot": True}
    assert "notification_error" in result


def test_live_dm_runner_retry_never_reexecutes_failed_claim(tmp_path, monkeypatch, capsys):
    from tools import bot_live_delivery as live

    home = _managed_home(tmp_path)
    target = home / "profiles" / "researcher"
    monkeypatch.setenv("HERMES_HOME", str(home))
    owner = dict(profile_home=str(target), session_id="bot", lease_id="lease", live_session_id="live")
    monkeypatch.setattr(live, "find_canonical_live_owner", lambda h: owner)
    monkeypatch.setattr(bot_mode_dm, "_LIVE_WAIT_SECONDS", 0)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("must not launch a model turn"))
    dm_file = tmp_path / "message.txt"
    dm_file.write_text("hello", encoding="utf-8")
    argv = ["hermes", "-p", "researcher"]
    assert bot_mode_dm._run_delivery(argv, str(dm_file), stdin_file=False) == 0
    queued = json.loads(capsys.readouterr().out)
    assert queued["status"] == "queued"
    claimed = live.claim_pending_delivery(target, owner)
    assert claimed is not None
    live.complete_delivery(target, claimed["delivery_id"], status="failed", error="HTTP 429 rate limit")
    monkeypatch.setattr(live, "find_canonical_live_owner", lambda h: None)
    assert bot_mode_dm._run_delivery(argv, str(dm_file), stdin_file=False) == 1
    failed = json.loads(capsys.readouterr().out)
    assert failed["status"] == "failed"
    assert failed["delivery_id"] == queued["delivery_id"]
    assert dm_file.read_text(encoding="utf-8") == "hello"


# ── plaintext tempfile lifecycle ─────────────────────────────────────────────


@pytest.mark.parametrize("stdin_file", [False, True])
def test_delivery_runner_keeps_file_for_child_then_unlinks(tmp_path, stdin_file):
    dm_file = tmp_path / "message with spaces.txt"
    dm_file.write_text("secret $(not shell)", encoding="utf-8")
    observed = tmp_path / "observed.txt"
    child = tmp_path / "child.py"
    child.write_text(
        textwrap.dedent(
            """\
            import pathlib
            import sys

            source = sys.stdin if sys.argv[1] == "-" else open(sys.argv[1], encoding="utf-8")
            with source:
                pathlib.Path(sys.argv[2]).write_text(source.read(), encoding="utf-8")
            """
        ),
        encoding="utf-8",
    )
    source_arg = "-" if stdin_file else str(dm_file)

    returncode = bot_mode_dm._run_delivery(
        [sys.executable, str(child), source_arg, str(observed)],
        str(dm_file),
        stdin_file=stdin_file,
    )

    assert returncode == 0
    assert observed.read_text(encoding="utf-8") == "secret $(not shell)"
    assert not dm_file.exists()


def test_delivery_runner_unlinks_when_child_launch_raises(tmp_path, monkeypatch):
    dm_file = tmp_path / "message.txt"
    dm_file.write_text("secret", encoding="utf-8")

    def boom(*args, **kwargs):
        raise RuntimeError("child launch failed")

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(RuntimeError, match="child launch failed"):
        bot_mode_dm._run_delivery(["hermes"], str(dm_file), stdin_file=False)
    assert not dm_file.exists()


def test_delivery_runner_preserves_child_failure_and_unlinks(tmp_path):
    dm_file = tmp_path / "message.txt"
    dm_file.write_text("secret", encoding="utf-8")
    child = tmp_path / "fail.py"
    child.write_text(
        "import pathlib, sys\n"
        "assert pathlib.Path(sys.argv[-1]).read_text(encoding='utf-8') == 'secret'\n"
        "raise SystemExit(7)\n",
        encoding="utf-8",
    )

    returncode = bot_mode_dm._run_delivery(
        [sys.executable, str(child)], str(dm_file), stdin_file=False
    )

    assert returncode == 7
    assert not dm_file.exists()


def test_delivery_runner_surfaces_live_owner_refusal(tmp_path, capsys):
    """#100523: the CLI's single-owner lease refusal is a delivery FAILURE the
    sender can read, not a raw exit-1 with the payload silently gone."""
    dm_file = tmp_path / "message.txt"
    dm_file.write_text("hi", encoding="utf-8")
    child = tmp_path / "owned.py"
    child.write_text(
        "import sys\n"
        "print('Session abc already has a live owner (desktop, pid 1).', file=sys.stderr)\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )

    returncode = bot_mode_dm._run_delivery(
        [sys.executable, str(child), "-p", "ops"], str(dm_file), stdin_file=False
    )

    assert returncode == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["reason"] == "target_busy"
    assert "NOT delivered" in payload["error"]


def test_query_file_delivery_closes_stdin_for_initial_attempt_and_retry(
    tmp_path, monkeypatch
):
    dm_file = tmp_path / "message.txt"
    dm_file.write_text("secret", encoding="utf-8")
    calls = []
    responses = [
        subprocess.CompletedProcess([], 1, stdout="", stderr="HTTP 429 rate limit"),
        subprocess.CompletedProcess([], 0, stdout="", stderr=""),
    ]

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return responses.pop(0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    returncode = bot_mode_dm._run_delivery(
        ["hermes", "-p", "researcher"], str(dm_file), stdin_file=False
    )

    assert returncode == 0
    assert len(calls) == 2
    assert [kwargs["stdin"] for _argv, kwargs in calls] == [
        subprocess.DEVNULL,
        subprocess.DEVNULL,
    ]
    assert not dm_file.exists()


@pytest.mark.parametrize("mode, author", [
    ("stdin", {"id": "bot:eri-mac.local/coder", "name": "coder", "is_bot": True}),
    ("query-file", {"id": "bot:coder", "name": "coder", "is_bot": True}),
    ("query-file", None),
], ids=["stdin", "query-file", "no author"])
def test_delivery_main_child_env_carries_only_the_argv_author(tmp_path, monkeypatch, mode, author):
    """The ``--author`` payload becomes HERMES_TURN_AUTHOR on the child. Without it the runner drops the
    variable it inherited from the sending bot's own turn instead of passing it on as the recipient's author."""
    from agent.turn_author import TURN_AUTHOR_ENV

    dm_file = tmp_path / "message.txt"
    dm_file.write_text("secret", encoding="utf-8")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("HERMES_DM_TEST_MARKER", "kept")
    monkeypatch.setenv(TURN_AUTHOR_ENV, json.dumps({"id": "bot:previous", "name": "previous", "is_bot": True}))
    author_args = ["--author", json.dumps(author)] if author else []

    returncode = bot_mode_dm._delivery_main(
        ["--run-delivery", *author_args, mode, str(dm_file), "hermes", "-p", "researcher"])

    assert returncode == 0
    [(argv, kwargs)] = calls
    assert argv[:3] == ["hermes", "-p", "researcher"]
    assert kwargs["env"]["HERMES_DM_TEST_MARKER"] == "kept"
    assert (json.loads(kwargs["env"][TURN_AUTHOR_ENV]) if TURN_AUTHOR_ENV in kwargs["env"] else None) == author
    assert not dm_file.exists()


def test_real_delivery_command_round_trip_carries_author(tmp_path):
    """Through a real subprocess, the runner argv built by ``_delivery_command`` sets HERMES_TURN_AUTHOR on the child."""
    dm_file = tmp_path / "message.txt"
    dm_file.write_text("secret", encoding="utf-8")
    observed = tmp_path / "observed.txt"
    child = tmp_path / "child.py"
    child.write_text(
        "import os, pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(os.environ.get('HERMES_TURN_AUTHOR', 'unset'), encoding='utf-8')\n",
        encoding="utf-8",
    )
    author = {"id": "bot:default", "name": "hermes", "is_bot": True}
    command = bot_mode_dm._delivery_command(
        [sys.executable, str(child), str(observed)], str(dm_file), stdin_file=False, author=author
    )

    result = subprocess.run(shlex.split(command), check=False)

    assert result.returncode == 0
    assert json.loads(observed.read_text(encoding="utf-8")) == author
    assert not dm_file.exists()


@pytest.mark.parametrize("mode", ["stdin", "query-file"])
def test_delivery_main_runs_valid_cli_and_unlinks(tmp_path, mode):
    dm_file = tmp_path / "message.txt"
    dm_file.write_text("secret", encoding="utf-8")
    observed = tmp_path / "observed.txt"
    child = tmp_path / "child.py"
    child.write_text(
        "import pathlib, sys\n"
        "source = sys.stdin if sys.argv[1] == '-' else open(sys.argv[1], encoding='utf-8')\n"
        "with source:\n"
        "    pathlib.Path(sys.argv[2]).write_text(source.read(), encoding='utf-8')\n",
        encoding="utf-8",
    )
    source_arg = "-" if mode == "stdin" else str(dm_file)

    returncode = bot_mode_dm._delivery_main(
        [
            "--run-delivery",
            mode,
            str(dm_file),
            sys.executable,
            str(child),
            source_arg,
            str(observed),
        ]
    )

    assert returncode == 0
    assert observed.read_text(encoding="utf-8") == "secret"
    assert not dm_file.exists()


def test_delivery_main_maps_launch_exception_to_one_and_unlinks(tmp_path, monkeypatch):
    dm_file = tmp_path / "message.txt"
    dm_file.write_text("secret", encoding="utf-8")

    def boom(*args, **kwargs):
        raise RuntimeError("child launch failed")

    monkeypatch.setattr(subprocess, "run", boom)
    assert (
        bot_mode_dm._delivery_main(
            ["--run-delivery", "query-file", str(dm_file), "missing-transport"]
        )
        == 1
    )
    assert not dm_file.exists()


@pytest.mark.parametrize("stdin_file", [False, True])
def test_real_delivery_command_round_trip(tmp_path, stdin_file):
    dm_file = tmp_path / "message with spaces.txt"
    dm_file.write_text("secret λ\nsecond line", encoding="utf-8")
    observed = tmp_path / "observed with spaces.txt"
    child = tmp_path / "child with spaces.py"
    child.write_text(
        "import pathlib, sys\n"
        "source = sys.stdin if sys.argv[1] == '-' else open(sys.argv[1], encoding='utf-8')\n"
        "with source:\n"
        "    pathlib.Path(sys.argv[2]).write_text(source.read(), encoding='utf-8')\n",
        encoding="utf-8",
    )
    source_arg = "-" if stdin_file else str(dm_file)
    command = bot_mode_dm._delivery_command(
        [sys.executable, str(child), source_arg, str(observed)],
        str(dm_file),
        stdin_file=stdin_file,
    )

    result = subprocess.run(shlex.split(command), check=False)

    assert result.returncode == 0
    assert observed.read_text(encoding="utf-8") == "secret λ\nsecond line"
    assert not dm_file.exists()


@pytest.mark.windows_only
def test_delivery_command_round_trip_through_windows_local_shell(tmp_path):
    """Native runner paths must survive the Git Bash process boundary."""
    from tools.environments.local import _find_shell

    dm_file = tmp_path / "message with spaces.txt"
    dm_file.write_text("secret", encoding="utf-8")
    observed = tmp_path / "observed with spaces.txt"
    child = tmp_path / "child with spaces.py"
    child.write_text(
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text('started', encoding='utf-8')\n",
        encoding="utf-8",
    )
    command = bot_mode_dm._delivery_command(
        [sys.executable, str(child), str(observed)],
        str(dm_file),
        stdin_file=False,
    )

    result = subprocess.run(
        [_find_shell(), "-lic", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert observed.read_text(encoding="utf-8") == "started"
    assert not dm_file.exists()


@pytest.mark.parametrize(
    ("terminal_result", "raises"),
    [
        (json.dumps({"error": "rejected"}), False),
        ("not json", False),
        (None, True),
    ],
)
def test_spawn_failure_unlinks_untransferred_file(
    tmp_path, monkeypatch, terminal_result, raises
):
    dm_file = tmp_path / "message.txt"
    dm_file.write_text("secret", encoding="utf-8")

    import tools.terminal_tool as terminal_tool_module

    def fail_spawn(command, **kwargs):
        assert dm_file.exists()
        if raises:
            raise RuntimeError("spawn failed")
        return terminal_result

    monkeypatch.setattr(terminal_tool_module, "terminal_tool", fail_spawn)
    result = json.loads(
        bot_mode_dm._spawn_delivery(
            "unused", "@researcher", dm_file=str(dm_file), task_id=None, agent=None
        )
    )

    assert "error" in result
    assert not dm_file.exists()


def test_successful_spawn_transfers_cleanup_to_runner(tmp_path, monkeypatch):
    dm_file = tmp_path / "message.txt"
    dm_file.write_text("secret", encoding="utf-8")

    import tools.terminal_tool as terminal_tool_module

    def launched(command, **kwargs):
        assert dm_file.exists()
        return json.dumps({"session_id": "proc_test1234"})

    monkeypatch.setattr(terminal_tool_module, "terminal_tool", launched)
    result = json.loads(
        bot_mode_dm._spawn_delivery(
            "unused", "@researcher", dm_file=str(dm_file), task_id=None, agent=None
        )
    )

    assert result["status"] == "sent"
    assert dm_file.exists(), "the parent must not delete before the background runner reads"


def test_write_dm_file_unlinks_partial_file_on_write_exception(tmp_path, monkeypatch):
    tmp_path / "partial.txt"
    real_mkstemp = bot_mode_dm.tempfile.mkstemp

    def fixed_mkstemp(**kwargs):
        kwargs["dir"] = tmp_path
        return real_mkstemp(**kwargs)

    class BrokenWriter:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def write(self, content):
            raise OSError("disk full")

    monkeypatch.setattr(bot_mode_dm.tempfile, "mkstemp", fixed_mkstemp)
    monkeypatch.setattr(bot_mode_dm.os, "fdopen", lambda *args, **kwargs: BrokenWriter())

    with pytest.raises(OSError, match="disk full"):
        bot_mode_dm._write_dm_file("secret")
    assert list(tmp_path.glob("dm-*.txt")) == []


def test_sweeper_removes_only_stale_dm_files(tmp_path, monkeypatch):
    monkeypatch.setattr(bot_mode_dm.tempfile, "gettempdir", lambda: str(tmp_path))
    dm_dir = bot_mode_dm._dm_dir()
    legacy_stale = tmp_path / "hermes-dm-stale.txt"
    stale = dm_dir / "dm-stale.txt"
    fresh = dm_dir / "dm-fresh.txt"
    unrelated = tmp_path / "other.txt"
    for path in (legacy_stale, stale, fresh, unrelated):
        path.write_text("secret", encoding="utf-8")
    now = time.time()
    old = now - bot_mode_dm._DM_STALE_SECONDS - 1
    os.utime(legacy_stale, (old, old))
    os.utime(stale, (old, old))
    bot_mode_dm.cleanup_bot_dm_cache(now=now)

    assert not legacy_stale.exists()
    assert not stale.exists()
    assert fresh.exists()
    assert unrelated.exists()


def test_dm_dir_is_private_and_uid_scoped_on_posix(tmp_path, monkeypatch):
    monkeypatch.setattr(bot_mode_dm.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(bot_mode_dm, "_dm_fallback_dir", lambda: tmp_path / "fallback")

    dm_dir = bot_mode_dm._dm_dir()

    if hasattr(os, "getuid"):
        assert dm_dir.name == f"{bot_mode_dm._DM_DIR_NAME}-{os.getuid()}"
        # POSIX contract only: a Windows directory always reports 0o777 and chmod() there
        # toggles just the read-only bit, so the directory ACL is what protects it.
        assert dm_dir.stat().st_mode & 0o777 == 0o700
    else:
        assert dm_dir.name == bot_mode_dm._DM_DIR_NAME


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX mode contract")
def test_dm_dir_repairs_restrictive_owner_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(bot_mode_dm.tempfile, "gettempdir", lambda: str(tmp_path))
    uid = os.getuid() if hasattr(os, "getuid") else None
    dirname = f"{bot_mode_dm._DM_DIR_NAME}-{uid}" if uid is not None else bot_mode_dm._DM_DIR_NAME
    dm_dir = tmp_path / dirname
    dm_dir.mkdir(mode=0o500)
    dm_dir.chmod(0o500)

    assert bot_mode_dm._dm_dir() == dm_dir
    assert dm_dir.stat().st_mode & 0o777 == 0o700


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX ownership contract")
def test_dm_dir_rejects_precreated_symlink(tmp_path, monkeypatch):
    target = tmp_path / "attacker-controlled"
    target.mkdir()
    expected = tmp_path / f"{bot_mode_dm._DM_DIR_NAME}-{os.getuid()}"
    expected.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(bot_mode_dm.tempfile, "gettempdir", lambda: str(tmp_path))

    with pytest.raises(PermissionError, match="not a directory"):
        bot_mode_dm._dm_dir()


# --- Unusable temp root: degrade to the private fallback (the reported WinError 5) ---


def _deny_canonical(monkeypatch, canonical: Path):
    """Make only the canonical directory's writability probe fail, WinError 5 shaped."""
    def probe(path):
        if Path(path) == canonical:
            raise PermissionError(13, "Acces refuse", str(path), 5)
    monkeypatch.setattr(bot_mode_dm, "_probe_writable", probe)


def test_dm_dir_falls_back_when_canonical_path_is_not_usable(tmp_path, monkeypatch):
    """A canonical path that cannot become a directory must degrade, not fail the send."""
    monkeypatch.setattr(bot_mode_dm.tempfile, "gettempdir", lambda: str(tmp_path))
    fallback = tmp_path / "home" / "cache" / "dm-tmp"
    monkeypatch.setattr(bot_mode_dm, "_dm_fallback_dir", lambda: fallback)
    (tmp_path / bot_mode_dm._DM_DIR_NAME).write_text("not a directory", encoding="utf-8")

    assert bot_mode_dm._dm_dir() == fallback
    assert fallback.is_dir()
    assert bot_mode_dm._dm_dir() == fallback  # idempotent: one stable fallback, not churn


def test_dm_dir_falls_back_when_write_probe_is_denied(tmp_path, monkeypatch):
    """[WinError 5]: the canonical dir exists but denies creating a file in it."""
    monkeypatch.setattr(bot_mode_dm.tempfile, "gettempdir", lambda: str(tmp_path))
    fallback = tmp_path / "home" / "cache" / "dm-tmp"
    monkeypatch.setattr(bot_mode_dm, "_dm_fallback_dir", lambda: fallback)
    _deny_canonical(monkeypatch, tmp_path / bot_mode_dm._DM_DIR_NAME)

    assert bot_mode_dm._dm_dir() == fallback


def test_write_dm_file_uses_the_fallback_when_temp_root_is_denied(tmp_path, monkeypatch):
    """Regression: a stale ACL on the temp dir must not disable agent-to-agent DMs."""
    monkeypatch.setattr(bot_mode_dm.tempfile, "gettempdir", lambda: str(tmp_path))
    fallback = tmp_path / "home" / "cache" / "dm-tmp"
    monkeypatch.setattr(bot_mode_dm, "_dm_fallback_dir", lambda: fallback)
    _deny_canonical(monkeypatch, tmp_path / bot_mode_dm._DM_DIR_NAME)

    payload = bot_mode_dm._write_dm_file("secret payload")

    assert Path(payload).parent == fallback
    assert Path(payload).read_text(encoding="utf-8") == "secret payload"


def test_dm_dir_fails_closed_when_no_location_is_usable(tmp_path, monkeypatch):
    """Both locations denied: a named error, never a silent write to an arbitrary path."""
    monkeypatch.setattr(bot_mode_dm.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(bot_mode_dm, "_dm_fallback_dir", lambda: tmp_path / "home" / "cache" / "dm-tmp")

    def probe(path):
        raise PermissionError(13, "Acces refuse", str(path), 5)

    monkeypatch.setattr(bot_mode_dm, "_probe_writable", probe)

    with pytest.raises(PermissionError, match="no usable DM directory"):
        bot_mode_dm._dm_dir()


def test_sweeper_covers_the_fallback_directory(tmp_path, monkeypatch):
    """Fallback payloads expire too: a later process with a healthy temp root would
    otherwise never sweep them, breaking the stale-payload bound."""
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    fallback = tmp_path / "home" / "cache" / "dm-tmp"
    fallback.mkdir(parents=True)
    monkeypatch.setattr(bot_mode_dm.tempfile, "gettempdir", lambda: str(temp_root))
    monkeypatch.setattr(bot_mode_dm, "_dm_fallback_dir", lambda: fallback)

    stale = fallback / "dm-stale.txt"
    fresh = fallback / "dm-fresh.txt"
    for path in (stale, fresh):
        path.write_text("secret", encoding="utf-8")
    now = time.time()
    old = now - bot_mode_dm._DM_STALE_SECONDS - 1
    os.utime(stale, (old, old))

    bot_mode_dm.cleanup_bot_dm_cache(now=now)

    assert not stale.exists()
    assert fresh.exists()
    assert fallback.exists()  # sweeping must not create or remove directories


def test_probe_leaves_no_file_behind(tmp_path):
    """The writability probe must be invisible: no residue in the DM directory."""
    target = tmp_path / "probe-target"  # dedicated dir: pytest's tmp_path also holds the
    target.mkdir()                      # suite's isolated HERMES_HOME
    bot_mode_dm._probe_writable(target)

    assert list(target.iterdir()) == []

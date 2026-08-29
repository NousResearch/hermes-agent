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
    parts = shlex.split(command)
    marker = parts.index("--run-delivery")
    return parts[marker + 1], parts[marker + 2], parts[marker + 3 :]


def test_local_delivery_command_and_ack(tmp_path, monkeypatch):
    calls = _capture_spawn(monkeypatch)
    home = _managed_home(tmp_path, teammates=("researcher",))
    agent = _FakeAgent(home, title="Bot Chat")

    result = json.loads(
        bot_mode_dm.message_agent_tool(
            target="@researcher",
            message='status? give me the "final" numbers $(and this is not shell)',
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
    assert "final" not in command
    assert "$(" not in command

    # attribution prefix applied server-side; body verbatim inside the file
    content = Path(dm_file).read_text(encoding="utf-8")
    assert content.startswith("Message from 🤖 hermes (@hermes): ")
    assert '$(and this is not shell)' in content


def test_peer_delivery_command(tmp_path, monkeypatch):
    calls = _capture_spawn(monkeypatch)
    home = _managed_home(tmp_path, peers=("spark",))
    agent = _FakeAgent(home, title="Bot Chat")

    result = json.loads(
        bot_mode_dm.message_agent_tool(target="spark/researcher", message="ping", agent=agent)
    )
    assert result["status"] == "sent"
    assert "spark" in result["to"]
    mode, _dm_file, transport_argv = _runner_parts(calls[0]["command"])
    assert mode == "stdin"
    assert transport_argv == ["hermes", "peer", "dm", "spark/researcher"]

    # bare peer name targets the peer's main agent
    result2 = json.loads(
        bot_mode_dm.message_agent_tool(target="spark", message="ping", agent=agent)
    )
    assert result2["status"] == "sent"
    mode, _dm_file, transport_argv = _runner_parts(calls[1]["command"])
    assert mode == "stdin"
    assert transport_argv == ["hermes", "peer", "dm", "spark"]


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


@pytest.mark.parametrize("args", [[], ["--run-delivery"], ["--run-delivery", "bad", "x"]])
def test_delivery_main_rejects_invalid_cli(args):
    assert bot_mode_dm._delivery_main(args) == 2


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
    dm_file = tmp_path / "partial.txt"
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
    bot_mode_dm._sweep_stale_dm_files(now=now)

    assert not legacy_stale.exists()
    assert not stale.exists()
    assert fresh.exists()
    assert unrelated.exists()


def test_dm_dir_is_private_and_uid_scoped_on_posix(tmp_path, monkeypatch):
    monkeypatch.setattr(bot_mode_dm.tempfile, "gettempdir", lambda: str(tmp_path))

    dm_dir = bot_mode_dm._dm_dir()

    if hasattr(os, "getuid"):
        assert dm_dir.name == f"{bot_mode_dm._DM_DIR_NAME}-{os.getuid()}"
    else:
        assert dm_dir.name == bot_mode_dm._DM_DIR_NAME
    assert dm_dir.stat().st_mode & 0o777 == 0o700


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

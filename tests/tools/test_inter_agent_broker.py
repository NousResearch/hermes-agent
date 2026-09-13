import json
import os
import stat
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from tools.comms import broker
from tools import inter_agent_tool
from tools.registry import registry


def _request(base_url, path, token, *, data=None):
    body = None if data is None else json.dumps(data).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            **({"Content-Type": "application/json"} if body else {}),
        },
        method="POST" if body else "GET",
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        return response.status, json.loads(response.read())


def _start_server(monkeypatch, tmp_path):
    monkeypatch.setattr(broker, "get_default_hermes_root", lambda: tmp_path)
    monkeypatch.setattr(inter_agent_tool, "get_default_hermes_root", lambda: tmp_path)
    token = "test-broker-token"
    (tmp_path / "inter-agent-broker.token").write_text(token, encoding="utf-8")
    server = broker._BrokerServer((broker.HOST, broker.PORT), broker._Handler)
    server.broker_id = "expected-broker"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, token, f"http://{broker.HOST}:{server.server_port}"


def test_health_requires_auth_and_matches_endpoint_identity(monkeypatch, tmp_path):
    server, _thread, token, base_url = _start_server(monkeypatch, tmp_path)
    try:
        request = urllib.request.Request(f"{base_url}/health")
        try:
            urllib.request.urlopen(request, timeout=2)
        except urllib.error.HTTPError as error:
            assert error.code == 401
        else:
            raise AssertionError("health must require broker authentication")

        endpoint = {"port": server.server_port, "broker_id": server.broker_id}
        (tmp_path / "inter-agent-broker.json").write_text(
            json.dumps(endpoint), encoding="utf-8"
        )
        assert inter_agent_tool._broker_is_ready()

        endpoint["broker_id"] = "stale-broker"
        (tmp_path / "inter-agent-broker.json").write_text(
            json.dumps(endpoint), encoding="utf-8"
        )
        assert not inter_agent_tool._broker_is_ready()
    finally:
        server.shutdown()
        server.server_close()


def test_broker_readiness_rejects_an_old_protocol(monkeypatch, tmp_path):
    server, _thread, _token, _base_url = _start_server(monkeypatch, tmp_path)
    try:
        (tmp_path / "inter-agent-broker.json").write_text(
            json.dumps({"port": server.server_port, "broker_id": server.broker_id}),
            encoding="utf-8",
        )
        monkeypatch.setattr(inter_agent_tool, "BROKER_PROTOCOL_VERSION", 2)
        assert not inter_agent_tool._broker_is_ready()
    finally:
        server.shutdown()
        server.server_close()


def test_ensure_broker_retires_rejected_owned_process(monkeypatch, tmp_path):
    monkeypatch.setattr(inter_agent_tool, "get_default_hermes_root", lambda: tmp_path)
    (tmp_path / "inter-agent-broker.token").write_text(
        "test-broker-token", encoding="utf-8"
    )
    (tmp_path / "inter-agent-broker.json").write_text(
        json.dumps(
            {
                "port": 12345,
                "broker_id": "old-broker",
                "protocol_version": 0,
                "pid": 4242,
            }
        ),
        encoding="utf-8",
    )
    retired = []

    class Process:
        def cmdline(self):
            return ["python", "-m", "tools.comms.broker"]

        def terminate(self):
            retired.append("terminate")

        def wait(self, timeout=None):
            retired.append(("wait", timeout))

    monkeypatch.setattr(inter_agent_tool.psutil, "Process", lambda _pid: Process())
    readiness = iter((False, True))
    monkeypatch.setattr(inter_agent_tool, "_broker_is_ready", lambda: next(readiness))
    monkeypatch.setattr(
        inter_agent_tool.subprocess, "Popen", lambda *_a, **_k: object()
    )

    inter_agent_tool._ensure_broker()

    assert retired == ["terminate", ("wait", 0.5)]


def test_empty_broker_token_file_is_repaired_atomically(monkeypatch, tmp_path):
    monkeypatch.setattr(inter_agent_tool, "get_default_hermes_root", lambda: tmp_path)
    path = tmp_path / "inter-agent-broker.token"
    path.write_text("\n", encoding="utf-8")

    token = inter_agent_tool._broker_token()

    assert token
    assert path.read_text(encoding="utf-8") == token


def test_receive_waits_for_a_message(monkeypatch, tmp_path):
    monkeypatch.setattr(broker, "RECEIVE_WAIT_SECONDS", 0.5)
    monkeypatch.setattr(broker, "RECEIVE_POLL_INTERVAL_SECONDS", 0.01)
    server, thread, token, base_url = _start_server(monkeypatch, tmp_path)
    result = []
    try:
        def receive():
            result.append(_request(base_url, "/receive?to=recipient", token))

        receiver = threading.Thread(target=receive)
        receiver.start()
        time.sleep(0.05)
        status, sent = _request(
            base_url,
            "/send",
            token,
            data={"from": "sender", "to": "recipient", "body": "hello"},
        )
        assert status == 200
        receiver.join(timeout=1)
        assert not receiver.is_alive()
        assert result == [
            (
                200,
                [
                    {
                        "id": sent["id"],
                        "from": "sender",
                        "to": "recipient",
                        "body": "hello",
                        "created_at": result[0][1][0]["created_at"],
                    }
                ],
            )
        ]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_registry_dispatch_uses_replyable_profile_address_for_round_trip(
    monkeypatch, tmp_path
):
    installation = tmp_path / "installation"
    profile_a = installation / "profiles" / "agent-a"
    profile_b = installation / "profiles" / "agent-b"
    profile_a.mkdir(parents=True)
    profile_b.mkdir(parents=True)
    server, thread, _token, _base_url = _start_server(monkeypatch, installation)
    (installation / "inter-agent-broker.json").write_text(
        json.dumps({"port": server.server_port, "broker_id": server.broker_id}),
        encoding="utf-8",
    )
    try:
        monkeypatch.setenv("HERMES_HOME", str(profile_a))
        monkeypatch.setenv("HERMES_PROFILE", "agent-a")
        sent = json.loads(
            registry.dispatch(
                "inter_agent",
                {"action": "send", "to": "agent-b", "body": "hello"},
                session_id="opaque-session-a",
            )
        )
        assert sent["ok"] is True
        assert (installation / "inter-agent-messages.db").exists()
        assert not (profile_a / "inter-agent-messages.db").exists()
        assert not (profile_b / "inter-agent-messages.db").exists()

        received = json.loads(
            registry.dispatch(
                "inter_agent", {"action": "receive", "to": "agent-b"}
            )
        )
        assert received[0]["from"] == "agent-a"

        monkeypatch.setenv("HERMES_HOME", str(profile_b))
        monkeypatch.setenv("HERMES_PROFILE", "agent-b")
        json.loads(
            registry.dispatch(
                "inter_agent",
                {"action": "send", "to": received[0]["from"], "body": "reply"},
                session_id="opaque-session-b",
            )
        )

        monkeypatch.setenv("HERMES_HOME", str(profile_a))
        monkeypatch.setenv("HERMES_PROFILE", "agent-a")
        reply = json.loads(
            registry.dispatch("inter_agent", {"action": "receive"})
        )
        assert reply[0]["from"] == "agent-b"
        assert reply[0]["to"] == "agent-a"
        assert reply[0]["body"] == "reply"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_broker_uses_platform_detach_helper(monkeypatch):
    calls = []
    readiness = iter((False, True))
    monkeypatch.setattr(inter_agent_tool, "_broker_is_ready", lambda: next(readiness))
    monkeypatch.setattr(
        inter_agent_tool,
        "windows_detach_popen_kwargs",
        lambda: {"creationflags": 0x08000200},
    )
    monkeypatch.setattr(
        inter_agent_tool.subprocess,
        "Popen",
        lambda argv, **kwargs: calls.append((argv, kwargs)),
    )

    inter_agent_tool._ensure_broker()

    assert calls == [
        (
            [inter_agent_tool.sys.executable, "-P", "-m", "tools.comms.broker"],
            {
                "stdin": inter_agent_tool.subprocess.DEVNULL,
                "stdout": inter_agent_tool.subprocess.DEVNULL,
                "stderr": inter_agent_tool.subprocess.DEVNULL,
                "env": inter_agent_tool._broker_process_env(),
                "creationflags": 0x08000200,
            },
        )
    ]


@pytest.mark.windows_only
def test_broker_retries_without_breakaway_when_job_rejects_spawn(monkeypatch):
    calls = []
    readiness = iter((False, False, True))

    monkeypatch.setattr(inter_agent_tool, "_broker_is_ready", lambda: next(readiness))

    def fake_popen(argv, **kwargs):
        calls.append((argv, kwargs))
        if len(calls) == 1:
            raise OSError(5, "Access is denied")
        return object()

    monkeypatch.setattr(inter_agent_tool.subprocess, "Popen", fake_popen)

    inter_agent_tool._ensure_broker()

    assert len(calls) == 2
    (argv1, kwargs1), (argv2, kwargs2) = calls
    assert argv1 == argv2 == [
        inter_agent_tool.sys.executable,
        "-P",
        "-m",
        "tools.comms.broker",
    ]
    assert kwargs1["env"] == kwargs2["env"] == inter_agent_tool._broker_process_env()
    assert kwargs1["creationflags"] == inter_agent_tool.windows_detach_popen_kwargs()[
        "creationflags"
    ]
    assert kwargs2["creationflags"] == inter_agent_tool.windows_detach_flags_without_breakaway()


def test_concurrent_first_use_starts_only_one_broker(monkeypatch):
    calls = []
    first_started = threading.Event()
    release_first = threading.Event()

    def fake_ready():
        return release_first.is_set()

    def fake_popen(argv, **kwargs):
        calls.append((argv, kwargs))
        if len(calls) == 1:
            first_started.set()
            assert release_first.wait(timeout=2)
        return object()

    monkeypatch.setattr(inter_agent_tool, "_broker_is_ready", fake_ready)
    monkeypatch.setattr(inter_agent_tool.subprocess, "Popen", fake_popen)

    workers = [threading.Thread(target=inter_agent_tool._ensure_broker) for _ in range(2)]
    workers[0].start()
    assert first_started.wait(timeout=2)
    workers[1].start()
    release_first.set()
    for worker in workers:
        worker.join(timeout=2)
        assert not worker.is_alive()

    assert len(calls) == 1


def test_broker_shutdown_does_not_remove_replaced_endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(broker, "get_default_hermes_root", lambda: tmp_path)
    endpoint = tmp_path / "inter-agent-broker.json"
    endpoint.write_text(
        json.dumps({"port": 1234, "broker_id": "replacement"}), encoding="utf-8"
    )

    broker._remove_endpoint_if_owned(endpoint, "original")

    assert endpoint.exists()


def test_message_database_and_sidecars_are_owner_only(monkeypatch, tmp_path):
    monkeypatch.setattr(broker, "get_default_hermes_root", lambda: tmp_path)
    path = tmp_path / "inter-agent-messages.db"
    for suffix in ("-journal", "-wal", "-shm"):
        sidecar = Path(f"{path}{suffix}")
        fd = os.open(sidecar, os.O_WRONLY | os.O_CREAT, 0o644)
        os.close(fd)
    connection = broker._database()
    connection.close()

    candidates = (
        path,
        *(Path(f"{path}{suffix}") for suffix in ("-journal", "-wal", "-shm")),
    )
    for candidate in candidates:
        if candidate.exists():
            assert stat.S_IMODE(candidate.stat().st_mode) == 0o600


@pytest.mark.windows_only
def test_broker_state_removes_inherited_windows_acl(tmp_path):
    path = tmp_path / "inter-agent-broker.token"
    path.write_text("secret", encoding="utf-8")

    broker._secure_state_permissions(path)

    acl = subprocess.run(
        ["icacls", str(path)], capture_output=True, text=True, check=True
    ).stdout
    assert "(I)" not in acl


@pytest.mark.windows_only
def test_broker_write_sidecars_have_private_acl(monkeypatch, tmp_path):
    monkeypatch.setattr(broker, "get_default_hermes_root", lambda: tmp_path)
    database = tmp_path / "inter-agent-messages.db"
    connection = broker._database()
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            "INSERT INTO messages(sender, recipient, body) VALUES (?, ?, ?)",
            ("sender", "recipient", "body"),
        )
        broker._commit_database(connection, database)
        for candidate in (database, Path(f"{database}-wal"), Path(f"{database}-shm")):
            if candidate.exists():
                acl = subprocess.run(
                    ["icacls", str(candidate)],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
                assert "BUILTIN\\Users" not in acl
                assert "Everyone" not in acl
                assert "Authenticated Users" not in acl
    finally:
        connection.close()


def test_history_query_has_sender_and_recipient_indexes(monkeypatch, tmp_path):
    monkeypatch.setattr(broker, "get_default_hermes_root", lambda: tmp_path)
    connection = broker._database()
    try:
        details = {
            row[3]
            for row in connection.execute(
                "EXPLAIN QUERY PLAN SELECT id FROM messages "
                "WHERE sender = ? OR recipient = ? ORDER BY id DESC LIMIT ?",
                ("peer", "peer", 100),
            )
        }
    finally:
        connection.close()

    assert any("messages_sender_id_idx" in detail for detail in details)
    assert any("messages_recipient_id_idx" in detail for detail in details)

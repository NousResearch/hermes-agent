from __future__ import annotations

import io
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import pytest

from gateway.session_context import reset_session_vars, set_session_vars
from tools import threadwire_telegram_worker_tool as worker


class FakeProcess:
    def __init__(self, returncode: int | None = 0):
        self.pid = 4242
        self._returncode = returncode
        self.stdin = io.StringIO()
        self.signals = []
        self.killed = False
        self.wait_timeouts = []

    def poll(self):
        return self._returncode

    def send_signal(self, sig):
        self.signals.append(sig)

    def kill(self):
        self.killed = True
        self._returncode = -9

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        return self._returncode


@pytest.fixture(autouse=True)
def clean_session_context():
    reset_session_vars()
    yield
    reset_session_vars()


@pytest.fixture
def available_threadwire(monkeypatch):
    monkeypatch.setattr(worker, "check_threadwire_requirements", lambda: True)


def _bind(*, platform="telegram", chat_id="123", thread_id=""):
    set_session_vars(platform=platform, chat_id=chat_id, thread_id=thread_id)


def _capture_spawn(monkeypatch, *, returncode=0):
    calls = []

    def fake_popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return FakeProcess(returncode)

    monkeypatch.setattr(worker.subprocess, "Popen", fake_popen)
    return calls


def test_dm_target_uses_authenticated_context_only(monkeypatch, tmp_path, available_threadwire):
    _bind(chat_id="123456")
    calls = _capture_spawn(monkeypatch)

    result = json.loads(worker.launch_telegram_coding_worker({
        "provider": "codex", "prompt": "do work", "cwd": str(tmp_path)
    }))

    assert result == {"status": "completed", "exit_code": 0}
    argv, kwargs = calls[0]
    assert argv[:2] == [worker.THREADWIRE_EXECUTABLE, "run"]
    assert argv[argv.index("--target") + 1] == "telegram:123456"
    assert kwargs["shell"] is False
    assert kwargs["cwd"] == str(tmp_path)
    assert kwargs["stdout"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.DEVNULL
    assert kwargs["start_new_session"] is True
    assert "--prompt" not in argv
    assert "do work" not in argv
    assert kwargs["stdin"].closed


def test_forum_topic_target_and_passthrough(monkeypatch, tmp_path, available_threadwire):
    _bind(chat_id="-100987", thread_id="42")
    calls = _capture_spawn(monkeypatch)
    result = json.loads(worker.launch_telegram_coding_worker({
        "provider": "claude",
        "prompt": "do work",
        "cwd": str(tmp_path),
        "process_number": 7,
        "max_output_length": 2048,
        "resume_session": "session-1",
        "provider_arguments": ["--model", "sonnet"],
    }))

    assert result["status"] == "completed"
    argv = calls[0][0]
    assert argv[argv.index("--target") + 1] == "telegram:-100987:42"
    assert ["--process-number", "7"] == argv[argv.index("--process-number"):argv.index("--process-number") + 2]
    assert ["--max-output-length", "2048"] == argv[argv.index("--max-output-length"):argv.index("--max-output-length") + 2]
    assert ["--resume-session", "session-1"] == argv[argv.index("--resume-session"):argv.index("--resume-session") + 2]
    assert "--activity-log" not in argv
    assert argv[-3:] == ["--", "--model", "sonnet"]


def test_prompt_file_is_forwarded_without_hermes_reading_it(
    monkeypatch, tmp_path, available_threadwire
):
    _bind()
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("file prompt sentinel", encoding="utf-8")
    calls = _capture_spawn(monkeypatch)

    result = json.loads(worker.launch_telegram_coding_worker({
        "provider": "opencode",
        "prompt_file": str(prompt_file),
        "cwd": str(tmp_path),
    }))

    assert result == {"status": "completed", "exit_code": 0}
    argv = calls[0][0]
    assert argv[argv.index("--prompt-file") + 1] == str(prompt_file)
    assert "file prompt sentinel" not in argv


@pytest.mark.parametrize(
    ("platform", "chat_id", "thread_id"),
    [
        ("discord", "123", ""),
        ("", "123", ""),
        ("telegram", "", ""),
        ("telegram", "0", ""),
        ("telegram", "+12", ""),
        ("telegram", " 12", ""),
        ("telegram", "12", "0"),
        ("telegram", "12", "-1"),
        ("telegram", "12", "topic"),
        ("telegram", "12", str(2**53)),
    ],
)
def test_invalid_context_fails_before_spawn(
    monkeypatch, tmp_path, available_threadwire, platform, chat_id, thread_id
):
    _bind(platform=platform, chat_id=chat_id, thread_id=thread_id)
    calls = _capture_spawn(monkeypatch)

    result = json.loads(worker.launch_telegram_coding_worker({
        "provider": "codex", "prompt": "do work", "cwd": str(tmp_path)
    }))

    assert result["status"] == "blocked"
    assert calls == []


def test_process_environment_cannot_spoof_telegram_context(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "123456")
    monkeypatch.setenv("HERMES_SESSION_THREAD_ID", "42")
    requirement_checks = []
    monkeypatch.setattr(
        worker,
        "check_threadwire_requirements",
        lambda: requirement_checks.append(True) or True,
    )
    calls = _capture_spawn(monkeypatch)

    result = json.loads(worker.launch_telegram_coding_worker({
        "provider": "codex", "prompt": "do work", "cwd": str(tmp_path)
    }))

    assert result["status"] == "blocked"
    assert requirement_checks == []
    assert calls == []


def test_malformed_authenticated_request_fails_before_service_probe(monkeypatch, tmp_path):
    _bind()
    requirement_checks = []
    monkeypatch.setattr(
        worker,
        "check_threadwire_requirements",
        lambda: requirement_checks.append(True) or True,
    )
    calls = _capture_spawn(monkeypatch)

    result = json.loads(worker.launch_telegram_coding_worker({
        "provider": "codex",
        "prompt": "work",
        "target": "telegram:999",
        "cwd": str(tmp_path),
    }))

    assert result["status"] == "blocked"
    assert requirement_checks == []
    assert calls == []


def test_authenticated_request_preserves_service_unavailable_error(monkeypatch, tmp_path):
    _bind()
    monkeypatch.setattr(worker, "check_threadwire_requirements", lambda: False)
    calls = _capture_spawn(monkeypatch)

    result = json.loads(worker.launch_telegram_coding_worker({
        "provider": "codex", "prompt": "work", "cwd": str(tmp_path)
    }))

    assert result == {
        "status": "error",
        "error": "Telegram coding-worker service is unavailable",
    }
    assert calls == []


@pytest.mark.parametrize(
    "override",
    [
        {"target": "telegram:999"},
        {"chat_id": "999"},
        {"thread_id": "9"},
        {"provider_arguments": ["--target", "telegram:999"]},
        {"provider_arguments": ["--target=telegram:999"]},
        {"command": "/opt/data/bin/codex"},
        {"executable": "/opt/data/bin/claude"},
        {"env": {"HERMES_SESSION_CHAT_ID": "999"}},
        {"shell": True},
    ],
)
def test_caller_overrides_fail_before_spawn(
    monkeypatch, tmp_path, available_threadwire, override
):
    _bind()
    calls = _capture_spawn(monkeypatch)
    args = {"provider": "codex", "prompt": "do work", "cwd": str(tmp_path), **override}

    result = json.loads(worker.launch_telegram_coding_worker(args))

    assert result["status"] == "blocked"
    assert calls == []


@pytest.mark.parametrize("provider", ["", "gpt", "Codex", None, 1])
def test_invalid_provider_fails_before_spawn(
    monkeypatch, tmp_path, available_threadwire, provider
):
    _bind()
    calls = _capture_spawn(monkeypatch)
    result = json.loads(worker.launch_telegram_coding_worker({
        "provider": provider, "prompt": "do work", "cwd": str(tmp_path)
    }))
    assert result["status"] == "blocked"
    assert calls == []


def test_exact_threadwire_argv_never_uses_provider_helpers(
    monkeypatch, tmp_path, available_threadwire
):
    _bind()
    calls = _capture_spawn(monkeypatch)

    result = json.loads(worker.launch_telegram_coding_worker({
        "provider": "opencode", "stdin": "private prompt", "cwd": str(tmp_path)
    }))

    argv, kwargs = calls[0]
    assert result == {"status": "completed", "exit_code": 0}
    assert argv[0] == "/opt/data/bin/threadwire"
    assert not any(
        item in argv
        for item in (
            "/opt/data/bin/codex",
            "/opt/data/bin/claude",
            "/opt/data/bin/opencode-local-fleet",
        )
    )
    assert kwargs["shell"] is False


def test_large_stdin_uses_private_file_not_child_pipe(
    monkeypatch, tmp_path, available_threadwire
):
    _bind()
    private_prompt = "private-large-prompt-" * 100_000
    captured = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["stdin"] = kwargs["stdin"]
        assert kwargs["stdin"] is not subprocess.PIPE
        assert kwargs["stdin"].read() == private_prompt
        return FakeProcess()

    monkeypatch.setattr(worker.subprocess, "Popen", fake_popen)

    result = worker.launch_telegram_coding_worker({
        "provider": "codex",
        "stdin": private_prompt,
        "cwd": str(tmp_path),
    })

    assert json.loads(result) == {"status": "completed", "exit_code": 0}
    assert private_prompt not in captured["argv"]
    assert private_prompt not in result
    assert captured["stdin"].closed


def test_direct_prompt_uses_private_stdin_file_and_never_argv(
    monkeypatch, tmp_path, available_threadwire
):
    _bind()
    private_prompt = "direct-prompt-process-list-sentinel"
    captured = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["stdin"] = kwargs["stdin"]
        assert kwargs["stdin"].read() == private_prompt
        return FakeProcess()

    monkeypatch.setattr(worker.subprocess, "Popen", fake_popen)

    result = worker.launch_telegram_coding_worker({
        "provider": "codex",
        "prompt": private_prompt,
        "cwd": str(tmp_path),
    })

    assert json.loads(result) == {"status": "completed", "exit_code": 0}
    assert "--prompt" not in captured["argv"]
    assert private_prompt not in captured["argv"]
    assert captured["stdin"].closed


def test_stdin_staging_honors_cancellation_before_spawn(
    monkeypatch, tmp_path, available_threadwire
):
    _bind()
    calls = _capture_spawn(monkeypatch)
    monkeypatch.setattr(worker, "is_interrupted", lambda: True)

    result = worker.launch_telegram_coding_worker({
        "provider": "codex",
        "stdin": "private prompt",
        "cwd": str(tmp_path),
    })

    assert json.loads(result) == {
        "status": "cancelled",
        "error": "Coding-worker launch was cancelled",
    }
    assert calls == []


def test_stdin_staging_honors_deadline_before_spawn(
    monkeypatch, tmp_path, available_threadwire
):
    _bind()
    calls = _capture_spawn(monkeypatch)
    clock = iter([0.0, 2.0])
    monkeypatch.setattr(worker.time, "monotonic", lambda: next(clock))

    result = worker.launch_telegram_coding_worker({
        "provider": "codex",
        "stdin": "private prompt",
        "cwd": str(tmp_path),
        "timeout": 1,
    })

    assert json.loads(result) == {
        "status": "timed_out",
        "error": "Coding-worker launch timed out",
    }
    assert calls == []


@pytest.mark.parametrize("raised", [KeyboardInterrupt(), SystemExit(7)])
def test_stdin_staging_closes_file_and_propagates_base_exception(monkeypatch, raised):
    staged = io.StringIO()
    monkeypatch.setattr(worker.tempfile, "TemporaryFile", lambda **_kwargs: staged)
    monkeypatch.setattr(worker, "is_interrupted", lambda: (_ for _ in ()).throw(raised))

    with pytest.raises(type(raised)):
        worker._stage_stdin("private prompt", float("inf"))

    assert staged.closed


def test_registry_discovery_does_not_probe_threadwire(monkeypatch):
    entry = worker.registry.get_entry("telegram_coding_worker")
    assert entry is not None
    assert entry.check_fn is None

    monkeypatch.setattr(
        worker,
        "check_threadwire_requirements",
        lambda: pytest.fail("discovery must not probe Threadwire"),
    )
    definitions = worker.registry.get_definitions({"telegram_coding_worker"})

    assert any(
        definition["function"]["name"] == "telegram_coding_worker"
        for definition in definitions
    )
    definition = next(
        item for item in definitions
        if item["function"]["name"] == "telegram_coding_worker"
    )
    assert "activity_log" not in definition["function"]["parameters"]["properties"]


def test_activity_log_is_rejected_before_spawn(monkeypatch, tmp_path):
    _bind()
    calls = _capture_spawn(monkeypatch)

    result = json.loads(worker.launch_telegram_coding_worker({
        "provider": "codex",
        "prompt": "private prompt",
        "cwd": str(tmp_path),
        "activity_log": str(tmp_path / "activity.jsonl"),
    }))

    assert result["status"] == "blocked"
    assert calls == []


def test_telegram_credentials_are_absent_from_child_and_logs(
    monkeypatch, tmp_path, available_threadwire, caplog
):
    _bind(chat_id="765", thread_id="8")
    calls = _capture_spawn(monkeypatch)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "source-token-sentinel")
    monkeypatch.setenv("THREADWIRE_TELEGRAM_BOT_TOKEN", "private-token-sentinel")
    monkeypatch.setenv("THREADWIRE_VALIDATE_ONLY", "1")

    result = worker.launch_telegram_coding_worker({
        "provider": "codex",
        "prompt": "prompt-sentinel",
        "cwd": str(tmp_path),
        "resume_session": "session-sentinel",
    })

    child_env = calls[0][1]["env"]
    assert "TELEGRAM_BOT_TOKEN" not in child_env
    assert "THREADWIRE_TELEGRAM_BOT_TOKEN" not in child_env
    assert "THREADWIRE_VALIDATE_ONLY" not in child_env
    combined = result + "\n" + caplog.text
    for secret in (
        "source-token-sentinel",
        "private-token-sentinel",
        "prompt-sentinel",
        "session-sentinel",
        "telegram:765:8",
    ):
        assert secret not in combined


def test_gateway_session_metadata_is_absent_from_child_environment(
    monkeypatch, tmp_path, available_threadwire
):
    _bind(chat_id="765", thread_id="8")
    monkeypatch.setenv("HERMES_SESSION_KEY", "session-key-sentinel")
    monkeypatch.setenv("HERMES_SESSION_USER_ID", "user-sentinel")
    monkeypatch.setenv("HERMES_SESSION_MESSAGE_ID", "message-sentinel")
    monkeypatch.setenv("HERMES_SESSION_FUTURE_METADATA", "future-sentinel")
    calls = _capture_spawn(monkeypatch)

    result = json.loads(worker.launch_telegram_coding_worker({
        "provider": "codex", "prompt": "work", "cwd": str(tmp_path)
    }))

    assert result == {"status": "completed", "exit_code": 0}
    child_env = calls[0][1]["env"]
    assert not any(name.startswith("HERMES_SESSION_") for name in child_env)


def test_child_env_builder_never_reads_excluded_telegram_values(monkeypatch):
    from tools.environments import local

    class GuardedEnvironment(dict):
        def __getitem__(self, key):
            if key in {"TELEGRAM_BOT_TOKEN", "THREADWIRE_TELEGRAM_BOT_TOKEN"}:
                raise AssertionError("Telegram credential value was read")
            return super().__getitem__(key)

    guarded = GuardedEnvironment({
        "PATH": os.defpath,
        "TELEGRAM_BOT_TOKEN": "must-not-be-read",
        "THREADWIRE_TELEGRAM_BOT_TOKEN": "must-not-be-read",
    })
    monkeypatch.setattr(local.os, "environ", guarded)

    env = local.hermes_subprocess_env(
        inherit_credentials=True,
        exclude_keys=frozenset({"THREADWIRE_TELEGRAM_BOT_TOKEN"}),
    )

    assert "TELEGRAM_BOT_TOKEN" not in env
    assert "THREADWIRE_TELEGRAM_BOT_TOKEN" not in env


def test_contextvars_isolate_concurrent_targets(monkeypatch, tmp_path, available_threadwire):
    barrier = threading.Barrier(2)
    targets = []
    lock = threading.Lock()

    def fake_popen(argv, **_kwargs):
        barrier.wait(timeout=2)
        with lock:
            targets.append(argv[argv.index("--target") + 1])
        return FakeProcess()

    monkeypatch.setattr(worker.subprocess, "Popen", fake_popen)

    def launch(chat_id, thread_id):
        set_session_vars(platform="telegram", chat_id=chat_id, thread_id=thread_id)
        return worker.launch_telegram_coding_worker({
            "provider": "codex", "prompt": "work", "cwd": str(tmp_path)
        })

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda pair: launch(*pair), [("111", ""), ("222", "9")]))

    assert sorted(targets) == ["telegram:111", "telegram:222:9"]
    assert all(json.loads(result)["status"] == "completed" for result in results)


def test_user_cancellation_is_forwarded_and_returns_only_safe_status(
    monkeypatch, tmp_path, available_threadwire
):
    _bind()

    class RunningProcess(FakeProcess):
        def __init__(self):
            super().__init__(None)

        def send_signal(self, sig):
            super().send_signal(sig)
            self._returncode = -int(sig)

    proc = RunningProcess()
    killed_groups = []
    interrupted = iter([False, True])
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *_a, **_k: proc)
    monkeypatch.setattr(worker, "is_interrupted", lambda: next(interrupted))
    monkeypatch.setattr(
        worker.os,
        "killpg",
        lambda pid, sig: (killed_groups.append((pid, sig)), setattr(proc, "_returncode", -int(sig))),
    )

    result = json.loads(worker.launch_telegram_coding_worker({
        "provider": "codex", "prompt": "secret prompt", "cwd": str(tmp_path)
    }))

    assert result == {"status": "cancelled", "error": "Coding-worker launch was cancelled"}
    assert killed_groups == [(proc.pid, worker.signal.SIGINT)]
    assert proc.signals == []


def test_timeout_terminates_threadwire_and_returns_only_safe_status(
    monkeypatch, tmp_path, available_threadwire
):
    _bind()

    class RunningProcess(FakeProcess):
        def __init__(self):
            super().__init__(None)

        def send_signal(self, sig):
            super().send_signal(sig)
            self._returncode = -int(sig)

    proc = RunningProcess()
    killed_groups = []
    clock = iter([0.0, 0.0, 2.0, 2.0])
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *_a, **_k: proc)
    monkeypatch.setattr(worker, "is_interrupted", lambda: False)
    monkeypatch.setattr(worker.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        worker.os,
        "killpg",
        lambda pid, sig: (killed_groups.append((pid, sig)), setattr(proc, "_returncode", -int(sig))),
    )

    result = json.loads(worker.launch_telegram_coding_worker({
        "provider": "claude", "prompt": "secret prompt", "cwd": str(tmp_path), "timeout": 1
    }))

    assert result == {"status": "timed_out", "error": "Coding-worker launch timed out"}
    assert killed_groups == [(proc.pid, worker.signal.SIGTERM)]
    assert proc.signals == []


def test_forced_kill_reaps_direct_child(monkeypatch):
    proc = FakeProcess(None)
    clock = iter([0.0, 4.0])
    killed_groups = []
    monkeypatch.setattr(worker.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        worker.os,
        "killpg",
        lambda pid, sig: (
            killed_groups.append((pid, sig)),
            setattr(proc, "_returncode", -9) if sig == worker.signal.SIGKILL else None,
        ),
    )

    worker._stop_process(cast(subprocess.Popen, proc), worker.signal.SIGTERM)

    assert proc.signals == []
    assert killed_groups == [
        (proc.pid, worker.signal.SIGTERM),
        (proc.pid, worker.signal.SIGKILL),
    ]
    assert proc.wait_timeouts == [worker._TERMINATE_GRACE_SECONDS]


@pytest.mark.parametrize("raised", [KeyboardInterrupt(), SystemExit(7)])
def test_base_exception_after_spawn_cleans_group_reaps_and_propagates(
    monkeypatch, tmp_path, available_threadwire, raised
):
    _bind()
    proc = FakeProcess(None)
    raised_once = False

    def interrupt_poll():
        nonlocal raised_once
        if not raised_once:
            raised_once = True
            raise raised
        return proc._returncode

    monkeypatch.setattr(proc, "poll", interrupt_poll)
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *_a, **_k: proc)
    killed_groups = []
    monkeypatch.setattr(
        worker.os,
        "killpg",
        lambda pid, sig: (killed_groups.append((pid, sig)), setattr(proc, "_returncode", -int(sig))),
    )

    with pytest.raises(type(raised)):
        worker.launch_telegram_coding_worker({
            "provider": "codex", "prompt": "work", "cwd": str(tmp_path)
        })

    assert killed_groups == [(proc.pid, worker.signal.SIGTERM)]
    assert proc.signals == []
    assert proc.wait_timeouts == [worker._TERMINATE_GRACE_SECONDS]


def test_display_redaction_hides_all_sensitive_worker_arguments(tmp_path):
    from agent.display import build_tool_preview, redact_tool_args_for_display

    args = {
        "provider": "codex",
        "prompt": "prompt-sentinel",
        "cwd": str(tmp_path),
        "provider_arguments": ["--model", "secret-model"],
        "resume_session": "session-sentinel",
    }
    safe = redact_tool_args_for_display("telegram_coding_worker", args)
    rendered = json.dumps(safe)
    assert "prompt-sentinel" not in rendered
    assert str(tmp_path) not in rendered
    assert "secret-model" not in rendered
    assert "session-sentinel" not in rendered
    assert build_tool_preview("telegram_coding_worker", args) == "coding worker"

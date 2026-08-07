"""Regression coverage for the hard caller deadline around Edge TTS."""

import asyncio
import concurrent.futures
import json
import threading
from pathlib import Path

from tools import tts_tool


def _select_edge(monkeypatch):
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "edge"})
    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object())


def test_edge_timeout_returns_before_cancellation_resistant_provider_finishes(tmp_path, monkeypatch):
    provider_started = threading.Event()
    cancellation_seen = threading.Event()
    provider_finished = threading.Event()
    release_provider = threading.Event()
    tool_returned = threading.Event()
    result_holder = {}
    output_path = tmp_path / "out.mp3"

    async def cancellation_resistant_provider(_text, path, _config):
        provider_started.set()
        try:
            await asyncio.to_thread(release_provider.wait)
        except asyncio.CancelledError:
            cancellation_seen.set()
            release_provider.wait()
        Path(path).write_bytes(b"late audio")
        provider_finished.set()
        return path

    _select_edge(monkeypatch)
    monkeypatch.setattr(tts_tool, "_generate_edge_tts", cancellation_resistant_provider)
    monkeypatch.setattr(tts_tool, "_edge_tts_timeout_seconds", lambda: 0.05)

    def invoke_tool():
        result_holder["result"] = json.loads(
            tts_tool.text_to_speech_tool("hello", output_path=str(output_path))
        )
        tool_returned.set()

    caller = threading.Thread(target=invoke_tool)
    caller.start()
    try:
        assert provider_started.wait(1), "provider did not start"
        assert tool_returned.wait(1), "caller remained joined to provider after its deadline"
        assert not provider_finished.is_set(), "provider completed naturally before caller returned"
        assert result_holder["result"]["success"] is False
        assert "timed out after 0.05 seconds" in result_holder["result"]["error"]
        assert not output_path.exists()
    finally:
        release_provider.set()
        caller.join(2)

    assert not caller.is_alive(), "caller did not shut down"
    assert cancellation_seen.wait(1), "deadline did not request coroutine cancellation"
    assert provider_finished.wait(1), "provider did not finish after test release"
    assert not output_path.exists(), "late provider output became current-call output"


def test_edge_timeout_registers_cleanup_before_observing_worker_completion(tmp_path, monkeypatch):
    provider_started = threading.Event()
    release_provider = threading.Event()
    provider_finished = threading.Event()
    staging_removed = threading.Event()
    output_path = tmp_path / "out.mp3"
    real_future_done = concurrent.futures.Future.done
    real_unlink = tts_tool.os.unlink
    done_checks = 0

    async def cancellation_resistant_provider(_text, path, _config):
        provider_started.set()
        release_provider.wait()
        Path(path).write_bytes(b"late audio")
        provider_finished.set()
        return path

    def finish_worker_during_done_check(future):
        nonlocal done_checks
        done_checks += 1
        if done_checks == 1:
            return False
        release_provider.set()
        assert provider_finished.wait(1), "provider did not finish in cleanup race"
        return real_future_done(future)

    def observe_unlink(path):
        real_unlink(path)
        staging_removed.set()

    monkeypatch.setattr(tts_tool, "_generate_edge_tts", cancellation_resistant_provider)
    monkeypatch.setattr(concurrent.futures.Future, "done", finish_worker_during_done_check)
    monkeypatch.setattr(tts_tool.os, "unlink", observe_unlink)

    try:
        tts_tool._run_edge_tts_with_timeout(
            "hello", str(output_path), {}, timeout=0.01
        )
    except tts_tool._EdgeTTSDeadlineExceeded:
        pass
    else:
        raise AssertionError("caller deadline did not expire")

    assert provider_started.is_set()
    release_provider.set()
    assert provider_finished.wait(1), "provider did not finish after test release"
    assert staging_removed.wait(1), "completion callback did not remove staging output"
    assert not list(tmp_path.glob("out.mp3.edge-pending-*"))


def test_edge_worker_drains_provider_child_tasks_before_loop_close(tmp_path, monkeypatch):
    child_started = asyncio.Event()
    child_cancelled = threading.Event()

    async def provider_with_child(_text, path, _config):
        async def child():
            child_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                child_cancelled.set()

        asyncio.create_task(child())
        await child_started.wait()
        Path(path).write_bytes(b"audio")
        return path

    monkeypatch.setattr(tts_tool, "_generate_edge_tts", provider_with_child)

    output_path = tmp_path / "out.mp3"
    tts_tool._run_edge_tts_with_timeout("hello", str(output_path), {})

    assert output_path.read_bytes() == b"audio"
    assert child_cancelled.is_set(), "worker loop closed without draining child task"


def test_edge_timeout_does_not_wait_for_worker_loop_publication(tmp_path, monkeypatch):
    loop_creation_allowed = threading.Event()
    provider_started = threading.Event()
    caller_returned = threading.Event()
    result_holder = {}
    real_new_event_loop = asyncio.new_event_loop

    def delayed_new_event_loop():
        assert loop_creation_allowed.wait(1), "test did not release worker-loop creation"
        return real_new_event_loop()

    async def provider(_text, path, _config):
        provider_started.set()
        Path(path).write_bytes(b"late audio")
        return path

    monkeypatch.setattr(asyncio, "new_event_loop", delayed_new_event_loop)
    monkeypatch.setattr(tts_tool, "_generate_edge_tts", provider)

    def invoke_helper():
        try:
            tts_tool._run_edge_tts_with_timeout(
                "hello", str(tmp_path / "out.mp3"), {}, timeout=0.01
            )
        except tts_tool._EdgeTTSDeadlineExceeded as exc:
            result_holder["error"] = exc
        finally:
            caller_returned.set()

    caller = threading.Thread(target=invoke_helper)
    caller.start()
    try:
        assert caller_returned.wait(0.2), "loop readiness extended the caller timeout budget"
        assert isinstance(result_holder.get("error"), tts_tool._EdgeTTSDeadlineExceeded)
        assert not provider_started.is_set()
    finally:
        loop_creation_allowed.set()
        caller.join(1)

    assert not caller.is_alive()
    assert not provider_started.wait(0.1), "late loop publication started cancelled work"
    assert not (tmp_path / "out.mp3").exists()


def test_edge_provider_timeout_preserves_provider_error(tmp_path, monkeypatch):
    calls = 0

    async def provider_timeout(_text, _path, _config):
        nonlocal calls
        calls += 1
        raise TimeoutError("socket handshake timed out")

    _select_edge(monkeypatch)
    monkeypatch.setattr(tts_tool, "_generate_edge_tts", provider_timeout)

    result = json.loads(
        tts_tool.text_to_speech_tool("hello", output_path=str(tmp_path / "out.mp3"))
    )

    assert result["success"] is False
    assert "socket handshake timed out" in result["error"]
    assert "timed out after 60 seconds" not in result["error"]
    assert calls == 1


def test_edge_runtime_error_does_not_retry_synthesis(tmp_path, monkeypatch):
    calls = 0

    async def provider_failure(_text, _path, _config):
        nonlocal calls
        calls += 1
        raise RuntimeError("provider runtime failure")

    _select_edge(monkeypatch)
    monkeypatch.setattr(tts_tool, "_generate_edge_tts", provider_failure)

    result = json.loads(
        tts_tool.text_to_speech_tool("hello", output_path=str(tmp_path / "out.mp3"))
    )

    assert result["success"] is False
    assert "provider runtime failure" in result["error"]
    assert calls == 1

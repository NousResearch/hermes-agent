"""Regression tests for the bounded edge-TTS worker (#126865).

_run_edge_tts must raise TimeoutError when the async Edge generator wedges — the
old ThreadPoolExecutor form bounded only the *wait*: leaving the ``with`` block
joined the worker thread (shutdown(wait=True)), so a hung generator blocked the
caller forever and the timeout never propagated. With voice.auto_tts enabled that
pinned the whole turn-delivery coroutine and froze the session lane.
"""

import asyncio
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

import tools.tts_tool as tts_tool


@pytest.fixture
def audio_dir(tmp_path):
    # The global home-sandbox fixture creates hermes_test under tmp_path.
    directory = tmp_path / "audio"
    directory.mkdir()
    return directory


@pytest.fixture(autouse=True)
def _fast_bound(monkeypatch):
    # Shrink the join bound so tests don't wait a real minute.
    monkeypatch.setattr(tts_tool, "_EDGE_TTS_TIMEOUT_S", 0.05)
    monkeypatch.setattr(tts_tool, "_EDGE_TTS_WORKER_SLOTS", threading.BoundedSemaphore(2))


def test_run_edge_tts_times_out_when_generator_hangs(audio_dir):
    """A wedged generator must raise TimeoutError promptly instead of blocking forever."""
    target = audio_dir / "out.mp3"

    async def hang(*_args, **_kwargs):
        await asyncio.sleep(300)

    original = tts_tool._generate_edge_tts
    try:
        tts_tool._generate_edge_tts = hang
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            tts_tool._run_edge_tts("hi", str(target), {})
        elapsed = time.monotonic() - start
    finally:
        tts_tool._generate_edge_tts = original

    # Bounded join: well under the old forever-block; generous ceiling for slow CI.
    assert elapsed < 10


def test_run_edge_tts_reraises_generator_errors(audio_dir):
    """Exceptions from the worker (non-Timeout) must propagate to the caller."""
    target = audio_dir / "out.mp3"

    async def boom(*_args, **_kwargs):
        raise ValueError("synthesis exploded")

    original = tts_tool._generate_edge_tts
    try:
        tts_tool._generate_edge_tts = boom
        with pytest.raises(ValueError, match="synthesis exploded"):
            tts_tool._run_edge_tts("hi", str(target), {})
    finally:
        tts_tool._generate_edge_tts = original


def test_run_edge_tts_success_writes_file(audio_dir):
    """The happy path still works: generator runs, file lands, no exception."""
    target = audio_dir / "out.mp3"

    async def quick(text, file_str, cfg):
        with open(file_str, "w") as fh:
            fh.write("audio")

    original = tts_tool._generate_edge_tts
    try:
        tts_tool._generate_edge_tts = quick
        tts_tool._run_edge_tts("hi", str(target), {})
    finally:
        tts_tool._generate_edge_tts = original

    assert target.exists() and target.read_text() == "audio"
    assert list(audio_dir.iterdir()) == [target]


def test_timeout_requests_cancellation_and_releases_worker(audio_dir, monkeypatch):
    cancelled = threading.Event()
    workers = []

    async def hang(*args):
        workers.append(threading.current_thread())
        try:
            await asyncio.sleep(300)
        finally:
            cancelled.set()

    monkeypatch.setattr(tts_tool, "_generate_edge_tts", hang)
    with pytest.raises(TimeoutError):
        tts_tool._run_edge_tts("hi", str(audio_dir / "out.mp3"), {})
    assert cancelled.wait(2)
    workers[0].join(2)
    assert not workers[0].is_alive()
    assert list(audio_dir.iterdir()) == []
    assert tts_tool._EDGE_TTS_WORKER_SLOTS.acquire(blocking=False)
    assert tts_tool._EDGE_TTS_WORKER_SLOTS.acquire(blocking=False)
    tts_tool._EDGE_TTS_WORKER_SLOTS.release()
    tts_tool._EDGE_TTS_WORKER_SLOTS.release()


@pytest.mark.parametrize("retry", [False, True])
def test_late_sdk_completion_never_publishes_or_overwrites_retry(audio_dir, monkeypatch, retry):
    """Keep the real generator and tool envelope; simulate an SDK stuck in native I/O."""
    release = threading.Event()
    workers, paths = [], []
    target = audio_dir / "reply.mp3"

    class Communicate:
        def __init__(self, text, **kwargs):
            self.text = text

        async def save(self, path):
            paths.append(path)
            if self.text == "old":
                workers.append(threading.current_thread())
                assert release.wait(5)
            Path(path).write_bytes(b"ID3" + self.text.encode())

    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: SimpleNamespace(Communicate=Communicate))
    def synthesize(text):
        return json.loads(tts_tool._text_to_speech_single(
            text, str(target), provider="edge", tts_config={}, command_provider_config=None,
            want_opus=False, instructions=None,
        ))

    try:
        result = synthesize("old")
        assert result["success"] is False
        assert "did not finish" in result["error"]
        assert not target.exists()
        if retry:
            result = synthesize("new")
            assert result["success"] is True
            assert target.read_bytes() == b"ID3new"
    finally:
        release.set()
        for worker in workers:
            worker.join(2)
            assert not worker.is_alive()
    assert all(Path(path) != target for path in paths)
    assert len(set(paths)) == len(paths)
    assert list(audio_dir.iterdir()) == ([target] if retry else [])
    if retry:
        assert target.read_bytes() == b"ID3new"


def test_stuck_workers_are_capped_and_capacity_recovers(audio_dir, monkeypatch):
    release = threading.Event()
    workers = []

    async def stuck(text, path, cfg):
        workers.append(threading.current_thread())
        assert release.wait(5)
        Path(path).write_text("late")

    monkeypatch.setattr(tts_tool, "_generate_edge_tts", stuck)
    try:
        for i in range(2):
            with pytest.raises(TimeoutError, match="did not finish"):
                tts_tool._run_edge_tts("hi", str(audio_dir / f"{i}.mp3"), {})
        start = time.monotonic()
        for i in range(10):
            with pytest.raises(TimeoutError, match="workers are still busy"):
                tts_tool._run_edge_tts("hi", str(audio_dir / "extra.mp3"), {})
        assert time.monotonic() - start < 1
        assert len(workers) == 2
    finally:
        release.set()
        for worker in workers:
            worker.join(2)
            assert not worker.is_alive()
    assert list(audio_dir.iterdir()) == []

    async def quick(text, path, cfg):
        Path(path).write_text("recovered")

    monkeypatch.setattr(tts_tool, "_generate_edge_tts", quick)
    target = audio_dir / "new.mp3"
    tts_tool._run_edge_tts("hi", str(target), {})
    assert target.read_text() == "recovered"


def test_configured_timeout_allows_slow_synthesis(audio_dir, monkeypatch):
    async def slow(text, path, cfg):
        await asyncio.sleep(0.1)
        Path(path).write_text("audio")

    monkeypatch.setattr(tts_tool, "_generate_edge_tts", slow)
    target = audio_dir / "slow.mp3"
    tts_tool._run_edge_tts("hi", str(target), {"edge": {"timeout": 1}})
    assert target.read_text() == "audio"


@pytest.mark.parametrize("timeout", [0, -1, "bad", None, True, float("inf"), float("nan")])
def test_invalid_timeout_rejected_before_starting(audio_dir, monkeypatch, timeout):
    async def unexpected(*args):
        pytest.fail("invalid timeout must not start synthesis")

    monkeypatch.setattr(tts_tool, "_generate_edge_tts", unexpected)
    with pytest.raises(ValueError, match="positive finite"):
        tts_tool._run_edge_tts("hi", str(audio_dir / "out.mp3"), {"edge": {"timeout": timeout}})
    assert list(audio_dir.iterdir()) == []


def test_runtime_error_does_not_retry_without_timeout(audio_dir, monkeypatch):
    calls = []

    async def boom(*args):
        calls.append(1)
        raise RuntimeError("SDK exploded")

    monkeypatch.setattr(tts_tool, "_generate_edge_tts", boom)
    with pytest.raises(RuntimeError, match="SDK exploded"):
        tts_tool._run_edge_tts("hi", str(audio_dir / "out.mp3"), {})
    assert len(calls) == 1
    assert list(audio_dir.iterdir()) == []


def test_thread_start_failure_cleans_staging_and_releases_capacity(audio_dir, monkeypatch):
    def fail_start(self):
        raise RuntimeError("cannot start thread")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="cannot start thread"):
        tts_tool._run_edge_tts("hi", str(audio_dir / "out.mp3"), {})
    assert list(audio_dir.iterdir()) == []
    assert tts_tool._EDGE_TTS_WORKER_SLOTS.acquire(blocking=False)
    assert tts_tool._EDGE_TTS_WORKER_SLOTS.acquire(blocking=False)
    tts_tool._EDGE_TTS_WORKER_SLOTS.release()
    tts_tool._EDGE_TTS_WORKER_SLOTS.release()

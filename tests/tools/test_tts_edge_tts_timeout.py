"""Regression tests for the bounded edge-TTS worker (#126865).

_run_edge_tts must raise TimeoutError when the async Edge generator wedges — the
old ThreadPoolExecutor form bounded only the *wait*: leaving the ``with`` block
joined the worker thread (shutdown(wait=True)), so a hung generator blocked the
caller forever and the timeout never propagated. With voice.auto_tts enabled that
pinned the whole turn-delivery coroutine and froze the session lane.
"""

import asyncio
import time

import pytest

import tools.tts_tool as tts_tool


@pytest.fixture(autouse=True)
def _fast_bound(monkeypatch):
    # Shrink the join bound so tests don't wait a real minute.
    monkeypatch.setattr(tts_tool, "_EDGE_TTS_TIMEOUT_S", 1)


def test_run_edge_tts_times_out_when_generator_hangs(tmp_path):
    """A wedged generator must raise TimeoutError promptly instead of blocking forever."""
    target = tmp_path / "out.mp3"

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


def test_run_edge_tts_reraises_generator_errors(tmp_path):
    """Exceptions from the worker (non-Timeout) must propagate to the caller."""
    target = tmp_path / "out.mp3"

    async def boom(*_args, **_kwargs):
        raise ValueError("synthesis exploded")

    original = tts_tool._generate_edge_tts
    try:
        tts_tool._generate_edge_tts = boom
        with pytest.raises(ValueError, match="synthesis exploded"):
            tts_tool._run_edge_tts("hi", str(target), {})
    finally:
        tts_tool._generate_edge_tts = original


def test_run_edge_tts_success_writes_file(tmp_path):
    """The happy path still works: generator runs, file lands, no exception."""
    target = tmp_path / "out.mp3"

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

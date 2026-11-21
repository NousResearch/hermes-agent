"""Edge TTS retries a transient "no audio" response instead of failing the call.

Edge's websocket intermittently closes without yielding any audio for a request that is otherwise
valid (same text, voice and rate); a fresh ``Communicate`` usually returns audio. The retry is
scoped to ``NoAudioReceived`` so a bad voice, an auth failure or an offline network still fails on
the first attempt instead of being repeated.
"""

from pathlib import Path

import pytest

from tools import tts_tool
from tools import tts_tool_providers as providers


class NoAudioReceived(Exception):
    """Stand-in for ``edge_tts.exceptions.NoAudioReceived``."""


class _FakeRequest:
    """One ``Communicate(...).save()`` call; replays the next scripted outcome."""

    def __init__(self, module: "_FakeEdgeModule"):
        self._module = module

    async def save(self, output_path: str) -> None:
        module = self._module
        module.calls += 1
        Path(output_path).write_bytes(b"partial")  # a failed attempt leaves a partial file behind
        outcome = module.outcomes.pop(0)
        if outcome is not None:
            raise outcome
        Path(output_path).write_bytes(b"audio")


class _FakeEdgeModule:
    """``edge_tts`` stand-in: ``exceptions.NoAudioReceived`` plus a scripted ``Communicate``."""

    def __init__(self, outcomes):
        self.exceptions = type("Exceptions", (), {"NoAudioReceived": NoAudioReceived})
        self.outcomes = list(outcomes)
        self.calls = 0
        self.kwargs: list[dict] = []

    def Communicate(self, text: str, **kwargs):
        self.kwargs.append(kwargs)
        return _FakeRequest(self)


@pytest.fixture
def edge(monkeypatch):
    """Install a fake ``edge_tts``; pass one outcome per attempt the call should make."""

    def install(*outcomes):
        module = _FakeEdgeModule(outcomes)
        monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: module)
        # Keep the retry path real but instant: the delay only paces production retries.
        monkeypatch.setattr(providers, "_EDGE_TTS_RETRY_DELAY_SECONDS", 0.0, raising=False)
        return module

    return install


@pytest.mark.asyncio
async def test_transient_no_audio_is_retried_into_audio(edge, tmp_path):
    module = edge(NoAudioReceived("no audio was received"), None)
    target = tmp_path / "reply.mp3"

    assert await providers._generate_edge_tts("hello", str(target), {}) == str(target)

    assert module.calls == 2
    assert target.read_bytes() == b"audio"
    assert module.kwargs[0] == {"voice": providers.DEFAULT_EDGE_VOICE}


@pytest.mark.asyncio
async def test_other_failures_are_not_retried(edge, tmp_path):
    module = edge(ValueError("unsupported voice"))

    with pytest.raises(ValueError):
        await providers._generate_edge_tts("hello", str(tmp_path / "reply.mp3"), {})

    assert module.calls == 1


@pytest.mark.asyncio
async def test_gives_up_after_three_attempts(edge, tmp_path):
    module = edge(NoAudioReceived(), NoAudioReceived(), NoAudioReceived())

    with pytest.raises(NoAudioReceived):
        await providers._generate_edge_tts("hello", str(tmp_path / "reply.mp3"), {})

    assert module.calls == 3

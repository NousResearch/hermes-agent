"""Speaker-side streaming pipeline: ``stream_tts_to_speaker``.

Turns a queue of LLM text deltas into audio the moment each sentence is complete. Two paths
share the sentence cutter (``tools.tts_streaming``): :class:`_StreamerPlayback` for a registered
chunked streamer (prefetch thread per sentence, one FIFO playback worker through a sounddevice
OutputStream or temp WAV + system player) and :class:`_SyncSentencePipeline` for every other
provider (per-sentence ``text_to_speech_tool`` on a single-thread executor, overlapped with
playback). Origin seams are resolved through :func:`_origin` at call time.
"""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import queue
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Iterable, Iterator, List, Optional

from tools.tts_text_normalize import _strip_markdown_for_tts
from tools.tts_tool_delivery import _origin, _remove_quietly as _unlink_quietly

logger = logging.getLogger("tools.tts_tool")

def _align_int16_chunks(chunks: Iterable[bytes], stop_evt: threading.Event, *, pad_tail: bool = True) -> Iterator[bytes]:
    """Yield int16-aligned byte chunks; a dangling odd byte is padded at the end (or dropped)."""
    leftover = b""
    for chunk in chunks:
        if stop_evt.is_set():
            break
        buf = leftover + chunk
        aligned_len = len(buf) - (len(buf) % 2)
        if aligned_len >= 2:
            yield buf[:aligned_len]
        leftover = buf[aligned_len:]
    if leftover and pad_tail:
        yield b"\x00"


def _play_via_tempfile(audio_iter: Iterable[bytes], stop_evt: threading.Event, sample_rate: int = 24000) -> None:
    """Write PCM chunks to a temp WAV file and play it with the system player."""
    tmp = tmp_path = None
    try:
        import wave
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            for aligned in _align_int16_chunks(audio_iter, stop_evt):
                wf.writeframes(aligned)
        # wave.open() on a file object does NOT close it; on Windows the open write
        # handle blocks the player and the unlink below (WinError 32).
        tmp.close()
        from tools.voice_mode import play_audio_file
        play_audio_file(tmp_path)
    except Exception as exc:
        logger.warning("Temp-file TTS fallback failed: %s", exc)
    finally:
        if tmp is not None:
            with contextlib.suppress(Exception):
                tmp.close()  # idempotent; ensures close on early error
        _unlink_quietly(tmp_path)


def _drain_chunks(chunk_queue: "queue.Queue[Optional[bytes]]") -> List[bytes]:
    """Collect one sentence's PCM chunks up to the ``None`` sentinel."""
    return list(iter(chunk_queue.get, None))


class _SyncSentencePipeline:
    """Overlap per-sentence synthesis with playback for non-streaming providers.

    One single-thread synthesis executor (FIFO; providers never see concurrent calls) feeds one
    playback worker through a small bounded queue, so sentence n+1 synthesizes while n plays;
    the bound keeps lookahead/temp files small and gives the caller backpressure.
    ``text_to_speech_tool`` / ``play_audio_file`` are resolved late so test patches apply."""

    def __init__(self, stop_event: threading.Event, *, lookahead: int = 2):
        self._stop = stop_event
        self._queue: "queue.Queue[Optional[tuple[str, Future]]]" = queue.Queue(maxsize=max(1, lookahead))
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-sync-synth")
        self._player = threading.Thread(target=self._drain, name="tts-sync-play", daemon=True)
        self._player.start()

    def speak(self, cleaned: str) -> None:
        """Queue one sentence. Blocks only when the lookahead bound is full."""
        if not self._stop.is_set():
            self._queue.put((cleaned, self._executor.submit(self._synthesize_to_tmp, cleaned)))

    def close(self) -> None:
        """Flush queued sentences in order (skipped if stopped), then join."""
        self._queue.put(None)
        self._player.join()
        self._executor.shutdown(wait=True)

    def _synthesize_to_tmp(self, cleaned: str) -> Optional[str]:
        if self._stop.is_set():
            return None
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            _origin().text_to_speech_tool(text=cleaned, output_path=tmp_path)
            return tmp_path
        except Exception as exc:
            logger.warning("Sync per-sentence TTS synthesis failed: %s", exc)
            _unlink_quietly(tmp_path)
            return None

    def _drain(self) -> None:
        for _sentence, future in iter(self._queue.get, None):
            tmp_path = None
            try:
                tmp_path = future.result()
                if tmp_path and not self._stop.is_set() and os.path.isfile(tmp_path) and os.path.getsize(tmp_path) > 0:
                    from tools.voice_mode import play_audio_file
                    play_audio_file(tmp_path)
            except Exception as exc:
                logger.warning("Sync per-sentence TTS failed: %s", exc)
            finally:
                _unlink_quietly(tmp_path)


class _StreamerPlayback:
    """Prefetch + FIFO playback for a chunked :class:`StreamingTTSProvider`.

    ``speak(text)`` starts ``streamer.stream()`` immediately on a prefetch thread (at most 3 in
    flight) buffering into a bounded per-sentence queue; one playback worker drains those in order,
    so sentence N+1 arrives while N plays. Output is a PortAudio stream when one opened, else temp
    WAV files; a failing write is retried on a reinitialized stream up to ``_MAX_REINIT`` times."""

    _MAX_REINIT = 3
    _CHUNK_QUEUE_MAX = 64

    def __init__(self, streamer, stop_event: threading.Event):
        self.streamer, self.stop_event = streamer, stop_event
        self.output_stream = self._open_output_stream()
        self._audio_queue: "queue.Queue[Optional[queue.Queue[Optional[bytes]]]]" = queue.Queue()
        self._prefetch_threads: List[threading.Thread] = []
        self._prefetch_sem = threading.Semaphore(3)
        self._worker = threading.Thread(target=self._playback_worker, daemon=True)
        self._worker.start()

    def _create_output_stream(self):
        sd = _origin()._import_sounddevice()
        stream = sd.OutputStream(
            samplerate=self.streamer.sample_rate, channels=self.streamer.channels, dtype="int16")
        stream.start()
        return stream

    def _open_output_stream(self):
        # macOS skips sounddevice entirely: PortAudio/CoreAudio init triggers a
        # kTCCServiceMediaLibrary prompt though output needs no media-library access.
        # None routes every sentence through tempfile -> afplay.
        # See PR #62601 / #13291.
        if platform.system() == "Darwin":
            return None
        try:
            return self._create_output_stream()
        except (ImportError, OSError) as exc:
            logger.debug("sounddevice not available, streamer→tempfile: %s", exc)
        except Exception as exc:
            logger.warning("sounddevice OutputStream failed: %s", exc)
        return None

    def close_output_stream(self) -> None:
        """Always release the device so a later stream can open it."""
        if self.output_stream is not None:
            with contextlib.suppress(Exception):
                self.output_stream.stop()
                self.output_stream.close()

    def speak(self, text: str) -> None:
        """Start ``streamer.stream(text)`` and prefetch its chunks immediately."""
        try:
            audio_iter = self.streamer.stream(text)
        except Exception as exc:
            logger.warning("Streaming TTS synthesis failed: %s", exc)
            return
        self._prefetch_sem.acquire()
        chunk_queue: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=self._CHUNK_QUEUE_MAX)
        self._audio_queue.put(chunk_queue)
        self._prefetch_threads.append(threading.Thread(
            target=self._consume_to_queue, args=(audio_iter, chunk_queue), daemon=True))
        self._prefetch_threads[-1].start()

    def _consume_to_queue(self, audio_iter: Iterator[bytes], chunk_queue: "queue.Queue[Optional[bytes]]") -> None:
        try:
            for chunk in audio_iter:
                if self.stop_event.is_set():
                    logger.info("TTS CUT: prefetch cancelled (stop_event set mid-sentence) — partial audio only")
                    break
                chunk_queue.put(chunk, timeout=30.0)
        except Exception as exc:
            logger.warning("TTS CUT: streaming TTS prefetch failed mid-sentence (partial audio only): %s", exc)
        finally:
            chunk_queue.put(None)  # sentinel: no more chunks
            self._prefetch_sem.release()

    def _play_sentence_via_tempfile(self, chunk_queue) -> None:
        _play_via_tempfile(_drain_chunks(chunk_queue), self.stop_event, self.streamer.sample_rate)

    def _for_each_sentence(self, play: Callable[[queue.Queue], None]) -> None:
        """Feed queued sentences to *play* in order until the end sentinel; stopped sentences are skipped."""
        for chunk_queue in iter(self._audio_queue.get, None):
            if not self.stop_event.is_set():
                play(chunk_queue)

    def _write_pcm(self, buf: bytes) -> None:
        self._current_stream.write(self._np.frombuffer(buf, dtype="<i2").reshape(-1, 1))

    def _recover_stream(self) -> bool:
        """Close the broken PortAudio stream and open a fresh one after a failed write; False once
        ``_MAX_REINIT`` is exhausted (remaining sentences go through temp files)."""
        if self._reinit_count >= self._MAX_REINIT:
            logger.warning(
                "TTS: PortAudio reinit exhausted after %d attempts, falling back to tempfile for remaining sentences",
                self._MAX_REINIT)
            self._current_stream = None
            return False
        self._reinit_count += 1
        self.close_output_stream()
        try:
            self.output_stream = self._create_output_stream()
            logger.info("TTS: PortAudio output stream reinitialized after error")
        except Exception as exc:
            logger.warning("TTS: PortAudio stream reinit failed: %s", exc)
            self.output_stream = None
        self._current_stream = self.output_stream
        return self._current_stream is not None

    def _play_sentence_via_stream(self, chunk_queue) -> None:
        """Write one sentence's PCM to PortAudio; after an unrecoverable write failure the rest is dropped."""
        if self._current_stream is None:
            self._play_sentence_via_tempfile(chunk_queue)
            return
        for aligned in _align_int16_chunks(iter(chunk_queue.get, None), self.stop_event, pad_tail=False):
            try:
                self._write_pcm(aligned)
            except Exception as write_exc:
                logger.warning("PortAudio write failed, attempting stream reinit: %s", write_exc)
                if not self._recover_stream():
                    return
                with contextlib.suppress(Exception):
                    self._write_pcm(aligned)

    def _playback_worker(self) -> None:
        """Single consumer: play audio segments from the queue in order."""
        if self.output_stream is None:
            self._for_each_sentence(self._play_sentence_via_tempfile)
            return
        import numpy as _np
        try:
            from tools.voice_mode import mark_audio_output_active
        except Exception:
            mark_audio_output_active = lambda _active: None  # noqa: E731
        self._np, self._reinit_count, self._current_stream = _np, 0, self.output_stream
        mark_audio_output_active(True)
        try:
            self._for_each_sentence(self._play_sentence_via_stream)
        finally:
            mark_audio_output_active(False)

    def finish(self) -> None:
        """Send the end sentinel, then wait for playback and prefetch threads."""
        self._audio_queue.put(None)
        self._worker.join(timeout=300.0)
        for t in self._prefetch_threads:
            t.join(timeout=10.0)
        self.close_output_stream()


def stream_tts_to_speaker(
    text_queue: queue.Queue, stop_event: threading.Event, tts_done_event: threading.Event,
    display_callback: Optional[Callable[[str], None]] = None, provider: Optional[str] = None):
    """Consume text deltas from *text_queue*, cut into sentences, speak each the moment it's ready.

    A registered streaming provider plays chunked PCM; every other provider is spoken
    per-sentence via ``text_to_speech_tool``, so audio still starts on sentence one. Protocol:
    ``str`` deltas, a ``None`` sentinel = end-of-text (flush), *stop_event* aborts (barge-in),
    *tts_done_event* is **set** in ``finally`` so continuous voice mode knows playback finished."""
    tts_done_event.clear()
    origin = _origin()
    sync_pipeline: Optional[_SyncSentencePipeline] = None
    playback: Optional[_StreamerPlayback] = None
    try:
        tts_config = origin._load_tts_config()
        # Prefer a chunked streamer for low time-to-first-audio; otherwise per-sentence sync
        # synthesis (universal — edge + every non-streamer).
        from tools.tts_streaming import SentenceChunker, resolve_streaming_provider
        streamer = resolve_streaming_provider(tts_config, preferred=provider)
        stream_max_len = 0
        if streamer is None:
            sync_pipeline = _SyncSentencePipeline(stop_event)
        else:
            with contextlib.suppress(Exception):
                stream_max_len = origin._resolve_max_text_length(
                    provider or origin._get_provider(tts_config), tts_config)
            playback = _StreamerPlayback(streamer, stop_event)
        chunker = SentenceChunker()
        spoken_sentences: list[str] = []  # skip duplicate/near-duplicate sentences (LLM repetition)

        def _speak_sentence(sentence: str) -> None:
            if stop_event.is_set():
                return
            cleaned = _strip_markdown_for_tts(sentence).strip()
            if not cleaned:
                return
            cleaned_lower = cleaned.lower().rstrip(".!,")
            if any(prev.lower().rstrip(".!,") == cleaned_lower for prev in spoken_sentences):
                return
            spoken_sentences.append(cleaned)
            if display_callback is not None:
                display_callback(sentence)  # raw sentence on screen before TTS processing
            if sync_pipeline is not None:
                sync_pipeline.speak(cleaned)
                return
            if stream_max_len and len(cleaned) > stream_max_len:
                cleaned = cleaned[:stream_max_len]
            playback.speak(cleaned)
        while not stop_event.is_set():
            try:
                delta = text_queue.get(timeout=0.5)
            except queue.Empty:
                delta = ""  # idle producer: flush a long buffer instead of sitting on it
                sentences = chunker.flush() if len(chunker.buf) > 100 else ()
            else:
                sentences = chunker.flush() if delta is None else chunker.feed(delta)
            for sentence in sentences:
                _speak_sentence(sentence)
            if delta is None:
                break
        with contextlib.suppress(queue.Empty):
            while True:
                text_queue.get_nowait()
    except Exception as exc:
        logger.warning("Streaming TTS pipeline error: %s", exc)
    finally:
        # Flush the sync pipeline first: queued sentences finish playing (or are skipped when
        # stop_event is set) BEFORE tts_done_event fires, so continuous voice mode never reopens
        # the mic over its own voice. The end sentinel lives in finally: so an exception in the
        # text pump still lets the playback worker exit.
        if sync_pipeline is not None:
            with contextlib.suppress(Exception):
                sync_pipeline.close()
        if playback is not None:
            playback.finish()
        tts_done_event.set()

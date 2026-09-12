"""Paragraph boundaries must survive cleanup, request caps and real WAV assembly."""
import json
import os
import struct
import wave
from types import SimpleNamespace

import pytest

from tools.tts_tool_segments import (
    DEFAULT_PARAGRAPH_PAUSE_MS, PCMBlock, SpeechChunk, SpeechSegment,
    decode_segments, encode_segments, local_paragraph_pause_ms, pcm_from_samples,
    pcm_from_wav, plan_speech_chunks, write_pcm_segments,
)


def read_wav(path):
    with wave.open(str(path), 'rb') as audio:
        return audio.getparams(), audio.readframes(audio.getnframes())


@pytest.mark.parametrize('value', [-1, 10001, True, False, None, '600', 600.0, float('inf')])
def test_invalid_local_pause_is_rejected(value):
    with pytest.raises(ValueError, match='paragraph_pause_ms'):
        local_paragraph_pause_ms('piper', {'piper': {'paragraph_pause_ms': value}})


@pytest.mark.parametrize('provider', ['piper', 'kittentts', 'neutts'])
def test_pause_config_defaults_and_disable(provider):
    assert local_paragraph_pause_ms(provider, {}) == DEFAULT_PARAGRAPH_PAUSE_MS
    assert local_paragraph_pause_ms(provider, {provider: None}) == DEFAULT_PARAGRAPH_PAUSE_MS
    assert local_paragraph_pause_ms(provider, {provider: {'paragraph_pause_ms': 0}}) == 0
    assert local_paragraph_pause_ms(provider, {provider: {'paragraph_pause_ms': 137}}) == 137
    assert local_paragraph_pause_ms('openai', {'openai': {'paragraph_pause_ms': 'invalid'}}) == 0


@pytest.mark.parametrize('cap', [1, 2, 3, 7, 13, 25, 1000])
def test_chunk_caps_do_not_drop_text_or_paragraph_pauses(cap):
    text = 'First paragraph is fairly long.\n\nSecond has more words.\n\nThird.'
    chunks = plan_speech_chunks(text, cap, 137)
    assert all(isinstance(chunk, SpeechChunk) and len(chunk.text) <= cap for chunk in chunks)
    segments = [segment for chunk in chunks for segment in chunk.segments]
    assert ''.join(s.text.replace(' ', '') for s in segments) == ''.join(text.split())
    assert [s.pause_before_ms for s in segments if s.pause_before_ms] == [137, 137]
    assert segments[0].pause_before_ms == 0


def test_paragraph_pause_survives_exact_request_boundary():
    chunks = plan_speech_chunks('First.\n\nNext.', len('First.'), 125)
    assert [chunk.text for chunk in chunks] == ['First.', 'Next.']
    assert chunks[1].segments[0].pause_before_ms == 125


def test_single_paragraph_and_disabled_paths_keep_legacy_chunking():
    from tools.tts_tool_delivery import _split_text_for_tts
    for text, pause in [('One long paragraph. More words.', 600), ('First.\n\nSecond.', 0)]:
        assert plan_speech_chunks(text, 12, pause) == _split_text_for_tts(text, 12)
    assert plan_speech_chunks('', 12, 600) == []


def test_literal_pilcrow_is_not_a_control_marker():
    text = 'Literal \u200b\u00b6\u200b character.\n\nNext.'
    chunks = plan_speech_chunks(text, 1000, 100)
    assert len(chunks[0].segments) == 2
    assert '\u200b\u00b6\u200b' in chunks[0].segments[0].text


def test_segment_protocol_roundtrip():
    segments = (SpeechSegment('Hello "world".\nПривет.'), SpeechSegment('Next.', 127))
    assert decode_segments(encode_segments(segments)) == segments


@pytest.mark.parametrize('payload', ['null', '{}', '[]', '[null]', '[{"text":""}]',
                                    '[{"text":"x","pause_before_ms":true}]',
                                    '[{"text":"x","extra":1}]'])
def test_invalid_segment_protocol(payload):
    with pytest.raises(ValueError):
        decode_segments(payload)


@pytest.mark.parametrize('rate', [8000, 22050, 24000])
@pytest.mark.parametrize('width,channels', [(1, 1), (2, 1), (2, 2), (3, 1), (4, 2)])
def test_writer_uses_actual_pcm_format_and_exact_silence(tmp_path, rate, width, channels):
    frame_size = width * channels
    first = PCMBlock(b'\x11' * (3 * frame_size), rate, width, channels)
    second = PCMBlock(b'\x22' * (5 * frame_size), rate, width, channels)
    output = tmp_path / 'speech.wav'
    write_pcm_segments(str(output), [(SpeechSegment('first'), first), (SpeechSegment('second', 137), second)])
    params, data = read_wav(output)
    gap_frames = rate * 137 // 1000
    assert (params.nchannels, params.sampwidth, params.framerate) == (channels, width, rate)
    assert params.nframes == 3 + gap_frames + 5
    silence = (b'\x80' if width == 1 else b'\x00' * width) * channels
    assert data == first.data + silence * gap_frames + second.data
    assert pcm_from_wav(output.read_bytes()) == PCMBlock(data, rate, width, channels)


def test_writer_retains_pause_before_first_segment_of_later_chunk(tmp_path):
    output = tmp_path / 'later.wav'
    write_pcm_segments(str(output), [(SpeechSegment('later', 125), PCMBlock(b'\x01\x02', 8000))])
    params, data = read_wav(output)
    assert params.nframes == 1001
    assert data == b'\x00' * 2000 + b'\x01\x02'


@pytest.mark.parametrize('bad', [PCMBlock(b'\x01', 24000), PCMBlock(b'', 24000),
                               PCMBlock(b'\x01\x02', 22050), PCMBlock(b'\x01\x02', 24000, channels=2)])
def test_writer_failure_preserves_existing_output_and_removes_temporary(tmp_path, bad):
    output = tmp_path / 'speech.wav'
    output.write_bytes(b'existing output')
    before = set(tmp_path.iterdir())
    with pytest.raises(ValueError):
        write_pcm_segments(str(output), [(SpeechSegment('first'), PCMBlock(b'\x01\x02', 24000)),
                                        (SpeechSegment('bad', 100), bad)])
    assert output.read_bytes() == b'existing output'
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize('after_first', [False, True])
def test_writer_preserves_synthesis_exception_and_cleans_up(tmp_path, after_first):
    def blocks():
        if after_first:
            yield SpeechSegment('first'), PCMBlock(b'\x01\x02', 24000)
        raise RuntimeError('model failure')
    output = tmp_path / 'speech.wav'
    before = set(tmp_path.iterdir())
    with pytest.raises(RuntimeError, match='model failure'):
        write_pcm_segments(str(output), blocks())
    assert set(tmp_path.iterdir()) == before


def test_writer_rejects_empty_stream(tmp_path):
    before = set(tmp_path.iterdir())
    with pytest.raises(ValueError, match='no audio segments'):
        write_pcm_segments(str(tmp_path / 'speech.wav'), [])
    assert set(tmp_path.iterdir()) == before


def test_samples_are_little_endian_and_not_silently_multichannel():
    np = pytest.importorskip("numpy")
    block = pcm_from_samples(np.array([-2, 0, 2], dtype=np.float32), 16000)
    assert block == PCMBlock(struct.pack('<hhh', -32767, 0, 32767), 16000)
    for bad in [np.array([[1, 2]]), np.array([np.nan]), np.array([np.inf])]:
        with pytest.raises(ValueError):
            pcm_from_samples(bad)


@pytest.mark.parametrize('separator', ['\n\n', '\r\n\r\n', '\n \t\n', '\n\n\n'])
def test_cleaner_preserves_real_paragraph_boundaries(separator):
    from tools.tts_text_normalize import prepare_spoken_text
    raw = f'**First**.{separator}Second.'
    assert prepare_spoken_text(raw, preserve_paragraphs=True) == 'First.\n\nSecond.'
    assert '\n' not in prepare_spoken_text(raw)


def test_cleaner_removes_multiline_nonspoken_blocks_before_segmentation():
    from tools.tts_text_normalize import prepare_spoken_text
    raw = '<think>hidden\n\nreasoning</think>First.\n\n```py\nsecret\n\ncode\n```\n\nSecond.'
    spoken = prepare_spoken_text(raw, max_chars=None, preserve_paragraphs=True)
    assert spoken == 'First.\n\nSecond.'
    assert '\n\n' not in prepare_spoken_text('First\nSecond', preserve_paragraphs=True)


def test_kittentts_loads_once_and_preserves_generation_options(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    from tools import tts_tool_local as local
    calls, loads = [], []
    def generate(text, **kwargs):
        calls.append((text, kwargs))
        return np.full(4, 0.5, dtype=np.float32)
    config = {'voice': 'voice-name', 'speed': 1.25, 'clean_text': False}
    def load(config):
        loads.append(config)
        return SimpleNamespace(generate=generate), config['kittentts']
    monkeypatch.setattr(local, '_load_kittentts_model_for_config', load)
    segments = (SpeechSegment('First.'), SpeechSegment('Second.', 100))
    output = tmp_path / 'kitten.wav'
    local._generate_kittentts('First. Second.', str(output), {'kittentts': config}, segments=segments)
    assert len(loads) == 1
    assert calls == [(s.text, config) for s in segments]
    assert read_wav(output)[0].nframes == 4 + 2400 + 4


@pytest.mark.parametrize('cap,pause', [(5000, 125), (7, 125), (5000, 0)])
def test_real_tool_config_dispatch_and_gateway_cleanup_write_correct_audio(tmp_path, monkeypatch, cap, pause):
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.base import BasePlatformAdapter
    from tools import tts_tool, tts_tool_local

    class Adapter(BasePlatformAdapter):
        def __init__(self):
            super().__init__(PlatformConfig(enabled=True, token='test'), Platform.TELEGRAM)
        async def connect(self):
            return True
        async def disconnect(self):
            pass
        async def send(self, chat_id, content, **kwargs):
            raise AssertionError('unused')
        async def get_chat_info(self, chat_id):
            return {'id': chat_id, 'type': 'dm'}

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_SESSION_PLATFORM', '')
    model_path = tmp_path / 'test.onnx'
    model_path.touch()
    (tmp_path / 'config.yaml').write_text(
        'tts:\n  provider: piper\n  piper:\n'
        f'    voice: {model_path.as_posix()}\n    paragraph_pause_ms: {pause}\n    max_text_length: {cap}\n')
    calls, loads = [], []

    class Voice:
        def synthesize_wav(self, text, output, **kwargs):
            calls.append(text)
            output.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
            output.writeframes(b'\x01\x02' * 4)

    def load(path, **kwargs):
        loads.append(path)
        return Voice()
    monkeypatch.setattr(tts_tool, '_import_piper', lambda: SimpleNamespace(load=load))
    monkeypatch.setattr(tts_tool_local, '_piper_voice_cache', {})
    raw = '**First**.\n\nSecond.'
    prepared = Adapter().prepare_tts_text(raw)
    assert prepared == 'First.\n\nSecond.'
    result = json.loads(tts_tool.text_to_speech_tool(prepared, output_path=str(tmp_path / 'out.wav')))
    assert result.get('success'), result
    assert len(loads) == 1
    assert calls == (['First.', 'Second.'] if pause else ['First. Second.'])
    blocks = [read_wav(path) for path in result['file_paths']]
    assert sum(p.nframes for p, _ in blocks) == (8 + 8000 * pause // 1000 if pause else 4)
    assert b''.join(data for _, data in blocks) == (
        b'\x01\x02' * 4 + b'\x00' * (2 * 8000 * pause // 1000) + b'\x01\x02' * 4
        if pause else b'\x01\x02' * 4)
    assert not list(tmp_path.glob('.*.wav'))


@pytest.fixture
def fake_neutts(tmp_path, monkeypatch):
    trace = tmp_path / 'trace.jsonl'
    (tmp_path / 'neutts.py').write_text('''
import json
import os
import numpy as np
TRACE = %r
class NeuTTS:
    def record(self, event, text=None):
        with open(TRACE, 'a') as f:
            f.write(json.dumps({'event': event, 'text': text, 'pid': os.getpid()}) + '\\n')
    def __init__(self, **kwargs):
        self.record('load')
    def encode_reference(self, path):
        self.record('reference')
        return 'encoded reference'
    def infer(self, text, reference, transcript):
        assert reference == 'encoded reference'
        self.record('infer', text)
        return np.full(3, 0.5, dtype=np.float32)
''' % str(trace))
    monkeypatch.setenv('PYTHONPATH', str(tmp_path) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    audio, transcript = tmp_path / 'ref.wav', tmp_path / 'ref.txt'
    audio.touch()
    transcript.write_text('reference transcript')
    return trace, {'neutts': {'ref_audio': str(audio), 'ref_text': str(transcript)}}


def test_neutts_actual_subprocess_loads_model_and_reference_once(tmp_path, fake_neutts):
    pytest.importorskip("numpy")
    from tools.tts_tool_local import _generate_neutts
    trace, config = fake_neutts
    output = tmp_path / 'neutts.wav'
    segments = (SpeechSegment('First.'), SpeechSegment('Second.', 100))
    _generate_neutts('First. Second.', str(output), config, segments=segments)
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [event['event'] for event in events] == ['load', 'reference', 'infer', 'infer']
    assert len({event['pid'] for event in events}) == 1
    assert events[-2]['text'] == 'First.' and events[-1]['text'] == 'Second.'
    assert read_wav(output)[0].nframes == 3 + 2400 + 3
    assert not list(tmp_path.glob('.*.wav'))


def test_neutts_legacy_text_path_remains_supported(tmp_path, fake_neutts):
    pytest.importorskip("numpy")
    from tools.tts_tool_local import _generate_neutts
    trace, config = fake_neutts
    output = tmp_path / 'legacy.wav'
    _generate_neutts('One paragraph.', str(output), config)
    assert read_wav(output)[0].nframes == 3
    assert json.loads(trace.read_text().splitlines()[-1])['text'] == 'One paragraph.'


def test_neutts_rejects_invalid_segments_before_loading_model(tmp_path, fake_neutts):
    from tools.tts_tool_local import _generate_neutts
    trace, config = fake_neutts
    output = tmp_path / 'invalid.wav'
    with pytest.raises(RuntimeError, match='non-empty list'):
        _generate_neutts('', str(output), config, segments=())
    assert not output.exists() and not trace.exists()


def test_config_defaults_match_the_local_runtime_default():
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    for provider in ('piper', 'kittentts', 'neutts'):
        assert DEFAULT_CONFIG['tts'][provider]['paragraph_pause_ms'] == local_paragraph_pause_ms(provider, {})

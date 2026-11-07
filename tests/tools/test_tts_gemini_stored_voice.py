"""Stored (designed) Gemini voice IDs belong to Interactions, not prebuiltVoiceConfig."""
import base64
import io
import json
import wave
from unittest.mock import MagicMock, patch

import pytest

MODEL = 'gemini-3.8-flash-lite-tts'


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ('GEMINI_API_KEY', 'GOOGLE_API_KEY', 'GEMINI_BASE_URL'):
        monkeypatch.delenv(key, raising=False)


def _wav(rate=24000, channels=1, width=2, frames=2400):
    out = io.BytesIO()
    with wave.open(out, 'wb') as f:
        f.setnchannels(channels)
        f.setsampwidth(width)
        f.setframerate(rate)
        f.writeframes(b'\x00\x01' * (frames * channels * width // 2))
    return out.getvalue()


@pytest.fixture
def wav_bytes():
    return _wav()


class _Resp:
    """A bounded-reader-compatible response double carrying a raw body."""

    def __init__(self, status_code, body):
        self.status_code = status_code
        self.content = body if isinstance(body, bytes) else json.dumps(body).encode()


def _audio(data, mime='audio/wav'):
    return {'type': 'audio', 'mime_type': mime, 'data': data}


def _ok(wav):
    return _Resp(200, {'steps': [{'type': 'model_output', 'content': [_audio(base64.b64encode(wav).decode())]}]})


def _synth(tmp_path, response, gemini=None, name='test.wav', tts_extra=None):
    from tools.tts_tool_providers import _generate_gemini_tts
    section = {'model': MODEL, 'voice': 'voice_example', **(gemini or {})}
    out = str(tmp_path / name)
    with patch('tools.tts_tool._resolve_provider_key', return_value='test-key'), \
            patch('requests.post', return_value=response) as post:
        result = _generate_gemini_tts('Hello there.', out, {'gemini': section, **(tts_extra or {})})
    return result, post


def _stream(response, section=None):
    from tools.tts_streaming import GeminiStreamer
    section = {'voice': 'voice_example', 'model': MODEL, **(section or {})}
    streamer = GeminiStreamer({'gemini': section}, section)
    with patch('tools.tts_tool._resolve_provider_key', return_value='test-key'), \
            patch('requests.post', return_value=response) as post:
        audio = b''.join(streamer.stream('Hello there.'))
    return audio, streamer, post


# ── sync ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize('response_shape', ['steps', 'output_audio'])
def test_stored_voice_uses_interactions_preserves_wav(tmp_path, wav_bytes, response_shape):
    audio = _audio(base64.b64encode(wav_bytes).decode())
    data = {'steps': [{'type': 'model_output', 'content': [audio]}]} if response_shape == 'steps' else {'output_audio': audio}
    out, post = _synth(tmp_path, _Resp(200, data))
    assert out == str(tmp_path / 'test.wav')
    args, kw = post.call_args
    assert args[0] == 'https://generativelanguage.googleapis.com/v1beta/interactions'
    assert kw['json'] == {
        'model': MODEL,
        'input': [{'type': 'user_input', 'content': [{'type': 'text', 'text': 'Hello there.'}]}],
        'response_format': {'type': 'audio'},
        'generation_config': {'speech_config': [{'voice': 'voice_example'}]},
    }
    assert kw['headers']['x-goog-api-key'] == 'test-key'
    assert 'test-key' not in args[0]
    assert (tmp_path / 'test.wav').read_bytes() == wav_bytes


def test_builtin_voice_keeps_generate_content(tmp_path):
    response = MagicMock(status_code=200)
    response.json.return_value = {'candidates': [{'content': {'parts': [
        {'inlineData': {'data': base64.b64encode(b'\x00' * 480).decode()}}]}}]}
    _, post = _synth(tmp_path, response, {'voice': 'Kore'})
    args, kw = post.call_args
    assert args[0].endswith(f'/models/{MODEL}:generateContent')
    assert kw['json']['generationConfig']['speechConfig'] == {
        'voiceConfig': {'prebuiltVoiceConfig': {'voiceName': 'Kore'}}}


def test_persona_prompt_becomes_style_annotation_not_transcript(tmp_path, wav_bytes):
    persona = tmp_path / 'persona.md'
    persona.write_text('Warm, unhurried, quietly amused.')
    _, post = _synth(tmp_path, _ok(wav_bytes), {'persona_prompt_file': str(persona)})
    content = post.call_args.kwargs['json']['input'][0]['content']
    assert content == [{
        'type': 'text', 'text': 'Hello there.',
        'annotations': [{'type': 'speech_metadata', 'style': 'Warm, unhurried, quietly amused.'}],
    }]


def test_persona_transcript_template_is_rejected_before_request(tmp_path, wav_bytes):
    persona = tmp_path / 'persona.md'
    persona.write_text("DIRECTOR'S NOTES: calm.\n\n#### TRANSCRIPT\n{{transcript}}")
    with patch('requests.post') as post, pytest.raises(ValueError, match='transcript'):
        from tools.tts_tool_providers import _generate_gemini_tts
        with patch('tools.tts_tool._resolve_provider_key', return_value='test-key'):
            _generate_gemini_tts('Hello there.', str(tmp_path / 'test.wav'), {'gemini': {
                'model': MODEL, 'voice': 'voice_example', 'persona_prompt_file': str(persona)}})
    post.assert_not_called()


def test_length_limit_counts_transcript_plus_style(tmp_path, wav_bytes):
    persona = tmp_path / 'persona.md'
    persona.write_text('x' * 50)
    _synth(tmp_path, _ok(wav_bytes), {'persona_prompt_file': str(persona), 'max_text_length': 62})
    with pytest.raises(ValueError, match=r'63 > 62'):
        persona.write_text('x' * 51)
        _synth(tmp_path, _ok(wav_bytes), {'persona_prompt_file': str(persona), 'max_text_length': 62})


def test_last_model_output_audio_is_selected(tmp_path, wav_bytes):
    first = _wav(frames=10)
    data = {'steps': [
        {'type': 'model_output', 'content': [_audio(base64.b64encode(first).decode())]},
        {'type': 'model_output', 'content': [
            _audio(base64.b64encode(first).decode()),
            _audio(base64.b64encode(wav_bytes).decode()),
            {'type': 'text', 'text': 'done'}]},
    ]}
    _synth(tmp_path, _Resp(200, data))
    assert (tmp_path / 'test.wav').read_bytes() == wav_bytes


@pytest.mark.parametrize('data', [
    {'steps': []},
    {},
    {'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': 'no audio'}]}]},
    {'steps': [{'type': 'model_output', 'content': [_audio('')]}]},
])
def test_missing_or_empty_audio_raises(tmp_path, data):
    with pytest.raises(RuntimeError, match='empty audio'):
        _synth(tmp_path, _Resp(200, data))
    assert not (tmp_path / 'test.wav').exists()


@pytest.mark.parametrize('body', [
    b'not json',
    b'[1, 2]',
    json.dumps({'steps': ['bogus']}).encode(),
    json.dumps({'steps': [{'type': 'model_output', 'content': [_audio('@@not-base64@@')]}]}).encode(),
    json.dumps({'output_audio': {'data': 42}}).encode(),
])
def test_malformed_response_raises_runtime_error(tmp_path, body):
    with pytest.raises(RuntimeError, match='malformed'):
        _synth(tmp_path, _Resp(200, body))
    assert not (tmp_path / 'test.wav').exists()


def test_non_wav_audio_is_rejected_not_wrapped(tmp_path):
    pcm = base64.b64encode(b'\x00\x01' * 100).decode()
    with pytest.raises(RuntimeError, match='non-WAV'):
        _synth(tmp_path, _Resp(200, {'output_audio': _audio(pcm)}))
    assert not (tmp_path / 'test.wav').exists()


def test_http_error_surfaces_status_and_message(tmp_path):
    error = _Resp(404, {'error': {'message': 'Voice voice_example not found'}})
    with pytest.raises(RuntimeError, match=r'HTTP 404.*voice_example not found'):
        _synth(tmp_path, error)
    assert not (tmp_path / 'test.wav').exists()


def test_configured_base_url_is_used(tmp_path, wav_bytes):
    _, post = _synth(tmp_path, _ok(wav_bytes), {'base_url': 'https://proxy.example/gemini/'})
    args, kw = post.call_args
    assert args[0] == 'https://proxy.example/gemini/v1beta/interactions'
    assert 'X-Goog-Api-Client' not in kw['headers']


def test_non_wav_output_converts_the_provider_wav(tmp_path, wav_bytes):
    with patch('tools.tts_tool_providers._write_wav_bytes_as', side_effect=lambda wav, path: path) as write:
        out, _ = _synth(tmp_path, _ok(wav_bytes), name='voice.ogg')
    assert out == str(tmp_path / 'voice.ogg')
    write.assert_called_once_with(wav_bytes, out)


# ── streaming ────────────────────────────────────────────────────────────


@pytest.mark.parametrize('rate', [24000, 48000])
def test_streamer_yields_raw_pcm_at_wav_rate(rate):
    wav = _wav(rate=rate)
    audio, streamer, post = _stream(_ok(wav))
    assert post.call_args[0][0] == 'https://generativelanguage.googleapis.com/v1beta/interactions'
    assert post.call_args.kwargs['json']['generation_config'] == {'speech_config': [{'voice': 'voice_example'}]}
    assert audio == b'\x00\x01' * 2400
    assert streamer.sample_rate == rate


def test_streamer_uses_configured_base_url(wav_bytes):
    _, _, post = _stream(_ok(wav_bytes), {'base_url': 'https://proxy.example/gemini'})
    assert post.call_args[0][0] == 'https://proxy.example/gemini/v1beta/interactions'


@pytest.mark.parametrize('wav', [_wav(channels=2), _wav(width=1), _wav(width=4)])
def test_streamer_rejects_non_mono_int16(wav):
    with pytest.raises(RuntimeError, match='mono 16-bit'):
        _stream(_ok(wav))


def test_streamer_rejects_truncated_wav(wav_bytes):
    with pytest.raises(RuntimeError, match='truncated'):
        _stream(_ok(wav_bytes[:-101]))


def test_streamer_rejects_corrupt_wav():
    with pytest.raises(RuntimeError, match='unreadable'):
        _stream(_ok(b'RIFF\x24\x00\x00\x00WAVEjunkjunkjunk'))


def test_streamer_rejects_empty_wav():
    with pytest.raises(RuntimeError, match='no audio'):
        _stream(_ok(_wav(frames=0)))


def test_streamer_http_error_raises(wav_bytes):
    with pytest.raises(RuntimeError, match='HTTP 429'):
        _stream(_Resp(429, {'error': {'message': 'quota'}}))


def test_streamer_cap_truncates_on_frame_boundary(monkeypatch):
    import tools.tts_streaming as ts
    monkeypatch.setattr(ts, '_STREAM_SENTENCE_BYTE_CAP', 10_001)
    audio, _, _ = _stream(_ok(_wav(frames=24000)))
    assert 0 < len(audio) <= 10_001
    assert len(audio) % 2 == 0

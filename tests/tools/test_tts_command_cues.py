"""Exercise command-provider cue fallback through the real tool and subprocess.

The command copies its text input, not synthesized audio: these assertions prove
what a local speech engine receives, not engine-specific prosody.
"""
import json
import sys
from pathlib import Path

import pytest

from tools import tts_tool


@pytest.mark.parametrize(('text', 'expected'), [
    ('Hello<break time="800ms"/>world.', 'Hello, world.'),
    ('This is <emphasis level="strong">important</emphasis>.', 'This is important.'),
    ('<speak>Hello <unsupported level="high">world</unsupported>.</speak>', 'Hello world.'),
    ('Plain speech stays unchanged.', 'Plain speech stays unchanged.'),
    ('&amp;lt;break/&amp;gt;Hello', 'and lt;break/ and gt;Hello'),
])
def test_command_cues_reach_local_process(tmp_path, monkeypatch, text, expected):
    command = (
        f'"{sys.executable}" -c "import shutil,sys; '
        'shutil.copyfile(sys.argv[1],sys.argv[2])" {input_path} {output_path}'
    )
    config = {'provider': 'kokoro-local', 'providers': {'kokoro-local': {
        'type': 'command', 'command': command, 'speech_cues': True,
        'output_format': 'wav',
    }}}
    monkeypatch.setattr(tts_tool, '_load_tts_config', lambda: config)
    result = json.loads(tts_tool.text_to_speech_tool(text, str(tmp_path / 'out.wav')))
    assert result['success'], result
    assert Path(result['file_path']).read_text(encoding='utf8') == expected


@pytest.mark.parametrize(('text', 'expected'), [
    ('Hello[PAUSE]world', 'Hello, world'),
    ('[whispers]Hello[/whispers]', 'Hello'),
    ('<tag note="a > b">Hello</tag>', 'Hello'),
    ('&lt;emphasis&gt;Hello&lt;/emphasis&gt;', 'Hello'),
    ('2 < 3 and 5 > 4 [reference]', '2 < 3 and 5 > 4 [reference]'),
    ('', ''),
])
def test_cue_fallback(text, expected):
    from tools.tts_text_normalize import strip_markdown_for_tts
    assert strip_markdown_for_tts(text, command_speech_cues=True) == expected


@pytest.mark.parametrize('setting', [False, 'false', None])
def test_disabled_cues_preserve_existing_pipeline(monkeypatch, setting):
    from tools.tts_text_normalize import prepare_spoken_text
    config = {'provider': 'custom', 'providers': {'custom': {
        'command': 'unused', 'speech_cues': setting,
    }}}
    monkeypatch.setattr(tts_tool, '_load_tts_config', lambda: config)
    observed = []
    def stop(chunks, *args, **kwargs):
        observed.extend(chunks)
        raise tts_tool._ChunkFailed('captured')
    monkeypatch.setattr(tts_tool, '_synthesize_chunks', stop)
    text = '<emphasis>Hello</emphasis>'
    tts_tool.text_to_speech_tool(text)
    assert observed == [prepare_spoken_text(text, max_chars=None)]


def test_cue_failure_does_not_dispatch_raw_markup(monkeypatch):
    from tools import tts_text_normalize
    monkeypatch.setattr(tts_tool, '_load_tts_config', lambda: {
        'provider': 'custom', 'providers': {'custom': {
            'command': 'unused', 'speech_cues': True,
        }},
    })
    def broken(text):
        raise ValueError('bad cue')
    monkeypatch.setattr(tts_text_normalize, 'prepare_command_speech_cues', broken)
    result = json.loads(tts_tool.text_to_speech_tool('<break/>Hello'))
    assert not result['success']
    assert 'TTS cue preprocessing failed' in result['error']


def test_markup_only_does_not_start_provider(monkeypatch):
    monkeypatch.setattr(tts_tool, '_load_tts_config', lambda: {
        'provider': 'custom', 'providers': {'custom': {
            'command': 'unused', 'speech_cues': True,
        }},
    })
    result = json.loads(tts_tool.text_to_speech_tool('<speak></speak>'))
    assert not result['success']
    assert 'empty after TTS cleanup' in result['error']


def test_call_selected_provider_controls_cues(monkeypatch):
    config = {'provider': 'other', 'providers': {'custom': {
        'command': 'unused', 'speech_cues': True,
    }}}
    monkeypatch.setattr(tts_tool, '_load_tts_config', lambda: config)
    observed = []
    def stop(chunks, *args, **kwargs):
        observed.extend(chunks)
        raise tts_tool._ChunkFailed('captured')
    monkeypatch.setattr(tts_tool, '_synthesize_chunks', stop)
    tts_tool.text_to_speech_tool('<emphasis>Hello</emphasis>', provider='custom')
    assert observed == ['Hello']
    assert config['provider'] == 'other'

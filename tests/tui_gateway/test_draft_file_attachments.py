"""Real file staging and narrowly authorized reference expansion."""
import base64
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from tui_gateway import server
from agent.context_references import preprocess_context_references


@pytest.fixture
def session(tmp_path, monkeypatch):
    cwd = tmp_path / 'workspace'
    cwd.mkdir()
    home = tmp_path / '.hermes' / 'profiles' / 'draft-test'
    home.mkdir(parents=True)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(server, '_start_agent_build', lambda *a: None)
    record = dict(cwd=str(cwd), profile_home=str(home), attached_images=[],
                  history_lock=threading.RLock(), agent=None, agent_ready=threading.Event(),
                  history=[], running=False, session_key='draft-test', transport=None)
    monkeypatch.setitem(server._sessions, 'draft-test', record)
    return record


def attach(**params):
    response = server.dispatch({'jsonrpc': '2.0', 'id': 1, 'method': 'file.attach',
                                'params': {'session_id': 'draft-test', **params}})
    assert isinstance(response, dict)
    return response


def png_bytes():
    out = BytesIO()
    Image.new('RGB', (8, 8), 'orange').save(out, format='PNG')
    return out.getvalue()


def test_file_attach_grants_exact_context_path_without_queuing_images(session, tmp_path):
    source = tmp_path / 'notes.txt'
    source.write_text('only authorized contents')
    response = attach(path=str(source))
    assert 'error' not in response, response
    staged = response['result']
    assert session.get('file_attachment_paths') == {staged['path']}
    assert session['attached_images'] == []
    assert 'image' not in staged
    context = preprocess_context_references(staged['ref_text'], cwd=session['cwd'],
        allowed_root=session['cwd'], allowed_paths=session['file_attachment_paths'], context_length=10000)
    assert not context.warnings
    assert 'only authorized contents' in context.message
    sibling = Path(staged['path']).with_name('not-granted.txt')
    sibling.write_text('not authorized contents')
    context = preprocess_context_references(f'@file:{sibling}', cwd=session['cwd'],
        allowed_paths=session['file_attachment_paths'], context_length=10000)
    assert 'outside the allowed workspace' in context.warnings[0]
    assert 'not authorized contents' not in context.message


def test_file_attach_image_metadata_requires_signature_not_filename(session):
    image = png_bytes()
    for name, payload, expected in [('picture.bin', image, True), ('fake.png', b'not an image', False)]:
        response = attach(name=name, data_url=base64.b64encode(payload).decode())
        assert 'error' not in response, response
        result = response['result']
        if expected:
            assert result.get('image') == {'name': result['name'], 'mime_type': 'image/png'}
        else:
            assert 'image' not in result
        assert session['attached_images'] == []


@pytest.mark.parametrize('payload', [b'BMI calculation notes: retain as text.\n',
    b'BM', b'\x89PNG\r\n\x1a\n', b'\xff\xd8\xff', b'GIF89a', b'RIFF\x00\x00\x00\x00WEBP'])
def test_plain_text_bm_prefix_and_malformed_candidates_stay_generic(session, payload):
    response = attach(name='notes.txt', data_url=base64.b64encode(payload).decode())
    assert 'error' not in response, response
    result = response['result']
    assert 'image' not in result, result
    assert session['attached_images'] == []
    if payload.startswith(b'BMI'):
        context = preprocess_context_references(result['ref_text'], cwd=session['cwd'],
            allowed_paths=session['file_attachment_paths'], context_length=10000)
        assert not context.warnings
        assert payload.decode().strip() in context.message


@pytest.mark.parametrize('format,mime', [('PNG', 'image/png'), ('JPEG', 'image/jpeg'),
    ('GIF', 'image/gif'), ('BMP', 'image/bmp'), ('WEBP', 'image/webp')])
def test_supported_rasters_require_decodable_content_and_preserve_provider_formats(session, format, mime):
    from agent.image_routing import build_native_content_parts
    out = BytesIO()
    Image.new('RGB', (8, 8), 'orange').save(out, format=format)
    data = out.getvalue()
    result = attach(name='picture.bin', data_url=base64.b64encode(data).decode())['result']
    assert result['image']['mime_type'] == mime
    assert server._validate_draft_image_paths(session, [result['path']]) == [result['path']]
    parts, skipped = build_native_content_parts('describe', [result['path']])
    assert not skipped
    images = [p['image_url']['url'] for p in parts if p['type'] == 'image_url']
    expected_mime = 'image/png' if format == 'BMP' else mime
    assert len(images) == 1 and images[0].startswith(f'data:{expected_mime};base64,')
    malformed = attach(name='truncated.bin', data_url=base64.b64encode(data[:len(data) // 2]).decode())['result']
    assert 'image' not in malformed


def test_path_staging_is_bounded_and_never_overwrites(session, tmp_path, monkeypatch):
    source = tmp_path / 'large.txt'
    source.write_bytes(b'contents')
    def no_read_bytes(path):
        raise AssertionError('staging must copy in bounded chunks')
    monkeypatch.setattr(Path, 'read_bytes', no_read_bytes)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: attach(path=str(source)), range(16)))
    assert all('result' in r for r in results), results
    paths = [r['result']['path'] for r in results]
    assert len(set(paths)) == len(paths)
    assert all(Path(p).read_text() == 'contents' for p in paths)


def test_oversized_path_and_base64_leave_no_staged_bytes_or_grants(session, tmp_path, monkeypatch):
    monkeypatch.setattr(server, '_FILE_ATTACH_MAX_BYTES', 4, raising=False)
    source = tmp_path / 'big.txt'
    source.write_bytes(b'too large')
    for params in [{'path': str(source)}, {'name': 'big.txt', 'data_url': base64.b64encode(b'too large').decode()}]:
        response = attach(**params)
        assert 'error' in response, response
        assert 'too large' in response['error']['message'].lower()
    assert not session.get('file_attachment_paths')
    root = Path(session['profile_home']) / 'attachments'
    assert not root.exists() or list(root.iterdir()) == []


def test_real_turn_preprocessing_expands_granted_outside_workspace_file(session, tmp_path, monkeypatch):
    import contextvars
    from types import SimpleNamespace
    source = tmp_path / 'notes.txt'
    source.write_text('the staged payload reaches the actual prompt')
    staged = attach(path=str(source))['result']
    agent = SimpleNamespace(model='test-model', provider='test', _config_context_length=10000)
    session['agent'] = agent
    monkeypatch.setattr(server, '_wire_callbacks', lambda *a: None)
    monkeypatch.setattr(server, '_sync_bot_capabilities', lambda *a: None)
    monkeypatch.setattr(server, '_start_turn_voice', lambda: (None, False))
    st = server._TurnRun(agent, True, None, True)
    prepared = contextvars.copy_context().run(server._prepare_turn_input,
        'draft-test', session, st, staged['ref_text'], [])
    assert prepared is not None
    assert 'the staged payload reaches the actual prompt' in prepared[1]
    assert 'outside the allowed workspace' not in prepared[1]


@pytest.fixture
def submit_env(session, monkeypatch):
    from types import SimpleNamespace
    session['agent'] = SimpleNamespace()
    session['agent_ready'].set()
    for name in ('_ensure_active_session_slot', '_persist_session_row_for_submit', '_reattach_refusal',
                 '_wait_agent_for_prompt'):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    monkeypatch.setattr(server, '_session_uses_compute_host', lambda *a: False)
    monkeypatch.setattr(server, '_emit', lambda *a, **k: None)
    monkeypatch.setattr(server, '_restart_completed_failed_agent_build', lambda *a: False)
    calls = []
    def run(rid, sid, record, text, **kwargs):
        calls.append((text, kwargs))
    monkeypatch.setattr(server, '_run_prompt_submit', run)
    class InlineThread:
        def __init__(self, target, **kwargs):
            self.target = target
        def start(self):
            self.target()
        def is_alive(self):
            return False
    gateway_threading = SimpleNamespace(**vars(threading))
    gateway_threading.Thread = InlineThread
    monkeypatch.setattr(server, 'threading', gateway_threading)
    return calls


def submit(**params):
    response = server.dispatch({'jsonrpc': '2.0', 'id': 2, 'method': 'prompt.submit',
        'params': {'session_id': 'draft-test', 'text': 'deliberate prompt', **params}})
    assert isinstance(response, dict)
    return response


def test_draft_images_travel_with_only_their_submit_and_add_to_legacy(session, submit_env):
    path = attach(name='shot.png', data_url=base64.b64encode(png_bytes()).decode())['result']['path']
    session['attached_images'] = ['legacy.png']
    response = submit(draft_image_paths=[path])
    assert 'error' not in response, response
    text, kwargs = submit_env.pop()
    assert kwargs.get('draft_image_paths') == [path]
    admitted = server._admit_prompt_turn('draft-test', session, text, None, None,
                                        draft_image_paths=kwargs['draft_image_paths'])
    assert admitted[0] == ['legacy.png', path]
    assert session['attached_images'] == []
    session['running'] = False
    submit()
    text, kwargs = submit_env.pop()
    assert not kwargs.get('draft_image_paths')
    assert server._admit_prompt_turn('draft-test', session, text, None, None)[0] == []


@pytest.mark.parametrize('kind', ['ungranted', 'foreign-session', 'malformed', 'changed', 'oversized',
                                 'dimensions', 'animation',
                                 pytest.param('symlink', marks=pytest.mark.linux_only)])
def test_draft_image_authorization_refuses_before_turn_mutation(session, submit_env, tmp_path, monkeypatch, kind):
    path = attach(name='shot.png', data_url=base64.b64encode(png_bytes()).decode())['result']['path']
    paths = [path]
    if kind == 'ungranted':
        other = Path(path).with_name('other.png')
        other.write_bytes(png_bytes())
        paths = [str(other)]
    elif kind == 'foreign-session':
        session['file_attachment_paths'] = set()
    elif kind == 'malformed':
        paths = path
    elif kind == 'changed':
        Path(path).write_text('not an image anymore')
    elif kind == 'oversized':
        monkeypatch.setattr(server, '_ATTACH_BYTES_MAX_BYTES', 4)
    elif kind == 'dimensions':
        import struct
        out = BytesIO()
        Image.new('RGB', (8, 8)).save(out, format='BMP')
        data = bytearray(out.getvalue())
        struct.pack_into('<ii', data, 18, 100000, 100000)  # huge header, no huge pixel allocation
        Path(path).write_bytes(data)
    elif kind == 'animation':
        from tools.vision_tools_image_prep import _VISION_MAX_VALIDATED_FRAME_COUNT
        frames = [Image.new('RGB', (2, 2), (i, 0, 0))
                  for i in range(_VISION_MAX_VALIDATED_FRAME_COUNT + 1)]
        frames[0].save(path, format='GIF', save_all=True, append_images=frames[1:])
    elif kind == 'symlink':
        other = tmp_path / 'other.png'
        other.write_bytes(png_bytes())
        Path(path).unlink()
        Path(path).symlink_to(other)
    session['attached_images'] = ['legacy.png']
    response = submit(draft_image_paths=paths)
    assert 'error' in response, response
    assert session['attached_images'] == ['legacy.png']
    assert not session['running']
    assert submit_env == []


def test_busy_draft_images_keep_fifo_envelopes_and_never_steer_or_take_later_images(session, submit_env, monkeypatch):
    monkeypatch.setattr(server, '_load_busy_input_mode', lambda: 'steer')
    steers = []
    session['agent'].steer = lambda text: steers.append(text) or True
    paths = [attach(name='shot.png', data_url=base64.b64encode(png_bytes()).decode())['result']['path']
             for _ in range(2)]
    session['running'] = True
    for i, path in enumerate(paths):
        assert submit(text=f'queued {i}', draft_image_paths=[path])['result']['status'] == 'queued'
    assert steers == []
    assert session['attached_images'] == []
    session['attached_images'] = ['later-legacy.png']
    for i, path in enumerate(paths):
        session['running'] = False
        assert server._drain_queued_prompt(3, 'draft-test', session)
        text, kwargs = submit_env.pop()
        assert text == f'queued {i}'
        assert kwargs.get('draft_image_paths') == [path]
        assert kwargs['image_paths'] == []
        admitted = server._admit_prompt_turn('draft-test', session, text,
            kwargs['image_paths'], kwargs['queued_prompt_generation'],
            draft_image_paths=kwargs['draft_image_paths'])
        assert admitted[0] == [path]
        assert session['attached_images'] == ['later-legacy.png']


def test_busy_to_idle_race_never_restores_draft_images_to_global_queue(session, submit_env, monkeypatch):
    path = attach(name='shot.png', data_url=base64.b64encode(png_bytes()).decode())['result']['path']
    session['running'] = True
    # Finish the old turn between the busy handler's two lock acquisitions.
    class FinishingLock:
        calls = 0
        def __enter__(self):
            self.calls += 1
            if self.calls == 2:
                session['running'] = False
        def __exit__(self, *args):
            pass
    session['history_lock'] = FinishingLock()
    response = server._handle_busy_submit(3, 'draft-test', session, 'next', None,
                                           draft_image_paths=[path])
    assert session['attached_images'] == []
    assert response is None


@pytest.mark.parametrize('queued', [False, True])
def test_compute_dispatch_carries_draft_images_and_grants_separately(session, submit_env, monkeypatch, queued):
    from types import SimpleNamespace
    path = attach(name='shot.png', data_url=base64.b64encode(png_bytes()).decode())['result']['path']
    frames = []
    monkeypatch.setattr(server, '_session_uses_compute_host', lambda *a: True)
    monkeypatch.setattr(server, '_get_compute_host_supervisor', lambda *a: SimpleNamespace(
        submit_turn=lambda frame, **kwargs: frames.append(frame)))
    session['attached_images'] = ['legacy.png']
    session['running'] = queued
    response = submit(draft_image_paths=[path], queued=queued)
    assert 'error' not in response, response
    if queued:
        session['running'] = False
        server._drain_queued_prompt(3, 'draft-test', session)
    frame = frames.pop()
    assert frame.get('draft_image_paths') == [path]
    assert set(frame.get('file_attachment_paths', [])) == {path}
    assert frame['attached_images'] == ['legacy.png']
    assert session['attached_images'] == []
    assert submit_env == []


def test_compute_child_preserves_turn_local_draft_paths(session, real_turn_env, monkeypatch):
    import json
    from tui_gateway.compute_host import ComputeHost
    from io import StringIO
    path = attach(name='shot.png', data_url=base64.b64encode(png_bytes()).decode())['result']['path']
    note = attach(name='note.txt', data_url=base64.b64encode(b'compute child reference contents').decode())['result']
    frame = server._compute_host_turn_frame(2, 'draft-test', session, note['ref_text'],
                                           draft_image_paths=[path])
    session['file_attachment_paths'] = set()  # independent child's record starts without grants
    monkeypatch.setattr(server, 'threading', threading)  # the child joins the actual turn worker
    monkeypatch.setattr(server, '_ensure_session_db_row', lambda *a: True)
    monkeypatch.setattr(server, '_persist_branch_seed', lambda *a: None)
    monkeypatch.setattr(server, '_session_info', lambda *a: {})
    out = StringIO()
    host = ComputeHost(stdout=out, heartbeat_secs=0)
    try:
        host._run_real_turn(frame)
        assert session['file_attachment_paths'] == {path, note['path']}
        assert len(real_turn_env) == 1
        text, kwargs = real_turn_env[0]
        assert path in str(text)
        assert 'compute child reference contents' in str(text)
        assert path in str(kwargs['persist_user_message'])
        assert session['attached_images'] == []
        assert not session['running']
        frames = [json.loads(line) for line in out.getvalue().splitlines()]
        assert frames[-1]['type'] == 'turn.end'
        assert frames[-1]['message_count'] == 2
    finally:
        host._executor.shutdown(wait=True)


@pytest.mark.parametrize('launch', ['default', 'named'])
@pytest.mark.parametrize('root,relative', [
    ('session', 'auth.json'), ('session', 'mcp-tokens/provider.json'),
    ('session', 'skills/.hub/cache.txt'), ('default', 'auth.json'),
    ('launch', 'auth.json'), ('home', '.ssh/id_ed25519'),
])
def test_profile_source_cannot_be_laundered(session, tmp_path, monkeypatch, launch, root, relative):
    from hermes_constants import (get_hermes_home, set_hermes_home_override,
                                  reset_hermes_home_override)
    default_home = tmp_path / '.hermes'
    launch_home = default_home if launch == 'default' else default_home / 'profiles' / 'gateway'
    launch_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(launch_home))
    monkeypatch.setattr(server, '_hermes_home', launch_home)
    roots = {'session': Path(session['profile_home']), 'default': default_home,
             'launch': launch_home, 'home': tmp_path}
    secret = roots[root] / relative
    secret.parent.mkdir(parents=True, exist_ok=True)
    dummy = 'THIS_IS_DUMMY_CREDENTIAL_DATA'
    secret.write_text(dummy)
    launch_token = set_hermes_home_override(launch_home)
    try:
        response = attach(path=str(secret))
        assert get_hermes_home() == launch_home  # temporary checks must not change the next RPC's scope
        if 'result' in response:
            # Exercise the real laundering path, not just the guard in isolation.
            token = set_hermes_home_override(session['profile_home'])
            try:
                context = preprocess_context_references(response['result']['ref_text'],
                    cwd=session['cwd'], allowed_paths=session['file_attachment_paths'], context_length=10000)
            finally:
                reset_hermes_home_override(token)
            assert dummy not in context.message, 'session-profile credential copied and expanded as a granted file'
        assert 'error' in response, response
        assert not session.get('file_attachment_paths')
        assert not (Path(session['profile_home']) / 'attachments').exists()
        # Same basename outside sensitive roots remains a legitimate generic file.
        safe = Path(session['cwd']) / 'auth.json'
        safe.write_text('ordinary project data')
        attached = attach(path=str(safe))['result']
        assert attached['path'] == str(safe)
        assert not attached['uploaded']
        assert get_hermes_home() == launch_home
    finally:
        reset_hermes_home_override(launch_token)


def test_file_attach_cannot_launder_sensitive_source_into_granted_staging(session):
    secret = Path(session['profile_home']) / '.env'
    secret.write_text('API_KEY=private-test-value')
    response = attach(path=str(secret))
    assert 'error' in response, response
    assert not session.get('file_attachment_paths')
    assert not (Path(session['profile_home']) / 'attachments').exists()


@pytest.mark.linux_only
def test_replaced_draft_fifo_is_rejected_without_blocking_rpc(session, tmp_path):
    import os
    path = attach(name='shot.png', data_url=base64.b64encode(png_bytes()).decode())['result']['path']
    Path(path).unlink()
    os.mkfifo(path)
    finished = threading.Event()
    errors = []
    def validate():
        try:
            server._validate_draft_image_paths(session, [path])
        except ValueError as exc:
            errors.append(str(exc))
        finally:
            finished.set()
    thread = threading.Thread(target=validate, daemon=True)
    thread.start()
    assert finished.wait(2), 'non-regular attachment blocked image validation'
    assert errors


@pytest.fixture
def real_turn_env(session, submit_env, monkeypatch):
    from tui_gateway import prompt_turn
    from tui_gateway.method_ctx import rebind
    messages = []
    def run_conversation(text, **kwargs):
        messages.append((text, kwargs))
        return {'final_response': 'ok', 'messages': [
            *kwargs.get('conversation_history', []),
            {'role': 'user', 'content': text}, {'role': 'assistant', 'content': 'ok'}]}
    session['agent'].run_conversation = run_conversation
    session['agent'].model = 'test-model'
    session['agent'].provider = 'test'
    session['agent']._config_context_length = 10000
    monkeypatch.setattr(server, '_run_prompt_submit', rebind(prompt_turn._run_prompt_submit, vars(server)))
    for name in ('_wire_callbacks', '_sync_agent_model_with_config', '_sync_bot_capabilities',
                 '_sync_session_key_after_compress', '_after_complete_turn'):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    monkeypatch.setattr(server, '_tts_stream_begin', lambda: None)
    monkeypatch.setattr(server, '_get_usage', lambda *a: {})
    return messages


@pytest.mark.parametrize('source', ['upload', 'workspace'])
def test_real_submit_turn_reaches_agent_with_only_admitted_draft_images(
        session, real_turn_env, tmp_path, source):
    messages = real_turn_env
    workspace_image = Path(session['cwd']) / 'workspace.png'
    workspace_image.write_bytes(png_bytes())
    params = ({'path': str(workspace_image)} if source == 'workspace'
              else {'name': 'shot.png', 'data_url': base64.b64encode(png_bytes()).decode()})
    attachment = attach(**params)['result']
    assert attachment['image']['mime_type'] == 'image/png'
    path = attachment['path']
    notes = tmp_path / 'notes.txt'
    notes.write_text('real prompt reference integration')
    ref = attach(path=str(notes))['result']['ref_text']
    response = submit(text=ref + ' [[ Image 1 ]]', draft_image_paths=[path])
    assert 'error' not in response, response
    assert len(messages) == 1
    assert path in str(messages[0][0])
    assert 'real prompt reference integration' in str(messages[0][0])
    assert path in str(messages[0][1]['persist_user_message'])
    assert session['attached_images'] == []
    assert not session['running']
    submit(text='next turn')
    assert len(messages) == 2
    assert path not in str(messages[1][0])


def test_authored_draft_turns_keep_sender_and_image_snapshots_through_fifo_drain(
        session, real_turn_env, monkeypatch):
    from tools.bot_relay import DeliveryAuthor

    author = {'id': 'bot:scout', 'name': 'scout', 'is_bot': True}
    paths = [attach(name='shot.png', data_url=base64.b64encode(png_bytes()).decode())['result']['path']
             for _ in range(2)]
    seen_authors = []
    run_conversation = session['agent'].run_conversation

    def authored_run(text, *, turn_author=None, **kwargs):
        seen_authors.append(turn_author)
        return run_conversation(text, **kwargs)

    monkeypatch.setattr(session['agent'], 'run_conversation', authored_run)
    session['running'] = True
    session['inflight_turn'] = {'user': 'shared request', 'assistant': '', 'streaming': True}
    assert submit(text='shared request', queued=True, draft_image_paths=[paths[0]],
                  _turn_author=DeliveryAuthor(author))['result']['status'] == 'queued'
    assert submit(text='shared request', queued=True,
                  draft_image_paths=[paths[1]])['result']['status'] == 'queued'
    envelopes = [session['queued_prompt'], *session.get('queued_prompts', [])]
    assert [(entry['text'], entry.get('turn_author'), entry['draft_image_paths'], entry['image_paths'])
            for entry in envelopes] == [
        ('shared request', author, [paths[0]], []),
        ('shared request', None, [paths[1]], []),
    ]
    # Neither queued envelope may acquire an image pasted after its acceptance.
    later = server.dispatch({'jsonrpc': '2.0', 'id': 3, 'method': 'image.attach_bytes',
        'params': {'session_id': 'draft-test', 'data': base64.b64encode(png_bytes()).decode()}})['result']['path']
    session['running'] = False
    assert server._drain_queued_prompt(4, 'draft-test', session)
    assert seen_authors == [author, None]
    assert len(real_turn_env) == len(paths)
    for (text, kwargs), path in zip(real_turn_env, paths):
        assert path in str(text) and path in str(kwargs['persist_user_message'])
        assert all(other not in str(text) for other in [later, *paths] if other != path)
    assert session['attached_images'] == [later]
    assert not session['running'] and not session.get('queued_prompt')
    assert 'turn_author' not in session


@pytest.mark.parametrize('gate', ['continue', 'cancel', 'generation', 'closing'])
@pytest.mark.parametrize('invalid', ['changed', 'deleted'])
def test_invalid_queued_image_does_not_strand_following_turn(
        session, real_turn_env, monkeypatch, tmp_path, gate, invalid):
    messages = real_turn_env
    bad = attach(name='bad.png', data_url=base64.b64encode(png_bytes()).decode())['result']['path']
    good = attach(name='good.png', data_url=base64.b64encode(png_bytes()).decode())['result']['path']
    note = tmp_path / 'queue.txt'
    note.write_text('next legitimate queued contents')
    ref = attach(path=str(note))['result']['ref_text']
    session['running'] = True
    assert submit(text='invalid image turn', draft_image_paths=[bad], queued=True)['result']['status'] == 'queued'
    legacy = server.dispatch({'jsonrpc': '2.0', 'id': 3, 'method': 'image.attach_bytes',
        'params': {'session_id': 'draft-test', 'data': base64.b64encode(png_bytes()).decode()}})['result']['path']
    assert submit(text=ref, draft_image_paths=[good], queued=True)['result']['status'] == 'queued'
    later = server.dispatch({'jsonrpc': '2.0', 'id': 4, 'method': 'image.attach_bytes',
        'params': {'session_id': 'draft-test', 'data': base64.b64encode(png_bytes()).decode()}})['result']['path']
    if invalid == 'changed':
        Path(bad).write_text('not an image any more')
    else:
        Path(bad).unlink()
    errors = []
    def emit(event, sid, payload=None):
        if event == 'error':
            errors.append(payload)
            with session['history_lock']:
                if gate == 'cancel':
                    session['_turn_cancel_requested'] = True
                elif gate == 'generation':
                    session['_queued_prompt_generation'] = session.get('_queued_prompt_generation', 0) + 1
                elif gate == 'closing':
                    session['_closing'] = True
    monkeypatch.setattr(server, '_emit', emit)
    session['running'] = False
    server._run_post_turn_followups(5, 'draft-test', session, {}, None)
    assert len(errors) == 1
    if gate != 'continue':
        assert messages == []
        assert session['queued_prompt']['text'] == ref
        session['_turn_cancel_requested'] = False
        session['_closing'] = False
        assert server._drain_queued_prompt(6, 'draft-test', session)
    assert len(messages) == 1, 'invalid admission stranded the next eligible prompt'
    text, kwargs = messages[0]
    assert 'next legitimate queued contents' in str(text)
    assert all(path in str(text) for path in [good, legacy])
    assert all(path not in str(text) for path in [bad, later])
    assert good in str(kwargs['persist_user_message'])
    assert session['attached_images'] == [later]
    assert not session['running']
    assert not session.get('queued_prompt')


@pytest.mark.parametrize('gate', ['continue', 'compression', 'cancel', 'closing', 'running'])
def test_failed_queue_continuation_keeps_original_generation_at_handoff(
        session, real_turn_env, monkeypatch, gate):
    from tui_gateway import session_compression
    from tui_gateway.method_ctx import rebind

    bad = attach(name='bad.png', data_url=base64.b64encode(png_bytes()).decode())['result']['path']
    good = attach(name='good.png', data_url=base64.b64encode(png_bytes()).decode())['result']['path']
    server._enqueue_prompt(session, 'bad image', None, draft_image_paths=[bad])
    server._enqueue_prompt(session, 'next turn', None, draft_image_paths=[good])
    next_envelope = session['queued_prompts'][0]
    original_generation = int(session.get('_queued_prompt_generation', 0))
    session['attached_images'] = ['later-legacy.png']
    Path(bad).write_text('changed image')

    # Keep the real compression re-anchor (the shared turn fixture normally stubs it).
    monkeypatch.setattr(server, '_sync_session_key_after_compress',
                        rebind(session_compression._sync_session_key_after_compress, vars(server)))
    monkeypatch.setattr(server, '_transfer_active_session_slot', lambda *a, **kw: True)
    errors = []
    monkeypatch.setattr(server, '_emit', lambda event, *args, **kw:
                        errors.append(args) if event == 'error' else None)
    handoff, invalidated = threading.Event(), threading.Event()
    worker_errors = []

    def invalidate():
        try:
            assert handoff.wait(3), 'failed dispatch never reached its continuation'
            with session['history_lock']:
                if gate == 'compression':
                    session['agent'].session_id = 'rotated-draft-test'
                    server._sync_session_key_after_compress('draft-test', session,
                        clear_pending_title=False, restart_slash_worker=False)
                elif field := {'cancel': '_turn_cancel_requested',
                               'closing': '_closing', 'running': 'running'}.get(gate):
                    session[field] = True
        except BaseException as exc:
            worker_errors.append(exc)
        finally:
            invalidated.set()

    real_drain = server._drain_queued_prompt
    drain_calls = 0

    def pause_continuation(*args, **kwargs):
        nonlocal drain_calls
        drain_calls += 1
        if drain_calls == 2:
            # Pause after eligibility was checked, before the recursive drainer
            # claims anything. The worker must acquire the real history lock.
            handoff.set()
            assert invalidated.wait(3), 'continuation held history_lock across dispatch'
        return real_drain(*args, **kwargs)

    monkeypatch.setattr(server, '_drain_queued_prompt', pause_continuation)
    worker = threading.Thread(target=invalidate, daemon=True)
    worker.start()
    try:
        server._run_post_turn_followups(5, 'draft-test', session, {}, None)
    finally:
        handoff.set()
        worker.join(3)
    assert not worker.is_alive()
    assert not worker_errors, worker_errors
    assert drain_calls >= 2
    assert len(errors) == 1  # Real admission rejected the changed image, not a fake failure.
    if gate == 'compression':
        assert session['session_key'] == 'rotated-draft-test'
        assert session['_queued_prompt_generation'] > original_generation
    if gate != 'continue':
        assert real_turn_env == [], 'stale continuation dispatched the next envelope'
        assert session['queued_prompt'] is next_envelope
        assert bool(session['running']) == (gate == 'running')
        # A fresh drain can still use the retained envelope after invalidation.
        session['_turn_cancel_requested'] = False
        session['_closing'] = False
        session['running'] = False
        assert real_drain(6, 'draft-test', session)
    assert len(real_turn_env) == 1
    text, _ = real_turn_env[0]
    assert 'next turn' in str(text) and good in str(text)
    assert bad not in str(text) and 'later-legacy.png' not in str(text)
    assert session['attached_images'] == ['later-legacy.png']
    assert not session.get('queued_prompt')
    assert not session['running']


def test_compute_send_failure_keeps_draft_for_inline_fallback(session, submit_env, monkeypatch):
    from types import SimpleNamespace
    path = attach(name='shot.png', data_url=base64.b64encode(png_bytes()).decode())['result']['path']
    monkeypatch.setattr(server, '_session_uses_compute_host', lambda *a: True)
    def failed_send(*args, **kwargs):
        raise BrokenPipeError('test pipe failed')
    monkeypatch.setattr(server, '_get_compute_host_supervisor', lambda *a: SimpleNamespace(submit_turn=failed_send))
    session['attached_images'] = ['legacy.png']
    response = submit(draft_image_paths=[path])
    assert 'error' not in response, response
    assert submit_env[-1][1]['draft_image_paths'] == [path]
    assert session['attached_images'] == ['legacy.png']


@pytest.mark.linux_only
def test_exact_file_grant_does_not_authorize_siblings_folders_or_retargeted_symlinks(session, tmp_path):
    source = tmp_path / 'note.txt'
    source.write_text('granted')
    result = attach(path=str(source))['result']
    granted = Path(result['path'])
    other = tmp_path / 'other.txt'
    other.write_text('must never inline this')
    granted.unlink()
    granted.symlink_to(other)
    for reference in [result['ref_text'], f'@folder:{granted.parent}', f'@file:{other}']:
        ctx = preprocess_context_references(reference, cwd=session['cwd'], context_length=10000,
                                           allowed_paths=session['file_attachment_paths'])
        assert ctx.warnings
        assert 'must never inline this' not in ctx.message


def test_binary_attachment_retains_real_sandbox_cache_path_mapping(session, tmp_path, monkeypatch):
    from tools.credential_files import to_agent_visible_cache_path
    monkeypatch.setenv('TERMINAL_ENV', 'docker')
    source = tmp_path / 'document.pdf'
    source.write_bytes(b'%PDF-1.7\n\x00binary document')
    result = attach(path=str(source))['result']
    ctx = preprocess_context_references(result['ref_text'], cwd=session['cwd'], context_length=10000,
                                       allowed_paths=session['file_attachment_paths'])
    mapped = to_agent_visible_cache_path(result['path'])
    assert mapped != result['path']
    assert mapped in ctx.message
    assert not ctx.warnings

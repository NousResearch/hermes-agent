"""Structured live image capture: bounded metadata, never prose or inline pixels."""
import base64
import json
from pathlib import Path
from types import SimpleNamespace

from tools.delegation_live_log import LiveTranscriptWriter, wrap_progress_callback

# Real minimal PNG; the cache's magic-byte check and later reads run unmocked.
PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=')


def bound_writer(tmp_path, monkeypatch):
    from tools import terminal_tool
    monkeypatch.setattr(terminal_tool, '_session_cwd', {})
    cwd = tmp_path / 'child'; cwd.mkdir()
    terminal_tool.record_session_cwd('child-task', str(cwd))
    child = SimpleNamespace(session_id='durable-child', _current_task_id='child-task')
    writer = LiveTranscriptWriter('images', 0, 'ignore MEDIA:/prompt.png', root=tmp_path / 'logs')
    writer.bind_child(child)
    return writer, child, cwd


def test_capture_structured_fields_before_log_truncation_with_child_cwd(tmp_path, monkeypatch):
    writer, child, cwd = bound_writer(tmp_path, monkeypatch)
    cb = wrap_progress_callback(None, writer)
    cb('tool.started', 'vision_analyze', 'short preview', {
        'question': 'MEDIA:/prompt.png', 'image_url': 'original.png'})
    cb('tool.completed', 'image_generate', result=json.dumps({
        'message': 'a' * 1000, 'images': ['render.png', 'render.png'],
        'output': 'not MEDIA:/incidental.png\nScreenshot: second.png\nMEDIA: clip.mp3',
        'page': {'images': ['/page-asset.png']}}))
    assert writer.image_snapshot() == {
        'images': [str(cwd / 'original.png'), str(cwd / 'render.png'), str(cwd / 'second.png')],
        'images_truncated': False, 'image_revision': 2}
    from tools.terminal_tool import record_session_cwd
    other = tmp_path / 'later'; other.mkdir()
    record_session_cwd(child._current_task_id, str(other))
    cb('tool.completed', 'tool', result={'image_paths': ['new.png']})
    assert writer.image_snapshot()['images'][-1] == str(other / 'new.png')
    assert 'render.png' not in writer.path.read_text()  # metadata is not scraped from the text tail


def test_native_pixels_replace_input_and_stay_outside_mounted_logs(tmp_path, monkeypatch):
    writer, _, cwd = bound_writer(tmp_path, monkeypatch)
    data_url = 'data:image/png;base64,' + base64.b64encode(PNG).decode()
    writer.observe('tool.started', 'vision_analyze', args={'image_url': 'original.png'})
    writer.observe('tool.completed', 'vision_analyze', result={
        '_multimodal': True, 'text_summary': 'Cropped image attached',
        'content': [{'type': 'text', 'text': 'Question: MEDIA:/prompt.png'},
                    {'type': 'image_url', 'image_url': {'url': data_url}}]})
    refs = writer.image_snapshot()['images']
    assert len(refs) == 1 and refs != [str(cwd / 'original.png')]
    assert Path(refs[0]).read_bytes() == PNG
    assert not Path(refs[0]).is_relative_to(writer.path.parent.parent)
    assert 'base64' not in writer.path.read_text()
    assert base64.b64encode(PNG).decode() not in writer.path.read_text()
    # Identical native bytes should not create a new file/reference on every tool result.
    writer.observe('tool.completed', 'mcp_image', result={
        'content': [{'type': 'image', 'mimeType': 'image/png', 'data': base64.b64encode(PNG).decode()}]})
    assert writer.image_snapshot()['images'] == refs


def test_untrusted_refs_and_remote_cwd_never_become_host_paths(tmp_path, monkeypatch):
    writer, _, cwd = bound_writer(tmp_path, monkeypatch)
    writer.observe('tool.completed', 'tool', result={'images': [
        'javascript:alert(1).png', 'file://foreign/share.png', 'data:image/svg+xml;base64,xxx',
        'https://example.com/a.png', '/secret.env', 'bad\nname.png', '//foreign/share.png']})
    assert writer.image_snapshot()['images'] == []
    monkeypatch.setenv('TERMINAL_ENV', 'docker')
    writer.observe('tool.started', 'vision_analyze', args={'image_url': 'sandbox.png'})
    writer.observe('tool.completed', 'tool', result={'image_path': '/sandbox/absolute.png'})
    assert writer.image_snapshot()['images'] == []


def test_task_backend_override_and_missing_cwd_fail_closed(tmp_path, monkeypatch):
    from tools import terminal_tool
    writer, child, _ = bound_writer(tmp_path, monkeypatch)
    monkeypatch.setattr(terminal_tool, '_task_env_overrides', {'child-task': {'env_type': 'docker'}})
    writer.observe('tool.completed', 'tool', result={'image_path': '/sandbox/not-host.png'})
    assert writer.image_snapshot()['images'] == []
    monkeypatch.setattr(terminal_tool, '_task_env_overrides', {})
    terminal_tool.clear_session_cwd(child._current_task_id)
    writer.observe('tool.started', 'vision_analyze', args={'image_url': 'unknown.png'})
    assert writer.image_snapshot()['images'] == []


def test_marker_variants_and_malformed_native_do_not_hide_valid_pixels(tmp_path, monkeypatch):
    writer, _, cwd = bound_writer(tmp_path, monkeypatch)
    writer.observe('tool.completed', 'browser_exec', result={
        'stdout': 'Screenshot saved to "screen with spaces.png"\nMEDIA: "quoted.png"\nprose Screenshot: no.png'})
    assert writer.image_snapshot()['images'] == [str(cwd / 'screen with spaces.png'), str(cwd / 'quoted.png')]
    writer.observe('tool.completed', 'mcp', result={'content': [
        {'type': 'image', 'mimeType': 'image/png', 'data': 'é!'},
        {'type': 'image', 'mimeType': 'image/png', 'data': base64.b64encode(PNG).decode()}]})
    assert Path(writer.image_snapshot()['images'][-1]).read_bytes() == PNG


def test_retention_and_parser_limits_are_explicit(tmp_path, monkeypatch):
    from tools.delegation_image_refs import MAX_IMAGES, MAX_REF_CHARS, MAX_EVENT_CHARS
    writer, _, cwd = bound_writer(tmp_path, monkeypatch)
    writer.observe('tool.completed', 'tool', result={'images': [f'{i}.png' for i in range(MAX_IMAGES + 5)]})
    snap = writer.image_snapshot()
    assert len(snap['images']) == MAX_IMAGES and snap['images_truncated']
    assert len(json.dumps(snap)) < 32768
    writer.observe('tool.completed', 'tool', result={'image_path': 'x' * MAX_REF_CHARS + '.png'})
    writer.observe('tool.completed', 'tool', result=' ' * (MAX_EVENT_CHARS + 1))
    assert writer.image_snapshot()['images'] == snap['images']
    assert writer.image_snapshot()['images_truncated']


def test_transient_terminal_workdir_does_not_guess_relative_result_paths(tmp_path, monkeypatch):
    writer, _, cwd = bound_writer(tmp_path, monkeypatch)
    writer.observe('tool.started', 'terminal', args={'command': 'make image', 'workdir': '/one/off'})
    writer.observe('tool.completed', 'terminal', result={'output': 'MEDIA: plot.png'})
    assert writer.image_snapshot()['images'] == []
    # Once the ambiguous call group ends, normal recorded-child-cwd results work again.
    writer.observe('tool.started', 'terminal', args={'command': 'make image'})
    writer.observe('tool.completed', 'terminal', result={'output': 'MEDIA: normal.png'})
    assert writer.image_snapshot()['images'] == [str(cwd / 'normal.png')]


def test_native_and_wire_budgets_and_cycles_are_bounded(tmp_path, monkeypatch):
    from tools import delegation_image_refs as refs
    writer, _, _ = bound_writer(tmp_path, monkeypatch)

    def native(data):
        writer.observe('tool.completed', 'mcp', result={'content': [
            {'type': 'image', 'mimeType': 'image/png', 'data': base64.b64encode(data).decode()}]})

    monkeypatch.setattr(refs, 'MAX_NATIVE_BYTES', len(PNG) + 1)
    monkeypatch.setattr(refs, 'MAX_NATIVE_TOTAL_BYTES', len(PNG))
    native(PNG)
    native(PNG + b'x')  # per-child aggregate budget, no second file
    native(PNG + b'xx')  # per-image budget
    snap = writer.image_snapshot()
    assert len(snap['images']) == 1 and snap['images_truncated']
    assert list(Path(snap['images'][0]).parent.glob('img_*')) == [Path(snap['images'][0])]
    cyclic = {}; cyclic['result'] = cyclic
    writer.observe('tool.completed', 'tool', result=cyclic)
    assert writer.image_snapshot() == snap
    # JSON escapes make the wire budget independent of Unicode string length.
    writer.observe('tool.completed', 'tool', result={'images': ['/' + 'ü' * 900 + str(i) + '.png' for i in range(32)]})
    assert len(json.dumps(writer.image_snapshot()).encode()) < refs.MAX_REFS_WIRE_BYTES + 128


def test_matches_existing_screenshot_and_host_image_contracts(tmp_path, monkeypatch):
    writer, _, cwd = bound_writer(tmp_path, monkeypatch)
    writer.observe('tool.completed', 'browser_exec', result={
        'output': 'Screenshot path: screen.png\nMEDIA:`quoted.png`'})
    writer.observe('tool.completed', 'image_generate', result={
        'result': {'host_image': str(cwd / 'host.png'), 'meta': {'screenshot_path': str(cwd / 'meta.png')}}})
    assert writer.image_snapshot()['images'] == [str(cwd / name) for name in
                                                ('screen.png', 'quoted.png', 'host.png', 'meta.png')]


def test_reference_retention_keeps_new_work_visible(tmp_path, monkeypatch):
    from tools.delegation_image_refs import MAX_IMAGES
    writer, _, cwd = bound_writer(tmp_path, monkeypatch)
    writer.observe('tool.completed', 'tool', result={'images': [f'{i}.png' for i in range(MAX_IMAGES + 2)]})
    snapshot = writer.image_snapshot()
    assert snapshot['images'][-1] == str(cwd / f'{MAX_IMAGES + 1}.png')
    assert str(cwd / '0.png') not in snapshot['images']
    assert len(snapshot['images']) == MAX_IMAGES and snapshot['images_truncated']


def test_inline_vision_input_is_available_before_auxiliary_analysis_finishes(tmp_path, monkeypatch):
    writer, _, _ = bound_writer(tmp_path, monkeypatch)
    writer.observe('tool.started', 'vision_analyze', args={
        'image_url': 'data:image/png;base64,' + base64.b64encode(PNG).decode(), 'question':'Inspect it'})
    refs = writer.image_snapshot()['images']
    assert len(refs) == 1 and Path(refs[0]).read_bytes() == PNG
    writer.observe('tool.completed', 'vision_analyze', result={'success':True,'analysis':'A text-only auxiliary analysis'})
    assert writer.image_snapshot()['images'] == refs


def test_unsupported_image_locations_are_reported_not_silently_hidden(tmp_path, monkeypatch):
    writer, _, _ = bound_writer(tmp_path, monkeypatch)
    writer.observe('tool.completed','image_generate',result={'image_url':'https://example.org/image.png'})
    assert writer.image_snapshot() == {'images':[], 'images_truncated':True, 'image_revision':0}


def test_repeated_image_event_at_same_path_advances_revision_but_text_does_not(tmp_path, monkeypatch):
    writer, _, cwd = bound_writer(tmp_path, monkeypatch)
    event = {'output': 'MEDIA: same.png'}
    writer.observe('tool.completed', 'terminal', result=event)
    first = writer.image_snapshot()
    writer.assistant_text('More progress, no new image')
    assert writer.image_snapshot() == first
    writer.observe('tool.completed', 'terminal', result=event)
    second = writer.image_snapshot()
    assert second['images'] == first['images'] == [str(cwd / 'same.png')]
    assert second['image_revision'] > first['image_revision']

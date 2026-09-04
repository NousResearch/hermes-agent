"""Tests for delegate_task(action=tail) — live transcript peek."""

import json
import os
import tempfile
import weakref

import pytest

from tools.delegate_tool import (
    _active_subagents,
    _handle_tail_action,
    _register_subagent,
)


class _StubChild:
    # Weakref-able stand-in for a live child AIAgent.
    def __init__(self, transcript_path=None):
        self._live_transcript_path = transcript_path


class _StubParent:
    pass


@pytest.fixture(autouse=True)
def _clear_registry():
    _active_subagents.clear()
    yield
    _active_subagents.clear()


def _register(sid, child, **extra):
    record = {
        'subagent_id': sid,
        'parent_id': None,
        'depth': 0,
        'goal': 'tail test',
        'model': 'test-model',
        'started_at': 1000.0,
        'status': 'running',
        'tool_count': 0,
        'agent': child,
    }
    record.update(extra)
    _register_subagent(record)


# ----- Missing subagent_id -----

def test_tail_requires_subagent_id():
    payload = json.loads(_handle_tail_action(None, _StubParent(), lines=40))
    assert 'error' in payload
    assert 'subagent_id' in payload['error']


def test_tail_with_blank_subagent_id():
    payload = json.loads(_handle_tail_action('   ', _StubParent(), lines=40))
    assert 'error' in payload


# ----- Unknown / foreign -----

def test_tail_unknown_subagent_id():
    payload = json.loads(_handle_tail_action('does-not-exist', _StubParent(), lines=40))
    assert 'error' in payload
    assert 'does-not-exist' in payload['error']


def test_tail_foreign_owned_subagent_is_not_visible():
    parent_a = _StubParent()
    parent_b = _StubParent()
    child = _StubChild()
    child._delegate_parent_ref = weakref.ref(parent_a)
    _register('task-x', child)
    payload = json.loads(_handle_tail_action('task-x', parent_b, lines=10))
    assert 'error' in payload
    assert 'task-x' in payload['error']


# ----- No transcript path -----

def test_tail_child_with_no_transcript_path():
    parent = _StubParent()
    child = _StubChild(transcript_path=None)
    child._delegate_parent_ref = weakref.ref(parent)
    _register('task-orphan', child)
    payload = json.loads(_handle_tail_action('task-orphan', parent, lines=10))
    assert 'error' in payload
    assert 'no live transcript' in payload['error']


# ----- Transcript pruned -----

def test_tail_reports_missing_transcript_file(tmp_path):
    parent = _StubParent()
    missing = tmp_path / 'never-existed.log'
    child = _StubChild(transcript_path=str(missing))
    child._delegate_parent_ref = weakref.ref(parent)
    _register('task-gone', child)
    payload = json.loads(_handle_tail_action('task-gone', parent, lines=10))
    assert payload['status'] == 'transcript_missing'
    assert payload['lines'] == 0
    assert payload['content'] == ''


# ----- Happy paths -----

def _write_lines(path, n):
    path.write_text('\n'.join(f'event-{i:03d}' for i in range(n)) + '\n', encoding='utf-8')


def test_tail_returns_last_n_lines_default(tmp_path):
    parent = _StubParent()
    log = tmp_path / 'live.log'
    _write_lines(log, 200)
    child = _StubChild(transcript_path=str(log))
    child._delegate_parent_ref = weakref.ref(parent)
    _register('task-default', child)
    payload = json.loads(_handle_tail_action('task-default', parent, lines=40))
    assert payload['status'] == 'running'
    assert payload['lines_requested'] == 40
    assert payload['lines_returned'] == 40
    assert payload['lines_total'] == 200
    assert payload['lines_truncated'] == 160
    assert 'event-199' in payload['content']
    assert 'event-160' in payload['content']
    assert 'event-159' not in payload['content']
    assert 'note' in payload


def test_tail_with_smaller_explicit_lines(tmp_path):
    parent = _StubParent()
    log = tmp_path / 'live.log'
    _write_lines(log, 100)
    child = _StubChild(transcript_path=str(log))
    child._delegate_parent_ref = weakref.ref(parent)
    _register('task-small', child)
    payload = json.loads(_handle_tail_action('task-small', parent, lines=5))
    assert payload['lines_requested'] == 5
    assert payload['lines_returned'] == 5
    assert payload['lines_total'] == 100
    for i in range(95, 100):
        assert f'event-{i:03d}' in payload['content']
    assert 'event-094' not in payload['content']


def test_tail_caps_oversized_request_at_1000(tmp_path):
    parent = _StubParent()
    log = tmp_path / 'live.log'
    _write_lines(log, 5000)
    child = _StubChild(transcript_path=str(log))
    child._delegate_parent_ref = weakref.ref(parent)
    _register('task-big', child)
    payload = json.loads(_handle_tail_action('task-big', parent, lines=99999))
    assert payload['lines_requested'] == 1000
    assert payload['lines_returned'] == 1000
    assert payload['lines_total'] == 5000
    assert payload['lines_truncated'] == 4000


def test_tail_with_non_integer_lines_falls_back_to_default(tmp_path):
    parent = _StubParent()
    log = tmp_path / 'live.log'
    _write_lines(log, 10)
    child = _StubChild(transcript_path=str(log))
    child._delegate_parent_ref = weakref.ref(parent)
    _register('task-typed', child)
    payload = json.loads(_handle_tail_action('task-typed', parent, lines='twenty'))
    assert payload['lines_requested'] == 40
    assert payload['lines_returned'] == 10
    assert payload['lines_truncated'] == 0
    assert 'note' not in payload


def test_tail_does_not_unregister_child(tmp_path):
    parent = _StubParent()
    log = tmp_path / 'live.log'
    _write_lines(log, 3)
    child = _StubChild(transcript_path=str(log))
    child._delegate_parent_ref = weakref.ref(parent)
    _register('task-stable', child)
    _handle_tail_action('task-stable', parent, lines=2)
    _handle_tail_action('task-stable', parent, lines=2)
    assert 'task-stable' in _active_subagents


def test_tail_empty_file_returns_zero_lines(tmp_path):
    parent = _StubParent()
    log = tmp_path / 'empty.log'
    log.write_text('', encoding='utf-8')
    child = _StubChild(transcript_path=str(log))
    child._delegate_parent_ref = weakref.ref(parent)
    _register('task-empty', child)
    payload = json.loads(_handle_tail_action('task-empty', parent, lines=10))
    assert payload['lines_returned'] == 0
    assert payload['lines_total'] == 0
    assert payload['content'] == ''
    assert 'note' not in payload


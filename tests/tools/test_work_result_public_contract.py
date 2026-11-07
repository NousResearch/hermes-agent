"""Public enrollment contract regressions for durable background results (#149248).

These exercise existing entry points on the base as well as the candidate: a
missing result API cannot turn the intended behavioral failure into an import error.
"""
import json
from unittest.mock import patch

import pytest


def test_terminal_dispatch_preserves_literal_work_bindings(monkeypatch):
    from tools import terminal_tool as terminal

    bindings = [{'work_id': ' build ', 'incarnation': ' 01 '}]
    received = []

    def execute(**kwargs):
        received.append(kwargs)
        return json.dumps({'output': 'fixture', 'error': None})

    monkeypatch.setattr(terminal, 'terminal_tool', execute)
    result = terminal._handle_terminal({
        'command': 'fixture-not-executed', 'background': True,
        'completion_work': bindings,
    }, task_id='fixture-parent')
    assert json.loads(result)['error'] is None
    assert received[0].get('completion_work') == bindings


@pytest.mark.parametrize('kind,errno', [(FileNotFoundError, 2), (PermissionError, 13)])
def test_spawn_rejection_is_identifiable_without_replacing_os_exception(kind, errno):
    from hermes_cli import process_identity

    original = kind(errno, 'fixture rejected spawn', '/fixture/missing')
    with patch.object(process_identity.subprocess, 'Popen', side_effect=original):
        with pytest.raises(kind) as caught:
            process_identity.spawn_owned(['not-executed'], purpose='test')
    assert caught.value is original
    assert getattr(original, '_hermes_child_not_started', False), 'definite rejection needs a lifecycle receipt'

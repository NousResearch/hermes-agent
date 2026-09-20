"""Ordinary Route text recovery does not require the optional Output owner."""
import builtins
import importlib
import sys


def test_text_receipt_loads_without_output_owner(monkeypatch):
    original_import = builtins.__import__

    def without_output(name, *args, **kwargs):
        if name == 'gateway.hosted_room_artifacts':
            raise ModuleNotFoundError('Output owner is not installed', name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, 'tui_gateway.hosted_room_driver', raising=False)
    monkeypatch.setattr(builtins, '__import__', without_output)
    driver = importlib.import_module('tui_gateway.hosted_room_driver')
    identity = driver.state.TaskIdentity('room', 'task', 'thread', 'turn')
    receipt = driver._find_terminal_receipt([
        {'role': 'assistant', 'task_id': 'task', 'execution_generation': 1,
         'status': 'settled', 'message_id': 'text-reply', 'content': 'Text remains available.'}
    ], identity, 1)
    assert receipt is not None
    assert receipt.result['text'] == 'Text remains available.'

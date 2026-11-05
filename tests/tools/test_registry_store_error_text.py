"""A tool whose handler raises a storage error reports it under the class of the SQLite error the store raised."""
import json
import sqlite3

from persistence.sqlite._backend import translated
from tools.registry import ToolRegistry


def _raise_store_error(_args):
    with translated():
        raise sqlite3.OperationalError("disk I/O error")


def test_dispatch_names_a_store_error_by_its_sqlite_class():
    reg = ToolRegistry()
    reg.register(name="store_tool", toolset="core", handler=_raise_store_error, schema={
        "name": "store_tool", "description": "A store tool", "parameters": {"type": "object", "properties": {}}})

    error = json.loads(reg.dispatch("store_tool", {}))["error"]
    assert "Tool execution failed: OperationalError: disk I/O error" in error

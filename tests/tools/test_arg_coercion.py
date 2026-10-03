"""Tool-argument coercion for union-typed (anyOf) properties.

Regression tests for #123345: ``notify`` (anyOf boolean/array, no "type")
reached neither the array wrapper nor the boolean branch, so a stringified
``"true"`` failed validation downstream.
"""
from tools import arg_coercion as ac

_NOTIFY_SCHEMA = {
    "parameters": {
        "properties": {
            "notify": {
                "anyOf": [
                    {"type": "boolean"},
                    {"type": "array", "items": {"type": "string"}},
                ]
            }
        }
    }
}


def _coerce(args, monkeypatch):
    monkeypatch.setattr(ac.registry, "get_schema", lambda name: _NOTIFY_SCHEMA)
    return ac.coerce_tool_args("terminal", dict(args))


def test_anyof_boolean_string_true_coerced(monkeypatch):
    assert _coerce({"notify": "true"}, monkeypatch) == {"notify": True}


def test_anyof_boolean_string_false_coerced(monkeypatch):
    assert _coerce({"notify": "False"}, monkeypatch) == {"notify": False}


def test_anyof_boolean_native_and_list_untouched(monkeypatch):
    assert _coerce({"notify": True}, monkeypatch) == {"notify": True}
    assert _coerce({"notify": ["err"]}, monkeypatch) == {"notify": ["err"]}
    assert _coerce({"notify": "some-pattern"}, monkeypatch) == {"notify": "some-pattern"}

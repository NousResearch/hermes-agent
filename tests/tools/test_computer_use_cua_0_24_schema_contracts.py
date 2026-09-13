"""Strict CuaDriver input-schema adapter contracts."""

from __future__ import annotations


class _StrictSchemaSession:
    def __init__(self, schemas: dict[str, set[str]]) -> None:
        self._schemas = schemas
        self.calls: list[tuple[str, dict[str, object]]] = []

    def supports_input_property(self, tool: str, property_name: str) -> bool:
        return property_name in self._schemas.get(tool, set())

    def _has_tool(self, name: str) -> bool:
        return name in self._schemas

    def call_tool(self, name: str, args: dict[str, object], timeout: float = 30.0) -> dict[str, object]:
        self.calls.append((name, dict(args)))
        unexpected = set(args) - self._schemas[name]
        if unexpected:
            raise AssertionError(f"{name} received schema-forbidden fields: {sorted(unexpected)}")
        return {"isError": False, "data": {}, "structuredContent": {"effect": "confirmed"}}


def _backend(session: _StrictSchemaSession):
    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend.__new__(CuaDriverBackend)
    backend._session = session
    backend._session_id = "schema-contract-run"
    backend._snapshot_tokens = {}
    backend._active_pid = 42
    backend._active_window_id = 7
    return backend


_COMMON = {"pid", "window_id", "session"}


def test_canonical_double_click_preserves_requested_semantics_when_schema_supports_count():
    session = _StrictSchemaSession({"click": _COMMON | {"element_index", "count", "button", "modifier"}})

    result = _backend(session).click(element=5, click_count=2, button="right", modifiers=["shift"])

    assert result.ok is True
    assert session.calls == [
        ("click", {"pid": 42, "element_index": 5, "window_id": 7, "count": 2,
                   "button": "right", "modifier": ["shift"], "session": "schema-contract-run"})
    ]


def test_legacy_double_click_keeps_unmodified_left_element_behavior():
    session = _StrictSchemaSession({
        "click": _COMMON | {"element_index"},
        "double_click": _COMMON | {"element_index"},
    })

    result = _backend(session).click(element=6, click_count=2)

    assert result.ok is True
    assert session.calls == [
        ("double_click", {"pid": 42, "element_index": 6, "window_id": 7, "session": "schema-contract-run"})
    ]


def test_legacy_double_click_refuses_semantics_it_cannot_preserve_before_transport():
    session = _StrictSchemaSession({"double_click": _COMMON | {"element_index"}})

    result = _backend(session).click(element=6, click_count=2, button="middle", modifiers=["shift"])

    assert result.ok is False
    assert result.code == "double_click_unsupported"
    assert session.calls == []


def test_legacy_coordinate_double_click_refuses_screen_relative_transport():
    session = _StrictSchemaSession({"double_click": _COMMON | {"x", "y"}})

    result = _backend(session).click(x=10, y=20, click_count=2)

    assert result.ok is False
    assert result.code == "double_click_coordinate_origin_unsupported"
    assert session.calls == []


def test_element_drag_refuses_before_transport_when_schema_lacks_element_addresses():
    session = _StrictSchemaSession({"drag": _COMMON | {"from_x", "from_y", "to_x", "to_y"}})

    result = _backend(session).drag(from_element=1, to_element=9)

    assert result.ok is False
    assert result.code == "element_drag_unsupported"
    assert session.calls == []

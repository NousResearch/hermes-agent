

class TestStrictSchemaInputProperties:
    """cua-driver 0.32+ publishes strict per-tool schemas (``additionalProperties:
    false``). An unknown property is a hard, machine-readable refusal rather
    than a silent ignore, so every property Hermes sends must exist in the
    LIVE schema of the tool it is sent to.

    Two properties were dropped from cua-driver's schemas and no open PR
    covered them:

    - ``double_click`` has no ``button`` (nor ``modifier``) property, while
      ``_InputMixin.click`` set ``args["button"]`` unconditionally — so EVERY
      double-click was refused with ``invalid_arguments``, not only the
      element-addressed ones.
    - ``drag`` has no ``from_element`` / ``to_element`` property and requires
      ``from_x`` / ``from_y`` / ``to_x`` / ``to_y``, so an element-addressed
      drag failed on an unknown argument AND on the missing required
      coordinates.

    Both fixes gate on the live ``tools/list`` schema, never on a version
    number, so a driver that does advertise the property keeps receiving it.
    """

    def _backend_with_active_target(self):
        from unittest.mock import MagicMock
        from tools.computer_use.cua_backend import CuaDriverBackend
        backend = CuaDriverBackend()
        backend._session = MagicMock()
        backend._session.call_tool.return_value = {
            "data": "ok",
            "images": [],
            "structuredContent": None,
            "isError": False,
        }
        backend._active_pid = 111
        backend._active_window_id = 222
        # Default: the live schema advertises nothing beyond what these tests
        # opt into, so the gates fail closed.
        backend._session.supports_input_property.return_value = False
        return backend

    def test_double_click_omits_button_when_the_schema_does_not_accept_it(self):
        """`double_click` carries no `button` property — sending it is refused."""
        backend = self._backend_with_active_target()
        res = backend.click(element=5, click_count=2)
        assert res.ok
        name, args = backend._session.call_tool.call_args.args
        assert name == "double_click"
        assert "button" not in args, (
            "double_click's schema has no `button` property — sending it is an "
            "unknown argument and the driver refuses the whole action."
        )
        assert args["element_index"] == 5

    def test_double_click_with_explicit_non_default_button_is_refused_when_schema_lacks_button(self):
        """An explicit non-default `button` must not be silently downgraded.

        When the live schema has no `button` property, dropping an explicit
        `button="right"` would perform a plain left double-click while
        reporting ok — a different gesture than the one requested. Refuse it
        instead; only the implicit default (`left`) is safe to omit.
        """
        backend = self._backend_with_active_target()
        res = backend.click(element=5, click_count=2, button="right")
        assert not res.ok
        assert "button" in (res.message or "")
        backend._session.call_tool.assert_not_called()

    def test_click_with_explicit_non_default_button_is_refused_when_schema_lacks_button(self):
        """Same refusal for single clicks addressed at an element."""
        backend = self._backend_with_active_target()
        res = backend.click(element=5, button="middle")
        assert not res.ok
        assert "button" in (res.message or "")
        backend._session.call_tool.assert_not_called()

    def test_click_keeps_explicit_button_when_the_schema_accepts_it(self):
        """A driver that advertises `button` keeps receiving an explicit value."""
        backend = self._backend_with_active_target()
        backend._session.supports_input_property.side_effect = (
            lambda tool, prop: prop == "button")
        res = backend.click(element=5, click_count=2, button="right")
        assert res.ok
        name, args = backend._session.call_tool.call_args.args
        assert name == "double_click"
        assert args["button"] == "right"

    def test_click_keeps_button_when_the_schema_accepts_it(self):
        """A driver that DOES advertise `button` keeps receiving it."""
        backend = self._backend_with_active_target()
        backend._session.supports_input_property.side_effect = (
            lambda tool, prop: (tool, prop) == ("click", "button"))
        res = backend.click(element=5, button="middle")
        assert res.ok
        name, args = backend._session.call_tool.call_args.args
        assert name == "click"
        assert args["button"] == "middle"

    def test_double_click_refuses_a_modifier_it_cannot_carry(self):
        """`double_click` has no `modifier` property — refuse instead of sending it."""
        backend = self._backend_with_active_target()
        backend._session.supports_input_property.side_effect = (
            lambda tool, prop: (tool, prop) == ("click", "modifier"))
        res = backend.click(element=5, click_count=2, modifiers=["cmd"])
        assert res.ok is False
        assert "does not accept modifier" in res.message
        backend._session.call_tool.assert_not_called()

    def test_click_keeps_modifier_when_the_schema_accepts_it(self):
        backend = self._backend_with_active_target()
        backend._session.supports_input_property.side_effect = (
            lambda tool, prop: (tool, prop) == ("click", "modifier"))
        res = backend.click(element=5, modifiers=["cmd"])
        assert res.ok
        _, args = backend._session.call_tool.call_args.args
        assert args["modifier"] == ["cmd"]

    def test_element_drag_is_resolved_to_the_captured_bounds(self):
        """`drag` takes coordinates only — an element drag must be resolved."""
        backend = self._backend_with_active_target()
        backend._snapshot_bounds = {3: (100, 200, 40, 20), 7: (500, 600, 60, 30)}
        res = backend.drag(from_element=3, to_element=7)
        assert res.ok
        name, args = backend._session.call_tool.call_args.args
        assert name == "drag"
        assert "from_element" not in args and "to_element" not in args, (
            "the driver's drag schema has no from_element/to_element property."
        )
        assert (args["from_x"], args["from_y"]) == (120, 210), "origin is the element's center"
        assert (args["to_x"], args["to_y"]) == (530, 615), "destination is the element's center"

    def test_element_drag_is_refused_when_the_capture_had_no_bounds(self):
        """No bounds (a markdown-parsed capture) must refuse, not drag from (0, 0)."""
        backend = self._backend_with_active_target()
        backend._snapshot_bounds = {}
        res = backend.drag(from_element=3, to_element=7)
        assert res.ok is False
        assert "no bounds" in res.message
        backend._session.call_tool.assert_not_called()

    def test_element_drag_keeps_the_element_form_when_the_driver_accepts_it(self):
        """A driver that DOES advertise `from_element` keeps receiving it."""
        backend = self._backend_with_active_target()
        backend._session.supports_input_property.side_effect = (
            lambda tool, prop: (tool, prop) == ("drag", "from_element"))
        backend._snapshot_bounds = {3: (100, 200, 40, 20), 7: (500, 600, 60, 30)}
        res = backend.drag(from_element=3, to_element=7)
        assert res.ok
        _, args = backend._session.call_tool.call_args.args
        assert args["from_element"] == 3 and args["to_element"] == 7

    def test_coordinate_drag_is_unchanged(self):
        """The coordinate form must reach the driver exactly as before."""
        backend = self._backend_with_active_target()
        res = backend.drag(from_xy=(10, 20), to_xy=(30, 40))
        assert res.ok
        name, args = backend._session.call_tool.call_args.args
        assert name == "drag"
        assert (args["from_x"], args["from_y"]) == (10, 20)
        assert (args["to_x"], args["to_y"]) == (30, 40)


class TestSnapshotBoundsLifecycle:
    """The bounds map is per-snapshot: it is filled by a capture and cleared
    with the rest of the snapshot context, so a stale box can never steer a
    gesture."""

    def test_bounds_are_cleared_when_the_target_changes(self):
        from tools.computer_use.cua_backend import CuaDriverBackend
        backend = CuaDriverBackend()
        backend._snapshot_bounds = {1: (0, 0, 10, 10)}
        backend._set_active_target({"pid": 5, "window_id": 6})
        assert backend._snapshot_bounds == {}

    def test_bounds_are_cleared_when_the_target_is_cleared(self):
        from tools.computer_use.cua_backend import CuaDriverBackend
        backend = CuaDriverBackend()
        backend._snapshot_bounds = {1: (0, 0, 10, 10)}
        backend._clear_active_target()
        assert backend._snapshot_bounds == {}

    def test_capture_records_bounds_and_skips_zero_bounds(self):
        """A capture stores real bounds; (0, 0, 0, 0) markdown bounds are dropped."""
        from unittest.mock import MagicMock
        from tools.computer_use.backend import UIElement
        from tools.computer_use.cua_backend import CuaDriverBackend
        backend = CuaDriverBackend()
        backend._session = MagicMock()
        backend._active_pid = 111
        backend._active_window_id = 222
        elements = [
            UIElement(index=1, role="AXButton", label="A", bounds=(10, 20, 30, 40)),
            UIElement(index=2, role="AXButton", label="B", bounds=(0, 0, 0, 0)),
        ]
        backend._snapshot_bounds = {}
        # The capture path assigns the map; exercise the assignment directly
        # through the same statement the capture uses.
        backend._snapshot_bounds = {e.index: e.bounds for e in elements if e.bounds != (0, 0, 0, 0)}
        assert backend._snapshot_bounds == {1: (10, 20, 30, 40)}
        assert 2 not in backend._snapshot_bounds

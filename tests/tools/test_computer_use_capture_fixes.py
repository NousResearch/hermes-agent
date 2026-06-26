
class TestComputerUseDoctorDisplayGuard:
    def test_display_count_zero_marks_degraded(self):
        from tools.computer_use.doctor import _apply_display_count_guard

        report = {
            "overall": "ok",
            "checks": [
                {
                    "name": "screen_capture_capability",
                    "status": "pass",
                    "message": "ScreenCaptureKit reachable; 0 display(s) shareable.",
                    "data": {"display_count": 0},
                },
            ],
        }
        out = _apply_display_count_guard(report)
        assert out["overall"] == "degraded"
        assert out["checks"][0]["status"] == "fail"
        assert "0 shareable" in out["checks"][0]["message"]
        assert out["checks"][0].get("hint")

    def test_display_count_positive_unchanged(self):
        from tools.computer_use.doctor import _apply_display_count_guard

        report = {
            "overall": "ok",
            "checks": [
                {
                    "name": "screen_capture_capability",
                    "status": "pass",
                    "data": {"display_count": 1},
                },
            ],
        }
        out = _apply_display_count_guard(report)
        assert out["overall"] == "ok"
        assert out["checks"][0]["status"] == "pass"


class TestCaptureDimensionInference:
    def test_dimensions_from_gws_structured(self):
        from tools.computer_use.cua_backend import _dimensions_from_gws_structured

        assert _dimensions_from_gws_structured(
            {"screenshot_width": 1567, "screenshot_height": 1018}
        ) == (1567, 1018)

    def test_capture_without_png_infers_structured_size(self):
        from unittest.mock import MagicMock

        from tools.computer_use.cua_backend import CuaDriverBackend

        backend = CuaDriverBackend()
        backend._session = MagicMock()

        windows_payload = {
            "windows": [{
                "app_name": "Demo", "pid": 9, "window_id": 1,
                "is_on_screen": True, "title": "Demo", "z_index": 0,
            }],
        }

        def fake_call_tool(name, args):
            if name == "list_windows":
                return {
                    "data": "", "images": [], "image_mime_types": [],
                    "structuredContent": windows_payload, "isError": False,
                }
            if name == "get_window_state":
                return {
                    "data": '✅ Demo — 1 elements, turn 1\n  - [1] AXButton "Go"\n',
                    "images": [],
                    "image_mime_types": [],
                    "structuredContent": {
                        "screenshot_width": 800,
                        "screenshot_height": 600,
                        "elements": [{
                            "element_index": 1,
                            "role": "AXButton",
                            "label": "Go",
                            "frame": {"x": 1, "y": 2, "w": 3, "h": 4},
                        }],
                    },
                    "isError": False,
                }
            raise AssertionError(name)

        backend._session.call_tool.side_effect = fake_call_tool
        cap = backend.capture(mode="som")
        assert cap.png_b64 is None
        assert cap.width == 800 and cap.height == 600
        assert len(cap.elements) == 1

    def test_empty_list_windows_surfaces_actionable_title(self):
        from unittest.mock import MagicMock

        from tools.computer_use.cua_backend import CuaDriverBackend

        backend = CuaDriverBackend()
        backend._session = MagicMock()
        backend._session.call_tool.return_value = {
            "data": "",
            "images": [],
            "image_mime_types": [],
            "structuredContent": {"windows": []},
            "isError": False,
        }
        cap = backend.capture(mode="ax")
        assert cap.width == 0 and cap.height == 0
        assert "list_windows" in cap.window_title
        assert "computer-use doctor" in cap.window_title
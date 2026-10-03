import json
import logging

from tui_gateway.transport import serialize_frame


def test_serialize_frame_escapes_line_separator_codepoints():
    frame = serialize_frame({"id": "x", "message": "a\u2028b\u2029c"}, "test", logging.getLogger(__name__))

    assert "\u2028" not in frame
    assert "\u2029" not in frame
    assert json.loads(frame)["message"] == "a\u2028b\u2029c"


def test_serialize_frame_escapes_line_separators_in_error_fallback():
    class _Sep:
        pass

    _Sep.__name__ = "Weird\u2028Type"
    frame = serialize_frame({"id": "x", "message": _Sep()}, "test", logging.getLogger(__name__))

    assert "\u2028" not in frame
    assert "\u2029" not in frame
    assert json.loads(frame)["error"]["code"] == -32603

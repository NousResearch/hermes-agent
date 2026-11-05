"""Regression probes for snapshot provenance and window-local coordinates."""
from unittest.mock import MagicMock

from tools.computer_use.backend import UIElement
from tools.computer_use.cua_backend import CuaDriverBackend
from tools.computer_use.tool import _bounds_hints, _format_elements


def _backend():
    b = CuaDriverBackend()
    b._session = MagicMock()
    b._session.supports_input_property.return_value = True
    b._session.supports_capability.return_value = False
    b._active_pid, b._active_window_id = 42, 99
    return b


def test_explicit_element_handles_need_latest_capture():
    b = _backend()
    for token, snap, index in [('madeup:1', None, None), (None, 'madeup', 1)]:
        result, args = b._element_addressing('click', index, token, snap)
        assert not result.ok and result.code == 'stale_snapshot'
        assert args == {}
    b._snapshot_tokens, b._snapshot_id = {1: 's0001:1'}, 's0001'
    result, args = b._element_addressing('click', 1, 's0001:2', None)
    assert not result.ok and not args  # matching prefix is NOT validation
    result, args = b._element_addressing('click', 1, 's0001:1', None)
    assert result is None and args['element_token'] == 's0001:1'


def test_vision_screenshot_refusal_preserves_target_for_fallback():
    b = _backend()
    b._snapshot_tokens, b._snapshot_id = {2: 'old:2'}, 'old'
    b._session._has_tool.return_value = True
    b._session.capabilities_discovered = True
    b._session.call_tool.side_effect = [
        {'isError': True, 'data': 'permission_denied'},
        {'isError': False, 'data': '', 'images': ['image'], 'image_mime_types': ['image/png'],
         'structuredContent': {'snapshot_id': 'new'}},
    ]
    image, mime, elements, _ = b._capture_vision()
    assert image == 'image' and mime == 'image/png' and elements == []
    assert b._session.call_tool.call_args_list[1].args[1]['pid'] == 42
    assert b._session.call_tool.call_args_list[1].args[1]['window_id'] == 99
    assert b._snapshot_tokens == {} and b._snapshot_id is None


def test_retarget_disarms_old_geometry_and_exact_target_omits_stale_frame():
    b = _backend()
    b._active_frame = (0, 0, 2920, 1582)
    b._last_capture_size = (1455, 791)
    b._set_active_target({'pid': 43, 'window_id': 100, 'bounds': (50, 20, 500, 400)})
    assert b._last_capture_size is None
    assert b._active_frame == (50, 20, 500, 400)
    b._set_active_target({'pid': 43, 'window_id': 100})
    assert b._active_frame is None


def test_small_bounds_still_show_frame_downscale_and_token_in_summary():
    e = UIElement(index=0, role='Button', label='OK', bounds=(100, 100, 20, 20), element_token='s0001:0')
    scale, note = _bounds_hints([e], 1455, 791, (0, 0, 2920, 1582))
    assert scale > 2 and 'WINDOW-LOCAL' in note
    assert 'token=s0001:0' in _format_elements([e])[0]

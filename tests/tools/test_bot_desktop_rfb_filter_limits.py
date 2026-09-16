"""RFB client-message framing at the bridge: a client-declared ClientCutText length is bounded at the header
(the bridge must not buffer up to 2 GiB for a viewer that holds a ticket but no lease), and every message
type Xvnc is configured to accept must be framed, or the stream dies on it."""

import pytest

from tools.bot_desktop.rfb_filter import _MAX_CUT_TEXT, RfbClientFilter

_HANDSHAKE = b"RFB 003.008\n\x01\x01"


def clipboard_header(length):
    return b"\x06\x00\x00\x00" + length.to_bytes(4, "big", signed=True)


@pytest.mark.parametrize("length", [_MAX_CUT_TEXT + 1, -_MAX_CUT_TEXT - 1, 2**31 - 1, -(2**31)])
@pytest.mark.parametrize("holder", [False, True])
def test_oversized_clipboard_is_rejected_at_header_without_waiting_for_payload(length, holder):
    parser = RfbClientFilter(lambda: holder)
    parser.feed(_HANDSHAKE)
    header = clipboard_header(length)
    for byte in header[:-1]:
        assert parser.feed(bytes([byte])) == b""
    with pytest.raises(ValueError, match="clipboard"):
        parser.feed(header[-1:])


def set_desktop_size(width, height, screens=1):
    # RFB 7.5.x SetDesktopSize: type, pad, u16 width, u16 height, u8 nScreens, pad, then 16 bytes per screen
    # (u32 id, u16 x, u16 y, u16 w, u16 h, u32 flags).
    head = bytes([251, 0]) + width.to_bytes(2, "big") + height.to_bytes(2, "big") + bytes([screens, 0])
    return head + b"".join(i.to_bytes(4, "big") + b"\x00\x00\x00\x00" + width.to_bytes(2, "big") + height.to_bytes(2, "big")
                           + b"\x00\x00\x00\x00" for i in range(screens))


@pytest.mark.parametrize("holder", [False, True])
def test_set_desktop_size_is_framed_and_forwarded_even_to_a_watcher(holder):
    """Regression for #110039: launcher.sh passes -AcceptSetDesktopSize (the viewer fits the screen to its
    pane), but the filter had no frame for client message 251 and killed the stream with 'unknown RFB client
    message type'. It carries no input, so it passes through like SetEncodings, lease or not."""
    parser = RfbClientFilter(lambda: holder)
    parser.feed(_HANDSHAKE)
    msg = set_desktop_size(1280, 800) + b"\x03\x00" + b"\x00" * 8  # followed by a FramebufferUpdateRequest
    assert parser.feed(msg) == msg


def test_truncated_set_desktop_size_waits_for_the_rest():
    parser = RfbClientFilter(lambda: False)
    parser.feed(_HANDSHAKE)
    msg = set_desktop_size(1280, 800, screens=2)
    for byte in msg[:-1]:
        assert parser.feed(bytes([byte])) == b""
    assert parser.feed(msg[-1:]) == msg

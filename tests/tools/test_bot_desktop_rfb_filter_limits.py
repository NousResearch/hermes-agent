"""A client-declared ClientCutText length is bounded at the header: the bridge must not buffer up to
2 GiB for a viewer that holds a ticket but no lease."""

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

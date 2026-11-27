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
def test_set_desktop_size_is_framed_and_gated_like_input(holder):
    """Regression for #110039: launcher.sh passes -AcceptSetDesktopSize, but the filter had no frame for client
    message 251 and killed the stream with 'unknown RFB client message type'. It is framed now, and because it
    resizes the bot's framebuffer under a working agent it is treated like input: the lease holder's resize
    is forwarded, a watcher's is dropped while the stream (and the request that follows) stays intact."""
    parser = RfbClientFilter(lambda: holder)
    parser.feed(_HANDSHAKE)
    resize = set_desktop_size(1280, 800)
    update_request = b"\x03\x00" + b"\x00" * 8
    assert parser.feed(resize + update_request) == (resize if holder else b"") + update_request


def test_truncated_set_desktop_size_waits_for_the_rest():
    parser = RfbClientFilter(lambda: True)
    parser.feed(_HANDSHAKE)
    msg = set_desktop_size(1280, 800, screens=2)
    for byte in msg[:-1]:
        assert parser.feed(bytes([byte])) == b""
    assert parser.feed(msg[-1:]) == msg


def fence(flags=0, payload=b""):
    # Fence (rfbproto §7.4.8, TigerVNC extension): u8 type 248, 3 pad, u32 flags, u8 payload
    # length, then payload. The length byte sits at offset 8, after the flags field.
    return bytes([248, 0, 0, 0]) + flags.to_bytes(4, "big") + bytes([len(payload)]) + payload


def test_fence_flags_do_not_widen_the_frame_and_gated_input_stays_dropped():
    """A watcher (no input lease) sends a Fence carrying the Request bit (bit 31, set on
    every fence noVNC emits) followed by a KeyEvent. Correct framing reads the payload
    length at offset 8: the fence emits as 9 bytes and the KeyEvent frames as the input
    message it is, so it is dropped. Reading the flags high byte at offset 4 as the length
    instead frames 9 + 0x80 bytes, and once the stream supplies them the KeyEvent rides
    inside the forwarded blob to Xvnc: input past the lease gate."""
    parser = RfbClientFilter(lambda: False)
    parser.feed(_HANDSHAKE)
    key_event = b"\x04\x00\x61\x00\x00\x00\x00\x00"  # KeyEvent, key 'a'
    blob = fence(flags=0x80000007) + key_event
    assert parser.feed(blob) == fence(flags=0x80000007)


def test_fence_payload_is_opaque_and_framed_by_the_length_byte():
    """A Fence may carry an opaque sync payload (Xvnc echoes it in the ServerFence reply);
    it is not re-parsed as messages, so a payload that merely looks like an input message
    is still safe to forward for a watcher. The filter must emit exactly 9 + length bytes
    and must not gate or split on payload content."""
    parser = RfbClientFilter(lambda: False)
    parser.feed(_HANDSHAKE)
    key_event_shaped = b"\x04\x00\x61\x00\x00\x00\x00\x00"
    msg = fence(flags=0x80000001, payload=key_event_shaped)
    assert parser.feed(msg) == msg


def test_zero_length_fence_with_any_flags_passes_through():
    """Flags live at offsets 4-7 and must never affect the frame: a zero-length fence emits
    9 bytes whatever the flags word is."""
    for flags in (0, 0x03, 0x80000007, 0xFFFFFFFF):
        for holder in (False, True):
            parser = RfbClientFilter(lambda: holder)
            parser.feed(_HANDSHAKE)
            update_request = b"\x03\x00" + b"\x00" * 8
            assert parser.feed(fence(flags=flags) + update_request) == fence(flags=flags) + update_request


def test_fence_frames_across_feed_boundaries():
    """A fence split mid-header and mid-payload buffers until whole, then emits verbatim."""
    parser = RfbClientFilter(lambda: True)
    parser.feed(_HANDSHAKE)
    msg = fence(flags=0x80000003, payload=b"sync\x04\x00")
    for byte in msg[:-1]:
        assert parser.feed(bytes([byte])) == b""
    assert parser.feed(msg[-1:]) == msg


@pytest.mark.parametrize("holder", [False, True])
def test_qemu_extended_key_event_is_framed_and_gated_like_input(holder):
    """QEMU Extended KeyEvent (255, sub-type 0) is keyboard input with a keycode: 12 bytes,
    forwarded for the lease holder and dropped for a watcher."""
    parser = RfbClientFilter(lambda: holder)
    parser.feed(_HANDSHAKE)
    qemu = bytes([255, 0]) + b"\x00\x01" + (0x61).to_bytes(4, "big") + (0x1E).to_bytes(4, "big")
    assert len(qemu) == 12
    update_request = b"\x03\x00" + b"\x00" * 8
    assert parser.feed(qemu + update_request) == (qemu if holder else b"") + update_request


@pytest.mark.parametrize("holder", [False, True])
def test_unframeable_qemu_subtype_drops_the_stream(holder):
    """QEMU sub-messages other than Extended KeyEvent (e.g. audio, sub-type 1) have their own
    lengths this parser cannot frame, and Xvnc does not support them: the stream drops rather
    than desync. Same for the holder, a misframe is not an input path."""
    parser = RfbClientFilter(lambda: holder)
    parser.feed(_HANDSHAKE)
    with pytest.raises(ValueError, match="QEMU"):
        parser.feed(bytes([255, 1, 0, 0]))  # audio enable


@pytest.mark.parametrize("holder", [False, True])
def test_qemu_subtype_split_across_feeds(holder):
    """A websocket frame boundary can split the QEMU type byte from its sub-type:
    the parser must wait for the sub-type byte (len < 2), then apply the drop or
    framing decision to the reassembled message."""
    parser = RfbClientFilter(lambda: holder)
    parser.feed(_HANDSHAKE)
    assert parser.feed(b"\xff") == b""  # type byte alone, waiting on the sub-type
    with pytest.raises(ValueError, match="QEMU"):
        parser.feed(b"\x01")  # sub-type 1 arrives in the next feed

    parser = RfbClientFilter(lambda: holder)
    parser.feed(_HANDSHAKE)
    assert parser.feed(b"\xff") == b""
    qemu_rest = b"\x00" + b"\x00\x01" + (0x61).to_bytes(4, "big") + (0x1E).to_bytes(4, "big")
    expected = (b"\xff" + qemu_rest) if holder else b""
    assert parser.feed(qemu_rest) == expected


@pytest.mark.parametrize("holder", [False, True])
def test_rfb_33_handshake_does_not_desync_the_stream(holder):
    """A client answering "RFB 003.003" sends ClientInit right after the version (13 bytes,
    no security-type pick). Consuming a fixed 14 reads the first message byte into the
    handshake and shifts every later frame by one: a watcher's KeyEvent then rides inside
    whatever blob the desynced parser happens to forward."""
    parser = RfbClientFilter(lambda: holder)
    handshake_33 = b"RFB 003.003\n\x01"  # version + ClientInit(shared=1)
    key_event = b"\x04\x00\x61\x00\x00\x00\x00\x00"
    got = parser.feed(handshake_33 + key_event)
    assert got == handshake_33 + (key_event if holder else b"")


def test_malformed_version_drops_the_stream():
    for bad in (b"not-a-version\n", b"RFB 004.000\n", b"RFB 0x3.008\n", b"RFB 003.00x\n"):
        parser = RfbClientFilter(lambda: True)
        with pytest.raises(ValueError):
            parser.feed(bad)

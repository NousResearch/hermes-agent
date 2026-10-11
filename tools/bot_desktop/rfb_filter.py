"""Byte-level RFB client→server gate for the Bot Desktop WebSocket bridge.

noVNC's ``viewOnly`` is a UI hint; anyone holding the socket could still inject input. The bridge
parses the client stream and forwards only non-input messages from viewers that do not hold the
lease. RFB messages do not align with WebSocket frames, so this is a stateful stream parser fed
arbitrary chunks (RFC 6143 §7.5 layouts; TigerVNC's EnableContinuousUpdates 150, Fence 248 and
SetDesktopSize 251 are framed too. 150 and 248 pass through untouched since they carry no input; QEMU
Extended KeyEvent 255 is keyboard input — noVNC switches to it as soon as Xvnc advertises the
pseudo-encoding — so it is gated like KeyEvent; SetDesktopSize resizes the bot's framebuffer under the
agent (Xvnc runs -AcceptSetDesktopSize), so it is gated like input: only the lease holder may send it).

Xvnc runs ``-SecurityTypes None``, so the handshake is short: 12-byte version, then a tail that
depends on the negotiated minor (clients <3.7 leave the security choice to the server and send
``ClientInit`` right away, 13 bytes total, while 3.7+ add their security pick, 14). Reading the
declared minor matters: consuming 14 bytes against a 13-byte handshake eats the first message byte
and desyncs every message after it, which is exactly the desync this filter exists to prevent.
``ServerInit`` is server→client and never crosses this filter.
"""

from __future__ import annotations

from typing import Callable

_INPUT_TYPES = {4, 5, 6, 251, 255}  # KeyEvent, PointerEvent, ClientCutText, SetDesktopSize, QEMU Extended KeyEvent

# Fixed-length client messages: type -> total length including the type byte.
_FIXED = {
    0: 20,   # SetPixelFormat
    3: 10,   # FramebufferUpdateRequest
    4: 8,    # KeyEvent
    5: 6,    # PointerEvent
    150: 10, # EnableContinuousUpdates
}
_SET_ENCODINGS = 2
_CLIENT_CUT_TEXT = 6
_FENCE = 248
_SET_DESKTOP_SIZE = 251  # u8 type, pad, u16 width, u16 height, u8 nScreens, pad, then 16 bytes per screen
_QEMU = 255  # u8 type, u8 sub-type; sub-type 0 = Extended KeyEvent (12 bytes), the only one Xvnc accepts

# TigerVNC's default MaxCutText, and the value launcher.sh passes as ``-MaxCutText`` so Xvnc and
# the bridge agree (keep the two in sync). The length is client-declared (int32); without a cap a
# watcher with a ticket but no lease could make the bridge buffer ~2 GiB waiting for a payload.
_MAX_CUT_TEXT = 256 * 1024


def _handshake_tail(version: bytes) -> int:
    """Client bytes that follow the 12-byte version reply (RFC 6143 §7.1).

    ``ClientInit`` is always the last handshake byte. Clients negotiating <3.7
    leave the security choice to the server (TigerVNC clamps them to 3.3) and
    send it right after the version, 1 byte; 3.7+ pick from the server's
    security list first, 2 bytes.
    """
    if (
        not version.startswith(b"RFB ")
        or version[7] != ord(".")
        or version[11] != 0x0A
        or not version[4:7].isdigit()
        or not version[8:11].isdigit()
    ):
        raise ValueError(f"malformed RFB version {version!r}")
    if int(version[4:7]) != 3:
        raise ValueError(f"unsupported RFB major version {version!r}")
    return 1 if int(version[8:11]) < 7 else 2


class RfbClientFilter:
    """Feed client bytes with :meth:`feed`; get back the bytes allowed to reach Xvnc.

    ``allow_input`` is consulted per message so a lease flip mid-stream applies to the very next
    key or pointer event.
    """

    def __init__(self, allow_input: Callable[[], bool]) -> None:
        self._allow_input = allow_input
        self._buf = bytearray()
        self._version_parsed = False
        self._handshake_left = 12 + 1 + 1  # >=3.7 worst case; shrunk once the version lands

    def feed(self, chunk: bytes) -> bytes:
        self._buf += chunk
        out = bytearray()
        if self._handshake_left:
            if not self._version_parsed:
                if len(self._buf) < 12:
                    return bytes(out)
                self._handshake_left = 12 + _handshake_tail(self._buf[:12])
                self._version_parsed = True
            take = min(self._handshake_left, len(self._buf))
            if take:
                head = bytes(self._buf[:take])
                # ClientInit shared-flag: force shared so a human viewer never disconnects the agent's
                # watcher or another observer (Xvnc also runs -AlwaysShared; belt and braces at zero cost).
                if self._handshake_left - take == 0 and take >= 1:
                    head = head[:-1] + b"\x01"
                out += head
                del self._buf[:take]
                self._handshake_left -= take
            if self._handshake_left:
                return bytes(out)
        while self._buf:
            length = self._message_length()
            if length is None or len(self._buf) < length:
                break
            msg = bytes(self._buf[:length])
            del self._buf[:length]
            if msg[0] in _INPUT_TYPES and not self._allow_input():
                continue
            out += msg
        return bytes(out)

    def _message_length(self) -> int | None:
        t = self._buf[0]
        if t in _FIXED:
            return _FIXED[t]
        if t == _SET_ENCODINGS:
            if len(self._buf) < 4:
                return None
            n = int.from_bytes(self._buf[2:4], "big")
            return 4 + 4 * n
        if t == _CLIENT_CUT_TEXT:
            if len(self._buf) < 8:
                return None
            n = int.from_bytes(self._buf[4:8], "big", signed=True)
            # Extended clipboard (RFB 3.8 + TigerVNC): negative length, |n| bytes follow.
            if abs(n) > _MAX_CUT_TEXT:
                raise ValueError("clipboard message too large")
            return 8 + abs(n)
        if t == _FENCE:
            if len(self._buf) < 9:
                return None
            return 9 + self._buf[8]  # payload length byte; flags live at 4-7
        if t == _SET_DESKTOP_SIZE:
            if len(self._buf) < 8:
                return None
            return 8 + 16 * self._buf[6]
        if t == _QEMU:
            if len(self._buf) < 2:
                return None
            # Only sub-type 0 (Extended KeyEvent, 12 bytes) has a known length; other QEMU
            # sub-messages (audio, op-keyed, variable) cannot be framed here, and Xvnc does
            # not support them anyway, so the stream drops rather than risk a misframe.
            if self._buf[1] != 0:
                raise ValueError(f"unframeable QEMU client message sub-type {self._buf[1]}")
            return 12
        # Unknown client message: we cannot frame it, and forwarding blind would let an input message
        # hide behind it. Drop the rest of the stream; the viewer reconnects.
        raise ValueError(f"unknown RFB client message type {t}")

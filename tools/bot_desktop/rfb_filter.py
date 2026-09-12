"""Byte-level RFB client→server gate for the Bot Desktop WebSocket bridge.

noVNC's ``viewOnly`` is a UI hint; anyone holding the socket could still inject input. The bridge
parses the client stream and forwards only non-input messages from viewers that do not hold the
lease. RFB messages do not align with WebSocket frames, so this is a stateful stream parser fed
arbitrary chunks (RFC 6143 §7.5 layouts; TigerVNC's EnableContinuousUpdates 150 and Fence 248 pass
through untouched since they carry no input; QEMU Extended KeyEvent 255 is keyboard input — noVNC
switches to it as soon as Xvnc advertises the pseudo-encoding — so it is gated like KeyEvent).

Xvnc runs ``-SecurityTypes None``, so the handshake is fixed-size: 12-byte version, 1-byte security
choice, then ``ClientInit`` (1 byte). ``ServerInit`` is server→client and never crosses this filter.
"""

from __future__ import annotations

from typing import Callable

_INPUT_TYPES = {4, 5, 6, 255}  # KeyEvent, PointerEvent, ClientCutText, QEMU Extended KeyEvent

# Fixed-length client messages: type -> total length including the type byte.
_FIXED = {
    0: 20,   # SetPixelFormat
    3: 10,   # FramebufferUpdateRequest
    4: 8,    # KeyEvent
    5: 6,    # PointerEvent
    150: 10, # EnableContinuousUpdates
    255: 12, # QEMU client message; sub-type 0 = Extended KeyEvent (the only one noVNC sends)
}
_SET_ENCODINGS = 2
_CLIENT_CUT_TEXT = 6
_FENCE = 248


class RfbClientFilter:
    """Feed client bytes with :meth:`feed`; get back the bytes allowed to reach Xvnc.

    ``allow_input`` is consulted per message so a lease flip mid-stream applies to the very next
    key or pointer event.
    """

    def __init__(self, allow_input: Callable[[], bool]) -> None:
        self._allow_input = allow_input
        self._buf = bytearray()
        self._handshake_left = 12 + 1 + 1  # version + security type + ClientInit(shared flag)

    def feed(self, chunk: bytes) -> bytes:
        self._buf += chunk
        out = bytearray()
        if self._handshake_left:
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
            return 8 + abs(n)
        if t == _FENCE:
            if len(self._buf) < 9:
                return None
            return 9 + self._buf[8]
        # Unknown client message: we cannot frame it, and forwarding blind would let an input message
        # hide behind it. Drop the rest of the stream; the viewer reconnects.
        raise ValueError(f"unknown RFB client message type {t}")

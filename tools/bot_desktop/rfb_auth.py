"""Remote RFB authentication before the input filter takes ownership of the stream.

The upstream negotiation and viewer handshake are separate: noVNC sees None auth, while the
password and challenge response stay between the gateway and the configured RFB host.
Option B keeps upstream reads sequential: authentication finishes before the bridge pumps start
reading framebuffer data. The same negotiation serves thumbnails without coupling them to the
WebSocket viewer. This module only uses the caller's outbound connection and never listens.
"""

from __future__ import annotations

import asyncio

_VERSION = b"RFB 003.008\n"
AUTH_TIMEOUT = 10.0


def vnc_response(password: str, challenge: bytes) -> bytes:
    from cryptography.hazmat.decrepit.ciphers.algorithms import TripleDES
    from cryptography.hazmat.primitives.ciphers import Cipher, modes
    if len(challenge) != 16:
        raise ValueError("invalid RFB challenge")
    # RFB uses eight Latin-1 bytes, with each key byte's bits in reverse order.
    key = password.encode("latin-1", errors="replace")[:8].ljust(8, b"\0")
    key = bytes(int(f"{byte:08b}"[::-1], 2) for byte in key)
    encryptor = Cipher(TripleDES(key * 3), modes.ECB()).encryptor()
    return encryptor.update(challenge) + encryptor.finalize()


async def authenticate(reader, writer, password: str) -> None:
    """Leave upstream awaiting ClientInit. Bound the whole exchange, including optional failure data."""
    async with asyncio.timeout(AUTH_TIMEOUT):
        version = await reader.readexactly(12)
        if version not in (b"RFB 003.003\n", b"RFB 003.007\n", _VERSION, b"RFB 003.889\n"):
            raise ValueError("unsupported remote RFB version")
        # Apple's 3.889 server also supports the standard 3.8 negotiation.
        minor = 8 if version == b"RFB 003.889\n" else int(version[8:11])
        writer.write(f"RFB 003.{minor:03d}\n".encode("ascii"))
        await writer.drain()
        security = 2 if password else 1
        if minor == 3:
            offered = int.from_bytes(await reader.readexactly(4), "big")
            if offered != security:
                raise ValueError("remote RFB does not offer the configured authentication")
        else:
            count = (await reader.readexactly(1))[0]
            if not count or security not in await reader.readexactly(count):
                raise ValueError("remote RFB does not offer the configured authentication")
            writer.write(bytes([security]))
            await writer.drain()
        if security == 2:
            writer.write(vnc_response(password, await reader.readexactly(16)))
            await writer.drain()
        if security == 2 or minor >= 8:
            result = int.from_bytes(await reader.readexactly(4), "big")
            if result:
                # A reason is optional on real servers. Bound both time and size, and never surface
                # untrusted text that could echo credentials. Failure must not wait for server EOF.
                try:
                    async with asyncio.timeout(0.2):
                        size = int.from_bytes(await reader.readexactly(4), "big")
                        if size <= 4096:
                            await reader.readexactly(size)
                except (TimeoutError, asyncio.IncompleteReadError):
                    pass
                raise ValueError("remote RFB authentication failed")


class ViewerHandshake:
    """Strip the viewer's version and None selection before the post-auth input filter."""

    greeting = _VERSION + b"\x01\x01"

    def __init__(self):
        self._buffer = bytearray()
        self.done = False

    def feed(self, chunk: bytes) -> tuple[bytes, bytes]:
        if self.done:
            return b"", chunk
        self._buffer.extend(chunk)
        if len(self._buffer) < 13:
            return b"", b""
        if self._buffer[:13] != _VERSION + b"\x01":
            raise ValueError("invalid viewer RFB handshake")
        self.done = True
        payload = bytes(self._buffer[13:])
        self._buffer.clear()
        return b"\0\0\0\0", payload

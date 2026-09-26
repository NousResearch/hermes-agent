"""One bounded raw RFB framebuffer grab for remote Screen previews."""

from __future__ import annotations

import asyncio
import struct

from tools.bot_desktop.rfb_auth import authenticate


async def grab(endpoint: tuple[str, int], password: str):
    from PIL import Image
    from tools.bot_desktop.runtime import validate_remote_peer
    async with asyncio.timeout(10):
        reader, writer = await asyncio.open_connection(*endpoint)
        try:
            validate_remote_peer(writer.get_extra_info("peername")[0])
            await authenticate(reader, writer, password)
            writer.write(b"\x01")  # Always shared: a thumbnail must not evict a viewer.
            await writer.drain()
            init = await reader.readexactly(24)
            width, height = struct.unpack(">HH", init[:4])
            name_size = int.from_bytes(init[20:24], "big")
            if not width or not height or width * height > 16_777_216 or name_size > 65536:
                raise ValueError("remote framebuffer exceeds thumbnail limits")
            await reader.readexactly(name_size)
            # Request a known true-color format and mandatory Raw encoding only.
            pixel_format = struct.pack(">BBBBHHHBBB3x", 32, 24, 0, 1, 255, 255, 255, 16, 8, 0)
            writer.write(b"\x00\0\0\0" + pixel_format)
            writer.write(b"\x02\0\0\x01" + struct.pack(">i", 0))
            writer.write(struct.pack(">BBHHHH", 3, 0, 0, 0, width, height))
            await writer.drain()
            image = Image.new("RGB", (width, height))
            while True:
                kind = (await reader.readexactly(1))[0]
                if kind == 2:  # Bell has no body.
                    continue
                if kind == 3:
                    header = await reader.readexactly(7)
                    size = int.from_bytes(header[3:], "big")
                    if size > 256 * 1024:
                        raise ValueError("remote clipboard exceeds thumbnail limits")
                    await reader.readexactly(size)
                    continue
                if kind != 0:
                    raise ValueError("unexpected remote framebuffer message")
                header = await reader.readexactly(3)
                count = int.from_bytes(header[1:], "big")
                if not count:
                    continue
                if count > 4096:
                    raise ValueError("too many remote framebuffer rectangles")
                for _ in range(count):
                    x, y, w, h, encoding = struct.unpack(">HHHHi", await reader.readexactly(12))
                    if encoding != 0 or not w or not h or x + w > width or y + h > height:
                        raise ValueError("invalid remote framebuffer rectangle")
                    pixels = await reader.readexactly(w * h * 4)
                    image.paste(Image.frombytes("RGB", (w, h), pixels, "raw", "BGRX"), (x, y))
                return image
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

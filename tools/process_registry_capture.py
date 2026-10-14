"""Separate byte capture and the terminal's merged text projection."""
import codecs
import select
import threading
from dataclasses import dataclass, field
from typing import Literal

StreamName = Literal["stdout", "stderr"]


@dataclass
class ProcessByteCapture:
    """Bounded prefixes with stable byte offsets; overflow is explicit, never a rolling gap."""

    limit: int
    streams: dict[str, bytearray] = field(default_factory=lambda: {"stdout": bytearray(), "stderr": bytearray()})
    truncated: dict[str, bool] = field(default_factory=lambda: {"stdout": False, "stderr": False})
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def append(self, stream: StreamName, data: bytes) -> None:
        with self.lock:
            target = self.streams[stream]
            remaining = self.limit - len(target)
            target.extend(data[:remaining])
            self.truncated[stream] |= len(data) > remaining

    def read(self, stream: StreamName, cursor: int, limit: int) -> bytes:
        if cursor < 0 or limit < 0:
            raise ValueError("Byte cursor and limit must be nonnegative")
        with self.lock:
            return bytes(self.streams[stream][cursor:cursor + limit])


def read_local_streams(registry, session) -> None:
    """Drain both native pipes before completion; preserve bytes and derive terminal text.

    POSIX select lets the existing reconciliation path request a bounded final drain when
    a grandchild inherits a pipe after the direct child exits. No additional process is spawned.
    """
    proc = session.process
    streams = {proc.stdout.buffer: "stdout", proc.stderr.buffer: "stderr"}
    decoders = {name: codecs.getincrementaldecoder("utf-8")(errors="replace") for name in streams.values()}
    head_noise = {name: True for name in streams.values()}
    session._reader_selectable = True
    idle_after_exit = 0
    final_bytes = 0

    def ingest(name, data, final=False):
        session.byte_capture.append(name, data)
        text = decoders[name].decode(data, final=final)
        if head_noise[name]:
            text = registry._clean_shell_noise(text)
            head_noise[name] = not text.strip()
        if text:
            registry._ingest_output(session, text)

    while streams:
        finishing = session._reader_finish_requested.is_set()
        ready, _, _ = select.select(list(streams), [], [], 0 if finishing else 0.2)
        if not ready:
            if finishing:
                break
            if proc.poll() is not None:
                idle_after_exit += 1
            if idle_after_exit >= 3:
                break
            continue
        for pipe in ready:
            data = pipe.read1(4096)
            if not data:
                del streams[pipe]
                continue
            ingest(streams[pipe], data)
            if finishing:
                final_bytes += len(data)
        if final_bytes >= 1_000_000:
            break
        idle_after_exit = 0
    for name in decoders:
        ingest(name, b"", final=True)
    proc.wait()
    registry._finish_exited(session, proc.returncode)

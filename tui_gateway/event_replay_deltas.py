"""Delta frames in the replay ring: one entry per open text block or tool tail, not one per token.

A streamed turn is mostly ``message.delta`` / ``reasoning.delta`` chunks, so a ring that kept each
one as its own entry covered only minutes of a long turn: a reconnecting client got ``truncated``
and had to refetch the transcript while its reply was still streaming. Deltas still get a seq each
(the live wire is unchanged), but a delta that continues the ring's newest entry folds into it.
Records (``message.start``, ``tool.*``, ``message.interim``, ``message.complete``…) are kept one per
entry, and any record closes the open run, so frame order is preserved.

On replay a run yields ONE catch-up delta holding every chunk newer than the client's watermark,
stamped with the seq of its last chunk and the ``offset`` of its first: exactly the text the client
would have received chunk by chunk, so append-only clients (Ink) and the desktop's upsert by
``turn_id``/``segment``/``offset`` both see the same reply. Every text run is closed by a record
carrying the full text (``message.interim`` / ``message.complete``), so a client that never saw a
delta still ends with the complete reply. An agent terminal's run keeps only its last
:data:`TOOL_TAIL_BYTES`; the desktop re-seeds the rest from the process snapshot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

#: Event type -> payload field holding the appended text. ``thinking.delta`` is absent on purpose:
#: its frames REPLACE a status line (spinner verbs, provider waits), so concatenating them is wrong.
TEXT_FIELDS: dict[str, str] = {
    "message.delta": "text",
    "reasoning.delta": "text",
    "agent.terminal.output": "chunk",
}
#: Tool output runs keep this much of their tail; a reply's text runs keep everything.
TOOL_TAIL_BYTES = 4096
_TAIL_TRIMMED = frozenset({"agent.terminal.output"})
# Per-chunk fields that describe the chunk, not the block: they never decide run identity.
_CHUNK_FIELDS = ("rendered", "offset")


def _utf8_len(text: str) -> int:
    return len(text.encode("utf-8", errors="surrogatepass"))


def delta_key(params: dict) -> str | None:
    """The block identity of a delta frame (consecutive frames with one key form a run), else None."""
    field_name = TEXT_FIELDS.get(str(params.get("type") or ""))
    payload = params.get("payload")
    if field_name is None or not isinstance(payload, dict) or not isinstance(payload.get(field_name), str):
        return None
    block = {k: v for k, v in payload.items() if k != field_name and k not in _CHUNK_FIELDS}
    envelope = {k: v for k, v in params.items() if k not in ("payload", "seq")}
    return json.dumps([envelope, block], ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass
class DeltaRun:
    """One open block's chunks, in seq order: ``(seq, text, rendered, offset)``."""

    key: str
    params: dict
    chunks: list[tuple[int, str, str | None, int | None]] = field(default_factory=list)
    size: int = 0

    @classmethod
    def start(cls, key: str, params: dict) -> DeltaRun:
        payload = params["payload"]
        field_name = TEXT_FIELDS[params["type"]]
        base = {k: v for k, v in params.items() if k not in ("payload", "seq")}
        base["payload"] = {k: v for k, v in payload.items() if k != field_name and k not in _CHUNK_FIELDS}
        # Detached copy (json round trip, like the ring's frozen records): caller mutation cannot leak in.
        return cls(key=key, params=json.loads(json.dumps(base, ensure_ascii=False)),
                   size=_utf8_len(key))

    @property
    def last_seq(self) -> int:
        return self.chunks[-1][0]

    def add(self, seq: int, params: dict) -> int:
        """Append one chunk; returns the change in retained bytes (negative when a tail trim wins)."""
        payload = params["payload"]
        text = payload[TEXT_FIELDS[params["type"]]]
        rendered = payload.get("rendered") if isinstance(payload.get("rendered"), str) else None
        offset = payload.get("offset") if isinstance(payload.get("offset"), int) else None
        grown = _utf8_len(text) + (_utf8_len(rendered) if rendered else 0)
        self.chunks.append((seq, text, rendered, offset))
        self.size += grown
        if params["type"] not in _TAIL_TRIMMED:
            return grown
        dropped = 0
        while len(self.chunks) > 1 and self.size - dropped - _utf8_len(self.key) > TOOL_TAIL_BYTES:
            _seq, old_text, _rendered, _offset = self.chunks.pop(0)
            dropped += _utf8_len(old_text)
        self.size -= dropped
        return grown - dropped

    def catch_up(self, last_seen: int) -> tuple[int, dict] | None:
        """``(seq, params)`` of the one delta that carries every chunk newer than *last_seen*."""
        fresh = [chunk for chunk in self.chunks if chunk[0] > last_seen]
        if not fresh:
            return None
        field_name = TEXT_FIELDS[self.params["type"]]
        payload = dict(self.params["payload"])
        payload[field_name] = "".join(text for _seq, text, _rendered, _offset in fresh)
        rendered = [r for _seq, _text, r, _offset in fresh if r is not None]
        if rendered:
            payload["rendered"] = "".join(rendered)
        if fresh[0][3] is not None:
            payload["offset"] = fresh[0][3]
        # Each read is its own object graph (the ring's records thaw fresh too).
        return fresh[-1][0], json.loads(json.dumps({**self.params, "payload": payload}, ensure_ascii=False))

"""Content-addressed projections in the canonical ArtifactStore."""
from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

from workstation.artifacts import ArtifactStore


def content_reference(store: ArtifactStore, task_id: str, raw: Any, *, schema: str = "tool_result") -> dict:
    content = json.dumps(raw, ensure_ascii=False, sort_keys=True) if not isinstance(raw, (str, bytes)) else raw
    encoded = content.encode("utf-8", "surrogatepass") if isinstance(content, str) else content
    digest = hashlib.sha256(encoded).hexdigest()
    uri = f"artifact://tasks/{task_id}/{digest}.data"
    hit = store.resolve_ref(uri) is not None
    ref = store.store(task_id, f"{digest}.data", encoded, schema=schema) if not hit else None
    return {"artifact_ref": ref.ref if ref else uri, "content_hash": digest,
            "size_bytes": len(encoded), "cache_hit": hit}


def blob_references(store: ArtifactStore, task_id: str, value: Any) -> Any:
    if isinstance(value, str) and "data:" in value and value.startswith(("{", "[")):
        try:
            return blob_references(store, task_id, json.loads(value))
        except ValueError:
            pass
    if isinstance(value, str) and value.startswith("data:") and ";base64," in value:
        header, encoded = value.split(";base64,", 1)
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, base64.binascii.Error):
            return value
        ref = content_reference(store, task_id, raw, schema="multimodal_blob")
        return {"blob_ref": ref["artifact_ref"], "sha256": ref["content_hash"],
                "mime_type": header[5:], "size_bytes": len(raw)}
    if isinstance(value, dict):
        projected = {k: blob_references(store, task_id, v) for k, v in value.items()}
        # Use supplied perception evidence; never invent visual meaning from pixels.
        digest = value.get("visual_digest")
        if isinstance(digest, str) and digest:
            digest_ref = content_reference(store, task_id, digest, schema="visual_digest")["artifact_ref"]
            for part in projected.values():
                if isinstance(part, dict) and part.get("blob_ref"):
                    part["visual_digest_ref"] = digest_ref
        return projected
    if isinstance(value, list):
        return [blob_references(store, task_id, v) for v in value]
    return value


class ReadCache:
    """Deduplicate authorized reads AFTER dispatch; never bypass security or freshness.

    Content hashes, not mtime, decide hits, including same-size edits and sections.
    A fresh scoped tool read remains necessary for remote/non-versioned sources.
    """
    def __init__(self, store: ArtifactStore, task_id: str):
        self.store, self.task_id = store, task_id

    def project(self, args: dict, raw: Any) -> dict:
        ref = content_reference(self.store, self.task_id, raw, schema="read_content")
        source_hash = hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest()[:16]
        index = self.store.store(self.task_id, f"{ref['content_hash']}_{source_hash}.index.json",
                                {"source": args, "content_ref": ref["artifact_ref"]})
        return {**ref, "status": "unchanged" if ref["cache_hit"] else "ok", "summary_ref": index.ref}

    def section(self, ref: str, *, offset: int, limit: int) -> str:
        if offset < 1 or not 1 <= limit <= 2000:
            raise ValueError("Invalid bounded section")
        return "\n".join(self.store.read(ref).splitlines()[offset - 1:offset - 1 + limit])


def schema_projection(store: ArtifactStore, task_id: str, schema: dict, fingerprint: str, *, full: bool = False) -> dict:
    identity = {"schema": schema, "capability_fingerprint": fingerprint}
    ref = content_reference(store, task_id, identity, schema="tool_schema")
    projection = {"tool": schema.get("name"), "schema_hash": ref["content_hash"],
                  "schema_ref": ref["artifact_ref"], "cache_hit": ref["cache_hit"]}
    projection["capabilities"] = schema.get("parameters", {}).get("properties", {}).get("action", {}).get("enum", [])[:32]
    if full or not ref["cache_hit"]:
        projection["schema"] = schema
    return projection

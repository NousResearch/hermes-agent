"""Verified operational recipes: versioned artifacts and a small atomic index.

No task state lives here. A recipe can only be promoted by the compiler after
external read verification; a caller-supplied status is never an admission gate.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading

from workstation.artifacts import ArtifactStore

_SECRET = re.compile(r"token|apikey|authorization|cookie|password|secret|credential|accesskey|privatekey|authentication|browsersession|storagestate|localstorage|sessionstorage", re.I)


def sanitize(value):
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()
                if not _SECRET.search(re.sub(r"[^a-z]", "", str(k).lower()))
                and k not in {"reasoning", "reasoning_content", "codex_reasoning_items", "chain-of-thought"}}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        # Structured JSON inside arguments must obey the same secret boundary.
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, (dict, list)):
            return json.dumps(sanitize(parsed), sort_keys=True)
        if re.search(r"(?i)\b(?:bearer\s+\S+|(?:api[_-]?key|token|password|cookie|secret|authorization)\s*[:=]\s*[^\s,;]+)", value):
            return "[REDACTED]"
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def recipe_fingerprint(graph, scope, preflight=()):
    from tools.registry import registry
    from tools.effects import tool_contract
    tools = {}
    for step in [*sum(graph.values(), []), *preflight]:
        name = step["tool"]
        entry = registry.get_entry(name)
        effect, metadata = tool_contract(name)
        tools[name] = {"schema": sanitize(entry.schema) if entry else {}, "effect": effect.value,
                       "routes": metadata.get("routes"), "runtime_family": "hermes-work-v1"}
    def stable(value, field=""):
        if isinstance(value, dict):
            return {k: stable(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [stable(v, field) for v in value]
        if field.lower() in {"card_id", "cardid", "timestamp", "created_at", "updated_at"} and not str(value).startswith("$"):
            return "[ENTITY]"
        return value
    return digest({"graph": stable(sanitize(graph)), "scope": sanitize(scope), "tools": tools,
                   "preflight": sanitize(list(preflight))})


def require_browser_scope(args, scope):
    """Bound navigation URLs cannot escape a cached browser procedure's scope."""
    if scope.get("route") != "native_browser" or not scope.get("host"):
        return
    from urllib.parse import urlparse
    def inspect(value):
        if isinstance(value, dict):
            for k, v in value.items():
                if k in {"url", "href", "target_url", "navigate_url"} and isinstance(v, str):
                    parsed = urlparse(v)
                    if parsed.scheme not in {"http", "https"} or parsed.hostname != scope.get("host"):
                        raise ValueError("recipe_scope_mismatch: browser URL is outside verified host")
                    family = scope.get("path_family", "/")
                    pattern = re.sub(r":[A-Za-z_][A-Za-z0-9_]*", "[^/]+", re.escape(family))
                    if not re.fullmatch(pattern, parsed.path):
                        raise ValueError("recipe_scope_mismatch: browser URL is outside verified path family")
                inspect(v)
        elif isinstance(value, list):
            for v in value:
                inspect(v)
    inspect(args)


class RecipeStore:
    def __init__(self, artifacts=None, root=None):
        self.artifacts = artifacts or ArtifactStore()
        self.root = Path(root) if root else self.artifacts.root.parent / "recipes"
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "index.json"
        self._lock = threading.RLock()

    @contextmanager
    def _transaction(self):
        # OS locks are released on process death; no stale lockfile admission.
        with self._lock, (self.root / "index.lock").open("a+b") as lock:
            if os.name == "nt":
                import msvcrt
                lock.seek(0)
                if not lock.read(1):
                    lock.write(b"0")
                    lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _load_index(self):
        return json.loads(self.index.read_text(encoding="utf-8")) if self.index.exists() else {}

    def get(self, key):
        with self._transaction():
            entry = self._load_index().get(key)
            if not entry:
                return None
            body = self.artifacts.read_json(entry["ref"])
            if digest(body) != entry["sha256"]:
                raise ValueError("Recipe artifact integrity mismatch")
            return body

    def put(self, key, recipe):
        if not isinstance(key, str) or not re.fullmatch(r"[\w.-]{1,128}", key):
            raise ValueError("Invalid recipe key")
        body = sanitize({**recipe, "recipe_id": key, "schema_version": 1})
        if body.get("status") not in {"VERIFIED", "STALE", "QUARANTINED"}:
            raise ValueError("Invalid recipe status")
        sha = digest(body)
        with self._transaction():
            ref = self.artifacts.store("recipes", sha + ".json", body, schema="verified_recipe_v1")
            index = self._load_index()
            index[key] = {"ref": ref.ref, "sha256": sha, "status": body["status"],
                          "fingerprint": body.get("fingerprint")}
            temp = self.index.with_name(f"index.{os.getpid()}.{threading.get_ident()}.tmp")
            with temp.open("w", encoding="utf-8") as stream:
                json.dump(index, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            temp.replace(self.index)
        return body

    def invalidate(self, key, status="STALE"):
        body = self.get(key)
        if body:
            body["status"] = status
            body.setdefault("stats", {})["failures"] = body.get("stats", {}).get("failures", 0) + 1
            self.put(key, body)

    def promote(self, key, graph, scope, preflight, verifier_ids, fingerprint):
        old = self.get(key)
        return self.put(key, {"operation_signature": digest(sanitize(graph)), "scope": scope,
            "graph": graph, "verification": {"verifier_step_ids": verifier_ids},
            "preflight": preflight, "fingerprint": fingerprint, "status": "VERIFIED",
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "stats": {"successes": (old or {}).get("stats", {}).get("successes", 0) + 1,
                      "failures": (old or {}).get("stats", {}).get("failures", 0)}})

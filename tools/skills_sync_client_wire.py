"""Skill Sync wire model: content-addressed objects, the HTTP client, tree walks.

Independent of local skill state (no ``~/.hermes`` reads); ``tools/skills_sync_client.py``
orchestrates push/pull on top of it and re-exports these names. It also owns the remote-trust
guards the sinks share: entry names on the wire are attacker-shaped, so ``_is_safe_tree_name``
(validates every tree component), ``confined_tree_dest`` (lexical-vs-resolved confinement of
``root / rel_path`` joins before ``rmtree``/materialize), ``get_object`` content rehashing, and
the materialize link/hardlink checks all live here. Wire contract version 1:
``hsp_version`` / ``X-HSP-Object-Type`` are deployed protocol identifiers and are NOT
renamed with the product name "Skill Sync".
"""

from __future__ import annotations

import hashlib
import json
import logging
import stat as _stat
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("tools.skills_sync_client")
WIRE_VERSION = "1"
DEFAULT_MAX_OBJECT_BYTES = 26214400  # 25 MiB, mirrors capabilities default
KIND_BLOB, KIND_TREE, KIND_COMMIT = "blob", "tree", "commit"
MODE_FILE, MODE_EXEC, MODE_DIR = "file", "exec", "dir"
ARTIFACT_TYPE_SKILL = "skill"
_EXEC_BITS = _stat.S_IXUSR | _stat.S_IXGRP | _stat.S_IXOTH
# Win32 reserves these device names in EVERY directory regardless of extension ("CON.txt" is the
# console). A remote entry named one fails mkdir/write on Windows; rejecting keeps pulls clean.
_WIN32_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)})
# Remote trees are attacker-shaped; cap descent so a degenerate deep chain cannot blow the stack
# (get_object rehashing already makes true cycles impossible under honest content addressing).
_MAX_TREE_DEPTH = 40

# `sync-manifest`: per-skill opt-in is CONTENT in the object model, not a device-local flag --
# a root-level blob in the tree at refs/user/<owner>/HEAD recording {name, enabled}. The plane
# manifest is authoritative; the local `.usage.json` `sync` flag is only the editable intent
# (reconciled FROM it on pull, TO it on push). Shape MUST match gateway-gateway src/sync/manifest.ts.
SYNC_MANIFEST_ENTRY_NAME = SYNC_MANIFEST_TYPE = "sync-manifest"
SYNC_MANIFEST_VERSION = 1


# Content addressing: the wire uses the FULL 64-hex sha256 -- a different namespace from the
# truncated 16-hex local `content_hash` (skills_guard.py).
def wire_address(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def canonical_json_bytes(obj: Dict[str, Any]) -> bytes:
    """Canonical JSON for hashing: UTF-8, sorted keys, no whitespace/trailing newline; arrays already in
    contract order. Client and server MUST agree byte-for-byte or a push fails ``422 hash_mismatch``."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def build_sync_manifest_bytes(skills: Dict[str, bool]) -> bytes:
    """Canonical ``sync-manifest`` bytes for ``{name: enabled}`` (sorted by name for a stable address)."""
    return canonical_json_bytes({
        "type": SYNC_MANIFEST_TYPE, "version": SYNC_MANIFEST_VERSION,
        "skills": [{"name": name, "enabled": bool(enabled)} for name, enabled in sorted(skills.items())],
    })


def parse_sync_manifest(data: bytes) -> Optional[Dict[str, bool]]:
    """``{name: enabled}`` from ``sync-manifest`` bytes, or None if malformed. Strict (mirrors gateway
    ``parseSyncManifest``): a bad manifest must not read as "nothing opted in"."""
    try:
        value = json.loads(data.decode("utf-8"))
    except Exception:
        return None
    if (not isinstance(value, dict) or value.get("type") != SYNC_MANIFEST_TYPE
            or value.get("version") != SYNC_MANIFEST_VERSION or not isinstance(value.get("skills"), list)):
        return None
    out: Dict[str, bool] = {}
    for raw in value["skills"]:
        name, enabled = (raw.get("name"), raw.get("enabled")) if isinstance(raw, dict) else (None, None)
        if not isinstance(name, str) or not name or not isinstance(enabled, bool):
            return None
        out[name] = enabled
    return out


# Object building
class ObjectSet:
    """Objects to push, ``hash -> (kind, bytes)``, deduped by content address."""

    def __init__(self) -> None:
        self.objects: Dict[str, Tuple[str, bytes]] = {}

    def add(self, kind: str, data: bytes) -> str:
        addr = wire_address(data)
        self.objects.setdefault(addr, (kind, data))
        return addr

    def __len__(self) -> int:
        return len(self.objects)


def _entry(name: str, kind: str, hash_: str, mode: str) -> Dict[str, str]:
    return {"name": name, "kind": kind, "hash": hash_, "mode": mode}


def _add_tree(entries: List[Dict[str, str]], objects: ObjectSet) -> str:
    """Canonicalize *entries* (sorted by name, byte order) into a tree object."""
    entries.sort(key=lambda e: e["name"])
    return objects.add(KIND_TREE, canonical_json_bytes({"type": KIND_TREE, "entries": entries}))


def _file_mode(path: Path) -> str:
    """``exec`` if +x else ``file``. No symlink / other modes are emitted."""
    with suppress(OSError):
        if path.stat().st_mode & _EXEC_BITS:
            return MODE_EXEC
    return MODE_FILE


def build_tree(dir_path: Path, objects: ObjectSet, *, max_object_bytes: int) -> str:
    """Build objects for *dir_path* recursively; return the tree address. Symlinks/special files are
    skipped (contract). A blob over *max_object_bytes* raises ValueError (server would 413)."""
    entries: List[Dict[str, str]] = []
    for child in sorted(dir_path.iterdir(), key=lambda p: p.name):
        if not _is_safe_tree_name(child.name):
            # POSIX-legal but wire-unsafe local names ('a\\b', 'x:y', 'CON', 'trailing ') upload
            # fine but every pull then skips them, so devices diverge with no signal; refuse at
            # the source instead.
            logger.warning("skills_sync_client: skipping %s: name not wire-safe", child)
        elif child.is_symlink():
            logger.debug("skills_sync_client: skipping symlink %s", child)
        elif child.is_dir():
            entries.append(_entry(child.name, KIND_TREE, build_tree(child, objects, max_object_bytes=max_object_bytes),
                                  MODE_DIR))
        elif child.is_file():
            data = child.read_bytes()
            if len(data) > max_object_bytes:
                raise ValueError(f"file {child} is {len(data)} bytes > max_object_bytes {max_object_bytes}")
            entries.append(_entry(child.name, KIND_BLOB, objects.add(KIND_BLOB, data), _file_mode(child)))
    return _add_tree(entries, objects)


def build_commit(tree_hash: str, parents: List[str], *, owner: str, device: str, message: str,
                 objects: ObjectSet, ts: Optional[str] = None) -> str:
    """Commit object address. ``parents``: 0 first commit, 1 edit, 2 merge (base first, other head second)."""
    return objects.add(KIND_COMMIT, canonical_json_bytes({
        "type": KIND_COMMIT, "tree": tree_hash, "parents": list(parents),
        "author": {"owner": owner, "device": device},
        "ts": ts or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "message": message, "artifact_type": ARTIFACT_TYPE_SKILL}))


def build_root_tree(node: Dict[str, Any], objects: ObjectSet, *, manifest_hash: Optional[str] = None) -> str:
    """Canonicalize a nested ``{name: {"__tree__": hash} | subdict}`` root into trees. ``manifest_hash``
    adds the root ``sync-manifest`` BLOB entry (cannot collide with a skill dir: those are trees)."""
    entries: List[Dict[str, str]] = []
    for name, child in node.items():
        leaf = isinstance(child, dict) and "__tree__" in child and len(child) == 1
        entries.append(_entry(name, KIND_TREE, child["__tree__"] if leaf else build_root_tree(child, objects),
                              MODE_DIR))
    if manifest_hash is not None:
        entries.append(_entry(SYNC_MANIFEST_ENTRY_NAME, KIND_BLOB, manifest_hash, MODE_FILE))
    return _add_tree(entries, objects)


def nest_skill_tree(root: Dict[str, Any], rel_parts: Tuple[str, ...], tree_hash: str) -> bool:
    """Insert a skill tree leaf into the nested root structure by path parts. Refuses parts that
    would be unsafe on the wire (``..``, separators, Win32-folded names) -- a POSIX-legal local
    dir named ``a\\b`` or ``.. `` pushed verbatim would traverse other members' mirrors."""
    if not rel_parts or not all(_is_safe_tree_name(p) for p in rel_parts):
        logger.warning("skills_sync_client: refusing to nest skill at unsafe path %r",
                       "/".join(str(p) for p in rel_parts))
        return False
    node = root
    for part in rel_parts[:-1]:
        node = node.setdefault(part, {})
    node[rel_parts[-1]] = {"__tree__": tree_hash}
    return True


def assemble_root_from_skill_trees(skill_trees: Dict[str, str], objects: ObjectSet) -> str:
    """Profile-root tree from ``{posix_rel_path: tree_hash}``; skill trees are assumed durable, only
    the new intermediate/root trees are added."""
    root: Dict[str, Any] = {}
    for path, tree_hash in skill_trees.items():
        nest_skill_tree(root, PurePosixPath(path).parts, tree_hash)
    return build_root_tree(root, objects)


# HTTP client (routes under /v1/sync/)
class SyncError(RuntimeError):
    """A non-recoverable wire error (4xx the client can't retry)."""

    def __init__(self, message: str, *, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class SyncConflict(RuntimeError):
    """CAS lost (409); NOT a rejection, pushed objects are durable. ``actual`` = head to merge against,
    or None when the ref does not exist (server sends ""): retry as a create, never fetch it."""

    def __init__(self, actual: Optional[str]):
        self.actual: Optional[str] = actual or None
        super().__init__(f"CAS conflict; actual head {self.actual}" if self.actual
                         else "CAS conflict; the ref does not exist yet")


def _check_version(caps: Dict[str, Any]) -> None:
    """Reject an incompatible server major version."""
    ver = str(caps.get("hsp_version") or "")  # wire field name
    if ver.split(".", 1)[0] != WIRE_VERSION:
        raise SyncError(f"this server speaks sync version {ver!r}, but this Hermes speaks "
                        f"{WIRE_VERSION} — update Hermes to sync with it")


def _body(r) -> Dict[str, Any]:
    return r.json() if r.content else {}


def checked_capabilities(client: "SyncClient") -> Tuple[Dict[str, Any], int]:
    """Version-checked ``(caps, max_object_bytes)`` for a sync session; also pins the negotiated
    cap on the client so ``get_object`` can bound what a hostile plane streams back."""
    caps = client.capabilities()
    _check_version(caps)
    client.max_object_bytes = int(caps.get("max_object_bytes") or DEFAULT_MAX_OBJECT_BYTES)
    return caps, client.max_object_bytes


class SyncClient:
    """Sync client bound to a base URL + Nous bearer. Org refs/objects live behind SEPARATE ``org/``
    routes: the personal routes are hard-scoped to the token's owner and would silently answer an
    org query with personal data, so org readers MUST pass ``org_scope=True`` on every hop."""

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 30.0):
        self.base = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_object_bytes = DEFAULT_MAX_OBJECT_BYTES  # refined by checked_capabilities
        import requests  # core dependency
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {api_key}"

    def _request(self, method: str, path: str, op: str, *, ok=(200,), errors: Optional[Dict[int, Any]] = None, **kw):
        """One wire call to ``/v1/sync/<path>``; SyncError unless the status is in *ok*. *errors* maps a
        status to a message (str or ``fn(response)``); anything else gets ``"<op> failed: <code>"``."""
        r = self._session.request(method, f"{self.base}/v1/sync/{path.lstrip('/')}", timeout=self.timeout, **kw)
        if r.status_code in ok:
            return r
        msg = (errors or {}).get(r.status_code)
        raise SyncError((msg(r) if callable(msg) else msg) or f"{op} failed: {r.status_code}", status=r.status_code)

    def capabilities(self) -> Dict[str, Any]:
        """GET capabilities (no auth required)."""
        return self._request("GET", "capabilities", "capabilities").json()

    def get_refs(self, prefix: str, *, org_scope: bool = False) -> List[Dict[str, str]]:
        """GET refs?prefix=... (or org/refs, filtered client-side by *prefix*)."""
        path, params = ("org/refs", None) if org_scope else ("refs", {"prefix": prefix})
        refs = (self._request("GET", path, "get_refs", params=params).json() or {}).get("refs", [])
        return [r_ for r_ in refs if str(r_.get("name", "")).startswith(prefix)] if org_scope else refs

    def get_object(self, obj_hash: str, *, org_scope: bool = False) -> Tuple[str, bytes]:
        """GET objects/:hash -> ``(kind, bytes)``; kind from the object-type header, blob default.
        Rehashes the body: content addressing is only a guarantee if the client enforces it, and
        a plane returning unverified bytes could serve a cyclic tree (honest trees cannot form
        cycles -- a tree's hash contains its children's) or arbitrary bytes under a known hash."""
        r = self._request("GET", f"{'org/' if org_scope else ''}objects/{obj_hash}", "get_object",
                          stream=True,
                          errors={404: f"object {obj_hash} not found", 403: f"object {obj_hash} not readable"})
        try:
            chunks, total = [], 0
            for chunk in r.iter_content(1 << 16):
                total += len(chunk)
                if total > self.max_object_bytes:
                    raise SyncError(f"object {obj_hash} exceeds max_object_bytes", status=r.status_code)
                chunks.append(chunk)
            data = b"".join(chunks)
        finally:
            r.close()
        if wire_address(data) != obj_hash:
            raise SyncError(f"object {obj_hash} content hash mismatch", status=r.status_code)
        return r.headers.get("X-HSP-Object-Type") or KIND_BLOB, data

    def _get_json_of_kind(self, obj_hash: str, expected: str, org_scope: bool) -> Dict[str, Any]:
        kind, data = self.get_object(obj_hash, org_scope=org_scope)
        if kind != expected:
            raise SyncError(f"{obj_hash} is {kind}, expected {expected}")
        return json.loads(data.decode("utf-8"))

    def get_commit_json(self, commit_hash: str, *, org_scope: bool = False) -> Dict[str, Any]:
        return self._get_json_of_kind(commit_hash, KIND_COMMIT, org_scope)

    def get_tree_json(self, tree_hash: str, *, org_scope: bool = False) -> Dict[str, Any]:
        return self._get_json_of_kind(tree_hash, KIND_TREE, org_scope)

    def put_objects(self, objects: Dict[str, Tuple[str, bytes]], *, org_scope: bool = False) -> Dict[str, Any]:
        """POST objects as multipart (field = claimed ``sha256:<hex>``, filename = kind, body = raw
        bytes; no base64-in-JSON). The server rehashes and 422s the whole batch on mismatch; known
        hashes are no-ops. ``org_scope`` adds ``?scope=org`` (required before an org CAS/propose)."""
        files = [(h, (kind, data, "application/octet-stream")) for h, (kind, data) in objects.items()]
        return _body(self._request(
            "POST", "objects", "put_objects", ok=(200, 201), files=files,
            params={"scope": "org"} if org_scope else None,
            errors={413: "object too large (413)", 422: lambda r: f"hash_mismatch (422): {r.text}"}))

    def cas_ref(self, name: str, from_hash: Optional[str], to_hash: str) -> Dict[str, Any]:
        """POST refs/:name -- atomic CAS; SyncConflict on 409. A member's CAS on an org HEAD becomes a
        proposal (202) -> ``{"proposal_pending": True, ...}``: success-shaped, never present as live."""
        r = self._request("POST", f"refs/{name}", "cas_ref", ok=(200, 202, 409),
                          json={"from": from_hash, "to": to_hash}, errors={403: "forbidden (403) -- owner/permission"})
        if r.status_code == 202:
            return {"proposal_pending": True, **_body(r)}
        if r.status_code == 409:  # "" actual = the ref does not exist server-side (-> None)
            raise SyncConflict((r.json() or {}).get("actual", ""))
        return _body(r)


# Reading remote trees
def read_ref_hash(client: SyncClient, ref: str, *, org_scope: bool = False) -> Optional[str]:
    """Hash of *ref* (queried with itself as prefix), or None if absent."""
    refs = client.get_refs(ref, org_scope=org_scope)
    return next((r.get("hash") for r in refs if r.get("name") == ref), None)


def root_tree_of_commit(client: SyncClient, commit_hash: str, *, org_scope: bool = False) -> str:
    return client.get_commit_json(commit_hash, org_scope=org_scope)["tree"]


def _is_safe_tree_name(name: Any) -> bool:
    """A remote tree entry name must be one relative path component: reject empties, dot entries,
    NUL, and BOTH separators (the wire is posix but ``\\`` still escapes on Windows, which this
    project ships). Win32 also strips trailing dots/spaces per component, so ``".. "`` lands on
    ``..`` -- the rstrip check below is what keeps that spelling out. Applied at
    ``skill_trees_of_root`` so walk callers never see a bad rel path, and re-checked per entry
    inside ``materialize_tree`` as the sink-side backstop for callers handing it a tree hash
    directly."""
    if not isinstance(name, str) or not name or "/" in name or "\\" in name:
        return False
    if any(ord(c) < 0x20 or c in '<>"|?*' for c in name):
        # NUL and the Win32-forbidden set: POSIX-legal names that fail or hit devices on Windows.
        return False
    if ":" in name:
        # NTFS alternate-data-stream separator; remote content has no legitimate use for it.
        return False
    stripped = name.rstrip(" .")
    if not stripped or stripped in (".", ".."):
        return False
    # Win32 device names are unwritable/traversable devices regardless of extension.
    if stripped.split(".", 1)[0].upper() in _WIN32_DEVICE_NAMES:
        return False
    return True


def fold_rel_to_disk(root: Path, rel_path: str) -> str:
    """Fold each component of *rel_path* onto the name an existing on-disk sibling already has.
    Win32/APFS fold case and trailing dots/spaces, so remote 'Deploy' lands in local 'deploy' --
    returns the disk-canonical rel path so alias dedupe, baseline lookups and rmtree all key off
    the directory that will actually be written. Components without a match pass through."""
    parts = rel_path.split("/")
    cur, out = root, []
    for part in parts:
        key = part.rstrip(" .").casefold()
        actual = part
        with suppress(OSError):
            for child in cur.iterdir():
                if child.name.rstrip(" .").casefold() == key:
                    actual = child.name
                    break
        out.append(actual)
        cur = cur / actual
    return "/".join(out)


def confined_tree_dest(root: Path, rel_path: str) -> Optional[Path]:
    """``root / rel_path`` confined under *root*; None when the path escapes (``..`` segments) or
    when any interior component is a link (the resolved path differs from the lexical join, so a
    symlinked interior would silently retarget writes/rmtree onto a sibling). Returns the LEXICAL
    path so ``rmtree`` on a symlink raises rather than deleting the link's in-root target.
    Backstop for the sinks: validation belongs to ``skill_trees_of_root``, this guards the join."""
    try:
        resolved_root = root.resolve()
        lexical = resolved_root / PurePosixPath(rel_path)
        resolved = lexical.resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if resolved_root not in resolved.parents:
        return None
    return lexical if resolved == lexical else None


def skill_trees_of_root(client: SyncClient, root_tree_hash: str, *, org_scope: bool = False) -> Dict[str, str]:
    """``{posix_rel_path: skill_tree_hash}`` for every subtree containing a ``SKILL.md`` blob.
    Unsafe remote names are skipped (warned): the result keys feed ``root / path`` joins that
    lead to ``rmtree``/blob writes, so ``..``/separator components must never leave this walk."""
    result: Dict[str, str] = {}
    def _walk(tree_hash: str, prefix: str, depth: int) -> None:
        entries = client.get_tree_json(tree_hash, org_scope=org_scope).get("entries", [])
        if not isinstance(entries, list):
            return
        if prefix and any(e.get("name") == "SKILL.md" and e.get("kind") == KIND_BLOB for e in entries):
            result[prefix] = tree_hash
            return
        if depth >= _MAX_TREE_DEPTH:
            logger.warning("skills_sync_client: remote tree nested past %d levels, skipping %r",
                           _MAX_TREE_DEPTH, prefix)
            return
        for e in entries:
            if not isinstance(e, dict) or e.get("kind") != KIND_TREE:
                continue
            if not _is_safe_tree_name(e.get("name")):
                logger.warning("skills_sync_client: skipping unsafe tree entry name %r", e.get("name"))
                continue
            child = e.get("hash")
            if not isinstance(child, str) or not child:
                continue
            _walk(child, f"{prefix}/{e['name']}" if prefix else e["name"], depth + 1)
    _walk(root_tree_hash, "", 0)
    return result


def read_manifest_of_root(client: SyncClient, root_tree_hash: str) -> Optional[Dict[str, bool]]:
    """``{name: enabled}`` from the root ``sync-manifest`` blob (how a device learns another's opt-ins)."""
    try:
        for e in client.get_tree_json(root_tree_hash).get("entries", []):
            if e.get("name") == SYNC_MANIFEST_ENTRY_NAME and e.get("kind") == KIND_BLOB:
                return parse_sync_manifest(client.get_object(e["hash"])[1])
    except Exception as e:
        logger.debug("skills_sync_client: manifest read failed: %s", e)
    return None


def _is_link(path: Path) -> bool:
    """Symlink or (Windows 3.12+) directory junction -- same check as tools/skills_hub_install.py's
    ``_is_path_redirect``. Junctions need no privilege on Windows, so a pre-existing one can
    redirect writes exactly like a symlink."""
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def materialize_tree(client: SyncClient, tree_hash: str, dest: Path, *, org_scope: bool = False,
                     _depth: int = 0) -> None:
    """Write the tree into *dest*: blobs -> files (+x for ``exec``), trees -> subdirs. Does NOT delete
    files absent from the tree (caller decides). Refuses path traversal."""
    if _depth > _MAX_TREE_DEPTH:
        logger.warning("skills_sync_client: remote tree nested past %d levels, truncating under %s",
                       _MAX_TREE_DEPTH, dest)
        return
    dest.mkdir(parents=True, exist_ok=True)
    for entry in client.get_tree_json(tree_hash, org_scope=org_scope).get("entries", []):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name", "")
        if not _is_safe_tree_name(name):
            logger.warning("skills_sync_client: skipping unsafe tree entry %r", name)
            continue
        child_hash = entry.get("hash")
        if not isinstance(child_hash, str) or not child_hash:
            continue
        target, kind = dest / name, entry.get("kind")
        if kind == KIND_TREE:
            if _is_link(target):
                # Writing inside a pre-existing symlinked/junctioned dir escapes dest.
                logger.warning("skills_sync_client: skipping tree entry over a symlink %r", name)
                continue
            materialize_tree(client, child_hash, target, org_scope=org_scope, _depth=_depth + 1)
        elif kind == KIND_BLOB:
            _, data = client.get_object(child_hash, org_scope=org_scope)
            # Pre-existing targets that must be replaced before write_bytes: links and junctions
            # redirect outside dest, hardlinks share an outside inode, and special files (FIFO,
            # socket, device) block open('wb') waiting for a peer. A plain regular file is safe
            # to truncate in place; a directory is a real collision (skip loudly).
            try:
                if target.is_dir() and not _is_link(target):
                    logger.warning("skills_sync_client: blob %r collides with a directory, skipping", name)
                    continue
                if _is_link(target) or target.exists():
                    target.unlink()
            except OSError as e:
                logger.warning("skills_sync_client: refusing blob write over %r: %s", name, e)
                continue
            target.write_bytes(data)
            if entry.get("mode") == MODE_EXEC:
                with suppress(OSError):
                    target.chmod(target.stat().st_mode | _EXEC_BITS)


def merge_skill(base: Optional[str], ours: Optional[str], theirs: Optional[str]) -> str:
    """Three-way decision: ``ours``/``theirs``/``either``/``overlap``/``none``. A side "modified" the
    skill when its hash differs from the base (same semantics as skills_sync.py)."""
    if ours == theirs:
        return "either" if ours is not None else "none"
    if theirs == base:  # only we moved
        return "ours"
    if ours == base:  # only they moved
        return "theirs"
    return "overlap"

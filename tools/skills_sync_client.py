#!/usr/bin/env python3
"""
HSP/1 sync client -- Hermes Sync Protocol version 1, client (personal skill sync).

This is the LOW-LEVEL sync layer. It builds content-addressed HSP objects
(blob/tree/commit) from local skills, talks the HSP/1 wire contract to a sync
plane (push objects + CAS a ref, pull the owner's HEAD, three-way merge on a
409), and is driven by:

  * a debounced push hook in ``skill_manage`` (after the write-gate passes),
  * a periodic pull hook (``maybe_pull_skills``) at the curator tick sites,
  * the ``hermes sync status|pull|push|now`` CLI.

It lives beside ``tools/skills_sync.py`` (NOT under ``hermes_cli/``) so the
low-level sync layer never imports the CLI -- same rule the bundled-skills
sync module documents at ``skills_sync.py:43-50``.

Contract: ``~/src/specs/collective-wisdom/hsp-1-contract.md`` (HSP/1, frozen
for Milestone 1). Endpoint shapes, object model, canonicalization, and status
codes below all trace to that document.

--- DEV-PHASE GATE (Milestone 1) -----------------------------------------
Client sync is INERT (no push, no pull, no-op) unless the resolved Nous
identity's access token carries ``tool_gateway_admin === true``. That claim is
minted by NAS (access-token-issuer.ts:312) and rides on the same bearer
``resolve_nous_runtime_credentials()`` returns. We decode the JWT payload
(no signature verification -- the server re-verifies) and check the claim
before doing any sync work. This is a temporary dev gate for the M1 rollout;
remove it (or replace it with a real ``sync:*`` scope / config toggle) when
sync ships to all users.

--- OPT-IN DEFAULT (M1-D, provisional) -----------------------------------
Nothing syncs unless the user marks a skill for sync. The user's local intent
is toggled via ``hermes sync enable/disable`` (a ``sync`` flag on the skill's
``.usage.json`` sidecar, alongside ``pinned``/``created_by``), but the DURABLE,
CROSS-DEVICE opt-in state is a committed ``sync-manifest`` object in the sync
plane (design.md §2.8): a root-level blob in the tree at
``refs/user/<owner>/HEAD`` recording per-skill ``{name, enabled}``. Push writes
the manifest from local intent; pull reconciles local intent FROM it, so a skill
opted in on one device becomes opted in on the others. The plane manifest is
authoritative; the local flag is just the editable intent. Only agent-created +
user-authored skills under ``~/.hermes/skills/`` are eligible; bundled and
hub-installed skills are excluded.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import stat as _stat
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# HSP/1 protocol constants (contract §1, §3.1)
HSP_VERSION = "1"
DEFAULT_MAX_OBJECT_BYTES = 26214400  # 25 MiB, mirrors capabilities default

# Object kinds (contract §2)
KIND_BLOB = "blob"
KIND_TREE = "tree"
KIND_COMMIT = "commit"

# Tree entry modes (contract §2.3)
MODE_FILE = "file"
MODE_EXEC = "exec"
MODE_DIR = "dir"

ARTIFACT_TYPE_SKILL = "skill"

# ---------------------------------------------------------------------------
# `sync-manifest` object convention (design.md §2.8).
#
# Per-skill sync opt-in ("this skill syncs / this one does not" — the M1-D
# opt-in state) is CONTENT inside the HSP object model, NOT a device-local flag
# or a mutable preference table. An owner's synced set is a small committed blob
# named ``sync-manifest`` at the ROOT of the tree referenced by
# ``refs/user/<owner>/HEAD``, recording per-skill ``{name, enabled}``. Toggling
# opt-in is a plain CAS ref update (upload the new manifest blob + root tree +
# commit, then CAS HEAD) — the same primitives push already uses.
#
# This makes opt-in durable and CROSS-DEVICE: device B learns which skills the
# user opted in on device A by reading the manifest on pull, rather than each
# device keeping its own local flag. The ``.usage.json`` ``sync`` flag is kept
# only as the local *intent* the user toggles via ``hermes sync enable`` — it is
# reconciled TO the manifest on pull and FROM it on push; the manifest in the
# plane is authoritative.
#
# MUST match gateway-gateway ``src/sync/manifest.ts`` byte-for-byte (the server
# reads + validates this exact shape). Entry name, ``type`` marker, ``version``,
# and the ``{name, enabled}`` skill shape are the shared contract.
# ---------------------------------------------------------------------------

SYNC_MANIFEST_ENTRY_NAME = "sync-manifest"
SYNC_MANIFEST_TYPE = "sync-manifest"
SYNC_MANIFEST_VERSION = 1


def build_sync_manifest_bytes(skills: Dict[str, bool]) -> bytes:
    """Serialize the per-skill opt-in map into canonical ``sync-manifest`` bytes.

    ``skills`` maps skill name -> enabled. Emits the shape gateway-gateway's
    ``parseSyncManifest`` validates: ``{type, version:1, skills:[{name,enabled}]}``.
    Skill entries are sorted by name for a stable content address.
    """
    manifest = {
        "type": SYNC_MANIFEST_TYPE,
        "version": SYNC_MANIFEST_VERSION,
        "skills": [
            {"name": name, "enabled": bool(enabled)}
            for name, enabled in sorted(skills.items())
        ],
    }
    return canonical_json_bytes(manifest)


def parse_sync_manifest(data: bytes) -> Optional[Dict[str, bool]]:
    """Parse ``sync-manifest`` bytes into ``{name: enabled}``, or ``None`` if the
    bytes are not a well-formed manifest.

    Strict (mirrors gateway-gateway ``parseSyncManifest``): an unknown ``type``,
    a missing/!=1 ``version``, a non-array ``skills``, or a malformed skill entry
    all reject rather than being coerced — a malformed manifest must not be
    mistaken for "no skills opted in."
    """
    try:
        value = json.loads(data.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(value, dict):
        return None
    if value.get("type") != SYNC_MANIFEST_TYPE:
        return None
    if value.get("version") != SYNC_MANIFEST_VERSION:
        return None
    raw_skills = value.get("skills")
    if not isinstance(raw_skills, list):
        return None
    out: Dict[str, bool] = {}
    for raw in raw_skills:
        if not isinstance(raw, dict):
            return None
        name = raw.get("name")
        enabled = raw.get("enabled")
        if not isinstance(name, str) or not name:
            return None
        if not isinstance(enabled, bool):
            return None
        out[name] = enabled
    return out


# ---------------------------------------------------------------------------
# Content addressing (contract §2.1 / OI-5)
#
# HSP uses the FULL 64-hex sha256 digest on the wire. This is a DIFFERENT
# namespace from hermes-agent's local ``content_hash`` (skills_guard.py:846),
# which is a truncated 16-hex digest used for local dedup. They must never be
# conflated -- we compute full digests here.
# ---------------------------------------------------------------------------

def hsp_address(data: bytes) -> str:
    """Return ``sha256:<64-hex>`` -- the HSP wire address of ``data`` (contract §2.1)."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def canonical_json_bytes(obj: Dict[str, Any]) -> bytes:
    """Canonical JSON serialization for tree/commit hashing (contract §2.5).

    UTF-8, keys sorted lexicographically, no insignificant whitespace
    (``separators=(",", ":")``), no trailing newline. Arrays must already be
    in the contract-specified order by the caller (tree entries by ``name``,
    commit ``parents`` in significance order). Both client and server MUST
    produce byte-identical output or a push fails ``422 hash_mismatch``.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# Identity & DEV-PHASE gate
#
# We reuse resolve_nous_runtime_credentials() for the bearer (it honors the
# cross-process file lock + portal host allowlist and refreshes as needed --
# we do NOT reimplement refresh). The returned api_key IS the JWT bearer; we
# decode its payload (unverified) to read the dev gate claim.
# ---------------------------------------------------------------------------

# Dev-phase gate claim (NAS access-token-issuer.ts:312). Sync is inert unless
# the resolved token carries this claim === true. Remove when sync ships GA.
DEV_GATE_CLAIM = "tool_gateway_admin"


class SyncInertError(RuntimeError):
    """Raised (and caught by the gate-and-swallow hooks) when sync must no-op:

    not logged in, no bearer, or the dev-phase gate claim is absent/false.
    """


def _decode_jwt_payload_unverified(token: str) -> Dict[str, Any]:
    """Decode a JWT payload WITHOUT signature verification.

    Safe here: we never trust these claims for authz -- the server re-verifies
    every call. We only read the dev-gate claim to decide whether to attempt
    sync at all. Mirrors the diagnostic decode in
    plugins/dashboard_auth/nous/__init__.py:463.
    """
    try:
        import jwt  # PyJWT, a core dependency

        return jwt.decode(
            token,
            options={"verify_signature": False, "verify_exp": False},
        ) or {}
    except Exception as e:
        logger.debug("skills_sync_client: JWT payload decode failed: %s", e)
        return {}


def resolve_identity() -> Dict[str, Any]:
    """Resolve the Nous bearer + owner + dev-gate flag.

    Returns a dict: ``{api_key, base_url, owner, dev_gate_ok, claims}``.
    Raises :class:`SyncInertError` if not logged in / no bearer.

    ``owner`` is the token-verified subject; the server derives the real owner
    from the bearer regardless (contract §0.4), so this is advisory for local
    ref naming only.
    """
    try:
        from hermes_cli.auth import resolve_nous_runtime_credentials

        creds = resolve_nous_runtime_credentials()
    except Exception as e:
        raise SyncInertError(f"no Nous credentials: {e}") from e

    api_key = (creds or {}).get("api_key")
    if not api_key:
        raise SyncInertError("no bearer token available")

    claims = _decode_jwt_payload_unverified(api_key)
    owner = (
        claims.get("sub")
        or claims.get("privy_did")
        or claims.get("tid")
        or "unknown"
    )
    dev_gate_ok = claims.get(DEV_GATE_CLAIM) is True
    return {
        "api_key": api_key,
        "base_url": (creds or {}).get("base_url"),
        "owner": str(owner),
        "dev_gate_ok": dev_gate_ok,
        "claims": claims,
    }


def dev_gate_open() -> bool:
    """Whether the DEV-PHASE gate permits sync. Never raises."""
    try:
        return bool(resolve_identity().get("dev_gate_ok"))
    except SyncInertError:
        return False
    except Exception as e:
        logger.debug("skills_sync_client: dev_gate_open check failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Sync-plane endpoint resolution
#
# The HSP routes are mounted under /v1/sync/ (contract §1). The base URL is
# configurable (config.yaml sync.base_url or HERMES_SYNC_BASE_URL bridge env);
# it is NOT the inference base_url. When unset, sync is inert -- there is no
# server to talk to yet (the server is being built in parallel).
# ---------------------------------------------------------------------------

def resolve_sync_base_url() -> Optional[str]:
    """Resolve the HSP sync-plane base URL, or None when unconfigured.

    Order: HERMES_SYNC_BASE_URL env bridge -> config.yaml ``sync.base_url``.
    Returns a base without a trailing slash (e.g. ``https://host``); the
    ``/v1/sync/`` prefix is appended by the client.
    """
    env = os.getenv("HERMES_SYNC_BASE_URL")
    if env and env.strip():
        return env.strip().rstrip("/")
    try:
        # Lazy import: the low-level sync layer must not import the CLI at
        # module load (skills_sync.py:43-50). A function-scoped import avoids
        # the cycle -- same pattern agent/curator.py:141 uses for config.
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        sync_cfg = cfg.get("sync") or {}
        base = sync_cfg.get("base_url")
        if isinstance(base, str) and base.strip():
            return base.strip().rstrip("/")
    except Exception as e:
        logger.debug("skills_sync_client: config sync.base_url read failed: %s", e)
    return None


# ---------------------------------------------------------------------------
# Local skill eligibility + the M1-D opt-in "sync" flag
#
# Only agent-created + user-authored skills under ~/.hermes/skills/ sync.
# Bundled (.bundled_manifest) and hub-installed skills are excluded. Sync is
# opt-in: a skill only syncs when its usage-sidecar carries ``sync: true``.
# ---------------------------------------------------------------------------

def _skills_dir() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "skills"


def is_sync_eligible(skill_name: str) -> bool:
    """Whether *skill_name* is a candidate for HSP sync (before the opt-in check).

    Eligible = present locally under ~/.hermes/skills/, NOT bundled, NOT
    hub-installed, NOT an external-dir skill. Mirrors the exclusion logic used
    by the curator (tools/skill_usage.py).
    """
    try:
        from tools.skill_usage import is_bundled, is_hub_installed, _find_skill_dir
        from agent.skill_utils import is_external_skill_path
    except Exception:
        return False
    if is_bundled(skill_name) or is_hub_installed(skill_name):
        return False
    skill_dir = _find_skill_dir(skill_name)
    if skill_dir is None:
        return False
    if is_external_skill_path(skill_dir):
        return False
    return True


def list_synced_skill_names() -> List[str]:
    """Return the names of skills the user has opted into sync (``sync: true``)
    AND that remain eligible. Sorted, deduped."""
    try:
        from tools.skill_usage import load_usage
    except Exception:
        return []
    names = []
    for name, rec in (load_usage() or {}).items():
        if isinstance(rec, dict) and rec.get("sync") is True and is_sync_eligible(name):
            names.append(name)
    return sorted(set(names))


# ---------------------------------------------------------------------------
# Object building -- turn a skill directory into HSP blob/tree/commit objects
#
# A skill dir becomes one tree (contract §2.3). Each file is a blob; each
# subdir a nested tree. The profile-root tree (contract §2.3: "a tree whose
# entries are category trees") is built from the set of synced skill trees.
# ---------------------------------------------------------------------------

class ObjectSet:
    """Accumulates HSP objects to push: hash -> (kind, bytes).

    Deduped by content address, so identical blobs across skills upload once.
    """

    def __init__(self) -> None:
        self.objects: Dict[str, Tuple[str, bytes]] = {}

    def add(self, kind: str, data: bytes) -> str:
        addr = hsp_address(data)
        self.objects.setdefault(addr, (kind, data))
        return addr

    def __len__(self) -> int:
        return len(self.objects)


def _file_mode(path: Path) -> str:
    """Return the HSP tree mode for a regular file: ``exec`` if +x else ``file``
    (contract §2.3). No symlinks / other modes are emitted."""
    try:
        if path.stat().st_mode & (_stat.S_IXUSR | _stat.S_IXGRP | _stat.S_IXOTH):
            return MODE_EXEC
    except OSError:
        pass
    return MODE_FILE


def build_tree(dir_path: Path, objects: ObjectSet, *, max_object_bytes: int) -> str:
    """Recursively build HSP objects for *dir_path*; return the tree address.

    Regular files become blobs; subdirectories become nested trees. Symlinks,
    sockets, and other special files are skipped (contract §2.3 security: no
    symlinks). Blobs over *max_object_bytes* raise :class:`ValueError` so the
    caller can surface / skip the artifact (contract §4.3 -> 413).
    """
    entries: List[Dict[str, str]] = []
    for child in sorted(dir_path.iterdir(), key=lambda p: p.name):
        if child.is_symlink():
            logger.debug("skills_sync_client: skipping symlink %s", child)
            continue
        if child.is_dir():
            sub_hash = build_tree(child, objects, max_object_bytes=max_object_bytes)
            entries.append(
                {"name": child.name, "kind": KIND_TREE, "hash": sub_hash, "mode": MODE_DIR}
            )
        elif child.is_file():
            data = child.read_bytes()
            if len(data) > max_object_bytes:
                raise ValueError(
                    f"file {child} is {len(data)} bytes > max_object_bytes "
                    f"{max_object_bytes} (contract §4.3)"
                )
            blob_hash = objects.add(KIND_BLOB, data)
            entries.append(
                {
                    "name": child.name,
                    "kind": KIND_BLOB,
                    "hash": blob_hash,
                    "mode": _file_mode(child),
                }
            )
        # else: skip special files
    # Entries sorted by name (byte order) for canonicalization (contract §2.3).
    entries.sort(key=lambda e: e["name"])
    tree_obj = {"type": KIND_TREE, "entries": entries}
    return objects.add(KIND_TREE, canonical_json_bytes(tree_obj))


def build_commit(
    tree_hash: str,
    parents: List[str],
    *,
    owner: str,
    device: str,
    message: str,
    objects: ObjectSet,
    ts: Optional[str] = None,
) -> str:
    """Build a commit object (contract §2.4) and return its address.

    ``parents``: 0 for first commit, 1 for a normal edit, 2 for a merge commit
    (order significant: parents[0] = base fast-forwarded from, parents[1] =
    the other head being merged).
    """
    commit_obj = {
        "type": KIND_COMMIT,
        "tree": tree_hash,
        "parents": list(parents),
        "author": {"owner": owner, "device": device},
        "ts": ts or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "message": message,
        "artifact_type": ARTIFACT_TYPE_SKILL,
    }
    return objects.add(KIND_COMMIT, canonical_json_bytes(commit_obj))


def stable_device_id() -> str:
    """Return an opaque, stable per-device id for commit ``author.device``
    (contract §2.4 -- advisory, never an auth input). Persisted under
    ~/.hermes/skills/.sync_device_id."""
    path = _skills_dir() / ".sync_device_id"
    try:
        if path.exists():
            val = path.read_text(encoding="utf-8").strip()
            if val:
                return val
    except OSError:
        pass
    import uuid

    val = uuid.uuid4().hex
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(val, encoding="utf-8")
    except OSError as e:
        logger.debug("skills_sync_client: could not persist device id: %s", e)
    return val


# ---------------------------------------------------------------------------
# HSP/1 wire client
#
# Thin requests-based client for the endpoints in contract §3-§4. Uploads all
# new objects (batch), then CAS-es the ref. A 409 returns the actual head for
# the caller's three-way merge. Auth is the Nous bearer resolved above.
# ---------------------------------------------------------------------------

class HSPError(RuntimeError):
    """A non-recoverable HSP wire error (4xx that the client can't retry)."""

    def __init__(self, message: str, *, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class HSPConflict(RuntimeError):
    """CAS lost (409). ``actual`` is the current head to merge against
    (contract §4.4). NOT a rejection -- pushed objects are already durable."""

    def __init__(self, actual: str):
        super().__init__(f"CAS conflict; actual head {actual}")
        self.actual = actual


class HSPClient:
    """HSP/1 client bound to a base URL + bearer (contract §1, routes under
    ``/v1/sync/``)."""

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 30.0):
        self.base = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        import requests  # core dependency

        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {api_key}"

    def _url(self, path: str) -> str:
        return f"{self.base}/v1/sync/{path.lstrip('/')}"

    # -- capability & read -------------------------------------------------

    def capabilities(self) -> Dict[str, Any]:
        """GET /v1/sync/capabilities (contract §3.1). No auth required."""
        r = self._session.get(self._url("capabilities"), timeout=self.timeout)
        if r.status_code != 200:
            raise HSPError(f"capabilities failed: {r.status_code}", status=r.status_code)
        return r.json()

    def get_refs(self, prefix: str) -> List[Dict[str, str]]:
        """GET /v1/sync/refs?prefix=... (contract §3.2)."""
        r = self._session.get(
            self._url("refs"), params={"prefix": prefix}, timeout=self.timeout
        )
        if r.status_code != 200:
            raise HSPError(f"get_refs failed: {r.status_code}", status=r.status_code)
        return (r.json() or {}).get("refs", [])

    def get_object(self, obj_hash: str) -> Tuple[str, bytes]:
        """GET /v1/sync/objects/:hash (contract §3.3). Returns (kind, bytes).

        Kind comes from ``X-HSP-Object-Type`` for tree/commit; a blob response
        (application/octet-stream) is returned as ``blob``.
        """
        r = self._session.get(self._url(f"objects/{obj_hash}"), timeout=self.timeout)
        if r.status_code == 404:
            raise HSPError(f"object {obj_hash} not found", status=404)
        if r.status_code == 403:
            raise HSPError(f"object {obj_hash} not readable", status=403)
        if r.status_code != 200:
            raise HSPError(f"get_object failed: {r.status_code}", status=r.status_code)
        kind = r.headers.get("X-HSP-Object-Type") or KIND_BLOB
        return kind, r.content

    def get_commit_json(self, commit_hash: str) -> Dict[str, Any]:
        """Fetch a commit object and parse its canonical JSON."""
        kind, data = self.get_object(commit_hash)
        if kind != KIND_COMMIT:
            raise HSPError(f"{commit_hash} is {kind}, expected commit")
        return json.loads(data.decode("utf-8"))

    def get_tree_json(self, tree_hash: str) -> Dict[str, Any]:
        """Fetch a tree object and parse its canonical JSON."""
        kind, data = self.get_object(tree_hash)
        if kind != KIND_TREE:
            raise HSPError(f"{tree_hash} is {kind}, expected tree")
        return json.loads(data.decode("utf-8"))

    # -- write -------------------------------------------------------------

    def put_objects(self, objects: Dict[str, Tuple[str, bytes]]) -> Dict[str, Any]:
        """POST /v1/sync/objects (contract §4.2). Batch multi-object upload.

        Contract §1 requires raw object bytes on the wire (NOT base64-in-JSON),
        and §4.2 specifies "a length-prefixed or multipart stream of
        {hash, type, bytes}". We use multipart/form-data: one part per object,
        the part's field name = the claimed ``sha256:<hex>`` hash, its
        ``filename`` carries the object ``type`` (blob|tree|commit), and the
        part body is the raw object bytes. The server recomputes each hash from
        the received bytes and rejects the whole batch with 422 on mismatch.
        Idempotent: a known hash is a no-op ``already_present``.

        NOTE (framing choice within contract latitude): §4.2 says "length-
        prefixed OR multipart"; this picks multipart/form-data with
        (field=hash, filename=type, body=raw-bytes). The server strand must
        parse the same framing -- flagged for cross-strand alignment.
        """
        # (field_name, (filename, raw_bytes, content_type))
        files = [
            (h, (kind, data, "application/octet-stream"))
            for h, (kind, data) in objects.items()
        ]
        r = self._session.post(
            self._url("objects"), files=files, timeout=self.timeout
        )
        if r.status_code == 413:
            raise HSPError("object too large (413)", status=413)
        if r.status_code == 422:
            raise HSPError(f"hash_mismatch (422): {r.text}", status=422)
        if r.status_code not in (200, 201):
            raise HSPError(f"put_objects failed: {r.status_code}", status=r.status_code)
        return r.json() if r.content else {}

    def cas_ref(self, name: str, from_hash: Optional[str], to_hash: str) -> Dict[str, Any]:
        """POST /v1/sync/refs/:name -- atomic compare-and-swap (contract §4.4).

        Raises :class:`HSPConflict` (carrying the actual head) on 409.
        """
        r = self._session.post(
            self._url(f"refs/{name}"),
            json={"from": from_hash, "to": to_hash},
            timeout=self.timeout,
        )
        if r.status_code == 409:
            actual = (r.json() or {}).get("actual", "")
            raise HSPConflict(actual)
        if r.status_code == 403:
            raise HSPError("forbidden (403) -- owner/permission", status=403)
        if r.status_code != 200:
            raise HSPError(f"cas_ref failed: {r.status_code}", status=r.status_code)
        return r.json() if r.content else {}


# ---------------------------------------------------------------------------
# HSP sync manifest (client-local, FULL-digest namespace)
#
# Records, per synced skill, the last commit HEAD we pushed/pulled and the
# tree hash of the on-disk content at that point. Distinct from the bundled
# manifest (skills_sync.py, truncated local content_hash namespace). Lives at
# ~/.hermes/skills/.sync_manifest as JSON.
# ---------------------------------------------------------------------------

def _sync_manifest_path() -> Path:
    return _skills_dir() / ".sync_manifest"


def read_sync_manifest() -> Dict[str, Any]:
    """Read the HSP sync manifest. Returns {} on missing/corrupt.

    Shape: ``{"head": "sha256:...|null", "skills": {name: {tree, commit}}}``.
    ``head`` is the last profile-root HEAD commit we reconciled with.
    """
    path = _sync_manifest_path()
    if not path.exists():
        return {"head": None, "skills": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("head", None)
            data.setdefault("skills", {})
            return data
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("skills_sync_client: sync manifest read failed: %s", e)
    return {"head": None, "skills": {}}


def write_sync_manifest(data: Dict[str, Any]) -> None:
    """Write the HSP sync manifest atomically. Best-effort."""
    import tempfile

    path = _sync_manifest_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".sync_manifest_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.debug("skills_sync_client: sync manifest write failed: %s", e)


# ---------------------------------------------------------------------------
# Tree materialization (pull) -- write an HSP tree back to a skill directory
# ---------------------------------------------------------------------------

def materialize_tree(client: HSPClient, tree_hash: str, dest: Path) -> None:
    """Write the HSP tree at *tree_hash* into *dest* (created if needed).

    Blobs become files (with +x restored for ``exec`` mode), nested trees
    become subdirectories. Does NOT delete files absent from the tree -- the
    caller decides removal semantics. Refuses path traversal via entry names.
    """
    dest.mkdir(parents=True, exist_ok=True)
    tree = client.get_tree_json(tree_hash)
    for entry in tree.get("entries", []):
        name = entry.get("name", "")
        if not name or "/" in name or name in (".", ".."):
            logger.warning("skills_sync_client: skipping unsafe tree entry %r", name)
            continue
        target = dest / name
        kind = entry.get("kind")
        if kind == KIND_TREE:
            materialize_tree(client, entry["hash"], target)
        elif kind == KIND_BLOB:
            _, data = client.get_object(entry["hash"])
            target.write_bytes(data)
            if entry.get("mode") == MODE_EXEC:
                try:
                    st = target.stat().st_mode
                    target.chmod(st | _stat.S_IXUSR | _stat.S_IXGRP | _stat.S_IXOTH)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Profile snapshot -- build the objects + per-skill tree map for a push
#
# The profile root is a tree whose entries mirror each synced skill's relative
# path under ~/.hermes/skills/ (contract §2.3: "the profile root is a tree
# whose entries are category trees"). Only opted-in, eligible skills are
# included (M1-D opt-in + eligibility).
# ---------------------------------------------------------------------------

def _skill_rel_path(skill_name: str) -> Optional[PurePosixPath]:
    """Return the skill's path relative to ~/.hermes/skills/ (posix), or None."""
    try:
        from tools.skill_usage import _find_skill_dir
    except Exception:
        return None
    skill_dir = _find_skill_dir(skill_name)
    if skill_dir is None:
        return None
    try:
        rel = skill_dir.resolve().relative_to(_skills_dir().resolve())
    except (OSError, ValueError):
        return None
    return PurePosixPath(rel.as_posix())


def snapshot_profile(
    skill_names: List[str], *, max_object_bytes: int = DEFAULT_MAX_OBJECT_BYTES
) -> Tuple[ObjectSet, str, Dict[str, str]]:
    """Build all HSP objects for *skill_names* + the profile-root tree.

    Returns ``(objects, root_tree_hash, skill_tree_map)`` where
    ``skill_tree_map`` is ``{skill_name: tree_hash}``. Skills whose blobs
    exceed *max_object_bytes* are skipped (surfaced via logger).

    The root tree nests category directories: a skill at ``devops/foo`` yields
    a root entry ``devops`` (tree) containing ``foo`` (tree). Flat skills yield
    a direct root entry.

    The root tree also carries a ``sync-manifest`` BLOB (design.md §2.8)
    recording the per-skill opt-in state, so opt-in is durable + cross-device
    rather than a device-local ``.usage.json`` flag. Every skill in
    ``skill_names`` is recorded ``enabled: true`` (they ARE the opted-in set);
    the manifest is the authoritative record the plane + other devices read.
    """
    from tools.skill_usage import _find_skill_dir

    objects = ObjectSet()
    skill_tree_map: Dict[str, str] = {}
    # Nested dict representing the root: {name: {"__tree__": hash} | subdict}
    root: Dict[str, Any] = {}

    for name in sorted(set(skill_names)):
        rel = _skill_rel_path(name)
        skill_dir = _find_skill_dir(name)
        if rel is None or skill_dir is None:
            continue
        try:
            tree_hash = build_tree(skill_dir, objects, max_object_bytes=max_object_bytes)
        except ValueError as e:
            logger.warning("skills_sync_client: skipping %s: %s", name, e)
            continue
        skill_tree_map[name] = tree_hash
        # Insert into the nested root structure by relative path parts.
        parts = list(rel.parts)
        node = root
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = {"__tree__": tree_hash}

    # §2.8 sync-manifest: record the opt-in state (the pushed set = enabled).
    # Only skills that actually made it into the tree are recorded, keyed by the
    # skill NAME (matching gateway-gateway's manifest shape + the read walk that
    # enumerates skill subtrees by name).
    manifest_map = {name: True for name in skill_tree_map}
    manifest_hash = objects.add(
        KIND_BLOB, build_sync_manifest_bytes(manifest_map)
    )

    root_hash = _build_root_tree(root, objects, manifest_hash=manifest_hash)
    return objects, root_hash, skill_tree_map


def _build_root_tree(
    node: Dict[str, Any], objects: ObjectSet, *, manifest_hash: Optional[str] = None
) -> str:
    """Recursively canonicalize the nested root structure into HSP trees.

    ``manifest_hash`` (only passed at the top level) adds a root-level
    ``sync-manifest`` BLOB entry (design.md §2.8) alongside the skill subtrees.
    It cannot collide with a skill dir (skill entries are trees; this is a blob).
    """
    entries: List[Dict[str, str]] = []
    for name, child in node.items():
        if isinstance(child, dict) and "__tree__" in child and len(child) == 1:
            entries.append(
                {"name": name, "kind": KIND_TREE, "hash": child["__tree__"], "mode": MODE_DIR}
            )
        else:
            sub_hash = _build_root_tree(child, objects)
            entries.append(
                {"name": name, "kind": KIND_TREE, "hash": sub_hash, "mode": MODE_DIR}
            )
    if manifest_hash is not None:
        entries.append(
            {
                "name": SYNC_MANIFEST_ENTRY_NAME,
                "kind": KIND_BLOB,
                "hash": manifest_hash,
                "mode": MODE_FILE,
            }
        )
    entries.sort(key=lambda e: e["name"])
    tree_obj = {"type": KIND_TREE, "entries": entries}
    return objects.add(KIND_TREE, canonical_json_bytes(tree_obj))


# ---------------------------------------------------------------------------
# Ref naming (contract §2.6)
# ---------------------------------------------------------------------------

def user_head_ref(owner: str) -> str:
    return f"refs/user/{owner}/HEAD"


def user_conflict_ref(owner: str, n: int) -> str:
    return f"refs/user/{owner}/conflict/{n}"


def _root_tree_of_commit(client: "HSPClient", commit_hash: str) -> str:
    """Return the tree hash referenced by a commit."""
    return client.get_commit_json(commit_hash)["tree"]


def _skill_trees_of_root(client: "HSPClient", root_tree_hash: str) -> Dict[str, str]:
    """Flatten a profile-root tree into ``{posix_rel_path: skill_tree_hash}``.

    A skill tree is any tree containing a ``SKILL.md`` blob entry. We walk the
    root tree; a subtree with a SKILL.md is treated as a skill leaf keyed by
    its path, so category nesting is preserved.
    """
    result: Dict[str, str] = {}

    def _walk(tree_hash: str, prefix: str) -> None:
        tree = client.get_tree_json(tree_hash)
        entries = tree.get("entries", [])
        has_skill_md = any(
            e.get("name") == "SKILL.md" and e.get("kind") == KIND_BLOB for e in entries
        )
        if has_skill_md and prefix:
            result[prefix] = tree_hash
            return
        for e in entries:
            if e.get("kind") == KIND_TREE:
                child_prefix = f"{prefix}/{e['name']}" if prefix else e["name"]
                _walk(e["hash"], child_prefix)

    _walk(root_tree_hash, "")
    return result


def read_manifest_of_root(
    client: "HSPClient", root_tree_hash: str
) -> Optional[Dict[str, bool]]:
    """Read the ``sync-manifest`` blob at the root of *root_tree_hash* into
    ``{name: enabled}`` (design.md §2.8), or ``None`` if there is no manifest
    entry / it is malformed.

    The manifest is a root-level BLOB entry named ``sync-manifest`` (never a
    skill subtree). This is how a device learns the cross-device opt-in state
    written by another device's push.
    """
    try:
        tree = client.get_tree_json(root_tree_hash)
    except Exception as e:
        logger.debug("skills_sync_client: manifest root read failed: %s", e)
        return None
    for e in tree.get("entries", []):
        if e.get("name") == SYNC_MANIFEST_ENTRY_NAME and e.get("kind") == KIND_BLOB:
            try:
                _kind, data = client.get_object(e["hash"])
            except Exception as ex:
                logger.debug("skills_sync_client: manifest blob fetch failed: %s", ex)
                return None
            return parse_sync_manifest(data)
    return None


def _check_version(caps: Dict[str, Any]) -> None:
    """Reject an incompatible server major version (contract §1)."""
    ver = str(caps.get("hsp_version") or "")
    major = ver.split(".", 1)[0]
    if major != HSP_VERSION:
        raise HSPError(f"incompatible HSP version {ver!r} (client speaks {HSP_VERSION})")


# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------

def push_skills(
    client: Optional["HSPClient"] = None,
    *,
    skill_names: Optional[List[str]] = None,
    identity: Optional[Dict[str, Any]] = None,
    message: str = "hermes skill sync",
) -> Dict[str, Any]:
    """Push opted-in skills to the owner's HEAD (contract §4).

    Uploads all new objects, then CAS-es ``refs/user/<owner>/HEAD``. On a 409,
    fetches the actual head, three-way merges, and retries once (§4.4 / M1-C).
    Returns a result dict; never raises for the inert / no-op cases.
    """
    if identity is None:
        identity = resolve_identity()
    owner = identity["owner"]
    if client is None:
        base = resolve_sync_base_url()
        if not base:
            return {"ok": False, "reason": "no sync base url configured", "noop": True}
        client = HSPClient(base, identity["api_key"])

    if skill_names is None:
        skill_names = list_synced_skill_names()
    if not skill_names:
        return {"ok": True, "reason": "no skills opted into sync", "noop": True}

    caps = client.capabilities()
    _check_version(caps)
    max_bytes = int(caps.get("max_object_bytes") or DEFAULT_MAX_OBJECT_BYTES)

    objects, root_hash, _ = snapshot_profile(skill_names, max_object_bytes=max_bytes)

    manifest = read_sync_manifest()
    base_head = manifest.get("head")

    # Idempotency: if the profile-root tree is unchanged since our last push,
    # there is nothing to propagate -- skip building an empty commit (contract
    # objects are immutable, so an identical tree hash means identical content).
    if base_head and manifest.get("root") == root_hash:
        return {"ok": True, "head": base_head, "reason": "unchanged", "noop": True}

    device = stable_device_id()
    parents = [base_head] if base_head else []
    commit_hash = build_commit(
        root_hash, parents, owner=owner, device=device, message=message, objects=objects
    )

    client.put_objects(objects.objects)
    ref = user_head_ref(owner)

    try:
        client.cas_ref(ref, base_head, commit_hash)
        manifest["head"] = commit_hash
        manifest["root"] = root_hash
        write_sync_manifest(manifest)
        return {"ok": True, "head": commit_hash, "pushed_objects": len(objects)}
    except HSPConflict as conflict:
        return _resolve_push_conflict(
            client, identity, conflict.actual, root_hash, commit_hash,
            objects, skill_names, message, base_head,
        )


# ---------------------------------------------------------------------------
# Conflict resolution / three-way merge (contract §4.4, M1-C)
#
# On a 409 the server hands back the actual head. We fetch it, three-way merge
# per skill against the base we forked from, reusing the origin/user/incoming
# decision semantics of skills_sync.py (_is_tracked_user_modification +
# the decision block at skills_sync.py:619-643):
#
#   * base == ours == theirs      -> nothing to do
#   * ours == base, theirs moved  -> take theirs (fast-forward incoming)
#   * theirs == base, ours moved  -> keep ours (our local edit)
#   * both moved, ours == theirs   -> converged; take either
#   * both moved, differ           -> TRUE OVERLAP -> conflict head
#
# Non-overlapping merges (each side changed a DIFFERENT skill) produce a merge
# commit (2 parents) and retry the CAS. A true overlap (both sides changed the
# SAME skill differently) is written to refs/user/<owner>/conflict/<n> and
# surfaced for out-of-band resolution.
# ---------------------------------------------------------------------------

def _resolve_push_conflict(
    client: "HSPClient",
    identity: Dict[str, Any],
    actual_head: str,
    our_root: str,
    our_commit: str,
    objects: "ObjectSet",
    skill_names: List[str],
    message: str,
    base_head: Optional[str],
) -> Dict[str, Any]:
    owner = identity["owner"]
    device = stable_device_id()

    theirs_root = _root_tree_of_commit(client, actual_head)
    base_root = _root_tree_of_commit(client, base_head) if base_head else None

    ours_trees = _skill_trees_of_root(client, our_root)
    theirs_trees = _skill_trees_of_root(client, theirs_root)
    base_trees = _skill_trees_of_root(client, base_root) if base_root else {}

    merged: Dict[str, str] = {}
    overlaps: List[str] = []
    all_paths = set(ours_trees) | set(theirs_trees) | set(base_trees)
    for path in all_paths:
        o = ours_trees.get(path)
        t = theirs_trees.get(path)
        b = base_trees.get(path)
        decision = _merge_skill(b, o, t)
        if decision == "overlap":
            overlaps.append(path)
            # Keep OURS on the surfaced conflict head; theirs is retained
            # server-side under the conflict ref for out-of-band resolution.
            if o is not None:
                merged[path] = o
        elif decision == "ours" and o is not None:
            merged[path] = o
        elif decision == "theirs" and t is not None:
            merged[path] = t
        elif decision == "either":
            merged[path] = o if o is not None else t  # type: ignore[assignment]
        # decision == "none": skill deleted on the winning side -> drop

    if overlaps:
        # TRUE OVERLAP -> write a conflict head and surface it (M1-C).
        n = _next_conflict_index(client, owner)
        conflict_ref = user_conflict_ref(owner, n)
        try:
            client.cas_ref(conflict_ref, None, our_commit)
        except HSPConflict:
            pass  # someone else grabbed this index; the head still exists
        return {
            "ok": False,
            "conflict": True,
            "conflict_ref": conflict_ref,
            "overlapping_skills": sorted(overlaps),
            "actual_head": actual_head,
            "message": (
                f"{len(overlaps)} skill(s) changed on both sides; wrote "
                f"{conflict_ref}. Resolve out-of-band (hermes sync / NAS UI)."
            ),
        }

    # Non-overlap -> build a merge commit (parents: base->actual, ours) and
    # retry the CAS against the actual head.
    merge_objects = ObjectSet()
    # Re-add our objects so the merge push is self-contained (idempotent).
    for h, (kind, data) in objects.objects.items():
        merge_objects.objects[h] = (kind, data)
    merged_root = _assemble_root_from_skill_trees(client, merged, merge_objects)
    merge_commit = build_commit(
        merged_root,
        [actual_head, our_commit],
        owner=owner,
        device=device,
        message=f"merge: {message}",
        objects=merge_objects,
    )
    client.put_objects(merge_objects.objects)
    try:
        client.cas_ref(user_head_ref(owner), actual_head, merge_commit)
    except HSPConflict as c2:
        return {
            "ok": False,
            "conflict": True,
            "message": f"merge CAS lost again (head now {c2.actual}); retry sync.",
            "actual_head": c2.actual,
        }
    manifest = read_sync_manifest()
    manifest["head"] = merge_commit
    manifest["root"] = merged_root
    write_sync_manifest(manifest)
    return {"ok": True, "head": merge_commit, "merged": True}


def _merge_skill(base: Optional[str], ours: Optional[str], theirs: Optional[str]) -> str:
    """Three-way decision for one skill's tree hash.

    Returns one of: ``ours``, ``theirs``, ``either``, ``overlap``, ``none``.
    Mirrors the origin/user/incoming decision block of skills_sync.py:619-643:
    a side "modified" the skill when its hash differs from the common base
    (analogous to ``_is_tracked_user_modification(origin, current)``).
    """
    if ours == theirs:
        return "either" if ours is not None else "none"
    ours_changed = ours != base
    theirs_changed = theirs != base
    if ours_changed and not theirs_changed:
        return "ours"
    if theirs_changed and not ours_changed:
        return "theirs"
    # both changed and differ
    return "overlap"


def _assemble_root_from_skill_trees(
    client: "HSPClient", skill_trees: Dict[str, str], objects: "ObjectSet"
) -> str:
    """Build a profile-root tree object from ``{posix_rel_path: tree_hash}``.

    Rebuilds the intermediate category trees. The referenced skill trees are
    assumed already durable (they came from either side of the merge); only
    the new intermediate/root tree objects are added to *objects*.
    """
    root: Dict[str, Any] = {}
    for path, tree_hash in skill_trees.items():
        parts = PurePosixPath(path).parts
        node = root
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = {"__tree__": tree_hash}
    return _build_root_tree(root, objects)


def _next_conflict_index(client: "HSPClient", owner: str) -> int:
    """Pick the next free conflict ref index for the owner."""
    try:
        refs = client.get_refs(f"refs/user/{owner}/conflict/")
    except HSPError:
        return 1
    used = []
    for r in refs:
        name = r.get("name", "")
        tail = name.rsplit("/", 1)[-1]
        if tail.isdigit():
            used.append(int(tail))
    return (max(used) + 1) if used else 1


# ---------------------------------------------------------------------------
# Pull
# ---------------------------------------------------------------------------

def pull_skills(
    client: Optional["HSPClient"] = None,
    *,
    identity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Pull the owner's HEAD and materialize opted-in skills to disk.

    Fetches ``refs/user/<owner>/HEAD``; if it advanced past our recorded head,
    walks the profile-root tree and writes each skill tree into
    ~/.hermes/skills/. Only paths the user has opted into (``sync: true``) are
    materialized, so a pull never resurrects a skill the user hasn't chosen.
    Best-effort; returns a result dict.
    """
    if identity is None:
        identity = resolve_identity()
    owner = identity["owner"]
    if client is None:
        base = resolve_sync_base_url()
        if not base:
            return {"ok": False, "reason": "no sync base url configured", "noop": True}
        client = HSPClient(base, identity["api_key"])

    caps = client.capabilities()
    _check_version(caps)

    refs = client.get_refs(user_head_ref(owner))
    head = None
    for r in refs:
        if r.get("name") == user_head_ref(owner):
            head = r.get("hash")
            break
    if not head:
        return {"ok": True, "reason": "no remote HEAD yet", "noop": True}

    manifest = read_sync_manifest()
    if head == manifest.get("head"):
        return {"ok": True, "reason": "already up to date", "head": head, "noop": True}

    root_tree = _root_tree_of_commit(client, head)
    remote_trees = _skill_trees_of_root(client, root_tree)

    # §2.8: reconcile local opt-in intent FROM the plane manifest, so a skill the
    # user opted in on another device becomes opted in here too (opt-in is
    # cross-device content, not a device-local flag). We only ADOPT enables from
    # the manifest for skills present in the remote tree; we never silently
    # disable a locally-enabled skill on pull (that stays the user's local call
    # until their next push reconciles it).
    reconciled_from_manifest: List[str] = []
    remote_manifest = read_manifest_of_root(client, root_tree)
    if remote_manifest:
        try:
            from tools.skill_usage import set_sync, is_curation_eligible, is_sync_enabled

            for sname, enabled in remote_manifest.items():
                if not enabled:
                    continue
                if not is_curation_eligible(sname):
                    continue
                if not is_sync_enabled(sname):
                    set_sync(sname, True)
                    reconciled_from_manifest.append(sname)
        except Exception as e:
            logger.debug("skills_sync_client: manifest opt-in reconcile failed: %s", e)

    opted_in = set(_opted_in_rel_paths())
    updated = []
    for path, tree_hash in remote_trees.items():
        # Opt-in gate on pull: only materialize skills the user chose to sync
        # (now including any adopted from the plane manifest above).
        if opted_in and path not in opted_in:
            continue
        dest = _skills_dir() / path
        materialize_tree(client, tree_hash, dest)
        updated.append(path)

    manifest["head"] = head
    write_sync_manifest(manifest)
    return {
        "ok": True,
        "head": head,
        "updated": sorted(updated),
        "opt_in_adopted": sorted(reconciled_from_manifest),
    }


def _opted_in_rel_paths() -> List[str]:
    """Relative posix paths of skills the user has opted into sync."""
    paths = []
    for name in list_synced_skill_names():
        rel = _skill_rel_path(name)
        if rel is not None:
            paths.append(rel.as_posix())
    return paths


# ---------------------------------------------------------------------------
# Gated public entrypoints (gate-and-swallow)
#
# maybe_pull_skills / maybe_push_skills clone the shape of the curator's
# maybe_run_curator (agent/curator.py:1998): best-effort, never raise, return
# a result dict or None. The DEV-PHASE gate is checked first -- sync is inert
# (no push, no pull, no-op) unless tool_gateway_admin === true on the token.
# ---------------------------------------------------------------------------

def maybe_push_skills(*, message: str = "hermes skill sync") -> Optional[Dict[str, Any]]:
    """Best-effort push if all gates pass. Returns a result dict or None.
    Never raises. Called from the debounced skill_manage push hook."""
    try:
        identity = resolve_identity()
        if not identity.get("dev_gate_ok"):
            return None  # DEV-PHASE gate: inert without tool_gateway_admin
        if not resolve_sync_base_url():
            return None
        if not list_synced_skill_names():
            return None
        return push_skills(identity=identity, message=message)
    except Exception as e:
        logger.debug("skills_sync_client: maybe_push_skills failed: %s", e, exc_info=True)
        return None


def maybe_pull_skills() -> Optional[Dict[str, Any]]:
    """Best-effort pull if all gates pass. Returns a result dict or None.
    Never raises. Invoked at the curator tick sites (gateway housekeeping loop
    + CLI startup)."""
    try:
        identity = resolve_identity()
        if not identity.get("dev_gate_ok"):
            return None  # DEV-PHASE gate: inert without tool_gateway_admin
        if not resolve_sync_base_url():
            return None
        return pull_skills(identity=identity)
    except Exception as e:
        logger.debug("skills_sync_client: maybe_pull_skills failed: %s", e, exc_info=True)
        return None


def sync_status() -> Dict[str, Any]:
    """Return a status snapshot for ``hermes sync status``. Never raises."""
    status: Dict[str, Any] = {
        "dev_gate_ok": False,
        "logged_in": False,
        "base_url": resolve_sync_base_url(),
        "opted_in_skills": [],
        "local_head": None,
        "owner": None,
    }
    try:
        identity = resolve_identity()
        status["logged_in"] = True
        status["owner"] = identity.get("owner")
        status["dev_gate_ok"] = bool(identity.get("dev_gate_ok"))
    except SyncInertError:
        pass
    except Exception as e:
        logger.debug("skills_sync_client: sync_status identity failed: %s", e)
    try:
        status["opted_in_skills"] = list_synced_skill_names()
        status["local_head"] = read_sync_manifest().get("head")
    except Exception:
        pass
    return status

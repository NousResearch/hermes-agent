#!/usr/bin/env python3
"""Skill Sync client -- the low-level sync layer (push objects + CAS a ref, pull the owner's
HEAD, three-way merge on a 409). Driven by the debounced ``skill_manage`` push hook, the curator
tick ``maybe_pull_skills`` and ``hermes sync``. Lives under tools/ so it never imports the CLI at
module load; ``skills_sync_client_wire`` / ``skills_sync_client_org`` are re-exported here.
ACCESS GATE (pre-launch): INERT unless the user is a Nous admin per the ``tool_gateway_admin``
JWT claim (NAS's misleading name for the global portal-admin permission; replace before shipping).
OPT-IN DEFAULT (provisional): local intent is the ``sync`` flag in ``.usage.json``; the DURABLE
cross-device state is the ``sync-manifest`` blob in the plane. Only ~/.hermes/skills/ skills qualify."""

from __future__ import annotations

import json
import logging
import os
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Optional, Tuple

from tools.skills_sync_client_wire import (
    DEFAULT_MAX_OBJECT_BYTES, KIND_BLOB, ObjectSet, SyncClient, SyncConflict, SyncError,
    assemble_root_from_skill_trees, build_commit, build_root_tree, build_sync_manifest_bytes, build_tree,
    checked_capabilities, materialize_tree, merge_skill, nest_skill_tree, read_manifest_of_root,
    read_ref_hash, root_tree_of_commit, skill_trees_of_root)
from tools.skills_sync_client_org import (
    ORG_DIR_NAME, list_locally_modified_org_skills, list_org_skill_names, resolve_org_identity)

logger = logging.getLogger(__name__)
# Gate claim (NAS's wire name; means "Nous admin" / Permissions.ADMIN_ACCESS). The bearer comes
# from resolve_nous_runtime_credentials(); its payload is decoded unverified to read this.
NOUS_ADMIN_CLAIM = "tool_gateway_admin"


class SyncInertError(RuntimeError):
    """Sync must no-op: not logged in, no bearer, or not a Nous admin. Caught by the gate-and-swallow hooks."""


def resolve_identity() -> Dict[str, Any]:
    """``{api_key, base_url, owner, nous_admin, claims}``; SyncInertError if not logged in / no bearer.
    ``owner`` is advisory (local ref naming; the server derives the real one). The JWT is decoded
    WITHOUT verification: safe, the claims only decide whether to attempt sync, never authz."""
    try:
        from hermes_cli.auth import resolve_nous_runtime_credentials
        creds = resolve_nous_runtime_credentials() or {}
    except Exception as e:
        raise SyncInertError(f"no Nous credentials: {e}") from e
    if not (api_key := creds.get("api_key")):
        raise SyncInertError("no bearer token available")
    try:
        import jwt  # PyJWT, a core dependency
        claims = jwt.decode(api_key, options={"verify_signature": False, "verify_exp": False}) or {}
    except Exception as e:
        logger.debug("skills_sync_client: JWT payload decode failed: %s", e)
        claims = {}
    owner = claims.get("sub") or claims.get("privy_did") or claims.get("tid") or "unknown"
    return {"api_key": api_key, "base_url": creds.get("base_url"), "owner": str(owner),
            "nous_admin": claims.get(NOUS_ADMIN_CLAIM) is True, "claims": claims}


# Configuration -- env-first so Hermes Cloud can enable sync via environment alone. Every knob:
# HERMES_SYNC_<KEY> env -> config.yaml ``sync.<key>`` -> default (base_url = the sync plane, NOT
# the inference URL; enabled; default_opt_in; org_auto_propose).
DEFAULT_SYNC_BASE_URL = "https://gateway-gateway.nousresearch.com"

_TRUE, _FALSE = {"1", "true", "yes", "on"}, {"0", "false", "no", "off", ""}


def _sync_config(key: str) -> Any:
    """``sync.<key>`` from config.yaml, or None. Lazy import: must not import the CLI at module load."""
    try:
        from hermes_cli.config import load_config
        return ((load_config() or {}).get("sync") or {}).get(key)
    except Exception as e:
        logger.debug("skills_sync_client: config sync.%s read failed: %s", key, e)
        return None


def resolve_sync_base_url() -> Optional[str]:
    """HERMES_SYNC_BASE_URL -> ``sync.base_url`` -> production plane, without trailing slash
    (``/v1/sync/`` is appended by the client). None only if the default is blanked out."""
    env = os.getenv("HERMES_SYNC_BASE_URL")
    if env and env.strip():
        return env.strip().rstrip("/")
    base = _sync_config("base_url")
    if isinstance(base, str) and base.strip():
        return base.strip().rstrip("/")
    return DEFAULT_SYNC_BASE_URL or None


def _parse_bool(value: Any) -> Optional[bool]:
    """Parse a config/env bool; None if unrecognized so callers fall through to the next layer."""
    if isinstance(value, bool) or value is None:
        return value
    s = str(value).strip().lower()
    return True if s in _TRUE else False if s in _FALSE else None


def _sync_config_bool(env_var: str, config_key: str, *, default: bool) -> bool:
    """``env_var`` -> ``sync.<config_key>`` -> default."""
    if (val := _parse_bool(os.getenv(env_var))) is not None:
        return val
    return default if (val := _parse_bool(_sync_config(config_key))) is None else val


def sync_feature_enabled() -> bool:
    """Master switch; the gate-and-swallow entrypoints ALSO require the Nous-admin gate and a base URL."""
    return _sync_config_bool("HERMES_SYNC_ENABLED", "enabled", default=False)


def sync_org_auto_propose() -> bool:
    """False (default): edits to an org skill stay LOCAL until ``hermes sync propose``. True: every
    edit is proposed right away (an admin still approves unless the editor is one)."""
    return _sync_config_bool("HERMES_SYNC_ORG_AUTO_PROPOSE", "org_auto_propose", default=False)


def sync_default_opt_in() -> bool:
    """False (default): opt-IN -- a skill syncs only after ``hermes sync enable`` or a plane manifest
    opting it in. True: opt-OUT -- every eligible skill syncs unless disabled (Hermes Cloud default)."""
    return _sync_config_bool("HERMES_SYNC_DEFAULT_OPT_IN", "default_opt_in", default=False)


# Local skill eligibility + the personal opt-in flag
def _skills_dir() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "skills"


def _org_dir() -> Path:  # local mirror root for org skills (read-only by convention)
    return _skills_dir() / ORG_DIR_NAME


def _rel_to_skills_dir(skill_dir: Path) -> Optional[Path]:
    """*skill_dir* relative to ~/.hermes/skills/, or None if outside/unresolvable."""
    try:
        return skill_dir.resolve().relative_to(_skills_dir().resolve())
    except (OSError, ValueError):
        return None


def _skill_rel_path(skill_name: str) -> Optional[PurePosixPath]:
    """The skill's path relative to ~/.hermes/skills/ (posix), or None."""
    from tools.skill_usage import _find_skill_dir
    skill_dir = _find_skill_dir(skill_name)
    rel = _rel_to_skills_dir(skill_dir) if skill_dir is not None else None
    return PurePosixPath(rel.as_posix()) if rel is not None else None


def is_sync_eligible(skill_name: str) -> bool:
    """Sync candidate (before opt-in): local, NOT bundled/hub-installed/external, NOT under ``_org/``
    (enterprise content never rides a personal push). Mirrors the curator's exclusions."""
    from tools.skill_usage import is_bundled, is_hub_installed, _find_skill_dir
    from agent.skill_utils import is_external_skill_path
    skill_dir = None if is_bundled(skill_name) or is_hub_installed(skill_name) else _find_skill_dir(skill_name)
    if skill_dir is None or is_external_skill_path(skill_dir):
        return False
    rel = _rel_to_skills_dir(skill_dir)
    return not (rel is not None and rel.parts and rel.parts[0] == ORG_DIR_NAME)


def list_synced_skill_names() -> List[str]:
    """Sorted skill names that should sync: opt-in -> eligible skills with ``sync: true``;
    opt-out (``sync_default_opt_in()``) -> every eligible skill unless ``sync: false``."""
    from tools.skill_usage import load_usage
    flags = {n: rec.get("sync") for n, rec in (load_usage() or {}).items() if isinstance(rec, dict)}
    if sync_default_opt_in():
        return sorted({n for n in _all_local_skill_names() if flags.get(n) is not False and is_sync_eligible(n)})
    return sorted({n for n, f in flags.items() if f is True and is_sync_eligible(n)})


def _all_local_skill_names() -> List[str]:
    """Every local skill name (dir under ~/.hermes/skills/ with SKILL.md; frontmatter ``name`` or dir name)."""
    names: List[str] = []
    root = _skills_dir()
    try:
        for skill_md in root.rglob("SKILL.md") if root.exists() else ():
            if skill_md.is_symlink():
                continue
            name = skill_md.parent.name
            with suppress(Exception):
                from tools.skill_usage import _read_skill_name
                name = _read_skill_name(skill_md, name)
            if name:
                names.append(name)
    except OSError as e:
        logger.debug("skills_sync_client: local skill enumeration failed: %s", e)
    return sorted(set(names))


def _opted_in_rel_paths() -> List[str]:
    """Relative posix paths of skills the user has opted into sync."""
    rels = (_skill_rel_path(name) for name in list_synced_skill_names())
    return [rel.as_posix() for rel in rels if rel is not None]


def _adopt_manifest_opt_ins(remote_manifest: Optional[Dict[str, bool]]) -> List[str]:
    """Enable local sync intent for skills the plane manifest enabled that are locally
    curation-eligible. Enables only -- a pull never silently disables. Returns adopted names."""
    adopted: List[str] = []
    try:
        from tools.skill_usage import set_sync, is_curation_eligible, is_sync_enabled
        for sname, enabled in (remote_manifest or {}).items():
            if enabled and is_curation_eligible(sname) and not is_sync_enabled(sname):
                set_sync(sname, True)
                adopted.append(sname)
    except Exception as e:
        logger.debug("skills_sync_client: manifest opt-in reconcile failed: %s", e)
    return adopted


# Device label (commit ``author.device``; advisory, never an auth input)
def _default_device_label() -> str:
    """Short hostname + random suffix (two machines can share a hostname); bare uuid if unusable."""
    import socket, uuid
    try:
        host = socket.gethostname() or ""
    except OSError:
        host = ""
    short = "".join(c for c in host.split(".")[0].strip() if c.isalnum() or c in "-_")
    return f"{short}-{uuid.uuid4().hex[:6]}" if short else uuid.uuid4().hex


def stable_device_id() -> str:
    """Per-device label at ~/.hermes/skills/.sync_device_id. An existing file always wins; else seeded
    from HERMES_SYNC_DEVICE_NAME (first use only, for Hermes Cloud) or a friendly default, then persisted."""
    with suppress(OSError):
        val = _device_id_path().read_text(encoding="utf-8").strip()
        if val:
            return val
    val = (os.environ.get("HERMES_SYNC_DEVICE_NAME") or "").strip() or _default_device_label()
    try:
        _write_device_id(val)
    except OSError as e:
        logger.debug("skills_sync_client: could not persist device id: %s", e)
    return val


def _device_id_path() -> Path:
    return _skills_dir() / ".sync_device_id"


def _write_device_id(val: str) -> None:
    _skills_dir().mkdir(parents=True, exist_ok=True)
    _device_id_path().write_text(val, encoding="utf-8")


def set_device_name(name: str) -> str:
    """Overwrite the device label with the trimmed *name*; returns it. ValueError on empty."""
    if not (cleaned := (name or "").strip()):
        raise ValueError("device name must be a non-empty string")
    _write_device_id(cleaned)
    return cleaned


# Local sync STATE: last HEAD pushed/pulled + its root tree (FULL-digest namespace). Distinct from
# the bundled manifest (skills_sync.py) and the plane's `sync-manifest`. ~/.hermes/skills/.sync_state.
_EMPTY_STATE: Dict[str, Any] = {"head": None, "skills": {}}


def _load_state_file(path: Path, what: str = "sync state read") -> Optional[Dict[str, Any]]:
    """Parse a state file; None if missing / corrupt / not a dict."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("skills_sync_client: %s failed: %s", what, e)
        return None
    return {**_EMPTY_STATE, **data} if isinstance(data, dict) else None


def read_sync_state() -> Dict[str, Any]:
    """``{"head": "sha256:...|null", "skills": {...}}``; a default on missing/corrupt. A legacy
    ``.sync_manifest`` is migrated to ``.sync_state`` on read so no head record is lost."""
    path, legacy = _skills_dir() / ".sync_state", _skills_dir() / ".sync_manifest"
    if path.exists():
        return _load_state_file(path) or dict(_EMPTY_STATE)
    data = _load_state_file(legacy, "legacy sync state migrate") if legacy.exists() else None
    if data is not None:
        write_sync_state(data)
        with suppress(OSError):
            legacy.unlink()
    return data if data is not None else dict(_EMPTY_STATE)


def write_sync_state(data: Dict[str, Any]) -> None:
    """Write the local sync state atomically. Best-effort."""
    try:
        from utils import atomic_write_text
        atomic_write_text(_skills_dir() / ".sync_state",
                          json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False),
                          tmp_prefix=".sync_state_")
    except Exception as e:
        logger.debug("skills_sync_client: sync state write failed: %s", e)


# Profile snapshot -- the root tree mirrors each skill's relative path (categories = intermediate trees).
def snapshot_profile(skill_names: List[str], *, max_object_bytes: int = DEFAULT_MAX_OBJECT_BYTES,
                     ) -> Tuple[ObjectSet, str, Dict[str, str]]:
    """All objects for *skill_names* + profile root -> ``(objects, root_hash, {name: tree_hash})``.
    Oversized skills are skipped (logged); the root carries a ``sync-manifest`` blob of included skills."""
    from tools.skill_usage import _find_skill_dir
    objects = ObjectSet()
    skill_tree_map: Dict[str, str] = {}
    root: Dict[str, Any] = {}
    for name in sorted(set(skill_names)):
        rel, skill_dir = _skill_rel_path(name), _find_skill_dir(name)
        if rel is None or skill_dir is None:
            continue
        try:
            tree_hash = build_tree(skill_dir, objects, max_object_bytes=max_object_bytes)
        except ValueError as e:
            logger.warning("skills_sync_client: skipping %s: %s", name, e)
            continue
        skill_tree_map[name] = tree_hash
        nest_skill_tree(root, rel.parts, tree_hash)
    manifest_hash = objects.add(KIND_BLOB, build_sync_manifest_bytes(dict.fromkeys(skill_tree_map, True)))
    return objects, build_root_tree(root, objects, manifest_hash=manifest_hash), skill_tree_map


# Personal refs, push, pull
def user_head_ref(owner: str) -> str:
    return f"refs/user/{owner}/HEAD"


def _personal_client(identity: Optional[Dict[str, Any]], client: Optional[SyncClient],
                     ) -> Tuple[Dict[str, Any], Optional[SyncClient]]:
    """Identity + client for a personal sync op; ``client`` is None when no base URL is configured."""
    identity = identity if identity is not None else resolve_identity()
    if client is None and (base := resolve_sync_base_url()):
        client = SyncClient(base, identity["api_key"])
    return identity, client


_NO_BASE_URL = {"ok": False, "reason": "no sync base url configured", "noop": True}


def push_skills(client: Optional[SyncClient] = None, *, skill_names: Optional[List[str]] = None,
                identity: Optional[Dict[str, Any]] = None, message: str = "hermes skill sync") -> Dict[str, Any]:
    """Push opted-in skills to ``refs/user/<owner>/HEAD`` (upload objects, CAS HEAD). 409 with an actual
    head -> three-way merge + one retry; 409 on a NON-EXISTENT ref (stale local head) -> CAS as a create."""
    identity, client = _personal_client(identity, client)
    if client is None:
        return dict(_NO_BASE_URL)
    owner = identity["owner"]
    skill_names = list_synced_skill_names() if skill_names is None else skill_names
    if not skill_names:
        return {"ok": True, "reason": "no skills opted into sync", "noop": True}
    _caps, max_bytes = checked_capabilities(client)
    objects, root_hash, _ = snapshot_profile(skill_names, max_object_bytes=max_bytes)
    state = read_sync_state()
    base_head = state.get("head")
    # Idempotency: objects are immutable, so an unchanged root hash means identical content.
    if base_head and state.get("root") == root_hash:
        return {"ok": True, "head": base_head, "reason": "unchanged", "noop": True}
    commit_hash = build_commit(root_hash, [base_head] if base_head else [], owner=owner,
                               device=stable_device_id(), message=message, objects=objects)
    client.put_objects(objects.objects)
    ref = user_head_ref(owner)
    result = {"ok": True, "head": commit_hash, "pushed_objects": len(objects)}
    try:
        client.cas_ref(ref, base_head, commit_hash)
    except SyncConflict as conflict:
        if conflict.actual:
            return _resolve_push_conflict(client, identity, conflict.actual, root_hash, commit_hash,
                                          objects, message, base_head)
        client.cas_ref(ref, None, commit_hash)
        result["recovered_stale_head"] = True
    write_sync_state({**state, "head": commit_hash, "root": root_hash})
    return result


def _resolve_push_conflict(client: SyncClient, identity: Dict[str, Any], actual_head: str, our_root: str,
                           our_commit: str, objects: ObjectSet, message: str, base_head: Optional[str],
                           ) -> Dict[str, Any]:
    """Per-skill three-way merge against the forked base (merge_skill): different skills changed ->
    merge commit + CAS retry; SAME skill -> OVERLAP -> refs/user/<owner>/conflict/<n> for out-of-band."""
    owner = identity["owner"]
    ours_trees = skill_trees_of_root(client, our_root)
    theirs_trees = skill_trees_of_root(client, root_tree_of_commit(client, actual_head))
    base_trees = skill_trees_of_root(client, root_tree_of_commit(client, base_head)) if base_head else {}
    merged: Dict[str, str] = {}
    overlaps: List[str] = []
    for path in set(ours_trees) | set(theirs_trees) | set(base_trees):
        o, t = ours_trees.get(path), theirs_trees.get(path)
        decision = merge_skill(base_trees.get(path), o, t)
        if decision == "overlap":
            overlaps.append(path)
        # overlap keeps OURS on the surfaced conflict head (theirs stays server-side);
        # "none" = deleted on the winning side -> drop.
        pick = {"overlap": o, "ours": o, "theirs": t, "either": o if o is not None else t}.get(decision)
        if pick is not None:
            merged[path] = pick
    if overlaps:
        conflict_ref = f"refs/user/{owner}/conflict/{_next_conflict_index(client, owner)}"
        with suppress(SyncConflict):  # someone else grabbed this index; the head still exists
            client.cas_ref(conflict_ref, None, our_commit)
        return {"ok": False, "conflict": True, "conflict_ref": conflict_ref, "overlapping_skills": sorted(overlaps),
                "actual_head": actual_head, "message": (f"{len(overlaps)} skill(s) changed on both sides; wrote "
                                                        f"{conflict_ref}. Resolve out-of-band (hermes sync / NAS UI).")}
    # Merge commit (parents: actual, ours); re-add our objects so the merge push is self-contained.
    merge_objects = ObjectSet()
    merge_objects.objects |= objects.objects
    merged_root = assemble_root_from_skill_trees(merged, merge_objects)
    merge_commit = build_commit(merged_root, [actual_head, our_commit], owner=owner,
                                device=stable_device_id(), message=f"merge: {message}", objects=merge_objects)
    client.put_objects(merge_objects.objects)
    try:
        client.cas_ref(user_head_ref(owner), actual_head, merge_commit)
    except SyncConflict as c2:
        return {"ok": False, "conflict": True, "actual_head": c2.actual,
                "message": f"merge CAS lost again (head now {c2.actual}); retry sync."}
    write_sync_state({**read_sync_state(), "head": merge_commit, "root": merged_root})
    return {"ok": True, "head": merge_commit, "merged": True}


def _next_conflict_index(client: SyncClient, owner: str) -> int:
    """Next free ``conflict/<n>`` index for the owner."""
    try:
        refs = client.get_refs(f"refs/user/{owner}/conflict/")
    except SyncError:
        return 1
    return 1 + max((int(t) for t in (r.get("name", "").rsplit("/", 1)[-1] for r in refs) if t.isdigit()), default=0)


def pull_skills(client: Optional[SyncClient] = None, *, identity: Optional[Dict[str, Any]] = None,
                ) -> Dict[str, Any]:
    """Pull the owner's HEAD (if it advanced) and materialize opted-in skills. Opt-ins are first
    adopted FROM the plane manifest; only opted-in paths are written (never resurrects a skill)."""
    identity, client = _personal_client(identity, client)
    if client is None:
        return dict(_NO_BASE_URL)
    owner = identity["owner"]
    checked_capabilities(client)
    head = read_ref_hash(client, user_head_ref(owner))
    if not head:
        return {"ok": True, "reason": "no remote HEAD yet", "noop": True}
    state = read_sync_state()
    if head == state.get("head"):
        return {"ok": True, "reason": "already up to date", "head": head, "noop": True}
    root_tree = root_tree_of_commit(client, head)
    remote_trees = skill_trees_of_root(client, root_tree)
    adopted = _adopt_manifest_opt_ins(read_manifest_of_root(client, root_tree))
    opted_in = set(_opted_in_rel_paths())
    updated = [path for path in remote_trees if not opted_in or path in opted_in]
    for path in updated:
        materialize_tree(client, remote_trees[path], _skills_dir() / path)
    write_sync_state({**state, "head": head})
    return {"ok": True, "head": head, "updated": sorted(updated), "opt_in_adopted": sorted(adopted)}


# Gated public entrypoints (gate-and-swallow, like maybe_run_curator): never raise; dict or None.
def _gate_and_swallow(op: str, run: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]):
    """Run *run(identity)* only if all gates hold (Nous admin, feature on, base URL); None if inert/error."""
    try:
        identity = resolve_identity()
        if not identity.get("nous_admin") or not sync_feature_enabled() or not resolve_sync_base_url():
            return None
        return run(identity)
    except Exception as e:
        logger.debug("skills_sync_client: %s failed: %s", op, e, exc_info=True)
        return None


def maybe_push_skills(*, message: str = "hermes skill sync") -> Optional[Dict[str, Any]]:
    """Best-effort push (debounced skill_manage hook). Never raises."""
    return _gate_and_swallow("maybe_push_skills", lambda identity: push_skills(
        identity=identity, message=message) if list_synced_skill_names() else None)


def maybe_pull_skills() -> Optional[Dict[str, Any]]:
    """Best-effort pull (curator tick sites: gateway housekeeping + CLI startup). Never raises."""
    return _gate_and_swallow("maybe_pull_skills", lambda identity: pull_skills(identity=identity))


def sync_status() -> Dict[str, Any]:
    """Snapshot for ``hermes sync status``; never raises. ``org_available`` False = not in a shared org."""
    status: Dict[str, Any] = {"nous_admin": False, "logged_in": False, "feature_enabled": sync_feature_enabled(),
                              "default_opt_in": sync_default_opt_in(), "base_url": resolve_sync_base_url(),
                              "opted_in_skills": [], "local_head": None, "owner": None, "org_available": False,
                              "org_id": None, "org_role": None, "org_skills": [], "org_skills_modified": []}
    try:
        identity = resolve_identity()
        status.update(logged_in=True, owner=identity.get("owner"), nous_admin=bool(identity.get("nous_admin")))
    except SyncInertError:
        pass
    except Exception as e:
        logger.debug("skills_sync_client: sync_status identity failed: %s", e)
    with suppress(Exception):
        status["opted_in_skills"] = list_synced_skill_names()
        status["local_head"] = read_sync_state().get("head")
    try:
        org_identity = resolve_org_identity()
        status.update(org_available=True, org_id=org_identity.get("org_id"), org_role=org_identity.get("org_role"),
                      org_skills=list_org_skill_names(),
                      org_skills_modified=list_locally_modified_org_skills(org_identity.get("org_id")))
    except SyncInertError:
        pass
    except Exception as e:
        logger.debug("skills_sync_client: sync_status org lookup failed: %s", e)
    return status


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from datetime import datetime  # noqa: F401,E402
import hashlib  # noqa: F401,E402
import time  # noqa: F401,E402
from datetime import timezone  # noqa: F401,E402

KIND_COMMIT = "commit"

KIND_TREE = "tree"

MODE_DIR = "dir"

MODE_EXEC = "exec"

MODE_FILE = "file"

SYNC_MANIFEST_TYPE = "sync-manifest"

def dev_gate_open() -> bool:
    """Whether the access gate permits sync. Never raises."""
    try:
        return bool(resolve_identity().get("nous_admin"))
    except SyncInertError:
        return False
    except Exception as e:
        logger.debug("skills_sync_client: dev_gate_open check failed: %s", e)
        return False

def org_sync_available() -> bool:
    """True iff this token can see the org-skill surface (multi-member org)."""
    try:
        resolve_org_identity()
        return True
    except Exception:
        return False

def user_conflict_ref(owner: str, n: int) -> str:
    return f"refs/user/{owner}/conflict/{n}"


_PLUGIN_COMPAT_LAZY = {
    'ARTIFACT_TYPE_SKILL': ('tools.skills_sync_client_wire', 'ARTIFACT_TYPE_SKILL'),
    'SYNC_MANIFEST_ENTRY_NAME': ('tools.skills_sync_client_wire', 'SYNC_MANIFEST_ENTRY_NAME'),
    'SYNC_MANIFEST_VERSION': ('tools.skills_sync_client_wire', 'SYNC_MANIFEST_VERSION'),
    'WIRE_VERSION': ('tools.skills_sync_client_wire', 'WIRE_VERSION'),
    'canonical_json_bytes': ('tools.skills_sync_client_wire', 'canonical_json_bytes'),
    'maybe_pull_org_skills': ('tools.skills_sync_client_org', 'maybe_pull_org_skills'),
    'org_head_ref': ('tools.skills_sync_client_org', 'org_head_ref'),
    'org_skill_is_locally_modified': ('tools.skills_sync_client_org', 'org_skill_is_locally_modified'),
    'parse_sync_manifest': ('tools.skills_sync_client_wire', 'parse_sync_manifest'),
    'propose_skill': ('tools.skills_sync_client_org', 'propose_skill'),
    'pull_org_skills': ('tools.skills_sync_client_org', 'pull_org_skills'),
    'wire_address': ('tools.skills_sync_client_wire', 'wire_address'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----

"""Org-shared skills: org pull + propose (``~/.hermes/skills/_org/<org_id>/``).
Org skills live in a DISTINCT local namespace (read-only to the runtime; a local edit is a personal
fork until proposed); the canonical set is ``refs/org/<org_id>/HEAD`` with the SAME object model.
PERSONAL-ORG GATE: NAS stamps ``org_role`` ONLY for multi-member orgs; no claim => pull/propose
raise SyncInertError and personal sync is untouched. ``propose_skill`` must stay non-interactive.
Module state (``_skills_dir``, ``_org_dir``, base URL, device id) stays in ``tools.skills_sync_client``
and is read lazily so tests can monkeypatch it."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Optional

from tools.skills_sync_client_wire import (
    ObjectSet, SyncClient, SyncConflict, SyncError, assemble_root_from_skill_trees, build_commit, build_tree,
    checked_capabilities, materialize_tree, read_ref_hash, root_tree_of_commit, skill_trees_of_root)

logger = logging.getLogger("tools.skills_sync_client")
ORG_DIR_NAME = "_org"
# Propose re-splices onto a moved org HEAD at most this many times. Small: contention means
# other members are actively proposing; unbounded would spin.
_ORG_CAS_MAX_ATTEMPTS = 5


def _ssc():
    from tools import skills_sync_client
    return skills_sync_client


def org_head_ref(org_id: str) -> str:
    return f"refs/org/{org_id}/HEAD"


def resolve_org_identity() -> Dict[str, Any]:
    """``resolve_identity()`` + ``org_id``/``org_role``; SyncInertError without an ``org_role`` claim
    (personal org / old issuer): org sync unavailable, NOT an error."""
    ssc = _ssc()
    identity = ssc.resolve_identity()
    claims = identity.get("claims") or {}
    org_id, org_role = claims.get("org_id"), claims.get("org_role")
    if not org_id:
        raise ssc.SyncInertError("no organisation associated with this account")
    if not isinstance(org_role, str) or not org_role:
        raise ssc.SyncInertError("this account isn't a member of a shared organisation")
    identity.update(org_id=str(org_id), org_role=org_role)
    return identity


def _org_client(identity: Optional[Dict[str, Any]], client: Optional[SyncClient]):
    """(identity, client, max_object_bytes) for an org operation; SyncInertError when the base URL is
    missing or the server lacks the ``org`` feature."""
    ssc = _ssc()
    identity = identity or resolve_org_identity()
    if client is None:
        if not (base_url := ssc.resolve_sync_base_url()):
            raise ssc.SyncInertError("no sync base URL configured")
        client = SyncClient(base_url, identity["api_key"])
    caps, max_bytes = checked_capabilities(client)
    if "org" not in (caps.get("features") or []):
        raise ssc.SyncInertError("this server does not support org-shared skills")
    return identity, client, max_bytes


def _read_org_head(client: SyncClient, org_id: str) -> Optional[str]:
    """Current org HEAD, or None. MUST read through the ORG endpoint."""
    return read_ref_hash(client, org_head_ref(org_id), org_scope=True)


# Local mirror sidecars
def _mirror_root(org_id: Optional[str]) -> Path:
    """``<_org_dir>/<org_id>``; the org-level ``_org/`` root itself when *org_id* is None."""
    return _ssc()._org_dir() / org_id if org_id else _ssc()._org_dir()


def _write_sidecar(what: str, path_fn: Callable[[], Path], text: str) -> None:
    """Best-effort sidecar write (path resolution included); never raises."""
    try:
        path = path_fn()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except Exception as e:
        logger.debug("skills_sync_client: %s write failed: %s", what, e)


def _skill_dir_fingerprint(path: Path) -> str:
    """Content hash of a skill dir (sorted relative path + bytes; mtime-independent). "" on read failure."""
    h = hashlib.sha256()
    try:
        for f in sorted(p for p in path.rglob("*") if p.is_file()):
            h.update(str(f.relative_to(path)).replace("\\", "/").encode("utf-8"))
            h.update(b"\0")
            h.update(f.read_bytes())
            h.update(b"\0")
    except OSError as e:
        logger.debug("skills_sync_client: fingerprint failed for %s: %s", path, e)
        return ""
    return h.hexdigest()


def _sidecar_path(org_id: Optional[str], const: str) -> Path:
    """``<mirror>/<agent.skill_utils.<const>>`` (org-level when org_id is None)."""
    import agent.skill_utils as sku
    return _mirror_root(org_id) / getattr(sku, const)


def _read_org_baseline(org_id: str) -> Dict[str, Any]:
    """The baseline sidecar: upstream fingerprint + tree of each mirrored skill ({} if absent)."""
    try:
        return json.loads(_sidecar_path(org_id, "ORG_BASELINE_FILE").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_org_baseline(org_id: str, baseline: Dict[str, Any]) -> None:
    _write_sidecar("baseline", lambda: _sidecar_path(org_id, "ORG_BASELINE_FILE"),
                   json.dumps(baseline, indent=2, sort_keys=True))


def _write_org_provenance(org_id: str, data: Dict[str, Any]) -> None:
    _write_sidecar("org provenance", lambda: _sidecar_path(org_id, "ORG_PROVENANCE_FILE"),
                   json.dumps(data, indent=2))


def _write_active_org_marker(org_id: str) -> None:
    """Record which org's mirror may resolve (agent/skill_utils.read_active_org_id)."""
    _write_sidecar("active-org marker", lambda: _sidecar_path(None, "ORG_ACTIVE_MARKER"), org_id)


def _clear_active_org_marker() -> None:
    """Remove the active-org marker so org skills stop resolving."""
    try:
        marker = _sidecar_path(None, "ORG_ACTIVE_MARKER")
        if marker.exists():
            marker.unlink()
            logger.info("skills_sync_client: cleared active-org marker "
                        "(token has no org workflow); org skills no longer resolve")
    except Exception as e:
        logger.debug("skills_sync_client: marker clear failed: %s", e)


def org_skill_is_locally_modified(skill_rel_path: str, org_id: str) -> bool:
    """Local copy differs from upstream's fingerprint. No baseline (pre-existing mirror) => unmodified."""
    dest = _mirror_root(org_id) / PurePosixPath(skill_rel_path)
    entry = _read_org_baseline(org_id).get(skill_rel_path) or {}
    recorded = entry.get("fingerprint") if isinstance(entry, dict) else entry
    return dest.is_dir() and bool(recorded) and _skill_dir_fingerprint(dest) != recorded


def _active_org_id() -> Optional[str]:
    from agent.skill_utils import read_active_org_id
    return read_active_org_id(_ssc()._skills_dir())


def list_locally_modified_org_skills(org_id: Optional[str] = None) -> List[str]:
    """Org skills with local edits that upstream has not seen."""
    try:
        org_id = org_id or _active_org_id()
        if not org_id:
            return []
        return sorted(rel for rel in _read_org_baseline(org_id) if org_skill_is_locally_modified(rel, org_id))
    except Exception as e:
        logger.debug("skills_sync_client: modified-scan failed: %s", e)
        return []


def list_org_skill_names() -> List[str]:
    """Skill names present in the local org mirror (empty when none pulled)."""
    names: List[str] = []
    try:
        org_id = _active_org_id()
        root = _mirror_root(org_id) if org_id else None
        if root and root.is_dir():
            names = [str(rel).replace("\\", "/") for rel in (p.parent.relative_to(root) for p in root.rglob("SKILL.md"))
                     if rel.parts]
    except Exception as e:
        logger.debug("skills_sync_client: org skill listing failed: %s", e)
    return sorted(names)


# Pull / propose
def pull_org_skills(client: Optional[SyncClient] = None, *, identity: Optional[Dict[str, Any]] = None,
                    ) -> Dict[str, Any]:
    """Pull the org canonical set into the mirror (fast-forward only, no client merge). A skill with
    LOCAL edits is never clobbered: skipped, and in ``conflicted`` when upstream also moved.
    Returns ``{ok, org_id, head, updated, conflicted}``."""
    identity = identity or resolve_org_identity()
    if "org_id" not in identity:
        raise _ssc().SyncInertError("no organisation context available")
    identity, client, _ = _org_client(identity, client)
    org_id = identity["org_id"]
    head = _read_org_head(client, org_id)
    # Marker written HERE: only after the token's org_id + org_role were verified, so a stale mirror
    # from a previous org stops resolving on a pull under another org.
    _write_active_org_marker(org_id)
    if not head:
        return {"ok": True, "org_id": org_id, "head": None, "updated": []}
    head_commit = client.get_commit_json(head, org_scope=True)
    updated, conflicted = [], []
    baseline = _read_org_baseline(org_id)
    for rel_path, tree_hash in sorted(skill_trees_of_root(client, head_commit["tree"], org_scope=True).items()):
        dest = _mirror_root(org_id) / PurePosixPath(rel_path)
        try:
            if dest.exists():
                if org_skill_is_locally_modified(rel_path, org_id):
                    if (baseline.get(rel_path) or {}).get("tree") != tree_hash:
                        conflicted.append(rel_path)
                    continue
                shutil.rmtree(dest)
            materialize_tree(client, tree_hash, dest, org_scope=True)  # creates dest
            baseline[rel_path] = {"fingerprint": _skill_dir_fingerprint(dest), "tree": tree_hash}
            updated.append(rel_path)
        except Exception as e:
            logger.warning("skills_sync_client: org skill materialize failed for %s: %s", rel_path, e)
    # Provenance for the skill_view header: the HEAD author is token-verified by the plane at
    # push time, so it is trustworthy to display.
    author = head_commit.get("author") or {}
    _write_org_provenance(org_id, {"org_id": org_id, "head": head, "author_user_id": author.get("owner", ""),
                                   "author_device": author.get("device", ""), "ts": head_commit.get("ts", ""),
                                   "skills": updated})
    _write_org_baseline(org_id, baseline)
    if conflicted:
        logger.warning("skills_sync_client: %d org skill(s) have local edits AND upstream "
                       "changes; left untouched: %s", len(conflicted), ", ".join(conflicted))
    return {"ok": True, "org_id": org_id, "head": head, "updated": updated, "conflicted": conflicted}


def propose_skill(skill_name: str, client: Optional[SyncClient] = None, *,
                  identity: Optional[Dict[str, Any]] = None, message: Optional[str] = None) -> Dict[str, Any]:
    """Propose a local skill to the org canonical set: an org-scoped commit splicing that ONE skill
    subtree into the current org HEAD (per-skill deltas, never a wholesale replace), uploaded with
    ``?scope=org``, then CAS. ADMIN/OWNER -> ``{ok, merged: True}``; MEMBER -> 202 -> ``{ok,
    proposal_pending: True, proposal_id, ref}`` (never presented as live). If HEAD moves before the
    CAS the skill is re-spliced onto the NEW head (replaying the old root would drop others' skills)."""
    ssc = _ssc()
    identity, client, max_bytes = _org_client(identity, client)
    org_id = identity["org_id"]
    rel = ssc._skill_rel_path(skill_name)
    if rel is None:
        raise SyncError(f"skill '{skill_name}' not found under the skills dir")
    skill_dir = ssc._skills_dir() / rel
    if not (skill_dir / "SKILL.md").exists():
        raise SyncError(f"skill '{skill_name}' has no SKILL.md")
    objects = ObjectSet()
    skill_tree = build_tree(skill_dir, objects, max_object_bytes=max_bytes)
    for attempt in range(1, _ORG_CAS_MAX_ATTEMPTS + 1):
        base_head = _read_org_head(client, org_id)
        skill_map = {} if not base_head else skill_trees_of_root(
            client, root_tree_of_commit(client, base_head, org_scope=True), org_scope=True)
        skill_map[str(rel)] = skill_tree
        root_hash = assemble_root_from_skill_trees(skill_map, objects)
        commit_hash = build_commit(root_hash, [base_head] if base_head else [], owner=identity["owner"],
                                   device=ssc.stable_device_id(), message=message or f"propose {skill_name}",
                                   objects=objects)
        client.put_objects(objects.objects, org_scope=True)
        try:
            result = client.cas_ref(org_head_ref(org_id), base_head, commit_hash)
            break
        except SyncConflict as conflict:
            if attempt >= _ORG_CAS_MAX_ATTEMPTS:
                raise SyncError("the organisation's skills changed while this was being "
                                f"proposed, and {attempt} attempts to catch up all lost "
                                "the race — run the command again", status=409) from conflict
            logger.debug("propose_skill: org HEAD moved (actual=%r), re-splicing (attempt %d)",
                         conflict.actual, attempt)
    if result.get("proposal_pending"):
        return {"ok": True, "proposal_pending": True, "proposal_id": result.get("proposal_id"),
                "ref": result.get("ref"), "commit": commit_hash, "org_id": org_id}
    return {"ok": True, "merged": True, "head": result.get("hash", commit_hash),
            "commit": commit_hash, "org_id": org_id}


def maybe_pull_org_skills() -> Optional[Dict[str, Any]]:
    """Best-effort org pull if all gates hold; never raises, None when inert. Marker hygiene: a token
    that VERIFIABLY lacks the org claim clears the active-org marker (org skills stop resolving); an
    unresolvable identity (offline, logged out) leaves it alone so pulled org skills keep working."""
    ssc = _ssc()
    try:
        identity = resolve_org_identity()
    except ssc.SyncInertError:
        with suppress(Exception):
            if not (ssc.resolve_identity().get("claims") or {}).get("org_role"):
                _clear_active_org_marker()
        return None
    except Exception as e:
        logger.debug("skills_sync_client: maybe_pull_org_skills inert/failed: %s", e)
        return None
    try:
        if not ssc.sync_feature_enabled() or not ssc.resolve_sync_base_url():
            return None
        return pull_org_skills(identity=identity)
    except Exception as e:
        logger.debug("skills_sync_client: maybe_pull_org_skills inert/failed: %s", e)
        return None

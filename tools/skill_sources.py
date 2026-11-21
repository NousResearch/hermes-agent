"""Provenance for locally authored skills: report when a recorded source has changed.

``hermes skills check`` (``hermes_cli.skills_hub.do_check``) answers the question for
hub-installed skills from the hub lock file
(``tools.skills_hub_install.check_for_skill_updates``). Skills authored by ``/learn``
or by hand have no install record: they declare where they came from in their own
frontmatter, under ``metadata.hermes.sources``, and nothing reads it. A skill learned
from a documentation page or a repository therefore goes stale in silence.

This module is the reader. It parses that block, checks each recorded source against
the live one, and keeps the small amount of state a ``url`` or ``path`` check needs
(previous content hash, ETag, Last-Modified) in a profile-scoped sidecar next to the
other skill sidecars — never in ``SKILL.md``, which is skill content and gets shared.

Statuses per source:

``unchanged``          the source still matches what was recorded or last checked
``changed``            the source moved since it was read or last checked
``unreachable``        the source could not be read (network, HTTP error, missing file)
``baseline_recorded``  first check of a content-checked source, or the first check after the
                       skill was refreshed; the baseline is now stored
``skipped``            deliberately not checked (policy, unsupported form, no revision)

A ``changed`` verdict is **sticky**: the pre-change hash stays recorded, so every later check
keeps reporting ``changed`` until the skill is actually refreshed. Re-baselining is keyed on
the entry's ``retrieved`` date — the frontmatter's own statement of when that source was last
read — so refreshing one source clears that source's alert and nothing else's. (Re-keying on
the SKILL.md hash instead would let any unrelated edit — a typo, a new tag — silently clear
every alert the skill carries.) The cost of that choice: a same-day refresh, where ``retrieved``
does not move, keeps showing ``changed``. That direction is deliberate — a stale skill silently
reading as fresh is the failure this exists to prevent.

Only local, non-bundled, non-hub, non-protected skills are in scope, and every fetch goes
through the existing URL safety and website policy code. Nothing here calls a model, and
nothing writes to a skill.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import urlparse

from agent.skill_utils import get_skills_dir, iter_skill_index_files, parse_frontmatter
from utils import atomic_write_text

logger = logging.getLogger(__name__)

STATE_FILENAME = ".source_state.json"
STATE_VERSION = 1
# Fetches are bounded: a skill check is a foreground command, not a crawl.
HTTP_TIMEOUT_SECONDS = 10.0
MAX_BODY_BYTES = 2 * 1024 * 1024
GIT_TIMEOUT_SECONDS = 15
_SOURCE_KINDS = ("url", "repo", "path")
_HTTP_SCHEMES = ("http", "https")
# Repository URLs we can check without a credential dance or an interactive prompt.
_CHECKABLE_REPO_SCHEMES = ("http", "https", "file")
_REMOTE_REPO_SCHEMES = ("ssh", "git")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_INTER_TAG_RE = re.compile(r">\s+<")
_WHITESPACE_RE = re.compile(r"\s+")


# ── Reading the declared sources ────────────────────────────────────────────


def _as_text(value: Any) -> str:
    """A frontmatter scalar as text; YAML turns ``2026-10-22`` into a date."""
    if value is None:
        return ""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value).strip()


def _repo_url_like(value: str) -> bool:
    """True when *value* can name a git repository at all (checked later for reachability)."""
    if not value or value.startswith("-"):  # a leading dash would be read as an option
        return False
    parsed = urlparse(value)
    if parsed.scheme in _HTTP_SCHEMES:
        return bool(parsed.netloc)
    if parsed.scheme in ("file",) + _REMOTE_REPO_SCHEMES:
        return bool(parsed.path or parsed.netloc)
    if parsed.scheme:
        return False
    return True  # a bare filesystem path


def _normalize_entry(item: Any) -> Optional[dict[str, Any]]:
    """One ``sources`` list item as ``{kind, value, revision, retrieved}``, or None if unusable.

    Unusable means: not a mapping, not exactly one of url/repo/path, a url that is not
    http(s), a repo that cannot name a repository, or a ``revision`` on a non-repo entry.
    A bad entry is skipped silently — a malformed block must never break skill loading.
    """
    if not isinstance(item, dict):
        return None
    present = [kind for kind in _SOURCE_KINDS if _as_text(item.get(kind))]
    if len(present) != 1:
        return None
    kind = present[0]
    value = _as_text(item.get(kind))
    if kind == "url":
        parsed = urlparse(value)
        if parsed.scheme.lower() not in _HTTP_SCHEMES or not parsed.netloc:
            return None
    elif kind == "repo":
        if not _repo_url_like(value):
            return None
    elif "\x00" in value:
        return None
    revision = item.get("revision")
    if revision is not None:
        if kind != "repo" or not _as_text(revision):
            return None
        revision = _as_text(revision)
    return {"kind": kind, "value": value,
            "revision": _as_text(revision) or None,
            "retrieved": _as_text(item.get("retrieved")) or None}


def read_sources(skill_dir: Path) -> list[dict[str, Any]]:
    """Valid ``metadata.hermes.sources`` entries for one skill directory.

    Returns ``[]`` for an absent block, a block that is not a list, unreadable file or
    unparseable frontmatter. Never raises: this runs over whatever the user has on disk.
    """
    try:
        content = (Path(skill_dir) / "SKILL.md").read_text(encoding="utf-8-sig", errors="replace")
        frontmatter, _ = parse_frontmatter(content)
    except (OSError, ValueError):
        return []
    metadata = frontmatter.get("metadata")
    hermes = metadata.get("hermes") if isinstance(metadata, dict) else None
    raw = hermes.get("sources") if isinstance(hermes, dict) else None
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    entries = (_normalize_entry(item) for item in raw)
    return [entry for entry in entries if entry]


def _skill_name(skill_md: Path) -> str:
    """Frontmatter name when present, else the directory name — what the user sees."""
    try:
        frontmatter, _ = parse_frontmatter(skill_md.read_text(encoding="utf-8-sig", errors="replace"))
    except (OSError, ValueError):
        return skill_md.parent.name
    return _as_text(frontmatter.get("name")) or skill_md.parent.name


def iter_skill_sources(name: Optional[str] = None) -> Iterator[tuple[str, Path, list[dict[str, Any]]]]:
    """Yield ``(skill_name, skill_dir, entries)`` for in-scope skills that declare sources.

    In scope: skills under the profile's own skills directory that are neither
    hub-installed (already covered), bundled/protected (never editable by the agent, so
    a "refresh it with /learn" hint would be wrong) nor external (owned elsewhere).
    """
    from tools import skill_usage

    root = get_skills_dir()
    if not root.is_dir():
        return
    for skill_md in iter_skill_index_files(root, "SKILL.md"):
        skill_dir = skill_md.parent
        skill_name = _skill_name(skill_md)
        if name and name not in (skill_name, skill_dir.name):
            continue
        if skill_usage.is_hub_installed(skill_name) or skill_usage.is_bundled(skill_name) \
                or skill_usage.is_protected_builtin(skill_name):
            continue
        entries = read_sources(skill_dir)
        if entries:
            yield skill_name, skill_dir, entries


# ── Baseline sidecar ────────────────────────────────────────────────────────


def state_path() -> Path:
    """``<skills dir>/.source_state.json`` — beside ``.usage.json``, never in a skill."""
    return get_skills_dir() / STATE_FILENAME


def load_state() -> dict[str, Any]:
    """The recorded baselines; an unreadable or foreign file reads as empty, never an error."""
    try:
        data = json.loads(state_path().read_text(encoding="utf-8-sig", errors="replace"))
    except (OSError, ValueError):
        return {"version": STATE_VERSION, "sources": {}}
    sources = data.get("sources") if isinstance(data, dict) else None
    if data.get("version") != STATE_VERSION or not isinstance(sources, dict):
        return {"version": STATE_VERSION, "sources": {}}
    return {"version": STATE_VERSION, "sources": sources}


def save_state(state: dict[str, Any]) -> bool:
    """Write the baselines atomically under the same cross-process lock the usage sidecar uses."""
    from tools.skill_usage import skill_file_lock

    path = state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with skill_file_lock(path.with_suffix(".json.lock")):
            atomic_write_text(path, json.dumps(state, indent=2, sort_keys=True, ensure_ascii=False),
                              tmp_prefix=".source_state_")
        return True
    except OSError as exc:
        logger.debug("Could not write skill source state: %s", exc)
        return False


# ── Checking one source ─────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalized_body_digest(body: str) -> str:
    """Hash the readable text of a page: drop scripts/styles, ignore re-flowed markup.

    Rebuilt pages re-indent their markup and rotate inline script payloads (build ids,
    nonces) without changing the documentation, and a check that cries wolf on those is
    useless. Whitespace between tags is therefore dropped and runs of whitespace inside
    text collapse to one space before hashing. Nothing here rewrites content the reader
    would see.
    """
    without_scripts = _SCRIPT_STYLE_RE.sub(" ", body)
    reflowed = _INTER_TAG_RE.sub("><", without_scripts)
    return _digest(_WHITESPACE_RE.sub(" ", reflowed).strip().encode("utf-8"))


class SourceRedirectBlocked(Exception):
    """A redirect target the website policy refuses (raised from the response hook)."""


def _policy_redirect_guard(response: Any) -> None:
    """Re-check the website policy on every redirect hop.

    The SSRF guard already validates the address each hop connects to, but not the
    *policy*: a public URL 302ing to a blocklisted domain would otherwise be followed and
    read. Same shape as ``tools/vision_tools._ssrf_redirect_guard``, for a sync client.
    """
    from tools.url_safety import redirect_target_from_response
    from tools.website_policy import check_website_access

    target = redirect_target_from_response(response)
    if not target:
        return
    blocked = check_website_access(target)
    if blocked:
        raise SourceRedirectBlocked(
            blocked.get("message", f"redirect to {target} blocked by website policy"))


def _check_url(url: str, baseline: dict[str, Any], retrieved: Optional[str]) -> tuple[str, str, Optional[dict[str, Any]]]:
    from tools.url_safety import create_ssrf_safe_client, is_safe_url
    from tools.website_policy import check_website_access

    blocked = check_website_access(url)
    if blocked:
        return "skipped", blocked.get("message", "blocked by website policy"), None
    if not is_safe_url(url):
        return "skipped", "url is not permitted by the URL safety policy", None

    known = isinstance(baseline.get("hash"), str) and bool(baseline["hash"])
    refreshed = known and baseline.get("retrieved") != retrieved
    # A conditional request only means "unchanged" while the recorded baseline is still the
    # one it was taken against. After a refresh (or on a first check) the body is needed.
    conditional = known and not refreshed
    headers = {"User-Agent": f"hermes-skills-check/{STATE_VERSION}"}
    if conditional:
        if isinstance(baseline.get("etag"), str) and baseline["etag"]:
            headers["If-None-Match"] = baseline["etag"]
        if isinstance(baseline.get("last_modified"), str) and baseline["last_modified"]:
            headers["If-Modified-Since"] = baseline["last_modified"]

    try:
        with create_ssrf_safe_client(timeout=HTTP_TIMEOUT_SECONDS, follow_redirects=True,
                                     event_hooks={"response": [_policy_redirect_guard]}) as client:
            with client.stream("GET", url, headers=headers) as response:
                if response.status_code == 304:
                    return "unchanged", "not modified since the last check", {
                        **baseline, "checked_at": _now_iso()}
                body = _read_capped(response)
                etag = response.headers.get("etag", "")
                last_modified = response.headers.get("last-modified", "")
                if response.status_code >= 400:
                    return "unreachable", f"HTTP {response.status_code}", None
    except SourceRedirectBlocked as exc:
        return "skipped", str(exc)[:200], None
    except Exception as exc:  # network, TLS, timeout, policy at connect time
        logger.debug("Source check failed for %s: %s", url, exc, exc_info=True)
        return "unreachable", f"{type(exc).__name__}: {exc}"[:200], None

    digest = _normalized_body_digest(body)
    fresh = {"hash": digest, "etag": etag, "last_modified": last_modified,
             "retrieved": retrieved, "checked_at": _now_iso()}
    if not known:
        return "baseline_recorded", "first check — baseline recorded", fresh
    if refreshed:
        return "baseline_recorded", "re-baselined after the skill was refreshed (retrieved moved on)", fresh
    if baseline["hash"] == digest:
        return "unchanged", "", {**fresh, "hash": baseline["hash"]}
    # Keep the pre-change baseline: the alert has to survive until the skill is refreshed,
    # and a stored ETag for the new body would turn the very next check into a 304.
    return "changed", "page content changed since the last check", {**baseline, "checked_at": _now_iso()}


def _read_capped(response: Any) -> str:
    """Read a response body as text, stopping at ``MAX_BODY_BYTES``."""
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        chunks.append(chunk)
        total += len(chunk)
        if total >= MAX_BODY_BYTES:
            break
    return b"".join(chunks)[:MAX_BODY_BYTES].decode("utf-8", errors="replace")


def _repo_target(value: str) -> tuple[Optional[str], Optional[str]]:
    """``(git target, None)`` for a checkable repository, else ``(None, reason)``."""
    parsed = urlparse(value)
    scheme = parsed.scheme.lower()
    if scheme in _REMOTE_REPO_SCHEMES:
        return None, f"unsupported for an automatic check: {scheme}:// needs credentials"
    if scheme and scheme not in _CHECKABLE_REPO_SCHEMES:
        return None, f"unsupported repository url scheme: {scheme}"
    return value, None


def _ls_remote_head(target: str) -> Optional[str]:
    """The remote's ``HEAD`` commit, ``""`` when it has none, or None when unreachable.

    Transports are pinned: only http(s) and local ``file`` remotes may be spoken, so a
    recorded value cannot reach a transport we did not reason about (``ext::`` runs a
    command, and ``protocol.allow=never`` closes anything not named explicitly).
    """
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    protocol_args = [
        "-c", "protocol.allow=never",
        "-c", "protocol.https.allow=always",
        "-c", "protocol.http.allow=always",
        "-c", "protocol.file.allow=always",
    ]
    try:
        completed = subprocess.run(  # fixed argv, no shell
            ["git", *protocol_args, "ls-remote", "--", target, "HEAD"],
            capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS, env=env, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    for line in completed.stdout.splitlines():
        sha, _, ref = line.partition("\t")
        if ref.strip() == "HEAD" and sha.strip():
            return sha.strip()
    return ""


def _check_repo(value: str, revision: Optional[str]) -> tuple[str, str, Optional[dict[str, Any]]]:
    target, reason = _repo_target(value)
    if target is None:
        return "skipped", reason or "unsupported repository", None
    if not revision:
        return "skipped", "no revision recorded for this repository", None
    if urlparse(value).scheme.lower() in _HTTP_SCHEMES:
        # `git ls-remote` would otherwise be an egress path around the two policies every
        # other fetch in Hermes obeys: policy first (a blocklisted host), then safety (an
        # internal address such as a cloud metadata endpoint).
        from tools.url_safety import is_safe_url
        from tools.website_policy import check_website_access

        blocked = check_website_access(value)
        if blocked:
            return "skipped", blocked.get("message", "blocked by website policy"), None
        if not is_safe_url(value):
            return "skipped", "repository url is not permitted by the URL safety policy", None
    head = _ls_remote_head(target)
    if head is None:
        return "unreachable", "git ls-remote failed", None
    if not head:
        return "unreachable", "remote has no HEAD commit", None
    if head == revision:
        return "unchanged", "", None
    return "changed", f"new commits since the recorded revision (HEAD {head[:12]}, recorded {revision[:12]})", None


def _check_path(value: str, baseline: dict[str, Any], retrieved: Optional[str]) -> tuple[str, str, Optional[dict[str, Any]]]:
    path = Path(os.path.expanduser(value))
    if not path.is_absolute():
        return "skipped", "path is not absolute after expanding ~", None
    try:
        data = path.read_bytes() if path.is_file() else None
    except OSError as exc:
        return "unreachable", f"{type(exc).__name__}: {exc}"[:200], None
    if data is None:
        return "unreachable", "file does not exist", None
    digest = _digest(data)
    known = isinstance(baseline.get("hash"), str) and bool(baseline["hash"])
    fresh = {"hash": digest, "retrieved": retrieved, "checked_at": _now_iso()}
    if not known:
        return "baseline_recorded", "first check — baseline recorded", fresh
    if baseline.get("retrieved") != retrieved:
        return "baseline_recorded", "re-baselined after the skill was refreshed (retrieved moved on)", fresh
    if baseline["hash"] == digest:
        return "unchanged", "", fresh
    # Sticky, exactly as for a url: a `changed` file keeps reading as changed until the
    # skill itself is refreshed.
    return "changed", "file content changed since the last check", {**baseline, "checked_at": _now_iso()}


def _check_entry(entry: dict[str, Any], baseline: dict[str, Any]):
    kind = entry["kind"]
    if kind == "url":
        return _check_url(entry["value"], baseline, entry.get("retrieved"))
    if kind == "repo":
        return _check_repo(entry["value"], entry.get("revision"))
    return _check_path(entry["value"], baseline, entry.get("retrieved"))


# ── Public entry point ──────────────────────────────────────────────────────


def check_sources(name: Optional[str] = None, *, state: Optional[dict[str, Any]] = None,
                  persist: bool = True) -> list[dict[str, Any]]:
    """One row per recorded source across in-scope skills, recording new baselines.

    Each row is ``{"skill", "kind", "source", "status", "detail"}``; ``source`` is the
    recorded value as the user wrote it, so the table points back at their frontmatter.
    ``state``/``persist`` exist for tests to run a check without touching the profile.
    """
    loaded = load_state() if state is None else state
    working = {"version": STATE_VERSION, "sources": dict(loaded.get("sources") or {})}
    dirty = False
    rows: list[dict[str, Any]] = []
    for skill_name, _skill_dir, entries in iter_skill_sources(name=name):
        cache = working["sources"].setdefault(skill_name, {})
        for entry in entries:
            key = f"{entry['kind']}:{entry['value']}"
            baseline = cache.get(key) if isinstance(cache.get(key), dict) else {}
            status, detail, fresh = _check_entry(entry, baseline)
            if fresh is not None and fresh != baseline:
                cache[key] = fresh
                dirty = True
            rows.append({"skill": skill_name, "kind": entry["kind"],
                         "source": entry["value"], "status": status, "detail": detail})
    if persist and state is None and dirty:
        save_state(working)
    return rows

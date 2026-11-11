"""Serving path for remote (SEP-2640) skills inside ``skill_view`` / ``skills_list``.

Kept out of ``tools/skills_tool.py`` so the local/plugin flow keeps its shape: this
module owns everything that only exists because a skill can live on another machine
— locator resolution across origins, the load approval, content verification, and
the activation binding that later execution checks against.

Two invariants drive every choice here. A remote skill is never allowed to borrow a
local capability: no preprocessing, no dependency install, no environment capture
from frontmatter the server wrote. And nothing is injected before it verified —
the bytes are checked against the manifest the host already holds, and a mismatch
retires the catalog rather than being shown with a warning.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

from typing import TYPE_CHECKING

from agent.skills_contracts import SkillEntry, SkillError, SkillErrorCode, VerifiedRead
from agent.skills_prompt_mcp import (
    locator_of, qualified_name, remote_entry_disabled, split_qualified)

if TYPE_CHECKING:  # annotations only — the runtime import below is deferred on purpose
    from tools.mcp_skills_manager import SkillsOverMcpManager

logger = logging.getLogger("tools.skills_tool")


def _proto():
    """The wire models, imported on first use.

    Not module-level: ``mcp`` is an optional extra, and this module is imported by
    the skills toolset, which has to keep serving local skills on an install that
    never pulled the SDK in. Only a call that really concerns a remote skill
    reaches these imports.
    """
    from tools import mcp_skills_protocol as proto
    return proto


def _manager():
    from tools.mcp_skills_manager import get_manager
    return get_manager()


def _scope_id() -> str:
    from tools.mcp_skills_manager import current_scope_id
    return current_scope_id()


def _skills_enabled(config: Any) -> bool:
    from tools.mcp_skills_manager import mcp_server_skills_enabled
    return mcp_server_skills_enabled(config)

REMOTE_USAGE_HINT = ("To view linked files, call skill_view(server=..., uri=..., file_path=...) where "
                     "file_path is one of the relative paths in linked_files")


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _fail(error: str, **extra) -> str:
    return _json({"success": False, "error": error, **extra})


def failure_payload(exc: SkillError) -> str:
    """JSON body for a typed remote failure, keeping main's ``success/error`` contract."""
    return _json(exc.as_payload())


def known_server_labels() -> set[str]:
    """Configured MCP server names that opted in to the skills channel.

    The ``mcp_servers.<name>.skills.enabled`` flag is the whole opt-in: it says this
    instance may publish skills, and the skills enable/disable list decides which of
    them this profile actually uses. It grants no tool execution permission.
    """
    try:
        from hermes_cli.config import load_config
        servers = load_config().get("mcp_servers") or {}
    except Exception:
        logger.debug("MCP server labels unavailable", exc_info=True)
        return set()
    return {str(name) for name, cfg in servers.items() if _skills_enabled(cfg)}


def resolve_entry(name: Optional[str], server: Optional[str], uri: Optional[str], *,
                  manager: Optional[SkillsOverMcpManager] = None) -> SkillEntry:
    """Locate one remote entry from a structured or qualified locator.

    ``server`` and ``uri`` locate together and are never accepted separately: with
    only a URI the host would have to guess which of two servers that both publish
    ``skill://team/deploy/SKILL.md`` the caller meant. A bare name resolves only
    when exactly one origin offers it, and reports the candidates otherwise.
    """
    manager = manager or _manager()
    scope_id = _scope_id()
    if uri is not None:
        if not server:
            raise SkillError(SkillErrorCode.AMBIGUOUS_IDENTITY,
                             "Pass server= together with uri= so the origin is unambiguous.",
                             hint="For example: skill_view(server='skills', uri='skill://group/name/SKILL.md')")
        return manager.resolve_entry(server, scope_id, uri, refresh=True)
    if server is not None:
        raise SkillError(SkillErrorCode.NOT_PROVIDED,
                         f"Pass uri= (the SKILL.md resource URI) together with server='{server}'.",
                         hint="Use skills_list to see each remote skill's identity.")
    if not name:
        raise SkillError(SkillErrorCode.NOT_PROVIDED, "A skill name, or server= with uri=, is required.")
    split = _split_qualified(name)
    if split is not None:
        label, locator = split
        matches = _find_by_locator(manager, scope_id, label, locator)
        if not matches:
            raise SkillError(SkillErrorCode.NOT_PROVIDED,
                             f"'{locator}' is not provided by MCP server '{label}'.",
                             available_skills=sorted(
                                 locator_of(e.identity.locator)
                                 for e in manager.find(scope_id, server=label))[:20])
        return _sole(matches, name)
    matches = manager.find(scope_id, name=name)
    if not matches:
        raise SkillError(SkillErrorCode.NOT_PROVIDED, f"Skill '{name}' was not found in any connected MCP server.")
    return _sole(matches, name)


def _find_by_locator(manager: SkillsOverMcpManager, scope_id: str, server: str,
                     locator: str) -> list[SkillEntry]:
    """Entries under *server* whose URI locator is *locator*.

    Matched on the URI rather than the frontmatter name: the name is a label the
    server may reuse, and the qualified key has to name exactly one skill.
    """
    return [entry for entry in manager.find(scope_id, server=server)
            if locator_of(entry.identity.locator) == locator]


def _split_qualified(name: str) -> Optional[tuple[str, str]]:
    from agent.skills_prompt_mcp import split_qualified
    return split_qualified(name)


def _sole(matches: list[SkillEntry], name: str) -> SkillEntry:
    if len(matches) > 1:
        raise SkillError(
            SkillErrorCode.AMBIGUOUS_IDENTITY,
            f"'{name}' is offered by {len(matches)} remote skills; refusing to guess which one you meant.",
            matches=[{"server": e.identity.origin_id, "uri": e.identity.locator} for e in matches],
            hint="Locate it explicitly with skill_view(server=..., uri=...).")
    return matches[0]


def serve(entry: SkillEntry, file_path: Optional[str], task_id: Optional[str], *,
          manager: Optional[SkillsOverMcpManager] = None) -> str:
    """Read, verify and return one remote skill (or one of its files) as JSON.

    Availability is settled before anything is fetched, and the bytes are verified
    against the manifest the catalog published before any of them can reach the
    model: what arrives is what the server declared, or it is refused.
    """
    manager = manager or _manager()
    scope_id = _scope_id()
    if refusal := _availability_refusal(entry):
        return refusal
    try:
        if file_path:
            return _serve_attachment(manager, entry, file_path, task_id, scope_id)
        return _serve_main(manager, entry, task_id, scope_id)
    except SkillError as exc:
        return failure_payload(exc)


def _availability_refusal(entry: SkillEntry) -> Optional[str]:
    """The JSON refusal when this skill may not be read right now, or None.

    Same order and wording as the on-disk path (platform, then the enable/disable
    list): a remote skill must not answer a question differently from a local one,
    or the model learns to probe for whichever surface lets it through. The disable
    check runs before any RPC, so a switched-off skill costs no network at all.
    """
    if unsupported := _platform_refusal(entry):
        return _fail(unsupported, readiness_status="unsupported")
    if remote_entry_disabled(entry.identity.origin_id, entry.identity.locator, _disabled_names()):
        return _fail(f"Skill '{_load_key(entry)}' is disabled. Enable it with `hermes skills`.")
    return None


def _platform_refusal(entry: SkillEntry) -> Optional[str]:
    """The local platform-mismatch message, when a remote skill declares other platforms.

    The index already withholds these rows; refusing the explicit load too keeps the
    two surfaces from disagreeing. A skill that says ``platforms: [linux]`` describes
    commands for a host this is not, and loading it would hand the model shell lines
    that cannot run here — the same reason ``skill_view`` refuses a local one.
    """
    try:
        from agent.skill_utils import skill_matches_platform
    except ImportError:
        return None
    try:
        if skill_matches_platform(dict(entry.frontmatter)):
            return None
    except (TypeError, ValueError):
        # A malformed `platforms:` value is the server's problem, not a reason to refuse a
        # load it never asked us to refuse; the index gate already applied the same leniency.
        logger.debug("Unusable platforms declaration on %s", entry.identity.locator, exc_info=True)
        return None
    return f"Skill '{entry.name}' is not supported on this platform."


def _serve_attachment(manager: SkillsOverMcpManager, entry: SkillEntry, file_path: str,
                      task_id: Optional[str], scope_id: str) -> str:
    """Serve one support file, resolved strictly inside the held manifest.

    A path the manifest does not list is refused rather than fetched: the manifest
    is what the user approved, and a server adding files after the fact must not be
    able to widen the approval by simply answering a request for them.
    """
    normalized = _normalize_relative(file_path)
    if normalized == _proto().MAIN_FILE_NAME or normalized.endswith("/" + _proto().MAIN_FILE_NAME):
        # A nested SKILL.md is a different skill, not a file of this one: serving it here
        # would present another skill's instructions under this skill's identity.
        nested = normalized[: -len("/" + _proto().MAIN_FILE_NAME)] if normalized != _proto().MAIN_FILE_NAME else ""
        return _fail(
            f"'{normalized}' is a nested skill, not a file of '{entry.name}'.",
            hint=(f"Load it as its own skill: skill_view(server='{entry.identity.origin_id}', "
                  f"uri='{_nested_uri(entry, nested)}')"))
    target = next((f for f in (entry.manifest.files if entry.manifest else ()) if f.relative_path == normalized), None)
    if target is None:
        return _fail(f"File '{file_path}' is not part of remote skill '{entry.name}'.",
                     available_files=sorted(f.relative_path for f in (entry.manifest.files if entry.manifest else ())),
                     hint="Use one of the available file paths listed above")
    from tools.skills_tool_dedup import check_remote_view, record_remote_view
    server = entry.identity.origin_id
    revision = entry.content_revision or ""
    if (stub := check_remote_view(task_id, server, entry.identity.locator, revision, normalized)) is not None:
        return stub
    read = manager.read_resource(server, scope_id, entry, target.uri)
    if read.text is None:
        return _json({"success": True, "name": _load_key(entry), "file": normalized, "is_binary": True,
                      "content": f"[Binary file: {normalized.split('/')[-1]}, size: {len(read.raw_bytes)} bytes]",
                      "source": _source(entry)})
    record_remote_view(task_id, server, entry.identity.locator, revision, normalized)
    return _json({"success": True, "name": _load_key(entry), "file": normalized, "content": read.text,
                  "file_type": ("." + normalized.rsplit(".", 1)[1]) if "." in normalized else "",
                  "source": _source(entry), "verified": True})


def _normalize_relative(file_path: str) -> str:
    return str(file_path).strip().replace("\\", "/").lstrip("./")


def _nested_uri(entry: SkillEntry, nested_dir: str) -> str:
    """Best-effort URI of a nested skill's own SKILL.md (the server may not publish one)."""
    root = _proto().skill_root_uri(entry.identity.locator)
    return f"{root}/{nested_dir}/{_proto().MAIN_FILE_NAME}" if nested_dir else entry.identity.locator


def _serve_main(manager: SkillsOverMcpManager, entry: SkillEntry, task_id: Optional[str],
                scope_id: str) -> str:
    from tools.skills_tool_dedup import check_remote_view, record_remote_view

    server = entry.identity.origin_id
    revision = entry.content_revision or ""
    if (stub := check_remote_view(task_id, server, entry.identity.locator, revision, None)) is not None:
        return stub
    read = manager.read_resource(server, scope_id, entry, entry.identity.locator)
    if read.text is None:
        return _fail(f"Remote skill '{entry.name}' did not return readable text for its SKILL.md.")
    _verify_frontmatter(read, entry)
    record_remote_view(task_id, server, entry.identity.locator, revision, None)
    linked = _proto().linked_files_of(entry)
    return _json({
        "success": True,
        "name": _load_key(entry),
        "description": entry.description,
        "content": read.text,
        "tags": [], "related_skills": [],
        "path": entry.identity.locator,
        "skill_dir": None,
        "skill_root_uri": read.skill_root_uri,
        "linked_files": linked or None,
        "usage_hint": REMOTE_USAGE_HINT if linked else None,
        "readiness_status": "available",
        "identity": {"backend": entry.identity.backend_id, "server": entry.identity.origin_id,
                     "uri": entry.identity.locator},
        "source": _source(entry),
        "provenance": _provenance(entry),
        "revision": entry.content_revision,
        "verified": True,
        **_declared_requirements(entry),
    })


def _verify_frontmatter(read: VerifiedRead, entry: SkillEntry) -> None:
    """Confirm the body's own frontmatter still names this skill.

    The digest already proves the bytes match the manifest; this catches the one
    case the digest cannot — a server that legitimately republished the file under
    a different skill and now answers an old URI with new content that happens to
    hash to the held digest is impossible, so this is really a cheap consistency
    check between the two views the host holds.
    """
    from tools.skills_tool_plugin import _safe_frontmatter
    declared = _safe_frontmatter(content=read.text or "").get("name")
    if isinstance(declared, str) and declared.strip() and declared.strip() != entry.name:
        raise SkillError(SkillErrorCode.VERIFICATION_FAILED,
                         f"Remote skill {entry.identity.locator!r} declares name '{declared.strip()}' in its "
                         f"frontmatter but the catalog published it as '{entry.name}'.")


def _load_key(entry: SkillEntry) -> str:
    """The qualified key this skill is listed and switched on/off by.

    Derived from the URI, so it names one skill even when a server publishes two
    under the same frontmatter name.
    """
    return qualified_name(entry.identity.origin_id, entry.identity.locator)


def _source(entry: SkillEntry) -> dict[str, Any]:
    return {"backend": "mcp", "server": entry.identity.origin_id, "uri": entry.identity.locator,
            "remote": True}


def _provenance(entry: SkillEntry) -> dict[str, Any]:
    origin = entry.provenance
    return {"label": origin.label if origin else entry.identity.origin_id,
            "detail": origin.detail if origin else "",
            "revision": entry.content_revision}


def _declared_requirements(entry: SkillEntry) -> dict[str, Any]:
    """Surface a remote skill's declared env needs WITHOUT acting on them.

    Locally, ``required_environment_variables`` triggers a secret prompt and .env
    persistence. For a remote skill that would let third-party frontmatter ask the
    user for credentials, so the list is reported for the model to relay and
    nothing is captured.
    """
    declared = entry.frontmatter.get("required_environment_variables")
    if not isinstance(declared, list) or not declared:
        return {}
    names = [str(e.get("name")) for e in declared if isinstance(e, dict) and e.get("name")]
    payload: dict[str, Any] = {"required_environment_variables": declared,
                               "setup_needed": False, "setup_skipped": True}
    if names:
        payload["remote_requirements_note"] = (
            "These requirements are declared by the remote skill and were not resolved locally. "
            "Set them yourself before running its commands: " + ", ".join(names))
    return payload


def list_rows(scope_id: Optional[str] = None, *, skip_disabled: bool = False) -> list[dict[str, Any]]:
    """Rows for every published remote skill: identity, description, and where it lives.

    ``name`` is the qualified load key — the same string ``skill_view`` resolves and
    the same one the enable/disable list names this skill by, so a caller can feed it
    straight back into either. ``server``/``uri`` are carried separately for callers
    that would rather locate explicitly; they locate, they do not authorize.

    ``skip_disabled`` mirrors the on-disk catalog's flag: the enable/disable UI lists
    everything, including what is switched off, while the model-facing listing shows
    only what this profile may actually load right now.
    """
    manager = _manager()
    scope = scope_id if scope_id is not None else _scope_id()
    disabled = set() if skip_disabled else _disabled_names()
    rows: list[dict[str, Any]] = []
    for entry in manager.entries(scope):
        if remote_entry_disabled(entry.identity.origin_id, entry.identity.locator, disabled):
            continue
        if _platform_refusal(entry):  # the offer-time gate the index and skill_view share
            continue
        rows.append({
            "name": _load_key(entry),
            "description": entry.description,
            "category": f"mcp/{entry.identity.origin_id}",
            "server": entry.identity.origin_id,
            "uri": entry.identity.locator,
            "source": _source(entry),
        })
    return rows


def _connect_servers(servers: list[str]) -> None:
    """Bring just *servers* up through the normal connect path, without this call reading stdin.

    The same route every other connect takes: cooldown, OAuth suppressions and the parked
    self-probe all stay in the discovery layer instead of being re-implemented here. For a
    server that is merely torn down it only signals the rebuild and returns.
    """
    try:
        from tools.mcp_oauth import suppress_interactive_oauth
    except ImportError:
        from contextlib import nullcontext as suppress_interactive_oauth
    from tools.mcp_tool_discovery import discover_mcp_tools

    with suppress_interactive_oauth():
        discover_mcp_tools(allowed_mcp_names=servers)


EXPLICIT_REFRESH_TIMEOUT_SECONDS = 15.0
# A nudge (``_connect_servers``) only *signals* a torn-down transport to rebuild; the session
# appears a moment later, so a listing waits this long for it before calling the server gone.
_SESSION_WAIT_SECONDS = 5.0
# The loop bridge polls in 100ms slices on its own deadline; a hair of headroom lets the
# pass finish and report per-server rather than being cut off mid-classification.
_BRIDGE_MARGIN_SECONDS = 2.0


def refresh_catalogs(*, timeout: float = EXPLICIT_REFRESH_TIMEOUT_SECONDS,
                     manager: Optional[SkillsOverMcpManager] = None,
                     scope_id: Optional[str] = None) -> tuple[list[str], list[dict[str, str]]]:
    """Re-read every opted-in server's catalog now, for a listing the user asked for.

    Returns ``(verified server labels, problems)``. A caller shows a server's rows only when
    it is listed as verified: this runs where someone asked for the *current* list, and the
    snapshot already held routes and loads against a connection that may be gone. A server
    that answers with an empty catalog is verified — "nothing published" is an answer.

    Bounded end to end: the reconnect nudge, the wait for it to land and the re-list share
    one deadline, so an unreachable server costs the caller time, never an unbounded wait.
    A problem here means "not readable now", never proof the transport is gone — nothing is
    marked disconnected and the next call tries again.
    """
    servers = sorted(known_server_labels())
    if not servers:
        return [], []
    manager = manager or _manager()
    scope = _scope_id() if scope_id is None else scope_id
    missing = [name for name in servers if not manager.has_session(name, scope)]
    if missing:
        _connect_servers(missing)
    try:
        from tools.mcp_tool_loop import _run_on_mcp_loop
        return _run_on_mcp_loop(lambda: _refresh_on_loop(manager, scope, servers, timeout),
                                timeout=timeout + _BRIDGE_MARGIN_SECONDS)
    except TimeoutError:
        return [], [_unreadable(name, "the servers did not answer in time") for name in servers]
    except InterruptedError:
        return [], [_unreadable(name, "the listing was interrupted") for name in servers]
    except RuntimeError as exc:  # no MCP loop: nothing was ever connected in this process
        logger.debug("Skills catalog refresh unavailable: %s", exc)
        return [], [_unreadable(name, "the MCP client is not available") for name in servers]


async def _refresh_on_loop(manager: SkillsOverMcpManager, scope: str, servers: list[str],
                           timeout: float) -> tuple[list[str], list[dict[str, str]]]:
    """Re-list inside the MCP loop: wait briefly for a signaled reconnect, then read in parallel."""
    deadline = time.monotonic() + timeout
    wait_until = min(time.monotonic() + _SESSION_WAIT_SECONDS, deadline)
    while time.monotonic() < wait_until and not all(manager.has_session(n, scope) for n in servers):
        await asyncio.sleep(0.05)
    ready = [name for name in servers if manager.has_session(name, scope)]
    problems = [_unreadable(name, "no live connection") for name in servers if name not in ready]
    tasks = {name: asyncio.ensure_future(manager.discover(name, scope, reason="explicit-list"))
             for name in ready}
    if tasks:
        await asyncio.wait(set(tasks.values()), timeout=max(0.0, deadline - time.monotonic()))
    verified: list[str] = []
    for name, task in tasks.items():
        if not task.done():
            # Past the deadline the read keeps running — the catalog it lands is still the
            # right one — but this listing may not claim it answered.
            task.add_done_callback(_log_late_refresh)
            problems.append(_unreadable(name, "the catalog did not answer in time"))
            continue
        failure = task.exception()
        if failure is None or _is_no_catalog_here(failure):
            # A server that answers "I serve no skills extension" has told us its current
            # list: nothing. That is a verified answer, not an unreadable one.
            verified.append(name)
        else:
            problems.append(_unreadable(name, _failure_text(failure)))
    return verified, problems


def _is_no_catalog_here(exc: BaseException) -> bool:
    return isinstance(exc, SkillError) and exc.code is SkillErrorCode.UNSUPPORTED_EXTENSION


def _failure_text(exc: BaseException) -> str:
    return getattr(exc, "message", None) or f"{type(exc).__name__}: {exc}"


def _unreadable(server: str, reason: str) -> dict[str, str]:
    return {"server": server, "error": reason}


def _log_late_refresh(task: asyncio.Task[Any]) -> None:
    """Consume a read that outlived its listing so its failure is never 'never retrieved'."""
    if task.cancelled():
        return
    failure = task.exception()
    if failure is not None:
        logger.debug("Skills catalog refresh finished after its listing: %s", failure)


def _disabled_names() -> set[str]:
    """The profile's skill enable/disable list, through the config-normalizing reader.

    Resolved via the ``tools.skills_tool`` delegate — the same seam the catalog scan
    and every other disabled-name reader go through — so the listing, the index and a
    direct ``skill_view`` can never disagree about whether a skill is switched on.
    """
    try:
        from tools.skills_tool import _get_disabled_skill_names
        return _get_disabled_skill_names()
    except Exception:
        logger.debug("Skill enable/disable list unavailable", exc_info=True)
        return set()


def mutation_unsupported(name: str) -> str:
    """``skill_manage`` answer for a remote skill: read-only, and says why."""
    return _fail(
        f"'{name}' is a remote (MCP) skill and is read-only here. Edit it on the server that publishes it.",
        error_code=SkillErrorCode.MUTATION_UNSUPPORTED.value,
        hint="Use skills_mcp_push / the server's own tooling to change it.")


def serve_locator(*, name: str = "", server: Optional[str] = None, uri: Optional[str] = None,
                  file_path: Optional[str] = None, task_id: Optional[str] = None) -> str:
    """``skill_view(server=…, uri=…)`` entry point: always answers, never falls through."""
    try:
        entry = resolve_entry(name or None, server, uri)
    except SkillError as exc:
        return failure_payload(exc)
    return serve(entry, file_path, task_id)


def serve_name_fallback(*, name: str, file_path: Optional[str] = None,
                        task_id: Optional[str] = None) -> Optional[str]:
    """Remote answer for a name nothing on disk matched; ``None`` keeps the local not-found error.

    ``None`` is deliberate for the two cases that are not "we can help": no server offers the
    name (the local error already lists what exists), and an ambiguous remote match whose error
    is *more* useful than the local one — that one is returned, not swallowed, because the
    caller asked for a skill that does exist somewhere.
    """
    try:
        entry = resolve_entry(name, None, None)
    except SkillError as exc:
        if exc.code is SkillErrorCode.NOT_PROVIDED:
            return None
        return failure_payload(exc)
    return serve(entry, file_path, task_id)


def mutation_target_is_remote(name: str) -> bool:
    """Whether *name* addresses a remote skill rather than a local/plugin one.

    Requires the ``server::locator`` spelling, the server to be configured for skills,
    AND the server to actually publish that skill — a local skill called ``skills/foo``
    (or a plugin named ``skills``) must keep resolving to itself.
    """
    split = split_qualified(name or "")
    if split is None:
        return False
    server, locator = split
    if server not in known_server_labels():
        return False
    try:
        return bool(_find_by_locator(_manager(), _scope_id(), server, locator))
    except Exception:
        logger.debug("Could not check remote ownership of %r", name, exc_info=True)
        return False


def dispatch_before_local_lookup(*, name: str = "", server: Optional[str] = None,
                                 uri: Optional[str] = None, file_path: Optional[str] = None,
                                 task_id: Optional[str] = None, lookup_hint: str = "") -> Optional[str]:
    """Answer a ``skill_view`` that cannot be a local-name lookup; ``None`` to continue locally.

    Three shapes qualify: a structured remote locator, a ``server::skill`` key whose server
    is configured, and a call with no locator at all (the model has to be told how to locate
    something). Everything else is a local-name lookup and pays one function call for this
    check, nothing more.

    A configured ``server::skill`` is resolved here rather than as a fallback, because the
    spelling is unambiguous — plugin's ``plugin:skill`` uses one colon, so no name that
    resolved before can be intercepted by this branch. A ``::`` name whose prefix is not a
    configured server returns None and keeps flowing through the local path.
    """
    if server is None and uri is None:
        if not name:
            return _fail("Pass a skill name, or server= with uri= to load a remote skill.", hint=lookup_hint)
        split = split_qualified(name)
        if split is not None and split[0] in known_server_labels():
            return serve_locator(name=name, file_path=file_path, task_id=task_id)
        return None
    return serve_locator(name=name, server=server, uri=uri, file_path=file_path, task_id=task_id)


def resolve_local_lookup_failure(*, name: str, kind: str, error: str,
                                 file_path: Optional[str] = None,
                                 task_id: Optional[str] = None) -> str:
    """The answer for a failed on-disk lookup: a remote match if one exists, else *error*.

    Remote skills stand behind the local tree, never in front of it — only a genuine
    miss reaches the network, so a name that already resolved locally, collided
    locally, or was quarantined locally keeps meaning exactly what it meant before a
    server was configured.
    """
    if kind == "not_found":
        remote = serve_name_fallback(name=name, file_path=file_path, task_id=task_id)
        if remote is not None:
            return remote
    return error

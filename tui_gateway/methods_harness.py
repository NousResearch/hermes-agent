"""Harness-specific JSON-RPC handlers preserved across upstream rebases."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

from .method_ctx import HandlerRegistry

# The wiki handlers below were carried into this module during the upstream
# rebase's handler split WITHOUT their imports — every wiki RPC (wiki.scan,
# wiki.page, wiki.list, wiki.taxonomy, wiki.expand_links, wiki.changesets)
# died with `NameError: name 'resolve_wiki' is not defined`, which the
# desktop rendered as "Failed to load page" on every wiki page. NameError,
# not ImportError, because Python resolves function-body names at CALL time —
# so the module imported cleanly and the break was invisible until the first
# wiki call. tests/gateway/test_methods_harness_imports.py now pins every
# name these handlers reference to an actual import.
from tui_gateway.wiki_api import (
    resolve_wiki,
    wiki_changesets,
    wiki_expand_links,
    wiki_flatten_taxonomy,
    wiki_list,
    wiki_page,
    wiki_scan,
    wiki_taxonomy,
)

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


# Skills helpers recovered from the pre-rebase server.py (60b71f5a6) — the
# handler split carried their callers here but dropped the definitions, so
# the skills.* handlers NameError'd on first call, same class as the wiki
# import loss above.
def _find_local_skill_md(skill_name: str) -> Optional[Path]:
    """Find a locally installed skill's SKILL.md file by name.

    Searches ~/.hermes/skills/ recursively for a SKILL.md whose YAML
    frontmatter ``name`` field matches *skill_name*.  Returns the
    absolute Path or None.
    """
    import yaml

    skills_dir = Path(get_hermes_home()) / "skills"
    if not skills_dir.is_dir():
        return None
    for candidate in sorted(skills_dir.rglob("SKILL.md")):
        try:
            raw = candidate.read_text(encoding="utf-8")[:4096]
        except (OSError, UnicodeDecodeError):
            continue
        # Extract YAML frontmatter between --- markers
        if not raw.startswith("---"):
            continue
        end = raw.find("---", 3)
        if end == -1:
            continue
        try:
            fm = yaml.safe_load(raw[3:end])
        except Exception:
            continue
        if isinstance(fm, dict) and fm.get("name") == skill_name:
            return candidate
    return None


def _parse_skill_frontmatter(content: str) -> dict[str, Any]:
    """Parse YAML frontmatter from a SKILL.md string."""
    import yaml

    if not content.startswith("---"):
        return {}
    end = content.find("---", 3)
    if end == -1:
        return {}
    try:
        return yaml.safe_load(content[3:end]) or {}
    except Exception:
        return {}


def _skill_info_from_path(skill_md: Path, fm: dict[str, Any]) -> dict[str, Any]:
    """Build a skill info dict consumable by HermesNative's SkillInfo."""
    category = skill_md.parent.name if skill_md.parent.parent != skill_md.parent.parent.parent else "general"
    # Use the relative path from skills/ dir as the category path
    try:
        skills_dir = Path(get_hermes_home()) / "skills"
        rel = skill_md.parent.relative_to(skills_dir)
        # category is the top-level directory name
        parts = rel.parts
        if parts:
            category = parts[0] if len(parts) >= 1 else "general"
    except ValueError:
        category = "general"

    info: dict[str, Any] = {
        "name": fm.get("name", skill_md.parent.name),
        "description": fm.get("description", ""),
        "category": category,
        "source": "local",
        "id": fm.get("name", skill_md.parent.name),
        "identifier": fm.get("name", skill_md.parent.name),
        "tags": fm.get("metadata", {}).get("hermes", {}).get("tags", []) if isinstance(fm.get("metadata"), dict) else [],
        "path": str(skill_md),
        "skill_md_preview": "",  # populated by caller after reading content
    }
    return info

@method("session.prompt_breakdown")
def _(rid, params: dict) -> dict:
    """Return the session's system prompt decomposed into sections with token counts."""
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    agent = session.get("agent")
    home = Path(get_hermes_home())

    try:
        import tiktoken
        _enc = tiktoken.get_encoding("cl100k_base")
        _count = lambda t: len(_enc.encode(t)) if t else 0
    except Exception:
        _count = lambda t: max(1, int(len(t) / 3.5)) if t else 0

    def _section(name, source, content, color):
        text = (content or "").strip()
        return dict(name=name, source=source,
                    content_preview=text[:200], full_content=text[:10240],
                    token_count=_count(text), char_count=len(text), color=color)

    persona = ""
    for p in [home / "SOUL.md", home / "persona.md"]:
        if p.exists():
            persona = p.read_text().strip()
            break
    memory = (home / "memories" / "MEMORY.md").read_text().strip() if (home / "memories" / "MEMORY.md").exists() else ""
    user = (home / "memories" / "USER.md").read_text().strip() if (home / "memories" / "USER.md").exists() else ""
    ephemeral = str(getattr(agent, "ephemeral_system_prompt", "") or "").strip()

    skills = ""
    if agent is not None and hasattr(agent, "tools"):
        from run_agent import get_toolset_for_tool
        toolsets = {ts for t in getattr(agent, "valid_tool_names", set()) if (ts := get_toolset_for_tool(t))}
        if toolsets:
            from agent.prompt_builder import build_skills_system_prompt
            skills = build_skills_system_prompt(getattr(agent, "valid_tool_names", set()), toolsets)

    tools_json = json.dumps(getattr(agent, "tools", []) or [], ensure_ascii=False)
    history = list(session.get("history", []))
    hist_json = json.dumps(history, ensure_ascii=False)

    sections = [
        _section("Persona", str(home / "SOUL.md"), persona, "#7c7cff"),
        _section("Memory", str(home / "memories" / "MEMORY.md"), memory, "#ff7c7c"),
        _section("User Profile", str(home / "memories" / "USER.md"), user, "#ffb87c"),
        _section("Ephemeral Prompt", "(session personality/prompt)", ephemeral, "#7cff7c"),
        _section("Active Skills", "~/.hermes/skills/", skills, "#ffd700"),
    ]

    return _ok(rid, dict(
        session_id=params.get("session_id", ""),
        model=getattr(agent, "model", "") if agent else "",
        context_limit=131072,
        total_system_tokens=sum(s["token_count"] for s in sections),
        sections=sections,
        tool_definitions=dict(token_count=_count(tools_json), count=len(getattr(agent, "tools", []) or [])),
        conversation_history=dict(token_count=_count(hist_json), message_count=len(history)),
    ))

@method("credits.view")
def _(rid, params: dict) -> dict:
    """Structured Nous credit view for the TUI /credits command.

    Account-independent (a portal fetch gated on "a Nous account is logged in"),
    so it works with no live agent / on a resumed session — same as the /usage
    credits block. Returns the surface-agnostic CreditsView fields so the TUI can
    render a clickable top-up <Link>. Fail-open: a portal hiccup or logged-out
    account yields {logged_in: false}, never an error the user has to parse.
    """
    try:
        from agent.account_usage import build_credits_view

        view = build_credits_view()
        return _ok(
            rid,
            {
                "logged_in": bool(view.logged_in),
                "balance_lines": [
                    line for line in view.balance_lines if not line.lstrip().startswith("📈")
                ],
                "identity_line": view.identity_line,
                "topup_url": view.topup_url,
                "depleted": bool(view.depleted),
            },
        )
    except Exception:
        # Fail-open: TUI treats this as "not logged in" and shows the prompt.
        return _ok(rid, {"logged_in": False, "balance_lines": [], "identity_line": None, "topup_url": None, "depleted": False})

@method("session.timeline")
def _(rid, params: dict) -> dict:
    """Return temporally-structured session events for playback visualization."""
    session, _ = _sess_nowait(params, rid)
    session_key = (session or {}).get("session_key") or params.get("session_id", "")
    messages = list((session or {}).get("history", []))

    db = _get_db()
    if db and session_key:
        try:
            messages = db.get_messages_as_conversation(session_key, include_ancestors=True)
        except Exception:
            pass

    events = []
    tool_starts = {}
    tool_count = 0
    in_tokens = out_tokens = 0

    for m in messages:
        role = m.get("role")
        ts = m.get("timestamp") or time.time()
        tc = m.get("token_count")
        if isinstance(tc, (int, float)):
            if role == "user": in_tokens += int(tc)
            elif role == "assistant": out_tokens += int(tc)

        if role == "user":
            if events and events[-1].get("type") != "turn_boundary":
                events.append(dict(type="turn_boundary", timestamp=ts))
            events.append(dict(type="user_message", timestamp=ts,
                               content=_coerce_message_text(m.get("content")),
                               token_count=tc))
        elif role == "assistant":
            for tc_item in m.get("tool_calls") or []:
                fn = tc_item.get("function", {})
                tid = tc_item.get("id", "")
                name = fn.get("name", "")
                evt = dict(type="tool_start", timestamp=ts, tool_name=name,
                           tool_id=tid, summary=_tool_ctx(name, {}))
                events.append(evt)
                tool_starts[tid] = evt
                tool_count += 1
            content = _coerce_message_text(m.get("content"))
            if content.strip():
                events.append(dict(type="assistant_message", timestamp=ts,
                                   content=content, token_count=tc))
        elif role == "tool":
            tid = m.get("tool_call_id", "")
            start = tool_starts.get(tid, {})
            dur = round((ts - start.get("timestamp", ts)) * 1000) if start.get("timestamp") and ts else None
            events.append(dict(type="tool_end", timestamp=ts,
                               tool_name=m.get("tool_name", ""), tool_id=tid,
                               content=_coerce_message_text(m.get("content"))[:300],
                               duration_ms=dur))

    tss = [e["timestamp"] for e in events if isinstance(e.get("timestamp"), (int, float))]
    total_duration = round(tss[-1] - tss[0], 1) if len(tss) >= 2 else 0.0

    cost = None
    if db and session_key:
        try:
            s = db.get_session(session_key) or {}
            cost = s.get("actual_cost_usd") or s.get("estimated_cost_usd")
        except Exception:
            pass

    return _ok(rid, dict(
        session_id=session_key, events=events,
        total_duration_seconds=total_duration, tool_calls=tool_count,
        input_tokens=in_tokens, output_tokens=out_tokens, cost_usd=cost,
    ))

@method("skills.get")
def _(rid, params: dict) -> dict:
    """Read a locally installed skill's SKILL.md content."""
    skill_name = params.get("skill_id", "")
    file_path = params.get("file_path")  # optional: relative path within skill dir

    if not skill_name:
        return _err(rid, 4018, "missing skill_id")

    try:
        skill_md = _find_local_skill_md(skill_name)
        if skill_md is None:
            return _err(rid, 4019, f"skill '{skill_name}' not found locally")

        # If a file_path is given, read a specific file from the skill dir
        if file_path:
            target = (skill_md.parent / file_path).resolve()
            # Safety: only allow reading within the skill directory
            try:
                target.relative_to(skill_md.parent)
            except ValueError:
                return _err(rid, 4020, f"path '{file_path}' escapes skill directory")
            if not target.is_file():
                return _err(rid, 4021, f"file '{file_path}' not found in skill dir")
            content = target.read_text(encoding="utf-8")
            return _ok(rid, {
                "skill": _skill_info_from_path(skill_md, _parse_skill_frontmatter(content)),
                "file_path": file_path,
                "content": content,
                "read_only": not os.access(target, os.W_OK),
            })

        content = skill_md.read_text(encoding="utf-8")
        fm = _parse_skill_frontmatter(content)

        info = _skill_info_from_path(skill_md, fm)
        info["skill_md_preview"] = content[:2000]

        read_only = not os.access(skill_md, os.W_OK)

        return _ok(rid, {
            "skill": info,
            "file_path": "SKILL.md",
            "content": content,
            "read_only": read_only,
        })
    except Exception as e:
        return _err(rid, 5024, str(e))

@method("skills.update")
def _(rid, params: dict) -> dict:
    """Update a locally installed skill's SKILL.md content."""
    skill_name = params.get("skill_id", "")
    new_content = params.get("content", "")

    if not skill_name:
        return _err(rid, 4022, "missing skill_id")
    if not new_content:
        return _err(rid, 4023, "missing content")

    try:
        skill_md = _find_local_skill_md(skill_name)
        if skill_md is None:
            return _err(rid, 4019, f"skill '{skill_name}' not found locally")

        if not os.access(skill_md, os.W_OK):
            return _err(rid, 4024, f"skill '{skill_name}' is read-only")

        skill_md.write_text(new_content, encoding="utf-8")
        fm = _parse_skill_frontmatter(new_content)
        info = _skill_info_from_path(skill_md, fm)
        info["skill_md_preview"] = new_content[:2000]

        # Reload skills so the agent picks up changes
        try:
            from agent.skill_commands import reload_skills
            reload_skills()
        except Exception:
            pass  # best-effort; don't fail the write

        return _ok(rid, {"skill": info})
    except Exception as e:
        return _err(rid, 5024, str(e))

@method("files.list")
def _(rid, params: dict) -> dict:
    """List a browsable root's file tree (the repo checkout or ~/.hermes).

    With no ``root`` param, returns the set of available root names so the
    desktop can render the top level; with a ``root``, returns that root's
    bounded, containment-checked tree. All logic lives in the pure
    tui_gateway.files_browse module (locally imported so it resolves against
    the real import system regardless of the handler's rebound globals)."""
    try:
        from tui_gateway.files_browse import FileBrowseError, file_roots, list_tree

        root = params.get("root")
        if not root:
            return _ok(rid, {"roots": sorted(file_roots().keys())})
        try:
            return _ok(rid, list_tree(root, params.get("path", "") or ""))
        except FileBrowseError as fe:
            return _err(rid, fe.code, fe.message)
    except Exception as e:
        logger.exception("files.list failed")
        return _err(rid, 5060, str(e))


@method("files.read")
def _(rid, params: dict) -> dict:
    """Read one file (read-only, UTF-8 text) from a browsable root."""
    try:
        from tui_gateway.files_browse import FileBrowseError, read_file

        root = params.get("root")
        path = params.get("path")
        if not root:
            return _err(rid, 4001, "root is required")
        try:
            return _ok(rid, read_file(root, path))
        except FileBrowseError as fe:
            return _err(rid, fe.code, fe.message)
    except Exception as e:
        logger.exception("files.read failed")
        return _err(rid, 5061, str(e))


@method("session.set_prompt")
def _(rid, params: dict) -> dict:
    """Set an ephemeral system prompt append on the live agent.

    The prompt is appended to the agent's system prompt on every API call
    but is NOT persisted to trajectories or the session database.
    Setting an empty string clears the ephemeral prompt.
    """
    session, err = _sess(params, rid)
    if err:
        return err
    agent = session.get("agent")
    if not agent:
        return _err(rid, 4001, "session not ready")
    if session.get("running"):
        return _err(
            rid,
            4009,
            "session busy — /interrupt the current turn before setting prompt",
        )
    prompt = str(params.get("prompt", "") or "").strip()
    agent.ephemeral_system_prompt = prompt or None
    agent._cached_system_prompt = None
    return _ok(rid, {"prompt": prompt})

@method("wiki.scan")
def _(rid, params: dict) -> dict:
    try:
        from tui_gateway.wiki_api import resolve_wiki, wiki_scan
        wiki_name = params.get("wiki") or params.get("path")
        wiki_path = resolve_wiki(wiki_name)
        result = wiki_scan(wiki_path)
        return _ok(rid, result)
    except Exception as e:
        logger.exception("wiki.scan failed")
        return _err(rid, 5050, str(e))

@method("wiki.page")
def _(rid, params: dict) -> dict:
    try:
        from tui_gateway.wiki_api import resolve_wiki, wiki_page
        page_path = params.get("path")
        if not page_path:
            return _err(rid, 4001, "path is required")
        wiki_name = params.get("wiki")
        wiki_path = resolve_wiki(wiki_name)
        result = wiki_page(page_path, wiki_path)
        if result is None:
            return _err(rid, 4040, f"page not found: {page_path}")
        return _ok(rid, result)
    except Exception as e:
        logger.exception("wiki.page failed")
        return _err(rid, 5051, str(e))

@method("wiki.list")
def _(rid, params: dict) -> dict:
    try:
        result = wiki_list()
        return _ok(rid, result)
    except Exception as e:
        logger.exception("wiki.list failed")
        return _err(rid, 5052, str(e))


@method("wiki.glossary")
def _(rid, params: dict) -> dict:
    """Read the validated glossary for one configured wiki."""
    try:
        from tui_gateway.wiki_glossary import (
            GlossaryValidationError,
            load_glossary,
            resolve_configured_wiki,
        )

        try:
            unknown = set(params) - {"wiki"}
            if unknown:
                raise GlossaryValidationError(
                    f"unknown parameter(s): {', '.join(sorted(map(str, unknown)))}"
                )
            wiki_path = resolve_configured_wiki(params.get("wiki"))
            return _ok(rid, load_glossary(wiki_path))
        except GlossaryValidationError as exc:
            return _err(rid, 4001, str(exc))
    except Exception as e:
        logger.exception("wiki.glossary failed")
        return _err(rid, 5062, str(e))


@method("wiki.glossary.update")
def _(rid, params: dict) -> dict:
    """Validate and atomically replace a configured wiki's glossary."""
    try:
        from tui_gateway.wiki_glossary import (
            GlossaryConflictError,
            GlossaryValidationError,
            resolve_configured_wiki,
            update_glossary,
        )

        try:
            allowed = {"wiki", "version", "mode", "proper_nouns", "if_match"}
            unknown = set(params) - allowed
            if unknown:
                raise GlossaryValidationError(
                    f"unknown parameter(s): {', '.join(sorted(map(str, unknown)))}"
                )
            wiki_path = resolve_configured_wiki(params.get("wiki"))
            if_match = params.get("if_match")
            if if_match is not None and not isinstance(if_match, str):
                raise GlossaryValidationError("if_match must be a string")
            payload = {
                "version": params.get("version"),
                "mode": params.get("mode"),
                "proper_nouns": params.get("proper_nouns"),
            }
            return _ok(
                rid,
                update_glossary(wiki_path, payload, if_match=if_match),
            )
        except GlossaryConflictError as exc:
            return _err(rid, 409, str(exc))
        except GlossaryValidationError as exc:
            return _err(rid, 4001, str(exc))
    except Exception as e:
        logger.exception("wiki.glossary.update failed")
        return _err(rid, 5063, str(e))


@method("wiki.taxonomy")
def _(rid, params: dict) -> dict:
    """Return the hierarchical taxonomy tree from taxonomy.yaml."""
    try:
        from tui_gateway.wiki_api import resolve_wiki, wiki_taxonomy, wiki_flatten_taxonomy
        wiki_name = params.get("wiki")
        wiki_path = resolve_wiki(wiki_name)
        tax = wiki_taxonomy(wiki_path)
        if tax is None:
            return _err(rid, 4040, "taxonomy.yaml not found")
        flat = wiki_flatten_taxonomy(wiki_path)
        return _ok(rid, {"taxonomy": tax, "flat_paths": flat})
    except Exception as e:
        logger.exception("wiki.taxonomy failed")
        return _err(rid, 5053, str(e))

@method("wiki.expand_links")
def _(rid, params: dict) -> dict:
    """Expand integration_links for a wiki page to live status objects."""
    try:
        from tui_gateway.wiki_api import resolve_wiki, wiki_expand_links
        slug = params.get("slug")
        if not slug:
            return _err(rid, 4001, "slug is required")
        wiki_name = params.get("wiki")
        wiki_path = resolve_wiki(wiki_name)
        result = wiki_expand_links(slug, wiki_path)
        if "error" in result:
            return _err(rid, 4040, result["error"])
        return _ok(rid, result)
    except Exception as e:
        logger.exception("wiki.expand_links failed")
        return _err(rid, 5054, str(e))

@method("wiki.changesets")
def _(rid, params: dict) -> dict:
    """Return paginated wiki changesets (timeline view)."""
    try:
        from tui_gateway.wiki_api import resolve_wiki, wiki_changesets
        wiki_name = params.get("wiki")
        wiki_path = resolve_wiki(wiki_name)
        result = wiki_changesets(
            wiki_path=wiki_path,
            page=params.get("page"),
            action=params.get("action"),
            trigger=params.get("trigger"),
            limit=params.get("limit", 50),
            offset=params.get("offset", 0),
            since=params.get("since"),
            until=params.get("until"),
        )
        return _ok(rid, result)
    except Exception as e:
        logger.exception("wiki.changesets failed")
        return _err(rid, 5055, str(e))

@method("wiki.events")
def _(rid, params: dict) -> dict:
    """Return the ingestion event log — what caused wiki updates.

    A join over data already on disk: raw sources under ``raw/`` are the
    events, and the changeset index records which events caused which page
    writes. Each event carries the changesets it produced, so a client can
    walk event → changeset → page and back.

    Params:
        - ``kind`` (str, optional): filter by event kind. Kinds are defined by
          ``type: event-type`` wiki pages, not a fixed list.
        - ``limit`` (int, default 200, max 1000) / ``offset`` (int, default 0)
        - ``since`` / ``until`` (ISO timestamps, optional)
        - ``wiki`` (str, optional): wiki name (omit for default).
    """
    try:
        from tui_gateway.wiki_api import resolve_wiki, wiki_events
        wiki_path = resolve_wiki(params.get("wiki"))

        result = wiki_events(
            wiki_path=wiki_path,
            kind=params.get("kind"),
            limit=params.get("limit", 200),
            offset=params.get("offset", 0),
            since=params.get("since"),
            until=params.get("until"),
        )
        return _ok(rid, result)
    except Exception as e:
        logger.exception("wiki.events failed")
        return _err(rid, 5059, str(e))

@method("wiki.changeset_diff")
def _(rid, params: dict) -> dict:
    """Return the unified git diff for a single changeset.

    Params:
        - ``id`` (str, required): changeset id from wiki.changesets.
        - ``wiki`` (str, optional): wiki name (omit for default).

    Returns ``{"diff": "<unified diff>", "changeset": {...}}``. When the wiki
    isn't git-initialized (older captures), returns error code 5057 with the
    changeset attached so clients can still show metadata.
    """
    try:
        changeset_id = params.get("id")
        if not changeset_id or not isinstance(changeset_id, str):
            return _err(rid, 4001, "id must be a non-empty string")
        from tui_gateway.wiki_api import resolve_wiki, wiki_changeset_diff
        wiki_path = resolve_wiki(params.get("wiki"))

        result = wiki_changeset_diff(changeset_id, wiki_path=wiki_path)
        if "error" in result:
            return _err(rid, 5057, result["error"])
        return _ok(rid, result)
    except Exception as e:
        logger.exception("wiki.changeset_diff failed")
        return _err(rid, 5056, str(e))

@method("wiki.update")
def _(rid, params: dict) -> dict:
    """Write a wiki page — the one mutating method on the wiki surface.

    Full-replace write with optimistic concurrency (see the ``wiki.update``
    semantics in Portal's docs/rpc-reference.md):

    - ``path`` (str, required): page path relative to the wiki root (.md,
      must resolve inside the root).
    - ``body`` (str, required): FULL replacement markdown body.
    - ``frontmatter`` (object, optional): REPLACES the entire frontmatter
      block when present; omitted preserves it. ``updated`` is always set
      server-side.
    - ``if_match`` (str, optional): optimistic-concurrency precondition —
      the ``updated`` the client read at load. Stale → error 409 with
      ``data.latest`` carrying the server's current page.
    - ``force`` (bool, default false): bypass the if_match precondition.
    - ``trigger`` (str, default "manual"): what kind of change this is.
      Was hardcoded to "manual", so every write through this method looked
      identical in the timeline regardless of what made it.
    - ``source_events`` (array of str, optional): the ingestion events that
      caused this write, as wiki-relative raw source paths. Recorded as the
      changeset's provenance; omitted reads downstream as *unknown*.
    - ``summary`` (str, optional): changeset summary.
    - ``wiki`` (str, optional): wiki name (omit for default).

    Records a changeset (action: update|create) with the usual git commit
    capture, so edits appear in wiki.changesets.
    """
    try:
        page_path = params.get("path")
        if not page_path or not isinstance(page_path, str):
            return _err(rid, 4001, "path is required")
        body = params.get("body")
        if not isinstance(body, str):
            return _err(rid, 4001, "body is required")
        frontmatter = params.get("frontmatter")
        if frontmatter is not None and not isinstance(frontmatter, dict):
            return _err(rid, 4001, "frontmatter must be an object")
        if_match = params.get("if_match")
        if if_match is not None and not isinstance(if_match, str):
            return _err(rid, 4001, "if_match must be a string")
        trigger = params.get("trigger", "manual")
        if not isinstance(trigger, str) or not trigger.strip():
            return _err(rid, 4001, "trigger must be a non-empty string")
        source_events = params.get("source_events")
        if source_events is not None and not isinstance(source_events, list):
            return _err(rid, 4001, "source_events must be an array of strings")
        summary = params.get("summary")
        if summary is not None and not isinstance(summary, str):
            return _err(rid, 4001, "summary must be a string")

        from tui_gateway.wiki_api import resolve_wiki, wiki_update

        wiki_path = resolve_wiki(params.get("wiki"))

        result = wiki_update(
            page_path,
            body,
            frontmatter=frontmatter,
            if_match=if_match,
            force=bool(params.get("force", False)),
            trigger=trigger.strip(),
            source_events=source_events,
            summary=summary,
            wiki_path=wiki_path,
        )
        if "error" in result:
            if result.get("code") == "conflict":
                return _err(rid, 409, result["error"],
                            data={"latest": result.get("latest")})
            return _err(rid, 4001, result["error"])
        return _ok(rid, result)
    except Exception as e:
        logger.exception("wiki.update failed")
        return _err(rid, 5058, str(e))

@method("artifact.set")
def _(rid, params: dict) -> dict:
    """Upsert a living artifact (a named model in the client render
    dialects — map/chart/graph/stats/table/markdown — that any writer
    maintains). Merges per kind server-side (map: markers union by label;
    others replace) unless replace=true; appends a revision; emits
    `artifact.changed` so connected clients stream the update live."""
    try:
        from tui_gateway.artifact_store import set_artifact

        raw_actions = params.get("actions")
        actions = raw_actions if isinstance(raw_actions, list) else None
        stored = set_artifact(
            artifact_id=str(params.get("id", "")),
            kind=str(params.get("kind", "")),
            content=str(params.get("content", "")),
            title=params.get("title"),
            updated_by=str(params.get("updated_by", "")),
            replace=bool(params.get("replace", False)),
            actions=actions,
        )
        _emit("artifact.changed", "", {
            "id": stored["id"], "kind": stored["kind"],
            "title": stored["title"], "rev": stored["rev"],
            "updated_at": stored["updated_at"],
            "updated_by": stored["updated_by"],
        })
        return _ok(rid, {"artifact": stored})
    except ValueError as e:
        return _err(rid, 4001, str(e))
    except Exception as e:
        logger.exception("artifact.set failed")
        return _err(rid, 5210, str(e))

@method("artifact.get")
def _(rid, params: dict) -> dict:
    """Fetch one artifact with content."""
    try:
        from tui_gateway.artifact_store import get_artifact

        artifact = get_artifact(str(params.get("id", "")))
        if artifact is None:
            return _err(rid, 4004, "artifact not found")
        return _ok(rid, {"artifact": artifact})
    except Exception as e:
        logger.exception("artifact.get failed")
        return _err(rid, 5211, str(e))

@method("artifact.list")
def _(rid, params: dict) -> dict:
    """All artifacts without content, newest first."""
    try:
        from tui_gateway.artifact_store import list_artifacts

        return _ok(rid, {"artifacts": list_artifacts()})
    except Exception as e:
        logger.exception("artifact.list failed")
        return _err(rid, 5212, str(e))

@method("artifact.delete")
def _(rid, params: dict) -> dict:
    """Remove an artifact (and its revisions); emits artifact.changed with
    deleted=true."""
    try:
        from tui_gateway.artifact_store import delete_artifact

        artifact_id = str(params.get("id", ""))
        if not delete_artifact(artifact_id):
            return _err(rid, 4004, "artifact not found")
        _emit("artifact.changed", "", {"id": artifact_id, "deleted": True})
        return _ok(rid, {"deleted": artifact_id})
    except Exception as e:
        logger.exception("artifact.delete failed")
        return _err(rid, 5213, str(e))

@method("artifact.revisions")
def _(rid, params: dict) -> dict:
    """Revision metadata for an artifact (no content), newest first —
    the audit trail: who changed what, when."""
    try:
        from tui_gateway.artifact_store import get_artifact, list_revisions

        artifact_id = str(params.get("id", ""))
        if get_artifact(artifact_id) is None:
            return _err(rid, 4004, "artifact not found")
        return _ok(rid, {"revisions": list_revisions(artifact_id)})
    except Exception as e:
        logger.exception("artifact.revisions failed")
        return _err(rid, 5214, str(e))

@method("artifact.revision")
def _(rid, params: dict) -> dict:
    """One revision's full content (time-travel view / restore source)."""
    try:
        from tui_gateway.artifact_store import get_revision

        revision = get_revision(str(params.get("id", "")), int(params.get("rev", 0)))
        if revision is None:
            return _err(rid, 4004, "revision not found")
        return _ok(rid, {"revision": revision})
    except (TypeError, ValueError):
        return _err(rid, 4001, "rev must be an integer")
    except Exception as e:
        logger.exception("artifact.revision failed")
        return _err(rid, 5215, str(e))

@method("artifact.action.invoke")
def _(rid, params: dict) -> dict:
    """Invoke a backend intent declared in an artifact's action manifest.

    The client sends only stable identifiers — artifact ID, pinned revision,
    binding ID, entity ref, and an idempotency key. The server resolves the
    registered handler from the artifact's declarations at that revision;
    a forged binding or substituted intent name is rejected because the
    server never trusts the caller's intent string.

    Returns: {"status": "needs_confirmation"|"succeeded"|"failed"|
              "conflict"|"unsupported", ...}
    """
    try:
        from tui_gateway.artifact_actions import invoke

        result = invoke(
            artifact_id=str(params.get("artifact_id", "")),
            artifact_rev=int(params.get("artifact_rev", 0)),
            binding_id=str(params.get("binding_id", "")),
            entity_ref=str(params.get("entity_ref", "")),
            idempotency_key=str(params.get("idempotency_key", "")),
        )
        if result.get("status") == "succeeded":
            # Emit artifact.changed so the client refreshes live.
            from tui_gateway.artifact_store import get_artifact
            artifact = get_artifact(str(params.get("artifact_id", "")))
            if artifact:
                _emit("artifact.changed", "", {
                    "id": artifact["id"], "kind": artifact.get("kind", ""),
                    "title": artifact.get("title", ""), "rev": artifact.get("rev", 0),
                    "updated_at": artifact.get("updated_at", ""),
                    "updated_by": artifact.get("updated_by", ""),
                })
        return _ok(rid, result)
    except (TypeError, ValueError) as e:
        return _err(rid, 4001, str(e))
    except Exception as e:
        logger.exception("artifact.action.invoke failed")
        return _err(rid, 5216, str(e))

@method("artifact.action.confirm")
def _(rid, params: dict) -> dict:
    """Complete a pending destructive intent after native confirmation.

    ``challenge`` is the short-lived token issued by artifact.action.invoke
    when the handler requires confirmation. It is bound to actor, artifact
    revision, binding, resolved target, and expiry on the server — the
    artifact cannot weaken confirmation policy by declaring confirm: false.
    """
    try:
        from tui_gateway.artifact_actions import confirm

        result = confirm(
            artifact_id=str(params.get("artifact_id", "")),
            challenge=str(params.get("challenge", "")),
        )
        if result.get("status") == "succeeded":
            from tui_gateway.artifact_store import get_artifact
            artifact = get_artifact(str(params.get("artifact_id", "")))
            if artifact:
                _emit("artifact.changed", "", {
                    "id": artifact["id"], "kind": artifact.get("kind", ""),
                    "title": artifact.get("title", ""), "rev": artifact.get("rev", 0),
                    "updated_at": artifact.get("updated_at", ""),
                    "updated_by": artifact.get("updated_by", ""),
                })
        return _ok(rid, result)
    except Exception as e:
        logger.exception("artifact.action.confirm failed")
        return _err(rid, 5217, str(e))

@method("artifact.action.log")
def _(rid, params: dict) -> dict:
    """Query the invocation ledger for an artifact.

    Params:
      artifact_id  (str, required)
      binding_id   (str, optional) — filter to one binding
      entity_ref   (str, optional) — filter to one entity
      limit        (int, optional, default 50, max 200)

    Returns newest-first list of invocation records. Native uses this to
    re-hydrate badge state on artifact-pane open after app restart.
    """
    try:
        artifact_id = (params.get("artifact_id") or "").strip()
        if not artifact_id:
            return _err(rid, 4001, "artifact_id required")
        from tui_gateway.artifact_invocation_ledger import query as _ledger_query
        records = _ledger_query(
            artifact_id=artifact_id,
            binding_id=params.get("binding_id"),
            entity_ref=params.get("entity_ref"),
            limit=int(params.get("limit", 50)),
        )
        return _ok(rid, {"records": records})
    except Exception as e:
        logger.exception("artifact.action.log failed")
        return _err(rid, 5219, str(e))

@method("actions.reload")
def _(rid, params: dict) -> dict:
    """Reload plugin action handlers from ~/.hermes/plugins/actions/.

    Returns a diff of added/changed/removed handler names and the list of
    files that were loaded. On any parse/exec error the live registry is
    unchanged and the traceback is returned in ``error``.

    Safe to call from an agent tool, CLI, or RPC — the security boundary is
    the plugins directory's non-agent-writability, not the trigger.
    """
    try:
        from tui_gateway.artifact_plugin_loader import reload as _reload
        result = _reload()
        return _ok(rid, result)
    except Exception as e:
        logger.exception("actions.reload failed")
        return _err(rid, 5218, str(e))

@method("gateway.restart")
def _(rid, params: dict) -> dict:
    """Re-exec the gateway process in place, loading all updated source files.

    Use after pulling new code onto the gateway host — the response is sent
    before the process replaces itself, so callers receive it reliably.

    The native app will experience a WebSocket disconnect followed by the
    standard reconnect sequence. On reconnect, gateway.capabilities will
    reflect any new capabilities added by the update.

    Safe to call from an agent tool (e.g. after ``git pull`` updates the
    fork). The process re-execs with the same ``sys.argv`` and inherits
    the environment, so all env vars (LINEAR_API_KEY, HERMES_HOME, etc.)
    are preserved.
    """
    import threading

    def _do_restart():
        import time
        time.sleep(0.15)  # let the response frame flush through the transport
        os.execv(sys.executable, [sys.executable] + sys.argv)

    threading.Thread(target=_do_restart, daemon=True).start()
    return _ok(rid, {"status": "restarting"})

@method("gateway.capabilities")
def _(rid, params: dict) -> dict:
    """Report gateway capabilities so native clients can feature-gate
    controls without trial-and-error method calls.

    ``capability_names`` is the authoritative set — clients check for
    substring matches (e.g. ``artifact.action``) rather than exact values
    so adding sub-capabilities doesn't break old clients.
    """
    try:
        from hermes_cli import __version__, __release_date__
        version = f"{__version__}+{__release_date__}"
    except Exception:
        version = "unknown"

    return _ok(rid, {
        "gateway_version": version,
        "capability_names": [
            "artifact.set",
            "artifact.get",
            "artifact.list",
            "artifact.delete",
            "artifact.revisions",
            "artifact.revision",
            "artifact.action",
            "artifact.action.invoke",
            "artifact.action.confirm",
            "artifact.action.reload",
            "artifact.action.log",
            "gateway.restart",
            "wiki.scan",
            "wiki.page",
            "wiki.list",
            "wiki.glossary",
            "wiki.glossary.update",
            "wiki.changesets",
            "wiki.events",
            "learning.course",
            "learning.deck",
            "learning.progress",
            "learning.review",
            "prompt.chat_mode",
        ],
    })

@method("feed.get")
def _(rid, params: dict) -> dict:
    """Return curated news feed from digest pipelines."""
    try:
        from tui_gateway.digest_store import get_feed as _feed_get
        sources = params.get("sources")
        if sources is not None and not isinstance(sources, list):
            return _err(rid, 4001, "sources must be a list or null")
        result = _feed_get(
            sources=sources, since=params.get("since"),
            limit=params.get("limit", 50), offset=params.get("offset", 0),
        )
        return _ok(rid, result)
    except Exception as e:
        logger.exception("feed.get failed")
        return _err(rid, 5200, str(e))

@method("feed.sources")
def _(rid, params: dict) -> dict:
    """Return available feed sources and article counts."""
    try:
        from tui_gateway.digest_store import get_sources as _feed_sources
        result = _feed_sources()
        return _ok(rid, result)
    except Exception as e:
        logger.exception("feed.sources failed")
        return _err(rid, 5201, str(e))

@method("feed.publish")
def _(rid, params: dict) -> dict:
    """Append articles to the news feed store (the producer side of feed.get).

    Params:
        - ``source`` (str, required): feed source name (e.g. ``"ai-digest"``),
          shown as a filter tab and used as the dedup key.
        - ``articles`` (list[dict], required): each may carry ``title``,
          ``url``, ``summary``, ``tags``, ``image_url``. Articles are deduped
          against what was already stored for the same source.

    Returns ``{"total": N}`` — the feed size after the append.
    """
    try:
        from tui_gateway.digest_store import append_digest as _feed_publish
        source = params.get("source")
        if not source or not isinstance(source, str):
            return _err(rid, 4001, "source must be a non-empty string")
        articles = params.get("articles")
        if not isinstance(articles, list):
            return _err(rid, 4001, "articles must be a list")
        total = _feed_publish(source, articles)
        return _ok(rid, {"total": total})
    except Exception as e:
        logger.exception("feed.publish failed")
        return _err(rid, 5202, str(e))

@method("push.register")
def _(rid, params: dict) -> dict:
    """Register an APNs device token for remote push notifications.

    Params:
        - ``token`` (str, required): hex APNs device token.
        - ``platform`` (str): "macos" (default) or "ios".
        - ``device_name`` (str, optional): human-readable device label.
        - ``bundle_id`` (str, optional): per-device topic override (macOS and
          iOS builds have different bundle ids).

    Returns the stored entry plus ``apns_configured`` so clients can tell the
    user when the gateway has no APNs credentials.
    """
    try:
        from tui_gateway.apns_sender import is_configured
        from tui_gateway.push_store import register_token

        token = params.get("token")
        if not token or not isinstance(token, str):
            return _err(rid, 4001, "token must be a non-empty string")
        entry = register_token(
            token,
            platform=params.get("platform", "macos"),
            device_name=params.get("device_name", ""),
            bundle_id=params.get("bundle_id"),
        )
        if "error" in entry:
            return _err(rid, 4001, entry["error"])
        return _ok(rid, {"registered": True, "apns_configured": is_configured(), "entry": entry})
    except Exception as e:
        logger.exception("push.register failed")
        return _err(rid, 5210, str(e))

@method("push.unregister")
def _(rid, params: dict) -> dict:
    """Remove an APNs device token (e.g. on sign-out)."""
    try:
        from tui_gateway.push_store import unregister_token

        token = params.get("token")
        if not token or not isinstance(token, str):
            return _err(rid, 4001, "token must be a non-empty string")
        return _ok(rid, {"removed": unregister_token(token)})
    except Exception as e:
        logger.exception("push.unregister failed")
        return _err(rid, 5211, str(e))


def register(server) -> None:
    _registry.install(server)

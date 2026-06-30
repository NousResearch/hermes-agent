"""Agent-facing Skills-Hub DISCOVERY (Phase 6 — the "employee model").

Lets an agent find and preview skills it does NOT already have, when a job
needs an approach outside its mastered set. This is the *discover* half of the
employee model (identify a limit -> discover -> acquire OR delegate); it is
strictly **READ-ONLY**: it searches/inspects remote skill registries but never
installs. Acquisition (install) is a separate, trust-gated step.

Wraps the existing Skills Hub library (``tools.skills_hub`` +
``hermes_cli.skills_hub``) that backs the ``hermes skills`` CLI, so it inherits
the same source router, trust ranking, and index-aware search.

Registered into the ``skills`` toolset (auto-imported by tools/registry.py).
"""
import json

from tools.registry import registry, tool_error
from tools.skills_tool import check_skills_requirements

_MAX_LIMIT = 20
_TRUST_RANK = {"builtin": 3, "trusted": 2, "community": 1}


def _search(query: str, limit: int):
    from tools.skills_hub import (
        GitHubAuth, create_source_router, parallel_search_sources,
    )

    sources = create_source_router(GitHubAuth())
    results, _counts, _timed_out = parallel_search_sources(
        sources, query=query, source_filter="all", overall_timeout=20,
    )
    # Dedupe by identifier, keeping the most-trusted variant.
    seen: dict = {}
    for r in results:
        rank = _TRUST_RANK.get(r.trust_level, 0)
        if r.identifier not in seen or rank > _TRUST_RANK.get(seen[r.identifier].trust_level, 0):
            seen[r.identifier] = r
    ranked = sorted(
        seen.values(),
        key=lambda r: (-_TRUST_RANK.get(r.trust_level, 0), r.name.lower()),
    )[:limit]
    return [
        {"name": r.name, "description": r.description, "source": r.source,
         "trust": r.trust_level, "identifier": r.identifier}
        for r in ranked
    ]


def skill_discover(query=None, identifier=None, limit=10, task_id=None):
    try:
        limit = max(1, min(int(limit or 10), _MAX_LIMIT))
    except (TypeError, ValueError):
        limit = 10

    try:
        if identifier:
            from hermes_cli.skills_hub import inspect_skill
            meta = inspect_skill(str(identifier))
            if not meta:
                return json.dumps({"success": False, "error": f"no skill found for '{identifier}'"})
            return json.dumps({"success": True, "skill": meta})

        if not query or not str(query).strip():
            return json.dumps({
                "success": False,
                "error": "provide 'query' to search the hub, or 'identifier' to inspect a skill",
            })

        results = _search(str(query).strip(), limit)
        return json.dumps({
            "success": True,
            "query": query,
            "count": len(results),
            "results": results,
            "note": "Discovery only — this does NOT install. To use a found skill, "
                    "request acquisition (a separate, approved step) or delegate the task.",
        })
    except Exception as exc:  # noqa: BLE001 - never crash the agent loop on a hub hiccup
        return tool_error(f"skill_discover failed: {exc}")


SKILL_DISCOVER_SCHEMA = {
    "name": "skill_discover",
    "description": (
        "Search the Skills Hub for skills you don't already have — use when a task "
        "needs an approach outside your mastered skills. Search by need (e.g. "
        "'parse PDF bank statements') to see candidates, then pass an `identifier` "
        "to preview that skill's details + SKILL.md. READ-ONLY: this does not "
        "install anything; acquiring a skill is a separate, approved step. Always "
        "check your own skills_list first — only reach for the hub when you're "
        "genuinely missing a capability the job needs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language description of the capability you need."},
            "identifier": {"type": "string", "description": "A skill identifier from a prior search result, to preview its details."},
            "limit": {"type": "integer", "description": f"Max results (default 10, max {_MAX_LIMIT})."},
        },
        "required": [],
    },
}


registry.register(
    name="skill_discover",
    toolset="skills",
    schema=SKILL_DISCOVER_SCHEMA,
    handler=lambda args, **kw: skill_discover(
        query=args.get("query"), identifier=args.get("identifier"),
        limit=args.get("limit", 10), task_id=kw.get("task_id"),
    ),
    check_fn=check_skills_requirements,
    emoji="🔎",
)


# Sources we'll auto-install from (builtin/official + curated trusted repos).
# Everything else (community/unknown) is held in quarantine for human approval.
_AUTO_INSTALL_TRUST = {"builtin", "trusted"}


def skill_acquire(identifier, category="", task_id=None):
    """Acquire a discovered skill (the *acquire* half of the employee model).

    Trust policy (Alon, 2026-06-18): a skill from a TRUSTED source (builtin /
    official / curated-trusted) installs automatically once the safety scan
    allows it; a skill from a community/unknown source — or one the scanner
    flags — needs a human's OK before it installs. Reuses the Hub's own
    quarantine + scanner; never bypasses them.

    Approval routing (2026-06-18): when the P5.3 tool-approval gate is enabled
    on this profile, an untrusted/flagged acquire is **staged** through that
    gate — it opens a ``hermes-action-approval`` Kanban card (rendered in the
    console with Approve/Reject, posted to the Mattermost approvals channel)
    and returns ``staged``. Approving it spawns a one-shot execution worker
    that REPLAYS this same call with a replay token; the body detects that
    token and installs directly (operator already approved), so the community
    install completes without a CLI step and without re-staging. With the gate
    OFF, behaviour is unchanged: the call returns ``requires_approval`` with the
    ``hermes skills install`` CLI hint.
    """
    if not identifier or not str(identifier).strip():
        return json.dumps({"success": False, "error": "provide a skill 'identifier' (from skill_discover)"})
    try:
        from tools.skills_hub import (
            GitHubAuth, create_source_router, ensure_hub_dirs,
            quarantine_bundle, install_from_quarantine, HubLockFile,
        )
        from tools.skills_guard import scan_skill, should_allow_install
        from hermes_cli.skills_hub import _resolve_source_meta_and_bundle, _resolve_short_name
        from tools import tool_gate

        # Is this invocation the deterministic replay of an already-approved
        # acquire? The choke-point gate does NOT consume the replay token for
        # skill_acquire (it is not in `require_approval`), so the token survives
        # into this body. A match means the operator approved THIS exact pending
        # acquisition — install regardless of trust, and do NOT re-stage.
        approved_pid = tool_gate.consume_replay_token("skill_acquire")

        ensure_hub_dirs()
        sources = create_source_router(GitHubAuth())

        class _Quiet:
            def print(self, *a, **k):
                pass

        ident = str(identifier).strip()
        if "/" not in ident:
            ident = _resolve_short_name(ident, sources, _Quiet())
            if not ident:
                return json.dumps({"success": False, "error": f"no skill found for '{identifier}'"})

        meta, bundle, _matched = _resolve_source_meta_and_bundle(ident, sources)
        if not bundle or not meta:
            return json.dumps({"success": False, "error": f"could not fetch '{identifier}' from any source"})

        if HubLockFile().get_installed(bundle.name):
            return json.dumps({"success": True, "installed": True, "already_installed": True,
                               "skill": bundle.name, "note": "Already installed — use skill_view to load it."})

        trust = getattr(meta, "trust_level", "community") or "community"
        qpath = quarantine_bundle(bundle)
        scan = scan_skill(qpath, source=trust)
        scan_allows, reason = should_allow_install(scan)
        verdict = getattr(scan, "verdict", "unknown")

        # 1. Trusted source + clean scan -> auto-install.
        # 2. Operator already approved THIS acquire (replay) -> install regardless
        #    of trust. The fresh re-fetch + re-scan above is a TOCTOU safeguard;
        #    the operator's approval overrides the trust requirement, not a hard
        #    scanner block.
        if (trust in _AUTO_INSTALL_TRUST and scan_allows) or approved_pid:
            install_from_quarantine(qpath, bundle.name, category or "", bundle, scan)
            note = ("Installed after operator approval." if approved_pid
                    else "Installed (trusted source, scan passed).")
            return json.dumps({
                "success": True, "installed": True, "skill": bundle.name,
                "trust": trust, "scan_verdict": verdict,
                "approved_via": approved_pid or None,
                "note": f"{note} Load it with skill_view.",
            })

        # 3. Untrusted source / flagged scan, first time. When the approval gate
        #    is enabled, stage the acquire through it (console + Mattermost card)
        #    so a human can approve in-band; otherwise fall back to the CLI hint.
        gate_cfg = tool_gate.get_tool_gate_config()
        if tool_gate.gate_enabled(gate_cfg):
            summary = (f"Install community skill '{bundle.name}' "
                       f"(source trust='{trust}', scan='{verdict}')")
            staged = tool_gate.stage_deferred(
                "skill_acquire",
                {"identifier": ident, "category": category or ""},
                summary=summary, config=gate_cfg,
            )
            return json.dumps({
                "success": True, "installed": False, "status": "staged",
                "skill": bundle.name, "trust": trust, "scan_verdict": verdict, "reason": reason,
                "pending_id": staged.get("pending_id"), "card_id": staged.get("card_id"),
                "note": (
                    f"Acquisition of '{bundle.name}' (source trust='{trust}', scan='{verdict}') "
                    f"is queued for operator approval — this is expected, do NOT retry. It "
                    f"installs automatically once approved; meanwhile, delegate the task or "
                    f"proceed without the skill."
                ),
            })

        # 4. Gate not enabled -> hold in quarantine for the CLI approval path.
        return json.dumps({
            "success": True, "installed": False, "status": "requires_approval",
            "skill": bundle.name, "trust": trust, "scan_verdict": verdict, "reason": reason,
            "note": (
                f"Not auto-installed: source trust='{trust}'"
                + ("" if trust in _AUTO_INSTALL_TRUST else " (untrusted)")
                + f", scan verdict='{verdict}'. Held in quarantine — an operator can approve "
                f"with `hermes skills install {ident}`. Meanwhile, consider delegating the task."
            ),
        })
    except Exception as exc:  # noqa: BLE001
        return tool_error(f"skill_acquire failed: {exc}")


SKILL_ACQUIRE_SCHEMA = {
    "name": "skill_acquire",
    "description": (
        "Acquire (install) a skill you found with skill_discover, so you can then use it. "
        "Only skills from TRUSTED sources install automatically (after a safety scan); skills "
        "from community/unknown sources are queued for an operator's approval (status "
        "'staged') and won't be available immediately — that 'staged' result is SUCCESS, not "
        "an error: do NOT retry it. It installs by itself once approved; meanwhile delegate "
        "the task or proceed without. Use sparingly: only when a task genuinely needs a "
        "capability you and your skills_list lack."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "identifier": {"type": "string", "description": "The skill identifier from a skill_discover result."},
            "category": {"type": "string", "description": "Optional category/subfolder to install under."},
        },
        "required": ["identifier"],
    },
}


registry.register(
    name="skill_acquire",
    toolset="skills",
    schema=SKILL_ACQUIRE_SCHEMA,
    handler=lambda args, **kw: skill_acquire(
        identifier=args.get("identifier"), category=args.get("category", ""),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_skills_requirements,
    emoji="📥",
)

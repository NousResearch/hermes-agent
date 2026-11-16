"""ownership — read and change automation ownership without a shell.

``hermes own`` is a CLI, and reaching a CLI from a messaging platform needs the
``terminal`` toolset, which only ``admin`` holds. That made every ownership
operation admin-only: the claim nudge told non-admins to run a command they
cannot run, an owner could not hand off their own automation, and nobody could
find out who owns something before editing it.

This tool exposes the same operations to any valid-role user. The acting
identity comes ONLY from session contextvars — never from tool arguments — so
unlike the CLI's ``--user`` / ``--by`` flags there is no way to act as someone
else. Permission checks live in ``agent.automation_ownership`` so every surface
shares them.

Not a security boundary: ownership stays an awareness + collaboration layer and
RBAC (gateway/tool_access.py) remains the real tool-access boundary. See
docs/superpowers/specs/2026-08-03-automation-ownership-tool-design.md.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import agent.automation_ownership as ao
from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

_KINDS = ("skill", "cron", "script", "automation")
_MUTATING = frozenset({"claim", "transfer", "collab_add", "collab_remove"})


OWNERSHIP_SCHEMA = {
    "name": "ownership",
    "description": (
        "Read and change who owns a user-built automation (skill, cron job, script, "
        "or automation bundle). Use this instead of the `hermes own` CLI — it works "
        "for any user, not just admins. Keys look like 'cron:9f3a1c2b', "
        "'skill:weekly-report', 'script:reports/weekly.py', 'automation:weekly-report' "
        "— the ownership nudge and cross-user gate messages print the exact key. "
        "Actions: list (what you own), show (who owns a key), claim (take an UNOWNED "
        "automation), transfer (hand yours to a teammate), collab_add / collab_remove "
        "(let a teammate edit yours without hitting the confirmation gate). Only the "
        "owner or an admin may transfer or change collaborators. Never claim or "
        "transfer on the user's behalf without asking them first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "show", "claim", "transfer", "collab_add", "collab_remove"],
                "description": "The ownership operation to perform.",
            },
            "key": {
                "type": "string",
                "description": (
                    "Automation key as '<kind>:<id>' where kind is one of "
                    "skill, cron, script, automation. Required for every action except list."
                ),
            },
            "to_user": {
                "type": "string",
                "description": (
                    "transfer only: the teammate to hand ownership to — a platform user id "
                    "or their name."
                ),
            },
            "user": {
                "type": "string",
                "description": (
                    "collab_add / collab_remove: the teammate to add or remove, as a user id "
                    "or name. On list (admins only): whose automations to list."
                ),
            },
            "confirm_capability_loss": {
                "type": "boolean",
                "description": (
                    "transfer of a cron job only: a cron job runs with its OWNER's role, "
                    "so a transfer to someone whose role lacks what the job needs is "
                    "refused. Set true only after telling the user exactly what the job "
                    "will lose and they confirm."
                ),
            },
        },
        "required": ["action"],
    },
}


# --------------------------------------------------------------------------- #
# Identity, admin resolution, target lookup
# --------------------------------------------------------------------------- #
def _current_chat_id() -> Optional[str]:
    """Active chat id from session contextvars, so ``channel_roles`` grants count."""
    try:
        from gateway.session_context import get_session_env

        return get_session_env("HERMES_SESSION_CHAT_ID", "") or None
    except Exception:
        return None


def _is_admin(identity: ao.Identity) -> bool:
    """True when *identity*'s RBAC role grants everything ("*").

    Mirrors cron/rbac_ceiling.cron_owner_grant's resolution. Fail-CLOSED: with
    RBAC disabled or unresolvable there is no admin, so only owners may
    transfer and the CLI stays the override path.
    """
    try:
        from gateway.tool_access import policy_for_platform

        policy = policy_for_platform(identity.platform or "")
        if policy is None or not policy.enabled:
            return False
        grant = policy.grant_for(identity.user_id, _current_chat_id())
        return bool(grant) and "*" in grant
    except Exception as err:
        logger.debug("[ownership] admin resolution failed (treating as non-admin): %s", err)
        return False


def _user_directory(platform: str) -> Dict[str, str]:
    """Return ``{user_id: display_name}`` for *platform* from RBAC config.

    Unions ``user_names`` (id → human name) with the ids in ``user_roles`` so a
    user with a role but no name entry is still resolvable. Empty when the
    platform has no directory configured.
    """
    try:
        from gateway.tool_access import _load_config_cached, _platform_extra
        from gateway.config import Platform

        config = _load_config_cached()
        if config is None:
            return {}
        extra = _platform_extra((getattr(config, "platforms", {}) or {}).get(Platform(platform)))
        out: Dict[str, str] = {}
        roles = extra.get("user_roles")
        if isinstance(roles, dict):
            for uid in roles:
                out[str(uid).strip()] = ""
        names = extra.get("user_names")
        if isinstance(names, dict):
            for uid, name in names.items():
                out[str(uid).strip()] = str(name).strip()
        return {k: v for k, v in out.items() if k}
    except Exception as err:
        logger.debug("[ownership] user directory lookup failed: %s", err)
        return {}


def _resolve_target(raw: str, platform: str) -> Tuple[Optional[ao.Identity], str]:
    """Resolve a user id or name to an Identity. Returns (identity, error).

    An unresolvable target is an error rather than a write: a record pointing at
    a typo'd id is invisible garbage — the owner is never notified and the cron
    toolset ceiling stops resolving a role for that job.
    """
    target = (raw or "").strip()
    if target.startswith("<@") and target.endswith(">"):
        target = target[2:-1].split("|", 1)[0]
    target = target.lstrip("@").strip()
    if not target:
        return None, "Name a user to act on (a platform user id or their name)."

    directory = _user_directory(platform)
    if not directory:
        # No configured directory to validate against — accept the id verbatim.
        return ao.Identity(platform=platform, user_id=target, display_name=target), ""

    if target in directory:
        return ao.Identity(
            platform=platform, user_id=target, display_name=directory[target] or target
        ), ""

    matches = [uid for uid, name in directory.items() if name and name.lower() == target.lower()]
    if len(matches) == 1:
        return ao.Identity(
            platform=platform, user_id=matches[0], display_name=directory[matches[0]]
        ), ""
    if len(matches) > 1:
        return None, (
            f"'{raw}' matches more than one user ({', '.join(sorted(matches))}). "
            "Use their user id instead."
        )
    return None, (
        f"No user matching '{raw}'. Use their platform user id, or the name they are "
        "registered under."
    )


def _validate_key(key: str) -> Tuple[str, str, str]:
    """Return (key, kind, error) for a '<kind>:<id>' automation key."""
    key = (key or "").strip()
    if not key:
        return "", "", "Pass the automation key, e.g. key=\"cron:9f3a1c2b\"."
    kind, _, ident = key.partition(":")
    kind = kind.strip().lower()
    if kind not in _KINDS or not ident.strip():
        return "", "", (
            f"'{key}' is not a valid automation key. Use '<kind>:<id>' where kind is one "
            f"of {', '.join(_KINDS)} — e.g. 'cron:9f3a1c2b' or 'skill:weekly-report'."
        )
    return key, kind, ""


def _describe(record: dict) -> dict:
    owner = record.get("owner") or {}
    return {
        "kind": record.get("kind", ""),
        "owner": owner.get("display_name") or owner.get("user_id") or "",
        "owner_user_id": owner.get("user_id") or "",
        "collaborators": [
            c.get("display_name") or c.get("user_id") or ""
            for c in record.get("collaborators", [])
        ],
        "source": record.get("source", ""),
    }


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #
def _do_list(args: dict, actor: ao.Identity) -> str:
    requested = (args.get("user") or "").strip()
    target = actor
    if requested:
        if not _is_admin(actor):
            return tool_error(
                "Only an admin can list another user's automations. Call this with no "
                "'user' argument to see your own."
            )
        target, err = _resolve_target(requested, actor.platform)
        if target is None:
            return tool_error(err)
    out = ao.list_for_user(target.user_id)
    return tool_result({
        "user": target.display_name or target.user_id,
        "owned": out["owned"],
        "collaborator": out["collaborator"],
    })


def _do_show(key: str, record: Optional[dict]) -> str:
    if record is None:
        return tool_result({
            "key": key,
            "owned": False,
            "message": (
                f"{key} has no recorded owner. Anyone can edit it; claim it with "
                f'action="claim" if it is yours.'
            ),
        })
    return tool_result({"key": key, "owned": True, **_describe(record)})


def _do_claim(key: str, kind: str, actor: ao.Identity) -> str:
    try:
        record = ao.claim(key, kind, actor)
    except PermissionError as err:
        return tool_error(str(err))
    ao.record_ownership_change("automation_claim", key, actor)
    return tool_result({
        "ok": True,
        "key": key,
        "owner": record["owner"]["display_name"],
        "message": f"You now own {key}.",
    })


def _do_transfer(args: dict, key: str, actor: ao.Identity, record: dict) -> str:
    new_owner, err = _resolve_target(args.get("to_user") or "", actor.platform)
    if new_owner is None:
        return tool_error(err)
    previous = dict(record.get("owner") or {})
    loss = _cron_capability_loss(key, new_owner)
    if loss and not args.get("confirm_capability_loss"):
        return tool_error(
            f"{key} runs with its owner's role, and under {new_owner.display_name}'s "
            f"role it cannot use what it needs: {loss}. The job would keep running but "
            "fail at that step. Tell the user, and either have an admin extend "
            f"{new_owner.display_name}'s role first, or re-run with "
            "confirm_capability_loss=true once they accept the loss."
        )
    try:
        updated = ao.transfer(key, new_owner, by=actor, by_is_admin=_is_admin(actor))
    except (KeyError, PermissionError) as exc:
        return tool_error(str(exc))
    actor_name = actor.display_name or actor.user_id
    ao.record_ownership_change(
        "automation_transfer", key, actor,
        notify=[previous, updated["owner"]],
        message=(
            f"{actor_name} transferred the automation `{key}` to "
            f"{new_owner.display_name}. — via Hermes automation ownership"
        ),
    )
    result = {
        "ok": True,
        "key": key,
        "owner": updated["owner"]["display_name"],
        "message": f"{key} now belongs to {new_owner.display_name}.",
    }
    capabilities = _cron_capabilities_after_transfer(key, new_owner.display_name)
    if capabilities:
        result["capabilities"] = capabilities
    return tool_result(result)


def _cron_capability_loss(key: str, new_owner: ao.Identity) -> str:
    """For a ``cron:`` key, what the job needs that ``new_owner``'s role would
    strip ("" when nothing / not a cron key / not evaluable). On 2026-09-14 a
    transfer silently removed ``terminal`` from the MRR automation's cron runs,
    and its sheet writes stopped without any error."""
    if not key.startswith("cron:"):
        return ""
    try:
        from cron.capability_preflight import evaluate_job_capabilities, grant_for_identity
        from cron.jobs import get_job
        from hermes_cli.config import load_config

        job = get_job(key.split(":", 1)[1])
        if not job or job.get("no_agent"):
            return ""
        chat_id = (job.get("origin") or {}).get("chat_id")
        report = evaluate_job_capabilities(
            job, load_config() or {}, grant_for_identity(new_owner, chat_id),
            include_enabled=True, check_availability=False,
        )
        return "; ".join(p for p in report.problems if "role" in p)
    except Exception:
        return ""


def _cron_capabilities_after_transfer(key: str, owner_name: str) -> dict:
    """For a ``cron:`` item, re-evaluate the job's toolsets under the NEW
    owner's role (the runtime ceiling follows the owner). A narrower role can
    silently strip what the job needs, and a wider one does not widen the
    job's stored ``enabled_toolsets`` — both are worth saying at transfer
    time. Best-effort: {} on any error or for non-cron keys."""
    if not key.startswith("cron:"):
        return {}
    try:
        from cron.capability_preflight import evaluate_job_capabilities
        from cron.jobs import get_job
        from cron.rbac_ceiling import cron_owner_grant
        from hermes_cli.config import load_config

        job = get_job(key.split(":", 1)[1])
        if not job or job.get("no_agent"):
            return {}
        report = evaluate_job_capabilities(
            job, load_config() or {}, cron_owner_grant(job),
            include_enabled=True, check_availability=False,
        )
        out: dict = {"effective_toolsets": report.effective or "all default toolsets"}
        if report.problems:
            out["warning"] = (
                f"Under {owner_name}'s role this job cannot use what it needs: "
                + "; ".join(report.problems)
            )
        return out
    except Exception:
        return {}


def _do_collab(action: str, args: dict, key: str, actor: ao.Identity, record: dict) -> str:
    target, err = _resolve_target(args.get("user") or "", actor.platform)
    if target is None:
        return tool_error(err)
    is_admin = _is_admin(actor)
    actor_name = actor.display_name or actor.user_id
    try:
        if action == "collab_add":
            ao.add_collaborator(key, target, by=actor, by_is_admin=is_admin)
            verb, audit = "added to", "automation_collab_add"
        else:
            ao.remove_collaborator(key, target.user_id, by=actor, by_is_admin=is_admin)
            verb, audit = "removed from", "automation_collab_remove"
    except (KeyError, PermissionError) as exc:
        return tool_error(str(exc))
    ao.record_ownership_change(
        audit, key, actor,
        notify=[{"platform": target.platform, "user_id": target.user_id}],
        message=(
            f"{actor_name} {verb} the automation `{key}` as a collaborator. "
            "— via Hermes automation ownership"
        ),
    )
    return tool_result({
        "ok": True,
        "key": key,
        "message": f"{target.display_name} was {verb} {key}.",
    })


async def _ownership_handler(args: dict, **_kw) -> str:
    action = (args.get("action") or "").strip().lower()
    if action not in {"list", "show", "claim", "transfer", "collab_add", "collab_remove"}:
        return tool_error(
            "Unknown action. Use one of: list, show, claim, transfer, collab_add, "
            "collab_remove."
        )
    if not ao.is_enabled():
        return tool_error("Automation ownership tracking is disabled on this install.")

    actor = ao.current_identity()
    if actor is None:
        return tool_error(
            "No acting user identity — ownership changes must be made by a person, not "
            "an autonomous run. Ask the user to do this from their own conversation."
        )

    if action == "list":
        return _do_list(args, actor)

    key, kind, err = _validate_key(args.get("key") or "")
    if err:
        return tool_error(err)

    record = ao.get_record(key)
    if action == "show":
        return _do_show(key, record)
    if action == "claim":
        return _do_claim(key, kind, actor)

    if record is None:
        return tool_error(
            f"{key} has no recorded owner yet, so there is nothing to "
            f"{'transfer' if action == 'transfer' else 'share'}. "
            'Claim it first with action="claim".'
        )
    if action == "transfer":
        return _do_transfer(args, key, actor, record)
    return _do_collab(action, args, key, actor, record)


registry.register(
    name="ownership",
    toolset="ownership",
    schema=OWNERSHIP_SCHEMA,
    handler=lambda args, **kw: _ownership_handler(args, **kw),
    check_fn=ao.is_enabled,
    requires_env=[],
    is_async=True,
    emoji="🔑",
    max_result_size_chars=8000,
)

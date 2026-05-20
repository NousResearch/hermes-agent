"""
tools/skills_scoped.py — S3-backed scoped skill resolver/writer for SaaS mode.

Scope model (personal wins over team wins over global):
    personal/{platform}/{team_id}/{user_id}/skills/{name}/SKILL.md
    team/{platform}/{team_id}/skills/{name}/SKILL.md
    global/skills/{name}/SKILL.md

Resolution:
    resolve_skill()  — walks identity.scope_chain, returns first S3 hit or None.
write_skill()    — writes to personal or team scope. Global writes are hard-blocked.
list_skills()    — returns all visible skills, annotated with their scope.
                   Personal skills shadow same-named team/global entries.
promote_skill_to_team() — copies a personal skill to team scope.

Assumptions surfaced explicitly:
- HERMES_MODE=saas must be set for callers that gate on this module. The
  functions here do NOT check HERMES_MODE themselves — gating lives in
  skill_manager_tool.py so this module stays unit-testable without env setup.
- S3 bucket name is configurable via HERMES_SKILLS_BUCKET env var, defaulting
  to "hermes-skills". The bucket and IAM policy are provisioned separately (AWS
  apply gate in Phase D).
- boto3 is an optional dependency. If it isn't installed (local dev without saas
  deps), import-time ImportError is caught and re-raised at call time with a
  clear message — so the local mode path is never broken by a missing dep.
- ClientError is used to detect missing objects (NoSuchKey equivalent). Using
  ClientError rather than s3.exceptions.NoSuchKey because the exceptions attr is
  only available on the client instance, not the module, making it awkward to
  mock in tests.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

BUCKET = os.environ.get("HERMES_SKILLS_BUCKET", "hermes-skills")

# Lazy import so local-mode paths never require boto3 to be installed.
_boto3_import_error: Optional[ImportError] = None
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError as _exc:
    boto3 = None  # type: ignore[assignment]
    ClientError = Exception  # type: ignore[assignment,misc]
    _boto3_import_error = _exc


def _require_boto3() -> None:
    """Raise a clear error if boto3 is not available."""
    if _boto3_import_error is not None:
        raise ImportError(
            "boto3 is required for scoped skill operations in SaaS mode. "
            f"Original import error: {_boto3_import_error}"
        ) from _boto3_import_error


def _s3_client():
    """Return a boto3 S3 client. Callers must have called _require_boto3() first."""
    return boto3.client("s3")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_skill(name: str, identity) -> Optional[str]:
    """
    Resolve a skill by name, walking personal → team → global.

    Returns the SKILL.md content string for the first scope that has a match,
    or None if the skill does not exist in any scope.

    Parameters
    ----------
    name : str
        Skill directory name (e.g. "deploy-to-prod").
    identity : HermesIdentity
        Caller identity from the current agent turn.

    Raises
    ------
    ImportError
        If boto3 is not installed (SaaS dep missing in local mode).
    """
    _require_boto3()
    s3 = _s3_client()

    for scope in identity.scope_chain:
        key = f"{scope}/skills/{name}/SKILL.md"
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=key)
            content = obj["Body"].read().decode("utf-8")
            logger.debug("resolve_skill(%r): hit at scope %r", name, scope)
            return content
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")  # type: ignore[attr-defined]
            if error_code in ("NoSuchKey", "404"):
                logger.debug("resolve_skill(%r): miss at scope %r", name, scope)
                continue
            # Any other S3 error is a genuine failure — let it propagate.
            raise

    logger.debug("resolve_skill(%r): not found in any scope", name)
    return None


def write_skill(name: str, content: str, scope: str, identity) -> None:
    """
    Write a skill to the specified scope.

    Parameters
    ----------
    name : str
        Skill directory name.
    content : str
        Full SKILL.md text content.
    scope : str
        One of "personal" or "team". Writing to "global" is permanently
        hard-blocked — no agent turn may overwrite platform defaults.
    identity : HermesIdentity
        Caller identity from the current agent turn.

    Raises
    ------
    PermissionError
        If scope == "global".
    ValueError
        If scope is not a recognised value.
    ImportError
        If boto3 is not installed.
    """
    if scope == "global":
        raise PermissionError(
            "Agent turns cannot write to the global skill scope. "
            "Global skills are platform-managed read-only defaults."
        )
    if scope not in ("personal", "team"):
        raise ValueError(f"Unknown scope {scope!r}. Use 'personal' or 'team'.")

    _require_boto3()
    s3 = _s3_client()

    prefix = identity.personal_scope if scope == "personal" else identity.team_scope
    key = f"{prefix}/skills/{name}/SKILL.md"

    logger.info("write_skill(%r): writing to %r (%s)", name, key, scope)
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=content.encode("utf-8"),
        ContentType="text/markdown",
    )


def list_skills(identity) -> list[dict]:
    """
    Return all skills visible to this identity, annotated with their scope.

    Resolution order: personal → team → global. A personal skill shadows a
    same-named team or global skill (only the most-specific entry is returned).

    Each entry is a dict:
        {
            "name":    str,   # skill directory name
            "scope":   str,   # "personal" | "team" | "global"
            "s3_key":  str,   # full S3 object key
        }

    Parameters
    ----------
    identity : HermesIdentity
        Caller identity from the current agent turn.

    Raises
    ------
    ImportError
        If boto3 is not installed.
    """
    _require_boto3()
    s3 = _s3_client()

    # scope_label maps scope string → human-readable label
    scope_labels = {
        identity.personal_scope: "personal",
        identity.team_scope: "team",
        identity.global_scope: "global",
    }

    # Walk scopes in resolution order; track seen names so personal shadows team/global.
    seen: set[str] = set()
    result: list[dict] = []

    for scope in identity.scope_chain:
        scope_label = scope_labels[scope]
        prefix = f"{scope}/skills/"

        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix, Delimiter="/"):
            # Each skill lives as a "subdirectory" whose key ends with "/"
            for common_prefix in page.get("CommonPrefixes", []):
                # e.g. "personal/slack/T01/U01/skills/deploy-to-prod/"
                skill_dir = common_prefix["Prefix"]
                # Extract the skill name: second-to-last segment
                name = skill_dir.rstrip("/").rsplit("/", 1)[-1]
                if not name or name in seen:
                    # Shadowed by a more-specific scope already collected.
                    continue
                seen.add(name)
                result.append(
                    {
                        "name": name,
                        "scope": scope_label,
                        "s3_key": f"{skill_dir}SKILL.md",
                    }
                )

    return result


def promote_skill_to_team(name: str, identity) -> None:
    """
    Copy a personal skill to team scope.

    This makes the skill visible to all members of the same team. It does NOT
    remove the personal copy — the user's personal version continues to shadow
    the team copy for them personally (per the resolution order).

    Future: require team admin role before promoting. Currently any team member
    can promote (per Phase A spec — role enforcement deferred to Phase C).

    Parameters
    ----------
    name : str
        Skill directory name to promote.
    identity : HermesIdentity
        Caller identity. Promotion reads from personal_scope, writes to team_scope.

    Raises
    ------
    FileNotFoundError
        If the named skill does not exist in the caller's personal scope.
    ImportError
        If boto3 is not installed.
    """
    # Resolve from personal scope specifically — not the full chain.
    _require_boto3()
    s3 = _s3_client()

    personal_key = f"{identity.personal_scope}/skills/{name}/SKILL.md"
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=personal_key)
        content = obj["Body"].read().decode("utf-8")
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")  # type: ignore[attr-defined]
        if error_code in ("NoSuchKey", "404"):
            raise FileNotFoundError(
                f"No personal skill '{name}' found at s3://{BUCKET}/{personal_key}. "
                "Cannot promote a skill that does not exist in personal scope."
            ) from exc
        raise

    team_key = f"{identity.team_scope}/skills/{name}/SKILL.md"
    logger.info(
        "promote_skill_to_team(%r): %r → %r", name, personal_key, team_key
    )
    s3.put_object(
        Bucket=BUCKET,
        Key=team_key,
        Body=content.encode("utf-8"),
        ContentType="text/markdown",
    )

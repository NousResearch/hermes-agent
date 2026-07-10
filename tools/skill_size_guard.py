"""Configurable prompt-efficiency ratchet for agent-managed SKILL.md files.

The storage hard limit remains in ``skill_manager_tool``.  This module owns the
softer policy that keeps the main skill body lean while allowing rich support
files under references/, templates/, scripts/, and assets/.
"""

from typing import Any, Dict, List, Optional, Tuple

from hermes_cli.config import cfg_get

DEFAULT_SKILL_MD_SOFT_LIMIT_CHARS = 20_000
DEFAULT_SKILL_MD_MAX_GROWTH_CHARS = 5_000
_SKILL_MD_SIZE_GUARD_MODES = {"off", "auto", "warn", "enforce"}


def _positive_int_config(value: Any, default: int) -> int:
    """Return a positive integer config value or *default*.

    ``bool`` is deliberately rejected even though it is an ``int`` subclass;
    ``true`` should never become a one-character skill limit.
    """
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _skill_md_size_guard_settings(skill_name: str) -> Tuple[str, int, int, str]:
    """Load the profile-scoped SKILL.md size policy.

    ``auto`` is deliberately non-breaking for foreground/user-directed work:
    it emits an advisory there, but enforces the same ratchet in autonomous
    background review. Explicit ``enforce`` blocks both origins; ``warn``
    advises both; ``off`` preserves the historical hard-limit-only behavior.
    """
    mode = "auto"
    soft_limit = DEFAULT_SKILL_MD_SOFT_LIMIT_CHARS
    growth_limit = DEFAULT_SKILL_MD_MAX_GROWTH_CHARS
    policy_source = "default"
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        raw_mode = cfg_get(cfg, "skills", "skill_md_size_guard")
        if raw_mode is not None:
            candidate = str(raw_mode).strip().lower()
            if candidate in _SKILL_MD_SIZE_GUARD_MODES:
                mode = candidate
        soft_limit = _positive_int_config(
            cfg_get(cfg, "skills", "skill_md_soft_limit_chars"),
            DEFAULT_SKILL_MD_SOFT_LIMIT_CHARS,
        )
        growth_limit = _positive_int_config(
            cfg_get(cfg, "skills", "skill_md_max_growth_chars"),
            DEFAULT_SKILL_MD_MAX_GROWTH_CHARS,
        )

        overrides = cfg_get(cfg, "skills", "skill_md_size_overrides")
        override = overrides.get(skill_name) if isinstance(overrides, dict) else None
        if isinstance(override, dict):
            policy_source = f"override:{skill_name}"
            raw_override_mode = override.get("mode")
            if raw_override_mode is not None:
                candidate = str(raw_override_mode).strip().lower()
                if candidate in _SKILL_MD_SIZE_GUARD_MODES:
                    mode = candidate
            soft_limit = _positive_int_config(
                override.get("soft_limit_chars"), soft_limit
            )
            growth_limit = _positive_int_config(
                override.get("max_growth_chars"), growth_limit
            )
    except Exception:
        # The hard 100k validation still applies. Config-read failures must not
        # make all skill writes unavailable, so fall back to the safe default.
        pass
    return mode, soft_limit, growth_limit, policy_source


def _background_review_origin() -> bool:
    try:
        from tools.skill_provenance import is_background_review

        return bool(is_background_review())
    except Exception:
        return False


def evaluate_skill_md_size_guard(
    skill_name: str,
    original_content: Optional[str],
    new_content: str,
) -> Optional[Dict[str, Any]]:
    """Return size-guard details when a SKILL.md write deserves action.

    Creation is judged against the soft total-size limit. Existing skills are
    also ratcheted on single-write growth. Shrinks are always allowed, even
    when a legacy SKILL.md remains above the soft limit.
    """
    configured_mode, soft_limit, growth_limit, policy_source = (
        _skill_md_size_guard_settings(skill_name)
    )
    if configured_mode == "off":
        return None

    before_chars = len(original_content) if original_content is not None else 0
    after_chars = len(new_content)
    delta_chars = after_chars - before_chars
    reasons: List[str] = []

    if after_chars > soft_limit and (original_content is None or delta_chars > 0):
        reasons.append(
            f"total size {after_chars:,} exceeds the {soft_limit:,}-character soft limit"
        )
    if original_content is not None and delta_chars > growth_limit:
        reasons.append(
            f"single-write growth +{delta_chars:,} exceeds the {growth_limit:,}-character limit"
        )
    if not reasons:
        return None

    if configured_mode == "auto":
        effective_mode = "enforce" if _background_review_origin() else "warn"
    else:
        effective_mode = configured_mode

    reason_text = "; ".join(reasons)
    message = (
        f"{reason_text}. Keep SKILL.md as a lean trigger/authority/router file; "
        "move session-specific detail, long examples, endpoint quirks, and "
        "incident notes into references/ (or templates/scripts) and leave a "
        "concise pointer in SKILL.md. Shrinking edits remain allowed."
    )
    return {
        "skill": skill_name,
        "policy_source": policy_source,
        "configured_mode": configured_mode,
        "effective_mode": effective_mode,
        "before_chars": before_chars,
        "after_chars": after_chars,
        "delta_chars": delta_chars,
        "soft_limit_chars": soft_limit,
        "max_growth_chars": growth_limit,
        "reasons": reasons,
        "message": message,
    }


def blocked_by_skill_md_size_guard(details: Dict[str, Any]) -> Dict[str, Any]:
    """Build the standard fail-closed tool result for an enforced policy."""
    return {
        "success": False,
        "error": f"SKILL.md size guard blocked this write: {details['message']}",
        "size_guard": details,
        "_fail_closed": True,
    }


def attach_skill_md_size_advisory(
    result: Dict[str, Any],
    details: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Attach a structured advisory to a successful warn-mode tool result."""
    if details and details.get("effective_mode") == "warn":
        result["size_advisory"] = details
    return result

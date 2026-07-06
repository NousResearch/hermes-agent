"""Delegation tier labels.

Derive a coarse cost/capability *tier* for a delegation profile from its
resolved model + reasoning effort. This is telemetry metadata only: it is
stamped onto the child session's ``model_config`` JSON at spawn time so that
after the fact you can tell WHICH lane a delegation ran on (many profiles share
one model, e.g. several profiles all use ``gpt-5.5``) and audit whether the
tiering policy is being respected.

Heuristic contract (NOT a hard guarantee):

- ``heavy``   -> opus family (any effort) OR gpt-5.5 at effort high/xhigh
- ``mid``     -> gpt-5.5 at low/medium, OR a sonnet lane at medium+ effort
- ``light``   -> cheap models (mini / haiku / sonnet-at-low). A mini model is
                 NEVER heavy regardless of effort.
- ``unprofiled`` -> no profile (bare delegate_task that inherited the parent
                    model)

Cost-class of the MODEL dominates; reasoning effort only breaks ties within a
class. This ordering matters: a cheap ``gpt-5.4-mini`` researcher lane running
at ``high`` effort must NOT be labelled heavy, or an over-provisioning audit
would false-positive on every cheap research run.
"""

from __future__ import annotations

from typing import Optional

_HEAVY_EFFORTS = {"high", "xhigh", "x-high", "extra-high"}


def _norm(s: Optional[str]) -> str:
    return str(s or "").strip().lower()


def _is_opus(model: str) -> bool:
    return "opus" in model


def _is_cheap(model: str) -> bool:
    # Cheap cost-class: mini variants, haiku, and (for tie-breaking) sonnet.
    return "mini" in model or "haiku" in model


def _is_sonnet(model: str) -> bool:
    return "sonnet" in model


def _is_gpt55(model: str) -> bool:
    # gpt-5.5 family (the heavy codex lane). Excludes gpt-5.4-mini / gpt-*-mini.
    return ("gpt-5.5" in model or "gpt5.5" in model) and "mini" not in model


def derive_tier_from_model(model: Optional[str], effort: Optional[str]) -> str:
    """Pure tier derivation from a model id + reasoning effort string."""
    m = _norm(model)
    e = _norm(effort)
    if not m:
        return "unprofiled"

    # 1. Cheap cost-class first: mini / haiku are never heavy.
    if _is_cheap(m):
        return "light"

    # 2. Opus is always heavy (top cost-class), any effort.
    if _is_opus(m):
        return "heavy"

    # 3. gpt-5.5 heavy codex lane: high/xhigh -> heavy, else mid.
    if _is_gpt55(m):
        return "heavy" if e in _HEAVY_EFFORTS else "mid"

    # 4. Sonnet mid-cost: medium+ effort -> mid, low -> light.
    if _is_sonnet(m):
        return "light" if e == "low" else "mid"

    # 5. Unknown model at heavy effort still signals a heavy lane; else mid.
    if e in _HEAVY_EFFORTS:
        return "heavy"
    return "mid"


def derive_tier(
    profile: Optional[str],
    cfg: Optional[dict] = None,
) -> str:
    """Derive the tier for a delegation.

    ``profile`` is the resolved profile name (or None/'' for a bare delegation).
    ``cfg`` is the MERGED delegation config for this child, which already
    carries the resolved ``model`` and ``reasoning_effort`` (see
    ``_merge_delegation_profile``). When ``cfg`` is omitted, the profile is
    looked up from the live ``config.yaml`` ``delegation.profiles`` map so the
    function is usable standalone (audit/tests) with just a profile name.
    """
    name = str(profile or "").strip()
    if not name:
        # Bare delegation: no profile means it silently inherited the parent
        # model. That is exactly the 'unprofiled' class the audit flags — do
        # NOT try to tier it by the inherited model.
        return "unprofiled"

    model = None
    effort = None
    if cfg is not None:
        model = cfg.get("model")
        effort = cfg.get("reasoning_effort")

    if not model:
        prof = _lookup_profile(name)
        if prof is None:
            return "unprofiled"
        model = prof.get("model")
        effort = prof.get("reasoning_effort")

    return derive_tier_from_model(model, effort)


def _lookup_profile(name: str) -> Optional[dict]:
    """Best-effort read of delegation.profiles.<name> from live config."""
    cfg = None
    try:
        from cli import CLI_CONFIG  # type: ignore
        if CLI_CONFIG:
            cfg = CLI_CONFIG
    except Exception:
        cfg = None
    if cfg is None:
        try:
            from hermes_cli.config import load_config  # type: ignore
            cfg = load_config() or {}
        except Exception:
            return None
    try:
        profiles = ((cfg.get("delegation") or {}).get("profiles") or {})
        prof = profiles.get(name)
        return prof if isinstance(prof, dict) else None
    except Exception:
        return None

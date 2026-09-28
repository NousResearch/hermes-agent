"""Blueprints: shareable plain-language automations layered on skills + cron.

A blueprint is NOT a new object type: it is an ordinary skill whose frontmatter declares
``metadata.hermes.blueprint`` (``schedule`` required; optional ``deliver`` [default "origin"],
``prompt``, ``no_agent``, ``model``, ``provider``, ``enabled_toolsets``), so it rides the whole
skills-hub pipeline for free. This module only bridges that block to cron ``create_job()``,
plus the inverse (``export_blueprint``) back to a SKILL.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["BlueprintSpec", "parse_blueprint", "blueprint_spec_for_installed", "blueprint_to_job_spec",
           "create_blueprint_job", "register_blueprint_suggestion", "export_blueprint", "BlueprintError"]


class BlueprintError(ValueError):
    """Raised when a blueprint block is present but malformed."""


@dataclass
class BlueprintSpec:
    """Parsed ``metadata.hermes.blueprint`` automation spec for a skill."""

    skill_name: str
    schedule: str
    deliver: str = "origin"
    prompt: Optional[str] = None
    no_agent: bool = False
    model: Optional[str] = None
    provider: Optional[str] = None
    enabled_toolsets: Optional[List[str]] = None
    raw: Dict[str, Any] = field(default_factory=dict)


def _split_frontmatter(text: str) -> Optional[Dict[str, Any]]:
    """Return the parsed YAML frontmatter mapping, or None if absent/invalid."""
    if not isinstance(text, str):
        return None
    stripped = text.lstrip("\ufeff").lstrip()  # BOM is not whitespace; strip explicitly
    if not stripped.startswith("---") or (end := stripped.find("\n---", 3)) == -1:
        return None
    try:
        import yaml

        data = yaml.safe_load(stripped[3:end])
    except Exception as e:  # pragma: no cover - malformed YAML
        logger.debug("blueprint: frontmatter YAML parse failed: %s", e)
        return None
    return data if isinstance(data, dict) else None


def parse_blueprint(skill_md_text: str) -> Optional[BlueprintSpec]:
    """Extract a BlueprintSpec from a SKILL.md string, or None if not a blueprint.

    A skill is a blueprint iff ``metadata.hermes.blueprint`` is a mapping with a
    non-empty ``schedule``. Raises BlueprintError if the block exists but is
    structurally invalid (so a typo surfaces instead of silently no-op'ing).
    """
    fm = _split_frontmatter(skill_md_text)
    if not fm:
        return None

    meta = fm.get("metadata")
    hermes = meta.get("hermes") if isinstance(meta, dict) else None
    blueprint = hermes.get("blueprint") if isinstance(hermes, dict) else None
    if blueprint is None:
        return None
    if not isinstance(blueprint, dict):
        raise BlueprintError("metadata.hermes.blueprint must be a mapping")

    schedule = str(blueprint.get("schedule", "")).strip()
    if not schedule:
        raise BlueprintError("blueprint.schedule is required and must be non-empty")

    prompt, model, provider = blueprint.get("prompt"), blueprint.get("model"), blueprint.get("provider")
    toolsets = blueprint.get("enabled_toolsets")
    if toolsets is not None and not isinstance(toolsets, list):
        raise BlueprintError("blueprint.enabled_toolsets must be a list when present")

    return BlueprintSpec(
        skill_name=str(fm.get("name", "")).strip(), schedule=schedule,
        deliver=str(blueprint.get("deliver", "origin")).strip() or "origin",
        prompt=str(prompt) if prompt is not None else None,
        no_agent=bool(blueprint.get("no_agent", False)),
        model=str(model).strip() if model else None,
        provider=str(provider).strip() if provider else None,
        enabled_toolsets=[str(t) for t in toolsets] if toolsets else None,
        raw=blueprint,
    )


def blueprint_spec_for_installed(skill_name: str) -> Optional[BlueprintSpec]:
    """Find ``<skill_name>/SKILL.md`` anywhere in the skills tree and parse its
    blueprint block; None when not found or not a blueprint."""
    try:
        from tools.skills_hub import SKILLS_DIR
    except Exception:  # pragma: no cover - import guard
        return None
    # Skills live at skills/<category>/<name>/SKILL.md or skills/<name>/SKILL.md.
    for path in Path(SKILLS_DIR).glob(f"**/{skill_name}/SKILL.md"):
        try:
            spec = parse_blueprint(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        if spec is not None:
            spec.skill_name = spec.skill_name or skill_name  # frontmatter name wins over dir name
            return spec
    return None


def blueprint_to_job_spec(spec: BlueprintSpec, *, name: Optional[str] = None) -> Dict[str, Any]:
    """``cron.jobs.create_job`` kwargs for a spec — the single translation used by
    both ``create_blueprint_job`` and the suggestion path so they never drift."""
    return {
        "prompt": spec.prompt, "schedule": spec.schedule, "name": name or f"blueprint:{spec.skill_name}",
        "deliver": spec.deliver, "skills": [spec.skill_name] if spec.skill_name else None,
        "model": spec.model, "provider": spec.provider, "enabled_toolsets": spec.enabled_toolsets,
        "no_agent": spec.no_agent,
    }


def create_blueprint_job(spec: BlueprintSpec, *, origin: Optional[Dict[str, Any]] = None,
                         name: Optional[str] = None) -> Dict[str, Any]:
    """Create the cron job for a spec (skill preloaded via ``skills=[name]``); returns the job dict."""
    from cron.scheduler import create_job_with_scheduler_registration

    job_spec = blueprint_to_job_spec(spec, name=name)
    if origin is not None:
        job_spec["origin"] = origin
    return create_job_with_scheduler_registration(**job_spec)


def register_blueprint_suggestion(spec: BlueprintSpec) -> Optional[Dict[str, Any]]:
    """Register an installed blueprint as a Suggested Cron Job (never auto-scheduled;
    the user accepts or dismisses it). None when skipped (seen/dismissed/backlog full)."""
    if not spec.skill_name:
        return None
    try:
        from cron.suggestions import add_suggestion
    except Exception:  # pragma: no cover - import guard
        return None

    deliver = f", delivering to {spec.deliver}" if spec.deliver and spec.deliver != "origin" else ""
    return add_suggestion(
        title=f"Schedule '{spec.skill_name}'",
        description=f"The '{spec.skill_name}' blueprint runs on schedule {spec.schedule}{deliver}.",
        source="blueprint",
        job_spec=blueprint_to_job_spec(spec),
        dedup_key=f"blueprint:{spec.skill_name}:{spec.schedule}",
    )


def export_blueprint(job: Dict[str, Any], body: str, *, blueprint_name: Optional[str] = None) -> str:
    """Inverse of ``create_blueprint_job``: render a cron job as a SKILL.md (with a
    ``metadata.hermes.blueprint`` block) ready for ``hermes skills publish``.
    ``body`` becomes the SKILL.md body; its first line is the description."""
    import yaml

    # Sanitize to a valid skill identifier.
    name = str(blueprint_name or job.get("name") or "shared-blueprint").lower()
    name = "".join(c if (c.isalnum() or c in "-_") else "-" for c in name).strip("-_") or "shared-blueprint"

    block: Dict[str, Any] = {"schedule": job.get("schedule_display") or _schedule_to_string(job.get("schedule"))}
    if job.get("deliver") and job["deliver"] != "origin":
        block["deliver"] = job["deliver"]
    if job.get("prompt"):
        block["prompt"] = job["prompt"]
    if job.get("no_agent"):
        block["no_agent"] = True
    block.update({k: job[k] for k in ("model", "provider", "enabled_toolsets") if job.get(k)})

    body = body.strip()
    frontmatter = {
        "name": name, "description": body.splitlines()[0][:200] if body else "Shared automation blueprint.",
        "version": "1.0.0", "license": "MIT",
        "metadata": {"hermes": {"tags": ["blueprint", "automation"], "blueprint": block}},
    }
    fm_yaml = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
    body_text = body or f"# {name}\n\nShared automation blueprint."
    return f"---\n{fm_yaml}\n---\n\n{body_text}\n"


def _schedule_to_string(schedule: Any) -> str:
    """Best-effort render of a parsed schedule dict back to a string."""
    if isinstance(schedule, str):
        return schedule
    if isinstance(schedule, dict):
        kind = schedule.get("kind")
        if kind == "cron" and schedule.get("expr"):
            return str(schedule["expr"])
        if kind == "interval":
            # parse_schedule stores interval periods as "minutes"; tolerate a legacy/foreign "seconds" form too.
            if schedule.get("minutes"):
                mins = int(schedule["minutes"])
                return f"every {mins // 60}h" if mins % 60 == 0 else f"every {mins}m"
            if schedule.get("seconds"):
                secs = int(schedule["seconds"])
                return (f"every {secs // 3600}h" if secs % 3600 == 0
                        else f"every {secs // 60}m" if secs % 60 == 0 else f"every {secs}s")
    return "0 9 * * *"  # safe daily fallback

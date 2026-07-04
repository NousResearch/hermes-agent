"""Blueprints: shareable plain-language automations layered on skills + cron.

A "blueprint" is NOT a new object type. It is an ordinary skill (a SKILL.md the
agent loads) that additionally declares an automation schedule in its
frontmatter:

    metadata:
      hermes:
        blueprint:
          schedule: "0 9 * * *"     # presence of `blueprint:` marks it runnable
          deliver: origin            # optional (default "origin")
          prompt: "..."              # optional task instruction for the run
          no_agent: false            # optional

Because a blueprint is just a skill, it flows through the ENTIRE existing
skills-hub pipeline for free — search, inspect, quarantine, security scan,
install, lock-file provenance, audit log, taps, the centralized index, and
`hermes skills publish` for sharing. No new source type, no new store, no new
transport. This module is the thin bridge between that skill metadata and the
existing cron `create_job()` API:

  * ``parse_blueprint(skill_md_text)``  -> BlueprintSpec | None
  * ``blueprint_spec_for_installed(name)`` -> BlueprintSpec | None
  * ``create_blueprint_job(spec, ...)`` -> the created cron job dict
  * ``export_blueprint(job, body)``      -> a shareable SKILL.md string

The dev guide's "Extend, Don't Duplicate" rule is the whole design: the blueprint
is a skill, the schedule is a cron job, sharing is the existing publish/tap/
index path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "BlueprintSpec",
    "parse_blueprint",
    "blueprint_spec_for_installed",
    "blueprint_to_job_spec",
    "create_blueprint_job",
    "register_blueprint_suggestion",
    "export_blueprint",
    "BlueprintError",
]


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
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return None
    # Find the closing fence after the opening one.
    after_open = stripped[3:]
    end = after_open.find("\n---")
    if end == -1:
        return None
    fm_text = after_open[:end]
    try:
        import yaml

        data = yaml.safe_load(fm_text)
    except Exception as e:  # pragma: no cover - malformed YAML
        logger.debug("blueprint: frontmatter YAML parse failed: %s", e)
        return None
    return data if isinstance(data, dict) else None


def parse_blueprint(skill_md_text: str) -> Optional[BlueprintSpec]:
    """Extract a BlueprintSpec from a SKILL.md string, or None if not a blueprint.

    A skill is a blueprint iff ``metadata.hermes.blueprint`` is a mapping containing
    a non-empty ``schedule``. Raises BlueprintError if the block exists but is
    structurally invalid (so a typo surfaces instead of silently no-op'ing).
    """
    fm = _split_frontmatter(skill_md_text)
    if not fm:
        return None

    name = str(fm.get("name", "")).strip()

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

    deliver = str(blueprint.get("deliver", "origin")).strip() or "origin"
    prompt = blueprint.get("prompt")
    if prompt is not None:
        prompt = str(prompt)
    no_agent = bool(blueprint.get("no_agent", False))
    model = blueprint.get("model")
    provider = blueprint.get("provider")
    toolsets = blueprint.get("enabled_toolsets")
    if toolsets is not None and not isinstance(toolsets, list):
        raise BlueprintError("blueprint.enabled_toolsets must be a list when present")

    return BlueprintSpec(
        skill_name=name,
        schedule=schedule,
        deliver=deliver,
        prompt=prompt,
        no_agent=no_agent,
        model=str(model).strip() if model else None,
        provider=str(provider).strip() if provider else None,
        enabled_toolsets=[str(t) for t in toolsets] if toolsets else None,
        raw=blueprint,
    )


def blueprint_spec_for_installed(skill_name: str) -> Optional[BlueprintSpec]:
    """Locate an installed skill's SKILL.md and parse its blueprint block.

    Searches the standard skills tree for ``<skill_name>/SKILL.md``. Returns
    None if the skill isn't found or isn't a blueprint.
    """
    try:
        from tools.skills_hub import SKILLS_DIR
    except Exception:  # pragma: no cover - import guard
        return None

    base = Path(SKILLS_DIR)
    # Skills live at skills/<category>/<name>/SKILL.md or skills/<name>/SKILL.md.
    candidates = list(base.glob(f"**/{skill_name}/SKILL.md"))
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        spec = parse_blueprint(text)
        if spec is not None:
            # Prefer the frontmatter name, fall back to the directory name.
            if not spec.skill_name:
                spec.skill_name = skill_name
            return spec
    return None


def blueprint_to_job_spec(
    spec: BlueprintSpec,
    *,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the ``cron.jobs.create_job`` kwargs dict for a BlueprintSpec.

    This is the single source of truth for translating a blueprint into a job.
    Both the direct ``create_blueprint_job`` path and the suggestion path
    (``register_blueprint_suggestion``) build on it, so a blueprint scheduled now and
    a blueprint accepted from a suggestion produce an identical job.
    """
    return {
        "prompt": spec.prompt,
        "schedule": spec.schedule,
        "name": name or f"blueprint:{spec.skill_name}",
        "deliver": spec.deliver,
        "skills": [spec.skill_name] if spec.skill_name else None,
        "model": spec.model,
        "provider": spec.provider,
        "enabled_toolsets": spec.enabled_toolsets,
        "no_agent": spec.no_agent,
    }


def create_blueprint_job(
    spec: BlueprintSpec,
    *,
    origin: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Create the cron job described by a BlueprintSpec via the existing cron API.

    The blueprint's skill is loaded before the run (cron ``skills=[name]``); the
    optional ``prompt`` becomes the task instruction. Delivery, model, and
    toolsets carry through. Returns the created job dict.
    """
    from cron.jobs import create_job

    job_spec = blueprint_to_job_spec(spec, name=name)
    if origin is not None:
        job_spec["origin"] = origin
    return create_job(**job_spec)


def register_blueprint_suggestion(spec: BlueprintSpec) -> Optional[Dict[str, Any]]:
    """Turn an installed blueprint into a pending Suggested Cron Job.

    Blueprints are source ``blueprint`` of the unified suggestion surface: installing
    a skill that carries a ``blueprint:`` block does NOT auto-schedule it — it
    registers a suggestion the user accepts (or dismisses) like any other.
    Returns the suggestion record, or None if it was skipped (already
    seen/dismissed, backlog full, etc.).
    """
    if not spec.skill_name:
        return None
    try:
        from cron.suggestions import add_suggestion
    except Exception:  # pragma: no cover - import guard
        return None

    return add_suggestion(
        title=f"Schedule '{spec.skill_name}'",
        description=(
            f"The '{spec.skill_name}' blueprint runs on schedule {spec.schedule}"
            + (f", delivering to {spec.deliver}" if spec.deliver and spec.deliver != "origin" else "")
            + "."
        ),
        source="blueprint",
        job_spec=blueprint_to_job_spec(spec),
        dedup_key=f"blueprint:{spec.skill_name}:{spec.schedule}",
    )


def export_blueprint(job: Dict[str, Any], body: str, *, blueprint_name: Optional[str] = None) -> str:
    """Render a shareable blueprint SKILL.md from an existing cron job dict.

    The inverse of ``create_blueprint_job``: take a cron job a user already built
    and emit a SKILL.md (with a ``metadata.hermes.blueprint`` block) they can hand
    to ``hermes skills publish`` to share. ``body`` is the plain-language
    description / instructions that become the SKILL.md body.
    """
    import yaml

    name = blueprint_name or job.get("name") or "shared-blueprint"
    # Sanitize to a valid skill identifier.
    name = "".join(c if (c.isalnum() or c in "-_") else "-" for c in str(name).lower())
    name = name.strip("-_") or "shared-blueprint"

    schedule = job.get("schedule_display") or _schedule_to_string(job.get("schedule"))
    skills = job.get("skills") or ([job["skill"]] if job.get("skill") else [])

    blueprint_block: Dict[str, Any] = {"schedule": schedule}
    deliver = job.get("deliver")
    if deliver and deliver != "origin":
        blueprint_block["deliver"] = deliver
    if job.get("prompt"):
        blueprint_block["prompt"] = job["prompt"]
    if job.get("no_agent"):
        blueprint_block["no_agent"] = True
    if job.get("model"):
        blueprint_block["model"] = job["model"]
    if job.get("provider"):
        blueprint_block["provider"] = job["provider"]
    if job.get("enabled_toolsets"):
        blueprint_block["enabled_toolsets"] = job["enabled_toolsets"]

    description = (
        (body.strip().splitlines() or ["Shared automation blueprint."])[0][:200]
        if body.strip()
        else "Shared automation blueprint."
    )

    frontmatter = {
        "name": name,
        "description": description,
        "version": "1.0.0",
        "license": "MIT",
        "metadata": {
            "hermes": {
                "tags": ["blueprint", "automation"],
                "blueprint": blueprint_block,
            }
        },
    }
    fm_yaml = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
    body_text = body.strip() or f"# {name}\n\nShared automation blueprint."
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
            # parse_schedule stores interval periods as "minutes"; tolerate a
            # legacy/foreign "seconds" form too.
            if schedule.get("minutes"):
                mins = int(schedule["minutes"])
                if mins % 60 == 0:
                    return f"every {mins // 60}h"
                return f"every {mins}m"
            if schedule.get("seconds"):
                secs = int(schedule["seconds"])
                if secs % 3600 == 0:
                    return f"every {secs // 3600}h"
                if secs % 60 == 0:
                    return f"every {secs // 60}m"
                return f"every {secs}s"
    return "0 9 * * *"  # safe daily fallback

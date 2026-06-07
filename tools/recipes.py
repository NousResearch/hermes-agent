"""Recipes: shareable plain-language automations layered on skills + cron.

A "recipe" is NOT a new object type. It is an ordinary skill (a SKILL.md the
agent loads) that additionally declares an automation schedule in its
frontmatter:

    metadata:
      hermes:
        recipe:
          schedule: "0 9 * * *"     # presence of `recipe:` marks it runnable
          deliver: origin            # optional (default "origin")
          prompt: "..."              # optional task instruction for the run
          no_agent: false            # optional

Because a recipe is just a skill, it flows through the ENTIRE existing
skills-hub pipeline for free — search, inspect, quarantine, security scan,
install, lock-file provenance, audit log, taps, the centralized index, and
`hermes skills publish` for sharing. No new source type, no new store, no new
transport. This module is the thin bridge between that skill metadata and the
existing cron `create_job()` API:

  * ``parse_recipe(skill_md_text)``  -> RecipeSpec | None
  * ``recipe_spec_for_installed(name)`` -> RecipeSpec | None
  * ``create_recipe_job(spec, ...)`` -> the created cron job dict
  * ``export_recipe(job, body)``      -> a shareable SKILL.md string

The dev guide's "Extend, Don't Duplicate" rule is the whole design: the recipe
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
    "RecipeSpec",
    "parse_recipe",
    "recipe_spec_for_installed",
    "recipe_to_job_spec",
    "create_recipe_job",
    "register_recipe_suggestion",
    "export_recipe",
    "RecipeError",
]


class RecipeError(ValueError):
    """Raised when a recipe block is present but malformed."""


@dataclass
class RecipeSpec:
    """Parsed ``metadata.hermes.recipe`` automation spec for a skill."""

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
        logger.debug("recipe: frontmatter YAML parse failed: %s", e)
        return None
    return data if isinstance(data, dict) else None


def parse_recipe(skill_md_text: str) -> Optional[RecipeSpec]:
    """Extract a RecipeSpec from a SKILL.md string, or None if not a recipe.

    A skill is a recipe iff ``metadata.hermes.recipe`` is a mapping containing
    a non-empty ``schedule``. Raises RecipeError if the block exists but is
    structurally invalid (so a typo surfaces instead of silently no-op'ing).
    """
    fm = _split_frontmatter(skill_md_text)
    if not fm:
        return None

    name = str(fm.get("name", "")).strip()

    meta = fm.get("metadata")
    hermes = meta.get("hermes") if isinstance(meta, dict) else None
    recipe = hermes.get("recipe") if isinstance(hermes, dict) else None
    if recipe is None:
        return None
    if not isinstance(recipe, dict):
        raise RecipeError("metadata.hermes.recipe must be a mapping")

    schedule = str(recipe.get("schedule", "")).strip()
    if not schedule:
        raise RecipeError("recipe.schedule is required and must be non-empty")

    deliver = str(recipe.get("deliver", "origin")).strip() or "origin"
    prompt = recipe.get("prompt")
    if prompt is not None:
        prompt = str(prompt)
    no_agent = bool(recipe.get("no_agent", False))
    model = recipe.get("model")
    provider = recipe.get("provider")
    toolsets = recipe.get("enabled_toolsets")
    if toolsets is not None and not isinstance(toolsets, list):
        raise RecipeError("recipe.enabled_toolsets must be a list when present")

    return RecipeSpec(
        skill_name=name,
        schedule=schedule,
        deliver=deliver,
        prompt=prompt,
        no_agent=no_agent,
        model=str(model).strip() if model else None,
        provider=str(provider).strip() if provider else None,
        enabled_toolsets=[str(t) for t in toolsets] if toolsets else None,
        raw=recipe,
    )


def recipe_spec_for_installed(skill_name: str) -> Optional[RecipeSpec]:
    """Locate an installed skill's SKILL.md and parse its recipe block.

    Searches the standard skills tree for ``<skill_name>/SKILL.md``. Returns
    None if the skill isn't found or isn't a recipe.
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
        spec = parse_recipe(text)
        if spec is not None:
            # Prefer the frontmatter name, fall back to the directory name.
            if not spec.skill_name:
                spec.skill_name = skill_name
            return spec
    return None


def recipe_to_job_spec(
    spec: RecipeSpec,
    *,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the ``cron.jobs.create_job`` kwargs dict for a RecipeSpec.

    This is the single source of truth for translating a recipe into a job.
    Both the direct ``create_recipe_job`` path and the suggestion path
    (``register_recipe_suggestion``) build on it, so a recipe scheduled now and
    a recipe accepted from a suggestion produce an identical job.
    """
    return {
        "prompt": spec.prompt,
        "schedule": spec.schedule,
        "name": name or f"recipe:{spec.skill_name}",
        "deliver": spec.deliver,
        "skills": [spec.skill_name] if spec.skill_name else None,
        "model": spec.model,
        "provider": spec.provider,
        "enabled_toolsets": spec.enabled_toolsets,
        "no_agent": spec.no_agent,
    }


def create_recipe_job(
    spec: RecipeSpec,
    *,
    origin: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Create the cron job described by a RecipeSpec via the existing cron API.

    The recipe's skill is loaded before the run (cron ``skills=[name]``); the
    optional ``prompt`` becomes the task instruction. Delivery, model, and
    toolsets carry through. Returns the created job dict.
    """
    from cron.jobs import create_job

    job_spec = recipe_to_job_spec(spec, name=name)
    if origin is not None:
        job_spec["origin"] = origin
    return create_job(**job_spec)


def register_recipe_suggestion(spec: RecipeSpec) -> Optional[Dict[str, Any]]:
    """Turn an installed recipe into a pending Suggested Cron Job.

    Recipes are source ``recipe`` of the unified suggestion surface: installing
    a skill that carries a ``recipe:`` block does NOT auto-schedule it — it
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
            f"The '{spec.skill_name}' recipe runs on schedule {spec.schedule}"
            + (f", delivering to {spec.deliver}" if spec.deliver and spec.deliver != "origin" else "")
            + "."
        ),
        source="recipe",
        job_spec=recipe_to_job_spec(spec),
        dedup_key=f"recipe:{spec.skill_name}:{spec.schedule}",
    )


def export_recipe(job: Dict[str, Any], body: str, *, recipe_name: Optional[str] = None) -> str:
    """Render a shareable recipe SKILL.md from an existing cron job dict.

    The inverse of ``create_recipe_job``: take a cron job a user already built
    and emit a SKILL.md (with a ``metadata.hermes.recipe`` block) they can hand
    to ``hermes skills publish`` to share. ``body`` is the plain-language
    description / instructions that become the SKILL.md body.
    """
    import yaml

    name = recipe_name or job.get("name") or "shared-recipe"
    # Sanitize to a valid skill identifier.
    name = "".join(c if (c.isalnum() or c in "-_") else "-" for c in str(name).lower())
    name = name.strip("-_") or "shared-recipe"

    schedule = job.get("schedule_display") or _schedule_to_string(job.get("schedule"))
    skills = job.get("skills") or ([job["skill"]] if job.get("skill") else [])

    recipe_block: Dict[str, Any] = {"schedule": schedule}
    deliver = job.get("deliver")
    if deliver and deliver != "origin":
        recipe_block["deliver"] = deliver
    if job.get("prompt"):
        recipe_block["prompt"] = job["prompt"]
    if job.get("no_agent"):
        recipe_block["no_agent"] = True
    if job.get("model"):
        recipe_block["model"] = job["model"]
    if job.get("provider"):
        recipe_block["provider"] = job["provider"]
    if job.get("enabled_toolsets"):
        recipe_block["enabled_toolsets"] = job["enabled_toolsets"]

    description = (
        (body.strip().splitlines() or ["Shared automation recipe."])[0][:200]
        if body.strip()
        else "Shared automation recipe."
    )

    frontmatter = {
        "name": name,
        "description": description,
        "version": "1.0.0",
        "license": "MIT",
        "metadata": {
            "hermes": {
                "tags": ["recipe", "automation"],
                "recipe": recipe_block,
            }
        },
    }
    fm_yaml = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
    body_text = body.strip() or f"# {name}\n\nShared automation recipe."
    return f"---\n{fm_yaml}\n---\n\n{body_text}\n"


def _schedule_to_string(schedule: Any) -> str:
    """Best-effort render of a parsed schedule dict back to a string."""
    if isinstance(schedule, str):
        return schedule
    if isinstance(schedule, dict):
        kind = schedule.get("kind")
        if kind == "cron" and schedule.get("expr"):
            return str(schedule["expr"])
        if kind == "interval" and schedule.get("seconds"):
            secs = int(schedule["seconds"])
            if secs % 3600 == 0:
                return f"every {secs // 3600}h"
            if secs % 60 == 0:
                return f"every {secs // 60}m"
            return f"every {secs}s"
    return "0 9 * * *"  # safe daily fallback

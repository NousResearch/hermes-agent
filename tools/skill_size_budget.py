"""Profile-scoped growth budgets for the main SKILL.md written by skill_manage."""

from pathlib import Path

from hermes_cli.config import cfg_get, load_config_readonly
from tools.skill_manager_guards import _refusal


def _positive_budget(key: str) -> int | None:
    value = cfg_get(load_config_readonly(), "skills", key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _is_main_skill_md(skill_dir: Path, target: Path) -> bool:
    return target.resolve() == (skill_dir / "SKILL.md").resolve()


def skill_write_budget_refusal(name: str, skill_dir: Path, target: Path,
                              content: str, original: str | None) -> dict | None:
    """Bound growth, while letting an already-over-budget skill be cleaned up."""
    if not _is_main_skill_md(skill_dir, target) or (cap := _positive_budget("max_skill_md_chars")) is None:
        return None
    current_chars = len(original or "")
    delta = len(content) - current_chars
    if len(content) <= cap or delta <= 0:
        return None
    return _refusal(
        f"Refusing to grow SKILL.md in skill '{name}' to {len(content):,} chars "
        f"(skills.max_skill_md_chars: {cap:,}). Move detail to references/ and keep "
        "SKILL.md concise; shrinking or equal-length maintenance remains allowed.",
        current_chars=current_chars, requested_delta=delta, cap=cap)


def attach_skill_size_warning(result: dict, name: str, skill_dir: Path,
                              target: Path, content: str) -> dict:
    """Annotate only a successful main-file write, using the active profile's threshold."""
    if _is_main_skill_md(skill_dir, target):
        threshold = _positive_budget("size_warn_chars")
        if threshold is not None and len(content) >= threshold:
            result["size_warning"] = {"name": name, "chars": len(content), "threshold": threshold}
    return result

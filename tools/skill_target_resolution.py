"""Exact target identity for opt-in skill creation approval; legacy lookup stays unchanged."""

from pathlib import Path


class AmbiguousSkillTarget(LookupError):
    def __init__(self, name, paths):
        self.candidates = sorted({str(Path(p).absolute()) for p in paths})
        super().__init__(f"Skill target {name!r} is ambiguous. Retry with one exact skill directory path in name: "
                         + "; ".join(self.candidates))


def unique_targets_enabled():
    from tools import write_approval as wa
    return wa.write_approval_enabled(wa.SKILLS) and wa.skill_write_approval_mode() == "create"


def choose_unique(name, paths, resolve):
    distinct = {}
    for path in paths:
        distinct.setdefault(resolve(path), path)
    if len(distinct) > 1:
        raise AmbiguousSkillTarget(name, distinct.values())
    return {"path": next(iter(distinct.values()))} if distinct else None


def explicit_targets(name, roots, *, exists, lexists):
    """Resolve only declared catalogs; supporting/excluded directories remain non-targets."""
    from agent.skill_utils import is_excluded_skill_path
    query = Path(name)
    if ".." in query.parts:
        return []
    matches = []
    for root in roots:
        root = root.absolute()
        path = query if query.is_absolute() else root / query
        if not path.is_relative_to(root) or not exists(path):
            continue
        manifest = path / "SKILL.md"
        if lexists(manifest) and not is_excluded_skill_path(manifest, root=root, exists=exists):
            matches.append(path)
    return matches


def metadata_names(name: str, skill_dir: Path | None = None) -> tuple[str, ...]:
    """Guard both directory identity and the displayed name used by curator receipts."""
    if not unique_targets_enabled():
        return (name,)
    from tools.skill_manager_tool import _find_skill, _read_frontmatter_name
    if skill_dir is None:
        hit = _find_skill(name)
        skill_dir = hit["path"] if hit else None
    if skill_dir is None:
        return (name,)
    directory = Path(skill_dir).resolve()
    display = _read_frontmatter_name(directory / "SKILL.md")
    return tuple(dict.fromkeys([directory.name, *([display] if display else [])]))


def metadata_name(name: str, skill_dir: Path | None = None) -> str:
    """Keep existing provenance keys rather than recording absolute path identities."""
    names = metadata_names(name, skill_dir)
    if len(names) > 1:
        from tools.skill_usage import load_usage
        records = load_usage()
        return next((key for key in names if key in records), names[0])
    return names[0]

"""Skills Hub "local-dir" source: skill folders already installed on disk by another tool (e.g. a
different agent CLI's skills directory), without vendoring them into the repo or hardcoding any
machine-specific path.

Directories are opt-in, added by the user via ``hermes skills local add <path>`` (persisted in
``tools.skills_hub.LocalDirsManager``, ``skills/.hub/local_dirs.json`` — per-user state, never
``config.yaml`` or the repo). Each configured directory is scanned one level deep for
``<name>/SKILL.md`` subfolders, matching the on-disk layout every agent-skill tool uses
(``~/.claude/skills``, ``~/.codex/skills``, ``~/.agents/skills``, ...).

Content here is third-party and unreviewed by Hermes, so it is always "community" trust — the
same security scan gate (``tools/skills_guard.py``) that every non-builtin install goes through
applies before anything lands in ``~/.hermes/skills/``.
"""

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

from tools.skills_hub_github import _skip_bundle_file
from tools.skills_hub_models import (
    SkillBundle, SkillMeta, SkillSource, _hermes_tags, _matches_query, _parse_frontmatter,
    _validate_skill_name,
)

logger = logging.getLogger("tools.skills_hub")


def _configured_dirs() -> List[Path]:
    from tools.skills_hub import LocalDirsManager
    dirs = []
    for raw in LocalDirsManager().list_dirs():
        path = Path(raw).expanduser()
        if path.is_dir():
            dirs.append(path)
        else:
            logger.debug("Configured local skills directory no longer exists: %s", raw)
    return dirs


class LocalFolderSource(SkillSource):
    """Browse/install skills from user-configured local directories of ``<name>/SKILL.md`` folders."""

    SOURCE_ID = "local-dir"
    TRUST_LEVEL = "community"

    def _meta(self, root: Path, skill_dir: Path) -> Optional[SkillMeta]:
        skill_md = skill_dir / "SKILL.md"
        try:
            content = skill_md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        fm = _parse_frontmatter(content)
        tags = _hermes_tags(fm)
        return SkillMeta(
            name=str(fm.get("name") or skill_dir.name), description=str(fm.get("description", ""))[:200],
            source="local-dir", identifier=f"local-dir:{root}/{skill_dir.name}", trust_level=self.TRUST_LEVEL,
            path=str(skill_dir), tags=tags if isinstance(tags, list) else [],
            extra={"root": str(root)},
        )

    def _iter_skill_dirs(self) -> Iterator[tuple]:
        """``(root, skill_dir)`` for every ``<root>/<name>/SKILL.md`` under a configured directory."""
        for root in _configured_dirs():
            try:
                entries = sorted(root.iterdir())
            except OSError:
                continue
            for entry in entries:
                if entry.is_dir() and (entry / "SKILL.md").is_file():
                    yield root, entry

    def list_all(self) -> List[SkillMeta]:
        """Every skill across every configured local directory (backs ``import-all``)."""
        return [meta for root, skill_dir in self._iter_skill_dirs()
                if (meta := self._meta(root, skill_dir)) is not None]

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        query_lower = query.lower()
        results: List[SkillMeta] = []
        for root, skill_dir in self._iter_skill_dirs():
            meta = self._meta(root, skill_dir)
            if meta is None:
                continue
            if not query_lower or _matches_query(query_lower, meta.name, meta.description, meta.tags):
                results.append(meta)
            if len(results) >= limit:
                break
        return results

    def _resolve(self, identifier: str) -> Optional[Path]:
        """``local-dir:<root>/<name>`` -> the skill directory, or a bare name matched against every
        configured directory (must be unique)."""
        rel = identifier[len("local-dir:"):] if identifier.startswith("local-dir:") else identifier
        path_text = rel.replace("\\", "/")
        parsed = Path(path_text)
        if parsed.parent != Path("."):
            root_path = parsed.parent.expanduser()
            name = parsed.name
            candidate = root_path / name
            configured_roots = {root.resolve() for root in _configured_dirs()}
            if (candidate.is_dir() and (candidate / "SKILL.md").is_file()
                    and candidate.parent.resolve() in configured_roots):
                return candidate
            rel = name  # fall through to bare-name lookup (root moved/stale)
        matches = [skill_dir for _, skill_dir in self._iter_skill_dirs() if skill_dir.name == rel]
        return matches[0] if len(matches) == 1 else None

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        skill_dir = self._resolve(identifier)
        return None if skill_dir is None else self._meta(skill_dir.parent, skill_dir)

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        skill_dir = self._resolve(identifier)
        if skill_dir is None:
            return None
        try:
            name = _validate_skill_name(skill_dir.name)
        except ValueError:
            logger.warning("Local skill directory has an unsafe name: %s", skill_dir)
            return None
        resolved_root = skill_dir.resolve()
        files: Dict[str, Union[str, bytes]] = {}
        for f in skill_dir.rglob("*"):
            if not f.is_file() or f.is_symlink():
                continue
            rel_path = f.relative_to(skill_dir).as_posix()
            if _skip_bundle_file(rel_path):
                continue
            try:
                # Symlinked files are already skipped above; this also rejects the case where an
                # ancestor directory is a symlink pointing outside skill_dir.
                if not f.resolve().is_relative_to(resolved_root):
                    logger.warning("Local skill %s references a path outside its directory: %s",
                                   identifier, rel_path)
                    return None
                files[rel_path] = f.read_bytes()
            except OSError:
                continue
        if "SKILL.md" not in files:
            return None
        return SkillBundle(
            name=name, files=files, source="local-dir", identifier=f"local-dir:{skill_dir.parent}/{name}",
            trust_level=self.TRUST_LEVEL, metadata={"local_path": str(skill_dir)},
        )

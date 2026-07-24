"""Offline analysis helpers for evolving Claude Code memory and skills.

This module intentionally does not register a Hermes tool. It powers the
interactive ``/evolve-cc`` CLI slash command only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence


CLAUDE_HOME = Path.home() / ".claude"
CLAUDE_PROJECTS_ROOT = CLAUDE_HOME / "projects"
CLAUDE_SKILLS_ROOT = CLAUDE_HOME / "skills"

_CORRECTION_RE = re.compile(
    r"\b(?:don't|dont|do not|stop|instead|no,|no\.)\b|不要|别|改成",
    re.IGNORECASE,
)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "by",
    "do",
    "for",
    "from",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "me",
    "my",
    "not",
    "of",
    "on",
    "or",
    "please",
    "that",
    "the",
    "this",
    "to",
    "use",
    "with",
    "you",
    "your",
}
_MAX_TOOL_NGRAM = 5


@dataclass(frozen=True)
class ClaudeProject:
    """A Claude Code project directory under ``~/.claude/projects``."""

    project_dir: Path
    session_files: tuple[Path, ...]
    cwd: Path | None = None


@dataclass(frozen=True)
class JsonlEvent:
    """One valid JSONL event from a Claude Code session transcript."""

    project_dir: Path
    session_file: Path
    session_id: str
    cwd: Path | None
    timestamp: datetime
    event_type: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class ToolInvocation:
    """A normalized assistant tool invocation."""

    tool_name: str
    args_shape: str
    timestamp: datetime
    day_key: str
    session_id: str
    project_dir: Path
    project_cwd: Path | None


@dataclass(frozen=True)
class CorrectionObservation:
    """A user correction anchored to the immediately preceding assistant step."""

    key: str
    text: str
    timestamp: datetime
    session_id: str
    project_dir: Path
    project_cwd: Path | None
    anchor_context: str


@dataclass(frozen=True)
class MemoryCandidate:
    """A draft memory inferred from repeated user corrections."""

    type: str
    name: str
    description: str
    body: str
    target_project_dirs: tuple[Path, ...]
    evidence: tuple[str, ...]
    frequency: int


@dataclass(frozen=True)
class SkillCandidate:
    """A draft skill inferred from repeated tool-call sequences."""

    slug: str
    title: str
    description: str
    body: str
    evidence: tuple[str, ...]
    frequency: int
    distinct_days: int
    source_projects: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisSummary:
    """Selection metadata for one `/evolve-cc` run."""

    scope: str
    anchor_path: Path
    since: datetime
    repo_root: Path | None
    active_worktree_paths: tuple[Path, ...]
    analysis_projects: tuple[ClaudeProject, ...]
    memory_target_project_dirs: tuple[Path, ...]
    historical_worktree_project_dirs: tuple[Path, ...]


@dataclass(frozen=True)
class AnalysisResult:
    """The full analysis output before reporting or writes."""

    summary: AnalysisSummary
    memory_candidates: tuple[MemoryCandidate, ...]
    skill_candidates: tuple[SkillCandidate, ...]


@dataclass(frozen=True)
class PlannedWrite:
    """A previewable write operation."""

    kind: str
    identifier: str
    path: Path
    content: str
    already_exists: bool


def discover_claude_projects(
    projects_root: Path = CLAUDE_PROJECTS_ROOT,
) -> tuple[ClaudeProject, ...]:
    """Discover Claude Code project directories under ``projects_root``."""

    projects: list[ClaudeProject] = []
    if not projects_root.exists():
        return ()

    for project_dir in sorted(p for p in projects_root.iterdir() if p.is_dir()):
        session_files = tuple(sorted(project_dir.glob("*.jsonl")))
        has_memory = (project_dir / "memory").exists()
        if not session_files and not has_memory:
            continue
        projects.append(
            ClaudeProject(
                project_dir=project_dir,
                session_files=session_files,
                cwd=_extract_project_cwd(session_files),
            )
        )
    return tuple(projects)


def iter_jsonl_events(
    projects: Sequence[ClaudeProject],
    since: datetime,
) -> Iterator[JsonlEvent]:
    """Yield JSONL events whose timestamp falls within the requested window."""

    for project in projects:
        for session_file in project.session_files:
            session_id = session_file.stem
            for payload in _iter_jsonl_payloads(session_file):
                timestamp = _parse_timestamp(payload.get("timestamp"))
                if timestamp is None or timestamp < since:
                    continue
                event_type = str(payload.get("type") or "")
                yield JsonlEvent(
                    project_dir=project.project_dir,
                    session_file=session_file,
                    session_id=str(payload.get("sessionId") or session_id),
                    cwd=_coerce_path(payload.get("cwd")) or project.cwd,
                    timestamp=timestamp,
                    event_type=event_type,
                    payload=payload,
                )


def select_scope(
    anchor_path: Path,
    scope: str,
    projects_root: Path = CLAUDE_PROJECTS_ROOT,
    discovered_projects: Sequence[ClaudeProject] | None = None,
    since: datetime | None = None,
) -> AnalysisSummary:
    """Select Claude Code projects for the requested scope."""

    anchor_path = anchor_path.expanduser().resolve(strict=False)
    if anchor_path.is_file():
        anchor_path = anchor_path.parent

    discovered = tuple(discovered_projects or discover_claude_projects(projects_root))
    project_by_dir = {project.project_dir: project for project in discovered}

    if scope == "all":
        analysis_projects = tuple(project for project in discovered if project.session_files)
        memory_targets = tuple(sorted((project.project_dir for project in analysis_projects), key=str))
        return AnalysisSummary(
            scope=scope,
            anchor_path=anchor_path,
            since=since or datetime.now(timezone.utc),
            repo_root=None,
            active_worktree_paths=(),
            analysis_projects=analysis_projects,
            memory_target_project_dirs=memory_targets,
            historical_worktree_project_dirs=(),
        )

    if scope == "cwd":
        project_dir = _project_dir_for_path(projects_root, anchor_path)
        analysis_project = project_by_dir.get(project_dir)
        return AnalysisSummary(
            scope=scope,
            anchor_path=anchor_path,
            since=since or datetime.now(timezone.utc),
            repo_root=None,
            active_worktree_paths=(),
            analysis_projects=((analysis_project,) if analysis_project and analysis_project.session_files else ()),
            memory_target_project_dirs=(project_dir,),
            historical_worktree_project_dirs=(),
        )

    if scope != "repo":
        raise ValueError(f"Unsupported scope: {scope}")

    repo_root = _git_repo_root(anchor_path)
    if repo_root is None:
        raise ValueError(f"{anchor_path} is not inside a git repository; use --scope cwd or --scope all")

    active_paths = _git_worktree_paths(repo_root)
    active_targets = {
        _project_dir_for_path(projects_root, repo_root),
        *(_project_dir_for_path(projects_root, worktree) for worktree in active_paths),
    }
    historical_dirs: set[Path] = set()
    analysis_dirs: set[Path] = set(active_targets)
    encoded_repo_prefix = re.sub(r"[^A-Za-z0-9]", "-", str(repo_root)) + "-"

    for project in discovered:
        cwd = project.cwd
        if cwd is None:
            if project.project_dir.name.startswith(encoded_repo_prefix):
                analysis_dirs.add(project.project_dir)
                historical_dirs.add(project.project_dir)
            continue
        within_repo = _path_is_within_root(cwd, repo_root)
        within_active_worktree = any(_path_is_within_root(cwd, path) for path in active_paths)
        if within_repo or within_active_worktree:
            analysis_dirs.add(project.project_dir)
            if not cwd.exists():
                historical_dirs.add(project.project_dir)

    analysis_projects = tuple(
        project_by_dir[project_dir]
        for project_dir in sorted(analysis_dirs, key=str)
        if project_dir in project_by_dir and project_by_dir[project_dir].session_files
    )
    memory_targets = tuple(sorted(active_targets | historical_dirs, key=str))

    return AnalysisSummary(
        scope=scope,
        anchor_path=anchor_path,
        since=since or datetime.now(timezone.utc),
        repo_root=repo_root,
        active_worktree_paths=tuple(sorted(active_paths, key=str)),
        analysis_projects=analysis_projects,
        memory_target_project_dirs=memory_targets,
        historical_worktree_project_dirs=tuple(sorted(historical_dirs, key=str)),
    )


def analyze_claude_code_history(
    anchor_path: Path,
    since: datetime,
    scope: str,
    projects_root: Path = CLAUDE_PROJECTS_ROOT,
) -> AnalysisResult:
    """Analyze Claude Code history for memory and skill candidates."""

    discovered = discover_claude_projects(projects_root)
    summary = select_scope(
        anchor_path=anchor_path,
        scope=scope,
        projects_root=projects_root,
        discovered_projects=discovered,
        since=since,
    )
    corrections, tool_segments = _collect_observations(summary.analysis_projects, since)
    memory_candidates = mine_memory_candidates(corrections, summary)
    skill_candidates = mine_skill_candidates(tool_segments)
    return AnalysisResult(
        summary=summary,
        memory_candidates=memory_candidates,
        skill_candidates=skill_candidates,
    )


def mine_memory_candidates(
    observations: Sequence[CorrectionObservation],
    summary: AnalysisSummary,
) -> tuple[MemoryCandidate, ...]:
    """Group repeated user corrections into memory candidates."""

    grouped: dict[str, list[CorrectionObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.key].append(observation)

    candidates: list[MemoryCandidate] = []
    for key, items in grouped.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda item: item.timestamp, reverse=True)
        headline = _summarize_feedback(items[0].text)
        base_slug = _slugify(headline) or _slugify(key) or _digest_text(items[0].text)
        name = f"feedback-{base_slug}"[:80]
        evidence = tuple(
            f"{item.timestamp.date().isoformat()}: {_truncate(item.text.strip(), 140)}"
            for item in items[:3]
        )
        most_recent_context = next((item.anchor_context for item in items if item.anchor_context), "")
        body = _render_memory_body(
            title=headline,
            frequency=len(items),
            context=most_recent_context,
            evidence=evidence,
        )
        if summary.scope == "repo":
            target_dirs = summary.memory_target_project_dirs
        elif summary.scope == "cwd":
            target_dirs = summary.memory_target_project_dirs[:1]
        else:
            target_dirs = tuple(sorted({item.project_dir for item in items}, key=str))
        candidates.append(
            MemoryCandidate(
                type="feedback",
                name=name,
                description=f"Repeated user correction: {headline}",
                body=body,
                target_project_dirs=target_dirs,
                evidence=evidence,
                frequency=len(items),
            )
        )

    candidates.sort(key=lambda candidate: (-candidate.frequency, candidate.name))
    return tuple(candidates)


def mine_skill_candidates(
    tool_segments: Sequence[Sequence[ToolInvocation]],
) -> tuple[SkillCandidate, ...]:
    """Mine repeated tool-call n-grams into skill candidates."""

    stats: dict[tuple[str, ...], dict[str, Any]] = {}
    for segment in tool_segments:
        if len(segment) < 2:
            continue
        max_n = min(_MAX_TOOL_NGRAM, len(segment))
        for n in range(2, max_n + 1):
            for idx in range(0, len(segment) - n + 1):
                window = tuple(segment[idx : idx + n])
                key = tuple(f"{item.tool_name}|{item.args_shape}" for item in window)
                head = window[0]
                entry = stats.get(key)
                if entry is None:
                    stats[key] = {
                        "exemplar": window,
                        "count": 1,
                        "days": {head.day_key},
                        "sources": {str(head.project_cwd or head.project_dir)},
                        "first_heads": [head],
                    }
                else:
                    entry["count"] += 1
                    entry["days"].add(head.day_key)
                    entry["sources"].add(str(head.project_cwd or head.project_dir))
                    if len(entry["first_heads"]) < 3:
                        entry["first_heads"].append(head)

    candidates: list[SkillCandidate] = []
    for key, entry in stats.items():
        count = entry["count"]
        days = entry["days"]
        if count < 3 or len(days) < 2:
            continue
        exemplar = entry["exemplar"]
        first_heads = entry["first_heads"]
        sources = entry["sources"]
        tool_names = [item.tool_name for item in exemplar]
        slug_base = _slugify("-".join(name.lower() for name in tool_names)) or "workflow"
        slug = f"{slug_base}-{hashlib.sha1('||'.join(key).encode('utf-8')).hexdigest()[:6]}"
        title = " -> ".join(tool_names)
        evidence = tuple(
            f"{head.day_key}: {head.project_cwd or head.project_dir}"
            for head in first_heads
        )
        source_projects = tuple(sorted(sources))
        body = _render_skill_body(
            title=title,
            sequence=exemplar,
            frequency=count,
            distinct_days=len(days),
            evidence=evidence,
        )
        candidates.append(
            SkillCandidate(
                slug=slug,
                title=title,
                description=(
                    f"Observed {count} times across {len(days)} days "
                    f"from Claude Code history"
                ),
                body=body,
                evidence=evidence,
                frequency=count,
                distinct_days=len(days),
                source_projects=source_projects,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            -candidate.frequency,
            -candidate.distinct_days,
            -len(candidate.title),
            candidate.slug,
        )
    )
    return tuple(candidates)


def format_candidates_report(
    result: AnalysisResult,
    memory_candidates: Sequence[MemoryCandidate],
    skill_candidates: Sequence[SkillCandidate],
    write_plan: Sequence[PlannedWrite],
) -> str:
    """Render an ASCII report for dry-run or apply mode."""

    lines = [
        "+------------------------------------------------------------------------------+",
        "| Claude Code Evolution                                                        |",
        "+------------------------------------------------------------------------------+",
        f"Scope:              {result.summary.scope}",
        f"Anchor:             {result.summary.anchor_path}",
        f"Since:              {result.summary.since.isoformat()}",
        f"Analyzed projects:  {len(result.summary.analysis_projects)}",
        f"Memory targets:     {len(result.summary.memory_target_project_dirs)}",
    ]
    if result.summary.repo_root:
        lines.append(f"Repo root:          {result.summary.repo_root}")
        lines.append(f"Active worktrees:   {len(result.summary.active_worktree_paths)}")
        lines.append(
            f"Historical worktree project dirs: {len(result.summary.historical_worktree_project_dirs)}"
        )

    planned_new = sum(1 for write in write_plan if not write.already_exists)
    planned_existing = sum(1 for write in write_plan if write.already_exists)
    lines.extend(
        [
            f"Planned writes:     {len(write_plan)} total ({planned_new} new, {planned_existing} existing/no-op)",
            "",
            f"Memory candidates shown: {len(memory_candidates)} / {len(result.memory_candidates)}",
        ]
    )

    if memory_candidates:
        for idx, candidate in enumerate(memory_candidates, start=1):
            lines.append(
                f"  [{idx}] {candidate.name}  x{candidate.frequency}  {candidate.description}"
            )
            for evidence in candidate.evidence[:2]:
                lines.append(f"      evidence: {evidence}")
            for write in _writes_for(write_plan, "memory", candidate.name)[:5]:
                suffix = " (exists)" if write.already_exists else ""
                lines.append(f"      -> {write.path}{suffix}")
            extra = len(_writes_for(write_plan, "memory", candidate.name)) - 5
            if extra > 0:
                lines.append(f"      ... {extra} more target(s)")
    else:
        lines.append("  (none)")

    lines.extend(
        [
            "",
            f"Skill candidates shown: {len(skill_candidates)} / {len(result.skill_candidates)}",
        ]
    )
    if skill_candidates:
        for idx, candidate in enumerate(skill_candidates, start=1):
            lines.append(
                f"  [{idx}] {candidate.slug}  x{candidate.frequency} / {candidate.distinct_days} day(s)"
            )
            lines.append(f"      {candidate.title}")
            for evidence in candidate.evidence[:2]:
                lines.append(f"      evidence: {evidence}")
            for write in _writes_for(write_plan, "skill", candidate.slug):
                suffix = " (exists)" if write.already_exists else ""
                lines.append(f"      -> {write.path}{suffix}")
    else:
        lines.append("  (none)")

    return "\n".join(lines)


def plan_candidate_writes(
    memory_candidates: Sequence[MemoryCandidate],
    skill_candidates: Sequence[SkillCandidate],
    projects_root: Path = CLAUDE_PROJECTS_ROOT,
    skills_root: Path = CLAUDE_SKILLS_ROOT,
) -> tuple[PlannedWrite, ...]:
    """Preview the exact write operations for a set of candidates."""

    reserved_paths: set[Path] = set()
    writes: list[PlannedWrite] = []

    for candidate in memory_candidates:
        content = render_memory_candidate(candidate)
        for project_dir in candidate.target_project_dirs:
            path, already_exists = _plan_memory_path(
                project_dir=project_dir,
                content=content,
                base_name=candidate.name,
                projects_root=projects_root,
                reserved_paths=reserved_paths,
            )
            reserved_paths.add(path)
            writes.append(
                PlannedWrite(
                    kind="memory",
                    identifier=candidate.name,
                    path=path,
                    content=content,
                    already_exists=already_exists,
                )
            )

    for candidate in skill_candidates:
        content = render_skill_candidate(candidate)
        path, already_exists = _plan_skill_path(
            skills_root=skills_root,
            content=content,
            base_slug=candidate.slug,
            reserved_paths=reserved_paths,
        )
        reserved_paths.add(path)
        writes.append(
            PlannedWrite(
                kind="skill",
                identifier=candidate.slug,
                path=path,
                content=content,
                already_exists=already_exists,
            )
        )

    return tuple(writes)


def apply_write_plan(write_plan: Sequence[PlannedWrite]) -> tuple[int, int]:
    """Execute a planned write set."""

    written = 0
    skipped = 0
    for write in write_plan:
        if write.already_exists:
            skipped += 1
            continue
        _atomic_write(write.path, write.content)
        written += 1
    return written, skipped


def render_memory_candidate(candidate: MemoryCandidate) -> str:
    """Return the markdown body for a memory candidate."""

    return candidate.body.rstrip() + "\n"


def render_skill_candidate(candidate: SkillCandidate) -> str:
    """Return the SKILL.md content for a skill candidate."""

    return (
        "---\n"
        f"name: {candidate.slug}_evolved\n"
        f"description: {candidate.description}\n"
        "---\n\n"
        f"{candidate.body.rstrip()}\n"
    )


def write_memory_candidate(
    project_dir: Path,
    candidate: MemoryCandidate,
    projects_root: Path = CLAUDE_PROJECTS_ROOT,
) -> Path:
    """Write one memory candidate into a Claude project directory."""

    content = render_memory_candidate(candidate)
    path, already_exists = _plan_memory_path(
        project_dir=project_dir,
        content=content,
        base_name=candidate.name,
        projects_root=projects_root,
        reserved_paths=set(),
    )
    if not already_exists:
        _atomic_write(path, content)
    return path


def write_skill_candidate(
    skills_root: Path,
    candidate: SkillCandidate,
) -> Path:
    """Write one skill candidate into ``skills_root``."""

    content = render_skill_candidate(candidate)
    path, already_exists = _plan_skill_path(
        skills_root=skills_root,
        content=content,
        base_slug=candidate.slug,
        reserved_paths=set(),
    )
    if not already_exists:
        _atomic_write(path, content)
    return path


def _collect_observations(
    projects: Sequence[ClaudeProject],
    since: datetime,
) -> tuple[tuple[CorrectionObservation, ...], tuple[tuple[ToolInvocation, ...], ...]]:
    corrections: list[CorrectionObservation] = []
    segments: list[tuple[ToolInvocation, ...]] = []
    since_ts = since.timestamp()
    cwd_cache: dict[str, Path | None] = {}

    def _cached_cwd(value: Any, fallback: Path | None) -> Path | None:
        if not isinstance(value, str) or not value:
            return fallback
        if value in cwd_cache:
            return cwd_cache[value] or fallback
        resolved = _coerce_path(value)
        cwd_cache[value] = resolved
        return resolved or fallback

    for project in projects:
        for session_file in project.session_files:
            try:
                if session_file.stat().st_mtime < since_ts:
                    continue
            except OSError:
                continue
            session_id = session_file.stem
            current_segment: list[ToolInvocation] = []
            last_assistant_context = ""
            for payload in _iter_jsonl_payloads(session_file):
                event_type = str(payload.get("type") or "")
                timestamp = _parse_timestamp(payload.get("timestamp"))
                if event_type == "user":
                    content = _message_content(payload)
                    if _is_tool_result_only(content):
                        continue
                    if current_segment:
                        segments.append(tuple(current_segment))
                        current_segment = []
                    if timestamp is None or timestamp < since:
                        continue
                    text = _extract_text_content(content)
                    if text and last_assistant_context and _looks_like_correction(text):
                        corrections.append(
                            CorrectionObservation(
                                key=_normalize_correction_key(text),
                                text=text,
                                timestamp=timestamp,
                                session_id=str(payload.get("sessionId") or session_id),
                                project_dir=project.project_dir,
                                project_cwd=_cached_cwd(payload.get("cwd"), project.cwd),
                                anchor_context=last_assistant_context,
                            )
                        )
                    continue

                if event_type != "assistant":
                    continue

                content = _message_content(payload)
                context = _assistant_context(content)
                if context:
                    last_assistant_context = context

                if timestamp is None or timestamp < since:
                    continue

                project_cwd = _cached_cwd(payload.get("cwd"), project.cwd)
                session_id_value = str(payload.get("sessionId") or session_id)
                day_key = timestamp.date().isoformat()
                invocations = [
                    ToolInvocation(
                        tool_name=name,
                        args_shape=args_shape,
                        timestamp=timestamp,
                        day_key=day_key,
                        session_id=session_id_value,
                        project_dir=project.project_dir,
                        project_cwd=project_cwd,
                    )
                    for name, args_shape in _extract_tool_calls(content)
                ]
                current_segment.extend(invocations)

            if current_segment:
                segments.append(tuple(current_segment))

    return tuple(corrections), tuple(segments)


_CWD_SCAN_LINE_LIMIT = 50


def _extract_project_cwd(session_files: Sequence[Path]) -> Path | None:
    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    for session_file in sorted(session_files, key=_mtime, reverse=True):
        for idx, payload in enumerate(_iter_jsonl_payloads(session_file)):
            if idx >= _CWD_SCAN_LINE_LIMIT:
                break
            cwd = _coerce_path(payload.get("cwd"))
            if cwd is not None:
                return cwd
    return None


def _iter_jsonl_payloads(session_file: Path) -> Iterator[dict[str, Any]]:
    try:
        with session_file.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    yield payload
    except OSError:
        return


def _extract_tool_calls(content: Any) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []
    if not isinstance(content, list):
        return calls
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        name = str(block.get("name") or "")
        if not name:
            continue
        calls.append((name, _normalize_tool_call(name, block.get("input"))))
    return calls


def _assistant_context(content: Any) -> str:
    tool_calls = _extract_tool_calls(content)
    if tool_calls:
        rendered = ", ".join(
            f"{name}({args_shape})" if args_shape else name
            for name, args_shape in tool_calls[:3]
        )
        return _truncate(rendered, 160)
    text = _extract_text_content(content)
    return _truncate(" ".join(text.split()), 160) if text else ""


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        if isinstance(block.get("text"), str):
            parts.append(str(block["text"]))
        elif block.get("type") in {"text", "output_text"} and isinstance(block.get("content"), str):
            parts.append(str(block["content"]))
    return "\n".join(part for part in parts if part)


def _looks_like_correction(text: str) -> bool:
    compact = " ".join(text.split())
    if len(compact) > 350:
        return False
    return bool(_CORRECTION_RE.search(compact))


def _normalize_correction_key(text: str) -> str:
    summary = _summarize_feedback(text)
    tokens = [
        token
        for token in re.findall(r"[a-z0-9_]+", summary.lower())
        if token not in _STOPWORDS
    ]
    if not tokens:
        return _digest_text(text)
    return "-".join(tokens[:8])


def _summarize_feedback(text: str) -> str:
    compact = " ".join(text.split())
    patterns = [
        r"(?:don't|dont|do not)\s+([^.!?\n]+)",
        r"(?:stop)\s+([^.!?\n]+)",
        r"(?:instead)\s+([^.!?\n]+)",
        r"(?:不要|别|改成)([^。！!\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, compact, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip(" ,.;:-")
            if candidate:
                return _truncate(candidate, 80)
    return _truncate(compact, 80)


def _render_memory_body(
    title: str,
    frequency: int,
    context: str,
    evidence: Sequence[str],
) -> str:
    lines = [
        f"# Evolved Memory: {title}",
        "",
        "## Why",
        f"- This correction appeared {frequency} times in Claude Code history.",
    ]
    if context:
        lines.append(f"- It was usually prompted by: `{context}`")
    lines.extend(
        [
            "",
            "## How To Apply",
            f"- Treat `{title}` as a standing preference until the user says otherwise.",
            "- Check for this preference before repeating the same workflow or tool choice.",
            "",
            "## Evidence",
        ]
    )
    lines.extend(f"- {item}" for item in evidence)
    return "\n".join(lines)


def _render_skill_body(
    title: str,
    sequence: Sequence[ToolInvocation],
    frequency: int,
    distinct_days: int,
    evidence: Sequence[str],
) -> str:
    lines = [
        f"# Evolved Workflow: {title}",
        "",
        "## When To Use",
        (
            f"This sequence appeared {frequency} times across {distinct_days} days in "
            "Claude Code history."
        ),
        "",
        "## Steps",
    ]
    for idx, item in enumerate(sequence, start=1):
        lines.append(f"{idx}. Run `{item.tool_name}` with args shape `{item.args_shape}`.")
    lines.extend(["", "## Evidence"])
    lines.extend(f"- {item}" for item in evidence)
    return "\n".join(lines)


def _normalize_tool_call(tool_name: str, args: Any) -> str:
    if not isinstance(args, dict):
        return type(args).__name__ if args is not None else ""

    keys = []
    for key in sorted(args):
        value = args[key]
        if isinstance(value, dict):
            keys.append(f"{key}:object")
        elif isinstance(value, list):
            keys.append(f"{key}:list")
        elif value is None:
            keys.append(f"{key}:null")
        else:
            keys.append(str(key))
    return ",".join(keys)


def _plan_memory_path(
    project_dir: Path,
    content: str,
    base_name: str,
    projects_root: Path,
    reserved_paths: set[Path],
) -> tuple[Path, bool]:
    _ensure_within_root(project_dir, projects_root)
    memory_dir = project_dir / "memory"
    return _plan_markdown_path(memory_dir, base_name, content, reserved_paths, skip_names={"MEMORY.md"})


def _plan_skill_path(
    skills_root: Path,
    content: str,
    base_slug: str,
    reserved_paths: set[Path],
) -> tuple[Path, bool]:
    skills_root = skills_root.expanduser().resolve(strict=False)
    _ensure_within_root(skills_root, skills_root)
    base_dir = skills_root / f"{base_slug}_evolved"
    exact_path = base_dir / "SKILL.md"
    if exact_path.exists():
        try:
            if exact_path.read_text(encoding="utf-8") == content:
                return exact_path, True
        except OSError:
            pass
    if exact_path not in reserved_paths and not exact_path.exists():
        return exact_path, False

    for idx in range(2, 1000):
        candidate = skills_root / f"{base_slug}_evolved-{idx}" / "SKILL.md"
        if candidate.exists():
            try:
                if candidate.read_text(encoding="utf-8") == content:
                    return candidate, True
            except OSError:
                pass
        if candidate not in reserved_paths and not candidate.exists():
            return candidate, False

    raise RuntimeError(f"Could not allocate a unique skill path for {base_slug}")


def _plan_markdown_path(
    target_dir: Path,
    base_name: str,
    content: str,
    reserved_paths: set[Path],
    skip_names: set[str] | None = None,
) -> tuple[Path, bool]:
    skip_names = skip_names or set()
    if target_dir.exists():
        for existing in sorted(target_dir.glob("*.md")):
            if existing.name in skip_names:
                continue
            try:
                if existing.read_text(encoding="utf-8") == content:
                    return existing, True
            except OSError:
                continue

    for idx in range(1, 1000):
        suffix = "" if idx == 1 else f"-{idx}"
        candidate = target_dir / f"{base_name}{suffix}.md"
        if candidate in reserved_paths:
            continue
        if not candidate.exists():
            return candidate, False
        try:
            if candidate.read_text(encoding="utf-8") == content:
                return candidate, True
        except OSError:
            pass

    raise RuntimeError(f"Could not allocate a unique markdown path under {target_dir}")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.tmp.",
        suffix="",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.chmod(temp_path, 0o644)
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def _git_repo_root(anchor_path: Path) -> Path | None:
    try:
        common_dir = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(anchor_path),
        )
    except Exception:
        common_dir = None

    if common_dir and common_dir.returncode == 0:
        path = common_dir.stdout.strip()
        if path:
            common_path = Path(path).resolve(strict=False)
            if common_path.name == ".git":
                return common_path.parent

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(anchor_path),
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    path = result.stdout.strip()
    return Path(path).resolve(strict=False) if path else None


def _git_worktree_paths(repo_root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    try:
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(repo_root),
        )
    except Exception:
        return ()
    if result.returncode != 0:
        return ()
    for line in result.stdout.splitlines():
        if not line.startswith("worktree "):
            continue
        path = line.split("worktree ", 1)[1].strip()
        if not path:
            continue
        resolved = Path(path).resolve(strict=False)
        if _path_eq(resolved, repo_root):
            continue
        paths.append(resolved)
    return tuple(sorted({path for path in paths}, key=str))


def _project_dir_for_path(projects_root: Path, path: Path) -> Path:
    encoded = re.sub(r"[^A-Za-z0-9]", "-", str(path.resolve(strict=False)))
    return projects_root.expanduser().resolve(strict=False) / encoded


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _coerce_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return Path(value).expanduser().resolve(strict=False)
    except Exception:
        return None


def _message_content(payload: dict[str, Any]) -> Any:
    message = payload.get("message")
    if isinstance(message, dict):
        return message.get("content")
    return None


def _is_tool_result_only(content: Any) -> bool:
    if not isinstance(content, list) or not content:
        return False
    for block in content:
        if not isinstance(block, dict):
            return False
        if block.get("type") != "tool_result":
            return False
    return True


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:64]


def _truncate(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def _digest_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _writes_for(
    write_plan: Sequence[PlannedWrite],
    kind: str,
    identifier: str,
) -> list[PlannedWrite]:
    return [write for write in write_plan if write.kind == kind and write.identifier == identifier]


def _ensure_within_root(path: Path, root: Path) -> None:
    resolved_root = root.expanduser().resolve(strict=False)
    resolved_path = path.expanduser().resolve(strict=False)
    if not _path_is_within_root(resolved_path, resolved_root):
        raise ValueError(f"{resolved_path} is outside {resolved_root}")


def _path_is_within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _path_eq(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)

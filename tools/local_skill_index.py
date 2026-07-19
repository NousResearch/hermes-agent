"""Local SQLite FTS5 index for token-efficient skill discovery.

The index is a derived cache. Skill files remain the source of truth; the
optional Obsidian dashboard contains metadata and relationships, never copied
instruction bodies.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

from agent.markdown_sections import split_markdown_sections
from agent.skill_utils import iter_skill_index_files, parse_frontmatter

_WORD_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
_INDEX_VERSION = "1"


def _normalise_roots(skill_roots: Iterable[Path | str]) -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()
    for raw_root in skill_roots:
        root = Path(raw_root).expanduser()
        key = str(root.resolve()) if root.exists() else str(root.absolute())
        if key in seen:
            continue
        seen.add(key)
        roots.append(root)
    return roots


def _support_markdown_files(skill_dir: Path) -> list[Path]:
    files: list[Path] = []
    references = skill_dir / "references"
    if not references.is_dir():
        return files
    for path in sorted(references.rglob("*.md")):
        if path.is_file() and not path.is_symlink():
            files.append(path)
    return files


def _skill_source_files(skill_roots: Sequence[Path]) -> list[tuple[Path, Path]]:
    """Return ``(root, SKILL.md)`` pairs with local-root precedence."""
    found: list[tuple[Path, Path]] = []
    seen_names: set[str] = set()
    for root in skill_roots:
        if not root.is_dir():
            continue
        for skill_md in iter_skill_index_files(root, "SKILL.md"):
            try:
                raw = skill_md.read_text(encoding="utf-8")
                frontmatter, _ = parse_frontmatter(raw)
            except (OSError, UnicodeError):
                continue
            name = str(frontmatter.get("name") or skill_md.parent.name).strip()
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            found.append((root, skill_md))
    return found


def _source_signature(skill_roots: Sequence[Path]) -> str:
    """Hash canonical skill sources; usage and Obsidian inputs are metadata only."""
    digest = hashlib.sha256()
    for root, skill_md in _skill_source_files(skill_roots):
        paths = [skill_md, *_support_markdown_files(skill_md.parent)]
        for path in paths:
            try:
                stat = path.stat()
                relative = path.relative_to(root)
            except (OSError, ValueError):
                continue
            digest.update(str(root).encode("utf-8"))
            digest.update(str(relative).encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _parse_tags(frontmatter: dict[str, Any]) -> list[str]:
    metadata = frontmatter.get("metadata")
    hermes = metadata.get("hermes") if isinstance(metadata, dict) else None
    raw_tags = hermes.get("tags") if isinstance(hermes, dict) else []
    if isinstance(raw_tags, str):
        raw_tags = [part.strip() for part in raw_tags.strip("[]").split(",")]
    if not isinstance(raw_tags, list):
        return []
    return [str(tag).strip() for tag in raw_tags if str(tag).strip()]


def _category_for(root: Path, skill_md: Path) -> str:
    try:
        parts = skill_md.relative_to(root).parts
    except ValueError:
        return "general"
    return parts[0] if len(parts) >= 3 else "general"


def _load_json_dict(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _obsidian_links(path: Path | None, skill_names: Iterable[str]) -> dict[str, list[str]]:
    if path is None or not path.is_file():
        return {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    result: dict[str, list[str]] = {}
    for name in skill_names:
        pattern = re.compile(rf"(?<![A-Za-z0-9_-]){re.escape(name)}(?![A-Za-z0-9_-])")
        links: set[str] = set()
        for line in lines:
            if pattern.search(line):
                links.update(_WIKILINK_RE.findall(line))
        if links:
            result[name] = sorted(links)
    return result


def _write_dashboard(
    output_path: Path,
    skills: list[dict[str, Any]],
    *,
    signature: str,
    routing_map_path: Path | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_chars = sum(int(skill["char_count"]) for skill in skills)
    oversized = [skill for skill in skills if int(skill["char_count"]) > 20_000]
    lines = [
        "# Hermes Skill Index",
        "",
        "> Generated from canonical `SKILL.md` files. Do not edit this dashboard; edit the skill or routing map and regenerate.",
        "",
        f"- Skills: **{len(skills)}**",
        f"- Router characters: **{total_chars:,}**",
        f"- Estimated router tokens: **{math.ceil(total_chars / 4):,}**",
        f"- Source signature: `{signature[:12]}`",
    ]
    if routing_map_path is not None:
        lines.append("- Human routing relationships: [[Operations/Hermes-Skill-Map]]")

    lines.extend(["", "## Oversized routers", ""])
    if oversized:
        lines.extend(["| Skill | Characters | Est. tokens | References |", "|---|---:|---:|---:|"])
        for skill in sorted(oversized, key=lambda row: int(row["char_count"]), reverse=True):
            lines.append(
                f"| `{skill['name']}` | {int(skill['char_count']):,} | "
                f"{int(skill['token_estimate']):,} | {int(skill['reference_count'])} |"
            )
    else:
        lines.append("No routers exceed 20,000 characters.")

    lines.extend(
        [
            "",
            "## All skills",
            "",
            "| Skill | Category | Characters | Est. tokens | Refs | Views | Uses | Related notes |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for skill in sorted(skills, key=lambda row: (str(row["category"]), str(row["name"]))):
        note_links = " ".join(f"[[{link}]]" for link in skill.get("obsidian_links", []))
        note_links = note_links.replace("|", "\\|")
        lines.append(
            f"| `{skill['name']}` | {skill['category']} | {int(skill['char_count']):,} | "
            f"{int(skill['token_estimate']):,} | {int(skill['reference_count'])} | "
            f"{int(skill['view_count'])} | {int(skill['use_count'])} | {note_links} |"
        )
    lines.append("")

    content = "\n".join(lines)
    fd, temp_name = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temp_name, output_path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def build_skill_index(
    skill_roots: Iterable[Path | str],
    db_path: Path | str,
    *,
    usage_path: Path | str | None = None,
    obsidian_map_path: Path | str | None = None,
    obsidian_output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Build an atomic SQLite FTS5 index and optional Obsidian dashboard."""
    roots = _normalise_roots(skill_roots)
    db_path = Path(db_path).expanduser()
    usage = Path(usage_path).expanduser() if usage_path else None
    obsidian_map = Path(obsidian_map_path).expanduser() if obsidian_map_path else None
    obsidian_output = Path(obsidian_output_path).expanduser() if obsidian_output_path else None
    signature = _source_signature(roots)
    usage_data = _load_json_dict(usage)

    records: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    for root, skill_md in _skill_source_files(roots):
        try:
            raw = skill_md.read_text(encoding="utf-8")
            frontmatter, body = parse_frontmatter(raw)
        except (OSError, UnicodeError):
            continue
        name = str(frontmatter.get("name") or skill_md.parent.name).strip()
        description = str(frontmatter.get("description") or "").strip()
        category = _category_for(root, skill_md)
        tags = _parse_tags(frontmatter)
        refs = _support_markdown_files(skill_md.parent)
        content_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        raw_usage = usage_data.get(name)
        skill_usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        record = {
            "name": name,
            "description": description,
            "category": category,
            "path": str(skill_md),
            "char_count": len(raw),
            "token_estimate": math.ceil(len(raw) / 4),
            "content_hash": content_hash,
            "reference_count": len(refs),
            "view_count": int(skill_usage.get("view_count") or 0),
            "use_count": int(skill_usage.get("use_count") or 0),
            "last_used_at": str(skill_usage.get("last_used_at") or ""),
        }
        records.append(record)

        files = [(skill_md, "router", body)]
        for reference in refs:
            try:
                files.append((reference, "reference", reference.read_text(encoding="utf-8")))
            except (OSError, UnicodeError):
                continue
        for path, kind, content in files:
            relative_path = "SKILL.md" if path == skill_md else path.relative_to(skill_md.parent).as_posix()
            for section, section_content, level in split_markdown_sections(content):
                documents.append(
                    {
                        "skill_name": name,
                        "category": category,
                        "kind": kind,
                        "file_path": relative_path,
                        "section": section,
                        "heading_level": level,
                        "title": section or name,
                        "description": description,
                        "tags": " ".join(tags),
                        "content": section_content,
                    }
                )

    links_by_skill = _obsidian_links(obsidian_map, (record["name"] for record in records))
    for record in records:
        record["obsidian_links"] = links_by_skill.get(record["name"], [])

    db_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{db_path.name}.", dir=db_path.parent)
    os.close(fd)
    try:
        connection = sqlite3.connect(temp_name)
        try:
            connection.executescript(
                """
                PRAGMA journal_mode=DELETE;
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE skills (
                    name TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    category TEXT NOT NULL,
                    path TEXT NOT NULL,
                    char_count INTEGER NOT NULL,
                    token_estimate INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    reference_count INTEGER NOT NULL,
                    view_count INTEGER NOT NULL,
                    use_count INTEGER NOT NULL,
                    last_used_at TEXT NOT NULL,
                    obsidian_links TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE skill_docs USING fts5(
                    skill_name,
                    category,
                    kind UNINDEXED,
                    file_path UNINDEXED,
                    section,
                    heading_level UNINDEXED,
                    title,
                    description,
                    tags,
                    content,
                    tokenize='porter unicode61'
                );
                """
            )
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                [("version", _INDEX_VERSION), ("source_signature", signature)],
            )
            connection.executemany(
                """
                INSERT INTO skills VALUES (
                    :name, :description, :category, :path, :char_count,
                    :token_estimate, :content_hash, :reference_count,
                    :view_count, :use_count, :last_used_at, :obsidian_links
                )
                """,
                [
                    {
                        **record,
                        "obsidian_links": json.dumps(record["obsidian_links"], ensure_ascii=False),
                    }
                    for record in records
                ],
            )
            connection.executemany(
                """
                INSERT INTO skill_docs(
                    skill_name, category, kind, file_path, section,
                    heading_level, title, description, tags, content
                ) VALUES (
                    :skill_name, :category, :kind, :file_path, :section,
                    :heading_level, :title, :description, :tags, :content
                )
                """,
                documents,
            )
            connection.commit()
        finally:
            connection.close()
        os.replace(temp_name, db_path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise

    if obsidian_output is not None:
        _write_dashboard(
            obsidian_output,
            records,
            signature=signature,
            routing_map_path=obsidian_map,
        )

    return {
        "rebuilt": True,
        "skill_count": len(records),
        "document_count": len(documents),
        "source_signature": signature,
        "db_path": str(db_path),
        "obsidian_output_path": str(obsidian_output) if obsidian_output else None,
    }


def _read_index_signature(db_path: Path) -> tuple[str, str] | None:
    if not db_path.is_file():
        return None
    try:
        with sqlite3.connect(db_path) as connection:
            rows = dict(connection.execute("SELECT key, value FROM metadata"))
    except (sqlite3.Error, OSError):
        return None
    version = rows.get("version")
    signature = rows.get("source_signature")
    return (str(version), str(signature)) if version and signature else None


def ensure_skill_index(
    skill_roots: Iterable[Path | str],
    db_path: Path | str,
    *,
    usage_path: Path | str | None = None,
    obsidian_map_path: Path | str | None = None,
    obsidian_output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Reuse a fresh index or atomically rebuild a stale one."""
    roots = _normalise_roots(skill_roots)
    db = Path(db_path).expanduser()
    usage = Path(usage_path).expanduser() if usage_path else None
    obsidian_map = Path(obsidian_map_path).expanduser() if obsidian_map_path else None
    obsidian_output = (
        Path(obsidian_output_path).expanduser() if obsidian_output_path else None
    )
    signature = _source_signature(roots)
    existing = _read_index_signature(db)
    if existing == (_INDEX_VERSION, signature) and (
        obsidian_output is None or obsidian_output.is_file()
    ):
        return {
            "rebuilt": False,
            "source_signature": signature,
            "db_path": str(db),
            "obsidian_output_path": str(obsidian_output_path) if obsidian_output_path else None,
        }
    return build_skill_index(
        roots,
        db,
        usage_path=usage,
        obsidian_map_path=obsidian_map,
        obsidian_output_path=obsidian_output_path,
    )


def _fts_query(query: str) -> str:
    words = [word for word in _WORD_RE.findall(query) if len(word) >= 2]
    # Quoted prefix terms prevent FTS operators in user text from changing the query.
    return " OR ".join(f'"{word.replace(chr(34), "")}"*' for word in words[:24])


def search_skill_index(
    db_path: Path | str,
    query: str,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return compact ranked routing cards, never full instruction bodies."""
    expression = _fts_query(query)
    if not expression:
        return []
    limit = max(1, min(int(limit), 20))
    db = Path(db_path).expanduser()
    sql = """
        SELECT
            d.skill_name,
            d.category,
            d.kind,
            d.file_path,
            d.section,
            d.title,
            snippet(skill_docs, 9, '**', '**', '…', 18) AS snippet,
            bm25(skill_docs, 10.0, 2.0, 0.0, 0.0, 5.0, 0.0, 6.0, 4.0, 5.0, 1.0) AS score,
            s.description,
            s.char_count,
            s.token_estimate,
            s.content_hash,
            s.reference_count,
            s.view_count,
            s.use_count,
            s.last_used_at,
            s.obsidian_links
        FROM skill_docs AS d
        JOIN skills AS s ON s.name = d.skill_name
        WHERE skill_docs MATCH ?
        ORDER BY score ASC, d.skill_name ASC
        LIMIT ?
    """
    try:
        with sqlite3.connect(db) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(sql, (expression, limit)).fetchall()
    except (sqlite3.Error, OSError):
        return []

    results: list[dict[str, Any]] = []
    for row in rows:
        file_path = str(row["file_path"])
        section = str(row["section"] or "")
        arguments = [f"name={json.dumps(str(row['skill_name']), ensure_ascii=False)}"]
        if file_path != "SKILL.md":
            arguments.append(
                f"file_path={json.dumps(file_path, ensure_ascii=False)}"
            )
        if section:
            arguments.append(
                f"section={json.dumps(section, ensure_ascii=False)}"
            )
        results.append(
            {
                "name": row["skill_name"],
                "category": row["category"],
                "description": row["description"],
                "kind": row["kind"],
                "file_path": file_path,
                "section": section or None,
                "title": row["title"],
                "snippet": row["snippet"],
                "score": float(row["score"]),
                "char_count": int(row["char_count"]),
                "token_estimate": int(row["token_estimate"]),
                "content_hash": row["content_hash"],
                "reference_count": int(row["reference_count"]),
                "view_count": int(row["view_count"]),
                "use_count": int(row["use_count"]),
                "last_used_at": row["last_used_at"] or None,
                "obsidian_links": json.loads(row["obsidian_links"] or "[]"),
                "recommended_skill_view": f"skill_view({', '.join(arguments)})",
            }
        )
    return results

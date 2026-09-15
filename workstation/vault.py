from __future__ import annotations

import atexit
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import yaml

from hermes_constants import get_config_path, get_hermes_home
from utils import atomic_roundtrip_yaml_update

_log = logging.getLogger(__name__)
VAULT_ROOT_CONFIG_KEY = "workstation.vault.root"

# Regular expressions for wikilinks and tags
# Matches [[Target]], [[Target#Heading]], [[Target|Alias]], [[Target#Heading|Alias]]
WIKILINK_REGEX = re.compile(r"\[\[([^\]\|#\n]+)(?:#([^\]\|\n]+))?(?:\|([^\]\n]+))?\]\]")
TAG_REGEX = re.compile(r"(?:^|\s)#([a-zA-Z0-9_\-\/]+)")


@dataclass
class Wikilink:
    target: str
    section: Optional[str] = None
    alias: Optional[str] = None
    raw: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "section": self.section,
            "alias": self.alias,
            "raw": self.raw,
        }


@dataclass
class VaultNote:
    path: str
    rel_path: str
    title: str
    content: str
    frontmatter: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    links: List[Wikilink] = field(default_factory=list)
    mtime: float = 0.0
    size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "rel_path": self.rel_path,
            "title": self.title,
            "content": self.content,
            "frontmatter": self.frontmatter,
            "tags": self.tags,
            "links": [link.to_dict() for link in self.links],
            "mtime": self.mtime,
            "size": self.size,
        }


def parse_frontmatter_and_content(raw_text: str) -> Tuple[Dict[str, Any], str]:
    """Parse YAML frontmatter if present, returning (frontmatter_dict, remaining_content)."""
    if not raw_text.startswith("---"):
        return {}, raw_text

    parts = raw_text.split("---", 2)
    if len(parts) >= 3:
        yaml_content = parts[1]
        body = parts[2].lstrip("\r\n")
        try:
            parsed = yaml.safe_load(yaml_content)
            if isinstance(parsed, dict):
                return parsed, body
        except Exception as e:
            _log.warning("Failed to parse YAML frontmatter: %s", e)
    return {}, raw_text


def extract_wikilinks(text: str) -> List[Wikilink]:
    """Extract all [[wikilinks]] from text."""
    links: List[Wikilink] = []
    for match in WIKILINK_REGEX.finditer(text):
        target = match.group(1).strip()
        section = match.group(2).strip() if match.group(2) else None
        alias = match.group(3).strip() if match.group(3) else None
        links.append(
            Wikilink(
                target=target,
                section=section,
                alias=alias,
                raw=match.group(0),
            )
        )
    return links


def extract_tags(text: str, frontmatter_tags: Optional[Any] = None) -> List[str]:
    """Extract unique tags from frontmatter and markdown body (e.g. #topic, #status/todo)."""
    tags_set: Set[str] = set()

    if frontmatter_tags:
        if isinstance(frontmatter_tags, list):
            for t in frontmatter_tags:
                cleaned = str(t).lstrip("#").strip()
                if cleaned:
                    tags_set.add(cleaned)
        elif isinstance(frontmatter_tags, str):
            for t in frontmatter_tags.split(","):
                cleaned = t.lstrip("#").strip()
                if cleaned:
                    tags_set.add(cleaned)

    # Extract inline tags from body, avoiding code blocks
    in_code_block = False
    for line in text.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block or trimmed.startswith("#"):  # skip markdown headers
            continue

        for match in TAG_REGEX.finditer(line):
            tag = match.group(1).strip()
            if tag:
                tags_set.add(tag)

    return sorted(tags_set)


def canonicalize_title(title_or_path: str) -> str:
    """Normalize a note title or relative path (e.g. 'folder/Note.md' -> 'Note')."""
    base = os.path.basename(title_or_path)
    if base.lower().endswith(".md"):
        base = base[:-3]
    return base.strip()


class VaultIndex:
    """In-memory bidirectional link and tag index for the vault."""

    def __init__(self) -> None:
        self.notes: Dict[str, VaultNote] = {}  # canonical title -> note
        self.path_to_title: Dict[str, str] = {}  # rel_path -> canonical title
        self.forward_links: Dict[str, Set[str]] = {}  # title -> set of target titles
        self.backlinks: Dict[str, Set[str]] = {}  # title -> set of source titles
        self.tag_index: Dict[str, Set[str]] = {}  # tag -> set of titles
        self.aliases: Dict[str, str] = {}  # alias -> canonical title

    def clear(self) -> None:
        self.notes.clear()
        self.path_to_title.clear()
        self.forward_links.clear()
        self.backlinks.clear()
        self.tag_index.clear()
        self.aliases.clear()

    def add_note(self, note: VaultNote) -> None:
        title = note.title
        self.notes[title] = note
        self.path_to_title[note.rel_path] = title

        # Record aliases
        raw_aliases = note.frontmatter.get("aliases") or note.frontmatter.get("alias")
        if raw_aliases:
            if isinstance(raw_aliases, list):
                for a in raw_aliases:
                    self.aliases[str(a).strip()] = title
            elif isinstance(raw_aliases, str):
                for a in raw_aliases.split(","):
                    self.aliases[a.strip()] = title

        # Record tags
        for tag in note.tags:
            self.tag_index.setdefault(tag, set()).add(title)

        # Forward links
        targets = set()
        for link in note.links:
            target_title = canonicalize_title(link.target)
            targets.add(target_title)
            self.backlinks.setdefault(target_title, set()).add(title)
        self.forward_links[title] = targets

    def remove_note(self, title_or_path: str) -> None:
        title = canonicalize_title(title_or_path)
        if title not in self.notes:
            title = self.path_to_title.get(title_or_path, title)

        if title not in self.notes:
            return

        note = self.notes.pop(title)
        self.path_to_title.pop(note.rel_path, None)

        # Remove from tags
        for tag in note.tags:
            if tag in self.tag_index:
                self.tag_index[tag].discard(title)

        # Remove forward links
        if title in self.forward_links:
            for target in self.forward_links.pop(title):
                if target in self.backlinks:
                    self.backlinks[target].discard(title)

        # Remove backlinks
        if title in self.backlinks:
            self.backlinks.pop(title)

        # Remove aliases
        to_del = [a for a, t in self.aliases.items() if t == title]
        for a in to_del:
            self.aliases.pop(a, None)

    def resolve_title(self, query: str) -> Optional[str]:
        """Resolve a note title from exact match, alias, or path."""
        canon = canonicalize_title(query)
        if canon in self.notes:
            return canon
        if query in self.aliases:
            return self.aliases[query]
        if query in self.path_to_title:
            return self.path_to_title[query]

        # Case-insensitive fallback
        query_lower = canon.lower()
        for t in self.notes:
            if t.lower() == query_lower:
                return t
        return None


class VaultManager:
    """Manages local-first Markdown knowledge base, link indexing, search, and graph generation."""

    def __init__(self, vault_dir: Optional[str] = None) -> None:
        if vault_dir:
            self.vault_dir = Path(vault_dir).expanduser().resolve()
        else:
            self.vault_dir = Path(get_hermes_home()) / "vault"
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        self.vault_dir = self.vault_dir.resolve(strict=True)
        self.index = VaultIndex()
        self._lock = threading.RLock()
        self._watch_stop = threading.Event()
        self._watch_thread: Optional[threading.Thread] = None
        self.scan()

    def _relative_parts(self, value: str, *, field_name: str) -> Tuple[str, ...]:
        """Validate path text using both host and Windows grammar.

        Checking Windows grammar even on POSIX prevents a value such as
        ``C:\\outside`` or a UNC path from becoming an innocent-looking local
        filename during tests or remote execution.
        """
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError(f"{field_name} must not be empty")
        value = value.strip()
        win = PureWindowsPath(value)
        if Path(value).is_absolute() or win.is_absolute() or bool(win.drive) or value.startswith(("\\\\", "//")):
            raise ValueError(f"{field_name} must be relative to the vault")
        normalized = value.replace("\\", "/")
        parts = tuple(normalized.split("/"))
        if any(part in ("", ".", "..") for part in parts):
            raise ValueError(f"{field_name} contains an unsafe path component")
        return parts

    def _assert_contained(self, path: Path, *, must_exist: bool = False) -> Path:
        resolved = path.resolve(strict=must_exist)
        try:
            resolved.relative_to(self.vault_dir)
        except (ValueError, OSError) as exc:
            raise ValueError("path escapes the configured vault root") from exc
        return resolved

    def _safe_note_path(self, title: str, subfolder: Optional[str] = None) -> Path:
        clean_title = canonicalize_title(title)
        if (
            not clean_title
            or clean_title in (".", "..")
            or clean_title != title.removesuffix(".md").strip()
            or any(ch in clean_title for ch in "/\\:\x00")
            or any(ord(ch) < 32 for ch in clean_title)
            or clean_title.endswith((" ", "."))
        ):
            raise ValueError("title must be a non-empty filename without path components")
        target_dir = self.vault_dir
        if subfolder is not None:
            target_dir = self.vault_dir.joinpath(*self._relative_parts(subfolder, field_name="subfolder"))
        self._assert_contained(target_dir)
        return target_dir / f"{clean_title}.md"

    def _safe_indexed_path(self, path: str) -> Path:
        candidate = Path(path)
        if candidate.is_symlink():
            raise ValueError("vault notes must not be symbolic links")
        return self._assert_contained(candidate, must_exist=True)

    def _atomic_write(self, file_path: Path, text: str) -> None:
        parent = file_path.parent
        self._assert_contained(parent)
        parent.mkdir(parents=True, exist_ok=True)
        safe_parent = self._assert_contained(parent, must_exist=True)
        if safe_parent != parent.resolve(strict=True):
            raise ValueError("vault directory changed during write")
        if file_path.is_symlink():
            raise ValueError("vault notes must not be symbolic links")
        fd, tmp_name = tempfile.mkstemp(prefix=".hermes-vault-", suffix=".tmp", dir=safe_parent)
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            self._assert_contained(safe_parent, must_exist=True)
            if file_path.is_symlink():
                raise ValueError("vault note changed to a symbolic link during write")
            os.replace(tmp_path, file_path)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def scan(self) -> Dict[str, Any]:
        """Scan all markdown files in the vault and rebuild the index."""
        with self._lock:
            return self._scan_locked()

    def _scan_locked(self) -> Dict[str, Any]:
        self.index.clear()
        count = 0

        for root, dirs, files in os.walk(self.vault_dir, followlinks=False):
            dirs[:] = [name for name in dirs if not (Path(root) / name).is_symlink()]
            for file in files:
                if file.lower().endswith(".md"):
                    full_path = Path(root) / file
                    try:
                        if full_path.is_symlink():
                            continue
                        safe_path = self._assert_contained(full_path, must_exist=True)
                        rel_path = str(safe_path.relative_to(self.vault_dir)).replace("\\", "/")
                        stat = safe_path.stat()
                        raw_text = safe_path.read_text(encoding="utf-8", errors="replace")

                        fm, body = parse_frontmatter_and_content(raw_text)
                        title = fm.get("title") or canonicalize_title(file)
                        links = extract_wikilinks(body)
                        tags = extract_tags(body, fm.get("tags"))

                        note = VaultNote(
                            path=str(safe_path),
                            rel_path=rel_path,
                            title=title,
                            content=raw_text,
                            frontmatter=fm,
                            tags=tags,
                            links=links,
                            mtime=stat.st_mtime,
                            size=stat.st_size,
                        )
                        self.index.add_note(note)
                        count += 1
                    except Exception as e:
                        _log.warning("Failed to scan note %s: %s", full_path, e)

        _log.info("Vault scanned %d notes in %s", count, self.vault_dir)
        return {"notes_count": count, "vault_dir": str(self.vault_dir)}

    def _snapshot(self) -> Tuple[Tuple[str, int, int], ...]:
        entries: List[Tuple[str, int, int]] = []
        for root, dirs, files in os.walk(self.vault_dir, followlinks=False):
            dirs[:] = [name for name in dirs if not (Path(root) / name).is_symlink()]
            for name in files:
                path = Path(root) / name
                if not name.lower().endswith(".md") or path.is_symlink():
                    continue
                try:
                    safe = self._assert_contained(path, must_exist=True)
                    stat = safe.stat()
                    entries.append((str(safe.relative_to(self.vault_dir)), stat.st_mtime_ns, stat.st_size))
                except (OSError, ValueError):
                    continue
        return tuple(sorted(entries))

    def start_watcher(
        self,
        *,
        interval: float = 0.5,
        debounce: float = 0.2,
        on_change: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """Start one bounded polling watcher for external Markdown changes."""
        if interval <= 0 or debounce < 0:
            raise ValueError("watcher interval must be positive and debounce non-negative")
        if self._watch_thread and self._watch_thread.is_alive():
            return
        self._watch_stop.clear()
        initial_snapshot = self._snapshot()

        def watch() -> None:
            previous = initial_snapshot
            pending_since: Optional[float] = None
            while not self._watch_stop.wait(interval):
                current = self._snapshot()
                if current != previous:
                    previous = current
                    pending_since = time.monotonic()
                if pending_since is not None and time.monotonic() - pending_since >= debounce:
                    result = self.scan()
                    pending_since = None
                    if on_change:
                        try:
                            on_change(result)
                        except Exception:
                            _log.exception("Vault watcher callback failed")

        self._watch_thread = threading.Thread(target=watch, name="hermes-vault-watcher", daemon=True)
        self._watch_thread.start()

    def stop_watcher(self, timeout: float = 2.0) -> None:
        self._watch_stop.set()
        thread = self._watch_thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, timeout))
        self._watch_thread = None

    def list_notes(self) -> List[Dict[str, Any]]:
        """List all notes with their metadata."""
        return [
            {
                "title": n.title,
                "rel_path": n.rel_path,
                "tags": n.tags,
                "mtime": n.mtime,
                "size": n.size,
                "links_count": len(n.links),
                "backlinks_count": len(self.index.backlinks.get(n.title, set())),
            }
            for n in sorted(self.index.notes.values(), key=lambda x: x.title.lower())
        ]

    def get_note(self, title_or_path: str) -> Optional[Dict[str, Any]]:
        """Retrieve a note along with its resolved backlinks and forward links."""
        resolved = self.index.resolve_title(title_or_path)
        if not resolved or resolved not in self.index.notes:
            return None

        note = self.index.notes[resolved]
        backlinks = sorted(list(self.index.backlinks.get(resolved, set())))
        forward_links = sorted(list(self.index.forward_links.get(resolved, set())))

        data = note.to_dict()
        data["backlinks"] = backlinks
        data["forward_links"] = forward_links
        return data

    def write_note(
        self,
        title: str,
        content: str,
        tags: Optional[List[str]] = None,
        frontmatter: Optional[Dict[str, Any]] = None,
        subfolder: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create or update a markdown note."""
        clean_title = canonicalize_title(title)
        file_path = self._safe_note_path(title, subfolder)

        # Desktop editing sends the complete source so unknown YAML properties
        # survive a round trip. Treat embedded frontmatter as source metadata,
        # then let explicit structured fields override it; never nest a second
        # frontmatter block into the Markdown body.
        embedded_fm, embedded_body = parse_frontmatter_and_content(content)
        fm = dict(embedded_fm)
        fm.update(frontmatter or {})
        fm["title"] = clean_title
        if tags:
            fm_tags = fm.get("tags") or []
            if isinstance(fm_tags, list):
                combined_tags = sorted(list(set(fm_tags + tags)))
            else:
                combined_tags = tags
            fm["tags"] = combined_tags

        # Format full markdown document
        body = embedded_body.strip()
        if fm:
            fm_str = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True).strip()
            full_text = f"---\n{fm_str}\n---\n\n{body}\n"
        else:
            full_text = f"{body}\n"

        self._atomic_write(file_path, full_text)
        stat = file_path.stat()

        rel_path = str(file_path.relative_to(self.vault_dir)).replace("\\", "/")
        links = extract_wikilinks(body)
        parsed_tags = extract_tags(body, fm.get("tags"))

        note = VaultNote(
            path=str(file_path),
            rel_path=rel_path,
            title=clean_title,
            content=full_text,
            frontmatter=fm,
            tags=parsed_tags,
            links=links,
            mtime=stat.st_mtime,
            size=stat.st_size,
        )

        # Update index
        self.index.remove_note(clean_title)
        self.index.add_note(note)

        res = note.to_dict()
        res["backlinks"] = sorted(list(self.index.backlinks.get(clean_title, set())))
        res["forward_links"] = sorted(list(self.index.forward_links.get(clean_title, set())))
        return res

    def append_note(self, title: str, append_content: str) -> Dict[str, Any]:
        """Append content to an existing note or create it if not found."""
        existing = self.get_note(title)
        if existing:
            current_body = existing["content"]
            updated = f"{current_body.rstrip()}\n\n{append_content.strip()}\n"
            file_path = self._safe_indexed_path(existing["path"])
            self._atomic_write(file_path, updated)
            stat = file_path.stat()

            fm, body = parse_frontmatter_and_content(updated)
            clean_title = existing["title"]
            links = extract_wikilinks(body)
            tags = extract_tags(body, fm.get("tags"))

            note = VaultNote(
                path=existing["path"],
                rel_path=existing["rel_path"],
                title=clean_title,
                content=updated,
                frontmatter=fm,
                tags=tags,
                links=links,
                mtime=stat.st_mtime,
                size=stat.st_size,
            )
            self.index.remove_note(clean_title)
            self.index.add_note(note)
            res = note.to_dict()
            res["backlinks"] = sorted(list(self.index.backlinks.get(clean_title, set())))
            res["forward_links"] = sorted(list(self.index.forward_links.get(clean_title, set())))
            return res
        else:
            return self.write_note(title=title, content=append_content)

    def delete_note(self, title_or_path: str) -> bool:
        """Delete a note from disk and index."""
        resolved = self.index.resolve_title(title_or_path)
        if not resolved or resolved not in self.index.notes:
            return False

        note = self.index.notes[resolved]
        file_path = self._safe_indexed_path(note.path)
        if file_path.exists():
            file_path.unlink()

        self.index.remove_note(resolved)
        return True

    def search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search notes by title, tags, or body content with relevance ranking."""
        tokens = query.lower().split()
        if not tokens:
            return []

        results: List[Tuple[int, VaultNote]] = []

        for note in self.index.notes.values():
            score = 0
            title_lower = note.title.lower()
            content_lower = note.content.lower()

            for token in tokens:
                if token in title_lower:
                    score += 15  # title match weight
                if any(token in t.lower() for t in note.tags):
                    score += 10  # tag match weight
                if token in content_lower:
                    score += 2  # content match weight

            if score > 0:
                results.append((score, note))

        results.sort(key=lambda x: x[0], reverse=True)

        output: List[Dict[str, Any]] = []
        for score, note in results[:limit]:
            output.append(
                {
                    "score": score,
                    "title": note.title,
                    "rel_path": note.rel_path,
                    "tags": note.tags,
                    "preview": note.content[:200].replace("\n", " ").strip(),
                }
            )
        return output

    def get_graph(self) -> Dict[str, Any]:
        """Generate graph nodes and edges for visualization (Obsidian-style / Starmap compatible)."""
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        # Map each note to a node
        for note in self.index.notes.values():
            backlinks_count = len(self.index.backlinks.get(note.title, set()))
            nodes.append(
                {
                    "id": note.title,
                    "label": note.title,
                    "path": note.rel_path,
                    "tags": note.tags,
                    "weight": 1 + backlinks_count,
                    "group": note.tags[0] if note.tags else "general",
                }
            )

        # Edges for all valid wikilinks
        for source_title, targets in self.index.forward_links.items():
            for target_title in targets:
                edges.append(
                    {
                        "source": source_title,
                        "target": target_title,
                    }
                )

        return {"nodes": nodes, "edges": edges}

    def suggest_wikilinks(self, prefix: str = "") -> List[Dict[str, str]]:
        """Suggest available note titles for autocomplete when typing [[."""
        pref = prefix.lower().strip()
        suggestions: List[Dict[str, str]] = []

        for title, note in self.index.notes.items():
            if not pref or pref in title.lower():
                suggestions.append({"title": title, "path": note.rel_path})

        for alias, title in self.index.aliases.items():
            if not pref or pref in alias.lower():
                suggestions.append({"title": title, "alias": alias, "path": self.index.notes[title].rel_path})

        return sorted(suggestions, key=lambda x: x["title"].lower())


_default_vault_manager: Optional[VaultManager] = None
_default_vault_lock = threading.Lock()


def configured_vault_dir() -> Path:
    """Resolve the configured Vault root, falling back to HERMES_HOME/vault."""
    config_path = get_config_path()
    configured: Optional[str] = None
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if isinstance(raw, dict):
            workstation = raw.get("workstation")
            vault = workstation.get("vault") if isinstance(workstation, dict) else None
            value = vault.get("root") if isinstance(vault, dict) else None
            if isinstance(value, str) and value.strip():
                configured = value.strip()
    except FileNotFoundError:
        pass
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Cannot read Vault root from {config_path}: {exc}") from exc

    if configured is None:
        return (Path(get_hermes_home()) / "vault").resolve()
    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        raise ValueError(f"{VAULT_ROOT_CONFIG_KEY} must be an absolute path")
    return candidate.resolve()


def configure_vault_dir(vault_dir: str) -> Path:
    """Persist a custom root in config.yaml and switch the canonical manager."""
    if not isinstance(vault_dir, str) or not vault_dir.strip() or "\x00" in vault_dir:
        raise ValueError("vault root must be a non-empty absolute path")
    candidate = Path(vault_dir.strip()).expanduser()
    if not candidate.is_absolute():
        raise ValueError("vault root must be an absolute path")
    candidate.mkdir(parents=True, exist_ok=True)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("vault root must be a directory")
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_roundtrip_yaml_update(config_path, VAULT_ROOT_CONFIG_KEY, str(resolved))
    replace_default_vault_manager(resolved)
    return resolved


def replace_default_vault_manager(vault_dir: Path) -> VaultManager:
    """Atomically replace the canonical manager and stop its old watcher."""
    global _default_vault_manager
    replacement = VaultManager(str(vault_dir))
    replacement.start_watcher()
    with _default_vault_lock:
        previous = _default_vault_manager
        _default_vault_manager = replacement
    if previous is not None:
        previous.stop_watcher()
    return replacement


def get_default_vault_manager() -> VaultManager:
    """Return the process-wide canonical Vault owner used by tools and UI RPC."""
    global _default_vault_manager
    with _default_vault_lock:
        if _default_vault_manager is None:
            _default_vault_manager = VaultManager(str(configured_vault_dir()))
            _default_vault_manager.start_watcher()
        return _default_vault_manager


def stop_default_vault_manager() -> None:
    global _default_vault_manager
    with _default_vault_lock:
        manager = _default_vault_manager
        _default_vault_manager = None
    if manager is not None:
        manager.stop_watcher()


atexit.register(stop_default_vault_manager)

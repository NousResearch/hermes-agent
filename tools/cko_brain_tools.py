#!/usr/bin/env python3
"""
CKO Brain Tools Module - Git operations, file management, and brain search
for the Cloud-Keeping Agent's /Brain/ knowledge management.

Tools:
  Git:       git_pull, git_push, git_commit
  File:      read_markdown, write_markdown, append_to_file
  Search:    brain_search, find_similar, find_orphan_notes, local_search, build_search_index
  Memory:    calibrate_confidence, merge_notes, age_notes
  Arbiter:   detect_conflicts, resolve_conflict, query_arbitration_history, learn_rules_from_history
  Feedback:  record_vote, recalibrate_from_feedback, feedback_summary
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import logging

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
_DEFAULT_BRAIN_ROOT = Path.home() / ".hermes" / "brain"
_GIT_COMMIT_TEMPLATE = "{author}\n\n{body}"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _run_git(cmd: list[str], cwd: str | None = None, timeout: int = 30) -> dict:
    """Run a git command, return dict with success/stdout/stderr."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or str(_get_brain_root()),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": "Git command timed out", "returncode": -1}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}


def _get_brain_root() -> Path:
    """Return the /Brain/ root directory, respecting HERMES_HOME."""
    from hermes_constants import get_hermes_home
    brain = get_hermes_home() / "brain"
    brain.mkdir(parents=True, exist_ok=True)
    return brain


def _validate_path(path: str) -> tuple[bool, str]:
    """Ensure a path is within brain root. Returns (valid, error_msg)."""
    try:
        brain_root = _get_brain_root().resolve()
        target = (brain_root / path).resolve()
        # Must be under brain_root
        if not str(target).startswith(str(brain_root)):
            return False, f"Path '{path}' is outside the /Brain/ directory"
        return True, ""
    except Exception as e:
        return False, f"Path validation failed: {e}"


# -----------------------------------------------------------------------------
# Tool implementations
# -----------------------------------------------------------------------------

def git_pull(repo_path: str = ".", remote: str = "origin", branch: str = "main") -> str:
    """
    Pull the latest changes from a remote Git repository.

    Args:
        repo_path: Path to the repository (default: '.', i.e., the brain root).
        remote:    Remote name (default: 'origin').
        branch:    Branch name (default: 'main').

    Returns:
        JSON string with success status and output.
    """
    valid, err = _validate_path(repo_path)
    if not valid:
        return json.dumps({"success": False, "error": err})

    full_path = _get_brain_root() / repo_path
    result = _run_git(["git", "pull", remote, branch], cwd=str(full_path))
    return json.dumps(result, ensure_ascii=False)


def git_push(
    repo_path: str = ".",
    remote: str = "origin",
    branch: str = "main",
    message: Optional[str] = None,
) -> str:
    """
    Push local commits to a remote repository.

    Args:
        repo_path: Path to the repository.
        remote:    Remote name.
        branch:    Branch name.
        message:   Optional commit message. If provided, performs add + commit + push.
                   If None, only pushes (assumes commits already exist).

    Returns:
        JSON string with success status and output.
    """
    valid, err = _validate_path(repo_path)
    if not valid:
        return json.dumps({"success": False, "error": err})

    full_path = _get_brain_root() / repo_path

    if message:
        _run_git(["git", "add", "."], cwd=str(full_path))
        commit_result = _run_git(
            ["git", "commit", "-m", message],
            cwd=str(full_path),
        )
        if not commit_result["success"]:
            return json.dumps({
                "success": False,
                "error": commit_result.get("stderr", commit_result.get("stdout", "Commit failed")),
            }, ensure_ascii=False)

    push_result = _run_git(["git", "push", remote, branch], cwd=str(full_path))

    if push_result["success"]:
        return json.dumps({
            "success": True,
            "pushed": True,
            "stdout": push_result.get("stdout", ""),
        }, ensure_ascii=False)

    # Push failed
    stderr = push_result.get("stderr", "")
    if "nothing to commit" in stderr.lower():
        return json.dumps({
            "success": True,
            "pushed": False,
            "message": "Nothing to push — working tree clean.",
        }, ensure_ascii=False)
    if "rejected" in stderr.lower() or "fetch first" in stderr.lower():
        return json.dumps({
            "success": False,
            "pushed": False,
            "error": "Push rejected — remote has changes. Run git_pull first.",
            "stderr": stderr,
        }, ensure_ascii=False)

    return json.dumps({
        "success": False,
        "pushed": False,
        "error": stderr or push_result.get("stdout", "Push failed"),
    }, ensure_ascii=False)


def git_commit(
    repo_path: str = ".",
    message: str = "",
    author: str = "CKO Agent",
    add_all: bool = True,
) -> str:
    """
    Stage and commit changes in a Git repository.

    Args:
        repo_path: Path to the repository.
        message:   Commit message (required).
        author:    Author name for the commit (default: 'CKO Agent').
        add_all:   Whether to stage all changes (default: True).

    Returns:
        JSON string with success status and commit hash.
    """
    if not message or not message.strip():
        return json.dumps({"success": False, "error": "Commit message is required"})

    valid, err = _validate_path(repo_path)
    if not valid:
        return json.dumps({"success": False, "error": err})

    full_path = _get_brain_root() / repo_path

    if add_all:
        _run_git(["git", "add", "."], cwd=str(full_path))
    else:
        _run_git(["git", "add", "-u"], cwd=str(full_path))

    # Set author via environment
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = author
    env["GIT_COMMITTER_NAME"] = author

    result = _run_git(
        ["git", "commit", "-m", message],
        cwd=str(full_path),
    )

    # Get commit hash on success
    if result["success"]:
        hash_result = _run_git(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(full_path),
        )
        return json.dumps({
            "success": True,
            "committed": True,
            "commit_hash": hash_result.get("stdout", "").strip(),
            "stdout": result.get("stdout", ""),
        }, ensure_ascii=False)

    # Commit failed — distinguish "nothing to commit" from real errors
    stderr = result.get("stderr", "")
    if "nothing to commit" in stderr.lower() or "no changes added" in stderr.lower():
        return json.dumps({
            "success": True,
            "committed": False,
            "message": "Nothing to commit — working tree clean.",
        }, ensure_ascii=False)

    return json.dumps({
        "success": False,
        "error": stderr or result.get("stdout", "Unknown error"),
    }, ensure_ascii=False)


def read_markdown(
    path: str,
    offset: int = 1,
    limit: int = 500,
) -> str:
    """
    Read a markdown file from the /Brain/ directory.

    Args:
        path:    Relative path within /Brain/ (e.g., 'MOCs/Daily.md').
        offset:  Line number to start reading from (1-indexed, default: 1).
        limit:   Maximum number of lines to read (default: 500).

    Returns:
        JSON string with file content and metadata.
    """
    valid, err = _validate_path(path)
    if not valid:
        return json.dumps({"success": False, "error": err})

    full_path = _get_brain_root() / path

    if not full_path.exists():
        return json.dumps({"success": False, "error": f"File not found: {path}"})

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        total_lines = len(lines)
        start = max(0, offset - 1)  # Convert to 0-indexed
        end = min(start + limit, total_lines)

        content = "".join(lines[start:end])

        return json.dumps({
            "success": True,
            "content": content,
            "path": path,
            "offset": offset,
            "limit": limit,
            "total_lines": total_lines,
            "has_more": end < total_lines,
        }, ensure_ascii=False)

    except UnicodeDecodeError:
        return json.dumps({"success": False, "error": "File is not valid UTF-8 text"})
    except Exception as e:
        return json.dumps({"success": False, "error": f"Read failed: {e}"})


def write_markdown(
    path: str,
    content: str,
    create_dirs: bool = True,
) -> str:
    """
    Write content to a markdown file in /Brain/.

    Args:
        path:        Relative path within /Brain/.
        content:     The markdown content to write.
        create_dirs: Whether to create parent directories if they don't exist (default: True).

    Returns:
        JSON string with success status and path.
    """
    valid, err = _validate_path(path)
    if not valid:
        return json.dumps({"success": False, "error": err})

    full_path = _get_brain_root() / path

    if create_dirs:
        full_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        return json.dumps({
            "success": True,
            "path": path,
            "bytes_written": len(content.encode("utf-8")),
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"success": False, "error": f"Write failed: {e}"})


def append_to_file(
    path: str,
    content: str,
    create_if_missing: bool = True,
) -> str:
    """
    Append content to a file in /Brain/.

    Args:
        path:            Relative path within /Brain/.
        content:         The content to append.
        create_if_missing: Create the file if it doesn't exist (default: True).

    Returns:
        JSON string with success status and bytes appended.
    """
    valid, err = _validate_path(path)
    if not valid:
        return json.dumps({"success": False, "error": err})

    full_path = _get_brain_root() / path

    if create_if_missing:
        full_path.parent.mkdir(parents=True, exist_ok=True)

    if not full_path.exists():
        if not create_if_missing:
            return json.dumps({"success": False, "error": f"File does not exist: {path}"})

    try:
        with open(full_path, "a", encoding="utf-8") as f:
            f.write(content)

        return json.dumps({
            "success": True,
            "path": path,
            "bytes_appended": len(content.encode("utf-8")),
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"success": False, "error": f"Append failed: {e}"})


def brain_search(
    query: str,
    path: Optional[str] = None,
    limit: int = 20,
    match_context: int = 2,
) -> str:
    """
    Search the /Brain/ directory for files containing a keyword or phrase.

    Uses ripgrep-style regex search. Supports Obsidian [[wikilinks]] and #tags.

    Args:
        query:         The search term or regex pattern.
        path:          Optional sub-path to search within (relative to /Brain/).
                       If None, searches the entire /Brain/.
        limit:         Maximum number of matches to return (default: 20).
        match_context:  Number of lines of context around each match (default: 2).

    Returns:
        JSON string with match results.
    """
    import re

    brain_root = _get_brain_root()
    search_root = brain_root / path if path else brain_root

    if not search_root.exists():
        return json.dumps({"success": False, "error": f"Search path not found: {path or '/Brain/'}"})

    try:
        results = []
        query_lower = query.lower()

        # Walk all markdown files
        for md_file in search_root.rglob("*.md"):
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception:
                continue

            # Search line by line
            for i, line in enumerate(lines):
                if query_lower in line.lower():
                    # Get context lines
                    start = max(0, i - match_context)
                    end = min(len(lines), i + match_context + 1)
                    context = "".join(lines[start:end]).strip()

                    # Calculate relative path
                    rel_path = str(md_file.relative_to(brain_root))

                    results.append({
                        "file": rel_path,
                        "line_number": i + 1,  # 1-indexed
                        "match": line.strip(),
                        "context": context,
                    })

                    if len(results) >= limit:
                        break

            if len(results) >= limit:
                break

        return json.dumps({
            "success": True,
            "query": query,
            "search_path": path or "/Brain/",
            "total_matches": len(results),
            "matches": results,
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"success": False, "error": f"Search failed: {e}"})


def _extract_keywords(text: str) -> set:
    """Extract significant keywords from text (simple TF-IDF-like approach).

    Strategy: tokenize by whitespace first, then for each word/token:
      - English/alphanumeric: keep as-is (len >= 2)
      - CJK single-char or mixed: extract bigrams
    This avoids frontmatter CJK characters forcing bigram mode on English body text.
    """
    import re

    # Remove frontmatter
    text = re.sub(r'^---[\s\S]*?---', '', text)
    # Remove wikilinks and tags
    text = re.sub(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', r'\1', text)
    text = re.sub(r'#[a-zA-Z0-9_-]+', '', text)
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    # Remove punctuation (keep CJK chars intact)
    text = re.sub(r'[^\w\s]', ' ', text)

    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this',
        'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'what', 'which', 'who', 'when', 'where', 'why', 'how', 'not', 'no',
        'yes', 'all', 'any', 'each', 'every', 'some', 'if', 'then', 'else',
        'so', 'than', 'too', 'very', 'just', 'only', 'also', 'now', 'here',
        'there', 'into', 'out', 'up', 'down', 'over', 'under', 'again', 'once',
    }

    tokens = text.split()
    keywords: set[str] = set()

    for token in tokens:
        token_lower = token.lower()
        if token_lower in stopwords:
            continue

        has_cjk = bool(re.search(r'[\u4e00-\u9fff]', token))

        if has_cjk:
            # CJK: extract bigrams from the token
            # (CJK text rarely has spaces so token is the whole phrase)
            clean = re.sub(r'[\s\d_]+', '', token)
            for i in range(len(clean) - 1):
                bigram = clean[i:i + 2]
                if len(bigram) == 2 and bigram not in stopwords:
                    keywords.add(bigram)
        else:
            # English/alphanumeric: keep as-is if long enough
            if len(token) >= 2:
                keywords.add(token_lower)

    return keywords


def _compute_similarity(text1: str, text2: str) -> float:
    """Compute keyword-overlap similarity using query-coverage for short-vs-long docs.

    For short queries (≤10 significant words): measures what fraction of query
    keywords appear in the document (handle the "needle in haystack" problem).

    For longer texts: falls back to standard Jaccard.
    """
    kw1 = _extract_keywords(text1)
    kw2 = _extract_keywords(text2)
    if not kw1 or not kw2:
        return 0.0

    intersection = kw1 & kw2
    union = kw1 | kw2

    # Query-coverage: better for short queries matching long documents
    if len(kw1) <= 10:
        return len(intersection) / len(kw1)  # what fraction of query is found

    # Standard Jaccard for similarly-sized texts
    return len(intersection) / len(union)


def find_similar(
    text: str,
    threshold: float = 0.25,
    path: Optional[str] = None,
    limit: int = 10,
) -> str:
    """
    Find notes in /Brain/ that are similar to the provided text.
    Uses keyword-overlap similarity (TF-IDF-like keyword extraction).

    Args:
        text:      The text to compare against existing notes.
        threshold: Minimum similarity score (0.0-1.0) to include in results.
                   Default 0.25 — notes below this are filtered out.
        path:      Optional sub-path to search within.
        limit:     Maximum number of results to return (default: 10).

    Returns:
        JSON string with similar notes ranked by similarity score.
    """
    brain_root = _get_brain_root()
    search_root = brain_root / path if path else brain_root

    if not search_root.exists():
        return json.dumps({
            "success": False,
            "error": f"Search path not found: {path or '/Brain/'}"
        })

    source_keywords = _extract_keywords(text)

    try:
        results = []

        for md_file in search_root.rglob("*.md"):
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    file_text = f.read()
            except Exception:
                continue

            # Skip very small files (< 50 chars)
            if len(file_text.strip()) < 50:
                continue

            score = _compute_similarity(text, file_text)

            if score >= threshold:
                rel_path = str(md_file.relative_to(brain_root))
                # Extract title from first H1 or filename
                title = ""
                for line in file_text.split('\n'):
                    if line.startswith('# '):
                        title = line[2:].strip()
                        break
                if not title:
                    title = md_file.stem

                results.append({
                    "file": rel_path,
                    "title": title,
                    "score": round(score, 4),
                    "matched_keywords": list(source_keywords & _extract_keywords(file_text)),
                })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:limit]

        return json.dumps({
            "success": True,
            "source_keywords": list(source_keywords),
            "threshold": threshold,
            "total_matches": len(results),
            "matches": results,
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"success": False, "error": f"find_similar failed: {e}"})


def find_orphan_notes(
    path: Optional[str] = None,
    ignore_patterns: Optional[list] = None,
) -> str:
    """
    Find notes in /Brain/ that have no outgoing or incoming links (orphan notes).

    An orphan note is a note that is NOT linked TO from any other note AND
    does NOT link TO any other note (no outlinks AND no backlinks).

    Args:
        path:            Optional sub-path to search within.
        ignore_patterns: List of path substrings to ignore (e.g. ['README', 'index']).
                         Default ignores 'README' and 'index'.

    Returns:
        JSON string with orphan notes and their link status.
    """
    import re

    brain_root = _get_brain_root()
    search_root = brain_root / path if path else brain_root

    if not search_root.exists():
        return json.dumps({
            "success": False,
            "error": f"Search path not found: {path or '/Brain/'}"
        })

    # Default ignore patterns
    if ignore_patterns is None:
        ignore_patterns = ["README", "index", "Heartbeat"]

    # Regex to find [[wikilinks]] in a file
    wikilink_re = re.compile(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]')

    # Collect all files
    all_files: dict[str, set] = {}  # path -> set of linked-to filenames (without extension)
    all_files_set: set = set()

    for md_file in search_root.rglob("*.md"):
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue

        rel_path = str(md_file.relative_to(brain_root))

        # Skip ignored patterns
        if any(pat.lower() in rel_path.lower() for pat in ignore_patterns):
            continue

        # Extract wikilinks
        links = wikilink_re.findall(content)
        # Normalize: strip extension, take basename
        linked_names = {Path(l).stem for l in links}
        all_files[rel_path] = linked_names
        all_files_set.add(rel_path)

    # Build backlink map: for each file, which files link TO it?
    backlink_map: dict[str, set] = {f: set() for f in all_files_set}

    for file_path, links in all_files.items():
        for linked_name in links:
            # Find which file this link points to
            for candidate in all_files_set:
                if Path(candidate).stem == linked_name:
                    backlink_map[candidate].add(file_path)

    # Find orphans: no outgoing links AND no incoming links (backlinks)
    orphans = []
    for file_path in all_files_set:
        outlinks = all_files[file_path]
        backlinks = backlink_map[file_path]

        if not outlinks and not backlinks:
            # Get title
            try:
                with open(brain_root / file_path, "r", encoding="utf-8") as f:
                    first_lines = f.readlines(200)
                title = ""
                for line in first_lines:
                    if line.startswith('# '):
                        title = line[2:].strip()
                        break
                if not title:
                    title = Path(file_path).stem
            except Exception:
                title = Path(file_path).stem

            orphans.append({
                "file": file_path,
                "title": title,
                "outlinks": [],
                "backlinks": [],
            })

    orphans.sort(key=lambda x: x["file"])

    return json.dumps({
        "success": True,
        "search_path": str(path) if path else "/Brain/",
        "total_files_scanned": len(all_files_set),
        "total_orphans": len(orphans),
        "orphans": orphans,
    }, ensure_ascii=False)


# ────────────────────────── Local Search (TF-IDF) ────────────────────────────

_STORE_DIR = None   # lazy init

def _ls_get_store_dir():
    global _STORE_DIR
    if _STORE_DIR is None:
        brain_root = os.environ.get("BRAIN_ROOT", "/root/.hermes/brain")
        _STORE_DIR = os.path.join(brain_root, ".local_search")
        os.makedirs(_STORE_DIR, exist_ok=True)
    return _STORE_DIR


def _ls_preprocess(text: str, expand_dict: bool = False) -> list:
    import re
    text = re.sub(r'^---[\s\S]*?---', '', text)
    text = re.sub(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', r'\1', text)
    text = re.sub(r'#[a-zA-Z0-9_-]+', '', text)
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'&[a-z]+;', ' ', text)
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
    text = text.lower()
    tokens = []
    for token in text.split():
        if re.search(r'[\u4e00-\u9fff]', token):
            clean = re.sub(r'[\s\d_]+', '', token)
            for i in range(len(clean) - 1):
                bigram = clean[i:i + 2]
                if bigram not in _LS_STOPWORDS:
                    tokens.append(bigram)
            if expand_dict:
                # Expand via cross-lingual dict (only the bigram itself, not sub-bigrams)
                expanded = _LS_CROSS_LINGUAL.get(clean, [])
                for ex in expanded:
                    ex_clean = ex.lower()
                    if ex_clean not in _LS_STOPWORDS and len(ex_clean) >= 2:
                        tokens.append(ex_clean)
        elif len(token) >= 2 and token not in _LS_STOPWORDS:
            tokens.append(token)
            if expand_dict:
                expanded = _LS_CROSS_LINGUAL.get(token, [])
                for ex in expanded:
                    ex_clean = ex.lower()
                    if ex_clean not in _LS_STOPWORDS and len(ex_clean) >= 2:
                        tokens.append(ex_clean)
    return tokens


_LS_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does',
    'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'what', 'which', 'who', 'when', 'where', 'why', 'how', 'not', 'no', 'yes', 'all', 'any',
    'each', 'every', 'some', 'if', 'then', 'else', 'so', 'than', 'too', 'very', 'just', 'only',
    'also', 'now', 'here', 'there', 'into', 'out', 'up', 'down', 'over', 'under', 'again', 'once',
    'more', 'other', 'same', 'such', '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
    '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有',
    '看', '好', '自己', '这', '什么', '可以', '可能', '已经', '或者', '如果',
}

# ── Cross-lingual translation dict (CN ↔ EN technical terms) ──
_LS_CROSS_LINGUAL = {
    # OS / 操作系统
    '注册表':        ['registry', 'regedit', 'reg', 'windows registry'],
    'windows注册表': ['windows registry', 'registry', 'regedit', 'reg'],
    'macos':         ['mac os', 'macos', '苹果系统', '苹果'],
    '苹果':          ['macos', 'mac os', 'apple', '苹果电脑', 'macbook'],
    'linux':         ['linux', 'linux系统', '发行版', 'ubuntu', 'debian'],
    'windows':       ['windows', 'win系统', '视窗', 'win11', 'win10'],
    'ubuntu':        ['ubuntu', 'linux发行版', 'debian系'],
    '安卓':          ['android', '安卓系统', '安卓手机'],
    'android':       ['android', '安卓', '安卓系统'],

    # Development / 开发
    '代码':          ['code', 'coding', '源代码', 'source code'],
    '函数':          ['function', 'method', 'func', '函数'],
    '变量':          ['variable', 'var', '变量'],
    '数组':          ['array', 'list', '数组', '列表'],
    '对象':          ['object', 'obj', '对象'],
    '类':            ['class', 'oop', '类'],
    '接口':          ['interface', '接口', 'api'],
    'api':           ['api', 'application programming interface', '接口', '应用程序接口'],
    'sdk':           ['sdk', 'software development kit', '开发包', '工具包'],
    'ide':           ['ide', '集成开发环境', '编辑器', 'vscode', 'pycharm'],
    '编辑器':        ['editor', 'ide', '编辑器', 'vscode', 'vim', 'nano'],
    '编译器':        ['compiler', '编译', '编译工具'],
    '调试':          ['debug', '调试', 'debugger', '断点'],
    '调试器':        ['debugger', 'debug', '调试工具'],
    '单元测试':      ['unit test', 'unittest', '测试', 'testing'],
    '测试':          ['test', 'testing', '测试', '单元测试'],
    '重构':          ['refactor', '重构', '代码重构'],
    '算法':          ['algorithm', '算法', '排序', 'search'],
    '数据结构':       ['data structure', '数据结构', 'tree', 'graph', '图'],
    '版本控制':       ['version control', 'git', 'svn', '版本管理'],
    'git':           ['git', '版本控制', 'github', 'git仓库'],
    'github':        ['github', 'git仓库', 'git', '远程仓库'],
    'docker':        ['docker', '容器', 'container', '镜像'],
    '容器':          ['container', 'docker', '镜像', 'containerization'],
    'kubernetes':    ['kubernetes', 'k8s', '容器编排', 'k8s'],
    'k8s':           ['k8s', 'kubernetes', '容器编排', '容器集群'],
    'ci':            ['ci', 'continuous integration', '持续集成', 'jenkins'],
    'cd':            ['cd', 'continuous deployment', '持续部署', '部署流水线'],
    '持续集成':       ['ci', 'continuous integration', 'jenkins', 'github actions'],
    '持续部署':       ['cd', 'continuous deployment', '部署', '流水线'],

    # Data / 数据
    '数据库':        ['database', 'db', '数据库', 'sql', 'nosql'],
    '数据库表':       ['database table', '表', 'table', '数据表'],
    'sql':           ['sql', '数据库查询', '数据库', 'mysql', 'postgresql'],
    'nosql':         ['nosql', '非关系数据库', 'mongodb', 'redis', '文档数据库'],
    'mysql':         ['mysql', '数据库', '关系数据库', 'sql数据库'],
    'mongodb':       ['mongodb', '文档数据库', 'nosql', '数据库'],
    'redis':         ['redis', '缓存', '键值数据库', 'nosql', '内存数据库'],
    '缓存':          ['cache', '缓存', 'redis', 'memcached'],
    '索引':          ['index', '索引', 'database index', 'search index'],
    '搜索引擎':       ['search engine', '搜索引擎', 'elasticsearch', '全文搜索'],
    'elasticsearch': ['elasticsearch', 'es', '搜索引擎', '全文搜索'],
    '爬虫':          ['crawler', 'spider', '爬虫', 'web scraping'],
    '数据挖掘':       ['data mining', '数据挖掘', 'analytics', '数据分析'],
    '数据分析':       ['data analysis', '数据分析', 'analytics', '数据挖掘'],
    '机器学习':       ['machine learning', 'ml', '机器学习', '深度学习'],
    '深度学习':       ['deep learning', 'dl', '深度学习', 'neural network', '神经网络'],
    '神经网络':       ['neural network', '深度学习', 'deep learning', 'nn'],
    '大模型':         ['llm', 'large language model', '大语言模型', '语言模型', 'gpt'],
    'llm':           ['llm', 'large language model', '大语言模型', '语言模型', 'gpt', '大模型'],
    'embedding':     ['embedding', '向量嵌入', '词向量', 'text embedding'],
    '向量':          ['vector', 'embedding', '向量', '向量数据库'],
    '向量数据库':     ['vector database', '向量数据库', 'pinecone', 'milvus', 'faiss'],
    '知识库':        ['knowledge base', '知识库', 'knowledge graph', '知识图谱'],
    '知识图谱':       ['knowledge graph', '知识图谱', '知识库', 'ontology'],
    'rag':           ['rag', 'retrieval augmented generation', '检索增强生成', 'rag系统'],

    # Config / 配置
    '配置文件':       ['config file', 'configuration', '配置文件', 'ini', 'yaml', 'toml', 'json配置'],
    '环境变量':       ['environment variable', 'env', '环境变量', '环境配置'],
    '路径':          ['path', '路径', 'filepath', '文件路径', 'directory'],
    '目录':          ['directory', 'folder', '目录', '文件夹', 'path'],
    '文件夹':        ['folder', 'directory', '文件夹', '目录'],
    '文件':          ['file', '文件', '文件管理'],
    '日志':          ['log', '日志', 'logging', '日志文件'],
    '日志文件':       ['log file', '日志', 'logging', '日志输出'],
    '权限':          ['permission', '权限', 'authorization', 'access control'],
    '端口':          ['port', '端口', '网络端口', 'socket'],
    '代理':          ['proxy', '代理', '反向代理', 'reverse proxy', 'forward proxy'],
    '反向代理':       ['reverse proxy', '反向代理', 'nginx', '代理服务器'],
    'nginx':         ['nginx', '反向代理', 'web服务器', '代理服务器'],
    'ssl':           ['ssl', 'tls', '证书', 'https', 'ssl证书', '安全证书'],
    '证书':          ['certificate', 'ssl', 'tls', 'ssl证书', '安全证书'],
    'https':         ['https', 'ssl', 'tls', '安全连接', '加密连接'],
    '域名':          ['domain', '域名', 'dns', '域名解析'],
    'dns':           ['dns', '域名解析', '域名系统', 'nameserver'],
    '防火墙':        ['firewall', '防火墙', 'iptables', 'ufw', '安全组'],
    '监控':          ['monitoring', '监控', 'metrics', '指标监控'],
    '告警':          ['alert', 'alerting', '告警', '报警', 'notification'],
    '备份':          ['backup', '备份', '数据备份', '灾难恢复'],
    '恢复':          ['recovery', 'restore', '恢复', '数据恢复', '容灾'],

    # Network / 网络
    'ip地址':        ['ip address', 'ip', 'ip地址', 'ipv4', 'ipv6'],
    'ip':            ['ip', 'ip地址', 'ipv4', 'ipv6', '网络协议'],
    'mac地址':       ['mac address', 'mac', '物理地址', '网卡地址'],
    '网关':          ['gateway', '网关', '路由器', 'router'],
    '路由器':        ['router', '路由器', '网关', 'router'],
    '子网掩码':       ['subnet mask', '子网掩码', 'netmask', '子网'],
    'vpn':           ['vpn', '虚拟专用网络', '翻墙', '梯子'],
    'http':          ['http', '超文本传输协议', '网络协议', 'https'],
    'websocket':     ['websocket', 'ws', '长连接', '双向通信'],
    'tcp':           ['tcp', '传输控制协议', '网络协议', 'udp'],
    'udp':           ['udp', '用户数据报协议', '网络协议', 'tcp'],
    '端口号':        ['port number', '端口号', '端口', '网络端口'],
    '局域网':        ['lan', '局域网', 'local network', '内网'],
    '广域网':        ['wan', '广域网', 'wide area network', '外网'],
    '内网':          ['intranet', '内网', '局域网', '私有网络'],
    '外网':          ['internet', '外网', '公网', '互联网'],
    'api调用':       ['api call', 'api invocation', 'api调用', '接口调用'],
    '网络请求':       ['network request', 'http request', '网络请求', 'api请求'],
    '请求':          ['request', 'http request', '网络请求', '请求'],
    '响应':          ['response', 'http response', '响应', '网络响应'],
    'webhook':       ['webhook', '回调', 'web钩子', 'http回调'],

    # Security / 安全
    '安全':          ['security', '安全', '信息安全的', 'cybersecurity'],
    '加密':          ['encryption', '加密', 'cryptography', '密码学'],
    '解密':          ['decryption', '解密', '解密', 'decode'],
    '密码':          ['password', '密码', 'credential', '凭据'],
    '认证':          ['authentication', 'auth', '认证', '身份验证'],
    '授权':          ['authorization', 'authorization', '授权', '权限管理'],
    'token':         ['token', '令牌', 'access token', 'access token', '刷新令牌'],
    'oauth':         ['oauth', 'oauth2', '开放授权', '第三方认证'],
    'jwt':           ['jwt', 'json web token', '令牌', 'token'],
    'cors':          ['cors', '跨域', 'cross-origin', '跨域资源共享'],
    '注入':          ['injection', '注入', 'sql injection', '代码注入'],
    'xss':           ['xss', '跨站脚本', 'cross-site scripting', '前端安全'],
    'csrf':          ['csrf', '跨站请求伪造', 'cross-site request forgery', '安全'],
    '渗透测试':       ['penetration test', 'pentest', '渗透测试', '安全测试'],
    '漏洞扫描':       ['vulnerability scan', '漏洞扫描', 'security scan'],

    # Cloud / 云
    '云服务器':       ['cloud server', '云主机', 'ecs', '云服务实例'],
    'ecs':           ['ecs', 'elastic compute service', '云服务器', '云主机'],
    '云存储':         ['cloud storage', '对象存储', 'oss', 's3', '云存储'],
    'oss':           ['oss', 'object storage service', '对象存储', '云存储'],
    's3':            ['s3', 'simple storage service', '对象存储', 'aws s3'],
    'cdn':           ['cdn', '内容分发网络', '内容分发', '加速'],
    '负载均衡':       ['load balancer', 'lb', '负载均衡', '负载均衡器'],
    '自动扩缩容':     ['autoscaling', 'auto scale', '自动扩缩容', '弹性伸缩'],
    '弹性伸缩':       ['elasticity', 'autoscaling', '自动扩缩容', '弹性'],

    # Tools / 工具
    '命令行':        ['command line', 'cli', '命令行', '终端', 'terminal'],
    '终端':          ['terminal', '终端', '命令行', 'cli', 'shell'],
    'shell':         ['shell', 'bash', 'zsh', '终端', '命令行'],
    'bash':          ['bash', 'shell脚本', 'bash脚本', 'shell'],
    'powershell':    ['powershell', 'ps', 'windows powershell', '脚本'],
    'vim':           ['vim', 'vi', '编辑器', '终端编辑器', '文本编辑器'],
    'gitbash':       ['git bash', 'bash', 'windows git', '命令行'],
    'tmux':          ['tmux', '终端多路复用', '会话管理', '分屏'],
    '正则表达式':     ['regex', 'regular expression', '正则', '正则表达式'],
    '正则':          ['regex', '正则表达式', 'regular expression', 'pattern'],

    # Package / 包管理
    'pip':           ['pip', 'python包管理器', 'python pip', '包管理'],
    'npm':           ['npm', 'node包管理器', 'node package manager', '包管理'],
    'yarn':          ['yarn', 'npm替代', 'node包管理', '包管理'],
    'conda':         ['conda', '环境管理器', 'python环境', 'conda环境'],
    'venv':          ['venv', 'virtualenv', 'python虚拟环境', '虚拟环境'],
    '依赖':          ['dependency', 'dependencies', '依赖', '包依赖'],
    '包':            ['package', '包', 'npm包', 'pip包', '依赖包'],
    '镜像源':        ['mirror', '源', '镜像', 'pip镜像', 'npm镜像', '仓库'],
    '仓库':          ['repository', 'repo', '仓库', '代码仓库', 'registry'],

    # AI/LLM specific
    '提示词':        ['prompt', '提示词', 'prompt engineering', '提示词工程'],
    '提示词工程':     ['prompt engineering', '提示词工程', 'prompt', '提示词优化'],
    '思维链':        ['chain of thought', 'cot', '思维链', '推理链'],
    'few-shot':      ['few-shot', 'few shot', '少样本', '小样本学习'],
    'zero-shot':     ['zero-shot', 'zero shot', '零样本', '零样本学习'],
    '微调':          ['fine-tuning', '微调', '模型微调', 'fine-tune'],
    'fine-tuning':   ['fine-tuning', '微调', '模型微调', 'fine-tune'],
    '训练':          ['training', '训练', '模型训练', 'train'],
    '推理':          ['inference', '推理', '模型推理', 'infer'],
    '模型蒸馏':       ['distillation', '模型蒸馏', '知识蒸馏', 'distill'],
    '量化':          ['quantization', '量化', '模型量化', 'quantize'],
    '剪枝':          ['pruning', '剪枝', '模型剪枝', 'network pruning'],
    'loss函数':      ['loss function', 'loss', '损失函数', 'objective function'],
    '梯度下降':       ['gradient descent', '梯度下降', '优化算法', 'optimizer'],
    '优化器':         ['optimizer', '优化器', 'adam', 'sgd', '梯度下降'],
    '超参数':        ['hyperparameter', '超参数', 'hyper params'],
    'epoch':         ['epoch', '轮次', '训练轮次', 'iteration'],
    'batch':         ['batch', '批次', 'batch size', '批量'],
    'token计费':     ['token billing', 'token计算', 'token计费', '用量计费'],
    '上下文长度':     ['context length', '上下文长度', 'max tokens', 'context window'],
    'context窗口':   ['context window', '上下文窗口', '上下文长度', 'context length'],

    # Brain / CKO
    '第二大脑':       ['second brain', '第二大脑', 'pkm', 'personal knowledge management'],
    '知识管理':       ['knowledge management', '知识管理', 'pkm', '知识库'],
    '笔记':          ['note', '笔记', 'notes', 'knowledge note'],
    '笔记系统':       ['note-taking system', '笔记系统', 'obsidian', '双链笔记'],
    '双链':          ['bidirectional link', '双链', 'obsidian', 'link'],
    'obsidian':      ['obsidian', '双链笔记', '笔记软件', 'md编辑器'],
    'zettelkasten':  ['zettelkasten', '卡片盒', '笔记法', '知识管理'],
    '定期回顾':       ['spaced repetition', '定期回顾', '记忆曲线', '艾宾浩斯'],
    '记忆曲线':       ['spaced repetition', '记忆曲线', '艾宾浩斯遗忘曲线', '定期回顾'],
    '记忆宫殿':       ['memory palace', '记忆宫殿', ' loci method', '记忆法'],
    '增量学习':       ['incremental learning', '增量学习', '持续学习', '在线学习'],
    '知识蒸馏':       ['knowledge distillation', '知识蒸馏', '模型蒸馏', 'distillation'],
    '跨平台':        ['cross-platform', '跨平台', '跨系统', '多平台'],
    '多设备同步':     ['multi-device sync', '多设备同步', '设备同步', '跨设备'],
    '代理':          ['proxy', '代理', '梯子', 'vpn', '网络代理'],
    'tailscale':     ['tailscale', '组网', '异地组网', 'mesh vpn', '零信任组网'],
    '内网穿透':       ['内网穿透', 'nat traversal', 'frp', 'ngrok', '穿透'],
    'frp':           ['frp', '内网穿透', 'nat traversal', '反向代理穿透'],
    'ngrok':         ['ngrok', '内网穿透', '隧道', '公网穿透'],

    # Memory / 记忆
    '上下文':        ['context', '上下文', 'context window', '上下文窗口'],
    '工作记忆':       ['working memory', '工作记忆', 'short-term memory', '短期记忆'],
    '长期记忆':       ['long-term memory', '长期记忆', 'persistent memory', '持久记忆'],
    '情景记忆':       ['episodic memory', '情景记忆', '记忆', '个人经历记忆'],
    '语义记忆':       ['semantic memory', '语义记忆', '知识记忆', '概念记忆'],
    '遗忘':          ['forgetting', '遗忘', '记忆衰退', 'aging'],
    '遗忘曲线':       ['forgetting curve', '遗忘曲线', '艾宾浩斯', '记忆衰退'],
    '元认知':        ['metacognition', '元认知', '认知监控', '自我认知'],
    '认知偏差':       ['cognitive bias', '认知偏差', '偏见', '思维偏差'],
    '确认偏差':       ['confirmation bias', '确认偏差', '认知偏差', '偏见'],
    '锚定效应':       ['anchoring bias', '锚定效应', '认知偏差', '心理锚定'],
    '可获得性启发':   ['availability heuristic', '可获得性启发', '认知启发', '判断偏差'],
    '后见之明偏差':   ['hindsight bias', '后见之明偏差', '认知偏差', '马后炮'],
    '过度自信效应':   ['overconfidence effect', '过度自信', '认知偏差', '自信偏差'],
    '计划谬误':       ['planning fallacy', '计划谬误', '低估耗时', '认知偏差'],
    '框架效应':       ['framing effect', '框架效应', '认知偏差', '表述偏差'],
    '损失规避':       ['loss aversion', '损失规避', '认知偏差', '损失厌恶'],
    '沉没成本谬误':    ['sunk cost fallacy', '沉没成本', '认知偏差', '成本谬误'],

    # Thinking / 思维
    '批判性思维':     ['critical thinking', '批判性思维', '分析思维', '逻辑思维'],
    '系统性思维':     ['systems thinking', '系统性思维', '系统思考', '整体思维'],
    '设计思维':       ['design thinking', '设计思维', '以人为本设计', '创新思维'],
    '第一性原理':     ['first principles', '第一性原理', '物理思维', '根本原理'],
    '类比思维':       ['analogical thinking', '类比思维', '类比推理', '比喻思维'],
    '逆向思维':       ['reverse thinking', '逆向思维', '反推法', '倒推法'],
    '发散思维':       ['divergent thinking', '发散思维', '头脑风暴', '创意思维'],
    '收敛思维':       ['convergent thinking', '收敛思维', '聚合思维', '逻辑收敛'],
    '水平思维':       ['lateral thinking', '水平思维', '横向思维', '创新思维'],
    '垂直思维':       ['vertical thinking', '垂直思维', '纵向思维', '深度思考'],
    '元认知监控':     ['metacognitive monitoring', '元认知监控', '自我监控', '思维监控'],
    '思维模型':       ['mental model', '思维模型', '心智模型', '认知框架'],
    '认知框架':       ['cognitive framework', '认知框架', '心智模型', '思维框架'],
    '心理模型':       ['mental model', '心理模型', '心智模型', '思维模型'],

    # Productivity / 效率
    '番茄工作法':     ['pomodoro technique', '番茄工作法', '时间管理', '专注'],
    '番茄钟':        ['pomodoro', '番茄钟', '专注计时', '时间块'],
    '时间管理':       ['time management', '时间管理', '效率', 'gtd', 'getting things done'],
    'gtd':           ['gtd', 'getting things done', '时间管理', '任务管理'],
    '优先级':        ['priority', '优先级', '重要紧急', '任务优先级'],
    '重要紧急':       ['eisenhower matrix', '重要紧急', '优先级', '四象限'],
    '四象限':        ['eisenhower matrix', '四象限', '重要紧急', '优先级矩阵'],
    '深度工作':       ['deep work', '深度工作', '专注工作', '心流'],
    '心流':          ['flow state', '心流', '心流状态', '深度专注'],
    '多任务处理':     ['multitasking', '多任务处理', '任务切换', '并行处理'],
    '委托':          ['delegation', '委托', '任务委托', '授权'],
    '自动化':         ['automation', '自动化', '工作流自动化', '流程自动化'],
    '模板':          ['template', '模板', '文档模板', '工作模板'],
    'checklist':     ['checklist', '清单', '检查清单', '任务清单'],
    'sop':           ['sop', 'standard operating procedure', '标准作业程序', '流程文档'],
    '流程':          ['process', '流程', '工作流', 'workflow', '步骤'],
    '工作流':        ['workflow', '工作流', '流程', 'pipeline', '流水线'],
    '流水线':        ['pipeline', '流水线', '工作流', 'ci cd pipeline', '流程'],

    # Communication / 沟通
    '异步沟通':       ['async communication', '异步沟通', '非同步', '消息异步'],
    '同步沟通':       ['sync communication', '同步沟通', '实时沟通', '会议'],
    '结构化沟通':     ['structured communication', '结构化沟通', '沟通框架', '金字塔原理'],
    '金字塔原理':     ['pyramid principle', '金字塔原理', '结构化沟通', '结论先行'],
    '电梯演讲':       ['elevator pitch', '电梯演讲', '电梯法则', '简报'],
    '反馈':          ['feedback', '反馈', '意见反馈', '回复'],
    '主动反馈':       ['proactive feedback', '主动反馈', '反馈', '定期反馈'],
    '绩效评估':       ['performance review', '绩效评估', '绩效面谈', '考核'],
    '1on1':          ['1-on-1', '一对一会议', 'one on one', '绩效沟通'],
    '周报':          ['weekly report', '周报', '周总结', '工作周报'],
    '月报':          ['monthly report', '月报', '月总结', '工作月报'],
    '站会':          ['standup', '站会', '每日站会', 'scrum standup'],
    '复盘':          ['retrospective', '复盘', '项目复盘', '总结反思'],
    '项目复盘':       ['project retrospective', '项目复盘', '复盘', '事后分析'],
    '知识分享':       ['knowledge sharing', '知识分享', '技术分享', '内部分享'],
    '技术分享':       ['tech talk', '技术分享', '知识分享', '技术演讲'],
    '文档':          ['documentation', '文档', 'docs', '技术文档'],
    '技术文档':       ['technical documentation', '技术文档', '文档', '开发文档'],
    'api文档':        ['api documentation', 'api文档', '接口文档', 'api docs'],
    'readme':        ['readme', 'readme文档', '项目说明', '项目文档'],
    'changelog':     ['changelog', '更新日志', '变更日志', '版本记录'],
    '协议':          ['protocol', '协议', '通信协议', '网络协议', '规则'],
    '规范':          ['specification', 'spec', '规范', '约定', '规格'],
    '标准':          ['standard', '标准', '规范', '规格', 'standard'],
    '约定':          ['convention', '约定', '规范', '惯例', 'convention'],
    '命名规范':       ['naming convention', '命名规范', '命名约定', '变量命名'],
    '代码审查':       ['code review', '代码审查', 'cr', 'review', '代码评审'],
    'pair编程':      ['pair programming', 'pair编程', '结对编程', '协作编程'],
    '结对编程':       ['pair programming', '结对编程', 'pair编程', '协作编程'],
    'mob编程':       ['mob programming', 'mob编程', '群体编程', '团队协作'],
    '技术债务':       ['technical debt', '技术债务', '债务', '代码债务'],
    '代码质量':       ['code quality', '代码质量', '质量', '代码整洁度'],
    '可维护性':       ['maintainability', '可维护性', '维护性', '代码可维护性'],
    '可读性':        ['readability', '可读性', '代码可读性', '阅读性'],
    '可扩展性':       ['scalability', '可扩展性', '扩展性', '系统可扩展性'],
    '鲁棒性':        ['robustness', '鲁棒性', '健壮性', '容错性'],
    '容错':          ['fault tolerance', '容错', '容错性', '高可用'],
    '高可用':        ['high availability', 'ha', '高可用', '可用性', 'ha系统'],
    '灾备':          ['disaster recovery', '灾备', '容灾', '业务连续性'],
    '业务连续性':     ['business continuity', '业务连续性', 'bc', '灾备', '容灾'],
    'sla':           ['sla', 'service level agreement', '服务级别协议', '服务水平'],
    '监控':          ['monitoring', '监控', '系统监控', '应用监控'],
    '可观测性':       ['observability', '可观测性', '监控', 'metrics', 'traces', 'logs'],
    'metrics':       ['metrics', '指标', '度量', '可观测性', '性能指标'],
    'tracing':       ['tracing', '链路追踪', '分布式追踪', 'trace'],
    '日志':          ['logging', '日志', '日志记录', 'log'],
    '告警':          ['alerting', '告警', '报警', 'alert', 'notification'],
    'incident':      ['incident', '事故', '故障', 'incident管理', '事件'],
    'incident管理':   ['incident management', 'incident管理', '事故管理', '故障响应'],
    '值班':          ['on-call', '值班', 'oncall', '运维值班'],
    '变更管理':       ['change management', '变更管理', 'change control', '变更流程'],
    '发布':          ['release', '发布', '上线', 'deployment', '部署'],
    '灰度发布':       ['canary release', '灰度发布', '金丝雀发布', '渐进式发布'],
    '回滚':          ['rollback', '回滚', '版本回退', '回退'],
    '热更新':        ['hot reload', '热更新', '热部署', '在线更新'],
    '冷更新':        ['cold update', '冷更新', '离线更新', '重启更新'],
    '蓝绿部署':       ['blue-green deployment', '蓝绿部署', '部署策略', '零停机部署'],
    'ab测试':        ['a/b testing', 'ab测试', 'ab test', '对比测试', '分桶测试'],
    '特性开关':       ['feature flag', '特性开关', 'feature toggle', '功能开关'],
    '金丝雀':        ['canary', '金丝雀', '灰度', 'canary release', '金丝雀发布'],
    '冒烟测试':       ['smoke test', '冒烟测试', '冒烟', '快速测试'],
    '回归测试':       ['regression test', '回归测试', '回归', '功能回归'],
    '集成测试':       ['integration test', '集成测试', '集成', '系统集成测试'],
    '端到端测试':     ['e2e test', 'end-to-end test', '端到端测试', 'e2e'],
    '压测':          ['load test', '压测', '压力测试', '性能测试', 'stress test'],
    '性能测试':       ['performance test', '性能测试', '压测', '基准测试', 'benchmark'],
    '基准测试':       ['benchmark', '基准测试', '性能基准', '性能测试'],
    '渗透测试':       ['penetration test', '渗透测试', 'pentest', '安全测试'],
    '混沌工程':       ['chaos engineering', '混沌工程', '故障注入', '混沌测试'],
}


def _ls_chunk_text(text: str, size: int = 500) -> list:
    import re
    text = re.sub(r'^---[\s\S]*?---', '', text)
    lines = text.split('\n')
    chunks, current, wc = [], [], 0
    for line in lines:
        w = len(line.split())
        if wc + w > size and current:
            chunks.append('\n'.join(current))
            current, wc = [], 0
        current.append(line)
        wc += w
    if current:
        chunks.append('\n'.join(current))
    return [c.strip() for c in chunks if c.strip()]


def _ls_brain_hash() -> str:
    import hashlib
    brain_root = os.environ.get("BRAIN_ROOT", "/root/.hermes/brain")
    mtimes = []
    for root, _, files in os.walk(brain_root):
        if any(x in root for x in [".local_search", ".git", ".Archive"]):
            continue
        for fn in sorted(files):
            if fn.endswith(".md"):
                fp = os.path.join(root, fn)
                mtime = int(os.path.getmtime(fp))
                mtimes.append(str(mtime))
    return hashlib.sha256("".join(mtimes).encode()).hexdigest()[:12]


def _ls_load_manifest():
    manifest_path = os.path.join(_ls_get_store_dir(), "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)
    return {"version": 1, "notes": [], "index_hash": ""}


def _ls_save_manifest(m):
    manifest_path = os.path.join(_ls_get_store_dir(), "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)


def _ls_tfidf_vector(terms: list, doc_freqs: dict, n_docs: int) -> dict:
    from collections import Counter
    import math
    tf = Counter(terms)
    max_tf = max(tf.values()) if tf else 1
    vec = {}
    for term, count in tf.items():
        tf_norm = count / max_tf
        idf = math.log((n_docs + 1) / (doc_freqs.get(term, 0) + 1)) + 1
        vec[term] = tf_norm * idf
    return vec


def _ls_cosine(vec_a: dict, vec_b: dict) -> float:
    import math
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm = math.sqrt(sum(v * v for v in vec_a.values())) * math.sqrt(sum(v * v for v in vec_b.values()))
    return dot / (norm + 1e-10)


def _ls_cosine_vec(a: list, b: list) -> float:
    """Cosine similarity between two raw float vectors."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b + 1e-10)


def _ls_embed(texts: list, model: str = "nomic-embed-text") -> list:
    """Get Ollama embeddings for a list of texts. Returns list of float vectors."""
    import urllib.request, json
    embeddings = []
    for text in texts:
        payload = json.dumps({"model": model, "prompt": text[:8192]}).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                embeddings.append(data.get("embedding", []))
        except Exception:
            embeddings.append([])
    return embeddings


def build_search_index(notes_dir: Optional[str] = None, chunk_size: int = 500) -> str:
    """
    Scan notes in /Brain/, chunk each note, and build a TF-IDF index for local_search.
    Call this after adding or modifying many notes. Results are cached in /.local_search/.

    Args:
        notes_dir:   Directory to index. Defaults to BRAIN_ROOT.
        chunk_size: Max words per chunk (default 500). Short notes = 1 chunk.

    Returns:
        JSON summary with notes_indexed, chunks, elapsed_s.
    """
    import time, re
    t0 = time.time()
    brain_root = os.environ.get("BRAIN_ROOT", "/root/.hermes/brain")
    notes_dir = notes_dir or brain_root

    md_files = []
    for root, _, files in os.walk(notes_dir):
        if any(x in root for x in [".local_search", ".git", ".Archive"]):
            continue
        for fn in sorted(files):
            if fn.endswith(".md"):
                md_files.append(os.path.join(root, fn))

    doc_freqs = {}
    doc_data = []
    for fp in md_files:
        rel = os.path.relpath(fp, brain_root)
        with open(fp) as f:
            content = f.read()
        title = ""
        m = re.match(r'^#\s+(.+)$', content, re.MULTILINE)
        if m:
            title = m.group(1).strip()
        chunks = _ls_chunk_text(content, chunk_size)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{rel}.{i}" if len(chunks) > 1 else rel
            terms = _ls_preprocess(chunk)
            for t in set(terms):
                doc_freqs[t] = doc_freqs.get(t, 0) + 1
            doc_data.append({"path": chunk_id, "title": title or rel, "chunk_text": chunk, "terms": terms})

    n_docs = len(doc_data)
    doc_vectors = [_ls_tfidf_vector(d["terms"], doc_freqs, n_docs) for d in doc_data]

    # Ollama embedding vectors
    emb_path = os.path.join(_ls_get_store_dir(), "ollama_vectors.json")
    try:
        chunk_texts = [d["chunk_text"] for d in doc_data]
        vecs = _ls_embed(chunk_texts)
        # Save as list of lists (raw float vectors)
        with open(emb_path, "w") as f:
            json.dump(vecs, f)
        n_embedded = sum(1 for v in vecs if v)
        print(f"[local_search] Ollama embedded {n_embedded}/{len(vecs)} chunks")
    except Exception as ex:
        print(f"[local_search] Ollama embedding skipped: {ex}")
        with open(emb_path, "w") as f:
            json.dump([[] for _ in doc_data], f)

    # Save index
    idx_path = os.path.join(_ls_get_store_dir(), "tfidf_index.json")
    with open(idx_path, "w") as f:
        json.dump({"doc_freqs": doc_freqs, "doc_vectors": doc_vectors,
                    "doc_data": [{"path": d["path"], "title": d["title"], "chunk_text": d["chunk_text"]}
                                 for d in doc_data]}, f, ensure_ascii=False)

    # Save manifest
    manifest = _ls_load_manifest()
    manifest["index_hash"] = _ls_brain_hash()
    manifest["notes"] = [{"path": d["path"], "title": d["title"]} for d in doc_data]
    manifest["n_chunks"] = len(doc_data)
    manifest["n_notes"] = len(md_files)
    manifest["built_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _ls_save_manifest(manifest)

    return json.dumps({
        "success": True,
        "notes_indexed": len(md_files),
        "chunks": len(doc_data),
        "elapsed_s": round(time.time() - t0, 2),
    }, ensure_ascii=False)


def local_search(query: str, top_k: int = 5, mode: str = "hybrid") -> str:
    """
    Hybrid search over /Brain/ notes — TF-IDF + Ollama vector fusion.

    Pipeline (hybrid mode):
      1. CJK-aware tokenization with cross-lingual expansion
      2. TF-IDF cosine similarity  (weight: 0.6)
      3. Ollama nomic-embed-text cosine similarity (weight: 0.4)
      4. Keyword boost: exact term matches add +0.05/term
      5. Final score = 0.6*tfidf_score + 0.4*emb_score + keyword_boost

    Vector layer requires Ollama running with 'nomic-embed-text' pulled.
    Falls back to TF-IDF only if Ollama is unavailable.

    Args:
        query:    Search query (supports Chinese and English).
        top_k:    Max results to return (default 5).
        mode:     'hybrid' (TF-IDF + Ollama) or 'tfidf' (score only).

    Returns:
        JSON with results [{path, title, snippet, score}, ...].
    """
    import time, math
    t0 = time.time()

    idx_path = os.path.join(_ls_get_store_dir(), "tfidf_index.json")
    manifest = _ls_load_manifest()

    if not os.path.exists(idx_path):
        build_search_index()

    with open(idx_path) as f:
        raw = json.load(f)

    doc_freqs = raw["doc_freqs"]
    doc_vectors = raw["doc_vectors"]
    doc_data = raw["doc_data"]
    n_docs = len(doc_data)

    # Use cross-lingual expansion for richer query representation
    q_terms = _ls_preprocess(query, expand_dict=True)
    if not q_terms:
        return json.dumps({"success": False, "error": "No valid query terms after stopword filtering"}, ensure_ascii=False)

    q_vec_tfidf = _ls_tfidf_vector(q_terms, doc_freqs, n_docs)

    # Load Ollama vectors (pre-computed at index time)
    emb_path = os.path.join(_ls_get_store_dir(), "ollama_vectors.json")
    emb_vectors = []
    emb_available = False
    if os.path.exists(emb_path):
        with open(emb_path) as f:
            emb_vectors = json.load(f)
        emb_available = any(v for v in emb_vectors)

    # Compute Ollama query embedding
    q_emb = None
    if emb_available:
        try:
            q_emb = _ls_embed([query])[0]
        except Exception:
            pass

    scores = []
    for i, vec in enumerate(doc_vectors):
        tfidf_score = _ls_cosine(q_vec_tfidf, vec)
        final_score = tfidf_score

        # Ollama vector component (only if both query and doc have embeddings)
        if mode == "hybrid" and q_emb is not None and i < len(emb_vectors) and emb_vectors[i]:
            doc_emb = emb_vectors[i]
            if doc_emb and len(doc_emb) == len(q_emb):
                emb_score = _ls_cosine_vec(q_emb, doc_emb)
                final_score = 0.6 * tfidf_score + 0.4 * emb_score

        # Keyword boost
        if mode == "hybrid":
            snippet_lower = doc_data[i]["chunk_text"].lower()
            for t in q_terms:
                if len(t) >= 2:
                    final_score += snippet_lower.count(t) * 0.05

        if final_score > 0:
            scores.append((final_score, i))

    scores.sort(key=lambda x: -x[0])
    results = []
    for score, i in scores[:top_k]:
        d = doc_data[i]
        results.append({
            "path": d["path"],
            "title": d.get("title", ""),
            "snippet": d["chunk_text"][:200],
            "score": round(score, 4),
        })

    emb_status = "available" if emb_available else "unavailable"
    return json.dumps({
        "success": True,
        "query": query,
        "mode": mode,
        "results": results,
        "total_indexed": manifest.get("n_chunks", 0),
        "emb_status": emb_status,
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
    }, ensure_ascii=False)


def merge_notes(
    existing_note: str,
    new_content: str,
    merge_type: str = "auto",
    archive_base: str = ".Archive",
) -> str:
    """
    Merge new content into an existing Brain note following the SOP.

    Args:
        existing_note: Relative path within /Brain/ to the existing note.
        new_content:   The new markdown content to merge in.
        merge_type:    One of 'auto', 'facts_merged', 'discrepancy_logged', 'superseded'.
                       'auto' detects based on content comparison.
        archive_base:  Subdirectory under /Brain/ for archived files.

    Returns:
        JSON string with merged content and archive info.
    """
    import re
    from datetime import date

    brain_root = _get_brain_root()
    existing_path = brain_root / existing_note
    today = date.today().isoformat()

    # Validate existing note exists
    if not existing_path.exists():
        return json.dumps({"success": False, "error": f"Existing note not found: {existing_note}"})

    # Parse frontmatter from existing note
    def parse_frontmatter(content: str) -> tuple[dict, str, str]:
        fm_match = re.match(r'^---\n(.*?)\n---\n(.*)$', content, re.DOTALL)
        if fm_match:
            fm_text, body = fm_match.groups()
            lines = fm_text.split('\n')
            fm = {}
            for line in lines:
                if ':' in line:
                    key, val = line.split(':', 1)
                    fm[key.strip()] = val.strip().strip('"').strip("'")
            return fm, body.strip(), fm_text
        return {}, content, ""

    def write_frontmatter(fm: dict) -> str:
        lines = ["---"]
        for k, v in fm.items():
            lines.append(f"{k}: {v}")
        lines.append("---")
        return "\n".join(lines)

    try:
        with open(existing_path, "r", encoding="utf-8") as f:
            existing_content = f.read()
        existing_fm, existing_body, _ = parse_frontmatter(existing_content)
    except Exception as e:
        return json.dumps({"success": False, "error": f"Read existing note failed: {e}"})

    new_fm, new_body, _ = parse_frontmatter(new_content)

    # Determine existing confidence
    existing_conf = float(existing_fm.get("Confidence", "0.0"))
    new_conf = float(new_fm.get("Confidence", new_fm.get("confidence", "0.0")))

    # ---- Merge frontmatter ----
    # Source: merge both
    sources = []
    if "Source" in existing_fm:
        sources.append(existing_fm["Source"])
    if "Source" in new_fm:
        sources.append(new_fm["Source"])
    merged_fm = dict(existing_fm)
    if sources:
        merged_fm["Source"] = ", ".join(dict.fromkeys(sources))  # dedup, preserve order

    # Confidence: higher wins
    merged_fm["Confidence"] = str(max(existing_conf, new_conf))

    # Tags: merge dedup
    existing_tags = set(re.findall(r'#[a-zA-Z0-9_-]+', existing_body + existing_content))
    new_tags = set(re.findall(r'#[a-zA-Z0-9_-]+', new_content))
    merged_tags = existing_tags | new_tags

    # Links: parse [[wikilinks]]
    wikilink_re = re.compile(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]')
    existing_links = set(wikilink_re.findall(existing_content))
    new_links = set(wikilink_re.findall(new_content))
    merged_links = existing_links | new_links

    # ---- Merge body ----
    if merge_type == "auto":
        # Simple heuristic: check if key sentences are shared
        existing_sents = set(re.sub(r'[^\w\s]', '', existing_body.lower()).split())
        new_sents = set(re.sub(r'[^\w\s]', '', new_body.lower()).split())
        overlap = len(existing_sents & new_sents)
        total = len(existing_sents | new_sents)
        jaccard = overlap / total if total > 0 else 0

        # Check for contradiction keywords
        contradiction_indicators = ["but", "however", "actually", "instead", "contradicts", "相反", "不对"]
        has_contradiction = any(ind in new_body.lower() for ind in contradiction_indicators)

        if has_contradiction or jaccard < 0.3:
            detected_type = "discrepancy_logged"
        else:
            detected_type = "facts_merged"
    else:
        detected_type = merge_type

    # Build merged body
    if detected_type == "superseded":
        merged_body = new_body
        note_body = (
            f"\n\n---\n"
            f"## 合并记录\n"
            f"- **合并日期**：{today}\n"
            f"- **来源1**：[{existing_note}]（Confidence: {existing_conf}）\n"
            f"- **来源2**：新内容（Confidence: {new_conf}）\n"
            f"- **合并类型**：superseded（完全替代）\n"
            f"- **备注**：新内容完全替代旧版本，旧版已归档。\n"
        )
    elif detected_type == "discrepancy_logged":
        # Prepend discrepancy section to new content, keep both
        merged_body = (
            f"## 合并记录 — 存在分歧\n\n"
            f"> **注意**：以下内容与已有笔记存在分歧，已并存记录，请人工确认。\n\n"
            f"### 旧版内容（来源：{existing_note}，Confidence: {existing_conf}）\n\n"
            f"{existing_body}\n\n"
            f"---\n\n"
            f"### 新版内容（Confidence: {new_conf}）\n\n"
            f"{new_body}\n\n"
            f"---\n"
            f"- **合并日期**：{today}\n"
            f"- **合并类型**：discrepancy_logged\n"
            f"- **建议**：请 Boss 确认哪个版本准确，或补充证据。\n"
        )
    else:  # facts_merged
        merged_body = (
            f"{existing_body}\n\n"
            f"---\n\n"
            f"## 增量补充（{today}）\n\n"
            f"{new_body}\n\n"
            f"---\n"
            f"- **合并日期**：{today}\n"
            f"- **来源1**：[{existing_note}]（Confidence: {existing_conf}）\n"
            f"- **来源2**：新内容（Confidence: {new_conf}）\n"
            f"- **合并类型**：facts_merged\n"
        )

    # Rebuild full content
    # Update tags and links in merged body if they appear as fields
    merged_content = write_frontmatter(merged_fm) + "\n\n" + merged_body

    # ---- Archive old version ----
    archive_dir = brain_root / archive_base / f"{today}-MERGED_{existing_note.replace('/', '_')}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_file = archive_dir / existing_path.name
    with open(archive_file, "w", encoding="utf-8") as f:
        f.write(existing_content)

    return json.dumps({
        "success": True,
        "merged_note": existing_note,
        "merge_type": detected_type,
        "archive_path": str(archive_dir.relative_to(brain_root)),
        "archived_file": str(archive_file.relative_to(brain_root)),
        "confidence_retained": max(existing_conf, new_conf),
        "merged_content_preview": merged_content[:500],
        "total_tags": len(merged_tags),
        "total_links": len(merged_links),
    }, ensure_ascii=False)


def calibrate_confidence(
    source_type: str,
    content_quality: str,
    evidence_count: int = 1,
    has_caveats: bool = False,
) -> str:
    """
    Calibrate the confidence score for a note based on source reliability.

    Args:
        source_type:    Type of source (see level table).
                       Values: 'direct_verification', 'official_doc', 'multi_source',
                       'single可信_source', 'inference', 'unverified', 'guess'
        content_quality: Assessment of content quality.
                       Values: 'high' (+0.1), 'medium' (0), 'low' (-0.1)
        evidence_count: Number of independent evidence sources (default: 1).
        has_caveats:    Whether the content includes limitations/caveats (default: False).

    Returns:
        JSON string with recommended confidence score and reasoning.
    """
    # Level table
    level_map = {
        "direct_verification":   (1.0, "L1: XiaoHu directly verified/executed"),
        "official_doc":          (0.95, "L2: Authoritative official documentation"),
        "multi_source":          (0.85, "L3: Multiple independent sources agree"),
        "single可信_source":     (0.7, "L4: Single credible source (Boss/reliable Agent)"),
        "inference":             (0.5, "L5: Logical inference/reasonable extrapolation"),
        "unverified":            (0.3, "L6: Single unverified statement"),
        "guess":                 (0.1, "L7: Pure speculation/hearsay"),
    }

    if source_type not in level_map:
        return json.dumps({
            "success": False,
            "error": f"Unknown source_type: '{source_type}'. "
                     f"Valid values: {list(level_map.keys())}"
        })

    base_score, level_desc = level_map[source_type]
    quality_adjustment = {"high": 0.1, "medium": 0.0, "low": -0.1}.get(content_quality, 0.0)
    evidence_bonus = 0.05 if source_type == "multi_source" and evidence_count >= 3 else 0.0
    caveats_bonus = 0.05 if has_caveats else 0.0
    final_score = min(1.0, round(base_score + quality_adjustment + evidence_bonus + caveats_bonus, 2))

    reasoning_parts = [level_desc, f"Base: {base_score}"]
    if quality_adjustment != 0:
        reasoning_parts.append(f"Quality ({content_quality}): {'+' if quality_adjustment > 0 else ''}{quality_adjustment}")
    if evidence_bonus > 0:
        reasoning_parts.append(f"Strong multi-source (≥3): +{evidence_bonus}")
    if caveats_bonus > 0:
        reasoning_parts.append(f"Contains caveats: +{caveats_bonus}")
    reasoning_parts.append(f"Final: {final_score}")

    return json.dumps({
        "success": True,
        "recommended_confidence": final_score,
        "reasoning": " | ".join(reasoning_parts),
    }, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Feedback Loop Tools
# -----------------------------------------------------------------------------

def _get_feedback_dir() -> Path:
    """Return the /.Feedback/ directory under brain root."""
    fb = _get_brain_root() / ".Feedback"
    fb.mkdir(parents=True, exist_ok=True)
    return fb


def _parse_votes_file(path: Path) -> list[dict]:
    """Parse a votes.md file. Returns list of vote dicts."""
    if not path.exists():
        return []
    entries = []
    current_note = None
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("## "):
            current_note = line[3:].strip()
        elif line.startswith("-") and current_note:
            # e.g. `- 👍 2025-01-15 +0.10`
            import re
            m = re.match(r"- (?P<vote>👍|👎)\s+(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<delta>[+-][\d.]+)", line)
            if m:
                entries.append({
                    "note": current_note,
                    "vote": m.group("vote"),
                    "date": m.group("date"),
                    "delta": float(m.group("delta")),
                })
    return entries


def _load_topic_stats() -> dict:
    """Load topic stats from /.Feedback/topic_stats.json."""
    stats_file = _get_feedback_dir() / "topic_stats.json"
    if not stats_file.exists():
        return {}
    try:
        import json
        return json.loads(stats_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_topic_stats(stats: dict) -> None:
    """Save topic stats to /.Feedback/topic_stats.json."""
    import json
    stats_file = _get_feedback_dir() / "topic_stats.json"
    stats_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def record_vote(
    note_path: str,
    vote: str,  # "up" or "down"
    note: bool = True,  # if True: vote affects the note's confidence; if False: affects a topic
    topic: str = "",
) -> str:
    """
    Record a Boss feedback vote for a specific Brain note.

    Args:
        note_path:   Relative path within /Brain/ of the note being voted on.
        vote:        'up' (👍) or 'down' (👎).
        note:        If True, vote applies to the note's Confidence. If False, applies to a topic keyword.
        topic:       Topic keyword (e.g. 'git', 'python') to tag this vote under. Used for
                     topic-level feedback aggregation. Required when note=False.

    Returns:
        JSON string with vote recorded, new confidence (if note=True), and cumulative topic stats.
    """
    import re
    from datetime import date

    if vote not in ("up", "down"):
        return json.dumps({"success": False, "error": "vote must be 'up' or 'down'"})

    brain_root = _get_brain_root()
    target_path = brain_root / note_path
    if not target_path.exists():
        return json.dumps({"success": False, "error": f"Note not found: {note_path}"})

    today = date.today().isoformat()
    delta = 0.1 if vote == "up" else -0.1
    emoji = "👍" if vote == "up" else "👎"

    fb_dir = _get_feedback_dir()
    # Personal votes file (e.g. Boss_votes.md)
    votes_file = fb_dir / "Boss_votes.md"

    # Append to votes file
    section_header = f"## {note_path}"
    new_entry = f"- {emoji} {today} {delta:+.2f}\n"

    if votes_file.exists():
        content = votes_file.read_text(encoding="utf-8")
        if section_header in content:
            # Append under existing section
            lines = content.splitlines()
            for i, line in enumerate(lines):
                if line.strip() == section_header:
                    # Find end of section (next ## or EOF)
                    j = i + 1
                    while j < len(lines) and not lines[j].startswith("## "):
                        j += 1
                    lines.insert(j, new_entry)
                    content = "\n".join(lines)
                    break
        else:
            content = content.rstrip() + f"\n\n{section_header}\n{new_entry}"
    else:
        content = f"# Boss 反馈记录\n\n{section_header}\n{new_entry}"

    votes_file.write_text(content, encoding="utf-8")

    # If voting on a note, update its Confidence in frontmatter
    new_confidence = None
    if note:
        try:
            raw = target_path.read_text(encoding="utf-8")
            fm_match = re.match(r'^---\n(.*?)\n---\n(.*)$', raw, re.DOTALL)
            if fm_match:
                fm_text, body = fm_match.group(1), fm_match.group(2)
                fm_lines = fm_text.split('\n')
                new_fm_lines = []
                conf_updated = False
                for fl in fm_lines:
                    if fl.startswith("Confidence:"):
                        old_val = float(fl.split(":", 1)[1].strip())
                        new_val = round(max(0.1, min(1.0, old_val + delta)), 2)
                        new_fm_lines.append(f"Confidence: {new_val}")
                        new_confidence = new_val
                        conf_updated = True
                    else:
                        new_fm_lines.append(fl)
                if not conf_updated:
                    new_fm_lines.append(f"Confidence: {delta}")
                    new_confidence = delta

                # Append feedback log entry
                log_line = f"- **{emoji}** {today} Boss vote {delta:+.2f} → new confidence {new_confidence}\n"
                new_raw = "---\n" + "\n".join(new_fm_lines) + "\n---\n" + body
                if "## 反馈记录" in body:
                    pass  # handled below
                target_path.write_text(new_raw, encoding="utf-8")
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Vote recorded but Confidence update failed: {e}"
            })

    # Topic aggregation
    topic_key = topic.strip().lower()
    if topic_key:
        stats = _load_topic_stats()
        if topic_key not in stats:
            stats[topic_key] = {"up": 0, "down": 0, "net": 0.0, "notes": []}
        stats[topic_key]["up" if vote == "up" else "down"] += 1
        stats[topic_key]["net"] = stats[topic_key]["up"] - stats[topic_key]["down"]
        if note_path not in stats[topic_key]["notes"]:
            stats[topic_key]["notes"].append(note_path)
        _save_topic_stats(stats)

    result = {
        "success": True,
        "vote_recorded": f"{emoji} {today} {delta:+.2f}",
        "note_path": note_path,
        "vote_type": vote,
    }
    if new_confidence is not None:
        result["new_confidence"] = new_confidence
    if topic_key:
        stats = _load_topic_stats()
        result["topic"] = topic_key
        result["topic_stats"] = stats.get(topic_key, {})

    return json.dumps(result, ensure_ascii=False)


def recalibrate_from_feedback(
    note_path: str,
) -> str:
    """
    Recalibrate a note's Confidence based on all accumulated Boss feedback.

    Reads the /.Feedback/Boss_votes.md file, sums all votes for the given note,
    and applies the cumulative delta to the note's current Confidence.

    Args:
        note_path: Relative path within /Brain/ of the note to recalibrate.

    Returns:
        JSON string with old confidence, accumulated vote delta, and new confidence.
    """
    import re
    from datetime import date

    brain_root = _get_brain_root()
    target_path = brain_root / note_path
    if not target_path.exists():
        return json.dumps({"success": False, "error": f"Note not found: {note_path}"})

    votes_file = _get_feedback_dir() / "Boss_votes.md"
    votes = _parse_votes_file(votes_file)

    # Sum votes for this note
    note_votes = [v for v in votes if v["note"] == note_path]
    total_delta = sum(v["delta"] for v in note_votes)

    # Read current frontmatter confidence
    raw = target_path.read_text(encoding="utf-8")
    fm_match = re.match(r'^---\n(.*?)\n---\n(.*)$', raw, re.DOTALL)
    if not fm_match:
        return json.dumps({"success": False, "error": "No valid frontmatter in note"})

    fm_text, body = fm_match.group(1), fm_match.group(2)
    fm_lines = fm_text.split('\n')
    old_conf = 1.0
    new_fm_lines = []
    for fl in fm_lines:
        if fl.startswith("Confidence:"):
            old_conf = float(fl.split(":", 1)[1].strip())
            new_conf = round(max(0.1, min(1.0, old_conf + total_delta)), 2)
            new_fm_lines.append(f"Confidence: {new_conf}")
        else:
            new_fm_lines.append(fl)

    # Append recalibration log
    today = date.today().isoformat()
    log_block = (
        f"\n## 反馈重校准\n"
        f"- **日期**: {today}\n"
        f"- **历史投票次数**: {len(note_votes)}\n"
        f"- **累计delta**: {total_delta:+.2f}\n"
        f"- **原Confidence**: {old_conf}\n"
        f"- **新Confidence**: {new_conf}\n"
    )

    new_raw = "---\n" + "\n".join(new_fm_lines) + "\n---\n" + body + log_block
    target_path.write_text(new_raw, encoding="utf-8")

    return json.dumps({
        "success": True,
        "note_path": note_path,
        "old_confidence": old_conf,
        "vote_count": len(note_votes),
        "total_delta": round(total_delta, 2),
        "new_confidence": round(max(0.1, min(1.0, old_conf + total_delta)), 2),
    }, ensure_ascii=False)


def feedback_summary(
    topic: str = "",
) -> str:
    """
    Get a summary of all Boss feedback votes.

    Args:
        topic: Optional topic keyword to filter to. If empty, returns all stats.

    Returns:
        JSON string with per-note votes, per-topic aggregates, and topic-level
        confidence adjustments (upvoted topics get +0.05, downvoted -0.05).
    """
    votes_file = _get_feedback_dir() / "Boss_votes.md"
    votes = _parse_votes_file(votes_file)
    topic_stats = _load_topic_stats()

    # Per-note summary
    note_map: dict[str, dict] = {}
    for v in votes:
        np = v["note"]
        if np not in note_map:
            note_map[np] = {"up": 0, "down": 0, "net": 0.0, "votes": []}
        note_map[np]["up" if v["vote"] == "👍" else "down"] += 1
        note_map[np]["net"] = note_map[np]["up"] - note_map[np]["down"]
        note_map[np]["votes"].append({"emoji": v["vote"], "date": v["date"], "delta": v["delta"]})

    # Topic-level confidence adjustment
    topic_adjustments = {}
    for tk, ts in topic_stats.items():
        net = ts["net"]
        if net >= 3:
            topic_adjustments[tk] = 0.05
        elif net <= -3:
            topic_adjustments[tk] = -0.05
        else:
            topic_adjustments[tk] = 0.0

    result = {
        "success": True,
        "total_votes": len(votes),
        "notes_with_feedback": len(note_map),
        "per_note": note_map,
        "topic_stats": topic_stats,
        "topic_confidence_adjustments": topic_adjustments,
    }

    if topic:
        result["topic"] = topic
        result["topic_stats"] = topic_stats.get(topic, {})
        result["topic_confidence_adjustment"] = topic_adjustments.get(topic, 0.0)

    return json.dumps(result, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Conflict Arbitration Tools
# -----------------------------------------------------------------------------

def _get_arbiter_dir() -> Path:
    """Return the /.Arbiter/ directory under brain root."""
    arb = _get_brain_root() / ".Arbiter"
    arb.mkdir(parents=True, exist_ok=True)
    return arb


def _load_arbiter_history() -> list[dict]:
    hist_file = _get_arbiter_dir() / "history.json"
    if not hist_file.exists():
        return []
    try:
        import json
        return json.loads(hist_file.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_arbiter_history(history: list[dict]) -> None:
    import json
    hist_file = _get_arbiter_dir() / "history.json"
    hist_file.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_rules() -> dict:
    rules_file = _get_arbiter_dir() / "rules.json"
    if not rules_file.exists():
        return {}
    try:
        import json
        return json.loads(rules_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_rules(rules: dict) -> None:
    import json
    rules_file = _get_arbiter_dir() / "rules.json"
    rules_file.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")


def detect_conflicts(
    repo_path: str = ".",
    check_remote: bool = False,
) -> str:
    """
    Scan git status in /Brain/ for unmerged changes and potential conflicts.

    Detects:
      - Unmerged (staged/unstaged) files: files with local changes not yet committed
      - Staged conflicts: files in git index with conflict markers
      - Remote-ahead: files that exist locally but are not yet on main branch (potential overwrite risk)
      - Conflicted files: files with conflict markers (<<<<<<<, =======, >>>>>>>)

    Args:
        repo_path:   Relative path within /Brain/ to the repository (default '.').
        check_remote: If True, also check if remote has changes not yet pulled (default False).

    Returns:
        JSON string with conflict and risk report.
    """
    import re

    brain_root = _get_brain_root()
    full_path = brain_root / repo_path

    # 1. Git status --short for uncommitted changes
    status_result = _run_git(["git", "status", "--short"], cwd=str(full_path))
    uncommitted = []
    if status_result["success"]:
        for line in status_result["stdout"].splitlines():
            if line.strip():
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    uncommitted.append({"status": parts[0], "file": parts[1]})

    # 2. Scan for conflict markers in .md files
    conflict_markers_re = re.compile(r'^<<<<<<<|^=======|^>>>>>>>', re.MULTILINE)
    conflicted_files = []
    for md_file in full_path.rglob("*.md"):
        if ".git" in str(md_file) or ".Archive" in str(md_file):
            continue
        try:
            content = md_file.read_text(encoding="utf-8")
            if conflict_markers_re.search(content):
                rel = str(md_file.relative_to(brain_root))
                lines = content.splitlines()
                # Find line numbers of conflict markers
                marker_lines = [i + 1 for i, l in enumerate(lines)
                               if l.startswith(('<<<<<<<', '=======', '>>>>>>>'))]
                conflicted_files.append({
                    "file": rel,
                    "marker_count": len(marker_lines),
                    "marker_lines": marker_lines[:6],  # first 6
                })
        except Exception:
            continue

    # 3. Check for files modified by multiple agents (compare commit timestamps)
    # Strategy: look at git log --name-status for files touched in recent commits
    recent_result = _run_git(
        ["git", "log", "--name-status", "-10", "--format=%H %ae %ai"],
        cwd=str(full_path),
    )
    recent_files = {}
    if recent_result["success"]:
        current_file = None
        for line in recent_result["stdout"].splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0] not in ("A", "M", "D", "R"):
                # This is a commit header line
                pass
            elif len(parts) == 1:
                current_file = parts[0]
            elif len(parts) >= 2 and parts[0] in ("A", "M"):
                mode, fname = parts[0], parts[1]
                if current_file:
                    if fname not in recent_files:
                        recent_files[fname] = []
                    recent_files[fname].append(current_file[:8])

    # Files touched by multiple recent commits (potential conflict points)
    multi_touched = {
        f: commits[:5]
        for f, commits in recent_files.items()
        if len(set(commits)) > 1
    }

    # 4. Check if remote is ahead (unpulled changes)
    remote_ahead = []
    if check_remote:
        fetch_result = _run_git(["git", "fetch"], cwd=str(full_path))
        log_result = _run_git(
            ["git", "log", "--oneline", "HEAD..origin/main", "--"],
            cwd=str(full_path),
        )
        if log_result["success"] and log_result["stdout"].strip():
            remote_ahead = log_result["stdout"].strip().splitlines()

    risk_level = "LOW"
    if conflicted_files:
        risk_level = "HIGH"
    elif uncommitted and len(uncommitted) > 5:
        risk_level = "MEDIUM"
    elif remote_ahead:
        risk_level = "MEDIUM"

    return json.dumps({
        "success": True,
        "repo": str(repo_path),
        "risk_level": risk_level,
        "conflicted_files": conflicted_files,
        "uncommitted_changes": uncommitted,
        "multi_touched_files": multi_touched,
        "remote_ahead_count": len(remote_ahead),
        "recommendation": _make_conflict_recommendation(
            conflicted_files, uncommitted, remote_ahead
        ),
    }, ensure_ascii=False)


def _make_conflict_recommendation(
    conflicted_files: list,
    uncommitted: list,
    remote_ahead: list,
) -> str:
    if conflicted_files:
        return (
            "HIGH RISK: Active merge conflicts detected. "
            "Run resolve_conflict for each conflicted file before pushing."
        )
    if remote_ahead:
        return (
            "MEDIUM RISK: Remote has unpulled changes. "
            "Pull before pushing to avoid merge conflicts."
        )
    if len(uncommitted) > 10:
        return (
            f"MEDIUM RISK: {len(uncommitted)} uncommitted files. "
            "Commit or stash changes before syncing."
        )
    return "LOW RISK: No immediate conflict detected. Safe to push."


def resolve_conflict(
    file_path: str,
    strategy: str = "auto",
    winning_version: str = "local",
    note: str = "",
) -> str:
    """
    Resolve a git merge conflict for a specific file in /Brain/.

    Performs conflict resolution in one of three modes:
      - 'auto'         — Detect and apply the appropriate strategy from rules.json
      - 'manual'       — Use winning_version ('local' or 'remote') directly
      - 'cko_override' — Use the version currently in /Brain/ (last XiaoHu-written version)

    On each resolution, appends an audit record to /.Arbiter/history.json.

    Args:
        file_path:        Relative path within /Brain/ of the conflicted file.
        strategy:         'auto', 'manual', or 'cko_override'. Default 'auto'.
        winning_version:  'local' or 'remote' — only used when strategy='manual'.
        note:             Optional note explaining the resolution rationale.

    Returns:
        JSON string with resolution result and audit record.
    """
    import re
    from datetime import date

    brain_root = _get_brain_root()
    full_path = brain_root / file_path

    if not full_path.exists():
        return json.dumps({"success": False, "error": f"File not found: {file_path}"})

    # Detect conflict markers
    conflict_start = re.compile(r'^<<<<<<<\s', re.MULTILINE)
    conflict_sep = re.compile(r'^=======\s*$', re.MULTILINE)
    conflict_end = re.compile(r'^>>>>>>>\s', re.MULTILINE)

    try:
        content = full_path.read_text(encoding="utf-8")
    except Exception as e:
        return json.dumps({"success": False, "error": f"Read failed: {e}"})

    has_conflicts = (
        conflict_start.search(content) or
        conflict_sep.search(content) or
        conflict_end.search(content)
    )

    if not has_conflicts:
        return json.dumps({
            "success": False,
            "error": f"No conflict markers found in {file_path}. Use resolve_conflict only on files with active merge conflicts."
        })

    # Strategy resolution
    if strategy == "auto":
        rules = _load_rules()
        applied_rule = None
        for pattern, strat in rules.items():
            if pattern in file_path:
                strategy = strat
                applied_rule = pattern
                break

        if not applied_rule:
            # No rule matched — apply decision weight hierarchy:
            # environment_priority (platform-specific) > timestamp_priority (generic)
            # For platform-specific detection: check path patterns
            platform_specific_patterns = ("Agents/Win", "Agents/Mac", "Agents/Linux",
                                          "Agents/Cloud", ".plist", ".reg", ".config")
            is_platform_specific = any(pat in file_path for pat in platform_specific_patterns)

            if is_platform_specific:
                # environment_priority: platform-specific content wins automatically
                strategy = "environment_priority"
            else:
                # timestamp_priority: newer/longer version wins
                strategy = "timestamp_priority"

    # Parse local vs remote sections from conflict markers
    lines = content.splitlines()
    local_lines = []
    remote_lines = []
    phase = "before"  # 'before', 'local', 'remote', 'after'

    for line in lines:
        if line.startswith('<<<<<<<'):
            phase = "local"
            continue
        elif line.startswith('======='):
            phase = "remote"
            continue
        elif line.startswith('>>>>>>>'):
            phase = "after"
            continue
        if phase == "local":
            local_lines.append(line)
        elif phase == "remote":
            remote_lines.append(line)

    local_content = "\n".join(local_lines)
    remote_content = "\n".join(remote_lines)

    # Determine winner content
    if strategy == "manual":
        if winning_version == "local":
            resolved = local_content
            winner = "local"
        else:
            resolved = remote_content
            winner = "remote"
    elif strategy == "cko_override":
        # Keep what's currently in the file (XiaoHu's last version)
        # Strip conflict markers from current content
        resolved = re.sub(r'^<<<<<<<\s.*\n', '', content)
        resolved = re.sub(r'^=======\s*$\n', '', resolved, flags=re.MULTILINE)
        resolved = re.sub(r'^>>>>>>>\s.*\n', '', resolved)
        winner = "cko_override"
    elif strategy == "environment_priority":
        # environment_priority: platform-specific version wins
        # Detect which side is platform-specific by path patterns
        # For now: assume the side with more platform-specific keywords wins
        platform_kw = ("Agents/Win", "Agents/Mac", "Agents/Linux", "Agents/Cloud",
                       ".plist", ".reg", ".config")
        local_has_platform = any(kw in local_content for kw in platform_kw)
        remote_has_platform = any(kw in remote_content for kw in platform_kw)

        if local_has_platform and not remote_has_platform:
            resolved = local_content
            winner = "local (platform-specific)"
        elif remote_has_platform and not local_has_platform:
            resolved = remote_content
            winner = "remote (platform-specific)"
        else:
            # Both or neither have platform keywords → use length heuristic
            resolved = local_content if len(local_content) >= len(remote_content) else remote_content
            winner = "local" if len(local_content) >= len(remote_content) else "remote"
            strategy = "environment_priority→timestamp_fallback"
    else:
        # timestamp_priority: longer version wins (heuristic for "more complete")
        resolved = local_content if len(local_content) >= len(remote_content) else remote_content
        winner = "local" if len(local_content) >= len(remote_content) else "remote"

    # Write resolved content
    full_path.write_text(resolved.strip() + "\n", encoding="utf-8")

    # Audit record
    today = date.today().isoformat()
    audit = {
        "date": today,
        "file": file_path,
        "strategy": strategy,
        "winner": winner,
        "note": note,
        "local_bytes": len(local_content),
        "remote_bytes": len(remote_content),
    }

    history = _load_arbiter_history()
    history.append(audit)
    _save_arbiter_history(history)

    return json.dumps({
        "success": True,
        "file": file_path,
        "strategy_used": strategy,
        "winner": winner,
        "audit_record": audit,
        "resolved_bytes": len(resolved),
    }, ensure_ascii=False)


def query_arbitration_history(
    file_path: str = "",
    strategy: str = "",
    limit: int = 20,
) -> str:
    """
    Query the arbitration history for past conflict resolutions.

    Can filter by:
      - file_path: exact file match (or prefix — e.g. 'Agents/' matches all agent files)
      - strategy: the resolution strategy used

    Returns the most recent matching records, useful for applying learned resolutions.

    Args:
        file_path: Optional file or directory path to filter by.
        strategy:  Optional strategy to filter by ('auto', 'manual', 'cko_override', 'timestamp_priority').
        limit:     Maximum number of records to return (default 20).

    Returns:
        JSON string with matching arbitration records and rule suggestions.
    """
    history = _load_arbiter_history()
    rules = _load_rules()

    # Filter
    filtered = history
    if file_path:
        filtered = [
            r for r in filtered
            if file_path in r.get("file", "")
        ]
    if strategy:
        filtered = [r for r in filtered if r.get("strategy") == strategy]

    filtered = filtered[-limit:]

    # Suggest new rules from history
    # Pattern: if same file appears ≥2 times with same strategy → suggest a rule
    file_strategy_counts: dict[str, dict] = {}
    for r in history:
        key = f"{r['file']}::{r.get('strategy', 'unknown')}"
        if key not in file_strategy_counts:
            file_strategy_counts[key] = {"file": r["file"], "strategy": r.get("strategy"), "count": 0}
        file_strategy_counts[key]["count"] += 1

    suggestions = [
        {"file_pattern": v["file"], "recommended_strategy": v["strategy"], "confidence": "high" if v["count"] >= 3 else "medium"}
        for v in file_strategy_counts.values()
        if v["count"] >= 2
    ]
    suggestions.sort(key=lambda x: x["confidence"] == "high", reverse=True)

    return json.dumps({
        "success": True,
        "total_records": len(history),
        "matching_records": len(filtered),
        "records": filtered,
        "rule_suggestions": suggestions[:10],
        "current_rules": rules,
    }, ensure_ascii=False)


# Built-in rules (platform-specific patterns)
_BUILTIN_RULES = {
    # Platform-specific configs
    "Agents/Win": "environment_priority",
    "Agents/Mac": "environment_priority",
    "Agents/Linux": "environment_priority",
    # Git and version control
    ".gitignore": "cko_override",
    ".gitattributes": "cko_override",
    # Protocol files
    "Protocols/": "cko_override",
    # Agent definitions
    "Agents/": "cko_override",
    # Configuration files
    "config.yaml": "cko_override",
    "config.json": "cko_override",
    # Knowledge notes
    "Notes/": "timestamp_priority",
    "MOCs/": "timestamp_priority",
}


def learn_rules_from_history(
    auto_apply: bool = False,
    min_confidence: str = "high",
) -> str:
    """
    Analyze arbitration history, generate learned rules, and optionally auto-apply them.

    Learned rule: same file resolved with the same strategy ≥2 times → candidate rule.
    Confidence levels:
      - 'high': same file resolved ≥3 times with same strategy
      - 'medium': same file resolved 2 times with same strategy

    Only 'high' confidence rules can be auto-applied (auto_apply=True).
    All learned rules are surfaced as suggestions regardless.

    Args:
        auto_apply:     If True, automatically apply 'high' confidence learned rules to rules.json.
                        If False, only return suggestions (default False).
        min_confidence: Minimum confidence level to include in suggestions: 'high' or 'medium'.

    Returns:
        JSON string with learned rule suggestions, current rules, and (if auto_apply=True) updated rules.json.
    """
    history = _load_arbiter_history()
    rules = _load_rules()

    # Count (file, strategy) occurrences
    counts: dict[str, dict] = {}
    for r in history:
        key = f"{r['file']}::{r.get('strategy', 'unknown')}"
        if key not in counts:
            counts[key] = {"file": r["file"], "strategy": r.get("strategy"), "count": 0, "dates": []}
        counts[key]["count"] += 1
        counts[key]["dates"].append(r.get("date", ""))

    # Generate suggestions
    learned = []
    for key, v in counts.items():
        conf = "high" if v["count"] >= 3 else "medium"
        if min_confidence == "high" and conf == "medium":
            continue
        # Don't override builtin rules
        builtin_override = any(pat in v["file"] for pat in _BUILTIN_RULES)
        learned.append({
            "file_pattern": v["file"],
            "recommended_strategy": v["strategy"],
            "confidence": conf,
            "occurrence_count": v["count"],
            "dates": v["dates"],
            "is_builtin_override": builtin_override,
            "note": "Would override built-in rule" if builtin_override else None,
        })

    learned.sort(key=lambda x: (x["confidence"] == "high", x["occurrence_count"]), reverse=True)

    # Auto-apply high-confidence rules
    applied = []
    if auto_apply:
        for rule in learned:
            if rule["confidence"] == "high" and not rule["is_builtin_override"]:
                # Apply as a new rule: store the file path as the pattern key
                rules[rule["file_pattern"]] = rule["recommended_strategy"]
                applied.append(rule["file_pattern"])
        if applied:
            _save_rules(rules)

    return json.dumps({
        "success": True,
        "total_history_records": len(history),
        "learned_rules": learned,
        "auto_applied": applied if auto_apply else [],
        "current_rules": rules,
        "builtin_rules": _BUILTIN_RULES,
    }, ensure_ascii=False)


# Initialize rules.json with built-in rules if not present
def _init_rules():
    rules = _load_rules()
    if not rules:
        _save_rules(_BUILTIN_RULES)

_init_rules()


def age_notes(
    path: Optional[str] = None,
    warning_threshold_days: int = 180,
    archive_threshold_days: int = 90,
    dry_run: bool = True,
    archive_base: str = ".Archive",
) -> str:
    """
    Scan notes in /Brain/ and classify them by aging status.

    Aging levels:
      ACTIVE   — Confidence >= 0.5 OR has incoming backlinks. Normal status, no action.
      WARNING  — Confidence < 0.5 AND no backlinks for >= warning_threshold_days.
                 Note is downgraded to 'stale' and flagged for Boss review.
      ARCHIVED — In WARNING state for >= archive_threshold_days with no improvement.
                 Note is moved to /.Archive/.

    Args:
        path:                  Optional sub-path to scan within /Brain/.
        warning_threshold_days: Days with no backlinks (and Confidence < 0.5) before
                                 WARNING status. Default 180 days (~6 months).
        archive_threshold_days: Days in WARNING state before archiving. Default 90 days.
        dry_run:               If True, only reports what would happen without making changes.
                                If False, actually demotes or archives notes.
        archive_base:          Subdirectory under /Brain/ for archived files.

    Returns:
        JSON string with aging status report for all scanned notes.
    """
    import re
    from datetime import date, datetime, timedelta

    brain_root = _get_brain_root()
    search_root = brain_root / path if path else brain_root

    if not search_root.exists():
        return json.dumps({
            "success": False,
            "error": f"Search path not found: {path or '/Brain/'}"
        })

    today = date.today()
    warning_delta = timedelta(days=warning_threshold_days)
    archive_delta = timedelta(days=archive_threshold_days)
    wikilink_re = re.compile(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]')

    # --- Step 1: Build backlink graph from all markdown files ---
    all_files: dict[str, set] = {}
    all_files_set: set = set()

    for md_file in search_root.rglob("*.md"):
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue

        rel_path = str(md_file.relative_to(brain_root))

        # Skip archive and system dirs
        if any(p in rel_path for p in [".Archive", ".git", "SOUL"]):
            continue

        links = wikilink_re.findall(content)
        linked_names = {Path(l).stem for l in links}
        all_files[rel_path] = linked_names
        all_files_set.add(rel_path)

    # Build backlink map
    backlink_map: dict[str, set] = {f: set() for f in all_files_set}
    for file_path, links in all_files.items():
        for linked_name in links:
            for candidate in all_files_set:
                if Path(candidate).stem == linked_name:
                    backlink_map[candidate].add(file_path)

    # --- Step 2: Parse frontmatter for each file ---
    def parse_frontmatter(content: str) -> dict:
        fm_match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
        if not fm_match:
            return {}
        fm = {}
        for line in fm_match.group(1).split('\n'):
            if ':' in line:
                key, val = line.split(':', 1)
                fm[key.strip()] = val.strip().strip('"').strip("'")
        return fm

    # --- Step 3: Classify each note ---
    ACTIVE = "ACTIVE"
    WARNING = "WARNING"
    ARCHIVED = "ARCHIVED"

    results_by_status = {ACTIVE: [], WARNING: [], ARCHIVED: []}

    for md_file in sorted(all_files_set):
        try:
            with open(brain_root / md_file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue

        fm = parse_frontmatter(content)
        confidence = float(fm.get("Confidence", "1.0"))
        backlinks = backlink_map.get(md_file, set())
        has_backlinks = len(backlinks) > 0

        # Extract title
        title = ""
        for line in content.split('\n'):
            if line.startswith('# '):
                title = line[2:].strip()
                break
        if not title:
            title = Path(md_file).stem

        # Parse last-modified from file stats
        try:
            mtime = datetime.fromtimestamp((brain_root / md_file).stat().st_mtime)
            last_modified = mtime.date()
        except Exception:
            last_modified = today

        # Parse Date field from frontmatter
        date_str = fm.get("Date", "")
        date_field = None
        if date_str:
            try:
                date_field = date.fromisoformat(date_str)
            except ValueError:
                pass

        effective_date = date_field or last_modified
        days_since = (today - effective_date).days

        # Classify
        is_low_conf = confidence < 0.5
        is_orphan = not has_backlinks

        if is_low_conf and is_orphan and days_since >= warning_threshold_days:
            status = WARNING
        else:
            status = ACTIVE

        note_entry = {
            "file": md_file,
            "title": title,
            "confidence": confidence,
            "backlink_count": len(backlinks),
            "last_modified": str(last_modified),
            "date_field": date_str or None,
            "days_since": days_since,
            "status": status,
        }

        results_by_status[status].append(note_entry)

    # --- Step 4: Apply archive action if not dry_run ---
    archive_moves = []
    warning_updates = []

    if not dry_run:
        for note in results_by_status[WARNING]:
            # Demote: add Status: stale to frontmatter
            note_path = brain_root / note["file"]
            try:
                with open(note_path, "r", encoding="utf-8") as f:
                    content = f.read()

                fm_match = re.match(r'^(---\n)(.*?)(\n---\n)', content, re.DOTALL)
                if fm_match:
                    fm_text = fm_match.group(2)
                    body = content[fm_match.end():]
                    # Add stale status
                    new_fm_lines = fm_text.rstrip('\n').split('\n')
                    new_fm_lines.append(f"Status: stale")
                    new_fm_lines.append(f"AgingNote: WARNING since {today.isoformat()}")
                    new_content = fm_match.group(1) + '\n'.join(new_fm_lines) + fm_match.group(3) + body
                else:
                    new_content = (
                        f"---\nStatus: stale\nAgingNote: WARNING since {today.isoformat()}\n---\n\n"
                        + content
                    )

                with open(note_path, "w", encoding="utf-8") as f:
                    f.write(new_content)

                warning_updates.append(note["file"])
            except Exception as e:
                pass  # Skip files that can't be written

        for note in results_by_status[ARCHIVED]:
            # Archive: move to .Archive directory
            note_path = brain_root / note["file"]
            archive_dir = brain_root / archive_base / f"{today.isoformat()}-AGED_{note['file'].replace('/', '_')}"
            try:
                archive_dir.mkdir(parents=True, exist_ok=True)
                import shutil
                archive_dest = archive_dir / note_path.name
                shutil.copy2(note_path, archive_dest)
                # Write aging metadata
                with open(archive_dest, "a", encoding="utf-8") as f:
                    f.write(f"\n\n> **Aging归档**: {today.isoformat()} | Confidence: {note['confidence']} | Backlinks: {note['backlink_count']}\n")
                # Remove original
                note_path.unlink()
                archive_moves.append(str(archive_dir.relative_to(brain_root)))
            except Exception as e:
                pass  # Skip files that can't be archived

    # --- Step 5: Build report ---
    total = sum(len(v) for v in results_by_status.values())

    report = {
        "success": True,
        "scan_path": str(path) if path else "/Brain/",
        "scan_date": today.isoformat(),
        "total_notes_scanned": total,
        "dry_run": dry_run,
        "aging_summary": {
            "active": len(results_by_status[ACTIVE]),
            "warning": len(results_by_status[WARNING]),
            "archived": len(results_by_status[ARCHIVED]),
        },
        "warning_threshold_days": warning_threshold_days,
        "archive_threshold_days": archive_threshold_days,
        "active_notes": results_by_status[ACTIVE],
        "warning_notes": results_by_status[WARNING],
        "archived_notes": results_by_status[ARCHIVED],
        "actions_taken": {
            "demoted_to_stale": warning_updates,
            "archived_to": archive_moves,
        } if not dry_run else {"demoted_to_stale": [], "archived_to": []},
    }

    return json.dumps(report, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# Availability check
# -----------------------------------------------------------------------------

def check_cko_brain_requirements() -> bool:
    """CKO brain tools are always available as long as the brain directory is accessible."""
    try:
        _get_brain_root().mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------

GIT_PULL_SCHEMA = {
    "name": "git_pull",
    "description": (
        "Pull the latest changes from a remote Git repository (typically the /Brain/ repo on GitHub). "
        "Use this before reading brain files to ensure you have the latest version."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "repo_path": {"type": "string", "default": ".", "description": "Path to the repository within /Brain/ (default: '.')"},
            "remote": {"type": "string", "default": "origin", "description": "Remote name (default: 'origin')"},
            "branch": {"type": "string", "default": "main", "description": "Branch name (default: 'main')"},
        },
    },
}

GIT_PUSH_SCHEMA = {
    "name": "git_push",
    "description": (
        "Push local commits to a remote Git repository. "
        "If 'message' is provided, also stages and commits all changes first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "repo_path": {"type": "string", "default": ".", "description": "Path to the repository within /Brain/ (default: '.')"},
            "remote": {"type": "string", "default": "origin", "description": "Remote name (default: 'origin')"},
            "branch": {"type": "string", "default": "main", "description": "Branch name (default: 'main')"},
            "message": {"type": "string", "description": "Optional commit message. If provided, commits before pushing."},
        },
    },
}

GIT_COMMIT_SCHEMA = {
    "name": "git_commit",
    "description": (
        "Stage and commit changes in a Git repository within /Brain/. "
        "Use after writing or modifying brain notes to save your changes locally."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "repo_path": {"type": "string", "default": ".", "description": "Path to the repository within /Brain/ (default: '.')"},
            "message": {"type": "string", "description": "Commit message describing the changes (required)."},
            "author": {"type": "string", "default": "CKO Agent", "description": "Author name for the commit."},
            "add_all": {"type": "boolean", "default": True, "description": "Whether to stage all changes before committing."},
        },
        "required": ["message"],
    },
}

READ_MARKDOWN_SCHEMA = {
    "name": "read_markdown",
    "description": (
        "Read a markdown file from the /Brain/ directory. "
        "Supports pagination via offset and limit parameters. "
        "Returns file content plus metadata (total lines, whether there's more)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path within /Brain/ (e.g., 'MOCs/Daily.md')"},
            "offset": {"type": "integer", "default": 1, "description": "Line number to start reading from (1-indexed, default: 1)"},
            "limit": {"type": "integer", "default": 500, "description": "Maximum number of lines to read (default: 500)"},
        },
        "required": ["path"],
    },
}

WRITE_MARKDOWN_SCHEMA = {
    "name": "write_markdown",
    "description": (
        "Write content to a markdown file in /Brain/. "
        "Creates parent directories automatically if needed. "
        "Use this to create or overwrite notes, MOCs, and other brain files."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path within /Brain/ (e.g., 'Notes/python-tips.md')"},
            "content": {"type": "string", "description": "The markdown content to write."},
            "create_dirs": {"type": "boolean", "default": True, "description": "Create parent directories if they don't exist."},
        },
        "required": ["path", "content"],
    },
}

APPEND_TO_FILE_SCHEMA = {
    "name": "append_to_file",
    "description": (
        "Append content to an existing file in /Brain/, or create it if it doesn't exist. "
        "Useful for adding new entries to log files, daily notes, or growing a document over time."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path within /Brain/ (e.g., 'Logs/2024-05-15.md')"},
            "content": {"type": "string", "description": "The content to append."},
            "create_if_missing": {"type": "boolean", "default": True, "description": "Create the file if it doesn't exist."},
        },
        "required": ["path", "content"],
    },
}

BRAIN_SEARCH_SCHEMA = {
    "name": "brain_search",
    "description": (
        "Search the /Brain/ directory for files containing a keyword or phrase. "
        "Supports regex patterns, [[wikilinks]], and #tags. "
        "Returns matching files with line numbers and surrounding context. "
        "Use this to check if knowledge already exists before creating a new note."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search term or regex pattern."},
            "path": {"type": "string", "description": "Optional sub-path to search within (relative to /Brain/). If None, searches everything."},
            "limit": {"type": "integer", "default": 20, "description": "Maximum number of matches to return (default: 20)."},
            "match_context": {"type": "integer", "default": 2, "description": "Number of context lines around each match (default: 2)."},
        },
        "required": ["query"],
    },
}

FIND_SIMILAR_SCHEMA = {
    "name": "find_similar",
    "description": (
        "Find notes in /Brain/ that are similar to a given text. "
        "Uses keyword-overlap similarity to detect potential duplicate content. "
        "Returns ranked results with similarity scores. "
        "Use this BEFORE creating a new note to check for existing knowledge."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The text to compare against existing notes."},
            "threshold": {"type": "number", "default": 0.25, "description": "Minimum similarity score (0.0-1.0) to include. Default 0.25."},
            "path": {"type": "string", "description": "Optional sub-path to search within."},
            "limit": {"type": "integer", "default": 10, "description": "Maximum results to return (default: 10)."},
        },
        "required": ["text"],
    },
}

FIND_ORPHAN_NOTES_SCHEMA = {
    "name": "find_orphan_notes",
    "description": (
        "Find notes in /Brain/ that have no outgoing or incoming links (orphan notes). "
        "Orphans are isolated — not linked TO from any other note AND do not link TO any other note. "
        "Returns a list of orphan files that need to be linked or archived. "
        "Use this periodically to keep the knowledge graph healthy."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Optional sub-path to search within."},
            "ignore_patterns": {"type": "array", "items": {"type": "string"}, "description": "Path substrings to ignore (e.g. ['README', 'index'])."},
        },
    },
}

CALIBRATE_CONFIDENCE_SCHEMA = {
    "name": "calibrate_confidence",
    "description": (
        "Calibrate the confidence score for a note based on source reliability. "
        "Maps source type to a confidence level (L1-L7), then applies content quality adjustments. "
        "Use this BEFORE finalizing a note's frontmatter to ensure the Confidence score is justified."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "source_type": {
                "type": "string",
                "enum": ["direct_verification", "official_doc", "multi_source",
                         "single可信_source", "inference", "unverified", "guess"],
                "description": "Type of information source."
            },
            "content_quality": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Quality of content: high (+0.1), medium (0), low (-0.1)."
            },
            "evidence_count": {
                "type": "integer",
                "default": 1,
                "description": "Number of independent evidence sources (default: 1)."
            },
            "has_caveats": {
                "type": "boolean",
                "default": False,
                "description": "Whether the content acknowledges limitations (bonus +0.05)."
            },
        },
        "required": ["source_type", "content_quality"],
    },
}

MERGE_NOTES_SCHEMA = {
    "name": "merge_notes",
    "description": (
        "Merge new content into an existing Brain note following the SOP. "
        "Performs field-level merge of frontmatter, body, tags, and links. "
        "Archives the original version to /.Archive/. "
        "Automatically detects merge type: facts_merged, discrepancy_logged, or superseded. "
        "Use this AFTER find_similar returns a match with similarity >= 0.25."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "existing_note": {
                "type": "string",
                "description": "Relative path within /Brain/ to the existing note to merge into."
            },
            "new_content": {
                "type": "string",
                "description": "The new markdown content to merge in."
            },
            "merge_type": {
                "type": "string",
                "enum": ["auto", "facts_merged", "discrepancy_logged", "superseded"],
                "default": "auto",
                "description": "Merge strategy. 'auto' detects based on content comparison."
            },
            "archive_base": {
                "type": "string",
                "default": ".Archive",
                "description": "Subdirectory under /Brain/ for archived files."
            },
        },
        "required": ["existing_note", "new_content"],
    },
}


RECORD_VOTE_SCHEMA = {
    "name": "record_vote",
    "description": (
        "Record a Boss feedback vote (👍 or 👎) for a Brain note. "
        "up = +0.10 to Confidence, down = -0.10. "
        "Also tracks topic-level aggregation: topics with net ≥ 3 upvotes get +0.05, "
        "net ≤ -3 downvotes get -0.05 future confidence bonus. "
        "Votes are stored in /.Feedback/Boss_votes.md. "
        "Use this whenever Boss explicitly approves or disapproves of a note."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "note_path": {"type": "string", "description": "Relative path within /Brain/ of the note being voted on."},
            "vote": {
                "type": "string",
                "enum": ["up", "down"],
                "description": "'up' = 👍 (+0.10), 'down' = 👎 (-0.10)."
            },
            "note": {
                "type": "boolean", "default": True,
                "description": "If True, vote applies to the note's Confidence. If False, votes on a topic keyword only."
            },
            "topic": {
                "type": "string", "default": "",
                "description": "Topic keyword to aggregate votes under (e.g. 'git', 'python'). Used for topic-level confidence bonus."
            },
        },
        "required": ["note_path", "vote"],
    },
}

RECALIBRATE_FROM_FEEDBACK_SCHEMA = {
    "name": "recalibrate_from_feedback",
    "description": (
        "Recalibrate a note's Confidence based on all accumulated Boss feedback votes. "
        "Reads all votes for this note from /.Feedback/Boss_votes.md, "
        "computes the cumulative delta, and applies it to the note's current Confidence. "
        "Appends a recalibration log block to the note. "
        "Use this before major reviews or periodic maintenance."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "note_path": {"type": "string", "description": "Relative path within /Brain/ of the note to recalibrate."},
        },
        "required": ["note_path"],
    },
}

FEEDBACK_SUMMARY_SCHEMA = {
    "name": "feedback_summary",
    "description": (
        "Get a summary of all Boss feedback votes. "
        "Returns per-note vote history, per-topic aggregates, "
        "and topic-level confidence adjustments (net ≥ 3 upvotes → +0.05, "
        "net ≤ -3 downvotes → -0.05). "
        "Use this to review the feedback landscape across the Brain."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string", "default": "",
                "description": "Optional topic keyword to filter the report to a specific topic."
            },
        },
    },
}


AGE_NOTES_SCHEMA = {
    "name": "age_notes",
    "description": (
        "Scan notes in /Brain/ and classify them by aging status. "
        "AGING LEVELS: ACTIVE = Confidence >= 0.5 OR has backlinks (normal). "
        "WARNING = Confidence < 0.5 AND no backlinks for >= warning_threshold_days (default 180 days). "
        "ARCHIVED = WARNING for >= archive_threshold_days with no improvement (default 90 days). "
        "Use dry_run=True to preview without making changes. "
        "Use dry_run=False to actually demote (add 'Status: stale') or archive notes. "
        "Archives go to /.Archive/YYYY-MM-DD-AGED_*/ . "
        "Run this periodically (e.g., weekly) to keep the knowledge base healthy."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Optional sub-path to scan within /Brain/."},
            "warning_threshold_days": {
                "type": "integer", "default": 180,
                "description": "Days with no backlinks (and Confidence < 0.5) before WARNING status. Default 180."
            },
            "archive_threshold_days": {
                "type": "integer", "default": 90,
                "description": "Days in WARNING state before archiving. Default 90."
            },
            "dry_run": {
                "type": "boolean", "default": True,
                "description": "If True, only reports what would happen. If False, applies changes."
            },
            "archive_base": {
                "type": "string", "default": ".Archive",
                "description": "Subdirectory under /Brain/ for archived files."
            },
        },
    },
}


# Arbitration schemas
DETECT_CONFLICTS_SCHEMA = {
    "name": "detect_conflicts",
    "description": (
        "Scan /Brain/ git repository for unmerged changes, conflict markers, and conflict risks. "
        "Detects: (1) files with conflict markers (<<<<<, =====, >>>>>), "
        "(2) uncommitted local changes, "
        "(3) files touched by multiple agents recently (multi_touched), "
        "(4) remote-ahead unpulled commits (if check_remote=True). "
        "Returns risk level: LOW / MEDIUM / HIGH. "
        "Use this BEFORE each git push to prevent merge conflicts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "repo_path": {
                "type": "string", "default": ".",
                "description": "Relative path within /Brain/ to the repository (default '.')."
            },
            "check_remote": {
                "type": "boolean", "default": False,
                "description": "If True, also check if remote has unpulled changes (default False)."
            },
        },
    },
}

RESOLVE_CONFLICT_SCHEMA = {
    "name": "resolve_conflict",
    "description": (
        "Resolve a git merge conflict for a specific file in /Brain/. "
        "Strategies: 'auto' (use rules.json), 'manual' (use winning_version), 'cko_override' (keep XiaoHu's version). "
        "Auto mode: reads /.Arbiter/rules.json to find matching pattern. "
        "Writes resolved file and appends audit record to /.Arbiter/history.json. "
        "CAUTION: Only use on files with active conflict markers. "
        "Use detect_conflicts first to find conflicted files."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Relative path within /Brain/ of the conflicted file."
            },
            "strategy": {
                "type": "string",
                "enum": ["auto", "manual", "cko_override"],
                "default": "auto",
                "description": "'auto' = rules.json, 'manual' = winning_version arg, 'cko_override' = keep current file."
            },
            "winning_version": {
                "type": "string",
                "enum": ["local", "remote"],
                "default": "local",
                "description": "Which version wins. Only used when strategy='manual'."
            },
            "note": {
                "type": "string", "default": "",
                "description": "Optional rationale for this resolution (saved to audit log)."
            },
        },
        "required": ["file_path"],
    },
}

QUERY_ARBITRATION_HISTORY_SCHEMA = {
    "name": "query_arbitration_history",
    "description": (
        "Query the arbitration history from /.Arbiter/history.json. "
        "Returns past conflict resolutions, filterable by file path or strategy. "
        "Also generates rule suggestions: if the same file was resolved with the same "
        "strategy ≥2 times → suggest it as a learned rule. "
        "Use this to apply learned resolutions automatically or review conflict patterns."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string", "default": "",
                "description": "Optional file path (or prefix like 'Agents/') to filter by."
            },
            "strategy": {
                "type": "string", "default": "",
                "description": "Optional strategy to filter by ('auto', 'manual', 'cko_override', 'timestamp_priority')."
            },
            "limit": {
                "type": "integer", "default": 20,
                "description": "Maximum number of history records to return (default 20)."
            },
        },
    },
}

BUILD_SEARCH_INDEX_SCHEMA = {
    "name": "build_search_index",
    "description": (
        "Build / rebuild the TF-IDF search index for /Brain/. "
        "Call this after adding or updating many notes. "
        "Results are cached in /.local_search/tfidf_index.json. "
        "The index is auto-built on first local_search call if missing. "
        "Use this to force a rebuild after bulk note changes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "chunk_size": {
                "type": "integer", "default": 500,
                "description": "Max words per chunk (default 500). Short notes = 1 chunk. Larger = fewer chunks, faster search but less granular."
            },
        },
    },
}

LOCAL_SEARCH_SCHEMA = {
    "name": "local_search",
    "description": (
        "Hybrid TF-IDF search over /Brain/ notes. "
        "Supports Chinese (bigram tokenization) and English. "
        "Pipeline: tokenize → TF-IDF cosine similarity → keyword boost → rank. "
        "Vector embedding layer is reserved for future integration. "
        "Use build_search_index after bulk note changes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query. Supports Chinese and English."
            },
            "top_k": {
                "type": "integer", "default": 5,
                "description": "Max results to return (default 5)."
            },
            "mode": {
                "type": "string", "default": "hybrid",
                "enum": ["hybrid", "tfidf"],
                "description": "'hybrid' = TF-IDF + keyword boost; 'tfidf' = score only."
            },
        },
        "required": ["query"],
    },
}


LEARN_RULES_SCHEMA = {
    "name": "learn_rules_from_history",
    "description": (
        "Analyze arbitration history and generate learned conflict-resolution rules. "
        "Learned rule = same file resolved with the same strategy ≥2 times. "
        "Confidence 'high' = ≥3 occurrences (auto-apply eligible). "
        "Confidence 'medium' = 2 occurrences (suggestion only). "
        "Built-in rules (Agents/, Protocols/, Notes/) are never auto-overwritten. "
        "Use this after several arbitrations to let the system learn from past decisions. "
        "Use auto_apply=True to automatically promote 'high' confidence rules to rules.json."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "auto_apply": {
                "type": "boolean", "default": False,
                "description": "If True, auto-apply 'high' confidence rules to rules.json. If False, suggestions only."
            },
            "min_confidence": {
                "type": "string",
                "enum": ["high", "medium"],
                "default": "high",
                "description": "Minimum confidence to include: 'high' (≥3 occurrences) or 'medium' (≥2)."
            },
        },
    },
}


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------
from tools.registry import registry

registry.register(
    name="git_pull",
    toolset="cko_brain",
    schema=GIT_PULL_SCHEMA,
    handler=lambda args, **kw: git_pull(
        repo_path=args.get("repo_path", "."),
        remote=args.get("remote", "origin"),
        branch=args.get("branch", "main"),
    ),
    check_fn=check_cko_brain_requirements,
    emoji="⬇️",
    description="Pull latest changes from /Brain/ Git repo",
)

registry.register(
    name="git_push",
    toolset="cko_brain",
    schema=GIT_PUSH_SCHEMA,
    handler=lambda args, **kw: git_push(
        repo_path=args.get("repo_path", "."),
        remote=args.get("remote", "origin"),
        branch=args.get("branch", "main"),
        message=args.get("message"),
    ),
    check_fn=check_cko_brain_requirements,
    emoji="⬆️",
    description="Push commits to /Brain/ Git repo",
)

registry.register(
    name="git_commit",
    toolset="cko_brain",
    schema=GIT_COMMIT_SCHEMA,
    handler=lambda args, **kw: git_commit(
        repo_path=args.get("repo_path", "."),
        message=args.get("message", ""),
        author=args.get("author", "CKO Agent"),
        add_all=args.get("add_all", True),
    ),
    check_fn=check_cko_brain_requirements,
    emoji="📝",
    description="Commit changes to /Brain/ Git repo",
)

registry.register(
    name="read_markdown",
    toolset="cko_brain",
    schema=READ_MARKDOWN_SCHEMA,
    handler=lambda args, **kw: read_markdown(
        path=args.get("path", ""),
        offset=args.get("offset", 1),
        limit=args.get("limit", 500),
    ),
    check_fn=check_cko_brain_requirements,
    emoji="📖",
    description="Read a markdown file from /Brain/",
)

registry.register(
    name="write_markdown",
    toolset="cko_brain",
    schema=WRITE_MARKDOWN_SCHEMA,
    handler=lambda args, **kw: write_markdown(
        path=args.get("path", ""),
        content=args.get("content", ""),
        create_dirs=args.get("create_dirs", True),
    ),
    check_fn=check_cko_brain_requirements,
    emoji="✍️",
    description="Write content to a /Brain/ markdown file",
)

registry.register(
    name="append_to_file",
    toolset="cko_brain",
    schema=APPEND_TO_FILE_SCHEMA,
    handler=lambda args, **kw: append_to_file(
        path=args.get("path", ""),
        content=args.get("content", ""),
        create_if_missing=args.get("create_if_missing", True),
    ),
    check_fn=check_cko_brain_requirements,
    emoji="➕",
    description="Append content to a /Brain/ file",
)

registry.register(
    name="brain_search",
    toolset="cko_brain",
    schema=BRAIN_SEARCH_SCHEMA,
    handler=lambda args, **kw: brain_search(
        query=args.get("query", ""),
        path=args.get("path"),
        limit=args.get("limit", 20),
        match_context=args.get("match_context", 2),
    ),
    check_fn=check_cko_brain_requirements,
    emoji="🔍",
    description="Search /Brain/ for keyword or phrase",
)

registry.register(
    name="find_similar",
    toolset="cko_brain",
    schema=FIND_SIMILAR_SCHEMA,
    handler=lambda args, **kw: find_similar(
        text=args.get("text", ""),
        threshold=args.get("threshold", 0.25),
        path=args.get("path"),
        limit=args.get("limit", 10),
    ),
    check_fn=check_cko_brain_requirements,
    emoji="🔎",
    description="Find notes similar to a given text in /Brain/",
)

registry.register(
    name="find_orphan_notes",
    toolset="cko_brain",
    schema=FIND_ORPHAN_NOTES_SCHEMA,
    handler=lambda args, **kw: find_orphan_notes(
        path=args.get("path"),
        ignore_patterns=args.get("ignore_patterns"),
    ),
    check_fn=check_cko_brain_requirements,
    emoji="👻",
    description="Find orphan (unlinked) notes in /Brain/",
)

registry.register(
    name="calibrate_confidence",
    toolset="cko_brain",
    schema=CALIBRATE_CONFIDENCE_SCHEMA,
    handler=lambda args, **kw: calibrate_confidence(
        source_type=args.get("source_type", ""),
        content_quality=args.get("content_quality", "medium"),
        evidence_count=args.get("evidence_count", 1),
        has_caveats=args.get("has_caveats", False),
    ),
    check_fn=check_cko_brain_requirements,
    emoji="⚖️",
    description="Calibrate note confidence score based on source reliability",
)

registry.register(
    name="merge_notes",
    toolset="cko_brain",
    schema=MERGE_NOTES_SCHEMA,
    handler=lambda args, **kw: merge_notes(
        existing_note=args.get("existing_note", ""),
        new_content=args.get("new_content", ""),
        merge_type=args.get("merge_type", "auto"),
        archive_base=args.get("archive_base", ".Archive"),
    ),
    check_fn=check_cko_brain_requirements,
    emoji="🔀",
    description="Merge new content into an existing Brain note per SOP",
)

registry.register(
    name="age_notes",
    toolset="cko_brain",
    schema=AGE_NOTES_SCHEMA,
    handler=lambda args, **kw: age_notes(
        path=args.get("path"),
        warning_threshold_days=args.get("warning_threshold_days", 180),
        archive_threshold_days=args.get("archive_threshold_days", 90),
        dry_run=args.get("dry_run", True),
        archive_base=args.get("archive_base", ".Archive"),
    ),
    check_fn=check_cko_brain_requirements,
    emoji="⏳",
    description="Scan and age-classify notes in /Brain/ — detect stale or archive-eligible notes",
)

registry.register(
    name="record_vote",
    toolset="cko_brain",
    schema=RECORD_VOTE_SCHEMA,
    handler=lambda args, **kw: record_vote(
        note_path=args.get("note_path", ""),
        vote=args.get("vote", "up"),
        note=args.get("note", True),
        topic=args.get("topic", ""),
    ),
    check_fn=check_cko_brain_requirements,
    emoji="👍",
    description="Record Boss feedback vote (👍/👎) for a Brain note",
)

registry.register(
    name="recalibrate_from_feedback",
    toolset="cko_brain",
    schema=RECALIBRATE_FROM_FEEDBACK_SCHEMA,
    handler=lambda args, **kw: recalibrate_from_feedback(
        note_path=args.get("note_path", ""),
    ),
    check_fn=check_cko_brain_requirements,
    emoji="🔄",
    description="Recalibrate note Confidence from accumulated Boss feedback votes",
)

registry.register(
    name="feedback_summary",
    toolset="cko_brain",
    schema=FEEDBACK_SUMMARY_SCHEMA,
    handler=lambda args, **kw: feedback_summary(
        topic=args.get("topic", ""),
    ),
    check_fn=check_cko_brain_requirements,
    emoji="📊",
    description="Get summary of all Boss feedback votes and topic-level confidence adjustments",
)

registry.register(
    name="detect_conflicts",
    toolset="cko_brain",
    schema=DETECT_CONFLICTS_SCHEMA,
    handler=lambda args, **kw: detect_conflicts(
        repo_path=args.get("repo_path", "."),
        check_remote=args.get("check_remote", False),
    ),
    check_fn=check_cko_brain_requirements,
    emoji="🔍",
    description="Scan /Brain/ for git conflicts and conflict risks — run before git push",
)

registry.register(
    name="resolve_conflict",
    toolset="cko_brain",
    schema=RESOLVE_CONFLICT_SCHEMA,
    handler=lambda args, **kw: resolve_conflict(
        file_path=args.get("file_path", ""),
        strategy=args.get("strategy", "auto"),
        winning_version=args.get("winning_version", "local"),
        note=args.get("note", ""),
    ),
    check_fn=check_cko_brain_requirements,
    emoji="⚖️",
    description="Resolve a git merge conflict in /Brain/ using auto or manual strategy",
)

registry.register(
    name="query_arbitration_history",
    toolset="cko_brain",
    schema=QUERY_ARBITRATION_HISTORY_SCHEMA,
    handler=lambda args, **kw: query_arbitration_history(
        file_path=args.get("file_path", ""),
        strategy=args.get("strategy", ""),
        limit=args.get("limit", 20),
    ),
    check_fn=check_cko_brain_requirements,
    emoji="📜",
    description="Query arbitration history for past conflict resolutions and learned rules",
)

registry.register(
    name="learn_rules_from_history",
    toolset="cko_brain",
    schema=LEARN_RULES_SCHEMA,
    handler=lambda args, **kw: learn_rules_from_history(
        auto_apply=args.get("auto_apply", False),
        min_confidence=args.get("min_confidence", "high"),
    ),
    check_fn=check_cko_brain_requirements,
    emoji="🧠",
    description="Learn conflict-resolution rules from arbitration history — auto-apply high-confidence rules",
)

registry.register(
    name="build_search_index",
    toolset="cko_brain",
    schema=BUILD_SEARCH_INDEX_SCHEMA,
    handler=lambda args, **kw: build_search_index(
        chunk_size=args.get("chunk_size", 500),
    ),
    check_fn=check_cko_brain_requirements,
    emoji="🔍",
    description="Build TF-IDF search index for /Brain/ — call after bulk note changes",
)

registry.register(
    name="local_search",
    toolset="cko_brain",
    schema=LOCAL_SEARCH_SCHEMA,
    handler=lambda args, **kw: local_search(
        query=args["query"],
        top_k=args.get("top_k", 5),
        mode=args.get("mode", "hybrid"),
    ),
    check_fn=check_cko_brain_requirements,
    emoji="🔎",
    description="Hybrid TF-IDF search over /Brain/ notes — Chinese + English, CJK bigrams, keyword boost",
)

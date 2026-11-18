"""read_file views that stand in for shell one-liners the model otherwise sends to the terminal.

``offset=-N`` reads the last N lines (``tail -n N``); a directory path returns a listing with
types, sizes and mtimes (``ls -la``); inside ``execute_code`` a read also carries ``text``, the
page without its ``N|`` gutter, so a cell can parse or rewrite it without stripping prefixes.
"""

import os
import re
import stat
import time

# A listing is a context payload like any read: past this many entries it points to a glob search.
DIR_LISTING_MAX_ENTRIES = 500
_GUTTER = re.compile(r"^\d+\|", re.MULTILINE)
_KINDS = ((stat.S_ISDIR, "dir"), (stat.S_ISLNK, "link"), (stat.S_ISREG, "file"))


def tail_request(offset) -> int | None:
    """N when ``offset`` asks for the last N lines (a negative integer), else None."""
    try:
        value = int(offset)
    except (TypeError, ValueError):
        return None
    return -value if value < 0 else None


def tail_offset(file_ops, path: str, lines: int) -> int:
    """First line of the last ``lines`` lines of ``path``. A failed probe (missing, binary) gives
    line 1, so the real read reports the failure exactly as a plain read would."""
    total = file_ops.read_file(path, 1, 1).total_lines or 0
    return max(1, total - lines + 1)


def raw_page_text(content: str) -> str:
    """A read page without its ``N|`` line-number gutter."""
    return _GUTTER.sub("", content)


def _entry(entry: os.DirEntry) -> dict | None:
    try:
        st = entry.stat(follow_symlinks=False)
    except FileNotFoundError:  # removed between scandir and stat
        return None
    kind = next((name for test, name in _KINDS if test(st.st_mode)), "other")
    item = {"name": entry.name, "type": kind, "size": st.st_size if kind == "file" else None,
            "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))}
    if kind == "link":
        item["target"] = os.readlink(entry.path)
    return item


def _row(item: dict) -> str:
    size = "-" if item["size"] is None else str(item["size"])
    name = item["name"] + ("/" if item["type"] == "dir" else "")
    link = f" -> {item['target']}" if item["type"] == "link" else ""
    return f"{size:>10}  {item['mtime']}  {name}{link}"


def directory_listing(path: str, *, structured: bool) -> dict:
    """``ls -la`` for one directory: dirs first, then by name. ``structured`` (a cell's read)
    adds the entries as a list of dicts."""
    with os.scandir(path) as it:
        items = [e for e in map(_entry, it) if e is not None]
    items.sort(key=lambda e: (e["type"] != "dir", e["name"]))
    shown = items[:DIR_LISTING_MAX_ENTRIES]
    dirs = sum(1 for e in items if e["type"] == "dir")
    header = f"{path}: {len(items)} entries ({dirs} dirs, {len(items) - dirs} other). size  mtime  name"
    result = {"path": path, "is_directory": True, "total_entries": len(items),
              "content": "\n".join([header, *map(_row, shown)])}
    if len(items) > len(shown):
        result["truncated"] = True
        result["hint"] = (f"Showing {len(shown)} of {len(items)} entries. Use search_files with "
                          "target='files' and a glob to narrow it.")
    if structured:
        result["entries"] = items
    return result

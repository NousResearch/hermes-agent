import os
import json
from executor import default_home

# Allowed roots for file operations.  Defaults to user home + /tmp.
# Override via OMARCHY_FILE_ROOTS (colon-separated).
def _allowed_roots() -> list[str]:
    env = os.environ.get("OMARCHY_FILE_ROOTS", "")
    if env:
        return [p.strip() for p in env.split(":") if p.strip()]
    return [default_home(), "/tmp"]


def _check_path(path: str) -> str:
    """Resolve *path* and check it's under one of the allowed roots.
    Returns the resolved path on success, or a JSON error string
    starting with ``{"error"`` when the path is rejected."""
    try:
        real_path = os.path.realpath(path)
    except (OSError, ValueError):
        return json.dumps({"error": f"cannot resolve path: {path}"})
    for root in _allowed_roots():
        if real_path == root or real_path.startswith(root + "/"):
            return real_path
    return json.dumps({"error": f"path must be under allowed roots"})


def _is_error(checked: str) -> bool:
    return checked.startswith("{")


def register(mcp):
    @mcp.tool()
    async def file_read(
        path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read a file's content from the Omarchy filesystem.

        Args:
            path: Absolute path to file
            offset: Starting line (0-indexed)
            limit: Maximum lines to return
        """
        checked = _check_path(path)
        if _is_error(checked):
            return checked  # error JSON

        try:
            with open(checked, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total = len(lines)
            selected = lines[offset : offset + limit]
            content = "".join(selected)

            return json.dumps({
                "content": content,
                "total_lines": total,
                "offset": offset,
                "returned_lines": len(selected),
                "truncated": (offset + limit) < total,
            })
        except FileNotFoundError:
            return json.dumps({"error": f"File not found: {path}"})
        except IsADirectoryError:
            return json.dumps({"error": f"Path is a directory: {path}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def file_write(
        path: str,
        content: str,
        mode: str = "overwrite",
    ) -> str:
        """Write content to a file on Omarchy.

        Args:
            path: Absolute path to file
            content: Content to write
            mode: 'overwrite' or 'append'
        """
        checked = _check_path(path)
        if _is_error(checked):
            return checked  # error JSON

        try:
            write_mode = "a" if mode == "append" else "w"
            os.makedirs(os.path.dirname(checked), exist_ok=True)
            with open(checked, write_mode, encoding="utf-8") as f:
                f.write(content)
            return json.dumps({"ok": True, "path": checked, "mode": mode, "bytes": len(content)})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def file_list(
        path: str | None = None,
    ) -> str:
        """List files and directories at a path on Omarchy.

        Args:
            path: Directory path to list (default: user home)
        """
        checked = _check_path(path or default_home())
        if _is_error(checked):
            return checked  # error JSON

        try:
            entries = os.listdir(checked)
            result = []
            for name in sorted(entries):
                full = os.path.join(checked, name)
                try:
                    st = os.stat(full)
                    result.append({
                        "name": name,
                        "type": "dir" if os.path.isdir(full) else "file",
                        "size": st.st_size,
                        "mtime": st.st_mtime,
                    })
                except OSError:
                    result.append({"name": name, "type": "unknown", "size": 0, "mtime": 0})
            return json.dumps({"entries": result, "count": len(result)})
        except FileNotFoundError:
            return json.dumps({"error": f"Directory not found: {path}"})
        except NotADirectoryError:
            return json.dumps({"error": f"Not a directory: {path}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

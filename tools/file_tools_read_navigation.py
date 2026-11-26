"""Read-mode routing after the shared file access guards."""

import json

from tools.binary_extensions import has_binary_extension
from tools.file_outline import outline_page


def validate_read_options(mode, cursor):
    if mode not in ("read", "outline"):
        raise ValueError("mode must be 'read' or 'outline'")
    if cursor is not None and (mode != "outline" or not isinstance(cursor, str)):
        raise ValueError("cursor must be a string used with mode='outline'")


def read_navigation_result(path, resolved, offset, limit, task_id, mode, cursor):
    """Return a navigation/extraction result, or None for an ordinary text read."""
    from tools import file_tools as ft

    if mode == "outline":
        return _read_outline(path, resolved, offset, limit, task_id, cursor)
    extracted = ft._read_extracted_document(path, resolved, offset, limit, task_id)
    if extracted is not None:
        return extracted
    # The extension is a claim; content sniffing names the actual magic-byte type.
    if has_binary_extension(str(resolved)):
        return ft.tool_error(
            f"Cannot read binary file '{path}' ({resolved.suffix.lower()}). "
            "Use vision_analyze for images, or terminal to inspect binary files.")
    return None


def _read_outline(path, resolved, offset, limit, task_id, cursor):
    from tools import file_tools as ft

    identity = str(resolved)
    cached = ft._check_not_found_cache("read", identity, task_id)
    if cached is not None:
        return cached
    ops = ft._get_file_ops(task_id)
    host_paths = ft._file_ops_uses_host_paths(ops)
    version = ft._file_metadata(identity) if host_paths else None
    dedup_key = (identity, "outline", offset, limit)
    with ft._read_tracker_lock:
        task_data = ft._task_data(task_id)
        cached_version = task_data["dedup"].get(dedup_key) if cursor is None else None
        served = dedup_key in task_data["dedup_generation_reads"]
    if version is not None and cached_version == version and served:
        return ft._dedup_stub_or_block(task_data, dedup_key, path)
    result = outline_page(ops, identity if host_paths else path, identity,
                          task_id, offset, limit, cursor)
    # Outline reads must neither establish nor refresh a body-read/write baseline.
    if "error" not in result:
        with ft._read_tracker_lock:
            if cursor is None:
                task_data["dedup_hits"].pop(dedup_key, None)
                task_data["dedup_generation_reads"].add(dedup_key)
                if version is not None and ft._file_metadata(identity) == version:
                    task_data["dedup"][dedup_key] = version
            count = ft._bump_consecutive(
                task_data, ("read", path, "outline", cursor or offset, limit))
            ft._cap_read_tracker_data(task_data)
        if count >= 4:
            return ft.tool_error(
                f"BLOCKED: You have requested this exact outline {count} times in a row. "
                "The file has NOT changed. Use the outline you already have.",
                path=path, already_read=count)
        if count >= 3:
            result["_warning"] = (
                f"You have requested this exact outline {count} times consecutively. "
                "The file has not changed; use the outline you already have.")
    return json.dumps(result, ensure_ascii=False)

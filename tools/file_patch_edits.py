"""Validation shared by atomic replacements and ACP previews."""


def validate_patch_edits(edits: list) -> None:
    """Reject malformed edit batches before applying any replacements."""
    if not isinstance(edits, list) or not 1 <= len(edits) <= 50:
        raise ValueError("edits must be a non-empty list of at most 50 edits")
    for i, edit in enumerate(edits, 1):
        if not isinstance(edit, dict) or not all(
            isinstance(edit.get(key), str) for key in ("old_string", "new_string")
        ):
            raise ValueError(f"edit {i} of {len(edits)}: old_string and new_string must be strings")
        if "replace_all" in edit and not isinstance(edit["replace_all"], bool):
            raise ValueError(f"edit {i} of {len(edits)}: replace_all must be a boolean")

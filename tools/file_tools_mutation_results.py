"""Validate captured target identities and map backend mutation receipts."""


def mutation_alias_error(targets, _content_write_paths):
    backend_owners: dict[str, str] = {}
    for display_path, target in targets.items():
        mutation_identities = set()
        if display_path in _content_write_paths:
            mutation_identities.add(target.backend_path)
        if target.entry_path is not None:
            mutation_identities.add(target.entry_path)
        for identity in mutation_identities:
            prior = backend_owners.setdefault(identity, display_path)
            if prior == display_path:
                continue
            return (
                "V4A patch resolves multiple display paths to the same backend target: "
                f"{prior!r} and {display_path!r} -> {identity!r}. "
                "Refusing before validation to preserve all-or-nothing patch semantics."
            )
    return None


def mapped_patch_result(result, targets, stale_warnings):
    result_dict = result.to_dict()
    if stale_warnings:
        result_dict["_warning"] = " | ".join(stale_warnings)
    def _map_result_path(label: str, *, entry: bool = False) -> str:
        target = targets.get(label)
        if target is not None:
            return (target.entry_path or target.backend_path) if entry else target.backend_path
        if " -> " in label:
            src, dst = label.split(" -> ", 1)
            if src in targets and dst in targets:
                src_path = targets[src].entry_path or targets[src].backend_path
                dst_path = targets[dst].entry_path or targets[dst].backend_path
                return f"{src_path} -> {dst_path}"
        return label
    for result_key in ("files_modified", "files_created", "files_deleted"):
        values = result_dict.get(result_key)
        if isinstance(values, list):
            result_dict[result_key] = [
                _map_result_path(str(value), entry=result_key == "files_deleted")
                for value in values
            ]
    if isinstance(result_dict.get("lint"), dict):
        result_dict["lint"] = {
            _map_result_path(str(key)): value for key, value in result_dict["lint"].items()
        }
    return result_dict

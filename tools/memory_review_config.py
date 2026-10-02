"""Read memory settings without importing the CLI's initializing config facade."""
import os
import re
from typing import Any

from hermes_constants import get_hermes_home
from hermes_cli.config_backend import config_exists, read_config_doc_readonly
from hermes_cli.managed_scope import load_managed_config


def _expand(value: Any, lookup) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item, lookup) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, lookup) for item in value]
    if isinstance(value, str):
        def replace(match):
            name = match[1].strip()
            if name.startswith("env:"):
                name = name[4:].strip()
            elif ":" in name:
                return match[0]
            resolved = lookup(name)
            return match[0] if resolved is None else resolved
        return re.sub(r"\$\{([^}]+)\}", replace, value)
    return value


def read_memory_review_config():
    from agent.secret_scope import get_secret
    path = get_hermes_home() / "config.yaml"
    # Read-only preview: the signature-cached backend keeps this off the CLI's
    # initializing facade, and a shared document is never mutated here.
    raw = read_config_doc_readonly(path) if config_exists(path) else {}
    raw = raw if isinstance(raw, dict) else {}
    memory = raw.get("memory", {})
    memory = _expand(memory if isinstance(memory, dict) else {}, get_secret)
    managed = load_managed_config().get("memory", {})
    if isinstance(managed, dict):
        memory.update(_expand(managed, os.environ.get))
    return {"memory": memory}

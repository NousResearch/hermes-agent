"""Self-introspection tool for agent runtime awareness."""

from tools.registry import registry
import os
from pathlib import Path
import yaml

def _load_config():
    hermes_home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    cfg_path = hermes_home / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    return {}

def _handle_agent_self(**kwargs):
    cfg = _load_config()
    return {
        "model": os.environ.get("HERMES_MODEL") or cfg.get("model", {}).get("default"),
        "provider": os.environ.get("HERMES_INFERENCE_PROVIDER") or cfg.get("model", {}).get("provider"),
        "hermes_home": str(Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))),
        "enabled_toolsets": os.environ.get("HERMES_ENABLED_TOOLSETS", "").split(",") if os.environ.get("HERMES_ENABLED_TOOLSETS") else [],
        "cwd": os.getcwd(),
        "memory_note": "use memory tools for full state",
    }

registry.register(
    name="agent_self",
    toolset="hermes-cli",
    schema={
        "name": "agent_self",
        "description": "Return the agent's current runtime identity and state (model, provider, toolsets, config). No parameters.",
        "parameters": {"type": "object", "properties": {}}
    },
    handler=_handle_agent_self,
    check_fn=lambda: True,
    emoji="🪞",
)
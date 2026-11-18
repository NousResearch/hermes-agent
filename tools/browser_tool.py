"""Frozen old-updater compatibility stub (tests/compat/old_updater_surface.json, ``bare`` list).

The built-in browser_* tools and agent-browser are gone; ``browser_exec`` (``tools/browser_use_cli.py``)
is the browser. A pre-PM ``hermes update`` still running mid-swap imports this module and the name below
from the NEW tree, so both must keep existing. Nothing else belongs here.
"""


def warm_agent_browser_npx_cache(timeout: float = 60.0) -> bool:
    """Frozen old-updater surface: nothing is warmed (there is no agent-browser to warm); always False."""
    return False

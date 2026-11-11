"""Frozen old-updater compatibility stub (tests/compat/old_updater_surface.json, ``bare`` list).

The extraction-era home of ``warm_agent_browser_npx_cache``: a pre-PM ``hermes update`` still running
mid-swap may import it from the NEW tree. agent-browser is gone; nothing else belongs here.
"""


def warm_agent_browser_npx_cache(timeout: float = 60.0) -> bool:
    """Frozen old-updater surface: nothing is warmed (there is no agent-browser to warm); always False."""
    return False

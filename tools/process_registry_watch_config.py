"""User-configured, finite watch-pattern budgets, captured on the spawning thread."""
import logging

logger = logging.getLogger('tools.process_registry')

DEFAULT_WATCH_LIFETIME_MAX_HITS = 8
MAX_WATCH_LIFETIME_MAX_HITS = 64


def normalize_watch_lifetime_max_hits(value):
    """Bad configuration/checkpoint values must never disable the lifetime breaker."""
    if type(value) is int and 1 <= value <= MAX_WATCH_LIFETIME_MAX_HITS:
        return value
    logger.warning('terminal.watch_lifetime_max_hits must be an integer from 1 to %d; using %d',
                   MAX_WATCH_LIFETIME_MAX_HITS, DEFAULT_WATCH_LIFETIME_MAX_HITS)
    return DEFAULT_WATCH_LIFETIME_MAX_HITS


def configured_watch_lifetime_max_hits():
    """Read the active profile once at spawn, not from an unscoped output-reader thread.

    The normal loader also respects managed settings and a managed worker's frozen
    config. No environment override or model-supplied tool argument is introduced.
    """
    from hermes_cli.config_effective import load_user_config_effective

    try:
        section = load_user_config_effective().get('terminal', {})
        value = section.get('watch_lifetime_max_hits', DEFAULT_WATCH_LIFETIME_MAX_HITS) if isinstance(section, dict) else None
    except (OSError, ValueError, TypeError):
        logger.warning('Could not load terminal.watch_lifetime_max_hits; using %d', DEFAULT_WATCH_LIFETIME_MAX_HITS)
        return DEFAULT_WATCH_LIFETIME_MAX_HITS
    return normalize_watch_lifetime_max_hits(value)

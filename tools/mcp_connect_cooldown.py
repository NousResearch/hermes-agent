import logging
import time
from typing import Dict

logger = logging.getLogger(__name__)

# Connection-retry cooldown (per-server isolation against restart storms).
#
# A single stdio MCP server that fails to spawn (bad PATH, ``exec: not
# found``, crash-on-start) is never recorded in ``_servers`` -- ``start()``
# raises and ``_discover_and_register_server`` aborts before the
# ``_servers[name] = server`` line. Without a cooldown, EVERY subsequent
# ``discover_mcp_tools()`` (one per agent worker session, i.e. every few
# seconds) sees the server as "not connected" and re-spawns it from
# scratch. That is the restart storm in #50394: the failing server is
# re-attempted on the shared MCP event loop on every worker session, the
# subprocesses pile up unreaped, and the churn destabilises the healthy
# co-located servers (their tools intermittently surface as
# "Unknown tool").
#
# Fix: after a failed connection attempt, stamp a monotonic
# ``retry_after`` deadline with exponential backoff. ``register_mcp_servers``
# skips a server whose cooldown has not elapsed, so a chronically failing
# server is retried on a backoff schedule instead of on every worker
# session -- isolating it from the rest of the bridge. A successful
# connection clears the state.
_server_connect_retry_after: Dict[str, float] = {}   # name -> monotonic deadline
_server_connect_failures: Dict[str, int] = {}        # name -> consecutive failures
_CONNECT_RETRY_BASE_BACKOFF_SEC = 30.0
_CONNECT_RETRY_MAX_BACKOFF_SEC = 600.0


def _record_connect_failure(server_name: str) -> None:
    """Stamp an exponential-backoff cooldown after a failed connect.

    Called (under ``_lock``) when a server fails its discovery/connect
    attempt. The cooldown grows geometrically with the consecutive
    failure count and is capped at :data:`_CONNECT_RETRY_MAX_BACKOFF_SEC`,
    so a permanently-broken server settles into infrequent retries
    rather than a tight respawn loop.
    """
    n = _server_connect_failures.get(server_name, 0) + 1
    _server_connect_failures[server_name] = n
    backoff = min(
        _CONNECT_RETRY_BASE_BACKOFF_SEC * (2 ** (n - 1)),
        _CONNECT_RETRY_MAX_BACKOFF_SEC,
    )
    _server_connect_retry_after[server_name] = time.monotonic() + backoff


def _clear_connect_failure(server_name: str) -> None:
    """Clear the connect-cooldown state after a successful connection."""
    _server_connect_failures.pop(server_name, None)
    _server_connect_retry_after.pop(server_name, None)


def _connect_cooldown_active(server_name: str) -> bool:
    """Return True if ``server_name`` is still within its retry cooldown."""
    deadline = _server_connect_retry_after.get(server_name)
    return deadline is not None and time.monotonic() < deadline

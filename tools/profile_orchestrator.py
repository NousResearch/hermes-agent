"""
Inter-profile orchestration for Hermes Agent.

Provides profile discovery, health checking, and port resolution
for managing multiple Hermes profiles under ~/.hermes/profiles/<name>/.

Each profile has its own HERMES_HOME, config.yaml, .env, and gateway.pid file.
This module allows tools to discover and interact with running profile instances.
"""

import asyncio  # For async operations and Future checking
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from hermes_cli.profiles import (
    _get_profiles_root,
    get_profile_dir,
    list_profiles,
    profile_exists,
    _check_gateway_running,
    ProfileInfo,
)

# Re-use gateway/status.py PID checking mechanisms
# Import late to avoid circular imports at module level
_gw_status = None


def _get_gateway_status():
    """Lazy import gateway/status.py to avoid circular imports."""
    global _gw_status
    if _gw_status is None:
        from gateway import status as _gw_status
    return _gw_status


# Logger for this module
logger = logging.getLogger(__name__)

# Default API server port when not configured in profile's config.yaml
DEFAULT_API_SERVER_PORT = 8642


def _get_profile_home(profile_name: str) -> Optional[Path]:
    """
    Get the HERMES_HOME path for a given profile.

    Args:
        profile_name: Name of the profile directory

    Returns:
        Path to the profile's HERMES_HOME, or None if not found.
    """
    if not profile_exists(profile_name):
        return None
    return get_profile_dir(profile_name)


def discover_profiles() -> List[str]:
    """
    Discover all available profiles.

    Returns:
        List[str]: Sorted list of profile names.
    """
    # list_profiles() returns List[ProfileInfo], extract names
    profiles = list_profiles()
    return sorted([p.name for p in profiles])


def is_profile_running(profile_name: str) -> bool:
    """
    Check if a profile's gateway is currently running.

    Uses _check_gateway_running from hermes_cli.profiles which checks
    the PID file and process existence.

    Args:
        profile_name: Name of the profile to check

    Returns:
        True if the profile's gateway is running, False otherwise.
    """
    profile_home = _get_profile_home(profile_name)
    if profile_home is None:
        return False

    # Use the new _check_gateway_running from profiles.py
    return _check_gateway_running(profile_home)


def get_profile_health(profile_name: str) -> Dict[str, Any]:
    """
    Get health status information for a profile's gateway.

    Attempts to check:
    1. PID file existence and process liveness
    2. HTTP /health endpoint if port is configured

    Args:
        profile_name: Name of the profile to check

    Returns:
        Dict with keys:
        - running: bool — whether the gateway appears to be running
        - pid: int | None — the PID if running
        - port: int | None — the configured API server port
        - health_check: str — "pid_only", "http_healthy", "http_unreachable", "not_running"
        - error: str | None — error message if any
    """
    result: Dict[str, Any] = {
        "running": False,
        "pid": None,
        "port": None,
        "health_check": "not_running",
        "error": None,
    }

    profile_home = _get_profile_home(profile_name)
    if profile_home is None:
        result["error"] = f"Profile '{profile_name}' not found"
        return result

    # Get port from profile config
    port = resolve_profile_port(profile_name)
    result["port"] = port

    # Check PID file
    pid_path = profile_home / "gateway.pid"
    if not pid_path.exists():
        result["health_check"] = "not_running"
        return result

    try:
        raw = pid_path.read_text().strip()
        if not raw:
            result["health_check"] = "not_running"
            return result

        try:
            payload = json.loads(raw)
            pid = int(payload.get("pid", 0)) if isinstance(payload, dict) else int(raw)
        except (json.JSONDecodeError, ValueError):
            pid = int(raw)
    except (OSError, ValueError) as e:
        result["error"] = f"Failed to read PID: {e}"
        result["health_check"] = "not_running"
        return result

    if pid <= 0:
        result["health_check"] = "not_running"
        return result

    # Check process existence
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        result["health_check"] = "not_running"
        result["error"] = "PID file exists but process is not running"
        return result

    result["running"] = True
    result["pid"] = pid
    result["health_check"] = "pid_only"

    # Try HTTP health check if port is configured
    if port and port > 0:
        try:
            import urllib.request
            import urllib.error

            health_url = f"http://127.0.0.1:{port}/health"
            req = urllib.request.Request(health_url, method="GET")
            req.add_header("Accept", "application/json")
            try:
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if resp.status == 200:
                        result["health_check"] = "http_healthy"
                    else:
                        result["health_check"] = "http_unreachable"
            except (urllib.error.URLError, TimeoutError, OSError):
                # Process is running but health endpoint not responding
                result["health_check"] = "pid_only"
        except Exception as e:
            logger.debug("Health check failed for profile %s: %s", profile_name, e)
            # Don't set error - PID check already succeeded

    return result


def resolve_profile_port(profile_name: str) -> int:
    """
    Resolve the API server port for a profile.

    Reads the profile's config.yaml and extracts api_server.port.
    Falls back to DEFAULT_API_SERVER_PORT (8642) if not configured.

    Args:
        profile_name: Name of the profile

    Returns:
        The configured port number, or DEFAULT_API_SERVER_PORT if not set.
    """
    config = load_profile_config(profile_name)
    if config is None:
        return DEFAULT_API_SERVER_PORT

    # Check api_server.port in config
    api_server = config.get("api_server", {})
    if isinstance(api_server, dict):
        port = api_server.get("port")
        if port is not None:
            try:
                return int(port)
            except (ValueError, TypeError):
                pass

    # Also check API_SERVER_PORT env var pattern (uppercase, common convention)
    env_override = config.get("environment", {}).get("API_SERVER_PORT")
    if env_override is not None:
        try:
            return int(env_override)
        except (ValueError, TypeError):
            pass

    return DEFAULT_API_SERVER_PORT


def load_profile_config(profile_name: str) -> Optional[Dict[str, Any]]:
    """
    Load the full config.yaml for a profile.

    Args:
        profile_name: Name of the profile directory

    Returns:
        The parsed config.yaml as a dict, or None if the profile
        doesn't exist or config.yaml can't be read.
    """
    profile_home = _get_profile_home(profile_name)
    if profile_home is None:
        logger.debug("Profile '%s' not found", profile_name)
        return None

    config_path = profile_home / "config.yaml"
    if not config_path.exists():
        logger.debug("Profile '%s' has no config.yaml", profile_name)
        return None

    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg if isinstance(cfg, dict) else {}
    except (OSError, yaml.YAMLError) as e:
        logger.warning("Failed to load config for profile '%s': %s", profile_name, e)
        return None


def get_profile_info(profile_name: str) -> Dict[str, Any]:
    """
    Get a summary of information about a profile.

    Convenience function that combines discovery, health, and config data.

    Args:
        profile_name: Name of the profile

    Returns:
        Dict with keys:
        - name: str — the profile name
        - exists: bool — whether the profile directory exists
        - running: bool — whether the gateway is running
        - health: Dict from get_profile_health()
        - config: Dict from load_profile_config() (or None)
    """
    info: Dict[str, Any] = {
        "name": profile_name,
        "exists": profile_exists(profile_name),
        "running": False,
        "health": {},
        "config": None,
    }

    if not info["exists"]:
        return info

    info["running"] = is_profile_running(profile_name)
    info["health"] = get_profile_health(profile_name)
    info["config"] = load_profile_config(profile_name)

    return info


def list_all_profiles_with_status() -> List[Dict[str, Any]]:
    """
    Discover all profiles and return their status information.

    Returns:
        List of profile info dicts (see get_profile_info), sorted by name.
    """
    profiles = discover_profiles()
    return [get_profile_info(name) for name in profiles]


# ---------------------------------------------------------------------------
# Worker lifecycle management
# ---------------------------------------------------------------------------

import sys
import signal
import subprocess
import time


def start_profile_worker(profile_name: str) -> subprocess.Popen:
    """
    Spawn a gateway subprocess for the named profile.

    The worker runs as a daemon — stdout/stderr are redirected to DEVNULL.
    The profile's HERMES_HOME is set via the HERMES_HOME env var so the
    subprocess picks up the correct config.yaml, .env, and writes gateway.pid
    into the profile directory.

    Args:
        profile_name: Name of the profile to start.

    Returns:
        The ``subprocess.Popen`` handle of the spawned gateway process.

    Raises:
        ValueError: If the profile directory does not exist.
        RuntimeError: If the subprocess cannot be started.
    """
    profile_home = _get_profile_home(profile_name)
    if profile_home is None:
        raise ValueError(f"Profile '{profile_name}' not found")

    # Build the command: run the gateway in daemon mode
    # Using `python -m hermes_cli.main gateway run`
    cmd = [sys.executable, "-m", "hermes_cli.main", "gateway", "run"]

    env = os.environ.copy()
    env["HERMES_HOME"] = str(profile_home)
    # Ensure quiet mode and interactive exec approval for messaging platforms
    env["HERMES_QUIET"] = "1"
    env["HERMES_EXEC_ASK"] = "1"

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,  # detach so SIGTERM propagates correctly
        )
    except OSError as e:
        raise RuntimeError(
            f"Failed to spawn gateway for profile '{profile_name}': {e}"
        ) from e

    logger.info(
        "Started gateway worker for profile '%s' (pid=%d, home=%s)",
        profile_name,
        proc.pid,
        profile_home,
    )
    return proc


def wait_for_profile_ready(profile_name: str, timeout: int = 30) -> bool:
    """
    Poll /health on a profile's gateway until it responds or the timeout expires.

    Uses exponential backoff starting at 0.5s, capped at 4s, until the
    deadline imposed by ``timeout``.  Only HTTP 200 is considered "ready".

    Args:
        profile_name: Name of the profile whose gateway to wait for.
        timeout:     Maximum number of seconds to wait (default 30).

    Returns:
        True if the /health endpoint returned 200 within the timeout,
        False otherwise.
    """
    port = resolve_profile_port(profile_name)
    if not port or port <= 0:
        logger.debug(
            "No port resolved for profile '%s', skipping health poll", profile_name
        )
        return False

    health_url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout
    delay = 0.5

    while time.monotonic() < deadline:
        # Check process still exists before wasting a request
        if not is_profile_running(profile_name):
            logger.debug("Profile '%s' process is no longer running", profile_name)
            return False

        try:
            import urllib.request
            import urllib.error

            req = urllib.request.Request(health_url, method="GET")
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    logger.debug(
                        "Profile '%s' is ready at %s", profile_name, health_url
                    )
                    return True
        except Exception:
            pass  # any failure → keep waiting

        time.sleep(delay)
        delay = min(delay * 2, 4.0)

    logger.warning(
        "Profile '%s' did not become ready within %ds (last check: %s)",
        profile_name,
        timeout,
        health_url,
    )
    return False


def ensure_profile_running(profile_name: str, timeout: int = 30) -> bool:
    """
    Ensure a profile's gateway worker is running, starting it if necessary.

    If the profile's gateway is already running (verified via PID file),
    this returns ``True`` immediately without spawning a new process.
    Otherwise it calls :func:`start_profile_worker` and then waits for
    readiness via :func:`wait_for_profile_ready`.

    Args:
        profile_name: Name of the profile to ensure is running.
        timeout:      Maximum seconds to wait for startup (default 30).

    Returns:
        True if the profile's gateway is running (either already running
        or successfully started), False if startup failed or timed out.
    """
    if is_profile_running(profile_name):
        logger.debug("Profile '%s' is already running", profile_name)
        return True

    logger.info("Profile '%s' is not running, spawning worker", profile_name)
    try:
        proc = start_profile_worker(profile_name)
    except (ValueError, RuntimeError) as e:
        logger.error("Failed to start profile '%s': %s", profile_name, e)
        return False

    # Verify the process is still alive after a brief moment
    time.sleep(0.5)
    if proc.poll() is not None:
        logger.error(
            "Profile '%s' gateway process exited immediately with code %d",
            profile_name,
            proc.returncode,
        )
        return False

    return wait_for_profile_ready(profile_name, timeout=timeout)


def stop_profile_worker(profile_name: str, timeout: int = 10) -> bool:
    """
    Gracefully shut down a profile's gateway worker via SIGTERM.

    Sends SIGTERM to the process identified in the profile's gateway.pid
    file and waits up to ``timeout`` seconds for it to exit.  If the process
    does not exit cleanly it is killed with SIGKILL.

    Args:
        profile_name: Name of the profile whose worker to stop.
        timeout:     Seconds to wait for graceful shutdown (default 10).

    Returns:
        True if the worker exited cleanly (or was not running),
        False if it had to be killed or the PID could not be resolved.
    """
    profile_home = _get_profile_home(profile_name)
    if profile_home is None:
        logger.warning("Cannot stop profile '%s': not found", profile_name)
        return False

    pid_path = profile_home / "gateway.pid"
    if not pid_path.exists():
        logger.debug(
            "Profile '%s' has no gateway.pid, assuming not running", profile_name
        )
        return True

    try:
        raw = pid_path.read_text().strip()
        if not raw:
            return True

        try:
            payload = json.loads(raw)
            pid = int(payload.get("pid", 0)) if isinstance(payload, dict) else int(raw)
        except (json.JSONDecodeError, ValueError):
            pid = int(raw)
    except (OSError, ValueError) as e:
        logger.error("Failed to read PID for profile '%s': %s", profile_name, e)
        return False

    if pid <= 0:
        return True

    try:
        # Check process exists before sending signal
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        logger.debug("Profile '%s' PID %d already gone", profile_name, pid)
        return True

    logger.info("Sending SIGTERM to profile '%s' (pid=%d)", profile_name, pid)
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        logger.debug("Profile '%s' PID %d already gone", profile_name, pid)
        return True

    # Wait for graceful exit
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            logger.info("Profile '%s' (pid=%d) exited cleanly", profile_name, pid)
            return True
        time.sleep(0.25)

    # Timeout expired — force kill
    logger.warning(
        "Profile '%s' (pid=%d) did not exit after %ds, sending SIGKILL",
        profile_name,
        pid,
        timeout,
    )
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass  # already gone

    return False


# ---------------------------------------------------------------------------
# HTTP client for cross-profile dispatch
# ---------------------------------------------------------------------------

import time
import uuid

# In-memory task store for async dispatch (persistence comes in T5)
_profile_tasks: Dict[str, Dict[str, Any]] = {}


def _build_chat_request(
    message: str,
    context: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Build an OpenAI-compatible /v1/chat/completions request body.

    Args:
        message: The current user message to send.
        context: Optional context dict. If provided, is included as a
            ``context`` field in the request (not part of the OpenAI spec,
            but the receiving profile's api_server handler can read it).
        conversation_history: Optional list of previous messages (user + assistant
            exchanges) to include in the conversation. Each dict should have
            "role" and "content" keys.

    Returns:
        A dict suitable for posting as JSON to /v1/chat/completions.
    """
    # Build messages array with full conversation history
    messages: List[Dict[str, str]] = []

    # Add conversation history if provided
    if conversation_history:
        messages.extend(conversation_history)
        logger.info(
            f"[DEBUG] _build_chat_request: Added {len(conversation_history)} history messages"
        )
    else:
        logger.info("[DEBUG] _build_chat_request: No conversation_history provided")

    # Add current message
    messages.append({"role": "user", "content": message})
    logger.info(f"[DEBUG] _build_chat_request: Final messages count: {len(messages)}")

    body: Dict[str, Any] = {
        "model": "hermes-agent",
        "messages": messages,
    }
    if context:
        body["context"] = context
    return body


def get_profile_api_key(profile_name: str) -> Optional[str]:
    """
    Load the Bearer token from a profile's .env file.

    Checks ``HERMES_API_KEY`` (preferred) then ``API_SERVER_KEY`` then
    ``OPENAI_API_KEY`` in the profile's .env.  Returns None if the profile
    does not exist or none of those keys are set.

    Args:
        profile_name: Name of the profile directory.

    Returns:
        The API key string, or None if not found.
    """
    profile_home = _get_profile_home(profile_name)
    if profile_home is None:
        logger.debug("Profile '%s' not found", profile_name)
        return None

    env_path = profile_home / ".env"
    if not env_path.exists():
        logger.debug("Profile '%s' has no .env file", profile_name)
        return None

    try:
        env_content = env_path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Failed to read .env for profile '%s': %s", profile_name, e)
        return None

    # Parse .env manually (avoiding external dependencies)
    for line in env_content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip surrounding quotes
        if len(value) >= 2 and (value[0] == value[-1]) and value[0] in ('"', "'"):
            value = value[1:-1]
        if key in ("HERMES_API_KEY", "API_SERVER_KEY", "OPENAI_API_KEY"):
            if value:
                logger.debug("Loaded API key '%s' for profile '%s'", key, profile_name)
                return value

    logger.debug("No known API key found in .env for profile '%s'", profile_name)
    return None


def send_to_profile(
    profile_name: str,
    message: str,
    context: Optional[Dict[str, Any]] = None,
    wait: bool = True,
    timeout: float = 300,
    source_profile: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    allow_subagents: bool = True,
    max_depth: int = 2,
) -> Dict[str, Any]:
    """
    Send a message to a profile's gateway via HTTP POST (blocking).

    POSTs to ``http://127.0.0.1:{port}/v1/chat/completions`` using an
    OpenAI-compatible request body.  Authentication uses the Bearer token
    from the profile's .env (see :func:`get_profile_api_key`).

    Args:
        profile_name: Target profile name.
        message:      User message to send.
        context:      Optional context dict forwarded to the request body.
        wait:         Ignored (present for API compatibility with
                      ``send_to_profile_async``).  This function always blocks.
        timeout:      Maximum seconds to wait for the HTTP response
                      (default 300).
        conversation_history: Optional list of previous messages to include
                      in the conversation for context continuity.
        allow_subagents: Whether the delegated agent may spawn one-off
                      subagents via delegate_task (default True).
        max_depth:    Maximum delegation depth (default 2). Prevents infinite
                      delegation chains.

    Returns:
        Dict with ``content`` from response and ``session_id``.

    Raises:
        ValueError: If the profile is not running or the port cannot be resolved.
        RuntimeError: If the HTTP request fails or returns a non-2xx status.
    """
    port = resolve_profile_port(profile_name)
    if not port or port <= 0:
        raise ValueError(f"Cannot resolve port for profile '{profile_name}'")

    if not is_profile_running(profile_name):
        raise ValueError(
            f"Profile '{profile_name}' is not running. "
            f"Start it with: hermes -p {profile_name} gateway start "
            f"(or hermes -p {profile_name} chat)"
        )

    api_key = get_profile_api_key(profile_name)
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Add delegation headers for inter-profile communication
    if source_profile:
        import uuid

        task_id = str(uuid.uuid4())[:8]
        headers["X-Hermes-Delegation"] = "true"
        headers["X-Hermes-Source-Profile"] = source_profile
        headers["X-Hermes-Task-ID"] = task_id
        headers["X-Hermes-Delegate-Depth"] = (
            "0"  # Initial depth, API server will increment
        )
        headers["X-Hermes-Allow-Subagents"] = "true" if allow_subagents else "false"
        headers["X-Hermes-Max-Depth"] = str(max_depth)

    body = _build_chat_request(message, context, conversation_history)
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    try:
        import urllib.request
        import urllib.error

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            error_body = e.read().decode("utf-8")
        except Exception:
            error_body = ""
        raise RuntimeError(
            f"HTTP {e.code} from profile '{profile_name}': {error_body}"
        ) from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise RuntimeError(
            f"Connection error contacting profile '{profile_name}' at {url}: {e}"
        ) from e

    choices = result.get("choices")
    if not choices or not isinstance(choices, list):
        raise RuntimeError(
            f"Unexpected response format from profile '{profile_name}': {result}"
        )

    content = choices[0].get("message", {}).get("content", "")
    return content


async def send_to_profile_async(
    profile_name: str,
    message: str,
    context: Optional[Dict[str, Any]] = None,
    source_profile: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    allow_subagents: bool = True,
    max_depth: int = 2,
) -> str:
    """
    Send a message to a profile's gateway via HTTP POST (non-blocking).

    Uses ThreadPoolExecutor for simplicity and reliability.
    Returns a task_id for polling via collect_profile_result.
    """
    import concurrent.futures
    import uuid

    task_id = str(uuid.uuid4())
    result_container: Dict[str, Any] = {}

    def _sync_request():
        try:
            result = send_to_profile(
                profile_name,
                message,
                context,
                conversation_history=conversation_history,
                allow_subagents=allow_subagents,
                max_depth=max_depth,
            )
            result_container["result"] = result
            return result
        except Exception as e:
            result_container["error"] = str(e)
            raise

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_sync_request)

    _profile_tasks[task_id] = {
        "profile_name": profile_name,
        "future": future,
        "started_at": time.time(),
    }
    executor.shutdown(wait=False)

    def _on_task_done(fut):
        try:
            from hermes_state import SessionDB

            db = SessionDB()
            try:
                result = fut.result()
                db.update_orchestration_task(
                    task_id=task_id,
                    status="completed",
                    result=result,
                )
            except Exception as e:
                db.update_orchestration_task(
                    task_id=task_id,
                    status="error",
                    error_message=str(e)[:500],
                )
        except Exception as e:
            logger.debug(f"Failed to update SQLite for task {task_id}: {e}")

    future.add_done_callback(_on_task_done)

    try:
        from hermes_state import SessionDB

        db = SessionDB()
        task_context = {
            "conversation_history": conversation_history,
            "source_profile": source_profile,
            "message": message,
        }
        db.create_orchestration_task(
            task_id=task_id,
            parent_session_id=None,
            target_profile=profile_name,
            goal=message,
            context=json.dumps(task_context),
        )
    except Exception as e:
        logger.debug(f"Failed to store orchestration task: {e}")

    return task_id

    port = resolve_profile_port(profile_name)
    if not port or port <= 0:
        raise ValueError(f"Cannot resolve port for profile '{profile_name}'")

    if not is_profile_running(profile_name):
        raise ValueError(
            f"Profile '{profile_name}' is not running. "
            f"Start it with: hermes -p {profile_name} gateway start "
            f"(or hermes -p {profile_name} chat)"
        )

    api_key = get_profile_api_key(profile_name)
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = _build_chat_request(message, context, conversation_history)
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    task_id = str(uuid.uuid4())

    # Store conversation_history in orchestration_tasks for async retrieval
    try:
        from hermes_state import SessionDB

        db = SessionDB()
        # Serialize conversation_history as JSON in context field
        task_context = {
            "conversation_history": conversation_history,
            "source_profile": source_profile,
            "message": message,
        }
        db.create_orchestration_task(
            task_id=task_id,
            parent_session_id=None,
            target_profile=profile_name,
            goal=message,
            context=json.dumps(task_context),
        )
    except Exception as e:
        logger.warning(f"Failed to store orchestration task: {e}")

    async def _make_request():
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                result = await resp.json()
                choices = result.get("choices")
                if not choices or not isinstance(choices, list):
                    raise RuntimeError(f"Unexpected response format: {result}")
                return choices[0].get("message", {}).get("content", "")

    future = asyncio.ensure_future(_make_request())

    _profile_tasks[task_id] = {
        "profile_name": profile_name,
        "future": future,
        "started_at": time.time(),
    }

    return task_id


def collect_profile_result(
    task_id: str,
    block: bool = False,
    timeout: float = 300,
) -> Optional[str]:
    """
    Poll for the result of an async profile dispatch task.

    When ``block=True`` this function blocks until the task completes or
    the ``timeout`` is reached.  When ``block=False`` it returns immediately
    with the result if ready or ``None`` if the task is still running.

    Args:
        task_id: The task ID returned by ``send_to_profile_async``.
        block:   Whether to block waiting for completion (default False).
        timeout: Maximum seconds to wait when ``block=True`` (default 300).

    Returns:
        The response content string if the task is complete, or ``None`` if
        ``block=False`` and the task has not yet finished.  Raises
        ``KeyError`` if the ``task_id`` is not known.
    """
    import concurrent.futures

    if task_id not in _profile_tasks:
        raise KeyError(f"Unknown task_id: {task_id}")

    entry = _profile_tasks[task_id]
    future = entry["future"]

    # Check cancelled flag first
    if entry.get("_cancelled", False):
        return None

    # Handle concurrent.futures.Future
    if isinstance(future, concurrent.futures.Future):
        if not block:
            if not future.done():
                return None
        try:
            result = future.result(timeout=timeout if block else None)
            return result
        except concurrent.futures.TimeoutError:
            return None
        except Exception as e:
            raise RuntimeError(f"Task '{task_id}' raised: {e}") from e

    raise RuntimeError(
        f"Task '{task_id}' has an unsupported future type: {type(future)}"
    )


# =============================================================================
# Honcho Integration for Cross-Session Memory
# =============================================================================


def save_delegation_to_honcho(
    source_profile: str,
    task_id: str,
    goal: str,
    context: Optional[str] = None,
    status: str = "received",
    result: Optional[str] = None,
) -> bool:
    """
    Save delegation information to Honcho for cross-session memory.

    This allows a profile to remember tasks received from other profiles
    even after the chat session ends. When the profile is asked directly
    later, it can retrieve this information using honcho_search or
    honcho_context.

    Args:
        source_profile: Name of the profile that sent the task (e.g., 'p3')
        task_id: Unique task identifier
        goal: The task description/goal
        context: Additional context provided by source
        status: Task status ("received", "in_progress", "completed", "failed")
        result: Task result (if completed)

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Import honcho tools (may not be available if honcho not enabled)
        from tools.honcho_tools import honcho_context

        # Store task with metadata
        task_data = {
            "source": source_profile,
            "task_id": task_id,
            "goal": goal,
            "context": context,
            "status": status,
            "result": result,
            "type": "profile_delegation",
        }

        # Save to honcho with structured key
        key = f"delegation_from_{source_profile}_{task_id}"
        honcho_context(
            action="store",
            key=key,
            value=json.dumps(task_data),
        )

        # Also update pending tasks list
        _update_pending_tasks_list(source_profile, task_id, goal, status)

        logger.info(f"Saved delegation from {source_profile} to honcho: {task_id}")
        return True

    except ImportError:
        logger.debug("Honcho tools not available, skipping delegation save")
        return False
    except Exception as e:
        logger.warning(f"Failed to save delegation to honcho: {e}")
        return False


def _update_pending_tasks_list(
    source_profile: str,
    task_id: str,
    goal: str,
    status: str,
) -> None:
    """Update the list of pending tasks from other profiles."""
    try:
        from tools.honcho_tools import honcho_context

        # Retrieve existing pending tasks
        existing = honcho_context(action="retrieve", key="pending_profile_tasks")
        tasks = []
        if existing:
            try:
                tasks = json.loads(existing)
            except json.JSONDecodeError:
                tasks = []

        # Add or update task
        task_entry = {
            "source": source_profile,
            "task_id": task_id,
            "goal": goal,
            "status": status,
            "timestamp": time.time(),
        }

        # Remove if already exists (update case)
        tasks = [t for t in tasks if t.get("task_id") != task_id]

        # Add if not completed
        if status != "completed":
            tasks.append(task_entry)

        # Save back
        honcho_context(
            action="store",
            key="pending_profile_tasks",
            value=json.dumps(tasks),
        )

    except Exception as e:
        logger.debug(f"Failed to update pending tasks list: {e}")


def get_delegation_history_from_honcho(
    source_profile: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve delegation history from Honcho.

    Args:
        source_profile: Filter by source profile (e.g., 'p3'), or None for all
        status: Filter by status, or None for all

    Returns:
        List of task dictionaries with source, task_id, goal, status, etc.
    """
    try:
        from tools.honcho_tools import honcho_search

        # Search for delegation entries
        query = "profile_delegation"
        if source_profile:
            query += f" {source_profile}"

        results = honcho_search(query=query, limit=50)

        tasks = []
        for result in results:
            try:
                task_data = json.loads(result.get("value", "{}"))
                if task_data.get("type") == "profile_delegation":
                    # Apply filters
                    if source_profile and task_data.get("source") != source_profile:
                        continue
                    if status and task_data.get("status") != status:
                        continue
                    tasks.append(task_data)
            except json.JSONDecodeError:
                continue

        return sorted(tasks, key=lambda x: x.get("timestamp", 0), reverse=True)

    except ImportError:
        logger.debug("Honcho tools not available, returning empty history")
        return []
    except Exception as e:
        logger.warning(f"Failed to retrieve delegation history: {e}")
        return []


def update_delegation_status_in_honcho(
    source_profile: str,
    task_id: str,
    new_status: str,
    result: Optional[str] = None,
) -> bool:
    """
    Update the status of a delegation in Honcho.

    Args:
        source_profile: Profile that sent the task
        task_id: Task identifier
        new_status: New status ("in_progress", "completed", "failed")
        result: Task result (if completed)

    Returns:
        True if updated successfully, False otherwise
    """
    try:
        from tools.honcho_tools import honcho_context

        # Retrieve existing task
        key = f"delegation_from_{source_profile}_{task_id}"
        existing = honcho_context(action="retrieve", key=key)

        if not existing:
            logger.warning(f"Task {task_id} from {source_profile} not found in honcho")
            return False

        # Update status
        task_data = json.loads(existing)
        task_data["status"] = new_status
        if result is not None:
            task_data["result"] = result
        task_data["updated_at"] = time.time()

        # Save back
        honcho_context(
            action="store",
            key=key,
            value=json.dumps(task_data),
        )

        # Update pending tasks list
        _update_pending_tasks_list(
            source_profile,
            task_id,
            task_data.get("goal", ""),
            new_status,
        )

        logger.info(f"Updated delegation status in honcho: {task_id} -> {new_status}")
        return True

    except ImportError:
        return False
    except Exception as e:
        logger.warning(f"Failed to update delegation status: {e}")
        return False


def get_pending_tasks_from_honcho() -> List[Dict[str, Any]]:
    """
    Get all pending tasks from other profiles.

    Returns:
        List of pending task dictionaries
    """
    try:
        from tools.honcho_tools import honcho_context

        pending = honcho_context(action="retrieve", key="pending_profile_tasks")
        if pending:
            tasks = json.loads(pending)
            # Filter out completed tasks
            return [t for t in tasks if t.get("status") != "completed"]
        return []

    except ImportError:
        return []
    except Exception as e:
        logger.warning(f"Failed to get pending tasks: {e}")
        return []

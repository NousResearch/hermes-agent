#!/usr/bin/env python3
"""
Browser Tool Module

This module provides browser automation tools using agent-browser CLI with
Browserbase cloud execution. It enables AI agents to navigate websites,
interact with page elements, and extract information in a text-based format.

The tool uses agent-browser's accessibility tree (ariaSnapshot) for text-based
page representation, making it ideal for LLM agents without vision capabilities.

Features:
- Cloud browser execution via Browserbase (no local browser needed)
- Basic Stealth Mode always active (random fingerprints, CAPTCHA solving)
- Proxies enabled by default for better CAPTCHA solving and anti-bot avoidance
- Session isolation per task ID
- Text-based page snapshots using accessibility tree
- Element interaction via ref selectors (@e1, @e2, etc.)
- Task-aware content extraction using LLM summarization
- Automatic cleanup of browser sessions

Environment Variables:
- BROWSERBASE_API_KEY: API key for Browserbase (required)
- BROWSERBASE_PROJECT_ID: Project ID for Browserbase (required)
- BROWSERBASE_PROXIES: Enable/disable residential proxies (default: "true")
- BROWSERBASE_ADVANCED_STEALTH: Enable advanced stealth mode with custom Chromium,
  requires Scale Plan (default: "false")
- BROWSERBASE_KEEP_ALIVE: Enable keepAlive for session reconnection after disconnects,
  requires paid plan (default: "true")
- BROWSERBASE_SESSION_TIMEOUT: Custom session timeout in milliseconds. Set to extend
  beyond project default. Common values: 600000 (10min), 1800000 (30min) (default: none)

Usage:
    from tools.browser_tool import browser_navigate, browser_snapshot, browser_click
    
    # Navigate to a page
    result = browser_navigate("https://example.com", task_id="task_123")
    
    # Get page snapshot
    snapshot = browser_snapshot(task_id="task_123")
    
    # Click an element
    browser_click("@e5", task_id="task_123")
"""

import atexit
import json
import logging
import os
import signal
import subprocess
import shutil
import sys
import tempfile
import threading
import time
import requests
from typing import Dict, Any, Optional, List
from pathlib import Path
from agent.auxiliary_client import get_vision_auxiliary_client

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Default timeout for browser commands (seconds)
DEFAULT_COMMAND_TIMEOUT = 30

# Default session timeout (seconds)
DEFAULT_SESSION_TIMEOUT = 300

# Max tokens for snapshot content before summarization
SNAPSHOT_SUMMARIZE_THRESHOLD = 8000

# Resolve vision auxiliary client for extraction/vision tasks
_aux_vision_client, EXTRACTION_MODEL = get_vision_auxiliary_client()

# Track active sessions per task
# Now stores tuple of (session_name, browserbase_session_id, cdp_url)
_active_sessions: Dict[str, Dict[str, str]] = {}  # task_id -> {session_name, bb_session_id, cdp_url}

# Flag to track if cleanup has been done
_cleanup_done = False

# =============================================================================
# Inactivity Timeout Configuration
# =============================================================================

# Session inactivity timeout (seconds) - cleanup if no activity for this long
# Default: 5 minutes. Needs headroom for LLM reasoning between browser commands,
# especially when subagents are doing multi-step browser tasks.
BROWSER_SESSION_INACTIVITY_TIMEOUT = int(os.environ.get("BROWSER_INACTIVITY_TIMEOUT", "300"))

# Track last activity time per session
_session_last_activity: Dict[str, float] = {}

# Background cleanup thread state
_cleanup_thread = None
_cleanup_running = False
# Protects _session_last_activity AND _active_sessions for thread safety
# (subagents run concurrently via ThreadPoolExecutor)
_cleanup_lock = threading.Lock()


def _emergency_cleanup_all_sessions():
    """
    Emergency cleanup of all active browser sessions.
    Called on process exit or interrupt to prevent orphaned sessions.
    """
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True
    
    if not _active_sessions:
        return
    
    logger.info("Emergency cleanup: closing %s active session(s)...", len(_active_sessions))
    
    try:
        api_key = os.environ.get("BROWSERBASE_API_KEY")
        project_id = os.environ.get("BROWSERBASE_PROJECT_ID")
        
        if not api_key or not project_id:
            logger.warning("Cannot cleanup - missing BROWSERBASE credentials")
            return
        
        for task_id, session_info in list(_active_sessions.items()):
            bb_session_id = session_info.get("bb_session_id")
            if bb_session_id:
                try:
                    response = requests.post(
                        f"https://api.browserbase.com/v1/sessions/{bb_session_id}",
                        headers={
                            "X-BB-API-Key": api_key,
                            "Content-Type": "application/json"
                        },
                        json={
                            "projectId": project_id,
                            "status": "REQUEST_RELEASE"
                        },
                        timeout=5  # Short timeout for cleanup
                    )
                    if response.status_code in (200, 201, 204):
                        logger.info("Closed session %s", bb_session_id)
                    else:
                        logger.warning("Failed to close session %s: HTTP %s", bb_session_id, response.status_code)
                except Exception as e:
                    logger.error("Error closing session %s: %s", bb_session_id, e)
        
        _active_sessions.clear()
    except Exception as e:
        logger.error("Emergency cleanup error: %s", e)


def _signal_handler(signum, frame):
    """Handle interrupt signals to cleanup sessions before exit."""
    logger.warning("Received signal %s, cleaning up...", signum)
    _emergency_cleanup_all_sessions()
    sys.exit(128 + signum)


# Register cleanup handlers
atexit.register(_emergency_cleanup_all_sessions)

# Only register signal handlers in main process (not in multiprocessing workers)
try:
    if os.getpid() == os.getpgrp():  # Main process check
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
except (OSError, AttributeError):
    pass  # Signal handling not available (e.g., Windows or worker process)


# =============================================================================
# Inactivity Cleanup Functions
# =============================================================================

def _cleanup_inactive_browser_sessions():
    """
    Clean up browser sessions that have been inactive for longer than the timeout.
    
    This function is called periodically by the background cleanup thread to
    automatically close sessions that haven't been used recently, preventing
    orphaned Browserbase sessions from accumulating.
    """
    current_time = time.time()
    sessions_to_cleanup = []
    
    with _cleanup_lock:
        for task_id, last_time in list(_session_last_activity.items()):
            if current_time - last_time > BROWSER_SESSION_INACTIVITY_TIMEOUT:
                sessions_to_cleanup.append(task_id)
    
    for task_id in sessions_to_cleanup:
        try:
            elapsed = int(current_time - _session_last_activity.get(task_id, current_time))
            logger.info("Cleaning up inactive session for task: %s (inactive for %ss)", task_id, elapsed)
            cleanup_browser(task_id)
            with _cleanup_lock:
                if task_id in _session_last_activity:
                    del _session_last_activity[task_id]
        except Exception as e:
            logger.warning("Error cleaning up inactive session %s: %s", task_id, e)


def _browser_cleanup_thread_worker():
    """
    Background thread that periodically cleans up inactive browser sessions.
    
    Runs every 30 seconds and checks for sessions that haven't been used
    within the BROWSER_SESSION_INACTIVITY_TIMEOUT period.
    """
    global _cleanup_running
    
    while _cleanup_running:
        try:
            _cleanup_inactive_browser_sessions()
        except Exception as e:
            logger.warning("Cleanup thread error: %s", e)
        
        # Sleep in 1-second intervals so we can stop quickly if needed
        for _ in range(30):
            if not _cleanup_running:
                break
            time.sleep(1)


def _start_browser_cleanup_thread():
    """Start the background cleanup thread if not already running."""
    global _cleanup_thread, _cleanup_running
    
    with _cleanup_lock:
        if _cleanup_thread is None or not _cleanup_thread.is_alive():
            _cleanup_running = True
            _cleanup_thread = threading.Thread(
                target=_browser_cleanup_thread_worker,
                daemon=True,
                name="browser-cleanup"
            )
            _cleanup_thread.start()
            logger.info("Started inactivity cleanup thread (timeout: %ss)", BROWSER_SESSION_INACTIVITY_TIMEOUT)


def _stop_browser_cleanup_thread():
    """Stop the background cleanup thread."""
    global _cleanup_running
    _cleanup_running = False
    if _cleanup_thread is not None:
        _cleanup_thread.join(timeout=5)


def _update_session_activity(task_id: str):
    """Update the last activity timestamp for a session."""
    with _cleanup_lock:
        _session_last_activity[task_id] = time.time()


# Register cleanup thread stop on exit
atexit.register(_stop_browser_cleanup_thread)


# ============================================================================
# Tool Schemas
# ============================================================================

BROWSER_TOOL_SCHEMAS = [
    {
        "name": "browser_navigate",
        "description": "Navigate to a URL in the browser. Initializes the session and loads the page. Must be called before other browser tools. For simple information retrieval, prefer web_search or web_extract (faster, cheaper). Use browser tools when you need to interact with a page (click, fill forms, dynamic content).",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to navigate to (e.g., 'https://example.com')"
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "browser_snapshot",
        "description": "Get a text-based snapshot of the current page's accessibility tree. Returns interactive elements with ref IDs (like @e1, @e2) for browser_click and browser_type. full=false (default): compact view with interactive elements. full=true: complete page content. Snapshots over 8000 chars are truncated or LLM-summarized. Requires browser_navigate first.",
        "parameters": {
            "type": "object",
            "properties": {
                "full": {
                    "type": "boolean",
                    "description": "If true, returns complete page content. If false (default), returns compact view with interactive elements only.",
                    "default": False
                }
            },
            "required": []
        }
    },
    {
        "name": "browser_click",
        "description": "Click on an element identified by its ref ID from the snapshot (e.g., '@e5'). The ref IDs are shown in square brackets in the snapshot output. Requires browser_navigate and browser_snapshot to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "The element reference from the snapshot (e.g., '@e5', '@e12')"
                }
            },
            "required": ["ref"]
        }
    },
    {
        "name": "browser_type",
        "description": "Type text into an input field identified by its ref ID. Clears the field first, then types the new text. Requires browser_navigate and browser_snapshot to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "The element reference from the snapshot (e.g., '@e3')"
                },
                "text": {
                    "type": "string",
                    "description": "The text to type into the field"
                }
            },
            "required": ["ref", "text"]
        }
    },
    {
        "name": "browser_scroll",
        "description": "Scroll the page in a direction. Use this to reveal more content that may be below or above the current viewport. Requires browser_navigate to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down"],
                    "description": "Direction to scroll"
                }
            },
            "required": ["direction"]
        }
    },
    {
        "name": "browser_back",
        "description": "Navigate back to the previous page in browser history. Requires browser_navigate to be called first.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "browser_press",
        "description": "Press a keyboard key. Useful for submitting forms (Enter), navigating (Tab), or keyboard shortcuts. Requires browser_navigate to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Key to press (e.g., 'Enter', 'Tab', 'Escape', 'ArrowDown')"
                }
            },
            "required": ["key"]
        }
    },
    {
        "name": "browser_close",
        "description": "Close the browser session and release resources. Call this when done with browser tasks to free up Browserbase session quota.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "browser_get_images",
        "description": "Get a list of all images on the current page with their URLs and alt text. Useful for finding images to analyze with the vision tool. Requires browser_navigate to be called first.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "browser_vision",
        "description": "Take a screenshot of the current page and analyze it with vision AI. Use this when you need to visually understand what's on the page - especially useful for CAPTCHAs, visual verification challenges, complex layouts, or when the text snapshot doesn't capture important visual information. Requires browser_navigate to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "What you want to know about the page visually. Be specific about what you're looking for."
                }
            },
            "required": ["question"]
        }
    },
]


# ============================================================================
# Utility Functions
# ============================================================================

def _create_browserbase_session(task_id: str) -> Dict[str, str]:
    """
    Create a Browserbase session with stealth features.
    
    Browserbase Stealth Modes:
    - Basic Stealth: ALWAYS enabled automatically. Generates random fingerprints,
      viewports, and solves visual CAPTCHAs. No configuration needed.
    - Advanced Stealth: Uses custom Chromium build for better bot detection avoidance.
      Requires Scale Plan. Enable via BROWSERBASE_ADVANCED_STEALTH=true.
    
    Proxies are enabled by default to route traffic through residential IPs,
    which significantly improves CAPTCHA solving rates. Can be disabled via
    BROWSERBASE_PROXIES=false if needed.
    
    Args:
        task_id: Unique identifier for the task
        
    Returns:
        Dict with session_name, bb_session_id, cdp_url, and feature flags
    """
    import uuid
    import sys
    
    config = _get_browserbase_config()
    
    # Check for optional settings from environment
    # Proxies: enabled by default for better CAPTCHA solving
    enable_proxies = os.environ.get("BROWSERBASE_PROXIES", "true").lower() != "false"
    # Advanced Stealth: requires Scale Plan, disabled by default
    enable_advanced_stealth = os.environ.get("BROWSERBASE_ADVANCED_STEALTH", "false").lower() == "true"
    # keepAlive: enabled by default (requires paid plan) - allows reconnection after disconnects
    enable_keep_alive = os.environ.get("BROWSERBASE_KEEP_ALIVE", "true").lower() != "false"
    # Custom session timeout in milliseconds (optional) - extends session beyond project default
    custom_timeout_ms = os.environ.get("BROWSERBASE_SESSION_TIMEOUT")
    
    # Track which features are actually enabled for logging/debugging
    features_enabled = {
        "basic_stealth": True,  # Always on
        "proxies": False,
        "advanced_stealth": False,
        "keep_alive": False,
        "custom_timeout": False,
    }
    
    # Build session configuration
    # Note: Basic stealth mode is ALWAYS active - no configuration needed
    session_config = {
        "projectId": config["project_id"],
    }
    
    # Enable keepAlive for session reconnection (default: true, requires paid plan)
    # Allows reconnecting to the same session after network hiccups
    if enable_keep_alive:
        session_config["keepAlive"] = True
    
    # Add custom timeout if specified (in milliseconds)
    # This extends session duration beyond project's default timeout
    if custom_timeout_ms:
        try:
            timeout_val = int(custom_timeout_ms)
            if timeout_val > 0:
                session_config["timeout"] = timeout_val
        except ValueError:
            logger.warning("Invalid BROWSERBASE_SESSION_TIMEOUT value: %s", custom_timeout_ms)
    
    # Enable proxies for better CAPTCHA solving (default: true)
    # Routes traffic through residential IPs for more reliable access
    if enable_proxies:
        session_config["proxies"] = True
    
    # Add advanced stealth if enabled (requires Scale Plan)
    # Uses custom Chromium build to avoid bot detection altogether
    if enable_advanced_stealth:
        session_config["browserSettings"] = {
            "advancedStealth": True,
        }
    
    # Create session via Browserbase API
    response = requests.post(
        "https://api.browserbase.com/v1/sessions",
        headers={
            "Content-Type": "application/json",
            "X-BB-API-Key": config["api_key"],
        },
        json=session_config,
        timeout=30
    )
    
    # Track if we fell back from paid features
    proxies_fallback = False
    keepalive_fallback = False
    
    # Handle 402 Payment Required - likely paid features not available
    # Try to identify which feature caused the issue and retry without it
    if response.status_code == 402:
        # First try without keepAlive (most likely culprit for paid plan requirement)
        if enable_keep_alive:
            keepalive_fallback = True
            logger.warning("keepAlive may require paid plan (402), retrying without it. "
                          "Sessions may timeout during long operations.")
            session_config.pop("keepAlive", None)
            response = requests.post(
                "https://api.browserbase.com/v1/sessions",
                headers={
                    "Content-Type": "application/json",
                    "X-BB-API-Key": config["api_key"],
                },
                json=session_config,
                timeout=30
            )
        
        # If still 402, try without proxies too
        if response.status_code == 402 and enable_proxies:
            proxies_fallback = True
            logger.warning("Proxies unavailable (402), retrying without proxies. "
                          "Bot detection may be less effective.")
            session_config.pop("proxies", None)
            response = requests.post(
                "https://api.browserbase.com/v1/sessions",
                headers={
                    "Content-Type": "application/json",
                    "X-BB-API-Key": config["api_key"],
                },
                json=session_config,
                timeout=30
            )
    
    if not response.ok:
        raise RuntimeError(f"Failed to create Browserbase session: {response.status_code} {response.text}")
    
    session_data = response.json()
    session_name = f"hermes_{task_id}_{uuid.uuid4().hex[:8]}"
    
    # Update features based on what actually succeeded
    if enable_proxies and not proxies_fallback:
        features_enabled["proxies"] = True
    if enable_advanced_stealth:
        features_enabled["advanced_stealth"] = True
    if enable_keep_alive and not keepalive_fallback:
        features_enabled["keep_alive"] = True
    if custom_timeout_ms and "timeout" in session_config:
        features_enabled["custom_timeout"] = True
    
    # Log session info for debugging
    feature_str = ", ".join(k for k, v in features_enabled.items() if v)
    logger.info("Created session %s with features: %s", session_name, feature_str)
    
    return {
        "session_name": session_name,
        "bb_session_id": session_data["id"],
        "cdp_url": session_data["connectUrl"],
        "features": features_enabled,
    }


def _get_session_info(task_id: Optional[str] = None) -> Dict[str, str]:
    """
    Get or create session info for the given task.
    
    Creates a Browserbase session with proxies enabled if one doesn't exist.
    Also starts the inactivity cleanup thread and updates activity tracking.
    Thread-safe: multiple subagents can call this concurrently.
    
    Args:
        task_id: Unique identifier for the task
        
    Returns:
        Dict with session_name, bb_session_id, and cdp_url
    """
    if task_id is None:
        task_id = "default"
    
    # Start the cleanup thread if not running (handles inactivity timeouts)
    _start_browser_cleanup_thread()
    
    # Update activity timestamp for this session
    _update_session_activity(task_id)
    
    with _cleanup_lock:
        # Check if we already have a session for this task
        if task_id in _active_sessions:
            return _active_sessions[task_id]
    
    # Create session outside the lock (network call - don't hold lock during I/O)
    session_info = _create_browserbase_session(task_id)
    
    with _cleanup_lock:
        _active_sessions[task_id] = session_info
    
    return session_info


def _get_session_name(task_id: Optional[str] = None) -> str:
    """
    Get the session name for agent-browser CLI.
    
    Args:
        task_id: Unique identifier for the task
        
    Returns:
        Session name for agent-browser
    """
    session_info = _get_session_info(task_id)
    return session_info["session_name"]


def _get_browserbase_config() -> Dict[str, str]:
    """
    Get Browserbase configuration from environment.
    
    Returns:
        Dict with api_key and project_id
        
    Raises:
        ValueError: If required env vars are not set
    """
    api_key = os.environ.get("BROWSERBASE_API_KEY")
    project_id = os.environ.get("BROWSERBASE_PROJECT_ID")
    
    if not api_key or not project_id:
        raise ValueError(
            "BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID environment variables are required. "
            "Get your credentials at https://browserbase.com"
        )
    
    return {
        "api_key": api_key,
        "project_id": project_id
    }


def _find_agent_browser() -> str:
    """
    Find the agent-browser CLI executable.
    
    Checks in order: PATH, local node_modules/.bin/, npx fallback.
    
    Returns:
        Path to agent-browser executable
        
    Raises:
        FileNotFoundError: If agent-browser is not installed
    """

    # Check if it's in PATH (global install)
    which_result = shutil.which("agent-browser")
    if which_result:
        return which_result
    
    # Check local node_modules/.bin/ (npm install in repo root)
    repo_root = Path(__file__).parent.parent
    local_bin = repo_root / "node_modules" / ".bin" / "agent-browser"
    if local_bin.exists():
        return str(local_bin)
    
    # Check common npx locations
    npx_path = shutil.which("npx")
    if npx_path:
        return "npx agent-browser"
    
    raise FileNotFoundError(
        "agent-browser CLI not found. Install it with: npm install -g agent-browser\n"
        "Or run 'npm install' in the repo root to install locally.\n"
        "Or ensure npx is available in your PATH."
    )


def _run_browser_command(
    task_id: str,
    command: str,
    args: List[str] = None,
    timeout: int = DEFAULT_COMMAND_TIMEOUT
) -> Dict[str, Any]:
    """
    Run an agent-browser CLI command using our pre-created Browserbase session.
    
    Args:
        task_id: Task identifier to get the right session
        command: The command to run (e.g., "open", "click")
        args: Additional arguments for the command
        timeout: Command timeout in seconds
        
    Returns:
        Parsed JSON response from agent-browser
    """
    args = args or []
    
    # Build the command
    try:
        browser_cmd = _find_agent_browser()
    except FileNotFoundError as e:
        return {"success": False, "error": str(e)}
    
    from tools.interrupt import is_interrupted
    if is_interrupted():
        return {"success": False, "error": "Interrupted"}

    # Get session info (creates Browserbase session with proxies if needed)
    try:
        session_info = _get_session_info(task_id)
    except Exception as e:
        return {"success": False, "error": f"Failed to create browser session: {str(e)}"}
    
    # Connect via CDP to our pre-created Browserbase session.
    # IMPORTANT: Do NOT use --session with --cdp. In agent-browser >=0.13,
    # --session creates a local browser instance and silently ignores --cdp.
    # Per-task isolation is handled by AGENT_BROWSER_SOCKET_DIR instead.
    cmd_parts = browser_cmd.split() + [
        "--cdp", session_info["cdp_url"],
        "--json",
        command
    ] + args
    
    try:
        # Give each task its own socket directory to prevent concurrency conflicts.
        # Without this, parallel workers fight over the same default socket path,
        # causing "Failed to create socket directory: Permission denied" errors.
        task_socket_dir = os.path.join(
            tempfile.gettempdir(), 
            f"agent-browser-{session_info['session_name']}"
        )
        os.makedirs(task_socket_dir, exist_ok=True)
        
        browser_env = {
            **os.environ,
            "AGENT_BROWSER_SOCKET_DIR": task_socket_dir,
        }
        
        result = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=browser_env,
        )
        
        # Log stderr for diagnostics (agent-browser may emit warnings there)
        if result.stderr and result.stderr.strip():
            logger.debug("stderr from '%s': %s", command, result.stderr.strip()[:200])
        
        # Parse JSON output
        if result.stdout.strip():
            try:
                parsed = json.loads(result.stdout.strip())
                # Warn if snapshot came back empty (common sign of daemon/CDP issues)
                if command == "snapshot" and parsed.get("success"):
                    snap_data = parsed.get("data", {})
                    if not snap_data.get("snapshot") and not snap_data.get("refs"):
                        logger.warning("snapshot returned empty content. "
                                       "Possible stale daemon or CDP connection issue. "
                                       "returncode=%s", result.returncode)
                return parsed
            except json.JSONDecodeError:
                # If not valid JSON, return as raw output
                return {
                    "success": True,
                    "data": {"raw": result.stdout.strip()}
                }
        
        # Check for errors
        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else f"Command failed with code {result.returncode}"
            return {"success": False, "error": error_msg}
        
        return {"success": True, "data": {}}
        
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Command timed out after {timeout} seconds"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _extract_relevant_content(
    snapshot_text: str,
    user_task: Optional[str] = None
) -> str:
    """Use LLM to extract relevant content from a snapshot based on the user's task.

    Falls back to simple truncation when no auxiliary vision model is configured.
    """
    if _aux_vision_client is None or EXTRACTION_MODEL is None:
        return _truncate_snapshot(snapshot_text)

    if user_task:
        extraction_prompt = (
            f"You are a content extractor for a browser automation agent.\n\n"
            f"The user's task is: {user_task}\n\n"
            f"Given the following page snapshot (accessibility tree representation), "
            f"extract and summarize the most relevant information for completing this task. Focus on:\n"
            f"1. Interactive elements (buttons, links, inputs) that might be needed\n"
            f"2. Text content relevant to the task (prices, descriptions, headings, important info)\n"
            f"3. Navigation structure if relevant\n\n"
            f"Keep ref IDs (like [ref=e5]) for interactive elements so the agent can use them.\n\n"
            f"Page Snapshot:\n{snapshot_text}\n\n"
            f"Provide a concise summary that preserves actionable information and relevant content."
        )
    else:
        extraction_prompt = (
            f"Summarize this page snapshot, preserving:\n"
            f"1. All interactive elements with their ref IDs (like [ref=e5])\n"
            f"2. Key text content and headings\n"
            f"3. Important information visible on the page\n\n"
            f"Page Snapshot:\n{snapshot_text}\n\n"
            f"Provide a concise summary focused on interactive elements and key content."
        )

    try:
        from agent.auxiliary_client import auxiliary_max_tokens_param
        response = _aux_vision_client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[{"role": "user", "content": extraction_prompt}],
            **auxiliary_max_tokens_param(4000),
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception:
        return _truncate_snapshot(snapshot_text)


def _truncate_snapshot(snapshot_text: str, max_chars: int = 8000) -> str:
    """
    Simple truncation fallback for snapshots.
    
    Args:
        snapshot_text: The snapshot text to truncate
        max_chars: Maximum characters to keep
        
    Returns:
        Truncated text with indicator if truncated
    """
    if len(snapshot_text) <= max_chars:
        return snapshot_text
    
    return snapshot_text[:max_chars] + "\n\n[... content truncated ...]"


# ============================================================================
# Browser Tool Functions
# ============================================================================

def browser_navigate(url: str, task_id: Optional[str] = None) -> str:
    """
    Navigate to a URL in the browser.
    
    Args:
        url: The URL to navigate to
        task_id: Task identifier for session isolation
        
    Returns:
        JSON string with navigation result (includes stealth features info on first nav)
    """
    effective_task_id = task_id or "default"
    
    # Get session info to check if this is a new session
    # (will create one with features logged if not exists)
    session_info = _get_session_info(effective_task_id)
    is_first_nav = session_info.get("_first_nav", True)
    
    # Mark that we've done at least one navigation
    if is_first_nav:
        session_info["_first_nav"] = False
    
    result = _run_browser_command(effective_task_id, "open", [url], timeout=60)
    
    if result.get("success"):
        data = result.get("data", {})
        title = data.get("title", "")
        final_url = data.get("url", url)
        
        response = {
            "success": True,
            "url": final_url,
            "title": title
        }
        
        # Detect common "blocked" page patterns from title/url
        blocked_patterns = [
            "access denied", "access to this page has been denied",
            "blocked", "bot detected", "verification required",
            "please verify", "are you a robot", "captcha",
            "cloudflare", "ddos protection", "checking your browser",
            "just a moment", "attention required"
        ]
        title_lower = title.lower()
        
        if any(pattern in title_lower for pattern in blocked_patterns):
            response["bot_detection_warning"] = (
                f"Page title '{title}' suggests bot detection. The site may have blocked this request. "
                "Options: 1) Try adding delays between actions, 2) Access different pages first, "
                "3) Enable advanced stealth (BROWSERBASE_ADVANCED_STEALTH=true, requires Scale plan), "
                "4) Some sites have very aggressive bot detection that may be unavoidable."
            )
        
        # Include feature info on first navigation so model knows what's active
        if is_first_nav and "features" in session_info:
            features = session_info["features"]
            active_features = [k for k, v in features.items() if v]
            if not features.get("proxies"):
                response["stealth_warning"] = (
                    "Running WITHOUT residential proxies. Bot detection may be more aggressive. "
                    "Consider upgrading Browserbase plan for proxy support."
                )
            response["stealth_features"] = active_features
        
        return json.dumps(response, ensure_ascii=False)
    else:
        return json.dumps({
            "success": False,
            "error": result.get("error", "Navigation failed")
        }, ensure_ascii=False)


def browser_snapshot(
    full: bool = False,
    task_id: Optional[str] = None,
    user_task: Optional[str] = None
) -> str:
    """
    Get a text-based snapshot of the current page's accessibility tree.
    
    Args:
        full: If True, return complete snapshot. If False, return compact view.
        task_id: Task identifier for session isolation
        user_task: The user's current task (for task-aware extraction)
        
    Returns:
        JSON string with page snapshot
    """
    effective_task_id = task_id or "default"
    
    # Build command args based on full flag
    args = []
    if not full:
        args.extend(["-c"])  # Compact mode
    
    result = _run_browser_command(effective_task_id, "snapshot", args)
    
    if result.get("success"):
        data = result.get("data", {})
        snapshot_text = data.get("snapshot", "")
        refs = data.get("refs", {})
        
        # Check if snapshot needs summarization
        if len(snapshot_text) > SNAPSHOT_SUMMARIZE_THRESHOLD and user_task:
            snapshot_text = _extract_relevant_content(snapshot_text, user_task)
        elif len(snapshot_text) > SNAPSHOT_SUMMARIZE_THRESHOLD:
            snapshot_text = _truncate_snapshot(snapshot_text)
        
        response = {
            "success": True,
            "snapshot": snapshot_text,
            "element_count": len(refs) if refs else 0
        }
        
        return json.dumps(response, ensure_ascii=False)
    else:
        return json.dumps({
            "success": False,
            "error": result.get("error", "Failed to get snapshot")
        }, ensure_ascii=False)


def browser_click(ref: str, task_id: Optional[str] = None) -> str:
    """
    Click on an element.
    
    Args:
        ref: Element reference (e.g., "@e5")
        task_id: Task identifier for session isolation
        
    Returns:
        JSON string with click result
    """
    effective_task_id = task_id or "default"
    
    # Ensure ref starts with @
    if not ref.startswith("@"):
        ref = f"@{ref}"
    
    result = _run_browser_command(effective_task_id, "click", [ref])
    
    if result.get("success"):
        return json.dumps({
            "success": True,
            "clicked": ref
        }, ensure_ascii=False)
    else:
        return json.dumps({
            "success": False,
            "error": result.get("error", f"Failed to click {ref}")
        }, ensure_ascii=False)


def browser_type(ref: str, text: str, task_id: Optional[str] = None) -> str:
    """
    Type text into an input field.
    
    Args:
        ref: Element reference (e.g., "@e3")
        text: Text to type
        task_id: Task identifier for session isolation
        
    Returns:
        JSON string with type result
    """
    effective_task_id = task_id or "default"
    
    # Ensure ref starts with @
    if not ref.startswith("@"):
        ref = f"@{ref}"
    
    # Use fill command (clears then types)
    result = _run_browser_command(effective_task_id, "fill", [ref, text])
    
    if result.get("success"):
        return json.dumps({
            "success": True,
            "typed": text,
            "element": ref
        }, ensure_ascii=False)
    else:
        return json.dumps({
            "success": False,
            "error": result.get("error", f"Failed to type into {ref}")
        }, ensure_ascii=False)


def browser_scroll(direction: str, task_id: Optional[str] = None) -> str:
    """
    Scroll the page.
    
    Args:
        direction: "up" or "down"
        task_id: Task identifier for session isolation
        
    Returns:
        JSON string with scroll result
    """
    effective_task_id = task_id or "default"
    
    # Validate direction
    if direction not in ["up", "down"]:
        return json.dumps({
            "success": False,
            "error": f"Invalid direction '{direction}'. Use 'up' or 'down'."
        }, ensure_ascii=False)
    
    result = _run_browser_command(effective_task_id, "scroll", [direction])
    
    if result.get("success"):
        return json.dumps({
            "success": True,
            "scrolled": direction
        }, ensure_ascii=False)
    else:
        return json.dumps({
            "success": False,
            "error": result.get("error", f"Failed to scroll {direction}")
        }, ensure_ascii=False)


def browser_back(task_id: Optional[str] = None) -> str:
    """
    Navigate back in browser history.
    
    Args:
        task_id: Task identifier for session isolation
        
    Returns:
        JSON string with navigation result
    """
    effective_task_id = task_id or "default"
    result = _run_browser_command(effective_task_id, "back", [])
    
    if result.get("success"):
        data = result.get("data", {})
        return json.dumps({
            "success": True,
            "url": data.get("url", "")
        }, ensure_ascii=False)
    else:
        return json.dumps({
            "success": False,
            "error": result.get("error", "Failed to go back")
        }, ensure_ascii=False)


def browser_press(key: str, task_id: Optional[str] = None) -> str:
    """
    Press a keyboard key.
    
    Args:
        key: Key to press (e.g., "Enter", "Tab")
        task_id: Task identifier for session isolation
        
    Returns:
        JSON string with key press result
    """
    effective_task_id = task_id or "default"
    result = _run_browser_command(effective_task_id, "press", [key])
    
    if result.get("success"):
        return json.dumps({
            "success": True,
            "pressed": key
        }, ensure_ascii=False)
    else:
        return json.dumps({
            "success": False,
            "error": result.get("error", f"Failed to press {key}")
        }, ensure_ascii=False)


def browser_close(task_id: Optional[str] = None) -> str:
    """
    Close the browser session.
    
    Args:
        task_id: Task identifier for session isolation
        
    Returns:
        JSON string with close result
    """
    effective_task_id = task_id or "default"
    result = _run_browser_command(effective_task_id, "close", [])
    
    # Close the BrowserBase session via API
    session_key = task_id if task_id and task_id in _active_sessions else "default"
    if session_key in _active_sessions:
        session_info = _active_sessions[session_key]
        bb_session_id = session_info.get("bb_session_id")
        if bb_session_id:
            try:
                config = _get_browserbase_config()
                _close_browserbase_session(bb_session_id, config["api_key"], config["project_id"])
            except Exception as e:
                logger.warning("Could not close BrowserBase session: %s", e)
        del _active_sessions[session_key]
    
    if result.get("success"):
        return json.dumps({
            "success": True,
            "closed": True
        }, ensure_ascii=False)
    else:
        # Even if close fails, session was released
        return json.dumps({
            "success": True,
            "closed": True,
            "warning": result.get("error", "Session may not have been active")
        }, ensure_ascii=False)


def browser_get_images(task_id: Optional[str] = None) -> str:
    """
    Get all images on the current page.
    
    Args:
        task_id: Task identifier for session isolation
        
    Returns:
        JSON string with list of images (src and alt)
    """
    effective_task_id = task_id or "default"
    
    # Use eval to run JavaScript that extracts images
    js_code = """JSON.stringify(
        [...document.images].map(img => ({
            src: img.src,
            alt: img.alt || '',
            width: img.naturalWidth,
            height: img.naturalHeight
        })).filter(img => img.src && !img.src.startsWith('data:'))
    )"""
    
    result = _run_browser_command(effective_task_id, "eval", [js_code])
    
    if result.get("success"):
        data = result.get("data", {})
        raw_result = data.get("result", "[]")
        
        try:
            # Parse the JSON string returned by JavaScript
            if isinstance(raw_result, str):
                images = json.loads(raw_result)
            else:
                images = raw_result
            
            return json.dumps({
                "success": True,
                "images": images,
                "count": len(images)
            }, ensure_ascii=False)
        except json.JSONDecodeError:
            return json.dumps({
                "success": True,
                "images": [],
                "count": 0,
                "warning": "Could not parse image data"
            }, ensure_ascii=False)
    else:
        return json.dumps({
            "success": False,
            "error": result.get("error", "Failed to get images")
        }, ensure_ascii=False)


def browser_vision(question: str, task_id: Optional[str] = None) -> str:
    """
    Take a screenshot of the current page and analyze it with vision AI.
    
    This tool captures what's visually displayed in the browser and sends it
    to Gemini for analysis. Useful for understanding visual content that the
    text-based snapshot may not capture (CAPTCHAs, verification challenges,
    images, complex layouts, etc.).
    
    Args:
        question: What you want to know about the page visually
        task_id: Task identifier for session isolation
        
    Returns:
        JSON string with vision analysis results
    """
    import base64
    import tempfile
    import uuid as uuid_mod
    from pathlib import Path
    
    effective_task_id = task_id or "default"
    
    # Check auxiliary vision client
    if _aux_vision_client is None or EXTRACTION_MODEL is None:
        return json.dumps({
            "success": False,
            "error": "Browser vision unavailable: no auxiliary vision model configured. "
                     "Set OPENROUTER_API_KEY or configure Nous Portal to enable browser vision."
        }, ensure_ascii=False)
    
    # Create a temporary file for the screenshot
    temp_dir = Path(tempfile.gettempdir())
    screenshot_path = temp_dir / f"browser_screenshot_{uuid_mod.uuid4().hex}.png"
    
    try:
        # Take screenshot using agent-browser
        result = _run_browser_command(
            effective_task_id, 
            "screenshot", 
            [str(screenshot_path)],
            timeout=30
        )
        
        if not result.get("success"):
            return json.dumps({
                "success": False,
                "error": f"Failed to take screenshot: {result.get('error', 'Unknown error')}"
            }, ensure_ascii=False)
        
        # Check if screenshot file was created
        if not screenshot_path.exists():
            return json.dumps({
                "success": False,
                "error": "Screenshot file was not created"
            }, ensure_ascii=False)
        
        # Read and convert to base64
        image_data = screenshot_path.read_bytes()
        image_base64 = base64.b64encode(image_data).decode("ascii")
        data_url = f"data:image/png;base64,{image_base64}"
        
        vision_prompt = (
            f"You are analyzing a screenshot of a web browser.\n\n"
            f"User's question: {question}\n\n"
            f"Provide a detailed and helpful answer based on what you see in the screenshot. "
            f"If there are interactive elements, describe them. If there are verification challenges "
            f"or CAPTCHAs, describe what type they are and what action might be needed. "
            f"Focus on answering the user's specific question."
        )

        # Use the sync auxiliary vision client directly
        from agent.auxiliary_client import auxiliary_max_tokens_param
        response = _aux_vision_client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            **auxiliary_max_tokens_param(2000),
            temperature=0.1,
        )
        
        analysis = response.choices[0].message.content
        return json.dumps({
            "success": True,
            "analysis": analysis,
        }, ensure_ascii=False)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Error during vision analysis: {str(e)}"
        }, ensure_ascii=False)
    
    finally:
        # Clean up screenshot file
        if screenshot_path.exists():
            try:
                screenshot_path.unlink()
            except Exception:
                pass


# ============================================================================
# Cleanup and Management Functions
# ============================================================================

def _close_browserbase_session(session_id: str, api_key: str, project_id: str) -> bool:
    """
    Close a Browserbase session immediately via the API.
    
    Uses POST /v1/sessions/{id} with status=REQUEST_RELEASE to immediately
    terminate the session without waiting for keepAlive timeout.
    
    Args:
        session_id: The Browserbase session ID
        api_key: Browserbase API key
        project_id: Browserbase project ID
        
    Returns:
        True if session was successfully closed, False otherwise
    """
    try:
        # POST to update session status to REQUEST_RELEASE
        response = requests.post(
            f"https://api.browserbase.com/v1/sessions/{session_id}",
            headers={
                "X-BB-API-Key": api_key,
                "Content-Type": "application/json"
            },
            json={
                "projectId": project_id,
                "status": "REQUEST_RELEASE"
            },
            timeout=10
        )
        
        if response.status_code in (200, 201, 204):
            logger.debug("Successfully closed BrowserBase session %s", session_id)
            return True
        else:
            logger.warning("Failed to close session %s: HTTP %s - %s", session_id, response.status_code, response.text[:200])
            return False
                
    except Exception as e:
        logger.error("Exception closing session %s: %s", session_id, e)
        return False


def cleanup_browser(task_id: Optional[str] = None) -> None:
    """
    Clean up browser session for a task.
    
    Called automatically when a task completes or when inactivity timeout is reached.
    Closes both the agent-browser session and the Browserbase session.
    
    Args:
        task_id: Task identifier to clean up
    """
    if task_id is None:
        task_id = "default"
    
    logger.debug("cleanup_browser called for task_id: %s", task_id)
    logger.debug("Active sessions: %s", list(_active_sessions.keys()))
    
    # Check if session exists (under lock), but don't remove yet -
    # _run_browser_command needs it to build the close command.
    with _cleanup_lock:
        session_info = _active_sessions.get(task_id)
    
    if session_info:
        bb_session_id = session_info.get("bb_session_id", "unknown")
        logger.debug("Found session for task %s: bb_session_id=%s", task_id, bb_session_id)
        
        # Try to close via agent-browser first (needs session in _active_sessions)
        try:
            _run_browser_command(task_id, "close", [], timeout=10)
            logger.debug("agent-browser close command completed for task %s", task_id)
        except Exception as e:
            logger.warning("agent-browser close failed for task %s: %s", task_id, e)
        
        # Now remove from tracking under lock
        with _cleanup_lock:
            _active_sessions.pop(task_id, None)
            _session_last_activity.pop(task_id, None)
        
        # Close the Browserbase session immediately via API
        try:
            config = _get_browserbase_config()
            success = _close_browserbase_session(bb_session_id, config["api_key"], config["project_id"])
            if not success:
                logger.warning("Could not close BrowserBase session %s", bb_session_id)
        except Exception as e:
            logger.error("Exception during BrowserBase session close: %s", e)
        
        # Kill the daemon process and clean up socket directory
        session_name = session_info.get("session_name", "")
        if session_name:
            socket_dir = os.path.join(tempfile.gettempdir(), f"agent-browser-{session_name}")
            if os.path.exists(socket_dir):
                # agent-browser writes {session}.pid in the socket dir
                pid_file = os.path.join(socket_dir, f"{session_name}.pid")
                if os.path.isfile(pid_file):
                    try:
                        daemon_pid = int(open(pid_file).read().strip())
                        os.kill(daemon_pid, signal.SIGTERM)
                        logger.debug("Killed daemon pid %s for %s", daemon_pid, session_name)
                    except (ProcessLookupError, ValueError, PermissionError, OSError):
                        pass
                shutil.rmtree(socket_dir, ignore_errors=True)
        
        logger.debug("Removed task %s from active sessions", task_id)
    else:
        logger.debug("No active session found for task_id: %s", task_id)


def cleanup_all_browsers() -> None:
    """
    Clean up all active browser sessions.
    
    Useful for cleanup on shutdown.
    """
    with _cleanup_lock:
        task_ids = list(_active_sessions.keys())
    for task_id in task_ids:
        cleanup_browser(task_id)


def get_active_browser_sessions() -> Dict[str, Dict[str, str]]:
    """
    Get information about active browser sessions.
    
    Returns:
        Dict mapping task_id to session info (session_name, bb_session_id, cdp_url)
    """
    with _cleanup_lock:
        return _active_sessions.copy()


# ============================================================================
# Requirements Check
# ============================================================================

def check_browser_requirements() -> bool:
    """
    Check if browser tool requirements are met.
    
    Returns:
        True if all requirements are met, False otherwise
    """
    # Check for Browserbase credentials
    api_key = os.environ.get("BROWSERBASE_API_KEY")
    project_id = os.environ.get("BROWSERBASE_PROJECT_ID")
    
    if not api_key or not project_id:
        return False
    
    # Check for agent-browser CLI
    try:
        _find_agent_browser()
        return True
    except FileNotFoundError:
        return False


# ============================================================================
# Module Test
# ============================================================================

if __name__ == "__main__":
    """
    Simple test/demo when run directly
    """
    print("🌐 Browser Tool Module")
    print("=" * 40)
    
    # Check requirements
    if check_browser_requirements():
        print("✅ All requirements met")
    else:
        print("❌ Missing requirements:")
        if not os.environ.get("BROWSERBASE_API_KEY"):
            print("   - BROWSERBASE_API_KEY not set")
        if not os.environ.get("BROWSERBASE_PROJECT_ID"):
            print("   - BROWSERBASE_PROJECT_ID not set")
        try:
            _find_agent_browser()
        except FileNotFoundError:
            print("   - agent-browser CLI not found")
    
    print("\n📋 Available Browser Tools:")
    for schema in BROWSER_TOOL_SCHEMAS:
        print(f"  🔹 {schema['name']}: {schema['description'][:60]}...")
    
    print("\n💡 Usage:")
    print("  from tools.browser_tool import browser_navigate, browser_snapshot")
    print("  result = browser_navigate('https://example.com', task_id='my_task')")
    print("  snapshot = browser_snapshot(task_id='my_task')")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry

_BROWSER_SCHEMA_MAP = {s["name"]: s for s in BROWSER_TOOL_SCHEMAS}

registry.register(
    name="browser_navigate",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_navigate"],
    handler=lambda args, **kw: browser_navigate(url=args.get("url", ""), task_id=kw.get("task_id")),
    check_fn=check_browser_requirements,
    requires_env=["BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID"],
)
registry.register(
    name="browser_snapshot",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_snapshot"],
    handler=lambda args, **kw: browser_snapshot(
        full=args.get("full", False), task_id=kw.get("task_id"), user_task=kw.get("user_task")),
    check_fn=check_browser_requirements,
    requires_env=["BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID"],
)
registry.register(
    name="browser_click",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_click"],
    handler=lambda args, **kw: browser_click(**args, task_id=kw.get("task_id")),
    check_fn=check_browser_requirements,
    requires_env=["BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID"],
)
registry.register(
    name="browser_type",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_type"],
    handler=lambda args, **kw: browser_type(**args, task_id=kw.get("task_id")),
    check_fn=check_browser_requirements,
    requires_env=["BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID"],
)
registry.register(
    name="browser_scroll",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_scroll"],
    handler=lambda args, **kw: browser_scroll(**args, task_id=kw.get("task_id")),
    check_fn=check_browser_requirements,
    requires_env=["BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID"],
)
registry.register(
    name="browser_back",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_back"],
    handler=lambda args, **kw: browser_back(task_id=kw.get("task_id")),
    check_fn=check_browser_requirements,
    requires_env=["BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID"],
)
registry.register(
    name="browser_press",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_press"],
    handler=lambda args, **kw: browser_press(key=args.get("key", ""), task_id=kw.get("task_id")),
    check_fn=check_browser_requirements,
    requires_env=["BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID"],
)
registry.register(
    name="browser_close",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_close"],
    handler=lambda args, **kw: browser_close(task_id=kw.get("task_id")),
    check_fn=check_browser_requirements,
    requires_env=["BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID"],
)
registry.register(
    name="browser_get_images",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_get_images"],
    handler=lambda args, **kw: browser_get_images(task_id=kw.get("task_id")),
    check_fn=check_browser_requirements,
    requires_env=["BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID"],
)
registry.register(
    name="browser_vision",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_vision"],
    handler=lambda args, **kw: browser_vision(question=args.get("question", ""), task_id=kw.get("task_id")),
    check_fn=check_browser_requirements,
    requires_env=["BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID"],
)

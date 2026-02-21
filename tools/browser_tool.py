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
import os
import signal
import subprocess
import shutil
import sys
import asyncio
import tempfile
import threading
import time
import requests
from typing import Dict, Any, Optional, List
from pathlib import Path
from hermes_constants import OPENROUTER_CHAT_URL

# Try to import httpx for async LLM calls
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


# ============================================================================
# Configuration
# ============================================================================

# Default timeout for browser commands (seconds)
DEFAULT_COMMAND_TIMEOUT = 30

# Default session timeout (seconds)
DEFAULT_SESSION_TIMEOUT = 300

# Max tokens for snapshot content before summarization
SNAPSHOT_SUMMARIZE_THRESHOLD = 8000

# Model for task-aware extraction
EXTRACTION_MODEL = "google/gemini-3-flash-preview"

# Track active sessions per task
# Now stores tuple of (session_name, browserbase_session_id, cdp_url)
_active_sessions: Dict[str, Dict[str, str]] = {}  # task_id -> {session_name, bb_session_id, cdp_url}

# Flag to track if cleanup has been done
_cleanup_done = False

# =============================================================================
# Inactivity Timeout Configuration
# =============================================================================

# Session inactivity timeout (seconds) - cleanup if no activity for this long
# Default: 2 minutes. Can be configured via environment variable.
BROWSER_SESSION_INACTIVITY_TIMEOUT = int(os.environ.get("BROWSER_INACTIVITY_TIMEOUT", "120"))

# Track last activity time per session
_session_last_activity: Dict[str, float] = {}

# Background cleanup thread state
_cleanup_thread = None
_cleanup_running = False
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
    
    print(f"\n[browser_tool] Emergency cleanup: closing {len(_active_sessions)} active session(s)...", file=sys.stderr)
    
    try:
        api_key = os.environ.get("BROWSERBASE_API_KEY")
        project_id = os.environ.get("BROWSERBASE_PROJECT_ID")
        
        if not api_key or not project_id:
            print("[browser_tool] WARNING: Cannot cleanup - missing BROWSERBASE credentials", file=sys.stderr)
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
                        print(f"[browser_tool] Closed session {bb_session_id}", file=sys.stderr)
                    else:
                        print(f"[browser_tool] Failed to close session {bb_session_id}: HTTP {response.status_code}", file=sys.stderr)
                except Exception as e:
                    print(f"[browser_tool] Error closing session {bb_session_id}: {e}", file=sys.stderr)
        
        _active_sessions.clear()
    except Exception as e:
        print(f"[browser_tool] Emergency cleanup error: {e}", file=sys.stderr)


def _signal_handler(signum, frame):
    """Handle interrupt signals to cleanup sessions before exit."""
    print(f"\n[browser_tool] Received signal {signum}, cleaning up...", file=sys.stderr)
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
            if not os.getenv("HERMES_QUIET"):
                elapsed = int(current_time - _session_last_activity.get(task_id, current_time))
                print(f"[browser_tool] Cleaning up inactive session for task: {task_id} "
                      f"(inactive for {elapsed}s)", file=sys.stderr)
            cleanup_browser(task_id)
            with _cleanup_lock:
                if task_id in _session_last_activity:
                    del _session_last_activity[task_id]
        except Exception as e:
            if not os.getenv("HERMES_QUIET"):
                print(f"[browser_tool] Error cleaning up inactive session {task_id}: {e}", file=sys.stderr)


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
            if not os.getenv("HERMES_QUIET"):
                print(f"[browser_tool] Cleanup thread error: {e}", file=sys.stderr)
        
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
            if not os.getenv("HERMES_QUIET"):
                print(f"[browser_tool] Started inactivity cleanup thread "
                      f"(timeout: {BROWSER_SESSION_INACTIVITY_TIMEOUT}s)", file=sys.stderr)


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
        "description": "Navigate to a URL in the browser. Opens the page and waits for it to load. Returns the final URL and page title. IMPORTANT: This should be the FIRST browser tool called - it initializes the browser session and loads the target page. Other browser tools require a page to be loaded first. NOTE: For simple information retrieval, prefer using web_search or web_extract first as they are faster and more cost-effective. Use browser tools when you need to interact with a page (click buttons, fill forms, handle dynamic content).",
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
        "description": "Get a text-based snapshot of the current page's accessibility tree. Returns interactive elements with ref IDs (like @e1, @e2) that can be used with browser_click and browser_type. Use full=true to get the complete page content including all text; use full=false (default) for a compact view focused on interactive elements. Requires browser_navigate to be called first.",
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
            print(f"[browser_tool] WARNING: Invalid BROWSERBASE_SESSION_TIMEOUT value: {custom_timeout_ms}", 
                  file=sys.stderr)
    
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
            print(f"[browser_tool] WARNING: keepAlive may require paid plan (402), retrying without it. "
                  f"Sessions may timeout during long operations.", file=sys.stderr)
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
            print(f"[browser_tool] WARNING: Proxies unavailable (402), retrying without proxies. "
                  f"Bot detection may be less effective.", file=sys.stderr)
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
    print(f"[browser_tool] Created session {session_name} with features: {feature_str}", file=sys.stderr)
    
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
    
    # Check if we already have a session for this task
    if task_id in _active_sessions:
        return _active_sessions[task_id]
    
    # Create a new Browserbase session with proxies
    session_info = _create_browserbase_session(task_id)
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


_stale_daemons_cleaned = False

def _kill_stale_agent_browser_daemons():
    """Kill any orphaned agent-browser daemon processes from previous runs."""
    global _stale_daemons_cleaned
    if _stale_daemons_cleaned:
        return
    _stale_daemons_cleaned = True

    try:
        result = subprocess.run(
            ["pgrep", "-f", "agent-browser.*daemon"],
            capture_output=True, text=True, timeout=5
        )
        pids = result.stdout.strip().split()
        if pids and pids[0]:
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except (ProcessLookupError, ValueError, PermissionError):
                    pass
            if not os.getenv("HERMES_QUIET"):
                print(f"[browser_tool] Cleaned up {len(pids)} stale daemon process(es)", file=sys.stderr)
    except Exception:
        pass


def _find_agent_browser() -> str:
    """
    Find the agent-browser CLI executable.
    
    Checks in order: PATH, local node_modules/.bin/, npx fallback.
    Also kills any stale daemon processes from prior runs on first call.
    
    Returns:
        Path to agent-browser executable
        
    Raises:
        FileNotFoundError: If agent-browser is not installed
    """
    _kill_stale_agent_browser_daemons()

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
    
    # Get session info (creates Browserbase session with proxies if needed)
    try:
        session_info = _get_session_info(task_id)
    except Exception as e:
        return {"success": False, "error": f"Failed to create browser session: {str(e)}"}
    
    # Connect via CDP to our pre-created Browserbase session (with proxies)
    # Use --cdp flag to connect to existing session instead of creating new one
    cmd_parts = browser_cmd.split() + [
        "--session", session_info["session_name"],
        "--cdp", session_info["cdp_url"],  # Connect to our proxied session
        "--json",  # Always request JSON output
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
        
        # Parse JSON output
        if result.stdout.strip():
            try:
                return json.loads(result.stdout.strip())
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


async def _extract_relevant_content(
    snapshot_text: str,
    user_task: Optional[str] = None
) -> str:
    """
    Use LLM to extract relevant content from a snapshot based on the user's task.
    
    This provides task-aware summarization that preserves meaningful text content
    (paragraphs, prices, descriptions) relevant to what the user is trying to accomplish.
    
    Args:
        snapshot_text: The full snapshot text
        user_task: The user's current task/goal (optional)
        
    Returns:
        Summarized/extracted content
    """
    if not HTTPX_AVAILABLE:
        # Fall back to simple truncation
        return _truncate_snapshot(snapshot_text)
    
    # Get API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return _truncate_snapshot(snapshot_text)
    
    # Build extraction prompt
    if user_task:
        extraction_prompt = f"""You are a content extractor for a browser automation agent.

The user's task is: {user_task}

Given the following page snapshot (accessibility tree representation), extract and summarize the most relevant information for completing this task. Focus on:
1. Interactive elements (buttons, links, inputs) that might be needed
2. Text content relevant to the task (prices, descriptions, headings, important info)
3. Navigation structure if relevant

Keep ref IDs (like [ref=e5]) for interactive elements so the agent can use them.

Page Snapshot:
{snapshot_text}

Provide a concise summary that preserves actionable information and relevant content."""
    else:
        extraction_prompt = f"""Summarize this page snapshot, preserving:
1. All interactive elements with their ref IDs (like [ref=e5])
2. Key text content and headings
3. Important information visible on the page

Page Snapshot:
{snapshot_text}

Provide a concise summary focused on interactive elements and key content."""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                OPENROUTER_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": EXTRACTION_MODEL,
                    "messages": [
                        {"role": "user", "content": extraction_prompt}
                    ],
                    "max_tokens": 4000,
                    "temperature": 0.1
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                # Fall back to truncation on API error
                return _truncate_snapshot(snapshot_text)
                
    except Exception:
        # Fall back to truncation on any error
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
            # Run async extraction
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            snapshot_text = loop.run_until_complete(
                _extract_relevant_content(snapshot_text, user_task)
            )
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
                print(f"[browser_tool] Warning: Could not close BrowserBase session: {e}", file=sys.stderr)
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
    
    # Check for OpenRouter API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return json.dumps({
            "success": False,
            "error": "OPENROUTER_API_KEY not set. Vision analysis requires this API key."
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
        
        # Prepare the vision prompt
        vision_prompt = f"""You are analyzing a screenshot of a web browser.

User's question: {question}

Provide a detailed and helpful answer based on what you see in the screenshot. 
If there are interactive elements, describe them. If there are verification challenges 
or CAPTCHAs, describe what type they are and what action might be needed.
Focus on answering the user's specific question."""

        # Call OpenRouter/Gemini for vision analysis
        if HTTPX_AVAILABLE:
            import asyncio
            
            async def analyze_screenshot():
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        OPENROUTER_CHAT_URL,
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "google/gemini-3-flash-preview",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": vision_prompt},
                                        {
                                            "type": "image_url",
                                            "image_url": {"url": data_url}
                                        }
                                    ]
                                }
                            ],
                            "max_tokens": 2000,
                            "temperature": 0.1
                        }
                    )
                    
                    if response.status_code != 200:
                        return {
                            "success": False,
                            "error": f"Vision API error: {response.status_code} - {response.text[:200]}"
                        }
                    
                    result_data = response.json()
                    analysis = result_data["choices"][0]["message"]["content"]
                    return {
                        "success": True,
                        "analysis": analysis
                    }
            
            # Run the async function
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            vision_result = loop.run_until_complete(analyze_screenshot())
            return json.dumps(vision_result, ensure_ascii=False)
        
        else:
            # Fallback: use synchronous requests
            response = requests.post(
                OPENROUTER_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "google/gemini-3-flash-preview",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": vision_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": data_url}
                                }
                            ]
                        }
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.1
                },
                timeout=60
            )
            
            if response.status_code != 200:
                return json.dumps({
                    "success": False,
                    "error": f"Vision API error: {response.status_code} - {response.text[:200]}"
                }, ensure_ascii=False)
            
            result_data = response.json()
            analysis = result_data["choices"][0]["message"]["content"]
            return json.dumps({
                "success": True,
                "analysis": analysis
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
            print(f"[browser_tool] Successfully closed BrowserBase session {session_id}", file=sys.stderr)
            return True
        else:
            print(f"[browser_tool] Failed to close session {session_id}: HTTP {response.status_code} - {response.text[:200]}", file=sys.stderr)
            return False
                
    except Exception as e:
        print(f"[browser_tool] Exception closing session {session_id}: {e}", file=sys.stderr)
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
    
    if not os.getenv("HERMES_QUIET"):
        print(f"[browser_tool] cleanup_browser called for task_id: {task_id}", file=sys.stderr)
        print(f"[browser_tool] Active sessions: {list(_active_sessions.keys())}", file=sys.stderr)
    
    if task_id in _active_sessions:
        session_info = _active_sessions[task_id]
        bb_session_id = session_info.get("bb_session_id", "unknown")
        print(f"[browser_tool] Found session for task {task_id}: bb_session_id={bb_session_id}", file=sys.stderr)
        
        # Try to close via agent-browser first
        try:
            _run_browser_command(task_id, "close", [], timeout=10)
            print(f"[browser_tool] agent-browser close command completed for task {task_id}", file=sys.stderr)
        except Exception as e:
            print(f"[browser_tool] agent-browser close failed for task {task_id}: {e}", file=sys.stderr)
        
        # Close the Browserbase session immediately via API
        try:
            config = _get_browserbase_config()
            success = _close_browserbase_session(bb_session_id, config["api_key"], config["project_id"])
            if not success:
                print(f"[browser_tool] WARNING: Could not close BrowserBase session {bb_session_id}", file=sys.stderr)
        except Exception as e:
            print(f"[browser_tool] Exception during BrowserBase session close: {e}", file=sys.stderr)
        
        # Clean up per-task socket directory
        session_name = session_info.get("session_name", "")
        if session_name:
            socket_dir = os.path.join(tempfile.gettempdir(), f"agent-browser-{session_name}")
            if os.path.exists(socket_dir):
                shutil.rmtree(socket_dir, ignore_errors=True)
        
        del _active_sessions[task_id]
        if not os.getenv("HERMES_QUIET"):
            print(f"[browser_tool] Removed task {task_id} from active sessions", file=sys.stderr)
    elif not os.getenv("HERMES_QUIET"):
        print(f"[browser_tool] No active session found for task_id: {task_id}", file=sys.stderr)
    
    # Clean up activity tracking
    with _cleanup_lock:
        if task_id in _session_last_activity:
            del _session_last_activity[task_id]


def cleanup_all_browsers() -> None:
    """
    Clean up all active browser sessions.
    
    Useful for cleanup on shutdown.
    """
    for task_id in list(_active_sessions.keys()):
        cleanup_browser(task_id)
    
    # Clear any remaining activity tracking
    with _cleanup_lock:
        _session_last_activity.clear()


def get_active_browser_sessions() -> Dict[str, Dict[str, str]]:
    """
    Get information about active browser sessions.
    
    Returns:
        Dict mapping task_id to session info (session_name, bb_session_id, cdp_url)
    """
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

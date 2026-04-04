#!/usr/bin/env python3
"""
A2A (Agent-to-Agent) Tool — discover and call remote A2A agents.

Gives the Hermes agent three capabilities:
  - a2a_discover    : fetch an Agent Card from any A2A endpoint
  - a2a_call        : send a task to any A2A agent and return the response
  - a2a_local_scan  : scan localhost ports to find running A2A agents

Remote agents are called via raw HTTP + JSON-RPC 2.0 (no SDK dependency needed
at runtime; the a2a-sdk is only required when *hosting* Hermes as an A2A server).

Named agents can be pre-configured in ``~/.hermes/config.yaml``::

    a2a_agents:
      researcher:
        url: http://192.168.1.100:9000
        description: "Remote research agent"
      coder:
        url: http://192.168.1.101:9000
        description: "Remote code-writing agent"

When named agents are configured, Hermes will list them in the tool description
so the model knows what remote help is available.
"""

import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT = 120.0  # seconds per A2A call

# ---------------------------------------------------------------------------
# Agent Card cache
# ---------------------------------------------------------------------------

_CARD_CACHE: Dict[str, Dict] = {}   # url → {"card": {...}, "expires": float}
_CARD_TTL = 300.0                   # 5 minutes


def _get_cached_card(url: str) -> Optional[Dict]:
    entry = _CARD_CACHE.get(url)
    if entry and time.monotonic() < entry["expires"]:
        return entry["card"]
    return None


def _set_cached_card(url: str, card: Dict) -> None:
    _CARD_CACHE[url] = {"card": card, "expires": time.monotonic() + _CARD_TTL}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_a2a_agents() -> Dict[str, Dict[str, str]]:
    """Load ``a2a_agents`` section from ~/.hermes/config.yaml."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        agents = cfg.get("a2a_agents") or {}
        if isinstance(agents, dict):
            return agents
    except Exception:
        logger.debug("Could not load a2a_agents from config", exc_info=True)
    return {}


def check_a2a_requirements() -> bool:
    """httpx is always available — A2A tool is always enabled."""
    return True


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _fetch_agent_card(url: str) -> Dict:
    """Fetch and cache the Agent Card for *url*. Raises on network/HTTP error."""
    cached = _get_cached_card(url)
    if cached is not None:
        logger.debug("a2a_discover: cache hit for %s", url)
        return cached

    card_url = f"{url}/.well-known/agent.json"
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(card_url)
        resp.raise_for_status()
        card = resp.json()

    _set_cached_card(url, card)
    return card


def a2a_discover(url: str) -> str:
    """
    Fetch the Agent Card from an A2A endpoint.

    Returns a JSON string describing the agent: name, description, skills,
    model, and streaming support. Results are cached for 5 minutes.
    """
    url = url.rstrip("/")

    try:
        card = _fetch_agent_card(url)
    except httpx.HTTPStatusError as e:
        return json.dumps({"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"})
    except Exception as e:
        return json.dumps({"error": f"Could not reach {url}/.well-known/agent.json: {e}"})

    name = card.get("name", "unknown")
    description = card.get("description", "")
    skills = [s.get("name", "") for s in (card.get("skills") or [])]
    model = (card.get("metadata") or {}).get("model", "")
    streaming = (card.get("capabilities") or {}).get("streaming", False)

    return json.dumps(
        {
            "name": name,
            "description": description,
            "skills": skills,
            "model": model,
            "streaming": streaming,
            "endpoint": url,
            "raw": card,
        },
        ensure_ascii=False,
        indent=2,
    )


def _extract_artifacts_text(result: Dict) -> str:
    """Extract concatenated text from a task result's artifacts."""
    texts: list[str] = []
    for artifact in (result.get("artifacts") or []):
        for part in (artifact.get("parts") or []):
            if part.get("type") == "text" and part.get("text"):
                texts.append(part["text"])
    return "\n".join(texts).strip()


def _call_streaming(url: str, payload: Dict, headers: Dict) -> str:
    """
    Call a remote A2A agent via ``tasks/sendSubscribe`` (SSE streaming).

    Accumulates artifact text from streamed events and returns the final
    response once a ``completed`` / ``failed`` / ``canceled`` status event
    is received.  Falls back gracefully if the server closes the stream
    without a terminal event.
    """
    payload = dict(payload, method="tasks/sendSubscribe")
    accumulated = ""

    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            with client.stream("POST", url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line.startswith("data:"):
                        continue
                    data_str = raw_line[len("data:"):].strip()
                    if not data_str:
                        continue
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    result = event.get("result") or {}

                    # Accumulate artifact text from partial updates
                    chunk = _extract_artifacts_text(result)
                    if chunk:
                        accumulated = chunk  # server sends cumulative text

                    # Terminal state — stop reading
                    state = (result.get("status") or {}).get("state", "")
                    if state in ("completed", "failed", "canceled"):
                        break

    except httpx.HTTPStatusError as e:
        return json.dumps({"error": f"HTTP {e.response.status_code}: {e.response.text[:300]}"})
    except Exception as e:
        return json.dumps({"error": f"Streaming request failed: {e}"})

    if not accumulated:
        return json.dumps({"error": "Agent returned an empty streaming response"})
    return accumulated


def a2a_call(
    url: str,
    message: str,
    session_id: Optional[str] = None,
    bearer_token: Optional[str] = None,
    stream: Optional[bool] = None,
) -> str:
    """
    Send a task to a remote A2A agent and return its response.

    Automatically uses SSE streaming (``tasks/sendSubscribe``) when the agent's
    Agent Card advertises ``capabilities.streaming: true``, falling back to
    ``tasks/send`` otherwise.  Pass ``stream=True`` or ``stream=False`` to
    override auto-detection.  Pass ``session_id`` to maintain conversation
    context across multiple calls to the same agent.

    Args:
        url:          Base URL of the A2A agent (e.g. http://192.168.1.100:9000)
        message:      The message / task to send
        session_id:   Optional session ID for multi-turn conversations
        bearer_token: Optional Bearer token if the agent requires auth
        stream:       Force streaming on/off; None = auto-detect from Agent Card

    Returns:
        The agent's response text, or a JSON error object.
    """
    url = url.rstrip("/")
    task_id = str(uuid.uuid4())[:8]
    ctx_id = session_id or str(uuid.uuid4())[:8]

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"

    # Auto-detect streaming from cached Agent Card
    if stream is None:
        try:
            card = _fetch_agent_card(url)
            stream = bool((card.get("capabilities") or {}).get("streaming", False))
        except Exception:
            stream = False

    payload = {
        "jsonrpc": "2.0",
        "id": task_id,
        "method": "tasks/send",  # overridden below for streaming
        "params": {
            "id": task_id,
            "sessionId": ctx_id,
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": message}],
            },
        },
    }

    if stream:
        return _call_streaming(url, payload, headers)

    # Non-streaming path
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        return json.dumps({"error": f"HTTP {e.response.status_code}: {e.response.text[:300]}"})
    except Exception as e:
        return json.dumps({"error": f"Request failed: {e}"})

    if "error" in data:
        err = data["error"]
        return json.dumps({"error": err.get("message", str(err))})

    result = data.get("result") or {}
    response_text = _extract_artifacts_text(result)
    if not response_text:
        return json.dumps({"error": "Agent returned an empty response", "raw": result})

    return response_text


# ---------------------------------------------------------------------------
# Tool wrappers (synchronous, matching Hermes tool interface)
# ---------------------------------------------------------------------------

def _tool_a2a_discover(args: Dict[str, Any], **_kw) -> str:
    url = args.get("url", "").strip()
    if not url:
        return json.dumps({"error": "url is required"})
    return a2a_discover(url)


def _tool_a2a_call(args: Dict[str, Any], **_kw) -> str:
    url = args.get("url", "").strip()
    message = args.get("message", "").strip()
    session_id = args.get("session_id")
    bearer_token = args.get("bearer_token")
    stream = args.get("stream")  # None = auto-detect

    if not url and args.get("agent_name"):
        # Resolve named agent from config
        agents = _load_a2a_agents()
        agent_cfg = agents.get(args["agent_name"])
        if agent_cfg:
            url = agent_cfg.get("url", "")
            if not bearer_token:
                bearer_token = agent_cfg.get("bearer_token")
        else:
            available = list(agents.keys())
            return json.dumps(
                {
                    "error": f"Agent '{args['agent_name']}' not found in config.yaml a2a_agents.",
                    "available_agents": available,
                }
            )

    if not url:
        return json.dumps({"error": "Provide 'url' or 'agent_name'"})
    if not message:
        return json.dumps({"error": "message is required"})

    return a2a_call(url, message, session_id=session_id, bearer_token=bearer_token, stream=stream)


# ---------------------------------------------------------------------------
# OpenAI Function Schemas
# ---------------------------------------------------------------------------

def _build_discover_description() -> str:
    return (
        "Fetch the Agent Card from any A2A-compatible agent endpoint. "
        "Returns the agent's name, description, skills, model, and capabilities. "
        "Use this to learn what a remote agent can do before calling it."
    )


def _build_call_description() -> str:
    agents = _load_a2a_agents()
    base = (
        "Send a task to a remote A2A agent and get its response. "
        "Supports any agent that implements the Google A2A protocol "
        "(Hermes, LangChain, CrewAI, AutoGen, Vertex AI agents, etc.).\n\n"
        "Automatically uses SSE streaming when the agent supports it (detected from "
        "its Agent Card). Override with stream=true/false if needed.\n\n"
        "Provide either 'url' (direct endpoint) or 'agent_name' (from config).\n"
    )
    if agents:
        names = ", ".join(
            f"{name} — {cfg.get('description', cfg.get('url', ''))}"
            for name, cfg in agents.items()
        )
        base += f"\nConfigured agents: {names}"
    return base


A2A_DISCOVER_SCHEMA = {
    "name": "a2a_discover",
    "description": _build_discover_description(),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Base URL of the A2A agent (e.g. http://192.168.1.100:9000)",
            }
        },
        "required": ["url"],
    },
}

A2A_CALL_SCHEMA = {
    "name": "a2a_call",
    "description": _build_call_description(),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Direct URL of the A2A agent. Required if agent_name is not provided.",
            },
            "agent_name": {
                "type": "string",
                "description": "Name of a pre-configured agent from config.yaml a2a_agents section.",
            },
            "message": {
                "type": "string",
                "description": "The task or question to send to the remote agent.",
            },
            "session_id": {
                "type": "string",
                "description": (
                    "Optional session ID for multi-turn conversation. "
                    "Reuse the same ID across calls to maintain context."
                ),
            },
            "bearer_token": {
                "type": "string",
                "description": "Optional Bearer token if the remote agent requires authentication.",
            },
            "stream": {
                "type": "boolean",
                "description": (
                    "Use SSE streaming (tasks/sendSubscribe). "
                    "Omit to auto-detect from the agent's Agent Card capabilities."
                ),
            },
        },
        "required": ["message"],
    },
}


# ---------------------------------------------------------------------------
# Local port scan
# ---------------------------------------------------------------------------

_SCAN_TIMEOUT = 2.0  # seconds per port probe
_DEFAULT_SCAN_START = 9000
_DEFAULT_SCAN_END = 9010


def a2a_local_scan(
    host: str = "localhost",
    port_start: int = _DEFAULT_SCAN_START,
    port_end: int = _DEFAULT_SCAN_END,
) -> str:
    """
    Scan a range of localhost ports for running A2A agents.

    Tries GET /{port}/.well-known/agent.json on each port in [port_start, port_end].
    Returns a JSON list of discovered agents with their name, description, skills,
    and endpoint URL.

    Args:
        host:       Host to scan (default: localhost)
        port_start: First port to probe (default: 9000)
        port_end:   Last port to probe inclusive (default: 9010)
    """
    found = []
    with httpx.Client(timeout=_SCAN_TIMEOUT) as client:
        for port in range(port_start, port_end + 1):
            url = f"http://{host}:{port}"
            card_url = f"{url}/.well-known/agent.json"
            try:
                resp = client.get(card_url)
                if resp.status_code == 200:
                    card = resp.json()
                    found.append({
                        "endpoint": url,
                        "name": card.get("name", "unknown"),
                        "description": card.get("description", ""),
                        "skills": [s.get("name", "") for s in (card.get("skills") or [])],
                        "streaming": (card.get("capabilities") or {}).get("streaming", False),
                    })
            except Exception:
                pass  # port not open or not an A2A agent — skip silently

    if not found:
        return json.dumps({
            "found": 0,
            "message": f"No A2A agents found on {host} ports {port_start}-{port_end}",
            "agents": [],
        })

    return json.dumps({
        "found": len(found),
        "agents": found,
    }, ensure_ascii=False, indent=2)


def _tool_a2a_local_scan(args: Dict[str, Any], **_kw) -> str:
    host = args.get("host", "localhost")
    port_start = int(args.get("port_start", _DEFAULT_SCAN_START))
    port_end = int(args.get("port_end", _DEFAULT_SCAN_END))
    if port_start > port_end:
        return json.dumps({"error": "port_start must be <= port_end"})
    if port_end - port_start > 100:
        return json.dumps({"error": "Scan range too large (max 100 ports)"})
    return a2a_local_scan(host=host, port_start=port_start, port_end=port_end)


A2A_LOCAL_SCAN_SCHEMA = {
    "name": "a2a_local_scan",
    "description": (
        "Scan localhost ports to discover running A2A agents. "
        "Probes each port for a /.well-known/agent.json endpoint. "
        "Use this when you need to find what A2A agents are currently running locally "
        "without knowing their ports in advance. "
        f"Default scan range: ports {_DEFAULT_SCAN_START}-{_DEFAULT_SCAN_END}."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "host": {
                "type": "string",
                "description": "Host to scan (default: localhost)",
            },
            "port_start": {
                "type": "integer",
                "description": f"First port to probe (default: {_DEFAULT_SCAN_START})",
            },
            "port_end": {
                "type": "integer",
                "description": f"Last port to probe inclusive (default: {_DEFAULT_SCAN_END})",
            },
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from tools.registry import registry

registry.register(
    name="a2a_discover",
    toolset="a2a",
    schema=A2A_DISCOVER_SCHEMA,
    handler=_tool_a2a_discover,
    check_fn=check_a2a_requirements,
    emoji="🔍",
)

registry.register(
    name="a2a_call",
    toolset="a2a",
    schema=A2A_CALL_SCHEMA,
    handler=_tool_a2a_call,
    check_fn=check_a2a_requirements,
    emoji="🤝",
)

registry.register(
    name="a2a_local_scan",
    toolset="a2a",
    schema=A2A_LOCAL_SCAN_SCHEMA,
    handler=_tool_a2a_local_scan,
    check_fn=check_a2a_requirements,
    emoji="📡",
)

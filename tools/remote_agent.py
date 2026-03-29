#!/usr/bin/env python3
"""
Remote Agent Module

Handles communication with remote Hermes instances via OpenAI-compatible
API endpoints. Supports synchronous calls and streaming.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import httpx, fall back to requests
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    try:
        import requests
        HAS_REQUESTS = True
    except ImportError:
        HAS_REQUESTS = False

# Default timeout for remote agent calls
DEFAULT_TIMEOUT = 300


def _resolve_env_var(value: str) -> str:
    """Resolve ${VAR_NAME} syntax to environment variable value."""
    if not value:
        return ""
    if value.startswith("${") and value.endswith("}"):
        var_name = value[2:-1]
        return os.environ.get(var_name, "")
    return value


def load_remote_agents_config() -> Dict[str, Any]:
    """
    Load remote agents configuration from ~/.hermes/config.yaml.
    
    Returns a dict mapping agent names to their configurations.
    """
    try:
        from hermes_cli.config import load_config
        config = load_config()
        remote_agents = config.get("remote_agents", {})
        
        # Resolve environment variables in config
        resolved = {}
        for name, agent_config in remote_agents.items():
            if agent_config and isinstance(agent_config, dict):
                resolved[name] = {
                    "endpoint": agent_config.get("endpoint", ""),
                    "api_key": _resolve_env_var(agent_config.get("api_key", "")),
                    "model": agent_config.get("model", "qwen3.5"),
                    "timeout": agent_config.get("timeout", DEFAULT_TIMEOUT),
                    "description": agent_config.get("description", f"Remote agent: {name}"),
                }
            else:
                resolved[name] = {
                    "endpoint": agent_config if isinstance(agent_config, str) else "",
                    "api_key": "",
                    "model": "qwen3.5",
                    "timeout": DEFAULT_TIMEOUT,
                    "description": f"Remote agent: {name}",
                }
        return resolved
    except ImportError:
        logger.debug("hermes_cli.config not available, no remote agents configured")
        return {}
    except Exception as e:
        logger.warning("Failed to load remote agents config: %s", e)
        return {}


# Cache for loaded config
_remote_agents_config = None


def get_remote_agents() -> Dict[str, Any]:
    """Get cached remote agents configuration."""
    global _remote_agents_config
    if _remote_agents_config is None:
        _remote_agents_config = load_remote_agents_config()
    return _remote_agents_config


def refresh_remote_agents_config():
    """Force reload of remote agents configuration."""
    global _remote_agents_config
    _remote_agents_config = None


def _build_remote_agent_prompt(
    goal: str,
    context: Optional[str],
    toolsets: Optional[list]
) -> str:
    """Build system prompt for remote agent."""
    parts = [
        "You are a remote Hermes agent working on a delegated task.",
        "",
        f"YOUR TASK:\n{goal}",
    ]
    
    if context and context.strip():
        parts.append(f"\nCONTEXT:\n{context}")
    
    if toolsets:
        parts.append(f"\nAVAILABLE TOOLS: {', '.join(toolsets)}")
    
    parts.append(
        "\nComplete this task using your tools. When finished, provide a "
        "clear, concise summary of:\n"
        "- What you did\n"
        "- What you found or accomplished\n"
        "- Any files you created or modified\n"
        "- Any issues encountered"
    )
    
    return "\n".join(parts)


def _blocking_request(
    agent_name: str,
    endpoint: str,
    agent_config: Dict[str, Any],
    messages: list
) -> Dict[str, Any]:
    """Make a blocking HTTP request to the remote agent."""
    url = endpoint.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }

    api_key = agent_config.get("api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    payload = {
        "model": agent_config.get("model", "hermes-agent"),
        "messages": messages,
        "stream": False,
    }
    
    timeout = agent_config.get("timeout", DEFAULT_TIMEOUT)
    
    try:
        if HAS_HTTPX:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return {"content": content, "agent": agent_name}
        elif HAS_REQUESTS:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"content": content, "agent": agent_name}
        else:
            return {"error": "No HTTP client available. Install 'httpx' or 'requests'."}
            
    except Exception as e:
        logger.error("HTTP error calling %s: %s", agent_name, e)
        return {"error": f"HTTP error: {str(e)}"}


def _stream_request(
    agent_name: str,
    endpoint: str,
    agent_config: Dict[str, Any],
    messages: list
) -> Dict[str, Any]:
    """Make a streaming HTTP request to the remote agent."""
    import httpx
    
    url = endpoint.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }
    
    api_key = agent_config.get("api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    payload = {
        "model": agent_config.get("model", "qwen3.5"),
        "messages": messages,
        "stream": True,
    }
    
    full_content = ""
    
    try:
        with httpx.Client(timeout=agent_config.get("timeout", DEFAULT_TIMEOUT)) as client:
            with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            json_data = json.loads(data)
                            delta = json_data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_content += content
                        except json.JSONDecodeError:
                            continue
        
        return {"content": full_content, "agent": agent_name}
        
    except httpx.HTTPError as e:
        logger.error("HTTP error calling %s: %s", agent_name, e)
        return {"error": f"HTTP error: {str(e)}"}


def call_remote_agent(
    agent_name: str,
    goal: str,
    context: Optional[str] = None,
    toolsets: Optional[list] = None,
    stream: bool = False,
) -> Dict[str, Any]:
    """
    Make a single request to a remote agent.
    
    Args:
        agent_name: Name of the remote agent from config
        goal: The task/goal to accomplish
        context: Additional context for the task
        toolsets: Toolsets to enable (optional)
        stream: Whether to stream the response
        
    Returns:
        Dict with 'content' key containing the response, or error dict
    """
    config = get_remote_agents()
    agent_config = config.get(agent_name)
    
    if not agent_config:
        return {
            "error": f"Unknown remote agent: {agent_name}. "
                     f"Available agents: {', '.join(config.keys()) or 'none'}"
        }
    
    endpoint = agent_config.get("endpoint")
    if not endpoint:
        return {"error": f"Remote agent {agent_name} has no endpoint configured"}
    
    # Build the request
    system_prompt = _build_remote_agent_prompt(goal, context, toolsets)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": goal}
    ]
    
    # Make the request
    try:
        if stream:
            return _stream_request(agent_name, endpoint, agent_config, messages)
        else:
            return _blocking_request(agent_name, endpoint, agent_config, messages)
    except Exception as e:
        logger.exception("Remote agent call failed: %s", e)
        return {"error": f"Remote agent {agent_name} failed: {str(e)}"}

"""Federation tools for cross-machine agent communication.

Registered tools:
    federation_discover — List and probe configured federation peers.
    federation_delegate — Send a task to a remote Hermes peer.
"""
import asyncio
import json
import logging

from tools.registry import registry

logger = logging.getLogger(__name__)


def federation_discover(tag: str = "", task_id: str = None) -> str:
    """Discovery federation peers and their status.

    Args:
        tag: Optional tag filter (e.g. "gpu", "code-review").
    """
    from agent.federation import get_federation_manager

    fm = get_federation_manager()
    if not fm:
        return json.dumps({
            "enabled": False,
            "peers": [],
            "message": ("Federation is not enabled. "
                        "Set federation.enabled=true in config.yaml and configure peers."),
        })

    peers = fm.list_peers(tag_filter=tag or None)
    return json.dumps({
        "enabled": True,
        "peers": [
            {
                "name": p.name,
                "url": p.url,
                "tags": p.tags,
                "status": p.status,
                "last_seen": p.last_seen,
            }
            for p in peers
        ],
    })


def federation_delegate(peer: str, goal: str, context: str = "",
                        skills: list = None, timeout: int = 300,
                        task_id: str = None) -> str:
    """Delegate a task to a remote Hermes peer via federation.

    Args:
        peer: Peer name as configured in config.yaml federation.peers[].name.
        goal: Task goal for the remote agent.
        context: Optional background context for the remote agent.
        skills: Optional list of skill names to load on the remote side.
        timeout: Max seconds to wait for remote result (default 300).
    """
    from agent.federation import get_federation_manager, FederationTask

    fm = get_federation_manager()
    if not fm:
        return json.dumps({
            "success": False,
            "error": "Federation is not enabled.",
        })

    task = FederationTask(
        peer_name=peer,
        goal=goal,
        context=context,
        skills=skills or [],
        timeout_seconds=timeout,
    )

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    result = loop.run_until_complete(fm.delegate(task))
    return json.dumps({
        "success": result.success,
        "peer": result.peer_name,
        "task_id": result.task_id,
        "result": result.result,
        "error": result.error,
        "elapsed_ms": round(result.elapsed_ms, 1),
    })


# ═══ Register ═══

registry.register(
    name="federation_discover",
    toolset="federation",
    schema={
        "name": "federation_discover",
        "description": (
            "Discover and list federation peers. Use this to see what remote Hermes "
            "instances are available for cross-machine task delegation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tag": {
                    "type": "string",
                    "description": "Optional tag filter to narrow peers by capability.",
                },
            },
            "required": [],
        },
    },
    handler=lambda args, **kw: federation_discover(tag=args.get("tag", ""), **kw),
    requires_env=[],
)

registry.register(
    name="federation_delegate",
    toolset="federation",
    schema={
        "name": "federation_delegate",
        "description": (
            "Delegate a task to a remote Hermes peer on another machine. "
            "The remote peer processes the task and returns the result. "
            "Use this for cross-machine work — GPU-heavy tasks, isolated environments, "
            "or specialized agents on different hardware."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "peer": {
                    "type": "string",
                    "description": "Name of the federation peer to delegate to.",
                },
                "goal": {
                    "type": "string",
                    "description": "Task goal for the remote agent.",
                },
                "context": {
                    "type": "string",
                    "description": "Optional background context.",
                },
                "skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional skill names to load remotely.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max seconds to wait (default 300).",
                },
            },
            "required": ["peer", "goal"],
        },
    },
    handler=lambda args, **kw: federation_delegate(
        peer=args["peer"],
        goal=args["goal"],
        context=args.get("context", ""),
        skills=args.get("skills"),
        timeout=args.get("timeout", 300),
        **kw,
    ),
    requires_env=[],
)
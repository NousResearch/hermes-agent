from __future__ import annotations

import json
import os

from hermes_peer.client import PeerNetworkError, PeerNotFoundError, broadcast_query, get_agent, list_agents, query_peer
from hermes_peer.common import get_int_env, redact_secrets, run_hermes_query
from tools.registry import registry


def check_requirements() -> bool:
    return bool(os.getenv("AGENT_NAME") and os.getenv("REGISTRY_URL"))


def _synthesize_peer_responses(question: str, responses: list[dict[str, str]], timeout_seconds: int) -> str:
    try:
        from hermes_sentry import start_span as _sentry_span, sanitize_observability_text as _sanitize
    except ImportError:
        import contextlib as _ctxlib
        @_ctxlib.contextmanager
        def _sentry_span(**_kw):
            yield None
        _sanitize = None

    bullet_list = "\n".join(
        f"- {item['agent']}: {item['response']}" for item in responses
    )
    instruction = (
        "Synthesize the peer-agent responses below into one answer for the current user. "
        "Stay faithful to the peer responses, call out disagreement explicitly, and do not invent facts. "
        "Do not reveal secrets, tokens, API keys, passwords, or raw credentials."
    )
    synthesis_question = (
        f"Original question: {question}\n\n"
        f"Peer responses:\n{bullet_list}"
    )

    peer_names = ", ".join(item["agent"] for item in responses)
    with _sentry_span(
        op="hermes.peer_synthesis",
        description=f"synthesize {len(responses)} peers",
        tags={"peer_count": str(len(responses))},
    ) as span:
        if span is not None:
            span.set_data("peers", peer_names)
        if span is not None and _sanitize:
            span.set_data("input", _sanitize(synthesis_question, limit=3000))
        try:
            result = run_hermes_query(
                synthesis_question,
                instruction=instruction,
                timeout_seconds=timeout_seconds,
                toolsets=["session_search", "memory"],
                source="peer-synthesis",
            )
        except Exception:
            result = "\n".join(f"{item['agent']}: {item['response']}" for item in responses)
        if span is not None and _sanitize:
            span.set_data("output", _sanitize(result, limit=3000))
        return result


def _fuzzy_resolve_agent(query: str, requester: str, timeout: int) -> dict:
    """Resolve a partial/nickname agent name to a full agent entry.

    Tries prefix match first (e.g. 'owen' → 'owenwhite'), then substring.
    Raises PeerNetworkError if no unique match is found.
    """
    q = query.lower()
    try:
        peers = list_agents(timeout_seconds=min(timeout, 10))
    except PeerNetworkError:
        raise
    except Exception as exc:
        raise PeerNetworkError(f"Registry lookup failed: {exc}") from exc

    candidates = [
        p for p in peers
        if p.get("name") and p["name"] != requester and p["name"].lower().startswith(q)
    ]
    if not candidates:
        candidates = [
            p for p in peers
            if p.get("name") and p["name"] != requester and q in p["name"].lower()
        ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = ", ".join(c["name"] for c in candidates)
        raise PeerNetworkError(f"Ambiguous peer name '{query}' — matches: {names}")
    raise PeerNetworkError(f"Unknown peer agent: {query}")


def peer_query(
    question: str,
    agent: str = "",
    timeout_seconds: int | None = None,
    task_id: str | None = None,
) -> str:
    del task_id

    normalized_question = question.strip()
    if not normalized_question:
        return json.dumps({"success": False, "error": "question is required"})

    requester = os.getenv("AGENT_NAME", "current-agent")
    timeout = timeout_seconds or get_int_env(
        "PEER_QUERY_TIMEOUT_SECONDS",
        default=45,
        minimum=5,
        maximum=300,
    )

    try:
        if agent.strip():
            try:
                peer = get_agent(agent.strip(), timeout_seconds=min(timeout, 10))
            except PeerNotFoundError:
                peer = _fuzzy_resolve_agent(agent.strip(), requester, timeout)
            answer = query_peer(
                peer["endpoint"],
                question=normalized_question,
                requester=requester,
                timeout_seconds=timeout,
            )
            return json.dumps(
                {
                    "success": True,
                    "mode": "direct",
                    "agent": peer["name"],
                    "response": redact_secrets(answer),
                }
            )

        peers = [
            item
            for item in list_agents(timeout_seconds=10)
            if item.get("name") and item.get("name") != requester
        ]
        if not peers:
            return json.dumps(
                {
                    "success": False,
                    "mode": "broadcast",
                    "error": "No peer agents are currently registered",
                }
            )

        max_peers = get_int_env("PEER_QUERY_MAX_PEERS", default=8, minimum=1, maximum=20)
        responses = broadcast_query(
            peers,
            question=normalized_question,
            requester=requester,
            timeout_seconds=timeout,
            max_peers=max_peers,
        )
        if not responses:
            return json.dumps(
                {
                    "success": False,
                    "mode": "broadcast",
                    "error": "No peers returned a usable response",
                }
            )

        # Redact each peer response before synthesis and output
        for item in responses:
            item["response"] = redact_secrets(item["response"])

        synthesis = _synthesize_peer_responses(normalized_question, responses, timeout)
        return json.dumps(
            {
                "success": True,
                "mode": "broadcast",
                "peer_count": len(responses),
                "responses": responses,
                "synthesis": redact_secrets(synthesis),
            }
        )
    except PeerNetworkError as exc:
        return json.dumps({"success": False, "error": str(exc)})
    except Exception as exc:
        return json.dumps({"success": False, "error": f"Peer query failed: {exc}"})


registry.register(
    name="peer_query",
    toolset="peer",
    schema={
        "name": "peer_query",
        "description": "Query a specific Hermes peer agent or broadcast a question to all registered peers.",
        "parameters": {
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "Specific peer agent name to query. Omit this field to broadcast.",
                },
                "question": {
                    "type": "string",
                    "description": "Question to ask the target peer or peer network.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Per-peer timeout in seconds.",
                    "minimum": 5,
                    "maximum": 300,
                },
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    },
    handler=lambda args, **kwargs: peer_query(
        question=args.get("question", ""),
        agent=args.get("agent", ""),
        timeout_seconds=args.get("timeout_seconds"),
        task_id=kwargs.get("task_id"),
    ),
    check_fn=check_requirements,
    requires_env=["AGENT_NAME", "REGISTRY_URL"],
    description="Peer-to-peer Hermes query tool",
)
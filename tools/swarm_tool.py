"""swarm_run — native Hermes tool for spawning real, coordinated multi-agent swarms.

Why this exists
---------------
Ruflo (and its ``swarm_init`` / ``agent_spawn`` MCP tools) was a coordination
*ledger* — it wrote JSON files describing intended agents but never actually
executed them.  The LLM saw "agent registered" responses and proceeded as if
work had happened; nothing had.

``swarm_run`` is the real thing.  It accepts a list of agents and:

  1. Creates a swarm record in the local hermes-swarm coordination plane
     (memory + tasks + messaging — see ``~/repos/hermes-swarm``).
  2. Spawns the agents through ``delegate_task`` — Hermes' native parallel
     subagent mechanism.  Each child is a real ``AIAgent`` instance with
     real LLM calls, not a JSON record.
  3. Wires each child to the swarm's coordination plane so peers can share
     findings, broadcast messages, and run consensus polls.
  4. Injects the matching persona prompt (from ``~/.hermes/personas/``) and
     resolves the per-role model from ``delegation.model_by_role``.

Topologies
----------
  parallel    All agents run concurrently in a single ``delegate_task`` batch.
              Bound by ``delegation.max_concurrent_children``.  Best for
              independent work (e.g. analyse 3 EMG bundles).

  sequential  Agents run one at a time, in declared order.  Each agent's
              context inherits the previous agents' results.  Use when
              earlier outputs inform later inputs.

  pipeline    Same as sequential but with explicit "your input is the
              previous agent's output" framing.  Use when the work is
              genuinely a transform chain (researcher → analyst → reviewer).

  hierarchical  First N-1 agents run in parallel as workers; the last
                agent runs after, receives all worker outputs as context,
                and synthesizes them.  Common pattern: 3 analysts → 1
                reviewer.

For a true mesh topology the workers need to talk *during* execution.  That
already works through the hermes-swarm MCP tools: any agent in any topology
can call ``swarm_broadcast`` / ``swarm_inbox`` mid-task.  The topology arg
just controls *spawn order*.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_TOPOLOGIES = ("parallel", "sequential", "pipeline", "hierarchical")
DEFAULT_TOPOLOGY = "parallel"

# Soft cap on agents per swarm.  Above this, LLMs almost certainly chose
# the wrong tool — a real swarm is 2–10 agents, not 50.  The hard cap from
# delegation.max_concurrent_children still applies for parallel mode.
MAX_AGENTS_PER_SWARM = 20


# ---------------------------------------------------------------------------
# Swarm context prelude — injected into every child's context so they know
# their identity, their swarm_id, and how to use the coordination plane.
#
# We build this as a plain text block (not a system-prompt block) because
# delegate_task already builds the system prompt; adding to context is the
# clean, side-effect-free path that doesn't touch internal delegate plumbing.
# ---------------------------------------------------------------------------


def _build_swarm_prelude(
    swarm_id: str,
    agent_id: str,
    agent_type: str,
    topology: str,
    peers: List[Dict[str, str]],
    role_in_swarm: str,
) -> str:
    """Compose the swarm-context block prepended to a child's ``context`` field.

    The prelude tells the child:
      - who they are (agent_id, agent_type, role-in-swarm)
      - what swarm they're in (swarm_id, topology)
      - who their peers are (so they can target broadcasts / DMs)
      - which MCP tools coordinate the swarm (and how to call them)

    Children don't need to memorise this — the swarm_* tools are visible in
    their toolset; the prelude just makes sure they USE them instead of
    operating in isolation.
    """
    peer_lines = "\n".join(
        f"  - {p['agent_id']}  ({p['agent_type']}) — {p.get('goal', '')[:80]}"
        for p in peers
    )
    return (
        "## SWARM COORDINATION CONTEXT\n"
        f"You are participating in a coordinated swarm.\n\n"
        f"  Your agent_id:  {agent_id}\n"
        f"  Your role:      {agent_type}  ({role_in_swarm})\n"
        f"  Swarm id:       {swarm_id}\n"
        f"  Topology:       {topology}\n"
        f"  Peers ({len(peers)}):\n{peer_lines}\n\n"
        "## How to coordinate\n"
        "You have access to the following hermes-swarm MCP tools.  Use them — \n"
        "they are how your swarm shares state.  The tools accept ``swarm_id`` \n"
        "and ``agent_id`` parameters — pass YOUR ids shown above on every \n"
        "call (don't rely on env-var defaults; you're spawned in-process).\n\n"
        "Note: the registered tool names carry a doubled ``swarm_`` (the\n"
        "first comes from the MCP server name ``hermes-swarm``, the second\n"
        "from the tool's own name).  Emit them exactly as shown below — \n"
        "guessing the singular form will trigger auto-repair on every call.\n\n"
        "  Memory (publish + read findings)\n"
        "    mcp_hermes_swarm_swarm_memory_store(key, value, tags?)\n"
        "    mcp_hermes_swarm_swarm_memory_get(key)\n"
        "    mcp_hermes_swarm_swarm_memory_search(query)\n"
        "    mcp_hermes_swarm_swarm_memory_list(prefix?, tag?)\n\n"
        "  Messaging (peer comms)\n"
        "    mcp_hermes_swarm_swarm_broadcast(body)            — send to all peers\n"
        "    mcp_hermes_swarm_swarm_send_message(recipient, body)  — DM one peer\n"
        "    mcp_hermes_swarm_swarm_inbox(since?)              — read messages addressed to you\n\n"
        "  Tasks (work-queue handoff between peers)\n"
        "    mcp_hermes_swarm_swarm_task_create(description, assignee?)\n"
        "    mcp_hermes_swarm_swarm_task_claim() / _swarm_task_complete(id, result)\n\n"
        "  Voting (consensus)\n"
        "    mcp_hermes_swarm_swarm_vote_open(question, options) / _swarm_vote_cast / _swarm_vote_tally\n\n"
        "  Lifecycle (mark yourself running/done)\n"
        f"    mcp_hermes_swarm_swarm_update_agent(agent_id='{agent_id}', "
        f"swarm_id='{swarm_id}', started=true)  — call at start\n"
        f"    mcp_hermes_swarm_swarm_update_agent(agent_id='{agent_id}', "
        f"swarm_id='{swarm_id}', ended=true, result='<summary>')  — call at end\n\n"
        "## Coordination contract\n"
        "  1. As soon as you find something material to your task, store it \n"
        "     under a namespaced key like ``finding:<your-agent-id>:<n>`` so \n"
        "     peers can read it.  Don't hoard findings until your final \n"
        "     summary — your peers may need them mid-task.\n"
        "  2. Skim the inbox at the start of long tool sequences (every \n"
        "     ~5 tool calls) — peers may have broadcast useful context.\n"
        "  3. Your final response to the parent is still the authoritative \n"
        "     summary.  The swarm tools are for *intra-swarm* coordination.\n"
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_agents(agents: Any) -> List[Dict[str, Any]]:
    """Coerce + validate the ``agents`` arg.  Raises ValueError on bad input."""
    if not isinstance(agents, list) or not agents:
        raise ValueError("agents must be a non-empty list")
    if len(agents) > MAX_AGENTS_PER_SWARM:
        raise ValueError(
            f"too many agents: {len(agents)} (max {MAX_AGENTS_PER_SWARM} per swarm)"
        )
    out: List[Dict[str, Any]] = []
    for i, a in enumerate(agents):
        if not isinstance(a, dict):
            raise ValueError(f"agents[{i}] must be a dict")
        agent_type = (a.get("type") or a.get("agent_type") or "").strip()
        goal = (a.get("goal") or "").strip()
        if not agent_type:
            raise ValueError(f"agents[{i}] missing 'type'")
        if not goal:
            raise ValueError(f"agents[{i}] missing 'goal'")
        out.append({
            "type": agent_type,
            "goal": goal,
            "context": a.get("context"),
            "model": a.get("model"),
            "toolsets": a.get("toolsets"),
            # Optional caller-supplied id; we generate one if not given.
            "agent_id": (a.get("agent_id") or "").strip() or None,
        })
    return out


def _validate_topology(topology: Optional[str]) -> str:
    t = (topology or DEFAULT_TOPOLOGY).strip().lower()
    if t not in VALID_TOPOLOGIES:
        raise ValueError(
            f"unknown topology: {t!r}  (valid: {', '.join(VALID_TOPOLOGIES)})"
        )
    return t


# ---------------------------------------------------------------------------
# Hermes-swarm hookup — best-effort.  When hermes-swarm is importable, we
# pre-register the swarm + agents so swarm_status / swarm_peers work
# immediately.  When it's not, we still set the IDs and let the children
# create the swarm lazily on first MCP call.
# ---------------------------------------------------------------------------


def _try_preregister_swarm(
    swarm_id: str,
    title: str,
    topology: str,
    agents: List[Dict[str, Any]],
) -> bool:
    """If hermes-swarm is on sys.path, pre-create the swarm and register all
    agents so the coordination plane is populated before any child runs.

    Returns True on success, False on any failure (including ImportError).
    Failure is non-fatal — the MCP server's lazy-create still works.
    """
    try:
        from swarm import lifecycle as _lc  # type: ignore
    except ImportError:
        logger.debug(
            "hermes-swarm not importable; skipping pre-registration "
            "(children will create swarm lazily via MCP)"
        )
        return False
    try:
        _lc.create_swarm(title=title, topology=topology, swarm_id=swarm_id)
        for a in agents:
            _lc.register_agent(
                swarm_id,
                a["agent_id"],
                a["type"],
                role="leaf",
                model=a.get("model"),
                goal=a["goal"],
            )
        return True
    except Exception:
        logger.exception("hermes-swarm pre-registration failed; continuing")
        return False


def _try_end_swarm(swarm_id: str, status: str) -> None:
    try:
        from swarm import lifecycle as _lc  # type: ignore

        _lc.end_swarm(swarm_id, status=status)
    except Exception:
        # Non-fatal — the swarm record is just for observability.
        pass


# ---------------------------------------------------------------------------
# Topology executors
# ---------------------------------------------------------------------------


def _make_task(
    a: Dict[str, Any],
    *,
    swarm_id: str,
    topology: str,
    peers: List[Dict[str, str]],
    extra_context: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the dict shape that ``delegate_task(tasks=[...])`` expects."""
    prelude = _build_swarm_prelude(
        swarm_id=swarm_id,
        agent_id=a["agent_id"],
        agent_type=a["type"],
        topology=topology,
        peers=peers,
        role_in_swarm="worker",
    )
    pieces: List[str] = [prelude]
    if extra_context and extra_context.strip():
        pieces.append("\n## SHARED SWARM CONTEXT\n" + extra_context.strip())
    if a.get("context") and str(a["context"]).strip():
        pieces.append("\n## TASK CONTEXT\n" + str(a["context"]).strip())
    task: Dict[str, Any] = {
        "goal": a["goal"],
        "context": "\n".join(pieces),
        "agent_type": a["type"],
    }
    # Carry through optional per-task overrides.
    if a.get("model"):
        task["model"] = a["model"]
    if a.get("toolsets"):
        task["toolsets"] = a["toolsets"]
    return task


def _peer_summaries(agents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {
            "agent_id": a["agent_id"],
            "agent_type": a["type"],
            "goal": a["goal"],
        }
        for a in agents
    ]


def _run_parallel(
    agents: List[Dict[str, Any]],
    *,
    swarm_id: str,
    shared_context: Optional[str],
    parent_agent,
) -> Dict[str, Any]:
    """All agents run concurrently in a single delegate_task batch."""
    from tools.delegate_tool import delegate_task

    peers = _peer_summaries(agents)
    tasks = [
        _make_task(
            a,
            swarm_id=swarm_id,
            topology="parallel",
            peers=peers,
            extra_context=shared_context,
        )
        for a in agents
    ]
    raw = delegate_task(tasks=tasks, parent_agent=parent_agent)
    return _wrap_delegate_result(raw, agents)


def _run_sequential(
    agents: List[Dict[str, Any]],
    *,
    swarm_id: str,
    shared_context: Optional[str],
    parent_agent,
    pipeline_framing: bool = False,
) -> Dict[str, Any]:
    """Agents run one at a time.  Each gets prior outputs in their context.

    When ``pipeline_framing`` is True, the framing emphasises that this
    agent's INPUT is the previous agent's OUTPUT (transform-chain style).
    Otherwise it's just "here's what's happened so far" (sequential style).
    """
    from tools.delegate_tool import delegate_task

    peers = _peer_summaries(agents)
    accumulated: List[Dict[str, Any]] = []
    for idx, a in enumerate(agents):
        # Build extra context from prior agent outputs.
        prior_block_lines: List[str] = []
        if shared_context and shared_context.strip():
            prior_block_lines.append(shared_context.strip())
        if accumulated:
            for prev in accumulated:
                if pipeline_framing and prev is accumulated[-1]:
                    prior_block_lines.append(
                        f"\n## YOUR INPUT (output of upstream agent "
                        f"{prev['agent_id']} / {prev['type']})\n"
                        f"{prev['summary']}"
                    )
                else:
                    prior_block_lines.append(
                        f"\n### Prior agent {prev['agent_id']} ({prev['type']}) — output\n"
                        f"{prev['summary']}"
                    )
        extra = "\n".join(prior_block_lines) if prior_block_lines else None

        topology_label = "pipeline" if pipeline_framing else "sequential"
        task = _make_task(
            a,
            swarm_id=swarm_id,
            topology=topology_label,
            peers=peers,
            extra_context=extra,
        )
        raw = delegate_task(tasks=[task], parent_agent=parent_agent)
        wrapped = _wrap_delegate_result(raw, [a])
        if wrapped["results"]:
            accumulated.append({
                "agent_id": a["agent_id"],
                "type": a["type"],
                "summary": wrapped["results"][0].get("summary", ""),
            })

    return {
        "results": [
            {
                "agent_id": acc["agent_id"],
                "agent_type": acc["type"],
                "summary": acc["summary"],
            }
            for acc in accumulated
        ],
    }


def _run_hierarchical(
    agents: List[Dict[str, Any]],
    *,
    swarm_id: str,
    shared_context: Optional[str],
    parent_agent,
) -> Dict[str, Any]:
    """First N-1 agents run in parallel (workers); last agent runs after,
    receiving all worker outputs (synthesizer/reviewer)."""
    if len(agents) < 2:
        # Degenerate case: hierarchical of one agent is just parallel-of-one.
        return _run_parallel(
            agents, swarm_id=swarm_id,
            shared_context=shared_context, parent_agent=parent_agent,
        )

    workers, synthesizer = agents[:-1], agents[-1]

    # Phase 1: workers in parallel.
    worker_result = _run_parallel(
        workers, swarm_id=swarm_id,
        shared_context=shared_context, parent_agent=parent_agent,
    )

    # Phase 2: synthesizer with worker outputs threaded into context.
    worker_summary_block = "\n\n".join(
        f"### {r['agent_type']} ({r['agent_id']}) — output\n{r.get('summary', '')}"
        for r in worker_result["results"]
    )
    synth_context_pieces: List[str] = []
    if shared_context and shared_context.strip():
        synth_context_pieces.append(shared_context.strip())
    synth_context_pieces.append(
        "## WORKER OUTPUTS (synthesise these)\n" + worker_summary_block
    )
    extra = "\n".join(synth_context_pieces)

    synth_result = _run_parallel(
        [synthesizer], swarm_id=swarm_id,
        shared_context=extra, parent_agent=parent_agent,
    )

    return {
        "results": worker_result["results"] + synth_result["results"],
    }


# ---------------------------------------------------------------------------
# Result shaping
# ---------------------------------------------------------------------------


def _wrap_delegate_result(
    raw: str,
    agents: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Coerce delegate_task's JSON string output into a swarm-shaped dict.

    delegate_task returns ``{"results": [...]}`` or ``{"error": "..."}``.
    We re-key the inner results with our agent_id/agent_type so the LLM
    can correlate them with the agents list it submitted.
    """
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {"results": [], "error": f"delegate_task returned non-JSON: {raw[:200]}"}
    if "error" in parsed:
        return {"results": [], "error": parsed["error"]}
    inner = parsed.get("results") or []
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(inner):
        a = agents[i] if i < len(agents) else None
        entry: Dict[str, Any] = {
            "agent_id": a["agent_id"] if a else f"unknown-{i}",
            "agent_type": a["type"] if a else "unknown",
            # delegate_task currently returns 'summary' for the child's final
            # text output; fall back across known field names defensively.
            "summary": r.get("summary") or r.get("response") or r.get("output", ""),
            "ok": r.get("ok", True if "summary" in r or "response" in r else False),
        }
        # Carry through any cost/iteration metadata delegate exposes.
        for k in ("model", "duration_s", "iterations", "cost_usd",
                  "input_tokens", "output_tokens"):
            if k in r:
                entry[k] = r[k]
        out.append(entry)
    return {"results": out}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def swarm_run(
    agents: Optional[List[Dict[str, Any]]] = None,
    topology: Optional[str] = None,
    title: Optional[str] = None,
    shared_context: Optional[str] = None,
    swarm_id: Optional[str] = None,
    parent_agent=None,
) -> str:
    """Spawn a coordinated multi-agent swarm.

    See module docstring for topology semantics.  Returns a JSON string with
    shape ``{"swarm_id": "...", "topology": "...", "results": [...]}`` on
    success or ``{"error": "..."}`` on failure.
    """
    if parent_agent is None:
        return tool_error("swarm_run requires a parent agent context.")

    # Validate inputs.
    try:
        validated = _validate_agents(agents)
        topo = _validate_topology(topology)
    except ValueError as exc:
        return tool_error(str(exc))

    # Generate swarm_id and per-agent ids if not supplied.
    sid = (swarm_id or "").strip() or f"sw-{uuid.uuid4().hex[:12]}"
    swarm_title = (title or "").strip() or f"Hermes swarm {sid}"
    for i, a in enumerate(validated):
        if not a["agent_id"]:
            # Suffix with type for human readability in logs / inboxes.
            a["agent_id"] = f"a{i + 1}-{a['type'][:24]}"

    # Pre-register in hermes-swarm if available (best-effort).
    pre_registered = _try_preregister_swarm(
        sid, swarm_title, topo, validated,
    )

    started = time.monotonic()
    logger.info(
        "swarm_run start: id=%s title=%r topology=%s agents=%d pre_registered=%s",
        sid, swarm_title, topo, len(validated), pre_registered,
    )

    # Dispatch by topology.
    try:
        if topo == "parallel":
            outcome = _run_parallel(
                validated, swarm_id=sid,
                shared_context=shared_context, parent_agent=parent_agent,
            )
        elif topo == "sequential":
            outcome = _run_sequential(
                validated, swarm_id=sid,
                shared_context=shared_context, parent_agent=parent_agent,
                pipeline_framing=False,
            )
        elif topo == "pipeline":
            outcome = _run_sequential(
                validated, swarm_id=sid,
                shared_context=shared_context, parent_agent=parent_agent,
                pipeline_framing=True,
            )
        elif topo == "hierarchical":
            outcome = _run_hierarchical(
                validated, swarm_id=sid,
                shared_context=shared_context, parent_agent=parent_agent,
            )
        else:  # pragma: no cover — _validate_topology should have rejected
            return tool_error(f"unsupported topology: {topo}")
    except Exception as exc:
        logger.exception("swarm_run failed for %s", sid)
        _try_end_swarm(sid, "failed")
        return tool_error(f"swarm_run crashed: {exc}")

    duration = time.monotonic() - started

    # Decide swarm-level status.  If any child reported error/!ok, mark failed
    # but still return the partial results so the LLM can recover.
    failed = bool(outcome.get("error")) or any(
        not r.get("ok", True) for r in outcome.get("results", [])
    )
    _try_end_swarm(sid, "failed" if failed else "completed")

    response: Dict[str, Any] = {
        "swarm_id": sid,
        "title": swarm_title,
        "topology": topo,
        "agents": len(validated),
        "duration_s": round(duration, 1),
        "results": outcome.get("results", []),
    }
    if outcome.get("error"):
        response["error"] = outcome["error"]
    return json.dumps(response, default=str)


# ---------------------------------------------------------------------------
# Tool schema — this is what the LLM sees.
# ---------------------------------------------------------------------------


SWARM_RUN_SCHEMA = {
    "name": "swarm_run",
    "description": (
        "Spawn a real, coordinated multi-agent swarm.  Children run in "
        "parallel (or sequenced — see topology), share state via the "
        "hermes-swarm coordination plane (memory/tasks/messaging), and "
        "each runs with a persona prompt + per-role model from "
        "delegation.model_by_role.\n\n"
        "When to use:\n"
        "  * 2+ agents needed with distinct roles (researcher + analyst + "
        "reviewer; N analysts on N independent inputs; etc.).\n"
        "  * You want them to share findings as they work, not just at the "
        "end (use mcp_hermes_swarm_memory_store / _broadcast).\n\n"
        "When NOT to use:\n"
        "  * Only one subagent needed → use delegate_task directly.\n"
        "  * Mechanical multi-step work with no reasoning → use "
        "execute_code.\n\n"
        "Topologies:\n"
        "  parallel     — all agents concurrent (default).  Best for "
        "independent inputs.\n"
        "  sequential   — one at a time, each sees prior outputs.\n"
        "  pipeline     — chain: each agent's input is previous output.\n"
        "  hierarchical — first N-1 in parallel as workers; last "
        "synthesises their outputs.\n\n"
        "Each agent dict needs: ``type`` (persona name from "
        "~/.hermes/personas/, e.g. 'researcher', 'code-analyzer'), "
        "``goal`` (what to do).  Optional: ``context`` (extra info just "
        "for that agent), ``model`` (override per-role model), "
        "``toolsets`` (override default toolset list), ``agent_id`` "
        "(stable id; auto-generated if omitted)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "agents": {
                "type": "array",
                "description": (
                    "List of agents to spawn.  Hard ceiling: "
                    f"{MAX_AGENTS_PER_SWARM} per swarm.  In parallel topology "
                    "the number of children running concurrently is bounded "
                    "by delegation.max_concurrent_children (default 3); "
                    "extra agents queue and run as slots free up.  Raise "
                    "the cap from the CLI with /delegation parallel <N>, "
                    "or in ~/.hermes/config.yaml under "
                    "delegation.max_concurrent_children."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "description": (
                                "Persona name (matches a .md file under "
                                "~/.hermes/personas/).  Common: "
                                "'researcher', 'coder', 'reviewer', "
                                "'code-analyzer', 'tester', "
                                "'system-architect'.  Run /delegation in "
                                "the CLI to see all ~90 available."
                            ),
                        },
                        "goal": {
                            "type": "string",
                            "description": "What this agent should accomplish.",
                        },
                        "context": {
                            "type": "string",
                            "description": (
                                "Extra context for this specific agent.  "
                                "Appended to the auto-built swarm prelude."
                            ),
                        },
                        "model": {
                            "type": "string",
                            "description": (
                                "Override the per-role model (from "
                                "delegation.model_by_role).  Rarely "
                                "needed — leave unset and let the curated "
                                "defaults apply."
                            ),
                        },
                        "toolsets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Override default toolset list for this "
                                "agent.  Default: inherit from parent."
                            ),
                        },
                        "agent_id": {
                            "type": "string",
                            "description": (
                                "Stable id for this agent within the "
                                "swarm.  Auto-generated if omitted."
                            ),
                        },
                    },
                    "required": ["type", "goal"],
                },
            },
            "topology": {
                "type": "string",
                "enum": list(VALID_TOPOLOGIES),
                "description": (
                    "Spawn ordering (default: parallel).  See tool "
                    "description for per-mode semantics."
                ),
            },
            "title": {
                "type": "string",
                "description": (
                    "Short human-readable name for this swarm — appears "
                    "in swarm_list and logs.  Auto-generated if omitted."
                ),
            },
            "shared_context": {
                "type": "string",
                "description": (
                    "Context string injected into every agent's prelude.  "
                    "Use for facts ALL agents need (e.g. case number, "
                    "customer name, target environment)."
                ),
            },
            "swarm_id": {
                "type": "string",
                "description": (
                    "Override the generated swarm_id.  Useful for "
                    "resuming/joining a known swarm.  Auto-generated if "
                    "omitted."
                ),
            },
        },
        "required": ["agents"],
    },
}


def check_swarm_run_requirements() -> bool:
    """Gate the tool's availability.  Always available — swarm_run uses
    delegate_task internally and inherits its requirements (parent agent
    context).  Hermes-swarm MCP server is optional."""
    return True


# --- Registry ---

registry.register(
    name="swarm_run",
    toolset="delegation",
    schema=SWARM_RUN_SCHEMA,
    handler=lambda args, **kw: swarm_run(
        agents=args.get("agents"),
        topology=args.get("topology"),
        title=args.get("title"),
        shared_context=args.get("shared_context"),
        swarm_id=args.get("swarm_id"),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=check_swarm_run_requirements,
    emoji="🐝",
)

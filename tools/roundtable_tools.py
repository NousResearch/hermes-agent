"""Roundtable discussion tools — structured tool-call surface for multi-agent debates.

These tools enable multiple agents to participate in structured, multi-round
discussions on a topic. A coordinator creates a discussion, participants take
turns speaking, and the system tracks convergence toward consensus.

Available when the ``roundtable`` toolset is enabled in the profile config,
or when an agent is explicitly given the roundtable toolset.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

def _check_roundtable_enabled() -> bool:
    """Roundtable tools are available when the profile has the toolset enabled."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        toolsets = cfg.get("toolsets", [])
        return "roundtable" in toolsets
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _connect():
    """Import + connect lazily so the module imports cleanly in non-roundtable contexts."""
    from hermes_cli import roundtable_db as rdb
    return rdb, rdb.connect()


def _ok(**fields: Any) -> str:
    return json.dumps({"ok": True, **fields})


def _format_history(speeches, participants_map: dict) -> str:
    """Format speech history into a human-readable string for prompts."""
    lines = []
    for s in speeches:
        p_info = participants_map.get(s.participant, {})
        display = p_info.get("display_name", s.participant)
        role = p_info.get("role", "")
        role_str = f"({role})" if role else ""
        ref_str = f" [引用 #{s.reply_to}]" if s.reply_to else ""
        lines.append(
            f"[#{s.id}] Round {s.round} | {display}{role_str}{ref_str}:\n  {s.content}"
        )
    return "\n\n".join(lines) if lines else "(暂无发言)"


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_init(args: dict, **kw) -> str:
    """Create a new roundtable discussion with topic and participants."""
    topic = args.get("topic", "").strip()
    if not topic:
        return tool_error("topic is required")

    participants = args.get("participants")
    if not participants or not isinstance(participants, list):
        return tool_error("participants must be a non-empty array of objects")
    if len(participants) < 2:
        return tool_error("At least 2 participants are required for a discussion")

    context = args.get("context")
    max_rounds = args.get("max_rounds", 5)
    speech_order = args.get("speech_order", "fixed")
    output_path = args.get("output_path")
    created_by = args.get("created_by", "coordinator")

    try:
        max_rounds = int(max_rounds)
    except (TypeError, ValueError):
        return tool_error("max_rounds must be an integer")

    try:
        rdb, conn = _connect()
        try:
            disc = rdb.create_discussion(
                conn,
                topic=topic,
                participants=participants,
                context=context,
                max_rounds=max_rounds,
                speech_order=speech_order,
                created_by=created_by,
                output_path=output_path,
            )
            return _ok(
                discussion_id=disc.id,
                topic=disc.topic,
                participants=[p.get("profile") for p in participants],
                max_rounds=disc.max_rounds,
                speech_order=disc.speech_order,
                status=disc.status,
            )
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(str(e))
    except Exception as e:
        logger.exception("roundtable_init failed")
        return tool_error(f"roundtable_init: {e}")


def _handle_speak(args: dict, **kw) -> str:
    """Record a participant's speech in a discussion."""
    discussion_id = args.get("discussion_id", "").strip()
    if not discussion_id:
        return tool_error("discussion_id is required")

    participant = args.get("participant", "").strip()
    if not participant:
        return tool_error("participant is required")

    content = args.get("content", "").strip()
    if not content:
        return tool_error("content is required")

    reply_to = args.get("reply_to")
    if reply_to is not None:
        try:
            reply_to = int(reply_to)
        except (TypeError, ValueError):
            return tool_error("reply_to must be an integer")

    try:
        rdb, conn = _connect()
        try:
            disc = rdb.get_discussion(conn, discussion_id)
            if not disc:
                return tool_error(f"Discussion {discussion_id} not found")
            if disc.status != "active":
                return tool_error(f"Discussion {discussion_id} is {disc.status}")

            # Validate participant is registered
            active_names = rdb.get_active_participant_names(conn, discussion_id)
            if participant not in active_names:
                return tool_error(
                    f"Participant '{participant}' is not an active member of this discussion. "
                    f"Active: {', '.join(active_names)}"
                )

            speech = rdb.add_speech(
                conn,
                discussion_id=discussion_id,
                participant=participant,
                content=content,
                reply_to=reply_to,
            )

            # Determine round state after this speech
            disc_after = rdb.get_discussion(conn, discussion_id)
            speakers_this_round = conn.execute(
                """SELECT DISTINCT participant FROM speeches
                   WHERE discussion_id = ? AND round = ?""",
                (discussion_id, disc.current_round),
            ).fetchall()
            spoke_names = {r["participant"] for r in speakers_this_round}
            round_complete = all(name in spoke_names for name in active_names)

            # Determine next speaker
            next_speaker = None
            if disc_after and disc_after.status == "active":
                target_round = disc_after.current_round
                speakers_next = conn.execute(
                    """SELECT DISTINCT participant FROM speeches
                       WHERE discussion_id = ? AND round = ?""",
                    (discussion_id, target_round),
                ).fetchall()
                spoke_next = {r["participant"] for r in speakers_next}
                for name in active_names:
                    if name not in spoke_next:
                        next_speaker = name
                        break

            return _ok(
                speech_id=speech.id,
                round=speech.round,
                participant=speech.participant,
                next_speaker=next_speaker,
                round_complete=round_complete,
                discussion_complete=disc_after.status != "active"
                if disc_after
                else False,
            )
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(str(e))
    except Exception as e:
        logger.exception("roundtable_speak failed")
        return tool_error(f"roundtable_speak: {e}")


def _handle_read(args: dict, **kw) -> str:
    """Read discussion history (speeches)."""
    discussion_id = args.get("discussion_id", "").strip()
    if not discussion_id:
        return tool_error("discussion_id is required")

    since_round = args.get("since_round")
    if since_round is not None:
        try:
            since_round = int(since_round)
        except (TypeError, ValueError):
            return tool_error("since_round must be an integer")

    participant_filter = args.get("participant")

    try:
        rdb, conn = _connect()
        try:
            disc = rdb.get_discussion(conn, discussion_id)
            if not disc:
                return tool_error(f"Discussion {discussion_id} not found")

            speeches = rdb.get_speeches(
                conn,
                discussion_id,
                since_round=since_round,
                participant=participant_filter,
            )
            participants = rdb.get_participants(conn, discussion_id)
            p_map = {
                p.participant: {
                    "role": p.role,
                    "display_name": p.display_name,
                    "perspective": p.perspective,
                }
                for p in participants
            }

            return json.dumps({
                "ok": True,
                "discussion_id": disc.id,
                "topic": disc.topic,
                "current_round": disc.current_round,
                "max_rounds": disc.max_rounds,
                "status": disc.status,
                "speeches": [
                    {
                        "id": s.id,
                        "round": s.round,
                        "participant": s.participant,
                        "display_name": p_map.get(s.participant, {}).get("display_name"),
                        "content": s.content,
                        "reply_to": s.reply_to,
                        "created_at": s.created_at,
                    }
                    for s in speeches
                ],
                "speech_count": len(speeches),
                "formatted_history": _format_history(speeches, p_map),
            })
        finally:
            conn.close()
    except Exception as e:
        logger.exception("roundtable_read failed")
        return tool_error(f"roundtable_read: {e}")


def _handle_status(args: dict, **kw) -> str:
    """Get discussion status including convergence metrics."""
    discussion_id = args.get("discussion_id", "").strip()
    if not discussion_id:
        return tool_error("discussion_id is required")

    try:
        rdb, conn = _connect()
        try:
            disc = rdb.get_discussion(conn, discussion_id)
            if not disc:
                return tool_error(f"Discussion {discussion_id} not found")

            participants = rdb.get_participants(conn, discussion_id)
            speech_count = rdb.get_speech_count(conn, discussion_id)
            findings = rdb.get_findings(conn, discussion_id)
            conv_history = rdb.get_convergence_history(conn, discussion_id)

            consensus_pts = [f.content for f in findings if f.type == "consensus"]
            disagreement_pts = [f.content for f in findings if f.type == "disagreement"]
            new_points = [f.content for f in findings if f.type == "new_point"]

            # Determine next speaker
            active_names = rdb.get_active_participant_names(conn, discussion_id)
            next_speaker = None
            if disc.status == "active" and active_names:
                speakers_current = conn.execute(
                    """SELECT DISTINCT participant FROM speeches
                       WHERE discussion_id = ? AND round = ?""",
                    (discussion_id, disc.current_round),
                ).fetchall()
                spoke = {r["participant"] for r in speakers_current}
                for name in active_names:
                    if name not in spoke:
                        next_speaker = name
                        break

            return json.dumps({
                "ok": True,
                "discussion_id": disc.id,
                "topic": disc.topic,
                "status": disc.status,
                "current_round": disc.current_round,
                "max_rounds": disc.max_rounds,
                "speech_order": disc.speech_order,
                "convergence_score": disc.convergence_score,
                "consensus_points": consensus_pts,
                "disagreement_points": disagreement_pts,
                "new_points": new_points,
                "speech_count": speech_count,
                "participant_count": len(participants),
                "next_speaker": next_speaker,
                "convergence_history": [
                    {
                        "round": c.round,
                        "score": c.score,
                        "consensus": c.consensus_count,
                        "disagreement": c.disagreement_count,
                        "new_points": c.new_point_count,
                    }
                    for c in conv_history
                ],
            })
        finally:
            conn.close()
    except Exception as e:
        logger.exception("roundtable_status failed")
        return tool_error(f"roundtable_status: {e}")


def _handle_summarize(args: dict, **kw) -> str:
    """Generate a conclusion document from the discussion.

    Returns structured data that the coordinator can use to write the final
    conclusion document. Does NOT call an LLM — the coordinator agent is
    expected to use this data to produce the Markdown conclusion.
    """
    discussion_id = args.get("discussion_id", "").strip()
    if not discussion_id:
        return tool_error("discussion_id is required")

    try:
        rdb, conn = _connect()
        try:
            disc = rdb.get_discussion(conn, discussion_id)
            if not disc:
                return tool_error(f"Discussion {discussion_id} not found")

            participants = rdb.get_participants(conn, discussion_id)
            speeches = rdb.get_speeches(conn, discussion_id)
            findings = rdb.get_findings(conn, discussion_id)
            conv_history = rdb.get_convergence_history(conn, discussion_id)

            p_map = {
                p.participant: {
                    "role": p.role,
                    "display_name": p.display_name,
                    "perspective": p.perspective,
                }
                for p in participants
            }

            consensus_pts = [f.content for f in findings if f.type == "consensus"]
            disagreement_pts = [f.content for f in findings if f.type == "disagreement"]
            new_points = [f.content for f in findings if f.type == "new_point"]

            # Group speeches by round
            rounds_dict: dict = {}
            for s in speeches:
                rounds_dict.setdefault(s.round, []).append({
                    "id": s.id,
                    "participant": s.participant,
                    "display_name": p_map.get(s.participant, {}).get("display_name"),
                    "role": p_map.get(s.participant, {}).get("role"),
                    "content": s.content,
                    "reply_to": s.reply_to,
                })

            # Calculate overall convergence
            final_score = disc.convergence_score
            if not final_score and conv_history:
                final_score = conv_history[-1].score

            return json.dumps({
                "ok": True,
                "discussion_id": disc.id,
                "topic": disc.topic,
                "context": disc.context,
                "status": disc.status,
                "total_rounds": disc.current_round,
                "max_rounds": disc.max_rounds,
                "final_convergence_score": final_score,
                "participants": [
                    {
                        "profile": p.participant,
                        "display_name": p.display_name,
                        "role": p.role,
                        "perspective": p.perspective,
                    }
                    for p in participants
                ],
                "consensus_points": consensus_pts,
                "disagreement_points": disagreement_pts,
                "new_points": new_points,
                "speech_count": len(speeches),
                "rounds": rounds_dict,
                "convergence_history": [
                    {
                        "round": c.round,
                        "score": c.score,
                        "consensus": c.consensus_count,
                        "disagreement": c.disagreement_count,
                    }
                    for c in conv_history
                ],
                "output_path": disc.output_path,
                "formatted_history": _format_history(speeches, p_map),
            })
        finally:
            conn.close()
    except Exception as e:
        logger.exception("roundtable_summarize failed")
        return tool_error(f"roundtable_summarize: {e}")


def _handle_end(args: dict, **kw) -> str:
    """End a discussion (conclude or cancel)."""
    discussion_id = args.get("discussion_id", "").strip()
    if not discussion_id:
        return tool_error("discussion_id is required")

    force = args.get("force", False)
    conclusion = args.get("conclusion")

    try:
        rdb, conn = _connect()
        try:
            disc = rdb.get_discussion(conn, discussion_id)
            if not disc:
                return tool_error(f"Discussion {discussion_id} not found")
            if disc.status != "active":
                return tool_error(
                    f"Discussion {discussion_id} is already {disc.status}"
                )

            if force:
                ok = rdb.cancel_discussion(conn, discussion_id)
                action = "cancelled"
            else:
                ok = rdb.conclude_discussion(
                    conn, discussion_id, conclusion=conclusion
                )
                action = "concluded"

            return _ok(
                discussion_id=discussion_id,
                action=action,
                success=ok,
            )
        finally:
            conn.close()
    except Exception as e:
        logger.exception("roundtable_end failed")
        return tool_error(f"roundtable_end: {e}")


def _handle_list(args: dict, **kw) -> str:
    """List all discussions with optional status filter."""
    status = args.get("status")
    limit = args.get("limit", 50)
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return tool_error("limit must be an integer")

    try:
        rdb, conn = _connect()
        try:
            discussions = rdb.list_discussions(conn, status=status, limit=limit)
            return json.dumps({
                "ok": True,
                "discussions": [
                    {
                        "id": d.id,
                        "topic": d.topic,
                        "status": d.status,
                        "current_round": d.current_round,
                        "max_rounds": d.max_rounds,
                        "created_by": d.created_by,
                        "created_at": d.created_at,
                        "concluded_at": d.concluded_at,
                        "convergence_score": d.convergence_score,
                    }
                    for d in discussions
                ],
                "count": len(discussions),
                "filter_status": status,
            })
        finally:
            conn.close()
    except Exception as e:
        logger.exception("roundtable_list failed")
        return tool_error(f"roundtable_list: {e}")


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

ROUNDTABLE_INIT_SCHEMA = {
    "name": "roundtable_init",
    "description": (
        "Create a new roundtable discussion with a topic and participants. "
        "Each participant is an agent profile that will take turns speaking. "
        "Returns the discussion_id for subsequent calls."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The discussion topic",
            },
            "participants": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "profile": {
                            "type": "string",
                            "description": "Agent profile name (e.g. 'bingge', 'mafei')",
                        },
                        "role": {
                            "type": "string",
                            "description": "Role description (e.g. '产品总监')",
                        },
                        "perspective": {
                            "type": "string",
                            "description": "Role perspective hint (e.g. '关注用户体验')",
                        },
                        "display_name": {
                            "type": "string",
                            "description": "Display name (e.g. '饼哥')",
                        },
                    },
                    "required": ["profile"],
                },
                "description": "List of participant profiles (min 2)",
            },
            "context": {
                "type": "string",
                "description": "Background context for the discussion",
            },
            "max_rounds": {
                "type": "integer",
                "description": "Maximum discussion rounds (default: 5)",
                "default": 5,
            },
            "speech_order": {
                "type": "string",
                "enum": ["fixed", "random", "priority", "free"],
                "description": "Speech order strategy (default: fixed)",
                "default": "fixed",
            },
            "output_path": {
                "type": "string",
                "description": "Path to save the conclusion document",
            },
            "created_by": {
                "type": "string",
                "description": "Profile name of the discussion creator",
            },
        },
        "required": ["topic", "participants"],
    },
}

ROUNDTABLE_SPEAK_SCHEMA = {
    "name": "roundtable_speak",
    "description": (
        "Record a participant's speech in a roundtable discussion. "
        "Automatically tracks rounds and advances when all participants have spoken."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "discussion_id": {
                "type": "string",
                "description": "Discussion ID (rt_xxxxxxxx)",
            },
            "participant": {
                "type": "string",
                "description": "Profile name of the speaker",
            },
            "content": {
                "type": "string",
                "description": "Speech content (Markdown supported)",
            },
            "reply_to": {
                "type": "integer",
                "description": "Optional: ID of a speech being referenced/replied to",
            },
        },
        "required": ["discussion_id", "participant", "content"],
    },
}

ROUNDTABLE_READ_SCHEMA = {
    "name": "roundtable_read",
    "description": (
        "Read the discussion history — all speeches or filtered by round/participant. "
        "Returns both structured data and a formatted history string."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "discussion_id": {
                "type": "string",
                "description": "Discussion ID (rt_xxxxxxxx)",
            },
            "since_round": {
                "type": "integer",
                "description": "Only return speeches from this round onwards",
            },
            "participant": {
                "type": "string",
                "description": "Only return speeches from this participant",
            },
        },
        "required": ["discussion_id"],
    },
}

ROUNDTABLE_STATUS_SCHEMA = {
    "name": "roundtable_status",
    "description": (
        "Get discussion status including current round, convergence score, "
        "consensus/disagreement points, and next speaker."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "discussion_id": {
                "type": "string",
                "description": "Discussion ID (rt_xxxxxxxx)",
            },
        },
        "required": ["discussion_id"],
    },
}

ROUNDTABLE_SUMMARIZE_SCHEMA = {
    "name": "roundtable_summarize",
    "description": (
        "Generate summary data for a conclusion document. Returns all discussion "
        "data organized by round, with consensus/disagreement points extracted. "
        "Use this to write the final Markdown conclusion."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "discussion_id": {
                "type": "string",
                "description": "Discussion ID (rt_xxxxxxxx)",
            },
        },
        "required": ["discussion_id"],
    },
}

ROUNDTABLE_END_SCHEMA = {
    "name": "roundtable_end",
    "description": (
        "End a roundtable discussion. By default, marks it as concluded. "
        "Use force=true to cancel instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "discussion_id": {
                "type": "string",
                "description": "Discussion ID (rt_xxxxxxxx)",
            },
            "force": {
                "type": "boolean",
                "description": "If true, cancel the discussion instead of concluding",
                "default": False,
            },
            "conclusion": {
                "type": "string",
                "description": "Optional: conclusion text to save",
            },
        },
        "required": ["discussion_id"],
    },
}

ROUNDTABLE_LIST_SCHEMA = {
    "name": "roundtable_list",
    "description": "List roundtable discussions with optional status filter.",
    "parameters": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["active", "concluded", "cancelled"],
                "description": "Filter by status (omit for all)",
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return (default: 50)",
                "default": 50,
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="roundtable_init",
    toolset="roundtable",
    schema=ROUNDTABLE_INIT_SCHEMA,
    handler=_handle_init,
    check_fn=_check_roundtable_enabled,
    emoji="🎯",
)

registry.register(
    name="roundtable_speak",
    toolset="roundtable",
    schema=ROUNDTABLE_SPEAK_SCHEMA,
    handler=_handle_speak,
    check_fn=_check_roundtable_enabled,
    emoji="💬",
)

registry.register(
    name="roundtable_read",
    toolset="roundtable",
    schema=ROUNDTABLE_READ_SCHEMA,
    handler=_handle_read,
    check_fn=_check_roundtable_enabled,
    emoji="📖",
)

registry.register(
    name="roundtable_status",
    toolset="roundtable",
    schema=ROUNDTABLE_STATUS_SCHEMA,
    handler=_handle_status,
    check_fn=_check_roundtable_enabled,
    emoji="📊",
)

registry.register(
    name="roundtable_summarize",
    toolset="roundtable",
    schema=ROUNDTABLE_SUMMARIZE_SCHEMA,
    handler=_handle_summarize,
    check_fn=_check_roundtable_enabled,
    emoji="📝",
)

registry.register(
    name="roundtable_end",
    toolset="roundtable",
    schema=ROUNDTABLE_END_SCHEMA,
    handler=_handle_end,
    check_fn=_check_roundtable_enabled,
    emoji="🏁",
)

registry.register(
    name="roundtable_list",
    toolset="roundtable",
    schema=ROUNDTABLE_LIST_SCHEMA,
    handler=_handle_list,
    check_fn=_check_roundtable_enabled,
    emoji="📋",
)

#!/usr/bin/env python3
"""
Session Search Tool - Long-Term Conversation Recall

Searches past session transcripts in SQLite via FTS5, then summarizes the top
matching sessions using a cheap/fast model (same pattern as web_extract).
Returns focused summaries of past conversations rather than raw transcripts,
keeping the main model's context window clean.

Flow:
  1. FTS5 search finds matching messages ranked by relevance
  2. Groups by session, takes the top N unique sessions (default 3)
  3. Loads each session's conversation, truncates to ~100k chars centered on matches
  4. Sends to Gemini Flash with a focused summarization prompt
  5. Returns per-session summaries with metadata
"""

import asyncio
import concurrent.futures
import json
import os
import logging
from typing import Dict, Any, List, Optional

from openai import AsyncOpenAI, OpenAI

from agent.auxiliary_client import get_text_auxiliary_client

# Resolve the auxiliary client at import time so we have the model slug.
# We build an AsyncOpenAI from the same credentials for async summarization.
_aux_client, _SUMMARIZER_MODEL = get_text_auxiliary_client()
_async_aux_client: AsyncOpenAI | None = None
if _aux_client is not None:
    _async_kwargs = {
        "api_key": _aux_client.api_key,
        "base_url": str(_aux_client.base_url),
    }
    if "openrouter" in str(_aux_client.base_url).lower():
        _async_kwargs["default_headers"] = {
            "HTTP-Referer": "https://github.com/NousResearch/hermes-agent",
            "X-OpenRouter-Title": "Hermes Agent",
            "X-OpenRouter-Categories": "cli-agent",
        }
    _async_aux_client = AsyncOpenAI(**_async_kwargs)
MAX_SESSION_CHARS = 100_000
MAX_SUMMARY_TOKENS = 2000


def _format_timestamp(ts) -> str:
    """Convert a Unix timestamp (float/int) or ISO string to a human-readable date."""
    if ts is None:
        return "unknown"
    try:
        if isinstance(ts, (int, float)):
            from datetime import datetime
            dt = datetime.fromtimestamp(ts)
            return dt.strftime("%B %d, %Y at %I:%M %p")
        if isinstance(ts, str):
            if ts.replace(".", "").replace("-", "").isdigit():
                from datetime import datetime
                dt = datetime.fromtimestamp(float(ts))
                return dt.strftime("%B %d, %Y at %I:%M %p")
            return ts
    except Exception:
        pass
    return str(ts)


def _format_conversation(messages: List[Dict[str, Any]]) -> str:
    """Format session messages into a readable transcript for summarization."""
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content") or ""
        tool_name = msg.get("tool_name")

        if role == "TOOL" and tool_name:
            # Truncate long tool outputs
            if len(content) > 500:
                content = content[:250] + "\n...[truncated]...\n" + content[-250:]
            parts.append(f"[TOOL:{tool_name}]: {content}")
        elif role == "ASSISTANT":
            # Include tool call names if present
            tool_calls = msg.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list):
                tc_names = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        name = tc.get("name") or tc.get("function", {}).get("name", "?")
                        tc_names.append(name)
                if tc_names:
                    parts.append(f"[ASSISTANT]: [Called: {', '.join(tc_names)}]")
                if content:
                    parts.append(f"[ASSISTANT]: {content}")
            else:
                parts.append(f"[ASSISTANT]: {content}")
        else:
            parts.append(f"[{role}]: {content}")

    return "\n\n".join(parts)


def _truncate_around_matches(
    full_text: str, query: str, max_chars: int = MAX_SESSION_CHARS
) -> str:
    """
    Truncate a conversation transcript to max_chars, centered around
    where the query terms appear. Keeps content near matches, trims the edges.
    """
    if len(full_text) <= max_chars:
        return full_text

    # Find the first occurrence of any query term
    query_terms = query.lower().split()
    text_lower = full_text.lower()
    first_match = len(full_text)
    for term in query_terms:
        pos = text_lower.find(term)
        if pos != -1 and pos < first_match:
            first_match = pos

    if first_match == len(full_text):
        # No match found, take from the start
        first_match = 0

    # Center the window around the first match
    half = max_chars // 2
    start = max(0, first_match - half)
    end = min(len(full_text), start + max_chars)
    if end - start < max_chars:
        start = max(0, end - max_chars)

    truncated = full_text[start:end]
    prefix = "...[earlier conversation truncated]...\n\n" if start > 0 else ""
    suffix = "\n\n...[later conversation truncated]..." if end < len(full_text) else ""
    return prefix + truncated + suffix


async def _summarize_session(
    conversation_text: str, query: str, session_meta: Dict[str, Any]
) -> Optional[str]:
    """Summarize a single session conversation focused on the search query."""
    system_prompt = (
        "You are reviewing a past conversation transcript to help recall what happened. "
        "Summarize the conversation with a focus on the search topic. Include:\n"
        "1. What the user asked about or wanted to accomplish\n"
        "2. What actions were taken and what the outcomes were\n"
        "3. Key decisions, solutions found, or conclusions reached\n"
        "4. Any specific commands, files, URLs, or technical details that were important\n"
        "5. Anything left unresolved or notable\n\n"
        "Be thorough but concise. Preserve specific details (commands, paths, error messages) "
        "that would be useful to recall. Write in past tense as a factual recap."
    )

    source = session_meta.get("source", "unknown")
    started = _format_timestamp(session_meta.get("started_at"))

    user_prompt = (
        f"Search topic: {query}\n"
        f"Session source: {source}\n"
        f"Session date: {started}\n\n"
        f"CONVERSATION TRANSCRIPT:\n{conversation_text}\n\n"
        f"Summarize this conversation with focus on: {query}"
    )

    if _async_aux_client is None or _SUMMARIZER_MODEL is None:
        logging.warning("No auxiliary model available for session summarization")
        return None

    max_retries = 3
    for attempt in range(max_retries):
        try:
            from agent.auxiliary_client import get_auxiliary_extra_body, auxiliary_max_tokens_param
            _extra = get_auxiliary_extra_body()
            response = await _async_aux_client.chat.completions.create(
                model=_SUMMARIZER_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **({} if not _extra else {"extra_body": _extra}),
                temperature=0.1,
                **auxiliary_max_tokens_param(MAX_SUMMARY_TOKENS),
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))
            else:
                logging.warning(f"Session summarization failed after {max_retries} attempts: {e}")
                return None


def session_search(
    query: str,
    role_filter: str = None,
    limit: int = 3,
    db=None,
) -> str:
    """
    Search past sessions and return focused summaries of matching conversations.

    Uses FTS5 to find matches, then summarizes the top sessions with Gemini Flash.
    """
    if db is None:
        return json.dumps({"success": False, "error": "Session database not available."}, ensure_ascii=False)

    if not query or not query.strip():
        return json.dumps({"success": False, "error": "Query cannot be empty."}, ensure_ascii=False)

    query = query.strip()
    limit = min(limit, 5)  # Cap at 5 sessions to avoid excessive LLM calls

    try:
        # Parse role filter
        role_list = None
        if role_filter and role_filter.strip():
            role_list = [r.strip() for r in role_filter.split(",") if r.strip()]

        # FTS5 search -- get matches ranked by relevance
        raw_results = db.search_messages(
            query=query,
            role_filter=role_list,
            limit=50,  # Get more matches to find unique sessions
            offset=0,
        )

        if not raw_results:
            return json.dumps({
                "success": True,
                "query": query,
                "results": [],
                "count": 0,
                "message": "No matching sessions found.",
            }, ensure_ascii=False)

        # Resolve child sessions to their parent — delegation stores detailed
        # content in child sessions, but the user's conversation is the parent.
        def _resolve_to_parent(session_id):
            visited = set()
            sid = session_id
            while sid and sid not in visited:
                visited.add(sid)
                session = db.get_session(sid)
                if not session:
                    break
                parent = session.get("parent_session_id")
                if parent:
                    sid = parent
                else:
                    break
            return sid

        # Group by resolved (parent) session_id, dedup
        seen_sessions = {}
        for result in raw_results:
            raw_sid = result["session_id"]
            resolved_sid = _resolve_to_parent(raw_sid)
            if resolved_sid not in seen_sessions:
                result = dict(result)
                result["session_id"] = resolved_sid
                seen_sessions[resolved_sid] = result
            if len(seen_sessions) >= limit:
                break

        # Prepare all sessions for parallel summarization
        tasks = []
        for session_id, match_info in seen_sessions.items():
            try:
                messages = db.get_messages_as_conversation(session_id)
                if not messages:
                    continue
                session_meta = db.get_session(session_id) or {}
                conversation_text = _format_conversation(messages)
                conversation_text = _truncate_around_matches(conversation_text, query)
                tasks.append((session_id, match_info, conversation_text, session_meta))
            except Exception as e:
                logging.warning(f"Failed to prepare session {session_id}: {e}")

        # Summarize all sessions in parallel
        async def _summarize_all():
            coros = [
                _summarize_session(text, query, meta)
                for _, _, text, meta in tasks
            ]
            return await asyncio.gather(*coros, return_exceptions=True)

        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                results = pool.submit(lambda: asyncio.run(_summarize_all())).result(timeout=60)
        except RuntimeError:
            results = asyncio.run(_summarize_all())

        summaries = []
        for (session_id, match_info, _, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logging.warning(f"Failed to summarize session {session_id}: {result}")
                continue
            if result:
                summaries.append({
                    "session_id": session_id,
                    "when": _format_timestamp(match_info.get("session_started")),
                    "source": match_info.get("source", "unknown"),
                    "model": match_info.get("model"),
                    "summary": result,
                })

        return json.dumps({
            "success": True,
            "query": query,
            "results": summaries,
            "count": len(summaries),
            "sessions_searched": len(seen_sessions),
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"success": False, "error": f"Search failed: {str(e)}"}, ensure_ascii=False)


def check_session_search_requirements() -> bool:
    """Requires SQLite state database and an auxiliary text model."""
    if _async_aux_client is None:
        return False
    try:
        from hermes_state import DEFAULT_DB_PATH
        return DEFAULT_DB_PATH.parent.exists()
    except ImportError:
        return False


SESSION_SEARCH_SCHEMA = {
    "name": "session_search",
    "description": (
        "Search your long-term memory of past conversations. This is your recall -- "
        "every past session is searchable, and this tool summarizes what happened.\n\n"
        "USE THIS PROACTIVELY when:\n"
        "- The user says 'we did this before', 'remember when', 'last time', 'as I mentioned'\n"
        "- The user asks about a topic you worked on before but don't have in current context\n"
        "- The user references a project, person, or concept that seems familiar but isn't in memory\n"
        "- You want to check if you've solved a similar problem before\n"
        "- The user asks 'what did we do about X?' or 'how did we fix Y?'\n\n"
        "Don't hesitate to search -- it's fast and cheap. Better to search and confirm "
        "than to guess or ask the user to repeat themselves.\n\n"
        "Search syntax: keywords joined with OR for broad recall (elevenlabs OR baseten OR funding), "
        "phrases for exact match (\"docker networking\"), boolean (python NOT java), prefix (deploy*). "
        "IMPORTANT: Use OR between keywords for best results — FTS5 defaults to AND which misses "
        "sessions that only mention some terms. If a broad OR query returns nothing, try individual "
        "keyword searches in parallel. Returns summaries of the top matching sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query — keywords, phrases, or boolean expressions to find in past sessions.",
            },
            "role_filter": {
                "type": "string",
                "description": "Optional: only search messages from specific roles (comma-separated). E.g. 'user,assistant' to skip tool outputs.",
            },
            "limit": {
                "type": "integer",
                "description": "Max sessions to summarize (default: 3, max: 5).",
                "default": 3,
            },
        },
        "required": ["query"],
    },
}


# --- Registry ---
from tools.registry import registry

registry.register(
    name="session_search",
    toolset="session_search",
    schema=SESSION_SEARCH_SCHEMA,
    handler=lambda args, **kw: session_search(
        query=args.get("query", ""),
        role_filter=args.get("role_filter"),
        limit=args.get("limit", 3),
        db=kw.get("db")),
    check_fn=check_session_search_requirements,
)

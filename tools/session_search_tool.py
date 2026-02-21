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

from tools.openrouter_client import get_async_client as _get_client

SUMMARIZER_MODEL = "google/gemini-3-flash-preview"
MAX_SESSION_CHARS = 100_000
MAX_SUMMARY_TOKENS = 2000


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
    started = session_meta.get("started_at", "unknown")

    user_prompt = (
        f"Search topic: {query}\n"
        f"Session source: {source}\n"
        f"Session started: {started}\n\n"
        f"CONVERSATION TRANSCRIPT:\n{conversation_text}\n\n"
        f"Summarize this conversation with focus on: {query}"
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await _get_client().chat.completions.create(
                model=SUMMARIZER_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=MAX_SUMMARY_TOKENS,
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

        # Group by session_id, keep order (highest ranked first)
        seen_sessions = {}
        for result in raw_results:
            sid = result["session_id"]
            if sid not in seen_sessions:
                seen_sessions[sid] = result
            if len(seen_sessions) >= limit:
                break

        # Summarize each matching session
        summaries = []
        for session_id, match_info in seen_sessions.items():
            try:
                # Load full conversation
                messages = db.get_messages_as_conversation(session_id)
                if not messages:
                    continue

                # Get session metadata
                session_meta = db.get_session(session_id) or {}

                # Format and truncate
                conversation_text = _format_conversation(messages)
                conversation_text = _truncate_around_matches(conversation_text, query)

                # Summarize with Gemini Flash (handle both async and sync contexts)
                coro = _summarize_session(conversation_text, query, session_meta)
                try:
                    asyncio.get_running_loop()
                    # Already in an async context (gateway) -- run in a thread
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        summary = pool.submit(lambda: asyncio.run(coro)).result(timeout=30)
                except RuntimeError:
                    # No running loop (normal CLI) -- use asyncio.run directly
                    summary = asyncio.run(coro)

                if summary:
                    summaries.append({
                        "session_id": session_id,
                        "source": match_info.get("source", "unknown"),
                        "model": match_info.get("model"),
                        "session_started": match_info.get("session_started"),
                        "summary": summary,
                    })

            except Exception as e:
                logging.warning(f"Failed to summarize session {session_id}: {e}")
                continue

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
    """Requires SQLite state database and OpenRouter API key."""
    if not os.getenv("OPENROUTER_API_KEY"):
        return False
    try:
        from hermes_state import DEFAULT_DB_PATH
        return DEFAULT_DB_PATH.parent.exists()
    except ImportError:
        return False


SESSION_SEARCH_SCHEMA = {
    "name": "session_search",
    "description": (
        "Search and recall past conversations. Finds matching sessions using "
        "full-text search, then provides a focused summary of each matching "
        "conversation.\n\n"
        "Use this when you need to recall:\n"
        "- A solution or approach from a previous session\n"
        "- Something the user said or asked about before\n"
        "- A command, file path, or technical detail from past work\n"
        "- The outcome of a previous task\n\n"
        "Supports search syntax:\n"
        "  Keywords: docker deployment\n"
        "  Phrases: '\"exact phrase\"'\n"
        "  Boolean: docker OR kubernetes, python NOT java\n"
        "  Prefix: deploy*\n\n"
        "Returns summaries (not raw transcripts) of the top matching sessions, "
        "focused on your search topic. Max 3 sessions per search."
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

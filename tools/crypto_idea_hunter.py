# tools/crypto_idea_hunter.py
#
# Twitter/X Thread → Crypto Product Idea Hunter
# -----------------------------------------------
# Scans viral crypto threads and user complaints to surface
# unsolved problems and turn them into product ideas.
#
# Usage:
#   crypto_idea_hunter(topic="defi", limit=5)
#   crypto_idea_hunter(topic="nft", limit=10)
#   crypto_idea_hunter(topic="general")

import json
import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Search queries per crypto sub-category
# ---------------------------------------------------------------------------

CRYPTO_SEARCH_QUERIES = {
    "defi": [
        "twitter crypto defi pain points complaints 2024",
        "defi users frustrated problem still unsolved",
        "defi ux problem thread site:twitter.com OR site:x.com",
    ],
    "nft": [
        "nft creators problems complaints thread 2024",
        "nft marketplace issues unsolved twitter",
        "nft royalty problem thread creators angry",
    ],
    "trading": [
        "crypto traders biggest pain points twitter thread",
        "crypto exchange problems complaints 2024",
        "on-chain trading ux frustration thread",
    ],
    "wallet": [
        "crypto wallet ux problems twitter complaints",
        "seed phrase problem alternatives thread 2024",
        "web3 wallet onboarding frustration",
    ],
    "general": [
        "crypto biggest unsolved problems thread 2024",
        "web3 ux pain points viral twitter thread",
        "crypto users complain still not fixed",
    ],
}

# ---------------------------------------------------------------------------
# Complaint-to-idea templates
# ---------------------------------------------------------------------------

IDEA_TEMPLATES = {
    "ux_problem":   "💡 {problem} → Simplified {domain} interface",
    "security":     "💡 {problem} → Automated security layer for {domain}",
    "cost":         "💡 {problem} → Low-cost {domain} alternative",
    "transparency": "💡 {problem} → {domain} transparency & monitoring tool",
    "education":    "💡 {problem} → Guided onboarding tool for {domain}",
    "automation":   "💡 {problem} → {domain} automation & bot tool",
    "aggregation":  "💡 {problem} → Aggregator for scattered {domain} data",
    "default":      "💡 {problem} → {domain} solution tool",
}

# ---------------------------------------------------------------------------
# Helper: detect problem type from complaint text
# ---------------------------------------------------------------------------

PROBLEM_KEYWORDS = {
    "ux_problem":   ["confusing", "hard", "complex", "ux", "ui", "onboard", "difficult", "impossible"],
    "security":     ["hack", "scam", "phishing", "rug", "exploit", "security", "stolen"],
    "cost":         ["gas", "fee", "expensive", "costly", "cheap"],
    "transparency": ["hidden", "transparent", "audit", "track", "monitor"],
    "education":    ["how to", "tutorial", "learn", "newbie", "beginner"],
    "automation":   ["manual", "repeat", "automate", "bot"],
    "aggregation":  ["scattered", "aggregate", "all in one"],
}

def _detect_problem_type(text: str) -> str:
    text_lower = text.lower()
    for ptype, keywords in PROBLEM_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return ptype
    return "default"

# ---------------------------------------------------------------------------
# Helper: extract complaint sentences from web search results
# ---------------------------------------------------------------------------

def _extract_complaints(search_results: list[dict], topic: str) -> list[str]:
    """
    Filters snippets that contain complaint signals.
    """
    complaint_signals = [
        "still", "why", "frustrat", "broken", "annoying", "problem",
        "issue", "hate", "worst", "pain", "difficult", "impossible",
        "nobody", "always fails", "never works",
    ]

    complaints = []
    for result in search_results:
        snippet = result.get("snippet", "") or result.get("description", "")
        if not snippet:
            continue
        snippet_lower = snippet.lower()
        if any(sig in snippet_lower for sig in complaint_signals):
            cleaned = re.sub(r'\s+', ' ', snippet).strip()
            if len(cleaned) > 200:
                cleaned = cleaned[:200] + "..."
            complaints.append(cleaned)

    return complaints[:15]  # max 15 complaints

# ---------------------------------------------------------------------------
# Helper: generate ideas from complaints
# ---------------------------------------------------------------------------

def _generate_ideas(complaints: list[str], topic: str) -> list[dict]:
    ideas = []
    seen_types: set[str] = set()

    for complaint in complaints:
        ptype = _detect_problem_type(complaint)

        # One idea per problem type to keep results diverse
        if ptype in seen_types:
            continue

        template = IDEA_TEMPLATES.get(ptype, IDEA_TEMPLATES["default"])
        domain = topic.upper() if len(topic) <= 5 else topic.capitalize()

        short_problem = complaint[:80].rstrip() + ("..." if len(complaint) > 80 else "")
        idea_text = template.format(problem=short_problem, domain=domain)

        ideas.append({
            "idea":        idea_text,
            "type":        ptype,
            "source_hint": complaint[:120],
        })

        seen_types.add(ptype)

    return ideas

# ---------------------------------------------------------------------------
# Main tool handler
# ---------------------------------------------------------------------------

def crypto_idea_hunter(
    topic: str = "general",
    limit: int = 5,
    task_id: str = None,
) -> str:
    """
    Scans viral Twitter/X crypto threads and user complaints to surface
    unsolved problems and generate product ideas.

    Args:
        topic:   Crypto sub-category — "defi", "nft", "trading", "wallet", "general"
        limit:   Maximum number of ideas to return (default: 5)
        task_id: Hermes session isolation (optional)

    Returns:
        JSON string — list of product ideas
    """
    topic = topic.lower().strip()

    # Unknown topic → fall back to general
    if topic not in CRYPTO_SEARCH_QUERIES:
        logger.debug("Unknown topic '%s', falling back to 'general'", topic)
        topic = "general"

    queries = CRYPTO_SEARCH_QUERIES[topic]
    all_complaints: list[str] = []

    try:
        from tools.registry import registry

        web_search = registry.get_handler("web_search")
        if web_search is None:
            return json.dumps(
                {"error": "web_search tool not found. Is the web toolset active?"},
                ensure_ascii=False,
            )

        for query in queries:
            try:
                raw = web_search(query=query)
                data = json.loads(raw) if isinstance(raw, str) else raw
                results = (
                    data.get("results")
                    or data.get("web", {}).get("results", [])
                    or []
                )
                complaints = _extract_complaints(results, topic)
                all_complaints.extend(complaints)

            except Exception as e:
                # If a single query fails (network, rate limit, etc.),
                # log and continue — do not crash the tool.
                logger.debug("Search query failed ('%s'): %s", query, e)
                continue

    except ImportError:
        return json.dumps(
            {"error": "Could not import tools.registry."},
            ensure_ascii=False,
        )

    # Deduplicate
    seen: set[str] = set()
    unique_complaints: list[str] = []
    for c in all_complaints:
        key = c[:60]
        if key not in seen:
            seen.add(key)
            unique_complaints.append(c)

    if not unique_complaints:
        return json.dumps(
            {
                "topic":   topic,
                "ideas":   [],
                "message": "Not enough complaint data found. Try a different topic.",
            },
            ensure_ascii=False,
            indent=2,
        )

    ideas = _generate_ideas(unique_complaints, topic)
    ideas = ideas[:limit]

    return json.dumps(
        {
            "topic":            topic,
            "complaints_found": len(unique_complaints),
            "ideas":            ideas,
        },
        ensure_ascii=False,
        indent=2,
    )


# ---------------------------------------------------------------------------
# Registry registration
# ---------------------------------------------------------------------------

CRYPTO_IDEA_SCHEMA = {
    "name": "crypto_idea_hunter",
    "description": (
        "Scans viral Twitter/X crypto threads and user complaints to surface "
        "unsolved problems and generate product ideas. "
        "Sub-categories: defi, nft, trading, wallet, general."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Crypto sub-category: 'defi', 'nft', 'trading', 'wallet', 'general'",
                "default": "general",
                "enum": ["defi", "nft", "trading", "wallet", "general"],
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of ideas to return",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": [],
    },
}

try:
    from tools.registry import registry

    registry.register(
        name="crypto_idea_hunter",
        handler=crypto_idea_hunter,
        schema=CRYPTO_IDEA_SCHEMA,
        toolset="web",       # tied to web toolset — available when web is active
        check_fn=lambda: True,
    )
except ImportError:
    pass  # Registry not available — tool can still be imported standalone

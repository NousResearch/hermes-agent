#!/usr/bin/env python3
"""Research lane launch orchestration.

Creates or refreshes the local research lane scaffolding, optionally generates a
standardized domain playbook from the canonical template, records/stubs the
Google Drive folder step, and runs the first 3-stage research pipeline with a
quiet messaging contract.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.registry import registry

logger = logging.getLogger(__name__)

DEFAULT_OBSIDIAN_ROOT = Path(os.path.expanduser("~/Obsidian/Jarvis-Operations"))
DEFAULT_PLAYBOOKS_DIR = DEFAULT_OBSIDIAN_ROOT / "🧠 Brain" / "playbooks"
DEFAULT_TEMPLATES_DIR = DEFAULT_PLAYBOOKS_DIR / "templates"
DEFAULT_RESEARCH_ROOT = DEFAULT_OBSIDIAN_ROOT / "🏗️ Projects" / "rww" / "research"

STAGE_SEQUENCE = [
    "stage-1-knowledge",
    "stage-2-web-research",
    "stage-3-writeup",
]

KNOWN_PLAYBOOK_ALIASES = {
    "seo": "seo-research",
    "seo-research": "seo-research",
    "gbp": "gbp-research",
    "gbp-research": "gbp-research",
    "geo": "geo-research",
    "geo-research": "geo-research",
    "website-craft": "website-research",
    "website-research": "website-research",
    "reviews": "reviews-reputation-research",
    "reviews-reputation": "reviews-reputation-research",
    "reputation": "reviews-reputation-research",
    "paid-advertising": "paid-advertising-research",
    "social-media": "social-media-research",
    "agency-business-model": "agency-business-model-research",
    "client-market-analysis": "client-market-analysis-research",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", (value or "").strip().lower())
    return text.strip("-") or "research-topic"


def _obsidian_root() -> Path:
    override = os.getenv("HERMES_RESEARCH_OBSIDIAN_ROOT", "").strip()
    return Path(override).expanduser() if override else DEFAULT_OBSIDIAN_ROOT


def _playbooks_dir() -> Path:
    override = os.getenv("HERMES_RESEARCH_PLAYBOOKS_DIR", "").strip()
    return Path(override).expanduser() if override else (_obsidian_root() / "🧠 Brain" / "playbooks")


def _templates_dir() -> Path:
    override = os.getenv("HERMES_RESEARCH_TEMPLATES_DIR", "").strip()
    return Path(override).expanduser() if override else (_playbooks_dir() / "templates")


def _research_root() -> Path:
    override = os.getenv("HERMES_RESEARCH_ROOT", "").strip()
    return Path(override).expanduser() if override else (_obsidian_root() / "🏗️ Projects" / "rww" / "research")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to parse JSON file %s", path)
        return None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _ensure_generic_index_standard_template() -> Path:
    path = _templates_dir() / "research-index-standard-template.md"
    if path.exists():
        return path
    content = f"""# Template: Research Index Standard

**Asset Class:** Template
**Last Updated:** {_today_str()}
**Canonical Path:** `{path}`
**Owner:** Atlas + Dax
**Type:** Canonical template — base standard for domain research indexes

---

## Purpose

This is the standard template for a domain research index. Each lane-specific index template should be generated from this structure so Stage 3 can create/update one consistent index for every domain.

## Rules
- Keep newest entries first within each section.
- Preserve history; mark older docs `superseded` or `stale` instead of deleting them.
- Every entry should point to both the durable markdown artifact and the Google Doc link record.
- The index is the durable source of truth for freshness and scope.

---

# {{domain_name}} Research Index
**Last Updated:** {{date}}
**Owner:** Atlas + Dax

## 1. Evergreen Foundation Docs

### {{domain_name}} Fundamentals
- **Doc Type:** foundation
- **Status:** evergreen
- **Last Updated:** {{date}}
- **Scope:** Broad, durable domain knowledge to preserve long-term
- **Google Doc:** {{url_or_tbd}}
- **Markdown Artifact:** `{{foundation_markdown_path}}`
- **Doc Link Record:** `{{foundation_doc_link_path}}`
- **Notes:** Update only when baseline understanding materially changes.

## 2. Central Index Records

### {{domain_name}} Research Index
- **Doc Type:** index
- **Status:** evergreen
- **Last Updated:** {{date}}
- **Google Doc:** {{url_or_tbd}}
- **Markdown Artifact:** `{{index_markdown_path}}`
- **Doc Link Record:** `{{index_doc_link_path}}`
- **Notes:** This file tracks all domain docs and their freshness.

## 3. Topic-Specific Docs

### {{date}} — {{topic_title}}
- **Doc Type:** topic | audit | strategy | question | update
- **Status:** current snapshot
- **Scope:** {{exact_scope}}
- **Freshness Window:** {{freshness_window}}
- **Google Doc:** {{url}}
- **Markdown Artifact:** `{{topic_markdown_pattern}}`
- **Doc Link Record:** `{{topic_doc_link_pattern}}`
- **Related Docs:** {{related_docs}}
- **Supersedes:** {{supersedes}}
- **Why This Exists:** {{why_this_exists}}

## 4. Refresh Queue
- {{refresh_item}}

## 5. Superseded / Historical Docs
- {{historical_item}}
"""
    _write_text(path, content)
    return path


def _render_lane_index_template(lane_title: str, lane_slug: str) -> str:
    return _read_text(_ensure_generic_index_standard_template()).format(
        domain_name=lane_title,
        date=_today_str(),
        url_or_tbd="{url-or-tbd}",
        foundation_markdown_path=f"🏗️ Projects/rww/research/{lane_slug}/{lane_slug}-fundamentals-guide.md",
        foundation_doc_link_path=f"🏗️ Projects/rww/research/{lane_slug}/doc-link.md",
        index_markdown_path=f"🏗️ Projects/rww/research/{lane_slug}/{lane_slug}-research-index.md",
        index_doc_link_path=f"🏗️ Projects/rww/research/{lane_slug}/index-doc-link.md",
        topic_title="Topic Title",
        exact_scope="{exact question, tactic, market, or subtopic}",
        freshness_window="{e.g. 90 days unless confirmed still current}",
        url="{url}",
        topic_markdown_pattern=f"🏗️ Projects/rww/research/{lane_slug}/topics/{{topic-slug}}/{{YYYY-MM-DD}}-{{topic-slug}}.md",
        topic_doc_link_pattern=f"🏗️ Projects/rww/research/{lane_slug}/topics/{{topic-slug}}/doc-link.md",
        related_docs="{foundation doc / related topic docs / none}",
        supersedes="{older doc or none}",
        why_this_exists="{why this deserved its own doc instead of a foundation update}",
        refresh_item="{doc name} — {why it needs refresh}",
        historical_item="{doc name} — superseded by {newer doc path/link}",
    )


def _standard_playbook_template_path() -> Path:
    return _templates_dir() / "research-playbook-standard-template.md"


def _extract_copyable_playbook_template() -> Optional[str]:
    template_text = _read_text(_standard_playbook_template_path())
    if not template_text:
        return None
    marker = "## Copyable Template"
    if marker not in template_text:
        return None
    copyable = template_text.split(marker, 1)[1].strip()
    return copyable or None


def _render_playbook_from_standard_template(
    lane_title: str,
    lane_slug: str,
    playbook_path: Path,
    index_template_path: Path,
) -> Optional[str]:
    template = _extract_copyable_playbook_template()
    if not template:
        return None

    # Research root for artifact paths
    research_root = str(_research_root()).replace(str(_obsidian_root()) + "/", "")

    replacements = {
        "{Domain Name}": f"{lane_title} Research",
        "{YYYY-MM-DD}": _today_str(),
        "{absolute path to this playbook}": str(playbook_path),
        "{profile-name or omit if not used}": "Not used in this domain",
        "{mode-1}": f"{lane_slug}-education",
        "{mode-2}": f"{lane_slug}-audit",
        "{mode-3}": f"{lane_slug}-market-scan",
        "{one-sentence statement of what this research lane exists to accomplish}": f"Build durable, general-purpose {lane_title.lower()} knowledge progressing from beginner to mastery level, so Atlas has authoritative domain expertise ready to apply in any context.",
        "{trigger 1}": f"Dax asks to learn, audit, compare, or evaluate anything in {lane_title.lower()}",
        "{trigger 2}": "Dax wants a repeatable research lane instead of a one-off answer",
        "{trigger 3}": f"A new scoped topic belongs under the {lane_title.lower()} lane",
        "{trigger 4}": "A recurring research workflow needs standard setup and persistence",
        "{phrase 1}": lane_title,
        "{phrase 2}": lane_slug.replace("-", " "),
        "{phrase 3}": "market scan",
        "{Business Name}": "Dax's system",
        "{what the business is}": "Atlas is building a general knowledge library. Research should produce domain expertise that is universally applicable, not tied to any single business.",
        "{what the business is really selling}": "Research must produce clear, accurate, general domain knowledge that Atlas can later apply to any business context.",
        "{primary}": "General domain knowledge applicable to any market",
        "{secondary}": "Specific applications will be determined when the knowledge is used",
        "{service or offer line 1}": "General domain expertise and research",
        "{service or offer line 2}": "Knowledge that progresses from beginner to mastery",
        "{relevant state of the business}": "This is a generated baseline playbook. Stage 1 should treat older lane docs as context and keep this lane clean as it matures.",
        "{constraint 1}": "Produce general domain knowledge, not business-specific content",
        "{constraint 2}": "Prefer durable doc structures over one-off note sprawl",
        "{constraint 3}": "Scoped / time-sensitive work should become a topic doc, not bloat the foundation doc",
        "{playbook-file}": playbook_path.stem,
        "{absolute paths}": f"{research_root}/{lane_slug}/ and 🧠 Brain/playbooks/research-pipeline.md",
        "{Domain}": lane_title,
        "{domain}": lane_title.lower(),
        "{absolute or vault-relative artifact path}": f"{research_root}/{lane_slug}/{lane_slug}-fundamentals-guide.md",
        "{doc-link path}": f"{research_root}/{lane_slug}/doc-link.md",
        "{index path}": f"{research_root}/{lane_slug}/{lane_slug}-research-index.md",
        "{index doc-link path}": f"{research_root}/{lane_slug}/index-doc-link.md",
        "{path pattern}": f"{research_root}/{lane_slug}/topics/{{topic-slug}}/{{YYYY-MM-DD}}-{{topic-slug}}.md",
        "{doc-link pattern}": f"{research_root}/{lane_slug}/topics/{{topic-slug}}/doc-link.md",
        "{market path pattern}": "Not used in this domain",
        "{audit path pattern}": f"{research_root}/{lane_slug}/topics/{{topic-slug}}/{{YYYY-MM-DD}}-{{topic-slug}}.md",
        "{experiment path pattern}": "Not used in this domain",
        "{freshness window}": "90 days",
        "{domain}-research-index-template.md": index_template_path.name,
        "{mode-slug}": f"{lane_slug}-education",
        "{what this mode is for}": f"Build general {lane_title.lower()} knowledge from the ground up — concepts, terminology, how it works, and why it matters.",
        "{step 1}": "Run Stage 1 document classification.",
        "{step 2}": "Research current domain fundamentals.",
        "{step 3}": "Write a clear, readable summary for Dax.",
        "{step 4}": "Persist findings to the correct foundation or topic artifact.",
        "{step 5}": "Update the lane index and doc-link records.",
        "{artifact(s) and where they go}": "Foundation doc or new topic doc, plus index/doc-link persistence in the lane folder.",
        "{mode-slug-2}": f"{lane_slug}-audit",
        "{max turns or budget}": "Max 20 turns per stage unless the pipeline playbook says otherwise",
        "{hard limit}": "30 minutes per run (hard limit)",
        "{Primary Artifact Name}": "Google Doc Link Record",
        "{file-name}": "doc-link",
        "{Artifact Title}": "<Doc Title>",
        "{date}": "<date>",
        "{exact scope}": "<exact scope this doc covers>",
        "{high-level summary bullets}": "<high-level summary bullets>",
        "{Telegram topic / destination}": f"{lane_title} Research topic in Library group chat (auto-created by lane launch)",
        "{allowed action 1}": "Run any research mode within the pipeline",
        "{allowed action 2}": "Create or update Obsidian artifacts within the lane folder",
        "{allowed action 3}": "Update the central domain index",
        "{approval-needed action 1}": "Deleting or significantly restructuring existing docs",
        "{approval-needed action 2}": "Running research outside the defined domain scope",
        "{approval-needed action 3}": "Sharing research externally beyond Telegram/Obsidian/Drive",
        "{forbidden action 1}": "Inventing new storage patterns outside the doc architecture",
        "{forbidden action 2}": "Combining unrelated topics into a single doc",
        "{forbidden action 3}": "Skipping Stage 1 document classification",
        "{what broad, durable knowledge belongs here}": f"Broad, evergreen {lane_title.lower()} knowledge that changes baseline understanding",
        "{path}": f"{research_root}/{lane_slug}/",
        "{path pattern}": f"{research_root}/{lane_slug}/topics/{{topic-slug}}/",
    }

    rendered = template
    for old, new in replacements.items():
        rendered = rendered.replace(old, new)
    rendered = rendered.replace("{Domain} Research Index", f"{lane_title} Research Index")
    rendered = rendered.replace("{Domain} Fundamentals", f"{lane_title} Fundamentals")
    rendered = rendered.replace("{Domain} Research", f"{lane_title} Research")
    # Strip any remaining "for Ridley Web Works" suffixes
    rendered = rendered.replace(" for Ridley Web Works", "")
    return rendered


def _render_generated_playbook(lane_title: str, lane_slug: str, playbook_path: Path, index_template_path: Path) -> str:
    rendered_from_template = _render_playbook_from_standard_template(
        lane_title=lane_title,
        lane_slug=lane_slug,
        playbook_path=playbook_path,
        index_template_path=index_template_path,
    )
    if rendered_from_template:
        return rendered_from_template
    return f"""# Playbook: {lane_title} Research

**Asset Class:** Playbook
**Last Updated:** {_today_str()}
**Canonical Path:** `{playbook_path}`
**Owner:** Atlas + Dax
**Type:** Generated from `🧠 Brain/playbooks/templates/research-playbook-standard-template.md`

---

## IMPORTANT: Research Execution Pattern
**All research uses the 3-stage pipeline.** See `🧠 Brain/playbooks/research-pipeline.md` for the execution pattern. This playbook provides the business context, output formats, domain rules, and persistence logic — the pipeline playbook provides the stage execution pattern.

## 1. Identity
- **Name:** {lane_title} Research
- **Type:** on-demand, executed via 3-stage research pipeline
- **Modes:** {lane_slug}-education, {lane_slug}-audit, {lane_slug}-market-scan
- **Standing mission:** Build durable, general-purpose {lane_title.lower()} knowledge progressing from beginner to mastery level, so Atlas has authoritative domain expertise ready to apply in any context.

## 2. When to Use This Playbook
Atlas should load this playbook into a subagent's `context` when:
- Dax asks to learn, audit, compare, or evaluate anything in {lane_title.lower()}
- Dax wants a repeatable research lane instead of a one-off answer
- A new scoped topic belongs under the {lane_title.lower()} lane
- Any task involving: `{lane_title}`, `{lane_slug.replace('-', ' ')}`, `research`, `audit`, `market scan`

## 3. Research Philosophy

**Purpose:** Build general domain knowledge that Atlas can master and later apply to any business context.

**Knowledge progression:** beginner → intermediate → adept → expert → mastery. Early runs should focus on foundational concepts. Later runs deepen into advanced topics.

**Value proposition:** Research must produce clear, accurate, general domain expertise — not content scoped to a single business.

**Current status:** This is a generated baseline playbook. Stage 1 should treat older lane docs as context and keep this lane clean as it matures.

**Operating constraints:**
- Produce general domain knowledge, not business-specific content
- Prefer durable doc structures over one-off note sprawl
- Scoped / time-sensitive work should become a topic doc, not bloat the foundation doc

## 4. Required Context for Subagent

Before spawning a research subagent, Atlas MUST load and pass in `context`:
1. **This playbook** (the full text of `{playbook_path.name}`)
2. **`🧠 Brain/playbooks/research-pipeline.md`**
3. **Existing lane docs** from `🏗️ Projects/rww/research/{lane_slug}/`
4. **The specific task** — what mode, what scope, what market, what deliverable

If any required files cannot be loaded, do NOT spawn — fix the missing file first.

## Durable {lane_title} Documentation Architecture

### The 3-layer {lane_title.lower()} doc stack

1. **Evergreen foundation doc**
   - Purpose: durable baseline understanding to preserve long-term
   - Canonical markdown artifact: `🏗️ Projects/rww/research/{lane_slug}/{lane_slug}-fundamentals-guide.md`
   - Canonical Google Doc link record: `🏗️ Projects/rww/research/{lane_slug}/doc-link.md`

2. **Central {lane_title.lower()} research index**
   - Purpose: one index doc linking the foundation doc plus every topic-specific, audit-specific, market-specific, and experiment-specific doc in this lane
   - Canonical markdown artifact: `🏗️ Projects/rww/research/{lane_slug}/{lane_slug}-research-index.md`
   - Canonical Google Doc link record: `🏗️ Projects/rww/research/{lane_slug}/index-doc-link.md`
   - Template source: `🧠 Brain/playbooks/templates/{index_template_path.name}`
   - If the index does not exist yet, Stage 3 must create it from the template before closing the run.

3. **Separate topic-specific / snapshot / audit docs**
   - Purpose: keep scoped, time-sensitive, tactic-specific, market-specific, and question-specific work out of the foundation doc
   - Topic / tactic / question path pattern: `🏗️ Projects/rww/research/{lane_slug}/topics/{{topic-slug}}/{{YYYY-MM-DD}}-{{topic-slug}}.md`
   - Topic / tactic / question Google Doc link record: `🏗️ Projects/rww/research/{lane_slug}/topics/{{topic-slug}}/doc-link.md`
   - Every one of these docs must also be added to the central {lane_title.lower()} research index.

### Stage 1 document-classification rule
Before Stage 2 web research begins, Stage 1 MUST read:
- the foundation doc + its `doc-link.md`
- the central index + its `doc-link.md` if they exist
- related topic docs, prior raw findings, and any related Obsidian notes

Then Stage 1 MUST return one of these document decisions:
- `update-foundation`
- `update-existing-doc`
- `new-topic-doc`
- `adjacent-topic-new-doc`

### Freshness, scope, and anti-duplication rules
- Every {lane_title.lower()} doc must state exact scope, date, and whether it is `evergreen` or a `snapshot`.
- Treat docs older than 90 days as context, not automatically current truth, unless Stage 1 explicitly confirms they still hold.
- When uncertain between `update` and `new doc`, choose the new scoped doc and link it back to the related prior doc.

### Stage 3 persistence rules
- Stage 3 must honor Stage 1's document decision.
- Every Google Doc must have a companion Obsidian `doc-link.md` record.
- Stage 3 must create or update the central research index on every run.
- If Google Docs or Drive capabilities are unavailable, Stage 3 must write a clear Obsidian stub showing the exact missing external boundary.

## 5. Research Modes

### Mode: {lane_slug}-education
**Goal:** Build general {lane_title.lower()} knowledge from the ground up — concepts, terminology, how it works, and why it matters.

**Workflow:**
1. Run Stage 1 document classification.
2. Research current domain fundamentals.
3. Write a clear, readable summary for Dax.
4. Persist findings to the correct foundation or topic artifact.
5. Update the lane index and doc-link records.

**Output:** foundation doc or new topic doc, plus index/doc-link persistence.

### Mode: {lane_slug}-audit
**Goal:** Audit a business, page set, competitor set, or market through the lens of this domain.

**Workflow:**
1. Read target context and any prior audits.
2. Use Stage 1 to classify whether this is a new snapshot or update.
3. Research the relevant gaps live.
4. Write prioritized findings.
5. Save the audit and update the index.

**Output:** scoped audit doc + doc-link + updated index entry.

### Mode: {lane_slug}-market-scan
**Goal:** Capture a market-specific or time-sensitive snapshot without polluting the evergreen foundation doc.

**Workflow:**
1. Read related lane docs and prior snapshots.
2. Use Stage 1 to classify the run.
3. Research the target market/topic live.
4. Synthesize findings into a dated scoped doc.
5. Update the index and doc-link records.

**Output:** dated topic or market-scan doc + doc-link + updated index entry.

## 6. Assigned Model
- **Model:** gpt-5.4 via openai-codex
- **Cost ceiling:** Max 20 turns per stage unless the pipeline playbook says otherwise
- **Timeout:** 30 minutes per run (hard limit)
- **Toolsets:** terminal, file, web

## 7. Output Formats

### Google Doc Link Record — doc-link.md
```markdown
# <Doc Title>
**Date:** <date>
**Doc Type:** <foundation|index|topic|audit|market-scan|experiment|update>
**Scope:** <exact scope this doc covers>
**Status:** <evergreen|current snapshot|superseded|stale>
**Google Doc:** <url-or-pending>

## Related Docs
- <related doc path or \"none\">

## Notes
- <freshness window, supersedes note, or why this doc exists>
```

## 8. Quality Bar
- [ ] Stage 1 checked the foundation doc, central index, and related prior docs before fresh research
- [ ] Stage 1 explicitly classified the run as `update-foundation`, `update-existing-doc`, `new-topic-doc`, or `adjacent-topic-new-doc`
- [ ] All claims are grounded in actual sources or actual local files
- [ ] Time-sensitive / scoped work went into a dated doc instead of bloating the foundation guide
- [ ] Every Google Doc has a paired `doc-link.md` record and an index entry
- [ ] Missing external capabilities are recorded explicitly when Google Docs/Drive steps cannot complete

## 9. Boundaries

### May do autonomously
- Create or refresh lane artifacts and playbooks in Obsidian
- Run the standard 3-stage research pipeline
- Create Google Docs when the tool is available and connected

### Requires Dax approval
- External outreach or sending results outside the normal research destinations
- Any paid external tool activation beyond existing connected services

## 10. Routing Contract
- Long-form research docs live in the requested research destination.
- Agent Activity receives concise spawn/completion visibility.
- The origin conversation gets only: short acknowledgment at launch, then short completion/blocker at the end.

## 11. Artifact Paths
- Lane root: `🏗️ Projects/rww/research/{lane_slug}/`
- Topic docs: `🏗️ Projects/rww/research/{lane_slug}/topics/{{topic-slug}}/`
- Pipeline state: `🏗️ Projects/rww/research/{lane_slug}/topics/{{topic-slug}}/pipeline-state.json` (or lane root for foundation runs)
- Google Drive note/stub: `drive-folder.md` in the active topic folder

## 12. Revision Memory
| Date | Change |
|------|--------|
| {_today_str()} | Generated lane playbook from the standard research template during lane launch. |
"""


def _resolve_playbook(lane_slug: str, playbook_slug: Optional[str]) -> Tuple[Path, str]:
    if playbook_slug:
        normalized = playbook_slug.strip().lower().removesuffix(".md")
    else:
        normalized = KNOWN_PLAYBOOK_ALIASES.get(lane_slug, f"{lane_slug}-research")
    return _playbooks_dir() / f"{normalized}.md", normalized


def _is_foundation_topic(lane_slug: str, topic_slug: str) -> bool:
    normalized_lane = (lane_slug or "").strip().lower()
    normalized_topic = (topic_slug or "").strip().lower()
    if not normalized_lane or not normalized_topic:
        return False
    return normalized_topic in {
        normalized_lane,
        f"{normalized_lane}-fundamentals",
        f"{normalized_lane}-fundamentals-guide",
    }


def _resolve_paths(lane_slug: str, topic_slug: str, playbook_path: Path, playbook_slug: str) -> Dict[str, Path]:
    lane_root = _research_root() / lane_slug
    topic_dir = lane_root if _is_foundation_topic(lane_slug, topic_slug) else (lane_root / "topics" / topic_slug)
    date_str = _today_str()
    return {
        "lane_root": lane_root,
        "topic_dir": topic_dir,
        "state_file": topic_dir / "pipeline-state.json",
        "lane_launch_note": topic_dir / "lane-launch.md",
        "drive_note": topic_dir / "drive-folder.md",
        "knowledge_brief": topic_dir / "knowledge-brief.md",
        "raw_findings": topic_dir / "raw-findings.md",
        "default_writeup": topic_dir / f"{date_str}-{topic_slug}.md",
        "foundation_doc": lane_root / f"{lane_slug}-fundamentals-guide.md",
        "index_doc": lane_root / f"{lane_slug}-research-index.md",
        "doc_link": topic_dir / "doc-link.md",
        "index_doc_link": lane_root / "index-doc-link.md",
        "playbook": playbook_path,
        "index_template": _templates_dir() / f"{lane_slug}-research-index-template.md",
    }


LIBRARY_CHAT_ID = "-1003727531067"
GOOGLE_DRIVE_RESEARCH_PARENT_ID = "0ABiwxqF63XS0Uk9PVA"


def _get_bot_token() -> Optional[str]:
    """Read Telegram bot token from .env or environment."""
    env_path = Path(os.path.expanduser("~/.hermes/.env"))
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == "TELEGRAM_BOT_TOKEN":
                return v.strip()
    return os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("HERMES_TELEGRAM_BOT_TOKEN")


def _create_telegram_topic(lane_title: str) -> Optional[Dict[str, Any]]:
    """Create a forum topic in the Library group chat via Telegram Bot API.

    Returns {"thread_id": int, "name": str} on success, None on failure.
    """
    import urllib.error
    import urllib.parse
    import urllib.request

    token = _get_bot_token()
    if not token:
        logger.warning("No Telegram bot token — cannot create topic for %s", lane_title)
        return None

    topic_name = f"{lane_title} Research"
    params = json.dumps({
        "chat_id": int(LIBRARY_CHAT_ID),
        "name": topic_name,
        "icon_color": 7322096,  # blue
    })

    url = f"https://api.telegram.org/bot{token}/createForumTopic"
    req = urllib.request.Request(
        url,
        data=params.encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        if body.get("ok") and body.get("result"):
            thread_id = body["result"].get("message_thread_id")
            logger.info("Created Telegram topic '%s' (thread_id=%s) in Library chat", topic_name, thread_id)
            return {"thread_id": thread_id, "name": topic_name}
        logger.warning("createForumTopic returned ok=false: %s", body)
        return None
    except Exception as exc:
        logger.warning("Failed to create Telegram topic for %s: %s", lane_title, exc)
        return None


def _register_topic_in_config(thread_id: int, topic_name: str) -> bool:
    """Add the new topic to config.yaml group_topics and group_system_prompts."""
    try:
        import yaml
        config_path = Path(os.path.expanduser("~/.hermes/config.yaml"))
        if not config_path.exists():
            return False
        text = config_path.read_text(encoding="utf-8")
        cfg = yaml.safe_load(text) or {}

        # Add to group_topics
        tg = cfg.setdefault("telegram", {}).setdefault("extra", {})
        group_topics = tg.setdefault("group_topics", [])
        library_group = None
        for group in group_topics:
            if str(group.get("chat_id")) == str(LIBRARY_CHAT_ID).lstrip("-"):
                library_group = group
                break
            if str(group.get("chat_id")) == LIBRARY_CHAT_ID:
                library_group = group
                break
        if library_group is None:
            library_group = {"chat_id": int(LIBRARY_CHAT_ID), "topics": []}
            group_topics.append(library_group)
        topics = library_group.setdefault("topics", [])
        # Check if already exists
        if not any(t.get("thread_id") == thread_id for t in topics):
            topics.append({"name": topic_name, "thread_id": thread_id})

        # Add to group_system_prompts
        prompts = cfg.setdefault("group_system_prompts", {})
        chat_prompts = prompts.setdefault(LIBRARY_CHAT_ID, {})
        tid_str = str(thread_id)
        if tid_str not in chat_prompts:
            clean_name = topic_name.upper()
            chat_prompts[tid_str] = {
                "name": topic_name,
                "prompt": (
                    f"You are Atlas in the {clean_name} workspace. "
                    f"Keep this topic focused on {topic_name} only, "
                    "and keep outputs research-first rather than execution management."
                ),
            }

        config_path.write_text(yaml.dump(cfg, default_flow_style=False, allow_unicode=True, sort_keys=False), encoding="utf-8")
        logger.info("Registered topic '%s' (thread_id=%s) in config.yaml", topic_name, thread_id)
        return True
    except Exception as exc:
        logger.warning("Failed to register topic in config.yaml: %s", exc)
        return False


def _create_google_drive_folder(lane_title: str, parent_folder_id: Optional[str] = GOOGLE_DRIVE_RESEARCH_PARENT_ID) -> Optional[Dict[str, Any]]:
    """Create a Google Drive folder for the research lane via MCP.

    Returns {"folder_id": str, "url": str} on success, None if MCP unavailable.
    """
    try:
        tool_names = set(registry.get_all_tool_names())
        create_folder_tool = "mcp_maton_google_docs_google_drive_create_folder"
        if create_folder_tool not in tool_names:
            logger.debug("Google Drive create_folder tool not registered — cannot create folder")
            return None

        folder_name = f"{lane_title} Research"
        args = {"name": folder_name}
        if parent_folder_id:
            args["parent_id"] = parent_folder_id

        raw = registry.dispatch(create_folder_tool, args)
        parsed = json.loads(raw) if isinstance(raw, str) else raw

        # Extract folder ID from MCP response
        folder_data = None
        if isinstance(parsed, dict):
            content = parsed.get("content")
            if isinstance(content, list) and content:
                text = content[0].get("text", "")
                folder_data = json.loads(text) if text else {}
            elif isinstance(parsed.get("result"), str):
                folder_data = json.loads(parsed["result"])
            elif isinstance(parsed.get("result"), dict):
                folder_data = parsed["result"]
            elif parsed.get("id"):
                folder_data = parsed

        if isinstance(folder_data, dict) and folder_data.get("id"):
            folder_id = folder_data["id"]
            url = f"https://drive.google.com/drive/folders/{folder_id}"
            logger.info("Created Google Drive folder '%s' (id=%s)", folder_name, folder_id)
            return {"folder_id": folder_id, "url": url, "name": folder_name}

        logger.warning("Drive create_folder returned unexpected response: %s", str(parsed)[:200])
        return None
    except Exception as exc:
        logger.warning("Google Drive folder creation failed: %s", exc)
        return None


def _ensure_lane_assets(
    *,
    lane_title: str,
    lane_slug: str,
    topic_title: str,
    topic_slug: str,
    request: str,
    deliver: Optional[str],
    playbook_path: Path,
    playbook_slug: str,
    create_playbook_if_missing: bool,
) -> Dict[str, Any]:
    paths = _resolve_paths(lane_slug, topic_slug, playbook_path, playbook_slug)
    for key in ("lane_root", "topic_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)

    _ensure_generic_index_standard_template()
    generated_playbook = False
    generated_index_template = False

    if create_playbook_if_missing and not paths["index_template"].exists():
        _write_text(paths["index_template"], _render_lane_index_template(lane_title, lane_slug))
        generated_index_template = True

    if create_playbook_if_missing and not playbook_path.exists():
        playbook_path.parent.mkdir(parents=True, exist_ok=True)
        _write_text(playbook_path, _render_generated_playbook(lane_title, lane_slug, playbook_path, paths["index_template"]))
        generated_playbook = True

    # --- Telegram topic creation ---
    # Only create a topic for THIS lane. Check 3 sources before creating:
    # 1. The lane's own topic-info file (survives config write failures)
    # 2. Config.yaml (via _parse_deliver_target)
    # 3. Explicit deliver target from the caller
    telegram_topic = None
    expected_topic_name = f"{lane_title} Research"
    topic_info_file = paths["lane_root"] / "telegram-topic.json"

    # Source 1: lane's own topic record (most reliable — not affected by config lock)
    existing_topic_info = _load_json(topic_info_file)
    if existing_topic_info and existing_topic_info.get("thread_id"):
        tid = existing_topic_info["thread_id"]
        deliver = f"telegram:{LIBRARY_CHAT_ID}:{tid}"
        logger.info("Found existing topic record for lane %s: thread_id=%s", lane_slug, tid)
    else:
        # Source 2: config.yaml
        from tools.delegate_tool import _parse_deliver_target
        existing_target = _parse_deliver_target(expected_topic_name)
        if existing_target:
            deliver = f"telegram:{existing_target['chat_id']}"
            if existing_target.get("thread_id"):
                deliver += f":{existing_target['thread_id']}"
            # Save to lane folder so we don't depend on config next time
            _write_json(topic_info_file, {"thread_id": existing_target.get("thread_id"), "name": expected_topic_name, "chat_id": existing_target["chat_id"]})
            logger.info("Found existing Telegram topic '%s' in config → deliver=%s", expected_topic_name, deliver)
        elif not deliver or deliver == "origin":
            # Source 3: No existing topic anywhere — create one for THIS lane only
            telegram_topic = _create_telegram_topic(lane_title)
            if telegram_topic and telegram_topic.get("thread_id"):
                _register_topic_in_config(telegram_topic["thread_id"], telegram_topic["name"])
                # Always save to lane folder as the durable record
                _write_json(topic_info_file, {"thread_id": telegram_topic["thread_id"], "name": telegram_topic["name"], "chat_id": LIBRARY_CHAT_ID})
                deliver = f"telegram:{LIBRARY_CHAT_ID}:{telegram_topic['thread_id']}"
                logger.info("Auto-created Telegram topic for lane %s: %s → deliver=%s", lane_slug, telegram_topic["name"], deliver)
        # If an explicit deliver was passed, use it as-is — do NOT create a new topic

    # --- Google Drive folder creation ---
    # Check if a Drive folder for this lane already exists (via drive note from prior run)
    drive_result = None
    existing_drive_note = _read_text(paths["drive_note"])
    if "folder_id" in existing_drive_note or "drive.google.com/drive/folders/" in existing_drive_note:
        logger.info("Drive folder already exists for lane %s — skipping creation", lane_slug)
    else:
        drive_result = _create_google_drive_folder(lane_title)
    drive_status = _google_drive_capability_status()
    if drive_result:
        drive_status["folder_id"] = drive_result.get("folder_id")
        drive_status["folder_url"] = drive_result.get("url")

    _write_text(
        paths["drive_note"],
        _render_drive_note(
            lane_title=lane_title,
            lane_slug=lane_slug,
            topic_title=topic_title,
            topic_slug=topic_slug,
            deliver=deliver,
            drive_status=drive_status,
        ),
    )

    _write_text(
        paths["lane_launch_note"],
        _render_lane_launch_note(
            lane_title=lane_title,
            lane_slug=lane_slug,
            topic_title=topic_title,
            topic_slug=topic_slug,
            request=request,
            deliver=deliver,
            playbook_path=playbook_path,
            drive_note=paths["drive_note"],
            state_file=paths["state_file"],
            generated_playbook=generated_playbook,
            generated_index_template=generated_index_template,
        ),
    )

    return {
        "paths": paths,
        "generated_playbook": generated_playbook,
        "generated_index_template": generated_index_template,
        "drive_status": drive_status,
        "telegram_topic": telegram_topic,
        "deliver": deliver,
    }


def _render_lane_launch_note(
    *,
    lane_title: str,
    lane_slug: str,
    topic_title: str,
    topic_slug: str,
    request: str,
    deliver: Optional[str],
    playbook_path: Path,
    drive_note: Path,
    state_file: Path,
    generated_playbook: bool,
    generated_index_template: bool,
) -> str:
    return f"""# Research Lane Launch — {topic_title}
**Date:** {_today_str()}
**Lane:** {lane_title} (`{lane_slug}`)
**Topic:** {topic_title} (`{topic_slug}`)
**Deliver Target:** {deliver or 'origin'}
**Quiet UX Contract:** Acknowledge once up front, then report only final completion or blocker.

## Request
{request}

## Assets
- **Playbook:** `{playbook_path}`{' (generated during launch)' if generated_playbook else ''}
- **Pipeline State:** `{state_file}`
- **Drive Step Record:** `{drive_note}`
- **Lane Root:** `🏗️ Projects/rww/research/{lane_slug}/`
- **Topic Folder:** `🏗️ Projects/rww/research/{lane_slug}/topics/{topic_slug}/` if scoped topic

## Setup Notes
- Generated lane playbook this launch: {'yes' if generated_playbook else 'no'}
- Generated lane index template this launch: {'yes' if generated_index_template else 'no'}
- Google Drive step handled via `drive-folder.md`
"""


def _google_drive_capability_status() -> Dict[str, Any]:
    tool_names = set(registry.get_all_tool_names())
    drive_tools = sorted(name for name in tool_names if "drive" in name.lower())
    folder_tools = sorted(name for name in drive_tools if "folder" in name.lower())
    if folder_tools:
        return {
            "status": "available",
            "detail": "Live Google Drive folder capability appears available.",
            "tools": folder_tools,
        }
    return {
        "status": "stubbed-unavailable",
        "detail": "No live Google Drive folder creation tool is registered in Hermes yet.",
        "tools": drive_tools,
    }


def _render_drive_note(
    *,
    lane_title: str,
    lane_slug: str,
    topic_title: str,
    topic_slug: str,
    deliver: Optional[str],
    drive_status: Dict[str, Any],
) -> str:
    desired_folder = f"Research / {lane_title} / {topic_title}"
    tools_line = ", ".join(drive_status.get("tools") or []) or "none"
    return f"""# Google Drive Folder — {topic_title}
**Date:** {_today_str()}
**Lane:** {lane_title}
**Topic:** {topic_title}
**Desired Folder:** {desired_folder}
**Status:** {drive_status.get('status')}

## Capability Boundary
- {drive_status.get('detail')}
- Registered Drive-related tools seen by Hermes: {tools_line}

## Next Action
- If/when Drive folder creation becomes available, create the folder above and update this file with the real Drive URL / folder ID.
- Until then, treat this file as the source-of-truth stub for the Drive step.

## Related Paths
- Lane root: `🏗️ Projects/rww/research/{lane_slug}/`
- Topic folder: `🏗️ Projects/rww/research/{lane_slug}/topics/{topic_slug}/`
- Deliver target: {deliver or 'origin'}
"""


def _default_state(
    *,
    lane_title: str,
    lane_slug: str,
    topic_title: str,
    topic_slug: str,
    playbook_slug: str,
    request: str,
    deliver: Optional[str],
    paths: Dict[str, Path],
    generated_playbook: bool,
    drive_status: Dict[str, Any],
) -> Dict[str, Any]:
    now = _utc_now_iso()
    return {
        "topic": topic_slug,
        "topic_title": topic_title,
        "lane": lane_slug,
        "lane_title": lane_title,
        "pipeline": "research",
        "playbook": playbook_slug,
        "started_at": now,
        "current_stage": "stage-1-knowledge",
        "request": request,
        "deliver": deliver or "origin",
        "lane_launch": {
            "status": "completed",
            "completed_at": now,
            "quiet_mode": True,
            "playbook_path": str(paths["playbook"]),
            "playbook_generated": generated_playbook,
            "lane_root": str(paths["lane_root"]),
            "topic_dir": str(paths["topic_dir"]),
            "artifacts": [
                str(paths["lane_launch_note"]),
                str(paths["drive_note"]),
            ],
            "google_drive": drive_status,
        },
        "stages": {
            "stage-1-knowledge": {"status": "pending", "artifact": str(paths["knowledge_brief"])} ,
            "stage-2-web-research": {"status": "pending", "artifact": str(paths["raw_findings"])},
            "stage-3-writeup": {"status": "pending", "artifact": str(paths["default_writeup"])} ,
        },
        "last_updated": now,
    }


def _artifact_exists(stage_name: str, stage_state: Dict[str, Any], paths: Dict[str, Path]) -> bool:
    candidates: List[Path] = []
    for key in ("artifact",):
        value = stage_state.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(Path(value))
    artifacts = stage_state.get("artifacts") or []
    if isinstance(artifacts, list):
        for item in artifacts:
            if isinstance(item, str) and item.strip():
                candidates.append(Path(item))
    if stage_name == "stage-1-knowledge":
        candidates.append(paths["knowledge_brief"])
    elif stage_name == "stage-2-web-research":
        candidates.append(paths["raw_findings"])
    elif stage_name == "stage-3-writeup":
        candidates.extend([paths["default_writeup"], paths["doc_link"], paths["index_doc"], paths["index_doc_link"]])
    return any(path.exists() for path in candidates)


def _required_stage3_artifacts_exist(paths: Dict[str, Path], artifacts: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
    required = [paths["doc_link"], paths["index_doc"], paths["index_doc_link"]]
    markdown_present = paths["default_writeup"].exists()
    if not markdown_present and isinstance(artifacts, list):
        markdown_present = any(isinstance(item, str) and item.strip() and Path(item).exists() for item in artifacts)
    missing = [str(path) for path in required if not path.exists()]
    if not markdown_present:
        missing.insert(0, str(paths["default_writeup"]))
    return (len(missing) == 0), missing


def _save_state(state: Dict[str, Any], state_file: Path) -> None:
    state["last_updated"] = _utc_now_iso()
    _write_json(state_file, state)


def _mark_stage_in_progress(state: Dict[str, Any], stage_name: str, state_file: Path) -> None:
    now = _utc_now_iso()
    state["current_stage"] = stage_name
    stage = state.setdefault("stages", {}).setdefault(stage_name, {})
    stage["status"] = "in-progress"
    stage["started_at"] = now
    stage.pop("error", None)
    _save_state(state, state_file)


def _mark_stage_completed(
    state: Dict[str, Any],
    stage_name: str,
    state_file: Path,
    *,
    artifact: Optional[str] = None,
    artifacts: Optional[List[str]] = None,
    extras: Optional[Dict[str, Any]] = None,
) -> None:
    now = _utc_now_iso()
    stage = state.setdefault("stages", {}).setdefault(stage_name, {})
    stage["status"] = "completed"
    stage["completed_at"] = now
    if artifact:
        stage["artifact"] = artifact
    if artifacts:
        stage["artifacts"] = artifacts
    if extras:
        stage.update(extras)
    _save_state(state, state_file)


def _mark_stage_failed(state: Dict[str, Any], stage_name: str, state_file: Path, error: str) -> None:
    now = _utc_now_iso()
    stage = state.setdefault("stages", {}).setdefault(stage_name, {})
    stage["status"] = "failed"
    stage["failed_at"] = now
    stage["error"] = error
    state["current_stage"] = stage_name
    _save_state(state, state_file)


def _next_stage(stage_name: str) -> Optional[str]:
    try:
        idx = STAGE_SEQUENCE.index(stage_name)
    except ValueError:
        return None
    return STAGE_SEQUENCE[idx + 1] if idx + 1 < len(STAGE_SEQUENCE) else None


def _resolve_resume_stage(state: Dict[str, Any], paths: Dict[str, Path], state_file: Path) -> Tuple[Optional[str], Optional[str]]:
    stages = state.setdefault("stages", {})
    for stage_name in STAGE_SEQUENCE:
        stage = stages.setdefault(stage_name, {"status": "pending"})
        status = stage.get("status", "pending")
        if status == "failed":
            # Clear the failed status so we can retry instead of permanently blocking
            stage["status"] = "pending"
            stage.pop("error", None)
            _save_state(state, state_file)
            logger.info("Cleared failed status for %s — will retry", stage_name)
            return stage_name, None
        if status == "in-progress":
            if _artifact_exists(stage_name, stage, paths):
                _mark_stage_completed(
                    state,
                    stage_name,
                    state_file,
                    artifact=str(stage.get("artifact") or paths["knowledge_brief" if stage_name == "stage-1-knowledge" else "raw_findings" if stage_name == "stage-2-web-research" else "default_writeup"]),
                    artifacts=stage.get("artifacts"),
                )
                return _next_stage(stage_name), None
            return stage_name, None
        if status == "completed":
            # Verify artifacts actually exist on disk — don't trust status alone
            if not _artifact_exists(stage_name, stage, paths):
                logger.warning("Stage %s marked completed but artifacts missing on disk — rerunning", stage_name)
                stage["status"] = "pending"
                _save_state(state, state_file)
                return stage_name, None
            continue
        # Pending or unknown status — start from here
        return stage_name, None
    return None, None


def _resolve_origin_target() -> Optional[Dict[str, str]]:
    try:
        from gateway.session_context import get_session_env
    except Exception:
        def get_session_env(name: str, default: str = "") -> str:
            return os.getenv(name, default)

    platform = (get_session_env("HERMES_SESSION_PLATFORM", "") or "").strip().lower()
    if platform and platform != "telegram":
        return None
    chat_id = (get_session_env("HERMES_SESSION_CHAT_ID", "") or "").strip()
    thread_id = (get_session_env("HERMES_SESSION_THREAD_ID", "") or "").strip()
    if not chat_id:
        return None
    return {"chat_id": chat_id, "thread_id": thread_id or None}


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text or not isinstance(text, str):
        return None
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            payload, _ = decoder.raw_decode(text[match.start():])
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


_TRANSIENT_ERROR_PATTERNS = (
    "overloaded",
    "try again later",
    "rate limit",
    "timeout",
    "timed_out",
    "connection refused",
    "connection reset",
    "502",
    "503",
    "504",
    "APIError",
    "request_id",
)


def _is_transient_error(error_text: str) -> bool:
    lower = (error_text or "").lower()
    return any(pattern.lower() in lower for pattern in _TRANSIENT_ERROR_PATTERNS)


def _delegate_single(
    *,
    goal: str,
    context: str,
    toolsets: List[str],
    max_iterations: int,
    parent_agent,
    max_retries: int = 1,
) -> Dict[str, Any]:
    import time as _time
    from tools.delegate_tool import delegate_task

    last_error = None
    for attempt in range(1 + max_retries):
        if attempt > 0:
            logger.info("Retrying stage delegation (attempt %d/%d) after transient error: %s",
                        attempt + 1, 1 + max_retries, last_error)
            _time.sleep(min(10 * attempt, 30))

        raw = delegate_task(
            goal=goal,
            context=context,
            toolsets=toolsets,
            max_iterations=max_iterations,
            background=False,
            parent_agent=parent_agent,
        )
        try:
            parsed = json.loads(raw)
        except Exception:
            return {"error": f"delegate_task returned non-JSON: {raw[:500]}"}

        if parsed.get("error"):
            error_str = str(parsed["error"])
            if attempt < max_retries and _is_transient_error(error_str):
                last_error = error_str
                continue
            return {"error": error_str}

        results = parsed.get("results") or []
        if not results:
            return {"error": "delegate_task returned no child results"}

        first = results[0]
        if first.get("error") and attempt < max_retries and _is_transient_error(str(first["error"])):
            last_error = str(first["error"])
            continue
        if first.get("status") == "failed" and attempt < max_retries and _is_transient_error(str(first.get("error") or first.get("summary") or "")):
            last_error = str(first.get("error") or first.get("summary") or "failed")
            continue

        return first

    return {"error": f"Stage failed after {1 + max_retries} attempts. Last error: {last_error}"}


def _google_docs_status(parent_agent=None) -> Dict[str, Any]:
    tool_names = set(registry.get_all_tool_names())
    required = {
        "mcp_maton_google_docs_google_docs_check_connection",
        "mcp_maton_google_docs_google_docs_create_document",
        "mcp_maton_google_docs_google_docs_insert_text",
    }
    if not required.issubset(tool_names):
        return {
            "status": "unavailable",
            "detail": "Google Docs MCP tools are not fully registered.",
        }
    try:
        raw = registry.dispatch(
            "mcp_maton_google_docs_google_docs_check_connection",
            {},
            parent_agent=parent_agent,
        )
        parsed = json.loads(raw)
        payload = None
        if isinstance(parsed, dict):
            content = parsed.get("content")
            if isinstance(content, list) and content:
                text = content[0].get("text", "")
                payload = json.loads(text) if text else {}
            elif isinstance(parsed.get("result"), str) and parsed.get("result", "").strip():
                payload = json.loads(parsed["result"])
            elif isinstance(parsed.get("result"), dict):
                payload = parsed.get("result")
        if isinstance(payload, dict):
            if payload.get("connected"):
                return {"status": "connected", "detail": "Google Docs connection is active."}
            if payload.get("connected") is False:
                return {"status": "disconnected", "detail": "Google Docs MCP is registered but no active connection was found."}
    except Exception as exc:
        logger.debug("Google Docs connection check failed: %s", exc)
    return {"status": "unknown", "detail": "Google Docs MCP tools are present but connection status could not be confirmed."}


def _load_authority_texts(playbook_path: Path) -> Dict[str, str]:
    pipeline_path = _playbooks_dir() / "research-pipeline.md"
    return {
        "pipeline": _read_text(pipeline_path),
        "playbook": _read_text(playbook_path),
    }


def _build_stage_context(
    *,
    lane_title: str,
    lane_slug: str,
    topic_title: str,
    topic_slug: str,
    request: str,
    paths: Dict[str, Path],
    authority: Dict[str, str],
    stage_name: str,
    stage_inputs: Optional[Dict[str, str]] = None,
    google_docs_state: Optional[Dict[str, Any]] = None,
) -> str:
    sections = [
        f"RESEARCH LANE: {lane_title} ({lane_slug})",
        f"TOPIC: {topic_title} ({topic_slug})",
        f"REQUEST: {request}",
        "QUIET UX CONTRACT: Do the work without asking the user for permission between stages unless genuinely blocked.",
        f"LANE ROOT: {paths['lane_root']}",
        f"TOPIC DIR: {paths['topic_dir']}",
        f"PIPELINE STATE: {paths['state_file']}",
        f"LANE LAUNCH NOTE: {paths['lane_launch_note']}",
        f"DRIVE NOTE: {paths['drive_note']}",
    ]
    if authority.get("pipeline"):
        sections.append("=== RESEARCH PIPELINE PLAYBOOK ===\n" + authority["pipeline"] + "\n=== END PLAYBOOK ===")
    if authority.get("playbook"):
        sections.append(f"=== DOMAIN PLAYBOOK: {paths['playbook'].name} ===\n" + authority["playbook"] + "\n=== END DOMAIN PLAYBOOK ===")
    if stage_inputs:
        if stage_inputs.get("knowledge_brief"):
            sections.append("=== STAGE 1 KNOWLEDGE BRIEF ===\n" + stage_inputs["knowledge_brief"] + "\n=== END STAGE 1 ===")
        if stage_inputs.get("raw_findings"):
            sections.append("=== STAGE 2 RAW FINDINGS ===\n" + stage_inputs["raw_findings"] + "\n=== END STAGE 2 ===")
    if google_docs_state:
        sections.append("GOOGLE DOCS STATUS: " + json.dumps(google_docs_state, ensure_ascii=False))
    sections.append(f"ACTIVE STAGE: {stage_name}")
    return "\n\n".join(sections)


def _stage_goal_stage1(topic_title: str, artifact_path: Path) -> str:
    return (
        f"[Stage 1 - Knowledge Assessment] {topic_title}. Read our existing local files and produce the knowledge brief. "
        f"Write the full markdown brief to {artifact_path}. Return JSON only with keys: "
        f"status, artifact, document_decision, summary."
    )


def _stage_goal_stage2(topic_title: str, artifact_path: Path) -> str:
    return (
        f"[Stage 2 - Web Research] {topic_title}. Research the gaps identified in the Stage 1 knowledge brief. "
        f"Write the full raw findings markdown to {artifact_path}. Return JSON only with keys: "
        f"status, artifact, sources_scraped, summary."
    )


def _stage_goal_stage3(
    *,
    lane_title: str,
    topic_title: str,
    default_artifact_path: Path,
    doc_link_path: Path,
    index_doc_path: Path,
    index_doc_link_path: Path,
    google_docs_state: Dict[str, Any],
) -> str:
    google_docs_instruction = (
        "Create the Google Doc as the primary output, write the content into it, and include the final Google Doc URL in your JSON return. "
        if google_docs_state.get("status") == "connected"
        else "Google Docs is not available for live creation in this run. Create the markdown/doc-link/index artifacts anyway and clearly mark the external boundary in doc-link.md and your JSON return. "
    )
    return (
        f"[Stage 3 - Write-up] {topic_title}. Write a clear, readable research doc for Dax. "
        f"Honor the Stage 1 document decision. Default markdown target if no better playbook-defined path applies: {default_artifact_path}. "
        f"Create or update the supporting doc-link at {doc_link_path}, the central index at {index_doc_path}, and the index doc-link at {index_doc_link_path}. "
        f"{google_docs_instruction}"
        f"Return JSON only with keys: status, artifact, artifacts, google_doc_url, google_doc_status, summary."
    )


def _stage_result_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    payload = _extract_json_object(result.get("summary") or "") or {}
    payload.setdefault("status", result.get("status"))
    payload.setdefault("summary", result.get("summary") or "")
    if result.get("error") and not payload.get("error"):
        payload["error"] = result.get("error")
    return payload


def _build_final_message(topic_title: str, stage3_payload: Dict[str, Any], state_file: Path) -> str:
    google_doc_url = str(stage3_payload.get("google_doc_url") or "").strip()
    artifact = str(stage3_payload.get("artifact") or "").strip()
    if google_doc_url:
        return f"{topic_title} research complete. {google_doc_url}"
    if artifact:
        return f"{topic_title} research complete. Markdown saved to {artifact}"
    return f"{topic_title} research complete. State: {state_file}"


def _build_blocker_message(topic_title: str, blocker: str, state_file: Path) -> str:
    short = re.sub(r"\s+", " ", blocker or "Unknown blocker").strip()
    if len(short) > 280:
        short = short[:277].rstrip() + "..."
    return f"{topic_title} research blocked. {short} State: {state_file}"


def _send_telegram(target: Optional[Dict[str, str]], text: str, *, context: str, target_label: Optional[str] = None) -> bool:
    if not target:
        return False
    from tools.delegate_tool import _send_telegram_sync

    return _send_telegram_sync(
        target["chat_id"],
        target.get("thread_id"),
        text,
        context=context,
        target_label=target_label,
    )


def _same_telegram_target(a: Optional[Dict[str, str]], b: Optional[Dict[str, str]]) -> bool:
    if not a or not b:
        return False
    return str(a.get("chat_id") or "") == str(b.get("chat_id") or "") and str(a.get("thread_id") or "") == str(b.get("thread_id") or "")


def _verify_stage3_artifacts(paths: Dict[str, Path], payload: Dict[str, Any]) -> Tuple[bool, str, List[str], str]:
    artifact_candidates: List[Path] = []
    primary_artifact = str(payload.get("artifact") or "").strip()
    if primary_artifact:
        artifact_candidates.append(Path(primary_artifact))
    artifact_candidates.append(paths["default_writeup"])

    found_primary = next((path for path in artifact_candidates if path.exists()), None)
    if not found_primary:
        return False, f"Stage 3 completed without writing a final markdown artifact at {paths['default_writeup']}", [], ""

    required_supporting = [paths["doc_link"], paths["index_doc"], paths["index_doc_link"]]
    missing_supporting = [str(path) for path in required_supporting if not path.exists()]
    if missing_supporting:
        return False, "Stage 3 completed without writing all required support artifacts", missing_supporting, str(found_primary)

    artifacts: List[str] = []
    raw_artifacts = payload.get("artifacts")
    if isinstance(raw_artifacts, list):
        for item in raw_artifacts:
            if isinstance(item, str) and item.strip():
                artifacts.append(item.strip())

    required_all = [str(found_primary)] + [str(path) for path in required_supporting]
    seen = set(artifacts)
    for item in required_all:
        if item not in seen:
            artifacts.append(item)
            seen.add(item)

    return True, "", artifacts, str(found_primary)


def _run_lane_pipeline(
    *,
    lane_title: str,
    lane_slug: str,
    topic_title: str,
    topic_slug: str,
    request: str,
    deliver: Optional[str],
    playbook_slug: str,
    state: Dict[str, Any],
    state_file: Path,
    paths: Dict[str, Path],
    parent_agent,
    origin_target: Optional[Dict[str, str]],
    deliver_target: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    authority = _load_authority_texts(paths["playbook"])
    google_docs_state = _google_docs_status(parent_agent)

    resume_stage, blocker = _resolve_resume_stage(state, paths, state_file)
    if blocker:
        blocker_msg = _build_blocker_message(topic_title, blocker, state_file)
        _send_telegram(origin_target, blocker_msg, context="research_lane_launch blocker")
        if deliver_target and not _same_telegram_target(origin_target, deliver_target):
            _send_telegram(deliver_target, blocker_msg, context="research_lane_launch blocker deliver", target_label=deliver)
        return {"status": "blocked", "blocker": blocker, "state_file": str(state_file)}
    if resume_stage is None:
        final_msg = _build_final_message(topic_title, state.get("stages", {}).get("stage-3-writeup", {}), state_file)
        # Only deliver to the research topic — origin doesn't need the message
        if deliver_target:
            _send_telegram(deliver_target, final_msg, context="research_lane_launch deliver", target_label=deliver)
        elif origin_target:
            _send_telegram(origin_target, final_msg, context="research_lane_launch completion")
        return {"status": "completed", "message": final_msg, "state_file": str(state_file)}

    stage_inputs: Dict[str, str] = {}
    if paths["knowledge_brief"].exists():
        stage_inputs["knowledge_brief"] = _read_text(paths["knowledge_brief"])
    if paths["raw_findings"].exists():
        stage_inputs["raw_findings"] = _read_text(paths["raw_findings"])

    for stage_name in STAGE_SEQUENCE[STAGE_SEQUENCE.index(resume_stage):]:
        _mark_stage_in_progress(state, stage_name, state_file)

        if stage_name == "stage-1-knowledge":
            goal = _stage_goal_stage1(topic_title, paths["knowledge_brief"])
            context = _build_stage_context(
                lane_title=lane_title,
                lane_slug=lane_slug,
                topic_title=topic_title,
                topic_slug=topic_slug,
                request=request,
                paths=paths,
                authority=authority,
                stage_name=stage_name,
            )
            result = _delegate_single(goal=goal, context=context, toolsets=["file"], max_iterations=30, parent_agent=parent_agent)
            if result.get("error"):
                _mark_stage_failed(state, stage_name, state_file, result["error"])
                msg = _build_blocker_message(topic_title, result["error"], state_file)
                _send_telegram(origin_target, msg, context="research_lane_launch blocker")
                if deliver_target and not _same_telegram_target(origin_target, deliver_target):
                    _send_telegram(deliver_target, msg, context="research_lane_launch blocker deliver", target_label=deliver)
                return {"status": "blocked", "blocker": result["error"], "state_file": str(state_file)}
            if not paths["knowledge_brief"].exists():
                error = f"Stage 1 completed without writing {paths['knowledge_brief']}"
                _mark_stage_failed(state, stage_name, state_file, error)
                msg = _build_blocker_message(topic_title, error, state_file)
                _send_telegram(origin_target, msg, context="research_lane_launch blocker")
                if deliver_target and not _same_telegram_target(origin_target, deliver_target):
                    _send_telegram(deliver_target, msg, context="research_lane_launch blocker deliver", target_label=deliver)
                return {"status": "blocked", "blocker": error, "state_file": str(state_file)}
            payload = _stage_result_payload(result)
            stage_inputs["knowledge_brief"] = _read_text(paths["knowledge_brief"])
            extras = {}
            if payload.get("document_decision"):
                extras["document_decision"] = payload.get("document_decision")
            _mark_stage_completed(state, stage_name, state_file, artifact=str(paths["knowledge_brief"]), extras=extras)

        elif stage_name == "stage-2-web-research":
            goal = _stage_goal_stage2(topic_title, paths["raw_findings"])
            context = _build_stage_context(
                lane_title=lane_title,
                lane_slug=lane_slug,
                topic_title=topic_title,
                topic_slug=topic_slug,
                request=request,
                paths=paths,
                authority=authority,
                stage_name=stage_name,
                stage_inputs=stage_inputs,
            )
            result = _delegate_single(goal=goal, context=context, toolsets=["web", "file"], max_iterations=25, parent_agent=parent_agent)
            if result.get("error"):
                _mark_stage_failed(state, stage_name, state_file, result["error"])
                msg = _build_blocker_message(topic_title, result["error"], state_file)
                _send_telegram(origin_target, msg, context="research_lane_launch blocker")
                if deliver_target and not _same_telegram_target(origin_target, deliver_target):
                    _send_telegram(deliver_target, msg, context="research_lane_launch blocker deliver", target_label=deliver)
                return {"status": "blocked", "blocker": result["error"], "state_file": str(state_file)}
            if not paths["raw_findings"].exists():
                error = f"Stage 2 completed without writing {paths['raw_findings']}"
                _mark_stage_failed(state, stage_name, state_file, error)
                msg = _build_blocker_message(topic_title, error, state_file)
                _send_telegram(origin_target, msg, context="research_lane_launch blocker")
                if deliver_target and not _same_telegram_target(origin_target, deliver_target):
                    _send_telegram(deliver_target, msg, context="research_lane_launch blocker deliver", target_label=deliver)
                return {"status": "blocked", "blocker": error, "state_file": str(state_file)}
            payload = _stage_result_payload(result)
            stage_inputs["raw_findings"] = _read_text(paths["raw_findings"])
            extras = {}
            if payload.get("sources_scraped") is not None:
                extras["sources_scraped"] = payload.get("sources_scraped")
            _mark_stage_completed(state, stage_name, state_file, artifact=str(paths["raw_findings"]), extras=extras)

        elif stage_name == "stage-3-writeup":
            goal = _stage_goal_stage3(
                lane_title=lane_title,
                topic_title=topic_title,
                default_artifact_path=paths["default_writeup"],
                doc_link_path=paths["doc_link"],
                index_doc_path=paths["index_doc"],
                index_doc_link_path=paths["index_doc_link"],
                google_docs_state=google_docs_state,
            )
            stage3_toolsets = ["file"]
            if google_docs_state.get("status") == "connected":
                stage3_toolsets.append("maton-google-docs")
            context = _build_stage_context(
                lane_title=lane_title,
                lane_slug=lane_slug,
                topic_title=topic_title,
                topic_slug=topic_slug,
                request=request,
                paths=paths,
                authority=authority,
                stage_name=stage_name,
                stage_inputs=stage_inputs,
                google_docs_state=google_docs_state,
            )
            result = _delegate_single(goal=goal, context=context, toolsets=stage3_toolsets, max_iterations=30, parent_agent=parent_agent)
            if result.get("error"):
                _mark_stage_failed(state, stage_name, state_file, result["error"])
                msg = _build_blocker_message(topic_title, result["error"], state_file)
                _send_telegram(origin_target, msg, context="research_lane_launch blocker")
                if deliver_target and not _same_telegram_target(origin_target, deliver_target):
                    _send_telegram(deliver_target, msg, context="research_lane_launch blocker deliver", target_label=deliver)
                return {"status": "blocked", "blocker": result["error"], "state_file": str(state_file)}
            payload = _stage_result_payload(result)
            verified, verification_error, artifacts, primary_artifact = _verify_stage3_artifacts(paths, payload)
            if not verified:
                _mark_stage_failed(state, stage_name, state_file, verification_error)
                msg = _build_blocker_message(topic_title, verification_error, state_file)
                _send_telegram(origin_target, msg, context="research_lane_launch blocker")
                if deliver_target and not _same_telegram_target(origin_target, deliver_target):
                    _send_telegram(deliver_target, msg, context="research_lane_launch blocker deliver", target_label=deliver)
                return {"status": "blocked", "blocker": verification_error, "state_file": str(state_file)}
            extras = {
                "google_doc_status": payload.get("google_doc_status") or google_docs_state.get("status"),
            }
            if payload.get("google_doc_url"):
                extras["google_doc_url"] = payload.get("google_doc_url")
            _mark_stage_completed(state, stage_name, state_file, artifact=primary_artifact or None, artifacts=artifacts or None, extras=extras)
            final_message = _build_final_message(topic_title, payload, state_file)
            # Only deliver to the research topic — origin doesn't need the message
            if deliver_target:
                _send_telegram(deliver_target, final_message, context="research_lane_launch deliver", target_label=deliver)
            elif origin_target:
                _send_telegram(origin_target, final_message, context="research_lane_launch completion")
            return {
                "status": "completed",
                "message": final_message,
                "state_file": str(state_file),
                "google_doc_url": payload.get("google_doc_url"),
            }

    msg = _build_blocker_message(topic_title, "Pipeline exited unexpectedly before Stage 3 completion.", state_file)
    _send_telegram(origin_target, msg, context="research_lane_launch blocker")
    if deliver_target and not _same_telegram_target(origin_target, deliver_target):
        _send_telegram(deliver_target, msg, context="research_lane_launch blocker deliver", target_label=deliver)
    return {"status": "blocked", "blocker": "unexpected pipeline exit", "state_file": str(state_file)}


def _launch_runner(
    *,
    lane_title: str,
    lane_slug: str,
    topic_title: str,
    topic_slug: str,
    request: str,
    deliver: Optional[str],
    playbook_slug: str,
    state: Dict[str, Any],
    state_file: Path,
    paths: Dict[str, Path],
    parent_agent,
) -> None:
    from tools.delegate_tool import _resolve_agent_activity_target

    activity_target = _resolve_agent_activity_target()
    if activity_target:
        _send_telegram(
            activity_target,
            f"Research lane launch: {lane_slug}/{topic_slug}",
            context="research_lane_launch activity start",
            target_label="Agent Activity",
        )

    deliver_target = None
    if deliver:
        from tools.delegate_tool import _parse_deliver_target
        deliver_target = _parse_deliver_target(deliver)
    origin_target = _resolve_origin_target()

    result = _run_lane_pipeline(
        lane_title=lane_title,
        lane_slug=lane_slug,
        topic_title=topic_title,
        topic_slug=topic_slug,
        request=request,
        deliver=deliver,
        playbook_slug=playbook_slug,
        state=state,
        state_file=state_file,
        paths=paths,
        parent_agent=parent_agent,
        origin_target=origin_target,
        deliver_target=deliver_target,
    )

    if activity_target:
        status = result.get("status", "unknown")
        text = f"Research lane {status}: {lane_slug}/{topic_slug}"
        _send_telegram(
            activity_target,
            text,
            context="research_lane_launch activity done",
            target_label="Agent Activity",
        )


def launch_research_lane(
    *,
    lane: str,
    topic: Optional[str] = None,
    request: Optional[str] = None,
    playbook_slug: Optional[str] = None,
    deliver: Optional[str] = None,
    create_playbook_if_missing: bool = True,
    resume_existing: bool = True,
    background: bool = True,
    parent_agent=None,
) -> str:
    if parent_agent is None:
        return json.dumps({"error": "launch_research_lane requires a parent agent context."})
    if not (lane or "").strip():
        return json.dumps({"error": "lane is required."})
    if not (topic or "").strip():
        # Default to fundamentals for new lanes
        topic = f"{lane.strip()} Fundamentals"

    lane_slug = _slugify(lane)
    topic_slug = _slugify(topic)
    lane_title = lane.strip()
    topic_title = topic.strip()
    request_text = (request or topic or lane).strip()

    playbook_path, resolved_playbook_slug = _resolve_playbook(lane_slug, playbook_slug)
    ensured = _ensure_lane_assets(
        lane_title=lane_title,
        lane_slug=lane_slug,
        topic_title=topic_title,
        topic_slug=topic_slug,
        request=request_text,
        deliver=deliver,
        playbook_path=playbook_path,
        playbook_slug=resolved_playbook_slug,
        create_playbook_if_missing=create_playbook_if_missing,
    )
    paths = ensured["paths"]
    state_file = paths["state_file"]
    # Pick up the deliver target — may have been updated by auto-created Telegram topic
    if ensured.get("deliver"):
        deliver = ensured["deliver"]

    state = _load_json(state_file)
    # Start fresh if: no state, resume disabled, OR the old run is fully completed
    # (a completed run means Dax wants a NEW research topic, not a replay)
    old_run_completed = (
        state
        and all(
            (state.get("stages") or {}).get(s, {}).get("status") == "completed"
            for s in STAGE_SEQUENCE
        )
    )
    # Also start fresh if the topic changed from the previous run
    topic_changed = state and state.get("topic") != topic_slug
    if not state or not resume_existing or old_run_completed or topic_changed:
        state = _default_state(
            lane_title=lane_title,
            lane_slug=lane_slug,
            topic_title=topic_title,
            topic_slug=topic_slug,
            playbook_slug=resolved_playbook_slug,
            request=request_text,
            deliver=deliver,
            paths=paths,
            generated_playbook=ensured["generated_playbook"],
            drive_status=ensured["drive_status"],
        )
        _save_state(state, state_file)
    else:
        state.setdefault("lane_launch", {}).update(
            {
                "status": "completed",
                "quiet_mode": True,
                "playbook_path": str(paths["playbook"]),
                "lane_root": str(paths["lane_root"]),
                "topic_dir": str(paths["topic_dir"]),
                "google_drive": ensured["drive_status"],
            }
        )
        _save_state(state, state_file)

    if background:
        thread = threading.Thread(
            target=_launch_runner,
            kwargs={
                "lane_title": lane_title,
                "lane_slug": lane_slug,
                "topic_title": topic_title,
                "topic_slug": topic_slug,
                "request": request_text,
                "deliver": deliver,
                "playbook_slug": resolved_playbook_slug,
                "state": state,
                "state_file": state_file,
                "paths": paths,
                "parent_agent": parent_agent,
            },
            daemon=True,
            name=f"research-lane-{lane_slug}-{topic_slug}",
        )
        thread.start()
        return json.dumps(
            {
                "status": "spawned",
                "lane": lane_slug,
                "topic": topic_slug,
                "playbook": str(playbook_path),
                "topic_dir": str(paths["topic_dir"]),
                "state_file": str(state_file),
                "deliver": deliver or "origin",
                "message": f"Research lane launched for {topic_title}. I'll report back when it's done.",
                "playbook_generated": ensured["generated_playbook"],
                "index_template_generated": ensured["generated_index_template"],
                "google_drive_status": ensured["drive_status"].get("status"),
                "telegram_topic_created": ensured.get("telegram_topic") is not None,
                "telegram_topic": ensured.get("telegram_topic"),
            },
            ensure_ascii=False,
        )

    if deliver:
        from tools.delegate_tool import _parse_deliver_target
        deliver_target = _parse_deliver_target(deliver)
    else:
        deliver_target = None

    result = _run_lane_pipeline(
        lane_title=lane_title,
        lane_slug=lane_slug,
        topic_title=topic_title,
        topic_slug=topic_slug,
        request=request_text,
        deliver=deliver,
        playbook_slug=resolved_playbook_slug,
        state=state,
        state_file=state_file,
        paths=paths,
        parent_agent=parent_agent,
        origin_target=_resolve_origin_target(),
        deliver_target=deliver_target,
    )
    result.setdefault("lane", lane_slug)
    result.setdefault("topic", topic_slug)
    result.setdefault("playbook", str(playbook_path))
    result.setdefault("state_file", str(state_file))
    return json.dumps(result, ensure_ascii=False)


LAUNCH_RESEARCH_LANE_SCHEMA = {
    "name": "launch_research_lane",
    "description": (
        "Launch or continue research in a lane. Handles everything automatically: "
        "creates the playbook, Telegram topic, Obsidian folder, and Google Drive folder "
        "if they don't exist, then runs the 3-stage research pipeline in the background.\n\n"
        "TWO MODES:\n"
        "1. NEW LANE: Dax says 'I want to learn about X' — provide lane name. "
        "The tool creates all setup and runs the first foundational research.\n"
        "2. GO DEEPER: Dax says 'do more research on X' or 'go deeper on X' — "
        "provide the lane name and pick the most logical next topic yourself based on "
        "what's already been researched. Check the lane's knowledge-brief.md and existing "
        "docs to decide what gap to fill next. Do NOT ask Dax follow-up questions unless "
        "he explicitly asked for something specific.\n\n"
        "If Dax specifies a particular topic or question, use that as the topic. "
        "If he just says 'go deeper' or 'do more research', YOU decide the topic.\n\n"
        "QUIET UX — THIS IS CRITICAL: After calling this tool, reply to Dax with ONLY "
        "'Got it, spawning now.' — nothing else. No explanation of what you're doing, "
        "no breakdown of the topic, no status updates. Just 'Got it, spawning now.' "
        "The Google Doc link will be delivered to the research topic automatically when done."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "lane": {
                "type": "string",
                "description": "Research lane/domain name, e.g. 'SEO', 'GBP', 'Website Capabilities'.",
            },
            "topic": {
                "type": "string",
                "description": (
                    "Specific research topic. If Dax specified one, use it. "
                    "If Dax just said 'go deeper' or 'do more research', YOU pick "
                    "the most logical next topic based on existing lane docs. "
                    "For a brand new lane, use '{Lane} Fundamentals'."
                ),
            },
            "request": {
                "type": "string",
                "description": "The actual research ask/scope. Defaults to the topic if omitted.",
            },
            "playbook_slug": {
                "type": "string",
                "description": "Optional explicit playbook slug (without .md), e.g. 'seo-research'.",
            },
            "deliver": {
                "type": "string",
                "description": "Optional delivery target. Usually auto-resolved from the lane's Telegram topic.",
            },
            "create_playbook_if_missing": {
                "type": "boolean",
                "description": "Generate playbook and index template if missing.",
                "default": True,
            },
            "resume_existing": {
                "type": "boolean",
                "description": "Resume interrupted pipeline instead of starting fresh.",
                "default": True,
            },
            "background": {
                "type": "boolean",
                "description": "Run in background thread.",
                "default": True,
            },
        },
        "required": ["lane"],
    },
}


registry.register(
    name="launch_research_lane",
    toolset="delegation",
    schema=LAUNCH_RESEARCH_LANE_SCHEMA,
    handler=lambda args, **kw: launch_research_lane(
        lane=args.get("lane", ""),
        topic=args.get("topic", ""),
        request=args.get("request"),
        playbook_slug=args.get("playbook_slug"),
        deliver=args.get("deliver"),
        create_playbook_if_missing=args.get("create_playbook_if_missing", True),
        resume_existing=args.get("resume_existing", True),
        background=args.get("background", True),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=lambda: True,
    emoji="🚀",
)

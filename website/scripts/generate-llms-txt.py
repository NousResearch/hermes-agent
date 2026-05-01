#!/usr/bin/env python3
"""Generate llms.txt and llms-full.txt for the Hermes docs site.

Outputs:
  website/static/llms.txt        — short curated index of the docs, one link per page,
                                    grouped by section. Conforms to https://llmstxt.org.
  website/static/llms-full.txt   — every `.md` file under `website/docs/` concatenated,
                                    with `# <title>` headings and `<!-- source: … -->`
                                    comments separating files.

Both publish at:
  https://hermes-agent.nousresearch.com/docs/llms.txt
  https://hermes-agent.nousresearch.com/docs/llms-full.txt

The `/docs/` prefix is not a mistake — Docusaurus serves `website/static/`
at the `docs/` base path. Clients and IDE plugins that probe the classic
`/llms.txt` root will miss these. Document the canonical URLs in the docs
index and in the repo README.

Called from `website/scripts/prebuild.mjs` on every `npm run start` /
`npm run build` so the output stays in sync with the docs tree.
"""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WEBSITE = SCRIPT_DIR.parent
DOCS = WEBSITE / "docs"
STATIC = WEBSITE / "static"

SITE_BASE = "https://hermes-agent.nousresearch.com/docs"

# Curated sections for llms.txt — mirrors the product story, not the filesystem.
# Each entry: (docs-relative path without .md, display title, optional short desc).
# `None` desc → pulled from frontmatter `description:` field.
SECTIONS: list[tuple[str, list[tuple[str, str, str | None]]]] = [
    ("Getting Started", [
        ("getting-started/installation", "Installation", None),
        ("getting-started/quickstart", "Quickstart", None),
        ("getting-started/learning-path", "Learning Path", None),
        ("getting-started/updating", "Updating", None),
        ("getting-started/termux", "Termux (Android)", None),
        ("getting-started/nix-setup", "Nix Setup", None),
    ]),
    ("Using Hermes", [
        ("user-guide/cli", "CLI", None),
        ("user-guide/tui", "TUI (Ink terminal UI)", None),
        ("user-guide/configuration", "Configuration", None),
        ("user-guide/configuring-models", "Configuring Models", None),
        ("user-guide/sessions", "Sessions", None),
        ("user-guide/profiles", "Profiles", None),
        ("user-guide/git-worktrees", "Git Worktrees", None),
        ("user-guide/docker", "Docker Backend", None),
        ("user-guide/security", "Security", None),
        ("user-guide/checkpoints-and-rollback", "Checkpoints & Rollback", None),
    ]),
    ("Core Features", [
        ("user-guide/features/overview", "Features Overview", None),
        ("user-guide/features/tools", "Tools", None),
        ("user-guide/features/skills", "Skills System", None),
        ("user-guide/features/curator", "Curator", None),
        ("user-guide/features/memory", "Memory", None),
        ("user-guide/features/memory-providers", "Memory Providers", None),
        ("user-guide/features/context-files", "Context Files", None),
        ("user-guide/features/context-references", "Context References", None),
        ("user-guide/features/personality", "Personality & SOUL.md", None),
        ("user-guide/features/plugins", "Plugins", None),
        ("user-guide/features/built-in-plugins", "Built-in Plugins", None),
    ]),
    ("Automation", [
        ("user-guide/features/cron", "Cron Jobs", None),
        ("user-guide/features/delegation", "Delegation", None),
        ("user-guide/features/kanban", "Kanban Multi-Agent", None),
        ("user-guide/features/kanban-tutorial", "Kanban Tutorial", None),
        ("user-guide/features/code-execution", "Code Execution", None),
        ("user-guide/features/hooks", "Hooks", None),
        ("user-guide/features/batch-processing", "Batch Processing", None),
    ]),
    ("Media & Web", [
        ("user-guide/features/voice-mode", "Voice Mode", None),
        ("user-guide/features/browser", "Browser", None),
        ("user-guide/features/vision", "Vision", None),
        ("user-guide/features/image-generation", "Image Generation", None),
        ("user-guide/features/tts", "Text-to-Speech", None),
    ]),
    ("Messaging Platforms", [
        ("user-guide/messaging/index", "Overview", None),
        ("user-guide/messaging/telegram", "Telegram", None),
        ("user-guide/messaging/discord", "Discord", None),
        ("user-guide/messaging/slack", "Slack", None),
        ("user-guide/messaging/whatsapp", "WhatsApp", None),
        ("user-guide/messaging/signal", "Signal", None),
        ("user-guide/messaging/email", "Email", None),
        ("user-guide/messaging/sms", "SMS", None),
        ("user-guide/messaging/matrix", "Matrix", None),
        ("user-guide/messaging/mattermost", "Mattermost", None),
        ("user-guide/messaging/homeassistant", "Home Assistant", None),
        ("user-guide/messaging/webhooks", "Webhooks", None),
    ]),
    ("Integrations", [
        ("integrations/index", "Integrations Overview", None),
        ("integrations/providers", "Providers", None),
        ("user-guide/features/mcp", "MCP (Model Context Protocol)", None),
        ("user-guide/features/acp", "ACP (Agent Context Protocol)", None),
        ("user-guide/features/api-server", "API Server", None),
        ("user-guide/features/honcho", "Honcho Memory", None),
        ("user-guide/features/provider-routing", "Provider Routing", None),
        ("user-guide/features/fallback-providers", "Fallback Providers", None),
        ("user-guide/features/credential-pools", "Credential Pools", None),
    ]),
    ("Guides & Tutorials", [
        ("guides/tips", "Tips & Best Practices", None),
        ("guides/local-llm-on-mac", "Local LLMs on Mac", None),
        ("guides/daily-briefing-bot", "Daily Briefing Bot", None),
        ("guides/team-telegram-assistant", "Team Telegram Assistant", None),
        ("guides/python-library", "Use Hermes as a Python Library", None),
        ("guides/use-mcp-with-hermes", "Use MCP with Hermes", None),
        ("guides/use-voice-mode-with-hermes", "Use Voice Mode with Hermes", None),
        ("guides/use-soul-with-hermes", "Use SOUL.md with Hermes", None),
        ("guides/build-a-hermes-plugin", "Build a Hermes Plugin", None),
        ("guides/automate-with-cron", "Automate with Cron", None),
        ("guides/work-with-skills", "Work with Skills", None),
        ("guides/delegation-patterns", "Delegation Patterns", None),
        ("guides/github-pr-review-agent", "GitHub PR Review Agent", None),
    ]),
    ("Developer Guide", [
        ("developer-guide/contributing", "Contributing", None),
        ("developer-guide/architecture", "Architecture", None),
        ("developer-guide/agent-loop", "Agent Loop", None),
        ("developer-guide/prompt-assembly", "Prompt Assembly", None),
        ("developer-guide/context-compression-and-caching", "Context Compression & Caching", None),
        ("developer-guide/gateway-internals", "Gateway Internals", None),
        ("developer-guide/session-storage", "Session Storage", None),
        ("developer-guide/provider-runtime", "Provider Runtime", None),
        ("developer-guide/adding-tools", "Adding Tools", None),
        ("developer-guide/adding-providers", "Adding Providers", None),
        ("developer-guide/adding-platform-adapters", "Adding Platform Adapters", None),
        ("developer-guide/creating-skills", "Creating Skills", None),
        ("developer-guide/extending-the-cli", "Extending the CLI", None),
    ]),
    ("Reference", [
        ("reference/cli-commands", "CLI Commands", None),
        ("reference/slash-commands", "Slash Commands", None),
        ("reference/profile-commands", "Profile Commands", None),
        ("reference/environment-variables", "Environment Variables", None),
        ("reference/tools-reference", "Tools Reference", None),
        ("reference/toolsets-reference", "Toolsets Reference", None),
        ("reference/mcp-config-reference", "MCP Config Reference", None),
        ("reference/model-catalog", "Model Catalog", None),
        ("reference/skills-catalog", "Bundled Skills Catalog", "Table of all ~90 skills bundled with Hermes"),
        ("reference/optional-skills-catalog", "Optional Skills Catalog", "Table of ~60 additional installable skills"),
        ("reference/faq", "FAQ & Troubleshooting", None),
    ]),
]


FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
DESC_RE = re.compile(r"^description:\s*[\"'](.+?)[\"']\s*$", re.MULTILINE)
TITLE_RE = re.compile(r"^title:\s*[\"'](.+?)[\"']\s*$", re.MULTILINE)


def read_frontmatter(path: Path) -> tuple[dict[str, str], str]:
    """Return ({title, description}, body-markdown) for a doc file."""
    text = path.read_text(encoding="utf-8")
    m = FRONTMATTER_RE.match(text)
    meta: dict[str, str] = {}
    body = text
    if m:
        fm = m.group(1)
        body = text[m.end():]
        dm = DESC_RE.search(fm)
        if dm:
            meta["description"] = dm.group(1)
        tm = TITLE_RE.search(fm)
        if tm:
            meta["title"] = tm.group(1)
    return meta, body


def resolve_desc(slug: str, provided: str | None) -> str:
    """Resolve short description for llms.txt entry."""
    if provided:
        return provided
    path = DOCS / f"{slug}.md"
    if not path.exists():
        path = DOCS / slug / "index.md"
    if not path.exists():
        return ""
    meta, _ = read_frontmatter(path)
    return meta.get("description", "")


def emit_llms_index() -> str:
    """Build the short llms.txt index."""
    lines: list[str] = []
    lines.append("# Hermes Agent")
    lines.append("")
    lines.append(
        "> The self-improving AI agent built by Nous Research. A terminal-native "
        "autonomous coding and task agent with persistent memory, agent-created skills, "
        "and a messaging gateway that lives on 15+ platforms (Telegram, Discord, Slack, "
        "SMS, Matrix, ...). Runs on local, Docker, SSH, Daytona, Modal, or Singularity "
        "backends. Works with Nous Portal, OpenRouter, OpenAI, Anthropic, Google, or any "
        "OpenAI-compatible endpoint."
    )
    lines.append("")
    lines.append(
        "Install: `curl -fsSL https://raw.githubusercontent.com/NousResearch/"
        "hermes-agent/main/scripts/install.sh | bash`  "
        "(Linux, macOS, WSL2, Termux)"
    )
    lines.append("")
    lines.append("Repo: https://github.com/NousResearch/hermes-agent")
    lines.append("")

    for section, items in SECTIONS:
        lines.append(f"## {section}")
        lines.append("")
        for slug, title, desc_override in items:
            desc = resolve_desc(slug, desc_override)
            url = f"{SITE_BASE}/{slug}"
            if desc:
                lines.append(f"- [{title}]({url}): {desc}")
            else:
                lines.append(f"- [{title}]({url})")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def emit_llms_full() -> str:
    """Concatenate every doc under website/docs/ into a single markdown file.

    Order: mirrors the curated SECTIONS list first (so the most important
    pages are front-loaded for agents that truncate on token budget), then
    appends any remaining .md files sorted by path.
    """
    seen: set[Path] = set()
    chunks: list[str] = [
        "# Hermes Agent — Full Documentation\n",
        (
            "This file is the entire Hermes Agent documentation concatenated for LLM "
            "context ingestion. Section order reflects docs-site navigation: Getting "
            "Started, Using Hermes, Features, Messaging, Integrations, Guides, "
            "Developer Guide, Reference, then everything else.\n"
        ),
        "Canonical site: https://hermes-agent.nousresearch.com/docs\n",
        "Short index: https://hermes-agent.nousresearch.com/docs/llms.txt\n",
        "\n---\n\n",
    ]

    def emit_file(rel: str) -> None:
        path = DOCS / f"{rel}.md"
        if not path.exists():
            path = DOCS / rel / "index.md"
        if not path.exists() or path in seen:
            return
        seen.add(path)
        meta, body = read_frontmatter(path)
        title = meta.get("title") or rel
        chunks.append(f"<!-- source: website/docs/{path.relative_to(DOCS)} -->\n")
        chunks.append(f"# {title}\n\n")
        chunks.append(body.rstrip() + "\n\n---\n\n")

    # Curated order first
    for _, items in SECTIONS:
        for slug, _t, _d in items:
            emit_file(slug)

    # Everything else (sorted, skipping already emitted and auto-gen skill pages
    # — those are covered by the two catalog reference pages, emitting every
    # individual skill would add ~1.4 MB of largely duplicative material).
    for path in sorted(DOCS.rglob("*.md")):
        if path in seen:
            continue
        rel = path.relative_to(DOCS)
        parts = rel.parts
        if len(parts) >= 3 and parts[0] == "user-guide" and parts[1] == "skills" \
                and parts[2] in ("bundled", "optional"):
            continue
        seen.add(path)
        meta, body = read_frontmatter(path)
        title = meta.get("title") or str(rel)
        chunks.append(f"<!-- source: website/docs/{rel} -->\n")
        chunks.append(f"# {title}\n\n")
        chunks.append(body.rstrip() + "\n\n---\n\n")

    return "".join(chunks).rstrip() + "\n"


def main() -> None:
    STATIC.mkdir(exist_ok=True)
    index = emit_llms_index()
    full = emit_llms_full()
    (STATIC / "llms.txt").write_text(index, encoding="utf-8")
    (STATIC / "llms-full.txt").write_text(full, encoding="utf-8")
    print(f"Wrote {STATIC / 'llms.txt'} ({len(index):,} bytes)")
    print(f"Wrote {STATIC / 'llms-full.txt'} ({len(full):,} bytes)")


if __name__ == "__main__":
    main()

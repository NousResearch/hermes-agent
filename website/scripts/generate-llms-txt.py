#!/usr/bin/env python3
"""Generate llms.txt and llms-full.txt for the Hermes docs site.

Outputs:
  website/static/llms.txt        — index of the docs, one link per page, grouped by
                                    section. Conforms to https://llmstxt.org.
  website/static/llms-full.txt   — every doc under `website/docs/` concatenated,
                                    with `# <title>` headings and `<!-- source: … -->`
                                    comments separating files.

Both are driven by `iter_docs()`, which walks the docs tree. `SECTIONS` below
curates *order and grouping*, never membership: a page nobody curated still
gets indexed, under the section its path belongs to. That distinction is the
reason this file was rewritten — when the section list also decided membership,
it silently drifted to 53% coverage, and Bot Mode, the desktop app, computer
use, web search, and 22 messaging platforms were absent from the index every
LLM reads to learn what Hermes does.

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

# The product story: which pages lead, and in what order. Everything not named
# here is still indexed — ABSORB decides where it lands — so this list is safe
# to leave alone as the docs grow, and worth editing only to promote a page.
# Each entry: (docs-relative path without extension, display title, optional
# short desc). `None` desc → pulled from frontmatter `description:` field.
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
        ("user-guide/features/goals", "Persistent Goals", None),
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
        ("user-guide/messaging", "Overview", None),
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
        ("integrations", "Integrations Overview", None),
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
        ("guides/automate-with-cron", "Automate with Cron", None),
        ("guides/work-with-skills", "Work with Skills", None),
        ("guides/delegation-patterns", "Delegation Patterns", None),
        ("guides/github-pr-review-agent", "GitHub PR Review Agent", None),
        ("guides/all-gods-must-die", "All Gods Must Die: Adversarially Verified Transformation", "Unified Hermes doctrine: god-file decomposition, documentation parity, cyber defense, interlock, skill refactoring, and attribution"),
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


DOC_EXTS = (".md", ".mdx")

# Per-skill pages are generated from the skill tree and summarized by the two
# catalog reference pages. Listing ~195 of them would bury the product docs in
# the index and add ~1.4 MB of duplicative material to llms-full.txt.
SKILL_CATALOG = ("user-guide/skills/bundled", "user-guide/skills/optional")

# Where a page nobody curated goes. First match wins, so a narrower prefix must
# precede the tree containing it. Anything matching nothing lands in
# MISC_SECTION — no path can drop a page out of the index.
ABSORB: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Getting Started", ("getting-started",)),
    ("Messaging Platforms", ("user-guide/messaging",)),
    ("Core Features", ("user-guide/features",)),
    ("Using Hermes", ("user-guide",)),
    ("Integrations", ("integrations",)),
    ("Guides & Tutorials", ("guides",)),
    ("Developer Guide", ("developer-guide",)),
    ("Reference", ("reference",)),
)
MISC_SECTION = "More"

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
DESC_RE = re.compile(r"^description:\s*(.+?)\s*$", re.MULTILINE)
TITLE_RE = re.compile(r"^title:\s*(.+?)\s*$", re.MULTILINE)
H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
# MDX pages open with component imports — markup plumbing, not prose.
MDX_IMPORT_RE = re.compile(r"^(?:import|export)\s.*$\n?", re.MULTILINE)


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def read_frontmatter(path: Path) -> tuple[dict[str, str], str]:
    """Return ({title, description}, body-markdown) for a doc file."""
    text = path.read_text(encoding="utf-8")
    m = FRONTMATTER_RE.match(text)
    meta: dict[str, str] = {}
    body = text
    if m:
        fm = m.group(1)
        body = text[m.end():]
        for key, pattern in (("description", DESC_RE), ("title", TITLE_RE)):
            found = pattern.search(fm)
            if found:
                meta[key] = _unquote(found.group(1))
    if path.suffix == ".mdx":
        body = MDX_IMPORT_RE.sub("", body)
    return meta, body


def slug_for(path: Path) -> str:
    """URL slug for a page: `user-guide/messaging/index.md` → `user-guide/messaging`."""
    rel = path.relative_to(DOCS).with_suffix("")
    if rel.name == "index":
        rel = rel.parent
    return "" if str(rel) == "." else str(rel)


def doc_path(slug: str) -> Path | None:
    """The file backing a slug, whether it's a page or a section landing page."""
    for ext in DOC_EXTS:
        for candidate in (DOCS / f"{slug}{ext}", DOCS / slug / f"index{ext}"):
            if candidate.exists():
                return candidate
    return None


def iter_docs() -> list[str]:
    """Every indexable page, as a slug — the one enumeration of the docs tree.

    llms.txt lists exactly these and llms-full.txt emits exactly these, so a
    page on disk cannot be missing from either output.
    """
    slugs = set()
    for ext in DOC_EXTS:
        for path in DOCS.rglob(f"*{ext}"):
            slug = slug_for(path)
            # The docs landing page is this index's subject, not an entry in it.
            if slug and not slug.startswith(SKILL_CATALOG):
                slugs.add(slug)
    return sorted(slugs)


def section_for(slug: str) -> str:
    for section, prefixes in ABSORB:
        if any(slug == prefix or slug.startswith(f"{prefix}/") for prefix in prefixes):
            return section
    return MISC_SECTION


def resolve_meta(slug: str) -> tuple[str, str]:
    """(title, description) for a page, falling back to its H1 then its slug."""
    path = doc_path(slug)
    if path is None:
        return slug, ""
    meta, body = read_frontmatter(path)
    title = meta.get("title")
    if not title:
        h1 = H1_RE.search(body)
        title = h1.group(1) if h1 else slug.rsplit("/", 1)[-1].replace("-", " ").title()
    return title, meta.get("description", "")


def resolve_desc(slug: str, provided: str | None) -> str:
    """Resolve short description for llms.txt entry."""
    return provided or resolve_meta(slug)[1]


def _entry(slug: str, title: str, desc: str) -> str:
    url = f"{SITE_BASE}/{slug}"
    return f"- [{title}]({url}): {desc}" if desc else f"- [{title}]({url})"


def emit_llms_index() -> str:
    """Build the llms.txt index: curated pages lead a section, the rest follow."""
    curated = {slug for _section, items in SECTIONS for slug, _title, _desc in items}
    absorbed: dict[str, list[str]] = {}
    for slug in iter_docs():
        if slug not in curated:
            absorbed.setdefault(section_for(slug), []).append(slug)

    lines: list[str] = []
    lines.append("# Hermes Agent")
    lines.append("")
    lines.append(
        "> The self-improving AI agent built by Nous Research. A terminal-native "
        "autonomous coding and task agent with persistent memory, agent-created skills, "
        "and a messaging gateway that lives on 21+ messaging platforms — 19 native to "
        "the gateway plus IRC and Microsoft Teams via plugins (Telegram, Discord, Slack, "
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
            lines.append(_entry(slug, title, resolve_desc(slug, desc_override)))
        for slug in absorbed.pop(section, []):
            lines.append(_entry(slug, *resolve_meta(slug)))
        lines.append("")

    # Only MISC_SECTION can survive the pops — a page whose path matched no
    # section still has to appear somewhere.
    for section, slugs in absorbed.items():
        lines.append(f"## {section}")
        lines.append("")
        for slug in slugs:
            lines.append(_entry(slug, *resolve_meta(slug)))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def emit_llms_full() -> str:
    """Concatenate every doc under website/docs/ into a single markdown file."""
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

    def emit_file(slug: str) -> None:
        path = doc_path(slug)
        if path is None or path in seen:
            return
        seen.add(path)
        title, _desc = resolve_meta(slug)
        _meta, body = read_frontmatter(path)
        chunks.append(f"<!-- source: website/docs/{path.relative_to(DOCS)} -->\n")
        chunks.append(f"# {title}\n\n")
        chunks.append(body.rstrip() + "\n\n---\n\n")

    # Curated order first, so a reader truncating on token budget keeps the
    # pages that matter most; then everything else the docs tree holds.
    for _section, items in SECTIONS:
        for slug, _title, _desc in items:
            emit_file(slug)
    for slug in iter_docs():
        emit_file(slug)

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

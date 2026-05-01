#!/usr/bin/env python3
"""Generate per-skill Docusaurus pages from skills/ and optional-skills/ SKILL.md files.

Each skill gets website/docs/user-guide/skills/<source>/<category>/<skill-name>.md
where <source> is "bundled" or "optional".

Also regenerates:
- website/docs/reference/skills-catalog.md
- website/docs/reference/optional-skills-catalog.md
(so their table rows link to the new dedicated pages)

Sidebar is updated to nest all per-skill pages under Skills → Bundled / Optional.
"""

from __future__ import annotations
import re
import sys
from collections import defaultdict
from pathlib import Path
from textwrap import dedent
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
DOCS = REPO / "website" / "docs"
SKILLS_PAGES = DOCS / "user-guide" / "skills"

SKILL_SOURCES = [
    ("bundled", REPO / "skills"),
    ("optional", REPO / "optional-skills"),
]

# Pages the user had previously hand-written in user-guide/skills/.
# We leave these alone (they get first-class sidebar treatment separately).
HAND_WRITTEN = {"godmode.md", "google-workspace.md"}


_FENCE_RE = re.compile(r"^(?P<indent>\s*)(?P<fence>```+|~~~+)", re.MULTILINE)

# Unicode box-drawing characters. If a generated fenced code block contains any
# of these, wrap it in `<!-- ascii-guard-ignore -->` so the docs-site-checks
# lint (which scans inside code fences) can't reject the page for a skill's
# own ASCII diagram. Skill authors shouldn't need to remember to add the
# ignore markers in every SKILL.md — the generator handles it defensively.
_BOX_DRAWING_CHARS = frozenset("┌┐└┘─│═║╔╗╚╝╠╣╦╩╬├┤┬┴┼╭╮╯╰▶◀▲▼")


def _wrap_ascii_art_code_blocks(code_segment: str) -> str:
    """Wrap a fenced code segment in ascii-guard-ignore markers if it contains
    box-drawing characters. No-op otherwise, so plain bash/python code blocks
    stay uncluttered.

    Already-wrapped segments (the SKILL.md source added its own markers) are
    left alone — double-wrapping is harmless but we'd rather keep the output
    clean.
    """
    if not any(ch in _BOX_DRAWING_CHARS for ch in code_segment):
        return code_segment
    return (
        "<!-- ascii-guard-ignore -->\n"
        f"{code_segment}\n"
        "<!-- ascii-guard-ignore-end -->"
    )


def mdx_escape_body(body: str) -> str:
    """Escape MDX-dangerous characters in markdown body, leaving fenced code blocks alone.

    Outside fenced code blocks:
      * `{` -> `&#123;`  (prevents MDX from parsing JSX expressions)
      * `}` -> `&#125;`
      * `<tag>` for bare tags that aren't whitelisted HTML get HTML-entity-escaped
      * inline `` `code` `` content is preserved (backticks handled naturally)
    Inside fenced code blocks: untouched.

    We also preserve `<br>`, `<br/>`, `<img ...>`, `<a ...>`, and a handful of
    other markup-safe tags because Docusaurus/MDX accepts them as HTML.
    """
    # Split the body into segments by fenced code blocks, alternating
    # (text, code, text, code, ...). A line like ``` or ~~~ opens a fence;
    # a matching marker closes it.
    lines = body.split("\n")
    segments: list[tuple[str, str]] = []  # ("text"|"code", content)
    buf: list[str] = []
    mode = "text"
    fence_char: str | None = None
    fence_len = 0
    for line in lines:
        stripped = line.lstrip()
        if mode == "text":
            if stripped.startswith("```") or stripped.startswith("~~~"):
                # Opening fence
                if buf:
                    segments.append(("text", "\n".join(buf)))
                    buf = []
                buf.append(line)
                # Detect fence char + length
                m = re.match(r"(`{3,}|~{3,})", stripped)
                if m:
                    fence_char = m.group(1)[0]
                    fence_len = len(m.group(1))
                mode = "code"
            else:
                buf.append(line)
        else:  # code mode
            buf.append(line)
            if fence_char is not None and stripped.startswith(fence_char * fence_len):
                # Closing fence
                segments.append(("code", "\n".join(buf)))
                buf = []
                mode = "text"
                fence_char = None
                fence_len = 0
    if buf:
        segments.append((mode, "\n".join(buf)))

    def escape_text(text: str) -> str:
        # Walk inline-code runs (backticks) and leave them alone.
        # Pattern matches runs of backticks, then the matched content, then the
        # same number of backticks.
        out: list[str] = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "`":
                # Find the run of backticks
                j = i
                while j < len(text) and text[j] == "`":
                    j += 1
                run = text[i:j]
                # Find matching run
                end = text.find(run, j)
                if end == -1:
                    # No closing -- just keep as-is
                    out.append(text[i:])
                    i = len(text)
                    continue
                out.append(text[i : end + len(run)])
                i = end + len(run)
            else:
                # Escape MDX metacharacters
                if ch == "{":
                    out.append("&#123;")
                elif ch == "}":
                    out.append("&#125;")
                elif ch == "<":
                    # Preserve full HTML comments (e.g. ascii-guard ignore markers) — they
                    # are not HTML tags, so the tag regex below would escape the leading <.
                    if text[i:].startswith("<!--"):
                        end = text.find("-->", i)
                        if end != -1:
                            out.append(text[i : end + 3])
                            i = end + 3
                            continue
                    # Look ahead to see if this is a valid HTML-ish tag.
                    # If it looks like a tag name then alnum/-/_ chars, leave it.
                    # Otherwise escape.
                    m = re.match(
                        r"<(/?)([A-Za-z][A-Za-z0-9]*)([^<>]*)>",
                        text[i:],
                    )
                    if m:
                        tag = m.group(2).lower()
                        # Whitelist known-safe HTML tags
                        safe_tags = {
                            "br",
                            "hr",
                            "img",
                            "a",
                            "b",
                            "i",
                            "em",
                            "strong",
                            "code",
                            "kbd",
                            "sup",
                            "sub",
                            "span",
                            "div",
                            "p",
                            "ul",
                            "ol",
                            "li",
                            "table",
                            "thead",
                            "tbody",
                            "tr",
                            "td",
                            "th",
                            "details",
                            "summary",
                            "blockquote",
                            "pre",
                            "mark",
                            "small",
                            "u",
                            "s",
                            "del",
                            "ins",
                            "h1",
                            "h2",
                            "h3",
                            "h4",
                            "h5",
                            "h6",
                        }
                        if tag in safe_tags:
                            out.append(m.group(0))
                            i += len(m.group(0))
                            continue
                    # Escape the `<`
                    out.append("&lt;")
                else:
                    out.append(ch)
                i += 1
        return "".join(out)

    processed: list[str] = []
    for kind, content in segments:
        if kind == "code":
            processed.append(_wrap_ascii_art_code_blocks(content))
        else:
            processed.append(escape_text(content))
    return "\n".join(processed)


def rewrite_relative_links(body: str, meta: dict[str, Any]) -> str:
    """Rewrite references/foo.md style links in the SKILL.md body.

    The source SKILL.md lives in `skills/<...>` and references sibling files
    with paths like `references/foo.md` or `./templates/bar.md`. Those files
    are NOT copied into docs/, so we rewrite these to absolute GitHub URLs
    pointing to the file in the repo.
    """
    source_dir = "skills" if meta["source_kind"] == "bundled" else "optional-skills"
    base = f"https://github.com/NousResearch/hermes-agent/blob/main/{source_dir}/{meta['rel_path']}"

    def sub_link(m: re.Match) -> str:
        text = m.group(1)
        url = m.group(2).strip()
        # Skip URLs that already start with a scheme or //
        if re.match(r"^[a-z]+://", url) or url.startswith("#") or url.startswith("/"):
            return m.group(0)
        # Skip mailto
        if url.startswith("mailto:"):
            return m.group(0)
        # Strip leading ./
        url_clean = url[2:] if url.startswith("./") else url
        full = f"{base}/{url_clean}"
        return f"[{text}]({full})"

    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", sub_link, body)


def parse_skill_md(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"{path}: no frontmatter")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"{path}: malformed frontmatter")
    fm_text, body = parts[1], parts[2]
    try:
        fm = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: YAML error: {exc}") from exc
    return {"frontmatter": fm, "body": body.lstrip("\n")}


def sanitize_yaml_string(s: str) -> str:
    """Make a string safe to embed in a YAML double-quoted scalar."""
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    # Collapse newlines to spaces.
    s = re.sub(r"\s+", " ", s).strip()
    return s


def derive_skill_meta(skill_path: Path, source_dir: Path, source_kind: str) -> dict[str, Any]:
    """Extract category + skill slug from filesystem layout.

    skills/<cat>/<skill>/SKILL.md           -> cat=<cat>, slug=<skill>
    skills/<cat>/<sub>/<skill>/SKILL.md     -> cat=<cat>, sub=<sub>, slug=<skill>
    optional-skills/<cat>/<skill>/SKILL.md  -> cat=<cat>, slug=<skill>
    """
    rel = skill_path.parent.relative_to(source_dir)
    parts = rel.parts
    if len(parts) == 1:
        # Top-level skill (e.g. skills/dogfood/SKILL.md) -- rare
        category = parts[0]
        sub = None
        slug = parts[0]
    elif len(parts) == 2:
        category, slug = parts
        sub = None
    elif len(parts) == 3:
        category, sub, slug = parts
    else:
        raise ValueError(f"Unexpected skill layout: {skill_path}")
    return {
        "source_kind": source_kind,  # bundled | optional
        "category": category,
        "sub": sub,
        "slug": slug,
        "rel_path": str(rel),
    }


def page_id(meta: dict[str, Any]) -> str:
    """Stable slug used for filename + sidebar id."""
    if meta["sub"]:
        return f"{meta['category']}-{meta['sub']}-{meta['slug']}"
    return f"{meta['category']}-{meta['slug']}"


def page_output_path(meta: dict[str, Any]) -> Path:
    return (
        SKILLS_PAGES
        / meta["source_kind"]
        / meta["category"]
        / f"{page_id(meta)}.md"
    )


def sidebar_doc_id(meta: dict[str, Any]) -> str:
    """Docusaurus sidebar id, relative to docs/."""
    return f"user-guide/skills/{meta['source_kind']}/{meta['category']}/{page_id(meta)}"


def render_skill_page(
    meta: dict[str, Any],
    fm: dict[str, Any],
    body: str,
    skill_index: dict[str, dict[str, Any]] | None = None,
) -> str:
    name = fm.get("name", meta["slug"])
    description = fm.get("description", "").strip()
    short_desc = description.split(".")[0].strip() if description else name
    if len(short_desc) > 160:
        short_desc = short_desc[:157] + "..."

    title = f"{name}"
    # Heuristic nicer title from name
    display_name = name.replace("-", " ").replace("_", " ").title()

    hermes_meta = (fm.get("metadata") or {}).get("hermes") or {}
    tags = hermes_meta.get("tags") or []
    related = hermes_meta.get("related_skills") or []
    platforms = fm.get("platforms")
    version = fm.get("version")
    author = fm.get("author")
    license_ = fm.get("license")
    deps = fm.get("dependencies")

    # Build metadata info block
    info_rows: list[tuple[str, str]] = []
    if meta["source_kind"] == "bundled":
        info_rows.append(("Source", "Bundled (installed by default)"))
    else:
        info_rows.append(
            (
                "Source",
                "Optional — install with `hermes skills install official/"
                + meta["category"]
                + "/"
                + meta["slug"]
                + "`",
            )
        )
    source_dir = "skills" if meta["source_kind"] == "bundled" else "optional-skills"
    info_rows.append(("Path", f"`{source_dir}/{meta['rel_path']}`"))
    if version:
        info_rows.append(("Version", f"`{version}`"))
    if author:
        info_rows.append(("Author", str(author)))
    if license_:
        info_rows.append(("License", str(license_)))
    if deps:
        if isinstance(deps, list):
            deps_str = ", ".join(f"`{d}`" for d in deps) if deps else "None"
        else:
            deps_str = f"`{deps}`"
        info_rows.append(("Dependencies", deps_str))
    if platforms:
        if isinstance(platforms, list):
            plat_str = ", ".join(platforms)
        else:
            plat_str = str(platforms)
        info_rows.append(("Platforms", plat_str))
    if tags:
        info_rows.append(("Tags", ", ".join(f"`{t}`" for t in tags)))
    if related:
        # link to sibling pages when possible -- fall back to plain code
        link_parts = []
        for r in related:
            target_meta = None
            if skill_index is not None:
                target_meta = skill_index.get(r)
            if target_meta is not None:
                href = (
                    f"/docs/user-guide/skills/{target_meta['source_kind']}"
                    f"/{target_meta['category']}/{page_id(target_meta)}"
                )
                link_parts.append(f"[`{r}`]({href})")
            else:
                link_parts.append(f"`{r}`")
        info_rows.append(("Related skills", ", ".join(link_parts)))

    info_block = "\n".join(f"| {k} | {v} |" for k, v in info_rows)
    info_table = (
        "| | |\n|---|---|\n" + info_block
    )

    # Frontmatter for Docusaurus
    fm_title = sanitize_yaml_string(display_name + " — " + (short_desc or name))
    if len(fm_title) > 120:
        fm_title = sanitize_yaml_string(display_name)
    fm_desc = sanitize_yaml_string(short_desc or description or name)
    sidebar_label = sanitize_yaml_string(display_name)

    body_clean = mdx_escape_body(rewrite_relative_links(body.strip(), meta))

    # Guard against the first heading in body being `# Xxx Skill` which would
    # duplicate the page title -- Docusaurus handles this fine because the
    # frontmatter `title` drives the page header and TOC.

    return (
        "---\n"
        f'title: "{fm_title}"\n'
        f'sidebar_label: "{sidebar_label}"\n'
        f'description: "{fm_desc}"\n'
        "---\n"
        "\n"
        "{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}\n"
        "\n"
        f"# {display_name}\n"
        "\n"
        f"{mdx_escape_body(description)}\n"
        "\n"
        "## Skill metadata\n"
        "\n"
        f"{info_table}\n"
        "\n"
        "## Reference: full SKILL.md\n"
        "\n"
        ":::info\n"
        "The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.\n"
        ":::\n"
        "\n"
        f"{body_clean}\n"
    )


def discover_skills() -> list[tuple[dict[str, Any], dict[str, Any]]]:
    results: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for kind, source_dir in SKILL_SOURCES:
        for skill_md in sorted(source_dir.rglob("SKILL.md")):
            meta = derive_skill_meta(skill_md, source_dir, kind)
            parsed = parse_skill_md(skill_md)
            results.append((meta, parsed))
    return results


def build_catalog_md_bundled(entries: list[tuple[dict[str, Any], dict[str, Any]]]) -> str:
    by_cat: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for meta, parsed in entries:
        if meta["source_kind"] != "bundled":
            continue
        by_cat[meta["category"]].append((meta, parsed))
    for k in by_cat:
        by_cat[k].sort(key=lambda e: e[0]["slug"])

    lines = [
        "---",
        "sidebar_position: 5",
        'title: "Bundled Skills Catalog"',
        'description: "Catalog of bundled skills that ship with Hermes Agent"',
        "---",
        "",
        "# Bundled Skills Catalog",
        "",
        "Hermes ships with a large built-in skill library copied into `~/.hermes/skills/` on install. Each skill below links to a dedicated page with its full definition, setup, and usage.",
        "",
        "If a skill is missing from this list but present in the repo, the catalog is regenerated by `website/scripts/generate-skill-docs.py`.",
        "",
    ]
    for category in sorted(by_cat):
        lines.append(f"## {category}")
        lines.append("")
        lines.append("| Skill | Description | Path |")
        lines.append("|-------|-------------|------|")
        for meta, parsed in by_cat[category]:
            fm = parsed["frontmatter"]
            name = fm.get("name", meta["slug"])
            desc = (fm.get("description") or "").strip()
            if len(desc) > 240:
                desc = desc[:237].rstrip() + "..."
            link_target = f"/docs/user-guide/skills/bundled/{meta['category']}/{page_id(meta)}"
            path = f"`{meta['rel_path']}`"
            desc_esc = mdx_escape_body(desc).replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| [`{name}`]({link_target}) | {desc_esc} | {path} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_catalog_md_optional(entries: list[tuple[dict[str, Any], dict[str, Any]]]) -> str:
    by_cat: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for meta, parsed in entries:
        if meta["source_kind"] != "optional":
            continue
        by_cat[meta["category"]].append((meta, parsed))
    for k in by_cat:
        by_cat[k].sort(key=lambda e: e[0]["slug"])

    lines = [
        "---",
        "sidebar_position: 9",
        'title: "Optional Skills Catalog"',
        'description: "Official optional skills shipped with hermes-agent — install via hermes skills install official/<category>/<skill>"',
        "---",
        "",
        "# Optional Skills Catalog",
        "",
        "Optional skills ship with hermes-agent under `optional-skills/` but are **not active by default**. Install them explicitly:",
        "",
        "```bash",
        "hermes skills install official/<category>/<skill>",
        "```",
        "",
        "For example:",
        "",
        "```bash",
        "hermes skills install official/blockchain/solana",
        "hermes skills install official/mlops/flash-attention",
        "```",
        "",
        "Each skill below links to a dedicated page with its full definition, setup, and usage.",
        "",
        "To uninstall:",
        "",
        "```bash",
        "hermes skills uninstall <skill-name>",
        "```",
        "",
    ]
    for category in sorted(by_cat):
        lines.append(f"## {category}")
        lines.append("")
        lines.append("| Skill | Description |")
        lines.append("|-------|-------------|")
        for meta, parsed in by_cat[category]:
            fm = parsed["frontmatter"]
            name = fm.get("name", meta["slug"])
            desc = (fm.get("description") or "").strip()
            if len(desc) > 240:
                desc = desc[:237].rstrip() + "..."
            link_target = f"/docs/user-guide/skills/optional/{meta['category']}/{page_id(meta)}"
            desc_esc = mdx_escape_body(desc).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| [**{name}**]({link_target}) | {desc_esc} |")
        lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## Contributing Optional Skills",
            "",
            "To add a new optional skill to the repository:",
            "",
            "1. Create a directory under `optional-skills/<category>/<skill-name>/`",
            "2. Add a `SKILL.md` with standard frontmatter (name, description, version, author)",
            "3. Include any supporting files in `references/`, `templates/`, or `scripts/` subdirectories",
            "4. Submit a pull request — the skill will appear in this catalog and get its own docs page once merged",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_sidebar_items(entries: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict:
    """Build a dict representing the Skills sidebar tree.

    Structure:
    Skills
    ├── (hand-written pages first: godmode, google-workspace)
    ├── Bundled
    │   ├── apple
    │   │   ├── apple-apple-notes
    │   │   └── ...
    │   └── ...
    └── Optional
        └── ...
    """
    bundled = defaultdict(list)
    optional = defaultdict(list)
    for meta, _ in entries:
        if meta["source_kind"] == "bundled":
            bundled[meta["category"]].append(meta)
        else:
            optional[meta["category"]].append(meta)

    def cat_section(bucket: dict[str, list[dict[str, Any]]]) -> list[dict]:
        result = []
        for category in sorted(bucket):
            items = sorted(bucket[category], key=lambda m: m["slug"])
            result.append(
                {
                    "type": "category",
                    "label": category,
                    "collapsed": True,
                    "items": [sidebar_doc_id(m) for m in items],
                }
            )
        return result

    return {
        "bundled_categories": cat_section(bundled),
        "optional_categories": cat_section(optional),
    }


def write_sidebar(entries):
    data = build_sidebar_items(entries)
    # Render just the "Skills" block TS for inclusion.
    def render_items(cats: list[dict]) -> str:
        lines = []
        for c in cats:
            lines.append("            {")
            lines.append("              type: 'category',")
            lines.append(f"              label: '{c['label']}',")
            lines.append("              collapsed: true,")
            lines.append("              items: [")
            for item in c["items"]:
                lines.append(f"                '{item}',")
            lines.append("              ],")
            lines.append("            },")
        return "\n".join(lines)

    bundled_block = render_items(data["bundled_categories"])
    optional_block = render_items(data["optional_categories"])

    skills_subtree = (
        "        {\n"
        "          type: 'category',\n"
        "          label: 'Skills',\n"
        "          collapsed: true,\n"
        "          items: [\n"
        "            'user-guide/skills/godmode',\n"
        "            'user-guide/skills/google-workspace',\n"
        "            {\n"
        "              type: 'category',\n"
        "              label: 'Bundled (by default)',\n"
        "              collapsed: true,\n"
        "              items: [\n"
        + bundled_block
        + "\n              ],\n"
        "            },\n"
        "            {\n"
        "              type: 'category',\n"
        "              label: 'Optional (installable)',\n"
        "              collapsed: true,\n"
        "              items: [\n"
        + optional_block
        + "\n              ],\n"
        "            },\n"
        "          ],\n"
        "        },\n"
    )

    sidebar_path = REPO / "website" / "sidebars.ts"
    text = sidebar_path.read_text(encoding="utf-8")
    # Replace the existing Skills block.
    pattern = re.compile(
        r"        \{\n"
        r"          type: 'category',\n"
        r"          label: 'Skills',\n"
        r"(?:.*?\n)*?"
        r"        \},\n",
        re.DOTALL,
    )
    # Safer: match the exact current block shape.
    old_block_start = "        {\n          type: 'category',\n          label: 'Skills',\n"
    i = text.find(old_block_start)
    if i == -1:
        raise RuntimeError("Could not find Skills sidebar block to replace")
    # Find matching closing of this block -- walk brace depth
    depth = 0
    j = i
    while j < len(text):
        ch = text[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                # Include the trailing ,\n after the closing brace
                end = text.find("\n", j) + 1
                break
        j += 1
    else:
        raise RuntimeError("Could not find end of Skills sidebar block")

    new_text = text[:i] + skills_subtree + text[end:]
    sidebar_path.write_text(new_text, encoding="utf-8")
    print(f"Updated sidebar: {sidebar_path}")


def main():
    entries = discover_skills()
    print(f"Discovered {len(entries)} skills")

    # Build name -> meta index for related-skill cross-linking
    skill_index: dict[str, dict[str, Any]] = {}
    for meta, parsed in entries:
        name = parsed["frontmatter"].get("name", meta["slug"])
        # Prefer bundled over optional if a name collision exists
        if name not in skill_index or meta["source_kind"] == "bundled":
            skill_index[name] = meta

    # Write per-skill pages
    written = 0
    for meta, parsed in entries:
        out_path = page_output_path(meta)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        content = render_skill_page(
            meta, parsed["frontmatter"], parsed["body"], skill_index=skill_index
        )
        out_path.write_text(content, encoding="utf-8")
        written += 1
    print(f"Wrote {written} per-skill pages under {SKILLS_PAGES}")

    # Regenerate catalogs
    bundled_catalog = build_catalog_md_bundled(entries)
    (DOCS / "reference" / "skills-catalog.md").write_text(bundled_catalog, encoding="utf-8")
    print("Updated reference/skills-catalog.md")

    optional_catalog = build_catalog_md_optional(entries)
    (DOCS / "reference" / "optional-skills-catalog.md").write_text(optional_catalog, encoding="utf-8")
    print("Updated reference/optional-skills-catalog.md")

    # Update sidebar
    write_sidebar(entries)


if __name__ == "__main__":
    main()

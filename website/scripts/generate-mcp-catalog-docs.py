#!/usr/bin/env python3
"""Generate per-MCP Docusaurus pages from optional-mcps/<name>/manifest.yaml files.

Mirrors the pattern of website/scripts/generate-skill-docs.py:

  * Discovers entries by globbing ``optional-mcps/*/manifest.yaml``.
  * Validates each manifest against the shape rules in
    ``hermes_cli/mcp_catalog.py::_parse_manifest`` (entries that fail to
    parse are skipped with a warning to stderr; the build is not aborted).
  * Writes per-entry detail pages at
    ``website/docs/user-guide/mcps/optional/<name>.md``. Each page shows
    overview, transport, auth, install steps, default-enabled tools,
    post-install notes, and embeds the full manifest YAML verbatim.
  * Regenerates ``website/docs/reference/optional-mcps-catalog.md`` with a
    table linking each entry to its detail page, trust-model notes, and a
    "how to contribute" section.
  * Idempotently updates ``website/sidebars.ts`` to nest the per-entry pages
    under a new MCPs section (parallel to the Skills section). The script
    walks brace depth (matching the skills generator) so it replaces the
    block in place — re-running the script does not duplicate entries.

The script intentionally does NOT touch ``generate-skill-docs.py``; any
shared helpers (``mdx_escape_body``) are copied here rather than refactored
into a shared module, to keep blast radius minimal.

If an ``optional-mcps/*/manifest.yaml`` fails validation, the script logs
the failure and proceeds with the remaining entries. CI will still build
docs from whatever entries parsed cleanly.
"""

from __future__ import annotations
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
DOCS = REPO / "website" / "docs"
MCP_PAGES_DIR = DOCS / "user-guide" / "mcps" / "optional"
CATALOG_PAGE = DOCS / "reference" / "optional-mcps-catalog.md"
OPTIONAL_MCPS_DIR = REPO / "optional-mcps"
SIDEBAR_PATH = REPO / "website" / "sidebars.ts"

_MANIFEST_VERSION = 1


# ─── mdx_escape_body (copied from generate-skill-docs.py) ─────────────────────
#
# Kept in sync with the version in generate-skill-docs.py. Copy rather than
# refactor — the skills generator is the canonical source; if you change the
# behaviour, copy the change here too. Tests for both generators rely on the
# same behaviour.

_BOX_DRAWING_CHARS = frozenset("┌┐└┘─│═║╔╗╚╝╠╣╦╩╬├┤┬┴┼╭╮╯╰▶◀▲▼")


def _wrap_ascii_art_code_blocks(code_segment: str) -> str:
    if not any(ch in _BOX_DRAWING_CHARS for ch in code_segment):
        return code_segment
    return (
        "<!-- ascii-guard-ignore -->\n"
        f"{code_segment}\n"
        "<!-- ascii-guard-ignore-end -->"
    )


def mdx_escape_body(body: str) -> str:
    """Escape MDX-dangerous characters in markdown body, leaving fenced code blocks alone.

    See website/scripts/generate-skill-docs.py for the canonical version.
    """
    lines = body.split("\n")
    segments: List[Tuple[str, str]] = []  # ("text"|"code", content)
    buf: List[str] = []
    mode = "text"
    fence_char: Optional[str] = None
    fence_len = 0
    for line in lines:
        stripped = line.lstrip()
        if mode == "text":
            if stripped.startswith("```") or stripped.startswith("~~~"):
                if buf:
                    segments.append(("text", "\n".join(buf)))
                    buf = []
                buf.append(line)
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
                segments.append(("code", "\n".join(buf)))
                buf = []
                mode = "text"
                fence_char = None
                fence_len = 0
    if buf:
        segments.append((mode, "\n".join(buf)))

    def escape_text(text: str) -> str:
        out: List[str] = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "`":
                j = i
                while j < len(text) and text[j] == "`":
                    j += 1
                run = text[i:j]
                end = text.find(run, j)
                if end == -1:
                    out.append(text[i:])
                    i = len(text)
                    continue
                out.append(text[i : end + len(run)])
                i = end + len(run)
            else:
                if ch == "{":
                    out.append("&#123;")
                elif ch == "}":
                    out.append("&#125;")
                elif ch == "<":
                    if text[i:].startswith("<!--"):
                        end = text.find("-->", i)
                        if end != -1:
                            out.append(text[i : end + 3])
                            i = end + 3
                            continue
                    m = re.match(
                        r"<(/?)([A-Za-z][A-Za-z0-9]*)([^<>]*)>",
                        text[i:],
                    )
                    if m:
                        tag = m.group(2).lower()
                        safe_tags = {
                            "br", "hr", "img", "a", "b", "i", "em", "strong",
                            "code", "kbd", "sup", "sub", "span", "div", "p",
                            "ul", "ol", "li", "table", "thead", "tbody", "tr",
                            "td", "th", "details", "summary", "blockquote",
                            "pre", "mark", "small", "u", "s", "del", "ins",
                            "h1", "h2", "h3", "h4", "h5", "h6",
                        }
                        if tag in safe_tags:
                            out.append(m.group(0))
                            i += len(m.group(0))
                            continue
                    out.append("&lt;")
                else:
                    out.append(ch)
                i += 1
        return "".join(out)

    processed: List[str] = []
    for kind, content in segments:
        if kind == "code":
            processed.append(_wrap_ascii_art_code_blocks(content))
        else:
            processed.append(escape_text(content))
    return "\n".join(processed)


# ─── Manifest validation (mirrors hermes_cli/mcp_catalog.py::_parse_manifest) ──


class ManifestError(Exception):
    """Manifest parse / shape validation failure."""


def _validate_env_spec(raw: Any, path: Path) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ManifestError(f"{path}: env entry must be a mapping, got {type(raw).__name__}")
    name = raw.get("name") or ""
    if not name or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise ManifestError(f"{path}: invalid env var name: {name!r}")
    return {
        "name": name,
        "prompt": raw.get("prompt") or name,
        "required": bool(raw.get("required", True)),
        "secret": bool(raw.get("secret", True)),
        "default": str(raw.get("default") or ""),
    }


def parse_manifest(path: Path) -> Dict[str, Any]:
    """Parse + validate a manifest. Raise ManifestError on any problem.

    Returns the raw parsed dict plus a normalized view used by the page
    renderer. We keep both because the per-entry page embeds the full YAML
    verbatim (so users see exactly what's in the repo).
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw_text) or {}
    except Exception as exc:
        raise ManifestError(f"failed to read {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ManifestError(f"{path}: manifest must be a mapping")

    mv = data.get("manifest_version")
    if mv != _MANIFEST_VERSION:
        raise ManifestError(
            f"{path}: manifest_version {mv!r} unsupported "
            f"(generator understands version {_MANIFEST_VERSION})"
        )

    name = data.get("name") or ""
    if not name or not re.match(r"^[A-Za-z0-9_-]+$", name):
        raise ManifestError(f"{path}: invalid or missing 'name'")

    description = str(data.get("description") or "").strip()
    if not description:
        raise ManifestError(f"{path}: 'description' required")

    source = str(data.get("source") or "").strip()

    transport_raw = data.get("transport") or {}
    if not isinstance(transport_raw, dict):
        raise ManifestError(f"{path}: 'transport' must be a mapping")
    t_type = transport_raw.get("type")
    if t_type not in ("stdio", "http"):
        raise ManifestError(f"{path}: transport.type must be 'stdio' or 'http'")
    t_args = transport_raw.get("args") or []
    if not isinstance(t_args, list):
        raise ManifestError(f"{path}: transport.args must be a list")
    if t_type == "stdio" and not transport_raw.get("command"):
        raise ManifestError(f"{path}: stdio transport requires 'command'")
    if t_type == "http" and not transport_raw.get("url"):
        raise ManifestError(f"{path}: http transport requires 'url'")

    auth_raw = data.get("auth") or {"type": "none"}
    if not isinstance(auth_raw, dict):
        raise ManifestError(f"{path}: 'auth' must be a mapping")
    a_type = auth_raw.get("type") or "none"
    if a_type not in ("api_key", "oauth", "none"):
        raise ManifestError(f"{path}: auth.type must be 'api_key'|'oauth'|'none'")
    env_list_raw = auth_raw.get("env") or []
    if not isinstance(env_list_raw, list):
        raise ManifestError(f"{path}: auth.env must be a list")
    env_list = [_validate_env_spec(e, path) for e in env_list_raw]

    tools_raw = data.get("tools") or {}
    if not isinstance(tools_raw, dict):
        raise ManifestError(f"{path}: 'tools' must be a mapping")
    default_enabled = tools_raw.get("default_enabled")
    if default_enabled is not None:
        if not isinstance(default_enabled, list) or not all(
            isinstance(t, str) for t in default_enabled
        ):
            raise ManifestError(
                f"{path}: tools.default_enabled must be a list of strings"
            )

    install_raw = data.get("install")
    if install_raw is not None:
        if not isinstance(install_raw, dict):
            raise ManifestError(f"{path}: 'install' must be a mapping")
        if install_raw.get("type") != "git":
            raise ManifestError(
                f"{path}: install.type must be 'git' (got {install_raw.get('type')!r})"
            )
        if not install_raw.get("url") or not install_raw.get("ref"):
            raise ManifestError(f"{path}: install.url and install.ref are required")
        bootstrap = install_raw.get("bootstrap") or []
        if not isinstance(bootstrap, list):
            raise ManifestError(f"{path}: install.bootstrap must be a list")

    return {
        "name": name,
        "description": description,
        "source": source,
        "transport": {
            "type": t_type,
            "command": transport_raw.get("command"),
            "args": [str(a) for a in t_args],
            "url": transport_raw.get("url"),
        },
        "auth": {
            "type": a_type,
            "env": env_list,
        },
        "tools": {"default_enabled": default_enabled},
        "install": install_raw,
        "post_install": str(data.get("post_install") or ""),
        "raw_text": raw_text.rstrip() + "\n",
    }


# ─── Discovery ─────────────────────────────────────────────────────────────────


def discover_entries() -> List[Dict[str, Any]]:
    """Glob optional-mcps/*/manifest.yaml; return parsed entries sorted by name.

    Bad manifests are skipped with a warning to stderr.
    """
    entries: List[Dict[str, Any]] = []
    if not OPTIONAL_MCPS_DIR.is_dir():
        print(
            f"[generate-mcp-catalog-docs] {OPTIONAL_MCPS_DIR} does not exist; "
            "no entries to generate.",
            file=sys.stderr,
        )
        return entries
    for manifest_path in sorted(OPTIONAL_MCPS_DIR.glob("*/manifest.yaml")):
        try:
            entry = parse_manifest(manifest_path)
            entries.append(entry)
        except ManifestError as exc:
            print(
                f"[generate-mcp-catalog-docs] skipping {manifest_path}: {exc}",
                file=sys.stderr,
            )
    entries.sort(key=lambda e: e["name"])
    return entries


# ─── Per-entry page rendering ──────────────────────────────────────────────────


def _sanitize_yaml_string(s: str) -> str:
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _transport_summary(transport: Dict[str, Any]) -> str:
    t_type = transport["type"]
    if t_type == "http":
        return f"http (remote) — `{transport['url']}`"
    # stdio
    cmd = transport.get("command") or ""
    args = transport.get("args") or []
    parts = [cmd] + list(args)
    cmd_line = " ".join(parts).strip()
    return f"stdio — `{cmd_line}`" if cmd_line else "stdio"


def _auth_summary(auth: Dict[str, Any]) -> str:
    t = auth["type"]
    if t == "none":
        return "none"
    if t == "oauth":
        return "oauth"
    # api_key
    env_names = [e["name"] for e in auth.get("env", [])]
    if env_names:
        return "api_key (" + ", ".join(f"`{n}`" for n in env_names) + ")"
    return "api_key"


def render_entry_page(entry: Dict[str, Any]) -> str:
    name = entry["name"]
    description = entry["description"]

    fm_title = _sanitize_yaml_string(name)
    fm_desc = _sanitize_yaml_string(description)
    fm_label = _sanitize_yaml_string(name)

    lines: List[str] = []
    lines.append("---")
    lines.append(f'id: {name}')
    lines.append(f'title: "{fm_title}"')
    lines.append(f'sidebar_label: "{fm_label}"')
    lines.append(f'description: "{fm_desc}"')
    lines.append("---")
    lines.append("")
    lines.append(
        "{/* This page is auto-generated from "
        f"`optional-mcps/{name}/manifest.yaml` by "
        "website/scripts/generate-mcp-catalog-docs.py. "
        "Edit the manifest, not this page. */}"
    )
    lines.append("")
    lines.append(f"# {name}")
    lines.append("")
    lines.append(mdx_escape_body(description))
    lines.append("")

    # Overview / install instructions
    lines.append("## Overview")
    lines.append("")
    if entry["source"]:
        lines.append(f"**Source:** [{entry['source']}]({entry['source']})")
        lines.append("")
    lines.append("Install this catalog entry with:")
    lines.append("")
    lines.append("```bash")
    lines.append(f"hermes mcp install {name}")
    lines.append("```")
    lines.append("")
    lines.append(
        "or pick it interactively with `hermes mcp`. "
        "Uninstall with `hermes mcp uninstall " + name + "` "
        "(the server's config block is removed; any credentials in "
        "`~/.hermes/.env` are preserved)."
    )
    lines.append("")

    # Transport
    transport = entry["transport"]
    lines.append("## Transport")
    lines.append("")
    lines.append(f"**Type:** `{transport['type']}`")
    lines.append("")
    if transport["type"] == "http":
        lines.append(f"**URL:** `{transport['url']}`")
    else:  # stdio
        lines.append(f"**Command:** `{transport['command']}`")
        if transport["args"]:
            lines.append("")
            lines.append("**Args:**")
            lines.append("")
            for arg in transport["args"]:
                lines.append(f"- `{arg}`")
    lines.append("")

    # Auth
    auth = entry["auth"]
    lines.append("## Auth")
    lines.append("")
    lines.append(f"**Type:** `{auth['type']}`")
    lines.append("")
    if auth["type"] == "api_key" and auth["env"]:
        lines.append("**Environment variables prompted at install time:**")
        lines.append("")
        lines.append("| Variable | Prompt | Required | Secret |")
        lines.append("|----------|--------|----------|--------|")
        for env in auth["env"]:
            prompt_safe = (env["prompt"] or "").replace("|", "\\|")
            lines.append(
                f"| `{env['name']}` | {prompt_safe} | "
                f"{'yes' if env['required'] else 'no'} | "
                f"{'yes' if env['secret'] else 'no'} |"
            )
        lines.append("")
        lines.append(
            "Values are written to `~/.hermes/.env`. "
            "Non-secret env vars also go to `.env` to keep one credential store."
        )
        lines.append("")
    elif auth["type"] == "oauth":
        lines.append(
            "OAuth is handled at first connection. For native MCP OAuth, "
            "Hermes's MCP client triggers the browser flow on the first probe."
        )
        lines.append("")
    else:
        lines.append("No credentials required.")
        lines.append("")

    # Install (git bootstrap)
    install = entry["install"]
    if install:
        lines.append("## Install")
        lines.append("")
        lines.append(f"**Type:** `{install.get('type')}`")
        lines.append("")
        lines.append(f"**Repository:** `{install.get('url')}` (ref: `{install.get('ref')}`)")
        lines.append("")
        bootstrap = install.get("bootstrap") or []
        if bootstrap:
            lines.append("**Bootstrap commands** (run inside the cloned directory after clone):")
            lines.append("")
            lines.append("```bash")
            for cmd in bootstrap:
                lines.append(cmd)
            lines.append("```")
            lines.append("")

    # Tools
    default_enabled = entry["tools"].get("default_enabled")
    lines.append("## Tools")
    lines.append("")
    if default_enabled:
        lines.append(
            "By default, only these tools are enabled at install time "
            "(others are hidden until the user opts in via the install-time checklist):"
        )
        lines.append("")
        for tool in default_enabled:
            lines.append(f"- `{tool}`")
        lines.append("")
    else:
        lines.append(
            "No default tool filter is declared. The install-time checklist "
            "starts with every probed tool pre-checked — users prune what "
            "they don't want."
        )
        lines.append("")

    # Post-install
    if entry["post_install"].strip():
        lines.append("## Post-install notes")
        lines.append("")
        lines.append(mdx_escape_body(entry["post_install"].strip()))
        lines.append("")

    # Manifest verbatim
    lines.append("## Manifest")
    lines.append("")
    lines.append(
        f"The manifest below is the source of truth. It lives at "
        f"`optional-mcps/{name}/manifest.yaml` in the hermes-agent repo."
    )
    lines.append("")
    lines.append("```yaml")
    lines.append(entry["raw_text"].rstrip())
    lines.append("```")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ─── Catalog page rendering ────────────────────────────────────────────────────


def render_catalog_page(entries: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("---")
    lines.append("sidebar_position: 10")
    lines.append('title: "Optional MCPs Catalog"')
    lines.append(
        'description: "Nous-approved optional MCP servers shipped with hermes-agent — '
        'install via hermes mcp install <name>"'
    )
    lines.append("---")
    lines.append("")
    lines.append(
        "{/* This page is auto-generated from optional-mcps/*/manifest.yaml by "
        "website/scripts/generate-mcp-catalog-docs.py. Edit the manifests, not "
        "this page. */}"
    )
    lines.append("")
    lines.append("# Optional MCPs Catalog")
    lines.append("")
    lines.append(
        "Optional MCP servers ship with hermes-agent under `optional-mcps/` but are "
        "**not active by default**. They are discovered through `hermes mcp catalog` "
        "and activated explicitly with `hermes mcp install <name>`."
    )
    lines.append("")
    lines.append(
        "Presence in `optional-mcps/` is the trust signal: an entry is in the catalog "
        "only because a maintainer merged a PR adding it. There is no community tier "
        "and no automatic refresh — the manifest you see is the manifest you get until "
        "you re-run `hermes mcp install` after a repo update."
    )
    lines.append("")

    lines.append("## CLI usage")
    lines.append("")
    lines.append("```bash")
    lines.append("hermes mcp                  # interactive picker (TUI) — toggle entries on/off")
    lines.append("hermes mcp catalog          # plain-text list of Nous-approved entries (scriptable)")
    lines.append("hermes mcp install <name>   # install a catalog entry by name (prompts for env/OAuth)")
    lines.append("hermes mcp uninstall <name> # remove the server's config block (.env credentials are preserved)")
    lines.append("```")
    lines.append("")
    lines.append(
        "`hermes mcp install` writes a `mcp_servers.<name>` block into "
        "`~/.hermes/config.yaml` using the manifest's `transport:` keys, runs any "
        "`install:` bootstrap (e.g. `git clone` + `pip install`), and prompts for "
        "any `auth:` env vars defined by the manifest. Secrets go to "
        "`~/.hermes/.env`; non-secret env vars also land in `.env` to keep one "
        "credential store."
    )
    lines.append("")
    lines.append(
        "For the general MCP config shape (independent of the catalog), see the "
        "[MCP Config Reference](/reference/mcp-config-reference). For the conceptual "
        "overview, see [MCP (Model Context Protocol)](/user-guide/features/mcp)."
    )
    lines.append("")

    # Catalog table
    lines.append("## Catalog entries")
    lines.append("")
    if entries:
        lines.append("| Name | Transport | Auth | Source | Description |")
        lines.append("|------|-----------|------|--------|-------------|")
        for entry in entries:
            name = entry["name"]
            link_target = f"/docs/user-guide/mcps/optional/{name}"
            transport_cell = entry["transport"]["type"]
            auth_cell = entry["auth"]["type"]
            source_cell = (
                f"[{entry['source']}]({entry['source']})"
                if entry["source"]
                else "—"
            )
            desc_cell = (
                mdx_escape_body(entry["description"])
                .replace("|", "\\|")
                .replace("\n", " ")
            )
            lines.append(
                f"| [**{name}**]({link_target}) | {transport_cell} | "
                f"{auth_cell} | {source_cell} | {desc_cell} |"
            )
        lines.append("")
    else:
        lines.append("_No catalog entries yet._")
        lines.append("")

    # Trust model — quoted directly from hermes_cli/mcp_catalog.py
    lines.append("## Trust model")
    lines.append("")
    lines.append(
        "The catalog policy is intentionally narrow, and is enforced at the "
        "directory level rather than via metadata:"
    )
    lines.append("")
    lines.append(
        "- **Approval is a merged PR.** Entries are added only by merging a PR into "
        "hermes-agent. Presence in the `optional-mcps/` directory equals Nous "
        "approval. There is no community tier and no trust signals beyond \"it's in "
        "the catalog\"."
    )
    lines.append(
        "- **Manifests pin transport details.** Each manifest fixes the command, "
        "args, install URL, and git ref. MCPs are never auto-updated — users re-run "
        "`hermes mcp install <name>` explicitly to pull a new manifest version "
        "after a repo update."
    )
    lines.append(
        "- **Secrets live in `~/.hermes/.env`.** Env vars prompted at install time go "
        "to `~/.hermes/.env` (the .env-is-for-secrets rule). Non-secret env vars "
        "also go to `.env` so there is one credential store."
    )
    lines.append(
        "- **Default tool surface is conservative.** When an entry specifies "
        "`tools.default_enabled`, the install-time checklist pre-prunes mutating or "
        "rarely-useful tools — users opt in to the full surface per their threat "
        "model."
    )
    lines.append("")

    # Contributing
    lines.append("## How to contribute an entry")
    lines.append("")
    lines.append("To propose a new optional MCP:")
    lines.append("")
    lines.append(
        "1. Add a directory under `optional-mcps/<name>/` containing a `manifest.yaml`."
    )
    lines.append(
        "2. Use the existing entries (`optional-mcps/linear/manifest.yaml`, "
        "`optional-mcps/n8n/manifest.yaml`) as templates. They cover the two "
        "supported transports (HTTP with native MCP OAuth; stdio with git-clone "
        "install)."
    )
    lines.append(
        "3. Set `manifest_version: 1` — the current schema version constant in "
        "`hermes_cli/mcp_catalog.py`. Manifests with a higher version than the "
        "running CLI are skipped, so bumping the version is a coordinated change "
        "with the catalog loader."
    )
    lines.append(
        "4. Submit a PR. Maintainers review transport, auth, default tool surface, "
        "and source provenance. Once merged, the entry appears in `hermes mcp "
        "catalog` and gets its own page in this catalog."
    )
    lines.append("")
    lines.append(
        "See [Contributing](https://github.com/NousResearch/hermes-agent/blob/main/"
        "CONTRIBUTING.md) for the general PR workflow."
    )
    lines.append("")

    # See also
    lines.append("## See also")
    lines.append("")
    lines.append("- [MCP Config Reference](/reference/mcp-config-reference)")
    lines.append("- [MCP (Model Context Protocol)](/user-guide/features/mcp)")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ─── Sidebar update ────────────────────────────────────────────────────────────


def _render_mcp_block(entries: List[Dict[str, Any]]) -> str:
    """Render the MCPs sidebar block as TypeScript source.

    Indentation matches the existing Skills block (8 leading spaces for the
    outer `{`).
    """
    pad = " " * 8
    inner = " " * 10
    inner2 = " " * 12
    inner3 = " " * 14

    out: List[str] = []
    out.append(f"{pad}{{")
    out.append(f"{inner}type: 'category',")
    out.append(f"{inner}label: 'MCPs',")
    out.append(f"{inner}collapsed: true,")
    out.append(f"{inner}items: [")
    out.append(f"{inner2}'reference/optional-mcps-catalog',")
    out.append(f"{inner2}{{")
    out.append(f"{inner3}type: 'category',")
    out.append(f"{inner3}label: 'Optional',")
    out.append(f"{inner3}key: 'mcps-optional',")
    out.append(f"{inner3}collapsed: true,")
    out.append(f"{inner3}items: [")
    for entry in entries:
        out.append(f"{inner3}  'user-guide/mcps/optional/{entry['name']}',")
    out.append(f"{inner3}],")
    out.append(f"{inner2}}},")
    out.append(f"{inner}],")
    out.append(f"{pad}}},")
    return "\n".join(out) + "\n"


def _add_reference_entry(text: str) -> str:
    """Idempotently add 'reference/optional-mcps-catalog' to the Reference section.

    Inserts directly after 'reference/mcp-config-reference' (the natural
    neighbour). If the entry is already present, returns text unchanged.
    """
    target_line = "        'reference/optional-mcps-catalog',\n"
    if target_line in text:
        return text
    anchor = "        'reference/mcp-config-reference',\n"
    idx = text.find(anchor)
    if idx == -1:
        raise RuntimeError(
            "Could not find 'reference/mcp-config-reference' in sidebars.ts "
            "to anchor the new optional-mcps-catalog entry."
        )
    insert_at = idx + len(anchor)
    return text[:insert_at] + target_line + text[insert_at:]


def _replace_or_insert_mcps_block(text: str, mcp_block: str) -> str:
    """Find an existing MCPs sidebar block and replace it; if absent, insert
    immediately after the Skills block.

    Brace-walking mirrors generate-skill-docs.py::write_sidebar.
    """
    anchor = "        {\n          type: 'category',\n          label: 'MCPs',\n"
    skills_anchor = (
        "        {\n          type: 'category',\n          label: 'Skills',\n"
    )

    i = text.find(anchor)
    if i != -1:
        # Replace existing MCPs block
        depth = 0
        j = i
        while j < len(text):
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = text.find("\n", j) + 1
                    break
            j += 1
        else:
            raise RuntimeError("Could not find end of MCPs sidebar block")
        return text[:i] + mcp_block + text[end:]

    # Insert after Skills block
    si = text.find(skills_anchor)
    if si == -1:
        raise RuntimeError(
            "Could not find Skills sidebar block to anchor the new MCPs section."
        )
    depth = 0
    j = si
    while j < len(text):
        ch = text[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = text.find("\n", j) + 1
                break
        j += 1
    else:
        raise RuntimeError("Could not find end of Skills sidebar block")
    return text[:end] + mcp_block + text[end:]


def update_sidebar(entries: List[Dict[str, Any]]) -> None:
    text = SIDEBAR_PATH.read_text(encoding="utf-8")
    text = _add_reference_entry(text)
    mcp_block = _render_mcp_block(entries)
    text = _replace_or_insert_mcps_block(text, mcp_block)
    SIDEBAR_PATH.write_text(text, encoding="utf-8")
    print(f"Updated sidebar: {SIDEBAR_PATH}")


# ─── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    entries = discover_entries()

    # Always (re)create the per-entry directory so stale entries don't linger
    # across renames. We don't blow away the whole tree — that would race with
    # other generators if they ever share the same parent — but we do remove
    # any *.md files we don't expect.
    MCP_PAGES_DIR.mkdir(parents=True, exist_ok=True)
    expected = {f"{e['name']}.md" for e in entries}
    for existing in MCP_PAGES_DIR.glob("*.md"):
        if existing.name not in expected:
            existing.unlink()

    # Per-entry pages
    written = 0
    for entry in entries:
        out_path = MCP_PAGES_DIR / f"{entry['name']}.md"
        out_path.write_text(render_entry_page(entry), encoding="utf-8")
        written += 1

    # Catalog page
    CATALOG_PAGE.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_PAGE.write_text(render_catalog_page(entries), encoding="utf-8")
    print(f"Updated {CATALOG_PAGE}")

    # Sidebar
    update_sidebar(entries)

    print(
        f"Discovered {len(entries)} MCP entries, "
        f"wrote {written} per-entry pages, "
        "updated catalog + sidebar"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

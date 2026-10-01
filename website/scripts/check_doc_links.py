#!/usr/bin/env python3
"""Keep cross-page links in website/docs resolvable on GitHub AND on the site.

Docusaurus turns a relative Markdown file link (`../user-guide/profiles.md#anchor`)
into the page route, and GitHub's file viewer follows the same path. A site
route written as a link (`/user-guide/profiles`, `/docs/user-guide/profiles`)
only works on the rendered site: GitHub resolves it as a repository path and
404s (issue #114428).

Check mode (default) lists every route-style link in hand-authored pages and
exits 1 when any exist. `--fix` rewrites them in place to relative file links,
preserving `#anchor`, and fails on a route that does not map to a doc file.

Exempt, in both modes:
- generated pages (per-skill pages and the skills catalogs produced by
  generate-skill-docs.py; the generator itself emits relative links);
- routes that are not docs: `/skills`, `/plugins` (React pages), `/img/...`,
  `/llms*.txt`, `/api/...` (static assets).

Usage:
    python3 website/scripts/check_doc_links.py            # lint EN + zh-Hans
    python3 website/scripts/check_doc_links.py --fix      # rewrite in place
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

WEBSITE = Path(__file__).resolve().parents[1]
EN_DOCS = WEBSITE / "docs"
ZH_DOCS = WEBSITE / "i18n" / "zh-Hans" / "docusaurus-plugin-content-docs" / "current"

# Written by generate-skill-docs.py; edit the generator, never these outputs.
GENERATED_PREFIXES = ("user-guide/skills/bundled/", "user-guide/skills/optional/")
GENERATED_FILES = {"reference/skills-catalog.md", "reference/optional-skills-catalog.md"}

# Site routes that are not documents: React pages and static assets.
NON_DOC_ROUTE_PREFIXES = ("/skills", "/plugins", "/img/", "/llms", "/api/")

LINK_RE = re.compile(r"\]\((/[^)\s]*)\)")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


def is_generated(rel: str) -> bool:
    return rel.startswith(GENERATED_PREFIXES) or rel in GENERATED_FILES


def frontmatter_slug(text: str) -> str | None:
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end < 0:
        return None
    m = re.search(r"^slug:\s*(\S+)\s*$", text[3:end], re.MULTILINE)
    return m.group(1).strip("'\"") if m else None


def route_for(rel: str, text: str) -> str:
    """Docusaurus route (relative to the docs root) for a source file."""
    slug = frontmatter_slug(text)
    if slug and slug.startswith("/"):
        return slug.rstrip("/") or "/"
    stem = re.sub(r"\.mdx?$", "", rel)
    if stem == "index":
        return "/"
    if stem.endswith("/index"):
        stem = stem[: -len("/index")]
    return "/" + stem


def route_map(docs_root: Path) -> dict[str, Path]:
    routes: dict[str, Path] = {}
    for path in sorted(docs_root.rglob("*.md*")):
        if path.suffix not in (".md", ".mdx"):
            continue
        rel = path.relative_to(docs_root).as_posix()
        routes[route_for(rel, path.read_text(encoding="utf-8"))] = path
    return routes


def normalize_route(target: str) -> tuple[str, str]:
    """Split a link target into (route, suffix) where suffix is `#anchor`/`?q`."""
    suffix = ""
    cut = min((i for i in (target.find("#"), target.find("?")) if i >= 0), default=-1)
    if cut >= 0:
        target, suffix = target[:cut], target[cut:]
    if target.startswith("/docs/"):
        target = target[len("/docs") :]
    elif target == "/docs":
        target = "/"
    if len(target) > 1:
        target = target.rstrip("/")
    return target, suffix


def relative_link(source: Path, source_root: Path, target: Path, target_root: Path) -> str:
    """Path from `source` to `target`, expressed inside the source's own tree.

    zh-Hans pages fall back to the EN file when no translation exists; the
    relative path is computed as if the target sat in the same tree, which is
    how both Docusaurus (locale content paths) and GitHub (EN files) resolve it.
    """
    target_rel = target.relative_to(target_root)
    virtual_target = source_root / target_rel
    rel = Path(os.path.relpath(virtual_target, source.parent)).as_posix()
    return rel if rel.startswith(".") else "./" + rel


def iter_links(text: str):
    """Yield (line_no, match) for markdown links outside fenced code blocks."""
    in_fence = False
    for line_no, line in enumerate(text.splitlines(), 1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for m in LINK_RE.finditer(line):
            # Skip links quoted inside inline code (`[x](/y)` examples).
            if line[: m.start()].count("`") % 2 == 1:
                continue
            yield line_no, m


def process_tree(
    docs_root: Path,
    en_routes: dict[str, Path],
    fix: bool,
    include_generated: bool,
) -> tuple[list[str], list[str], int]:
    """Return (route_style_findings, unresolved_findings, files_rewritten)."""
    local_routes = route_map(docs_root) if docs_root != EN_DOCS else en_routes
    findings: list[str] = []
    unresolved: list[str] = []
    rewritten = 0
    for path in sorted(docs_root.rglob("*.md*")):
        if path.suffix not in (".md", ".mdx"):
            continue
        rel = path.relative_to(docs_root).as_posix()
        if is_generated(rel) and not include_generated:
            continue
        text = path.read_text(encoding="utf-8")
        replacements: dict[str, str] = {}
        for line_no, m in iter_links(text):
            target = m.group(1)
            route, suffix = normalize_route(target)
            if route.startswith(NON_DOC_ROUTE_PREFIXES):
                continue
            shown = path.relative_to(WEBSITE) if path.is_relative_to(WEBSITE) else path
            where = f"{shown}:{line_no}: {target}"
            hit = local_routes.get(route) or en_routes.get(route)
            if hit is None:
                unresolved.append(where)
                continue
            findings.append(where)
            if fix:
                root = docs_root if hit.is_relative_to(docs_root) else EN_DOCS
                replacements[m.group(0)] = "](" + relative_link(path, docs_root, hit, root) + suffix + ")"
        if fix and replacements and not unresolved:
            new_text = text
            for old, new in replacements.items():
                new_text = new_text.replace(old, new)
            path.write_text(new_text, encoding="utf-8")
            rewritten += 1
    return findings, unresolved, rewritten


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fix", action="store_true", help="rewrite route-style links to relative file links")
    parser.add_argument(
        "--include-generated",
        action="store_true",
        help="also process generator outputs (only for a one-off migration; the generator owns them)",
    )
    parser.add_argument("--en-only", action="store_true", help="skip the zh-Hans mirror")
    args = parser.parse_args(argv)

    en_routes = route_map(EN_DOCS)
    trees = [EN_DOCS] if args.en_only or not ZH_DOCS.exists() else [EN_DOCS, ZH_DOCS]
    all_findings: list[str] = []
    all_unresolved: list[str] = []
    rewritten = 0
    for tree in trees:
        f, u, r = process_tree(tree, en_routes, args.fix, args.include_generated)
        all_findings += f
        all_unresolved += u
        rewritten += r

    if all_unresolved:
        print("Route-style links that match no doc file (fix the target first):", file=sys.stderr)
        for line in all_unresolved:
            print(f"  {line}", file=sys.stderr)
        return 1
    if args.fix:
        print(f"Rewrote {len(all_findings)} route-style links in {rewritten} files.")
        return 0
    if all_findings:
        print(
            "Route-style links found in hand-authored docs. They 404 on GitHub; write a relative\n"
            "Markdown path instead (`../user-guide/profiles.md#anchor`) or run\n"
            "`python3 website/scripts/check_doc_links.py --fix`:",
            file=sys.stderr,
        )
        for line in all_findings:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("OK: no route-style links in hand-authored docs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

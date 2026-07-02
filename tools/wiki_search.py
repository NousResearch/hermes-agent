#!/usr/bin/env python3
"""Search the yallaplay-wiki submodule with compact, agent-friendly output."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_WIKI_ROOT = Path(__file__).resolve().parents[1] / "yallaplay-wiki"
SKIP_DIRS = {".git", "node_modules", "__pycache__"}
DOMAIN_PREFIXES = {
    "company": ["company.md", "index.md", "dictionary.md", "glossary.md"],
    "dictionary": ["dictionary.md", "glossary.md", "index.md"],
    "products": ["products"],
    "engineering": ["engineering", "reference/warehouse", "reference/formula_dsl.md"],
    "operations": ["operations"],
    "liveops": ["operations", "tools/product", "products"],
    "analytics": ["engineering/data_warehouse.md", "reference/analytics_rules.md", "reference/warehouse", "products"],
    "collaboration": ["people", "tools/team", "journal/meetings"],
    "infrastructure": ["engineering/infrastructure.md", "engineering/backend_services.md", "tools/product/app_insights.md"],
    "tools": ["tools"],
    "market": ["market"],
    "literature": ["literature"],
    "people": ["people"],
}


@dataclass(frozen=True)
class Match:
    score: int
    path: Path
    line_no: int
    line: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="+", help="term(s) to search for")
    parser.add_argument("--wiki-root", type=Path, default=DEFAULT_WIKI_ROOT)
    parser.add_argument("--domain", choices=sorted(DOMAIN_PREFIXES), help="limit search to a wiki domain/facet")
    parser.add_argument("--limit", type=int, default=20, help="maximum matches to print")
    parser.add_argument("--context", type=int, default=0, help="context lines before/after each match")
    parser.add_argument("--files-only", action="store_true", help="print matching files, not matching lines")
    return parser.parse_args()


def iter_markdown_files(root: Path, domain: str | None) -> list[Path]:
    prefixes = DOMAIN_PREFIXES.get(domain, []) if domain else []
    files: list[Path] = []
    for path in root.rglob("*.md"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        rel = path.relative_to(root).as_posix()
        if prefixes and not any(rel == prefix or rel.startswith(f"{prefix}/") for prefix in prefixes):
            continue
        files.append(path)
    return sorted(files)


def score_line(line: str, terms: list[str]) -> int:
    lower = line.lower()
    score = 0
    for term in terms:
        count = lower.count(term)
        score += count * (10 if " " in term else 4)
    if line.lstrip().startswith("#"):
        score += 8
    if "[" in line and "](" in line:
        score += 2
    return score


def collect_matches(files: list[Path], terms: list[str], root: Path, context: int) -> list[Match]:
    matches: list[Match] = []
    for path in files:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        file_bonus = sum(8 for term in terms if term in path.relative_to(root).as_posix().lower())
        matched_lines: set[int] = set()
        for index, line in enumerate(lines):
            line_score = score_line(line, terms)
            if line_score <= 0:
                continue
            start = max(0, index - context)
            end = min(len(lines), index + context + 1)
            for ctx_index in range(start, end):
                if ctx_index in matched_lines:
                    continue
                matched_lines.add(ctx_index)
                ctx_line = lines[ctx_index].strip()
                if not ctx_line:
                    continue
                score = (line_score if ctx_index == index else max(1, line_score // 3)) + file_bonus
                matches.append(Match(score, path.relative_to(root), ctx_index + 1, ctx_line))
    return sorted(matches, key=lambda match: (-match.score, match.path.as_posix(), match.line_no))


def highlight(line: str, terms: list[str]) -> str:
    result = line
    for term in sorted(terms, key=len, reverse=True):
        result = re.sub(f"({re.escape(term)})", r"**\1**", result, flags=re.IGNORECASE)
    return result


def main() -> int:
    args = parse_args()
    root = args.wiki_root.resolve()
    if not root.exists():
        print(f"wiki root not found: {root}", file=sys.stderr)
        return 2

    query = " ".join(args.query).strip()
    terms = [part.lower() for part in re.findall(r'"([^"]+)"|([^\s]+)', query) for part in part if part]
    if not terms:
        print("empty query", file=sys.stderr)
        return 2

    files = iter_markdown_files(root, args.domain)
    matches = collect_matches(files, terms, root, args.context)

    if args.files_only:
        seen: set[Path] = set()
        for match in matches:
            if match.path in seen:
                continue
            seen.add(match.path)
            print(match.path.as_posix())
            if len(seen) >= args.limit:
                break
        return 0 if seen else 1

    for match in matches[: args.limit]:
        print(f"{match.path.as_posix()}:{match.line_no}: {highlight(match.line, terms)}")
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Benchmark ``website/scripts/extract-skills.py`` before/after #96029.

Generates a synthetic ``skills-index.json`` matching the shape the live
``scripts/build_skills_index.py`` produces (~88k entries spread across
the same sources the real catalog uses), then runs the *current*
``extract-skills.py`` against it and reports:

  * wall-clock time for ``extract_local_skills``
  * wall-clock time for ``extract_unified_index_skills``
  * wall-clock time for the whole ``main()`` pass
  * the final ``skills.json`` size on disk

If ``--baseline`` is passed, the script also runs an inlined copy of the
*previous* implementation (no thread pool, ``os.walk``, pre-built prefix
list not present, no field-stripping, no ``_search`` pre-computation)
and prints a head-to-head table. This is what the PR's Evidence section
cites.

Usage::

    python website/scripts/benchmark_extract_skills.py [--rows 88000]
    python website/scripts/benchmark_extract_skills.py --baseline

Output is printed as a tab-separated table suitable for pasting into
the PR description or running through ``column -t``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from contextlib import contextmanager
from typing import Iterator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(REPO_ROOT, "website", "scripts")


@contextmanager
def _synthetic_index(path: str, rows: int) -> Iterator[None]:
    """Write a fake ``skills-index.json`` with the catalog shape #96029 expects.

    Mixes the same source distribution the real index carries so the
    unified-index loop exercises every branch (github-prefix matching,
    skills.sh install cmd synthesis, clawhub without owner, etc.).
    """
    skills: list = []
    sources = [
        ("skills.sh", 0.55, lambda i: {
            "source": "skills.sh",
            "identifier": f"skills-sh/owner-{i % 500}/repo-{i % 1200}/skill-{i}",
            "name": f"skill-{i}",
            "description": f"Synthetic skills.sh skill {i} that does something cool and helpful.",
            "tags": ["productivity", "api"],
            "repo": f"owner-{i % 500}/repo-{i % 1200}",
        }),
        ("clawhub", 0.25, lambda i: {
            "source": "clawhub",
            "identifier": f"clawhub-slug-{i}",
            "name": f"clawhub-skill-{i}",
            "description": "Synthetic ClawHub skill description.\nLonger line.",
            "tags": ["automation"],
            "extra": {},
        }),
        ("github", 0.12, lambda i: {
            "source": "github",
            "identifier": (
                "openai/skills/foo" if i % 7 == 0
                else "anthropics/skills/bar" if i % 11 == 0
                else "huggingface/skills/baz" if i % 13 == 0
                else f"random-owner-{i % 9000}/random-repo-{i % 13000}/skill-{i}"
            ),
            "name": f"gh-skill-{i}",
            "description": "Synthetic GitHub skill.",
            "tags": ["ai", "ml"] if i % 3 else ["devops"],
        }),
        ("lobehub", 0.04, lambda i: {
            "source": "lobehub",
            "identifier": f"lobehub/synthetic-{i}",
            "name": f"lobehub-{i}",
            "description": "Synthetic LobeHub skill.",
            "tags": ["productivity"],
        }),
        ("browse-sh", 0.02, lambda i: {
            "source": "browse-sh",
            "identifier": f"browse-sh/example.com/login-{i}",
            "name": f"browse-{i}",
            "description": "Synthetic browse.sh skill.",
            "tags": ["automation"],
            "extra": {"source_url": f"https://example.com/task-{i}"},
        }),
        ("well-known", 0.01, lambda i: {
            "source": "well-known",
            "identifier": f"well-known-{i}",
            "name": f"wk-{i}",
            "description": "Synthetic well-known skill.",
            "tags": ["integration"],
        }),
        ("official", 0.01, lambda i: None),  # skipped by extract-skills.py
    ]
    cumulative = 0.0
    for label, frac, build in sources:
        cumulative += frac
    for i in range(rows):
        # Pick a source by stable bucket so the mix doesn't drift row-to-row.
        pick = (i * 0.000137) % 1.0
        cumulative = 0.0
        chosen = sources[0]
        for label, frac, build in sources:
            cumulative += frac
            if pick < cumulative:
                chosen = (label, frac, build)
                break
        entry = chosen[2](i)
        if entry is None:
            continue
        skills.append(entry)

    payload = {
        "version": 1,
        "generated_at": "2026-08-27T00:00:00+00:00",
        "skill_count": len(skills),
        "skills": skills,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    try:
        yield
    finally:
        if os.path.isfile(path):
            os.remove(path)


def _run_current(rows: int, repeats: int, workdir: str) -> dict:
    """Import the current ``extract-skills.py`` and benchmark it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "extract_skills",
        os.path.join(SCRIPTS, "extract-skills.py"),
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    index_path = os.path.join(workdir, "skills-index.json")
    samples_local = []
    samples_unified = []
    samples_total = []
    out_size = 0
    total_rows = 0
    for _ in range(repeats):
        with _synthetic_index(index_path, rows):
            mod.UNIFIED_INDEX_PATH = index_path
            mod.OUTPUT = os.path.join(workdir, "skills.json")
            mod.META_OUTPUT = os.path.join(workdir, "skills-meta.json")
            mod.REPO_ROOT = workdir
            # The local dir doesn't exist in the workdir — empty list,
            # which matches what a fresh clone sees on first run.
            t0 = time.perf_counter()
            local = mod.extract_local_skills()
            t_local = time.perf_counter() - t0
            t0 = time.perf_counter()
            external, _ = mod.extract_unified_index_skills()
            t_unified = time.perf_counter() - t0
            t0 = time.perf_counter()
            mod.main()
            t_total = time.perf_counter() - t0
        samples_local.append(t_local)
        samples_unified.append(t_unified)
        samples_total.append(t_total)
        if os.path.isfile(mod.OUTPUT):
            out_size = os.path.getsize(mod.OUTPUT)
        total_rows = len(local) + len(external)
    return {
        "label": "current (post-#96029)",
        "rows": total_rows,
        "local_s": statistics.median(samples_local),
        "unified_s": statistics.median(samples_unified),
        "total_s": statistics.median(samples_total),
        "out_bytes": out_size,
    }


# --- baseline copy of the OLD implementation, inlined -------------------------
# Kept here (and not in extract-skills.py) so we don't carry the legacy
# behaviour in production. The benchmark imports this only when --baseline
# is passed; it never runs from import-time of the module under test.


def _baseline_extract_local_skills(repo_root: str) -> list:
    """Original os.walk + synchronous read implementation."""
    import yaml
    LOCAL_SKILL_DIRS = [
        ("skills", "built-in"),
        ("optional-skills", "optional"),
    ]
    CATEGORY_LABELS = {
        "software-development": "Software Dev",
        "other": "Other",
    }
    skills: list = []
    for base_dir, source_label in LOCAL_SKILL_DIRS:
        base_path = os.path.join(repo_root, base_dir)
        if not os.path.isdir(base_path):
            continue
        for root, _dirs, files in os.walk(base_path):
            if "SKILL.md" not in files:
                continue
            skill_path = os.path.join(root, "SKILL.md")
            with open(skill_path, encoding="utf-8") as f:
                content = f.read()
            if not content.startswith("---"):
                continue
            parts = content.split("---", 2)
            if len(parts) < 3:
                continue
            try:
                fm = yaml.safe_load(parts[1])
            except yaml.YAMLError:
                continue
            if not fm or not isinstance(fm, dict):
                continue
            skills.append({
                "name": fm.get("name", os.path.basename(root)),
                "description": fm.get("description", ""),
                "category": root.split(os.sep)[0],
                "categoryLabel": CATEGORY_LABELS.get(
                    root.split(os.sep)[0], "Other"
                ),
                "source": source_label,
                "tags": [],
            })
    return skills


def _baseline_extract_unified(index_path: str) -> list:
    """Original single-pass, per-row dict.items() github prefix lookup."""
    if not os.path.isfile(index_path):
        return []
    with open(index_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return []
    GITHUB_TAP_LABELS = {
        "openai/skills": "OpenAI",
        "anthropics/skills": "Anthropic",
        "huggingface/skills": "HuggingFace",
    }
    UNIFIED_SOURCE_LABELS = {
        "skills.sh": "skills.sh",
        "skills-sh": "skills.sh",
        "clawhub": "ClawHub",
        "github": "GitHub",
        "lobehub": "LobeHub",
        "browse-sh": "browse.sh",
    }
    out: list = []
    for entry in data.get("skills", []):
        if not isinstance(entry, dict):
            continue
        source_id = (entry.get("source") or "").lower()
        identifier = entry.get("identifier", "") or ""
        name = entry.get("name") or identifier.split("/")[-1] or "unknown"
        description = (entry.get("description") or "").split("\n")[0]
        if len(description) > 280:
            description = description[:277] + "…"
        if source_id == "github":
            label = "GitHub"
            for prefix, lbl in GITHUB_TAP_LABELS.items():
                if identifier.startswith(prefix + "/") or identifier == prefix:
                    label = lbl
                    break
            source_label = label
        else:
            source_label = UNIFIED_SOURCE_LABELS.get(source_id, source_id or "community")
        # Old implementation did NOT pre-compute _search or strip empties.
        out.append({
            "name": name,
            "description": description,
            "category": "uncategorized",
            "categoryLabel": "",
            "source": source_label,
            "tags": [],
            "overview": "",
            "platforms": [],
            "author": "",
            "version": "",
            "license": "",
            "envVars": [],
            "commands": [],
            "docsPath": "",
            "identifier": identifier,
            "installCmd": f"hermes skills install {identifier or name}",
            "sourceUrl": "",
        })
    return out


def _run_baseline(rows: int, repeats: int, workdir: str) -> dict:
    index_path = os.path.join(workdir, "skills-index.json")
    samples_local = []
    samples_unified = []
    samples_total = []
    out_size = 0
    total_rows = 0
    for _ in range(repeats):
        with _synthetic_index(index_path, rows):
            t0 = time.perf_counter()
            local = _baseline_extract_local_skills(workdir)
            t_local = time.perf_counter() - t0
            t0 = time.perf_counter()
            external = _baseline_extract_unified(index_path)
            t_unified = time.perf_counter() - t0
            t0 = time.perf_counter()
            payload = local + external
            out_path = os.path.join(workdir, "skills.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)
            t_total = time.perf_counter() - t0
        samples_local.append(t_local)
        samples_unified.append(t_unified)
        samples_total.append(t_total)
        out_size = os.path.getsize(out_path)
        total_rows = len(payload)
    return {
        "label": "baseline (pre-#96029)",
        "rows": total_rows,
        "local_s": statistics.median(samples_local),
        "unified_s": statistics.median(samples_unified),
        "total_s": statistics.median(samples_total),
        "out_bytes": out_size,
    }


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def _print_table(results: list) -> None:
    headers = ("implementation", "rows", "local_s", "unified_s", "total_s", "out_bytes")
    rows = [(
        r["label"], f"{r['rows']:,}",
        f"{r['local_s']:.3f}",
        f"{r['unified_s']:.3f}",
        f"{r['total_s']:.3f}",
        _fmt_bytes(r["out_bytes"]),
    ) for r in results]
    widths = [max(len(str(c)) for c in col) for col in zip(headers, *rows)]
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(str(c).ljust(w) for c, w in zip(row, widths)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=88000,
                        help="Number of rows in the synthetic unified index")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Repeat each measurement; report the median")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip the inlined baseline (only run the current code)")
    args = parser.parse_args()

    workdir = tempfile.mkdtemp(prefix="skills-bench-")
    try:
        results = [_run_current(args.rows, args.repeats, workdir)]
        if not args.no_baseline:
            results.append(_run_baseline(args.rows, args.repeats, workdir))
        _print_table(results)
        # Sanity: the strip should be measurably happening per row even
        # though the absolute on-disk size may grow because of the
        # precomputed ``_search`` field (the strip saves ~96 bytes/row,
        # the haystack adds ~150 bytes/row). Surface a warning if the
        # per-row empty-field strip is gone.
        if len(results) == 2 and results[0]["rows"] > 100:
            stripped = results[0]["out_bytes"] - results[0]["rows"] * 150
            unstripped = results[1]["out_bytes"]
            if stripped > unstripped:
                print(
                    f"WARNING: per-row strip estimate ({stripped/1024/1024:.1f} MB after "
                    f"subtracting ~150 B/row for the _search field) is larger than "
                    f"baseline ({unstripped/1024/1024:.1f} MB); "
                    "_strip_empty_community_fields may have regressed.",
                    file=sys.stderr,
                )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""Isolated graphify runner for the service code-graph builder.

Reads a job ``{"files": [<abs path>, ...], "directed": true}`` on stdin and
writes ``{"nodes": [...], "edges": [...], "communities": {cid: [id, ...]}}`` on
stdout as JSON. All ``graphifyy`` usage lives here, in a short-lived subprocess,
so a missing/heavy/broken dependency (or a hang) degrades the caller to "code
graph unavailable" instead of destabilising the long-lived gateway process.

The module is deliberately self-contained — it imports only ``graphify`` and the
standard library, never the harness package — so any interpreter that has
``graphifyy`` installed can run it by path, regardless of ``sys.path``.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


def _run(job: dict) -> dict:
    files = [Path(f) for f in job.get("files") or []]
    directed = bool(job.get("directed", True))
    # A base dir the caller wants each node's ``source_file`` reported relative
    # to — otherwise graphify collapses absolute paths to a bare basename, which
    # cannot be mapped back to a unique file.
    root = Path(job["root"]) if job.get("root") else None

    # Imported lazily so an absent dependency surfaces as a clean non-zero exit
    # (caught by the caller) rather than an import error at module load.
    from graphify.build import build_from_json
    from graphify.cluster import cluster
    from graphify.extract import extract

    # graphify persists a ``graphify-out/cache/`` dir under an inferred prefix
    # (the CWD when ``cache_root`` is unset). This runner is a one-shot, so a
    # persistent cache buys nothing and would litter the gateway's working dir
    # and leak state across services/runs. Pin it to a private temp dir that is
    # torn down on exit → fully hermetic, and every run extracts from scratch
    # (a determinism guarantee the byte-stable Observatory build relies on).
    # ``root`` is passed separately, so ``cache_root`` never affects node
    # ids/source_file — it only relocates the throwaway cache.
    with tempfile.TemporaryDirectory(prefix="graphify-cache-") as cache_dir:
        extraction = (
            extract(files, cache_root=Path(cache_dir), root=root)
            if root
            else extract(files, cache_root=Path(cache_dir))
        )
    nodes = extraction.get("nodes", [])
    edges = extraction.get("edges", [])
    graph = build_from_json({"nodes": nodes, "edges": edges}, directed=directed)
    try:
        communities = cluster(graph)
    except Exception:
        # Clustering is best-effort colour; a graph with no discernible
        # communities is still a useful graph.
        communities = {}

    return {
        "nodes": nodes,
        "edges": edges,
        "communities": {str(key): list(value) for key, value in communities.items()},
    }


def main() -> int:
    try:
        job = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"invalid job json: {exc}\n")
        return 2
    try:
        result = _run(job)
    except Exception as exc:  # noqa: BLE001 — report any failure to the caller
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 1
    sys.stdout.write(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

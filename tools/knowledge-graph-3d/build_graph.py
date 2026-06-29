#!/usr/bin/env python3
# สร้างข้อมูลจุด/เส้นจากคลัง Obsidian -> graph.json (เลียนแบบที่เว็บ :9722 ทำ)
import os, re, json
from collections import Counter

VAULT = os.environ.get("VAULT", "/Users/rattanasak/ObsidianVault/HermesAgent")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph.json")

SKIP_DIRS = {".git", ".obsidian", "node_modules", ".trash"}
link_re = re.compile(r"\[\[([^\]]+)\]\]")

files = []
for root, dirs, fnames in os.walk(VAULT):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    for fn in fnames:
        if fn.endswith(".md"):
            files.append(os.path.relpath(os.path.join(root, fn), VAULT))

by_base, by_path = {}, {}
for rel in files:
    no_ext = rel[:-3]
    by_path[no_ext.lower()] = rel
    by_base.setdefault(os.path.basename(no_ext).lower(), rel)

def resolve(target):
    t = target.split("|")[0].split("#")[0].strip()
    if not t:
        return None
    key = t.lower()
    if key in by_path:
        return by_path[key]
    if key in by_base:
        return by_base[key]
    if key.endswith(".md") and key[:-3] in by_path:
        return by_path[key[:-3]]
    return None

edges, edge_seen = [], set()
deg = {rel: 0 for rel in files}
for rel in files:
    try:
        text = open(os.path.join(VAULT, rel), encoding="utf-8", errors="ignore").read()
    except Exception:
        continue
    for m in link_re.finditer(text):
        tgt = resolve(m.group(1))
        if tgt and tgt != rel:
            key = tuple(sorted((rel, tgt)))
            if key not in edge_seen:
                edge_seen.add(key)
                edges.append({"source": rel, "target": tgt})
                deg[rel] += 1
                deg[tgt] += 1

def group_of(rel):
    top = rel.split(os.sep)[0]
    return top if "." not in top else "(root)"

nodes = [{
    "id": rel,
    "name": os.path.basename(rel)[:-3],
    "path": rel,
    "group": group_of(rel),
    "val": 1 + deg[rel],
} for rel in files]

json.dump({"nodes": nodes, "links": edges}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
print(f"nodes={len(nodes)} links={len(edges)} -> {OUT}")
print("groups:", dict(Counter(n["group"] for n in nodes)))

#!/usr/bin/env bash
# LINE-LEVEL diff-closure proof: union(40 feature-PR diffs) ∪ deferred-set added lines
# == full ./src/ added lines vs v0.16.0, with unmapped == 0.
# Stronger than file/hunk closure: compares every non-blank ADDED line by content.
set -uo pipefail
SRC="${1:-<REPO_ROOT>}"
DEFDIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"   # the #50111 checkout root
V016=3c231eb3979ab9c57d5cd6d02f1d577a3b718b43
GH="env -u GITHUB_TOKEN -u GH_TOKEN gh"
cd "$SRC"
git fetch -q https://github.com/NousResearch/hermes-agent.git main 2>/dev/null
MAIN=$(git rev-parse FETCH_HEAD)
git remote add fork https://github.com/arminanton/hermes-agent.git 2>/dev/null || true
git fetch -q fork 2>/dev/null

python3 - "$MAIN" "$DEFDIR" <<'PY'
import subprocess, sys, json, glob, os
MAIN, DEFDIR = sys.argv[1], sys.argv[2]
V016='3c231eb3979ab9c57d5cd6d02f1d577a3b718b43'
def run(c): return subprocess.run(c,shell=True,capture_output=True,text=True).stdout
def added(t):
    o=set(); cur=None
    for ln in t.splitlines():
        if ln.startswith('+++ b/'): cur=ln[6:]
        elif ln.startswith('+') and not ln.startswith('+++'):
            c=ln[1:].strip()
            if c: o.add((cur,c))
    return o
full=added(run(f"git diff {V016} HEAD -- . ':(exclude)*.bak' ':(exclude)*.bak.*' ':(exclude).project-intel/*'"))
prs=json.loads(run("env -u GITHUB_TOKEN -u GH_TOKEN gh pr list --repo NousResearch/hermes-agent --author arminanton --state open --limit 60 --json number,headRefName") or "[]")
pr=set()
for p in prs:
    if p['number']==50111: continue
    run(f"git fetch -q fork {p['headRefName']} 2>/dev/null")
    pr|=added(run(f"git diff {MAIN}...fork/{p['headRefName']}"))
df=set()
for f in glob.glob(f'{DEFDIR}/deferred/*/*.patch')+glob.glob(f'{DEFDIR}/forward-compat/*.patch'):
    df|=added(open(f,errors='ignore').read())
cov=pr|df; un=full-cov
print(f"full src added lines    : {len(full)}")
print(f"feature-PR added lines   : {len(pr)}")
print(f"deferred+fwdcompat lines : {len(df)}")
print(f"covered (PR ∪ deferred) : {len(full&cov)}")
print(f"UNMAPPED (must be 0)    : {len(un)}")
for fp,l in list(un)[:20]: print(f"  UNMAPPED {fp}: {l[:70]}")
sys.exit(0 if not un else 1)
PY
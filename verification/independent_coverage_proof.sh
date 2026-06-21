#!/usr/bin/env bash
# independent_coverage_proof.sh — coverage check INDEPENDENT of reconcile_campaign.sh.
#
# Different method from the reconciler (which builds one big normalized blob): here we go
# FILE-BY-FILE. For every source file changed v0.16.0..src-HEAD, take its ADDED lines
# (git diff) and check each is present (exact, after lstrip) in EITHER:
#   (a) that file's content at some open feature-PR head, OR
#   (b) the union of #50111 deferred .patch added-lines.
# Emits per-file UNCOVERED counts. A file with >0 uncovered prints its lines.
# This is auditable line-by-line and does not depend on the reconciler's hashing.
set -uo pipefail
FORK="fork"
V016="3c231eb3979ab9c57d5cd6d02f1d577a3b718b43"
SRC="$(git rev-parse --show-toplevel)"
PY="$SRC/venv/bin/python"; [ -x "$PY" ] || PY="python3"

env -u GITHUB_TOKEN -u GH_TOKEN git fetch -q "$FORK" 2>/dev/null || true
PRNUMS="$(env -u GITHUB_TOKEN -u GH_TOKEN gh pr list --repo NousResearch/hermes-agent \
  --author arminanton --state open --limit 100 --json number,headRefOid \
  -q '.[] | (.number|tostring)+":"+.headRefOid' 2>/dev/null)"

"$PY" - "$V016" "$SRC" "$FORK" "$PRNUMS" <<'PYEOF'
import subprocess, sys
V016, SRC, FORK, prnums = sys.argv[1:5]
def git(*a):
    return subprocess.run(["git","-C",SRC,*a], capture_output=True, text=True).stdout

# PR head shas (exclude #50111 deferred tracker — handled as the deferred set)
pr_heads = []
deferred_sha = None
for tok in prnums.split():
    num, sha = tok.split(":")
    if num == "50111":
        deferred_sha = sha
        continue
    git("fetch","-q",FORK,sha)
    pr_heads.append((num, sha))

# Build the deferred added-line set (from #50111 .patch files)
deferred_lines = set()
if deferred_sha:
    git("fetch","-q",FORK,deferred_sha)
    for p in git("ls-tree","-r","--name-only",deferred_sha).splitlines():
        if p.endswith(".patch"):
            for ln in git("show",f"{deferred_sha}:{p}").splitlines():
                if ln.startswith("+") and not ln.startswith("+++"):
                    deferred_lines.add(ln[1:].lstrip())

HEAD = git("rev-parse","HEAD").strip()
# Every source file changed v0.16.0..HEAD (exclude generated artifacts)
files = [f for f in git("diff","--name-only",V016,HEAD,
            "--","." , ":(exclude)*.bak", ":(exclude)*.bak.*", ":(exclude).project-intel/**"
         ).splitlines() if f.strip()]

# Cache each PR's file contents lazily
pr_file_cache = {}
def pr_file_lines(sha, f):
    key=(sha,f)
    if key not in pr_file_cache:
        content = git("show",f"{sha}:{f}")
        pr_file_cache[key] = set(l.lstrip() for l in content.splitlines())
    return pr_file_cache[key]

total_added=0; total_uncovered=0; bad_files=[]
for f in files:
    diff = git("diff",V016,HEAD,"--",f)
    added=[l[1:] for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++") and l[1:].strip()]
    if not added: continue
    total_added += len(added)
    # candidate PR contents = PRs whose head changed this file
    cand_sets=[]
    for num,sha in pr_heads:
        # cheap check: does this PR touch f? (diff name-only against its own base)
        base = git("merge-base","origin/main",sha).strip() or V016
        if f in git("diff","--name-only",base,sha).splitlines():
            cand_sets.append(pr_file_lines(sha,f))
    uncovered=[]
    for line in added:
        s=line.lstrip()
        if any(s in cs for cs in cand_sets): continue
        if s in deferred_lines: continue
        uncovered.append(line)
    if uncovered:
        total_uncovered += len(uncovered)
        bad_files.append((f,len(uncovered),uncovered[:8]))

print(f"=== INDEPENDENT FILE-BY-FILE COVERAGE (method: per-file content containment) ===")
print(f"source files changed v0.16.0..HEAD: {len(files)}")
print(f"total added source lines: {total_added}")
print(f"UNCOVERED (not in any touching-PR's file content nor #50111 deferred): {total_uncovered}")
if bad_files:
    print("--- files with uncovered lines ---")
    for f,n,sample in bad_files:
        print(f"  {f}: {n} uncovered")
        for l in sample: print(f"      {l[:90]}")
print(f"\nRESULT: {'PASS — 0 uncovered (independent of reconcile_campaign.sh)' if total_uncovered==0 else 'FAIL — '+str(total_uncovered)+' uncovered'}")
sys.exit(0 if total_uncovered==0 else 1)
PYEOF

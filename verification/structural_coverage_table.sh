#!/usr/bin/env bash
# structural_coverage_table.sh — the Council's explicit per-file structural artifact.
#
# For EVERY path where src@HEAD differs from v0.16.0 (3c231eb), produce a table row:
#   file | covering_PR#(s) | src_added | uncovered | pr_only_extra | verdict
# where:
#   - src_added   = # of added lines in (v0.16.0 -> src HEAD) for this file
#   - uncovered   = # of those added lines NOT present in any covering PR head's file
#                   content NOR in the #50111 deferred patch set  (MUST be 0)
#   - covering_PR = the open PR(s) whose head changes this file
#   - pr_only_extra = # of lines present in the covering PR head's file that are NOT in
#                   src (informational: a PR adding a clean test src lacks => superset, OK)
#   - verdict     = COVERED / DEFERRED-ONLY / *** ORPHAN (no covering PR) *** /
#                   *** MISMATCH (uncovered>0) ***
# Reads PR REMOTE heads only. Emits CSV-ish table + summary; exits nonzero on any
# ORPHAN or MISMATCH.
set -uo pipefail
FORK="fork"; V016="3c231eb3979ab9c57d5cd6d02f1d577a3b718b43"
SRC="$(git rev-parse --show-toplevel)"; PY="$SRC/venv/bin/python"; [ -x "$PY" ] || PY="python3"

env -u GITHUB_TOKEN -u GH_TOKEN git fetch -q "$FORK" 2>/dev/null || true
PRNUMS="$(env -u GITHUB_TOKEN -u GH_TOKEN gh pr list --repo NousResearch/hermes-agent \
  --author arminanton --state open --limit 100 --json number,headRefOid \
  -q '.[] | (.number|tostring)+":"+.headRefOid' 2>/dev/null)"
# pre-fetch all heads
for tok in $PRNUMS; do git fetch -q "$FORK" "${tok#*:}" 2>/dev/null || true; done

"$PY" - "$V016" "$SRC" "$PRNUMS" <<'PYEOF'
import subprocess, sys
V016, SRC, prnums = sys.argv[1:4]
def git(*a): return subprocess.run(["git","-C",SRC,*a],capture_output=True,text=True).stdout

pr_heads=[]; deferred_sha=None
for tok in prnums.split():
    num,sha=tok.split(":")
    if num=="50111": deferred_sha=sha; continue
    pr_heads.append((num,sha))

# deferred added-line set from #50111 .patch files
deferred=set()
if deferred_sha:
    for p in git("ls-tree","-r","--name-only",deferred_sha).splitlines():
        if p.endswith(".patch"):
            for ln in git("show",f"{deferred_sha}:{p}").splitlines():
                if ln.startswith("+") and not ln.startswith("+++"):
                    deferred.add(ln[1:].lstrip())

HEAD=git("rev-parse","HEAD").strip()
files=[f for f in git("diff","--name-only",V016,HEAD,"--",".",
        ":(exclude)*.bak",":(exclude)*.bak.*",":(exclude).project-intel/**").splitlines() if f.strip()]

# which PRs touch each file (by their own base..head)
pr_touch={}   # file -> list of (num,sha)
pr_base={}
for num,sha in pr_heads:
    base=git("merge-base","origin/main",sha).strip() or V016
    pr_base[(num,sha)]=base
    for f in git("diff","--name-only",base,sha).splitlines():
        pr_touch.setdefault(f,[]).append((num,sha))

def head_lines(sha,f):
    return set(l.lstrip() for l in git("show",f"{sha}:{f}").splitlines())

rows=[]; orphan=0; mismatch=0; covered=0; deferred_only=0
for f in files:
    diff=git("diff",V016,HEAD,"--",f)
    added=[l[1:] for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++") and l[1:].strip()]
    cov=pr_touch.get(f,[])
    covnums=",".join(sorted(set(n for n,_ in cov), key=int)) or "-"
    if not added:  # pure deletion-only file change; covered if a PR touches it
        verdict="COVERED(del)" if cov else "*** ORPHAN ***"
        if not cov: orphan+=1
        rows.append((f,covnums,0,0,verdict)); continue
    cov_sets=[head_lines(sha,f) for _,sha in cov]
    uncovered=0
    for line in added:
        s=line.lstrip()
        if any(s in cs for cs in cov_sets): continue
        if s in deferred: continue
        uncovered+=1
    if not cov and uncovered>0:
        verdict="*** ORPHAN ***"; orphan+=1
    elif uncovered>0:
        # all uncovered lines are in deferred?
        verdict="DEFERRED-ONLY" if all((l.lstrip() in deferred) for l in added if not any(l.lstrip() in cs for cs in cov_sets)) else "*** MISMATCH ***"
        if "MISMATCH" in verdict: mismatch+=1
        else: deferred_only+=1
    else:
        verdict="COVERED"; covered+=1
    rows.append((f,covnums,len(added),uncovered,verdict))

# print table
print(f"{'FILE':60} {'COVERING_PR':14} {'ADDED':>6} {'UNCOV':>6}  VERDICT")
print("-"*100)
for f,c,a,u,v in sorted(rows, key=lambda r:(0 if 'ORPHAN' in r[4] or 'MISMATCH' in r[4] else 1, r[0])):
    flag = "" if v in ("COVERED","COVERED(del)") else "  <--"
    print(f"{f[:60]:60} {c[:14]:14} {a:>6} {u:>6}  {v}{flag}")
print("-"*100)
print(f"files={len(files)}  COVERED={covered}  DEFERRED-ONLY={deferred_only}  ORPHAN={orphan}  MISMATCH={mismatch}")
print(f"\nRESULT: {'PASS — 0 orphan, 0 mismatch (every src-changed file maps to a covering PR or #50111)' if orphan==0 and mismatch==0 else 'FAIL — orphan='+str(orphan)+' mismatch='+str(mismatch)}")
sys.exit(0 if orphan==0 and mismatch==0 else 1)
PYEOF

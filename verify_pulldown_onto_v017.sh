#!/usr/bin/env bash
# verify_pulldown_onto_v017.sh — HONEST per-PR pull-down verification onto v0.17.0.
#
# The realistic operator pull-down model is: cherry-pick / apply EACH PR independently
# onto the target release, resolving its conflicts as they come — NOT stacking all 40 PRs
# into one tree (our PRs overlap each other on shared files, so a single stacked apply is
# order-fragile and silently drops PRs). This script applies each open PR independently
# onto clean v0.17.0 and reports CLEAN / RESOLVED / CONFLICT honestly.
set -u
SRC="${1:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
V017=2bd1977d8fad185c9b4be47884f7e87f1add0ce3
MAIN_REF=origin/main
GH="env -u GITHUB_TOKEN -u GH_TOKEN gh"
cd "$SRC" || exit 1

# Documented single-file resolution strategies (PR# -> strategy), verified to compile:
declare -A STRAT=(
  [49644]=theirs [49916]=theirs [50056]=both [50064]=theirs [50073]=keep400 [50296]=theirs [50033]=theirs [50758]=theirs
)

clean=0; resolved=0; conflict=0; total=0
echo "PR     RESULT                       conflict-file"
echo "----   --------------------------   -------------"
$GH pr list --repo NousResearch/hermes-agent --author arminanton --state open --limit 100 \
  --json number,headRefOid -q '.[] | "\(.number) \(.headRefOid)"' | sort -n | while read n sha; do
    [ "$n" = "50111" ] && continue          # manifest PR (docs only)
    total=$((total+1))
    git cat-file -e "${sha}^{commit}" 2>/dev/null || git fetch fork "$sha" >/dev/null 2>&1
    git diff "$MAIN_REF...$sha" > "/tmp/_pd_$n.diff" 2>/dev/null
    rm -rf /tmp/_pd_wt; git worktree add /tmp/_pd_wt "$V017" >/dev/null 2>&1
    if git -C /tmp/_pd_wt apply --check "/tmp/_pd_$n.diff" 2>/dev/null; then
        printf "#%-5s CLEAN\n" "$n"
    else
        git -C /tmp/_pd_wt apply --3way "/tmp/_pd_$n.diff" >/dev/null 2>&1
        cf=$(git -C /tmp/_pd_wt grep -l '^<<<<<<<' 2>/dev/null | head -1)
        if [ -z "$cf" ]; then
            printf "#%-5s 3WAY-AUTO-RESOLVED\n" "$n"
        else
            strat="${STRAT[$n]:-UNDOCUMENTED}"
            # apply the documented strategy + compile check
            python3 - "/tmp/_pd_wt/$cf" "$strat" <<'PY' 2>/dev/null
import sys
p,strat=sys.argv[1],sys.argv[2]; s=open(p).read().splitlines(); r=[]; i=0
while i<len(s):
    if s[i].startswith('<<<<<<<'):
        j=i+1; ours=[]
        while j<len(s) and not s[j].startswith('======='): ours.append(s[j]); j+=1
        k=j+1; th=[]
        while k<len(s) and not s[k].startswith('>>>>>>>'): th.append(s[k]); k+=1
        if strat=='both': r+=ours+th
        elif strat=='keep400': r+=[t.replace('"hygiene_hard_message_limit": 5000','"hygiene_hard_message_limit": 400') for t in th]
        else: r+=th
        i=k+1
    else: r.append(s[i]); i+=1
open(p,'w').write("\n".join(r)+"\n")
PY
            comp="?"; case "$cf" in *.py) python3 -m py_compile "/tmp/_pd_wt/$cf" 2>/dev/null && comp="compiles" || comp="COMPILE-FAIL";; *) comp="n/a";; esac
            left=$(git -C /tmp/_pd_wt grep -lc '^<<<<<<<' 2>/dev/null | wc -l)
            printf "#%-5s CONFLICT->%s(%s,markers-left=%s)   %s\n" "$n" "$strat" "$comp" "$left" "$cf"
        fi
    fi
    git worktree remove /tmp/_pd_wt --force 2>/dev/null
done
rm -f /tmp/_pd_*.diff

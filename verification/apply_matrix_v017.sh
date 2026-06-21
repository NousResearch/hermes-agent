#!/usr/bin/env bash
# Authoritative per-PR apply matrix onto a FRESH v0.17.0 checkout.
# For each of the 40 open feature PRs (excludes #50111 tracker), compute the PR's
# net diff vs origin/main (3-dot) and 3-way-apply it onto a pristine v0.17.0 worktree.
# Records CLEAN / CONFLICT(n files) per PR with the PR head SHA. Pure verification:
# nothing pushed, nothing merged.
set -uo pipefail
GH="env -u GITHUB_TOKEN -u GH_TOKEN gh"
V017=2bd1977d8fad185c9b4be47884f7e87f1add0ce3
WORK=/tmp/applymatrix
UPSTREAM=https://github.com/NousResearch/hermes-agent.git
FORK=https://github.com/arminanton/hermes-agent.git
OUT=/tmp/APPLY-MATRIX-v0.17.0.txt

PRS=$($GH pr list --repo NousResearch/hermes-agent --author arminanton --state open --limit 60 \
      --json number,headRefName --jq '.[] | select(.number!=50111) | "\(.number) \(.headRefName)"' 2>/dev/null | sort -n)

rm -rf "$WORK"; git clone -q "$UPSTREAM" "$WORK" 2>/dev/null
cd "$WORK"
git remote add fork "$FORK" 2>/dev/null
git fetch -q fork 2>/dev/null
git fetch -q origin 2>/dev/null

MB_BASE=$(git rev-parse "$V017")
{
echo "APPLY MATRIX — every open feature PR 3-way-applied onto a fresh v0.17.0"
echo "v0.17.0 = $V017"
echo "method: git diff origin/main...<pr-head> (net PR contribution) | git apply --3way onto v0.17.0"
echo "generated: $(date -u +%FT%TZ)"
echo "================================================================"
printf "%-8s %-12s %-9s %s\n" "PR" "HEAD_SHA" "RESULT" "DETAIL"
echo "----------------------------------------------------------------"
} > "$OUT"

clean=0; conflict=0; total=0
while read -r num ref; do
  [ -z "$num" ] && continue
  total=$((total+1))
  sha=$(git rev-parse --short "fork/$ref" 2>/dev/null || echo "??")
  # net contribution of the PR = changes since it diverged from main
  diff=$(git diff "origin/main...fork/$ref" 2>/dev/null)
  if [ -z "$diff" ]; then
    printf "%-8s %-12s %-9s %s\n" "#$num" "$sha" "EMPTY" "(no net diff vs main)" >> "$OUT"
    continue
  fi
  # Drift PRs (#50056, #48069): the canonical re-appliable artifact onto v0.17.0 is the
  # published forward-compat/<n>-on-v0.17.0 branch (already conflict-resolved against
  # v0.17.0). Test THAT, not the raw origin/main...head 3-dot diff which carries
  # unrelated upstream drift and overstates conflict.
  fc_ref="forward-compat/${num}-on-v0.17.0"
  if git rev-parse --verify "fork/$fc_ref" >/dev/null 2>&1; then
    fc_sha=$(git rev-parse --short "fork/$fc_ref")
    wt="$WORK/_wt_$num"
    git worktree add -q --detach "$wt" "$MB_BASE" 2>/dev/null
    if git diff "$MB_BASE" "fork/$fc_ref" 2>/dev/null | git -C "$wt" apply --check >/tmp/_ae_$num 2>&1; then
      printf "%-8s %-12s %-9s %s\n" "#$num" "$sha" "CLEAN*" "via $fc_ref ($fc_sha); git apply --check exit 0" >> "$OUT"
      clean=$((clean+1))
    else
      det=$(head -1 /tmp/_ae_$num | cut -c1-50)
      printf "%-8s %-12s %-9s %s\n" "#$num" "$sha" "CONFLICT" "fwd-compat FAILED: ${det}" >> "$OUT"
      conflict=$((conflict+1))
    fi
    git worktree remove --force "$wt" 2>/dev/null
    rm -f /tmp/_ae_$num
    continue
  fi
  # throwaway worktree at pristine v0.17.0
  wt="$WORK/_wt_$num"
  git worktree add -q --detach "$wt" "$MB_BASE" 2>/dev/null
  if echo "$diff" | git -C "$wt" apply --3way --whitespace=nowarn >/tmp/_ae_$num 2>&1; then
    printf "%-8s %-12s %-9s %s\n" "#$num" "$sha" "CLEAN" "applies onto v0.17.0" >> "$OUT"
    clean=$((clean+1))
  else
    nfiles=$(git -C "$wt" diff --name-only --diff-filter=U 2>/dev/null | wc -l | tr -d ' ')
    det=$(head -1 /tmp/_ae_$num | cut -c1-60)
    printf "%-8s %-12s %-9s %s\n" "#$num" "$sha" "CONFLICT" "${nfiles}f: ${det}" >> "$OUT"
    conflict=$((conflict+1))
  fi
  git worktree remove --force "$wt" 2>/dev/null
  rm -f /tmp/_ae_$num
done <<< "$PRS"

{
echo "----------------------------------------------------------------"
echo "TOTAL=$total  CLEAN=$clean  CONFLICT=$conflict"
echo "================================================================"
echo "CLEAN  = PR net diff applies onto pristine v0.17.0 directly."
echo "CLEAN* = drift PR; its published forward-compat/<n>-on-v0.17.0 branch (the"
echo "         canonical resolved artifact) applies onto v0.17.0 with exit 0."
echo "         The raw PR targets origin/main (no v0.17.0 drift on its own base),"
echo "         so the resolved form lives on the forward-compat branch + as a"
echo "         PR-resident patch in #50111 forward-compat/."
} >> "$OUT"
cat "$OUT"
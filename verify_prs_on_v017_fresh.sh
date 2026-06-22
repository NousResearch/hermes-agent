#!/usr/bin/env bash
# Independent fresh-clone verification: apply every open arminanton PR onto a
# pristine v0.17.0 checkout, capture conflict-markers + py_compile per PR.
# Self-contained: clones fresh, uses only git + gh + python3. No reuse of any
# prior worktree. Run: bash verify_prs_on_v017_fresh.sh
set -u
V017=2bd1977d8fad185c9b4be47884f7e87f1add0ce3
WORK=/tmp/fresh-verify
FORK=https://github.com/arminanton/hermes-agent.git
UPSTREAM=https://github.com/NousResearch/hermes-agent.git
OUT="$WORK/MATRIX.txt"

rm -rf "$WORK"; mkdir -p "$WORK"; cd "$WORK" || exit 1
echo "[1] fresh clone of upstream at v0.17.0 ($V017)"
git init -q repo && cd repo
git remote add origin "$UPSTREAM"
git remote add fork "$FORK"
git fetch -q --depth 1 origin "$V017" || { echo "FETCH v017 FAILED"; exit 1; }
git checkout -q "$V017" || { echo "CHECKOUT FAILED"; exit 1; }
echo "    HEAD=$(git rev-parse --short HEAD)  pre-markers=$(grep -rlE '^<<<<<<<' --include='*.py' . 2>/dev/null | wc -l)"

echo "[2] enumerate open PRs"
env -u GITHUB_TOKEN -u GH_TOKEN gh pr list --repo NousResearch/hermes-agent \
  --author arminanton --state open --limit 100 \
  --json number,headRefName -q '.[] | "\(.number) \(.headRefName)"' > "$WORK/prs.txt"
echo "    $(wc -l < "$WORK/prs.txt") PRs"

: > "$OUT"
clean=0; conf=0; skip=0
while read -r num br; do
  # #50111 is the manifest PR (all-additive docs) — it applies clean too; count it like any other.
  # fetch the PR branch + its merge-base content (need enough history for 3-way)
  git fetch -q --force fork "$br:pr_$num" 2>/dev/null
  mb=$(git merge-base "$V017" "pr_$num" 2>/dev/null)
  [ -z "$mb" ] && git fetch -q --deepen 200 fork "$br" 2>/dev/null && mb=$(git merge-base "$V017" "pr_$num" 2>/dev/null)
  git diff "$mb..pr_$num" > "$WORK/d_$num.diff" 2>/dev/null
  # pristine reset
  git reset -q --hard "$V017"; git clean -fdq
  out=$(git apply --3way "$WORK/d_$num.diff" 2>&1); rc=$?
  mk=$(grep -rlE '^<<<<<<<' --include='*.py' . 2>/dev/null | wc -l)
  miss=$(echo "$out" | grep -c 'does not exist in index')
  # compile changed py files
  cf=0
  for f in $(git diff --name-only 2>/dev/null; git ls-files --others --exclude-standard 2>/dev/null); do
    case "$f" in *.py) python3 -m py_compile "$f" 2>/dev/null || cf=$((cf+1));; esac
  done
  if [ "$rc" -eq 0 ] && [ "$mk" -eq 0 ]; then
    printf "#%-6s CLEAN          compile_fail=%s\n" "$num" "$cf" >> "$OUT"; clean=$((clean+1))
  else
    printf "#%-6s CONFLICT mk=%s miss=%s compile_fail=%s\n" "$num" "$mk" "$miss" "$cf" >> "$OUT"; conf=$((conf+1))
  fi
done < "$WORK/prs.txt"
git reset -q --hard "$V017"; git clean -fdq

echo "[3] RESULTS"
sort -t'#' -k2 -n "$OUT"
echo
echo "TOTALS: clean=$clean conflict=$conf skip=$skip  (target v0.17.0 = $V017)"

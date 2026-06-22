#!/usr/bin/env bash
# reproduce-coverage.sh — re-runnable proof that every v0.16.0->overlay src-delta file is
# accounted for: in a live open PR, or in an explicit DISCARD/WITHDRAWN class. Zero orphans.
#
# Honest scope note: this is FILE-level coverage. A pure LINE-level "union of PR diffs ==
# overlay-vs-v0.16.0 diff" does NOT hold cleanly and we do not claim it, because the PRs are
# based on CURRENT origin/main (~1800 commits past v0.16.0), so each PR's line delta is relative
# to main, while the overlay-vs-v0.16.0 delta carries our change PLUS upstream's intervening
# drift. File-level ownership + the per-PR v0.17.0 apply-clean proof (verify-patches.sh) together
# establish "pullable onto v0.17.0"; line-equality against v0.16.0 is not a meaningful invariant
# given the differing bases.
#
# Usage: ./reproduce-coverage.sh    (run from a hermes-agent checkout that has the overlay HEAD)
set -u
REPO="${1:-$PWD}"
V016=3c231eb3979ab9c57d5cd6d02f1d577a3b718b43
FORK=https://github.com/arminanton/hermes-agent.git
cd "$REPO" || { echo "FATAL: no repo at $REPO"; exit 2; }

echo "== 1. src-delta files (git diff v0.16.0..HEAD) =="
git diff --name-only "$V016"..HEAD > /tmp/_cov_delta.txt
TOTAL=$(wc -l < /tmp/_cov_delta.txt)
echo "   delta files: $TOTAL"

echo "== 2. union of files across all OPEN arminanton PRs (paginated, no 100-cap) =="
env -u GITHUB_TOKEN -u GH_TOKEN gh pr list --repo NousResearch/hermes-agent --author arminanton \
    --state open --limit 80 --json number --jq '.[].number' 2>/dev/null | sort -n > /tmp/_cov_prs.txt
echo "   open PRs: $(wc -l < /tmp/_cov_prs.txt)"
: > /tmp/_cov_union.txt
while read -r pr; do
  env -u GITHUB_TOKEN -u GH_TOKEN gh api --paginate "repos/NousResearch/hermes-agent/pulls/$pr/files" \
      --jq '.[].filename' 2>/dev/null >> /tmp/_cov_union.txt
done < /tmp/_cov_prs.txt
sort -u /tmp/_cov_union.txt -o /tmp/_cov_union.txt
echo "   distinct files across open PRs: $(wc -l < /tmp/_cov_union.txt)"

echo "== 3. explicit non-PR classes =="
# WITHDRAWN (maintainer-aligned: agy-cli + gemini-cli-UA, ban-risk/superseded)
cat > /tmp/_cov_withdrawn.txt <<'EOF'
agent/agy_cli_client.py
agent/gemini_native_adapter.py
agent/google_user_agent.py
plugins/model-providers/agy-cli/__init__.py
plugins/model-providers/agy-cli/plugin.yaml
tests/agent/conftest.py
tests/agent/test_agy_cli_client_v2.py
tests/agent/test_agy_cli_client_v3.py
tests/plugins/test_agy_cli_plugin_v2.py
EOF

echo "== 4. classify every delta file =="
COVERED=0; DISCARD=0; WITHDRAWN=0; ORPHAN=0; : > /tmp/_cov_orphans.txt
while read -r f; do
  case "$f" in
    *.bak|*.bak.*|*/.project-intel/*|.project-intel/*|transcripts/*) DISCARD=$((DISCARD+1)); continue;;
  esac
  if grep -qxF "$f" /tmp/_cov_withdrawn.txt; then WITHDRAWN=$((WITHDRAWN+1)); continue; fi
  if grep -qxF "$f" /tmp/_cov_union.txt; then COVERED=$((COVERED+1)); else ORPHAN=$((ORPHAN+1)); echo "$f" >> /tmp/_cov_orphans.txt; fi
done < /tmp/_cov_delta.txt

echo
echo "==================== COVERAGE RESULT ===================="
printf "  total src-delta files : %d\n" "$TOTAL"
printf "  covered by open PR    : %d\n" "$COVERED"
printf "  DISCARD (.bak/intel/txt): %d\n" "$DISCARD"
printf "  WITHDRAWN (maintainer): %d\n" "$WITHDRAWN"
printf "  ORPHANS               : %d\n" "$ORPHAN"
printf "  check: %d + %d + %d + %d = %d (== %d ? %s)\n" \
  "$COVERED" "$DISCARD" "$WITHDRAWN" "$ORPHAN" \
  "$((COVERED+DISCARD+WITHDRAWN+ORPHAN))" "$TOTAL" \
  "$([ $((COVERED+DISCARD+WITHDRAWN+ORPHAN)) -eq "$TOTAL" ] && echo YES || echo NO)"
if [ "$ORPHAN" -ne 0 ]; then echo "  ORPHAN FILES:"; sed 's/^/    /' /tmp/_cov_orphans.txt; fi
echo "========================================================"
[ "$ORPHAN" -eq 0 ] && exit 0 || exit 1

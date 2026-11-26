#!/usr/bin/env bash
# Fail when website/scripts/generate-skill-docs.py output differs from the committed docs.
#
# GitHub renders the committed copies, so a regeneration that changes anything
# means the published catalogs/pages are stale. Fix: run the generator and commit.
#
# Usage: check_generated_skill_docs.sh [generator command...]
# With a command (`just docs-check`), the docs are snapshotted before it runs, so
# uncommitted hand edits are not reported as generator drift. A scratch index
# keeps the snapshot and `git add -N` (which makes newly generated pages visible
# to `git diff`) out of the real index.
set -euo pipefail

paths=(website/docs website/sidebars.ts website/i18n)
index="$(mktemp)"
trap 'rm -f "$index"' EXIT
cp "$(git rev-parse --git-path index)" "$index"
export GIT_INDEX_FILE="$index"
if [ "$#" -gt 0 ]; then
  git add -A -- "${paths[@]}"
  "$@"
fi
git add -N website/docs website/sidebars.ts
if ! git diff --exit-code --stat -- "${paths[@]}"; then
  echo "::error::website/scripts/generate-skill-docs.py output differs from the committed docs; run it and commit the result"
  exit 1
fi

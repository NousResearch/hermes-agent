#!/usr/bin/env bash
#
# update-fork.sh — refresh this fork's develop branch against upstream.
#
# What it does:
#   1. Fetches the latest from upstream (NousResearch/hermes-agent).
#   2. Squashes all local fork-only commits into a single commit.
#   3. Rebases that single commit on top of the latest upstream/main.
#   4. Force-pushes (with lease) the result to fork/develop.
#
# The end state is exactly: upstream/main + ONE commit containing the full
# diff of all local fork changes.
#
# Squashing *before* the rebase means any conflict with upstream is resolved
# once, against the combined diff, instead of once per original commit.
#
# A safety backup branch (backup/<branch>-before-upstream-rebase-<UTC>) is
# created before anything destructive happens.
#
# Usage:
#   ./update-fork.sh             # interactive: confirms before force-pushing
#   ./update-fork.sh -y          # skip the confirmation prompt
#   ./update-fork.sh -n          # dry run: rebase/squash locally, do NOT push
#   ./update-fork.sh -m "msg"    # custom squashed-commit message
#
# Overridable via env vars (defaults shown):
#   UPSTREAM_REMOTE=upstream  UPSTREAM_BRANCH=main
#   FORK_REMOTE=fork          FORK_BRANCH=develop
#
set -euo pipefail

UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
UPSTREAM_BRANCH="${UPSTREAM_BRANCH:-main}"
FORK_REMOTE="${FORK_REMOTE:-fork}"
FORK_BRANCH="${FORK_BRANCH:-develop}"

ASSUME_YES=0
DRY_RUN=0
COMMIT_MSG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes)      ASSUME_YES=1; shift ;;
    -n|--dry-run)  DRY_RUN=1; shift ;;
    -m|--message)  COMMIT_MSG="${2:-}"; shift 2 ;;
    -h|--help)
      sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *)
      echo "error: unknown argument: $1" >&2
      echo "try: $0 --help" >&2
      exit 2 ;;
  esac
done

# --- colored logging helpers --------------------------------------------------
if [[ -t 1 ]]; then
  C_BLUE=$'\033[34m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'
  C_RED=$'\033[31m'; C_BOLD=$'\033[1m'; C_RESET=$'\033[0m'
else
  C_BLUE=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_BOLD=""; C_RESET=""
fi
step() { echo "${C_BLUE}${C_BOLD}==>${C_RESET} ${C_BOLD}$*${C_RESET}"; }
info() { echo "    $*"; }
ok()   { echo "${C_GREEN}  ✓${C_RESET} $*"; }
warn() { echo "${C_YELLOW}  ! $*${C_RESET}"; }
die()  { echo "${C_RED}error:${C_RESET} $*" >&2; exit 1; }

# --- locate the repo (the worktree this script lives in) ----------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 \
  || die "not inside a git work tree (script dir: $SCRIPT_DIR)"
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

UPSTREAM_REF="${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}"

# --- preflight checks ---------------------------------------------------------
step "Preflight checks"

git remote get-url "$UPSTREAM_REMOTE" >/dev/null 2>&1 \
  || die "remote '$UPSTREAM_REMOTE' not found"
git remote get-url "$FORK_REMOTE" >/dev/null 2>&1 \
  || die "remote '$FORK_REMOTE' not found"

CURRENT_BRANCH="$(git symbolic-ref --short HEAD 2>/dev/null || echo '')"
[[ "$CURRENT_BRANCH" == "$FORK_BRANCH" ]] \
  || die "expected branch '$FORK_BRANCH' to be checked out here, but on '$CURRENT_BRANCH'.
       This repo uses worktrees — run this from the worktree where '$FORK_BRANCH' lives
       (e.g. cd into it, or 'git checkout $FORK_BRANCH' here if it isn't checked out elsewhere)."

if [[ -n "$(git status --porcelain)" ]]; then
  git status --short
  die "working tree is not clean — commit or stash changes first."
fi
ok "on '$FORK_BRANCH', working tree clean"

# --- fetch upstream + fork ----------------------------------------------------
step "Fetching $UPSTREAM_REMOTE/$UPSTREAM_BRANCH and $FORK_REMOTE/$FORK_BRANCH"
git fetch --prune "$UPSTREAM_REMOTE" "$UPSTREAM_BRANCH"
git fetch --prune "$FORK_REMOTE" "$FORK_BRANCH" || true
ok "fetched; upstream tip: $(git rev-parse --short "$UPSTREAM_REF")"

# --- figure out what we're squashing ------------------------------------------
BASE="$(git merge-base HEAD "$UPSTREAM_REF")"
AHEAD_COUNT="$(git rev-list --count "${BASE}..HEAD")"
UPSTREAM_NEW="$(git rev-list --count "${BASE}..${UPSTREAM_REF}")"

if [[ "$AHEAD_COUNT" -eq 0 ]]; then
  warn "no local commits ahead of upstream — nothing to squash."
  if [[ "$UPSTREAM_NEW" -gt 0 ]]; then
    info "fast-forwarding $FORK_BRANCH to $UPSTREAM_REF ($UPSTREAM_NEW new upstream commits)"
  else
    ok "already up to date. Nothing to do."
    exit 0
  fi
fi

step "Local commits to squash ($AHEAD_COUNT), $UPSTREAM_NEW new upstream commit(s) to absorb"
git log --reverse --format='    %C(auto)%h%Creset %s' "${BASE}..HEAD" || true

# --- build the squashed commit message ----------------------------------------
if [[ -z "$COMMIT_MSG" ]]; then
  STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  UPSHORT="$(git rev-parse --short "$UPSTREAM_REF")"
  COMMIT_MSG="$(
    printf 'Fork changes (squashed) on top of %s @ %s\n\n' "$UPSTREAM_REF" "$UPSHORT"
    printf 'Rebased and squashed by update-fork.sh at %s.\n' "$STAMP"
    printf 'Combines the following local commits:\n'
    git log --reverse --format='  - %s' "${BASE}..HEAD"
  )"
fi

# --- safety backup ------------------------------------------------------------
BACKUP_BRANCH="backup/${FORK_BRANCH}-before-upstream-rebase-$(date -u +%Y%m%d%H%M%S)"
step "Creating safety backup branch: $BACKUP_BRANCH"
git branch "$BACKUP_BRANCH"
ok "backup created (delete later with: git branch -D $BACKUP_BRANCH)"

# --- squash, then rebase the single commit onto upstream ----------------------
if [[ "$AHEAD_COUNT" -gt 0 ]]; then
  step "Squashing $AHEAD_COUNT local commit(s) into one"
  git reset --soft "$BASE"
  git commit --no-verify -m "$COMMIT_MSG"
  ok "squashed to $(git rev-parse --short HEAD)"
fi

step "Rebasing onto $UPSTREAM_REF"
if ! git rebase "$UPSTREAM_REF"; then
  echo
  die "rebase hit conflicts. Resolve them, then:
       git rebase --continue   (and re-run with the push step)
   or abort and restore:
       git rebase --abort
       git reset --hard $BACKUP_BRANCH"
fi
ok "rebased; $FORK_BRANCH is now $(git rev-parse --short HEAD)"

NEW_AHEAD="$(git rev-list --count "${UPSTREAM_REF}..HEAD")"
step "Result: $FORK_BRANCH = $UPSTREAM_REF + $NEW_AHEAD commit(s)"
git --no-pager log --oneline -n "$((NEW_AHEAD + 2))" --graph --decorate || true

# --- push ---------------------------------------------------------------------
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo
  warn "dry run: NOT pushing. To publish:"
  info "git push --force-with-lease $FORK_REMOTE $FORK_BRANCH"
  info "Backup branch kept at: $BACKUP_BRANCH"
  exit 0
fi

if [[ "$ASSUME_YES" -ne 1 ]]; then
  echo
  read -r -p "${C_BOLD}Force-push to ${FORK_REMOTE}/${FORK_BRANCH}? [y/N] ${C_RESET}" reply
  case "$reply" in
    [yY]|[yY][eE][sS]) ;;
    *) warn "aborted by user — nothing pushed. Local $FORK_BRANCH is updated; backup at $BACKUP_BRANCH."; exit 0 ;;
  esac
fi

step "Force-pushing to $FORK_REMOTE/$FORK_BRANCH"
git push --force-with-lease "$FORK_REMOTE" "$FORK_BRANCH"
ok "done. $FORK_REMOTE/$FORK_BRANCH updated."
info "Backup branch kept locally: $BACKUP_BRANCH"

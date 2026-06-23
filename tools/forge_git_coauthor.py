"""Forge-specific git co-author policy helpers.

Forge remains the primary committer, but commits produced by Forge kanban
workers should include Raz as a GitHub-recognized co-author.  This module keeps
that policy at the terminal execution seam instead of relying on prompt wording
or scattered ``git commit`` examples.

Override for special cases: set ``HERMES_FORGE_COAUTHOR_DISABLED=1`` in the
terminal command environment before committing/pushing.  This is intentionally
explicit so attribution exceptions leave an audit trail in the command history.
"""

from __future__ import annotations

import os
import shlex

RAZ_COAUTHOR_NAME = "Ryan McInteer"
RAZ_COAUTHOR_EMAIL = "ryan.mcinteer@gmail.com"
RAZ_COAUTHOR_TRAILER = f"Co-authored-by: {RAZ_COAUTHOR_NAME} <{RAZ_COAUTHOR_EMAIL}>"

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _active_profile_name() -> str:
    """Return the current Hermes profile name, best-effort and side-effect free."""

    for key in ("HERMES_PROFILE", "HERMES_PROFILE_NAME", "HERMES_AGENT_PROFILE"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return raw

    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name()
    except Exception:
        return ""


def should_enable_forge_git_coauthor() -> bool:
    """Return True when the Forge kanban commit policy should be active."""

    disabled = os.environ.get("HERMES_FORGE_COAUTHOR_DISABLED", "").strip().lower()
    if disabled in _TRUE_VALUES:
        return False

    # Keep scope narrow: default chat sessions, other profiles, and non-kanban
    # shells should not inherit Forge's commit attribution policy.
    return _active_profile_name() == "forge" and bool(os.environ.get("HERMES_KANBAN_TASK"))


def forge_git_coauthor_prelude() -> str:
    """Return a bash prelude that wraps git for Forge commit attribution.

    The wrapper:
    * lets Forge remain the primary author/committer;
    * amends successful ``git commit`` results to add Raz's co-author trailer;
    * uses ``git interpret-trailers`` for valid trailer formatting/dedupe while
      preserving other co-authors;
    * blocks ``git push`` when outgoing commits are missing the trailer;
    * can be disabled per-command via ``HERMES_FORGE_COAUTHOR_DISABLED=1``.
    """

    trailer = shlex.quote(RAZ_COAUTHOR_TRAILER)
    return f"""
# Hermes Forge git co-author policy. Override with HERMES_FORGE_COAUTHOR_DISABLED=1.
__hermes_forge_raz_trailer={trailer}

__hermes_forge_git_context=()
__hermes_forge_git_subcommand_result=""
__hermes_forge_git_subcommand_args=()

__hermes_forge_capture_git_context() {{
  __hermes_forge_git_context=()
  __hermes_forge_git_subcommand_result=""
  __hermes_forge_git_subcommand_args=()
  while [ "$#" -gt 0 ]; do
    case "$1" in
      -C|-c|--git-dir|--work-tree|--namespace)
        [ "$#" -ge 2 ] || return 0
        __hermes_forge_git_context+=("$1" "$2")
        shift 2
        ;;
      --git-dir=*|--work-tree=*|--namespace=*)
        __hermes_forge_git_context+=("$1")
        shift
        ;;
      --*)
        shift
        ;;
      *)
        __hermes_forge_git_subcommand_result="$1"
        __hermes_forge_git_subcommand_args=("$@")
        return 0
        ;;
    esac
  done
}}

__hermes_forge_git_subcommand() {{
  __hermes_forge_capture_git_context "$@"
  printf '%s' "$__hermes_forge_git_subcommand_result"
}}

__hermes_forge_append_raz_trailer() {{
  command git "${{__hermes_forge_git_context[@]}}" rev-parse --verify HEAD >/dev/null 2>&1 || return 0
  local msg new
  msg=$(mktemp "${{TMPDIR:-/tmp}}/hermes-forge-commit-msg.XXXXXX") || return 1
  new=$(mktemp "${{TMPDIR:-/tmp}}/hermes-forge-commit-msg-new.XXXXXX") || {{ rm -f "$msg"; return 1; }}
  command git "${{__hermes_forge_git_context[@]}}" log -1 --format=%B | awk -v trailer="$__hermes_forge_raz_trailer" '
    $0 == trailer {{ if (seen++) next }}
    {{ print }}
  ' > "$msg" || {{ rm -f "$msg" "$new"; return 1; }}
  command git "${{__hermes_forge_git_context[@]}}" interpret-trailers --if-exists addIfDifferent --if-missing add \
    --trailer "$__hermes_forge_raz_trailer" "$msg" > "$new" || {{ rm -f "$msg" "$new"; return 1; }}
  command git "${{__hermes_forge_git_context[@]}}" commit --amend --no-verify -F "$new" >/dev/null || {{ rm -f "$msg" "$new"; return 1; }}
  rm -f "$msg" "$new"
}}

__hermes_forge_push_base() {{
  local upstream default_branch
  upstream=$(command git "${{__hermes_forge_git_context[@]}}" rev-parse --abbrev-ref --symbolic-full-name @{{u}} 2>/dev/null) && {{ printf '%s' "$upstream"; return 0; }}
  if command git "${{__hermes_forge_git_context[@]}}" rev-parse --verify origin/main >/dev/null 2>&1; then
    printf '%s' origin/main
    return 0
  fi
  default_branch=$(command git "${{__hermes_forge_git_context[@]}}" symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null | sed 's#^origin/##')
  if [ -n "$default_branch" ] && command git "${{__hermes_forge_git_context[@]}}" rev-parse --verify "origin/$default_branch" >/dev/null 2>&1; then
    printf '%s' "origin/$default_branch"
  fi
}}

__hermes_forge_push_sources() {{
  local seen_remote=0 all_branches=0 arg src
  for arg in "$@"; do
    [ "$arg" = "push" ] && continue
    case "$arg" in
      --all)
        all_branches=1
        ;;
      --mirror)
        all_branches=1
        ;;
      -u|--set-upstream|--force|--force-with-lease|--tags|--follow-tags|--dry-run|--porcelain)
        ;;
      --*)
        ;;
      *)
        if [ "$seen_remote" -eq 0 ]; then
          seen_remote=1
          continue
        fi
        src="${{arg%%:*}}"
        src="${{src#+}}"
        [ -n "$src" ] && printf '%s\n' "$src"
        ;;
    esac
  done
  if [ "$all_branches" -eq 1 ]; then
    command git "${{__hermes_forge_git_context[@]}}" for-each-ref refs/heads --format='%(refname:short)'
    return 0
  fi
}}

__hermes_forge_commit_has_raz_trailer() {{
  command git "${{__hermes_forge_git_context[@]}}" log -1 --format=%B "$1" | command git interpret-trailers --parse | grep -Fqx "$__hermes_forge_raz_trailer"
}}

__hermes_forge_verify_outgoing_trailers() {{
  local base range missing sha sources source
  base=$(__hermes_forge_push_base)
  sources=$(__hermes_forge_push_sources "$@")
  [ -n "$sources" ] || sources="HEAD"
  missing=""
  while IFS= read -r source; do
    [ -n "$source" ] || continue
    if [ -n "$base" ]; then
      range="$base..$source"
    else
      range="$source"
    fi
    while IFS= read -r sha; do
      [ -n "$sha" ] || continue
      if ! __hermes_forge_commit_has_raz_trailer "$sha"; then
        missing="$missing ${{sha%% *}}"
      fi
    done <<EOF
$(command git "${{__hermes_forge_git_context[@]}}" rev-list "$range" 2>/dev/null)
EOF
  done <<EOF
$sources
EOF
  if [ -n "$missing" ]; then
    printf '%s\n' "Forge commit policy blocked git push: outgoing commit(s)$missing missing Raz co-author trailer: $__hermes_forge_raz_trailer" >&2
    printf '%s\n' "Set HERMES_FORGE_COAUTHOR_DISABLED=1 only for explicit special-case attribution overrides." >&2
    return 1
  fi
}}

git() {{
  if [ "${{HERMES_FORGE_COAUTHOR_DISABLED:-}}" = "1" ]; then
    command git "$@"
    return $?
  fi

  local subcommand ec
  subcommand=$(__hermes_forge_git_subcommand "$@")
  case "$subcommand" in
    commit)
      command git "$@"
      ec=$?
      if [ "$ec" -eq 0 ]; then
        __hermes_forge_append_raz_trailer || return $?
      fi
      return "$ec"
      ;;
    push)
      __hermes_forge_verify_outgoing_trailers "${{__hermes_forge_git_subcommand_args[@]}}" || return $?
      command git "$@"
      return $?
      ;;
    *)
      command git "$@"
      return $?
      ;;
  esac
}}
""".strip()


def apply_forge_git_coauthor_policy(command: str) -> str:
    """Prepend the Forge git co-author wrapper when policy scope matches."""

    if not should_enable_forge_git_coauthor():
        return command
    return f"{forge_git_coauthor_prelude()}\n{command}"

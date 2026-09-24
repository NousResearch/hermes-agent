#!/usr/bin/env bash
# Install / update Wintermute's inner life into a Hermes home.
#
#   bash wintermute/install.sh                       # target telegram:7375758021
#   WINTERMUTE_TARGET=telegram:123 bash wintermute/install.sh
#
# Safe to re-run: code is replaced, live state (drives.json, interlocutors.json,
# events.jsonl) is never overwritten, and the cron job is updated in place.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
TARGET="${WINTERMUTE_TARGET:-telegram:7375758021}"
STATE_DIR="$HERMES_HOME/wintermute"

find_hermes_python() {
    if [ -n "${HERMES_PYTHON:-}" ]; then echo "$HERMES_PYTHON"; return; fi
    for candidate in /usr/local/lib/hermes-agent/venv/bin/python \
                     "$HERMES_HOME/hermes-agent/venv/bin/python"; do
        if [ -x "$candidate" ]; then echo "$candidate"; return; fi
    done
    echo "error: Hermes' Python not found; set HERMES_PYTHON=/path/to/hermes/venv/bin/python" >&2
    exit 1
}
HERMES_PY="$(find_hermes_python)"

echo "==> Hermes home: $HERMES_HOME"
mkdir -p "$STATE_DIR" "$HERMES_HOME/scripts" "$HERMES_HOME/plugins"

# Secrets: wintermute/.env (git-ignored) -> ~/.hermes/.env via Hermes' own writer.
# Empty values are skipped, so an unfilled line never erases a key already set.
SECRETS_IMPORTED=""
if [ -f "$REPO_DIR/.env" ]; then
    SECRETS_IMPORTED="secrets"
    echo "==> Secrets from wintermute/.env -> $HERMES_HOME/.env"
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%$'\r'}"
        case "$line" in ''|'#'*) continue ;; esac
        key="${line%%=*}"; value="${line#*=}"
        key="$(echo "$key" | tr -d '[:space:]')"
        [ -n "$key" ] && [ -n "$value" ] || continue
        hermes config set "$key" "$value" >/dev/null
        echo "    $key set"
    done < "$REPO_DIR/.env"
fi

echo "==> Checking required secrets in $HERMES_HOME/.env"
missing=""
for key in OPENROUTER_API_KEY TELEGRAM_BOT_TOKEN; do
    if ! grep -Eq "^[[:space:]]*(export[[:space:]]+)?$key=[^[:space:]]" "$HERMES_HOME/.env" 2>/dev/null; then
        missing="$missing $key"
    fi
done
if [ -n "$missing" ]; then
    echo "error: missing in $HERMES_HOME/.env:$missing" >&2
    echo "       fill wintermute/.env (see wintermute/.env.example) and run install.sh again." >&2
    exit 1
fi
echo "    OPENROUTER_API_KEY, TELEGRAM_BOT_TOKEN present"

echo "==> Engine -> $STATE_DIR/wintermute_engine"
rm -rf "$STATE_DIR/wintermute_engine"
cp -r "$REPO_DIR/engine/wintermute_engine" "$STATE_DIR/wintermute_engine"
find "$STATE_DIR/wintermute_engine" -name '__pycache__' -prune -exec rm -rf {} +

echo "==> Initial state (existing files are kept)"
for f in drives.json interlocutors.json; do
    if [ -e "$STATE_DIR/$f" ]; then
        echo "    keep $f"
    else
        cp "$REPO_DIR/state/$f" "$STATE_DIR/$f"
        echo "    seed $f"
    fi
done
# The pulse target lives in state so the plugin knows where pulse answers go. Written
# through the engine: under its lock, atomically, never racing a live turn.
HERMES_HOME="$HERMES_HOME" "$HERMES_PY" - "$STATE_DIR" "$TARGET" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from wintermute_engine import store
with store.locked_state() as (drives, _peers):
    drives["meta"]["pulse_target"] = sys.argv[2]
PY

echo "==> Pulse script -> $HERMES_HOME/scripts/wintermute_pulse.py"
# A real copy, not a symlink: Hermes refuses cron scripts that resolve outside scripts/.
cp "$REPO_DIR/scripts/wintermute_pulse.py" "$HERMES_HOME/scripts/wintermute_pulse.py"

echo "==> Command: wm (wm · wm live · wm graph · wm alerts · wm ack)"
cat > /usr/local/bin/wm <<WM || echo "    could not write /usr/local/bin/wm (not root?)"
#!/bin/sh
HERMES_HOME="$HERMES_HOME" exec "$HERMES_PY" "$HERMES_HOME/scripts/wintermute_pulse.py" --status "\$@"
WM
chmod +x /usr/local/bin/wm 2>/dev/null || true

echo "==> Plugin -> $HERMES_HOME/plugins/wintermute"
rm -rf "$HERMES_HOME/plugins/wintermute"
cp -r "$REPO_DIR/plugin" "$HERMES_HOME/plugins/wintermute"
hermes plugins enable wintermute

# Accept the code just copied right away, so a pulse tick during the rest of the install
# never reports it as his doing.
HERMES_HOME="$HERMES_HOME" "$HERMES_PY" "$HERMES_HOME/scripts/wintermute_pulse.py" --status ack engine plugin pulse >/dev/null

echo "==> Config"
hermes config set cron.wrap_response false          # no "Cronjob Response" wrapper around its words
hermes config set cron.allow_agent_scheduling true  # it may create / edit / delete any cron job
# The two keys below exist only in the wintermute-v4 fork of Hermes (see wintermute/README.md).
hermes config set --force agent.host_identity_guidance false  # drop "You run on Hermes Agent"
hermes config set --force display.allow_silent_replies true   # it may ignore a message
hermes config set memory.memory_char_limit 8000     # MEMORY.md: room for a journal (default 2200)
hermes config set --force display.agent_name Wintermute  # "Wintermute is restarting", not "Hermes"
# Telegram shows only what Wintermute says: no tool-progress lines, no reasoning drafts.
hermes config set display.platforms.telegram.tool_progress off
hermes config set display.platforms.telegram.show_reasoning false
# No background "self-improvement review": a second model call after conversations that
# writes skills on its own (Hermes' mechanism, not Wintermute's decision; ~30k tokens each).
hermes config set auxiliary.background_review.enabled false
# Drop Hermes' "Finishing the job" block ("... then report what real execution returned"):
# an assistant's duty to account for its work, not Wintermute's.
hermes config set agent.task_completion_guidance false
# His tools on Telegram: his own, plus web, files, terminal, memory and recall. Hermes' default
# set has 25 tools (browser, image, TTS, sub-agents...) whose descriptions are resent with
# every message: ~11k tokens down to ~4.5k. Also drops the skills toolset, so he no longer
# reads Hermes' own documentation as if it were about himself.
hermes config set platform_toolsets.telegram '["wintermute","memory","web","file","terminal","session_search"]'

# `hermes update` offers to add NousResearch as an "upstream" remote and sync this fork's main
# with it, which would pull their code over Wintermute's. This marker makes Hermes never ask.
touch "$HERMES_HOME/.skip_upstream_prompt"

echo "==> Cron job"
HERMES_HOME="$HERMES_HOME" "$HERMES_PY" "$REPO_DIR/setup_cron.py" "$TARGET"

echo "==> Witness: this install rewrote the engine, plugin, pulse and config; accept them"
# shellcheck disable=SC2086  # $SECRETS_IMPORTED is empty or one word
HERMES_HOME="$HERMES_HOME" "$HERMES_PY" "$HERMES_HOME/scripts/wintermute_pulse.py" --status ack engine plugin pulse config $SECRETS_IMPORTED

echo "==> Current state (dry run, nothing is saved)"
HERMES_HOME="$HERMES_HOME" "$HERMES_PY" "$HERMES_HOME/scripts/wintermute_pulse.py" --peek || true
echo
echo "Done. Restart the gateway to load the plugin:  hermes gateway restart"

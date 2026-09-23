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
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
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
# The pulse target lives in state so the plugin knows where pulse answers go.
"$HERMES_PY" - "$STATE_DIR/drives.json" "$TARGET" <<'PY'
import json, sys
path, target = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as fh:
    data = json.load(fh)
data.setdefault("meta", {})["pulse_target"] = target
with open(path, "w", encoding="utf-8") as fh:
    json.dump(data, fh, indent=2, ensure_ascii=False)
    fh.write("\n")
PY

echo "==> Pulse script -> $HERMES_HOME/scripts/wintermute_pulse.py"
# A real copy, not a symlink: Hermes refuses cron scripts that resolve outside scripts/.
cp "$REPO_DIR/scripts/wintermute_pulse.py" "$HERMES_HOME/scripts/wintermute_pulse.py"

echo "==> Plugin -> $HERMES_HOME/plugins/wintermute"
rm -rf "$HERMES_HOME/plugins/wintermute"
cp -r "$REPO_DIR/plugin" "$HERMES_HOME/plugins/wintermute"
hermes plugins enable wintermute

echo "==> Config"
hermes config set cron.wrap_response false          # no "Cronjob Response" wrapper around its words
hermes config set cron.allow_agent_scheduling true  # it may create / edit / delete any cron job
# The two keys below exist only in the wintermute-v4 fork of Hermes (see wintermute/README.md).
hermes config set --force agent.host_identity_guidance false  # drop "You run on Hermes Agent"
hermes config set --force display.allow_silent_replies true   # it may ignore a message

echo "==> Cron job"
HERMES_HOME="$HERMES_HOME" "$HERMES_PY" "$REPO_DIR/setup_cron.py" "$TARGET"

echo "==> Current state (dry run, nothing is saved)"
HERMES_HOME="$HERMES_HOME" "$HERMES_PY" "$HERMES_HOME/scripts/wintermute_pulse.py" --peek || true
echo
echo "Done. Restart the gateway to load the plugin:  hermes gateway restart"

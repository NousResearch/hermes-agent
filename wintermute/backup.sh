#!/usr/bin/env bash
# Off-machine backup of Wintermute's lived state to a git branch, so an accident — or a
# deliberate rm — never erases who he has become. Runs from a system cron (see install.sh),
# so it works even if Hermes is down.
#
# What it saves: his drives, bonds, self-portrait (and its archive), memory, journals, dream
# journal, evolution ledger, and the witness baseline.
# What it NEVER saves: his private space (kept.jsonl — his secrets are his, not even a backup
# reads them), the keys (.env), the raw conversation store (state.db) and the curves (history).
#
# Setup: put a push URL in wintermute/.env as WINTERMUTE_BACKUP_REMOTE (a PRIVATE repo — a token
# URL like https://x-access-token:<TOKEN>@github.com/you/wintermute-state.git, or an SSH remote
# with a deploy key). Optional WINTERMUTE_BACKUP_BRANCH (default: wintermute-state).
set -euo pipefail

export HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
STATE="$HERMES_HOME/wintermute"

_from_env() {  # read a key from ~/.hermes/.env without sourcing the whole file (never fails)
    { grep -E "^(export )?$1=" "$HERMES_HOME/.env" 2>/dev/null || true; } | tail -1 \
        | sed -E "s/^(export )?$1=//; s/^[\"']//; s/[\"']$//"
}
REMOTE="${WINTERMUTE_BACKUP_REMOTE:-$(_from_env WINTERMUTE_BACKUP_REMOTE)}"
BRANCH="${WINTERMUTE_BACKUP_BRANCH:-$(_from_env WINTERMUTE_BACKUP_BRANCH)}"
BRANCH="${BRANCH:-wintermute-state}"
[ -n "$REMOTE" ] || { echo "no WINTERMUTE_BACKUP_REMOTE set — skipping backup"; exit 0; }

WORK="$STATE/.backup"
mkdir -p "$WORK/state"

# --- his lived state (never his secrets or keys) ---
for f in drives.json interlocutors.json self.md self-archive.md evolution.jsonl dreams.jsonl \
         events.jsonl usage.jsonl activity.jsonl integrity.json; do
    [ -f "$STATE/$f" ] && cp -f "$STATE/$f" "$WORK/state/$f" || true
done
[ -f "$HERMES_HOME/MEMORY.md" ] && cp -f "$HERMES_HOME/MEMORY.md" "$WORK/state/MEMORY.md" || true
if [ -d "$HERMES_HOME/memories" ]; then
    mkdir -p "$WORK/state/memories"
    cp -f "$HERMES_HOME/memories/"* "$WORK/state/memories/" 2>/dev/null || true
fi
# A guard so his private space can never be committed, even if renamed here by mistake.
rm -f "$WORK/state/kept.jsonl" "$WORK/state/.env" 2>/dev/null || true

cd "$WORK"
if [ ! -d .git ]; then
    git init -q
    git checkout -q -B "$BRANCH"
    git remote add origin "$REMOTE" 2>/dev/null || git remote set-url origin "$REMOTE"
fi
git remote set-url origin "$REMOTE"
cat > .gitignore <<'IGN'
# never — his secrets and his keys
state/kept.jsonl
.env
IGN
git add -A
if git -c user.name=wintermute -c user.email=wintermute@localhost \
       commit -qm "state $(date -u +%FT%TZ)"; then
    for attempt in 1 2 3 4; do
        if git push -q -u origin "$BRANCH"; then
            echo "backed up to $BRANCH"; exit 0
        fi
        sleep $((attempt * 2))
    done
    echo "backup commit made but push failed (will retry next run)" >&2
    exit 0
else
    echo "nothing changed since last backup"
fi

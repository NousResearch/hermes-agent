---
sidebar_position: 19
title: "HashiCorp Vault Secrets"
description: "Source provider credentials from HashiCorp Vault at startup via the command secret source, with a last-known-good cache that keeps the agent booting when Vault is sealed or unreachable"
---

# HashiCorp Vault Secrets

Pull your provider credentials (Telegram, Discord, ElevenLabs, model API keys, anything Hermes reads from the environment) out of `~/.hermes/.env` and into a HashiCorp Vault KV path, rotated centrally like the rest of your infrastructure.

Vault is not a bundled backend and, per the [secret source policy](/user-guide/secrets/), never will be: bundled sources are deliberately limited to Bitwarden and 1Password, and everything else lives out of tree. You do not need a plugin for Vault, though. The bundled [command helper](/user-guide/secrets/command) runs any script that prints `KEY=VALUE` lines, and this guide ships a reference helper with one property a plain `vault kv get` lacks: a **last-known-good cache**, so your agent still boots with the previous credentials when Vault is sealed after a host reboot, the network is down, or the machine holding the token can't reach the cluster. Availability failures are routine with self-hosted Vault; this pattern has repeatedly saved unattended agents from starting credential-less.

## What you get

- One `vault kv put` rotates a credential for every consumer, Hermes included.
- Fresh fetch at every Hermes startup; values flow through the standard [precedence ladder](/user-guide/secrets/) with `(from Command helper)` provenance.
- Atomic, `0600`, last-known-good cache: sealed or unreachable Vault degrades to the previous good credential set instead of an agent with no credentials.
- Revocation still works: a `permission denied` from Vault does NOT fall back to the cache (flag-gated), so pulling a token has its intended effect.
- A one-line status breadcrumb per run (`.vault-env.cache.status`), because Hermes deliberately discards helper stderr and you still want to know whether the last boot ran fresh or stale.

## Prerequisites

- `vault` and `jq` on the PATH of the user running Hermes.
- A KV mount (v1 or v2) holding your keys, e.g. `kv/hermes/env`.
- Non-interactive Vault auth for that user (see [Auth patterns](#auth-patterns)). The helper must never prompt: Hermes runs it headless with a hard timeout.

## 1. Store your keys in Vault

Field names lowercase; the helper upcases them into env var names (`telegram_bot_token` becomes `TELEGRAM_BOT_TOKEN`):

```bash
vault kv put kv/hermes/env \
  telegram_bot_token='123456:ABC...' \
  discord_bot_token='...' \
  elevenlabs_api_key='...'
```

## 2. Install the helper

Save the script below as `~/.hermes/scripts/hermes-vault-env.sh` and `chmod 700` it. Full source also available as a standalone file alongside this guide.

<details>
<summary><code>hermes-vault-env.sh</code> (reference implementation)</summary>

```sh
#!/bin/sh
# hermes-vault-env.sh - HashiCorp Vault helper for the Hermes `command` secret source.
#
# Prints KEY=VALUE lines on stdout for Hermes' secrets.command loader, and
# maintains an atomic, 0600, last-known-good cache so the agent still boots
# when Vault is sealed or unreachable.
#
# usage: hermes-vault-env.sh [-c CACHE] [-a MAX_AGE] [-t TIMEOUT] [-A] KV_PATH [FIELD[:ENV_NAME] ...]
#
#   KV_PATH        KV path exactly as `vault kv get` expects it (v1 and v2
#                  mounts both work; v2 data nesting is handled).
#   FIELD[:ENV]    Map a Vault field to an env var name. Without :ENV the
#                  field name is upcased (dashes become underscores). With
#                  no FIELD args at all, every single-line string field at
#                  the path is exported under its upcased name.
#                  Listed fields are REQUIRED: if any is missing or blank,
#                  the whole fresh fetch is rejected and the cache is used,
#                  so a half-populated Vault path can never leave the agent
#                  running on a mixed credential set.
#
#   -c CACHE       Cache file (default: $HOME/.hermes/.vault-env.cache).
#   -a MAX_AGE     Max cache age in seconds for fallback; 0 = unbounded
#                  (default). Past the bound the helper fails instead of
#                  serving arbitrarily old secrets.
#   -t TIMEOUT     Vault client timeout in seconds (default 5). Keep it
#                  below secrets.command.helper_timeout_seconds: if the
#                  fetch outlives Hermes' own window, Hermes kills the
#                  helper and the cache fallback never runs.
#   -A             Allow cache fallback even when Vault answered
#                  "permission denied". By default auth failures do NOT
#                  fall back (so revoking a token actually revokes);
#                  sealed / unreachable / timeout always fall back.
#
# Requires: vault, jq. Auth comes from the normal Vault client environment:
# VAULT_ADDR plus a token (VAULT_TOKEN, ~/.vault-token, or a token helper).
# For unattended startup prefer Vault Agent auto-auth (AppRole); see the
# Hermes "HashiCorp Vault secrets" guide.
#
# Never log secret values: stdout is the only channel that carries them.
# One status line per run lands in CACHE.status (timestamp + outcome, no
# values) because Hermes discards helper stderr by design.

set -u

CACHE="$HOME/.hermes/.vault-env.cache"
MAX_AGE=0
TIMEOUT=5
AUTH_FALLBACK=0
REASON=""
NL='
'

usage() {
  awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0" >&2
}

while getopts c:a:t:Ah opt; do
  case $opt in
    c) CACHE=$OPTARG ;;
    a) MAX_AGE=$OPTARG ;;
    t) TIMEOUT=$OPTARG ;;
    A) AUTH_FALLBACK=1 ;;
    h) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
[ $# -ge 1 ] || { usage; exit 2; }
KV_PATH=$1; shift

case $KV_PATH in
  -*) printf 'hermes-vault-env: refusing KV path that starts with "-"\n' >&2; exit 2 ;;
esac
case $MAX_AGE$TIMEOUT in
  *[!0-9]*) printf 'hermes-vault-env: -a and -t take plain seconds\n' >&2; exit 2 ;;
esac
for spec in "$@"; do
  case $spec in
    ''|*[!A-Za-z0-9_:.-]*)
      printf 'hermes-vault-env: bad field spec: %s\n' "$spec" >&2; exit 2 ;;
  esac
done

STATUS="$CACHE.status"

# One-line breadcrumb (timestamp + outcome, never values). Hermes discards
# helper stderr, so this file is the only post-hoc signal of fresh vs stale.
note() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" > "$STATUS.tmp.$$" 2>/dev/null \
    && mv -f "$STATUS.tmp.$$" "$STATUS" 2>/dev/null
  printf 'hermes-vault-env: %s\n' "$1" >&2
}

serve_cache() {
  [ -r "$CACHE" ] || return 1
  if [ "$MAX_AGE" -gt 0 ]; then
    # GNU stat first: BSD stat also accepts -f (filesystem mode) and would
    # "succeed" with garbage if probed the other way around on GNU systems.
    mtime=$(stat -c %Y "$CACHE" 2>/dev/null || stat -f %m "$CACHE" 2>/dev/null)
    case $mtime in ''|*[!0-9]*) return 1 ;; esac
    age=$(( $(date +%s) - mtime ))
    if [ "$age" -gt "$MAX_AGE" ]; then
      note "fail: cache is ${age}s old, past the ${MAX_AGE}s bound ($1)"
      return 1
    fi
  fi
  note "fallback: served last-known-good cache ($1)"
  cat "$CACHE"
}

for bin in vault jq; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    serve_cache "missing dependency: $bin" && exit 0
    note "fail: missing dependency: $bin and no usable cache"
    exit 1
  fi
done

upcase_env() { printf '%s' "$1" | tr '[:lower:]-' '[:upper:]_'; }

# Values are always double-quoted. The Hermes dotenv parser strips exactly
# one layer of matching surrounding quotes, so this round-trips every value
# byte-for-byte (including leading/trailing spaces and embedded quotes).
# render() runs in a command substitution, so failure reasons go to stderr
# (captured by the caller); they name fields only, never values.
render() {
  if [ $# -eq 0 ]; then
    printf '%s' "$JSON" | jq -r '
      (.data.data // .data // {})
      | to_entries[]
      | select((.value | type) == "string")
      | select(.value | test("^\\s*$") | not)
      | select(.value | test("\n") | not)
      | select(.key | test("^[A-Za-z_][A-Za-z0-9_-]*$"))
      | ((.key | ascii_upcase | gsub("-"; "_")) + "=\"" + .value + "\"")
    '
  else
    for spec in "$@"; do
      f=${spec%%:*}
      e=${spec#*:}
      [ "$e" = "$spec" ] && e=$(upcase_env "$f")
      if ! printf '%s' "$e" | grep -qE '^[A-Za-z_][A-Za-z0-9_]*$'; then
        printf 'field %s maps to invalid env name %s' "$f" "$e" >&2; return 1
      fi
      v=$(printf '%s' "$JSON" | jq -r --arg f "$f" \
            '(.data.data // .data // {})[$f] // "" | tostring')
      case $v in
        *[![:space:]]*) ;;
        *) printf 'required field %s missing or blank at %s' "$f" "$KV_PATH" >&2; return 1 ;;
      esac
      case $v in
        *"$NL"*) printf 'field %s contains a newline (dotenv cannot carry it)' "$f" >&2; return 1 ;;
      esac
      printf '%s="%s"\n' "$e" "$v"
    done
  fi
}

ERRF=$(mktemp "${TMPDIR:-/tmp}/hve-err.XXXXXX") || ERRF=/dev/null
RSNF=$(mktemp "${TMPDIR:-/tmp}/hve-rsn.XXXXXX") || RSNF=/dev/null
JSON=$(VAULT_CLIENT_TIMEOUT="${TIMEOUT}s" vault kv get -format=json "$KV_PATH" 2>"$ERRF")
RC=$?

OUT=""
if [ "$RC" -eq 0 ]; then
  OUT=$(render "$@" 2>"$RSNF") || OUT=""
fi

if [ -n "$OUT" ]; then
  rm -f "$ERRF" "$RSNF"
  umask 077
  TMP="$CACHE.tmp.$$"
  if printf '%s\n' "$OUT" > "$TMP" && mv "$TMP" "$CACHE"; then
    note "fresh: $(printf '%s\n' "$OUT" | grep -c .) vars from $KV_PATH; cache refreshed"
  else
    rm -f "$TMP" 2>/dev/null
    note "fresh: served, but cache write failed"
  fi
  printf '%s\n' "$OUT"
  exit 0
fi

# Fresh fetch unusable: classify before falling back. A denied token must
# not be papered over by the cache (revocation has to actually revoke),
# while sealed / unreachable / timed out is exactly what the cache is for.
if [ "$RC" -ne 0 ] && [ "$AUTH_FALLBACK" -eq 0 ] \
   && grep -qiE 'permission denied|invalid token|code: 403' "$ERRF" 2>/dev/null; then
  rm -f "$ERRF" "$RSNF"
  note "auth-denied: Vault refused this token; not falling back (-A overrides)"
  exit 1
fi

if [ "$RC" -ne 0 ]; then
  REASON="vault kv get failed (sealed, unreachable, or timed out)"
elif [ -s "$RSNF" ]; then
  REASON=$(cat "$RSNF")
else
  REASON="no exportable fields at $KV_PATH"
fi
rm -f "$ERRF" "$RSNF"

serve_cache "$REASON" && exit 0
note "fail: $REASON and no usable cache"
exit 1
```

</details>

Behavior in one paragraph: it runs `vault kv get -format=json` with a client timeout, renders the selected fields as `KEY="VALUE"` lines, and only on a fully valid result atomically rewrites the `0600` cache and prints. On a sealed, unreachable, or timed-out Vault it prints the cached copy instead (optionally bounded by `-a MAX_AGE`). On `permission denied` it fails rather than serving the cache, so revocation propagates. Every run leaves a one-line, value-free status breadcrumb next to the cache.

## 3. Wire it into config.yaml

```yaml
secrets:
  command:
    enabled: true
    command: "$HOME/.hermes/scripts/hermes-vault-env.sh kv/hermes/env"
    helper_timeout_seconds: 10   # must stay ABOVE the helper's -t (default 5)
```

Two timeout layers matter here. Hermes kills the whole helper at `helper_timeout_seconds`; the helper caps the Vault fetch at `-t` seconds. Keep `-t` comfortably below `helper_timeout_seconds`, otherwise a hanging Vault eats the entire window and Hermes kills the helper before the cache fallback ever runs. Note the defaults make this real: Hermes ships `helper_timeout_seconds: 3` while the helper defaults to `-t 5`, so either raise the Hermes window as the example does, or pass `-t 2` if you keep the default window.

By default only explicitly listed fields or all fields at the path are exported, and `.env`/shell values win (`override_existing: false`). To pin the export list (recommended: an explicit list means an unexpected field in Vault can never inject an env var you didn't plan for):

```yaml
    command: "$HOME/.hermes/scripts/hermes-vault-env.sh kv/hermes/env telegram_bot_token discord_bot_token elevenlabs_api_key"
```

Listed fields are treated as required: if any one is missing or blank in Vault, the whole fresh result is rejected in favor of the cache, so a half-edited Vault path cannot leave the agent running on a mixed credential set.

## 4. Verify

```bash
# Run it as the Hermes user; you should see KEY="VALUE" lines:
~/.hermes/scripts/hermes-vault-env.sh kv/hermes/env

# Then check what Hermes applied:
hermes model   # detected keys show "(from Command helper)"

# And the breadcrumb:
cat ~/.hermes/.vault-env.cache.status
# 2026-07-26T18:02:11Z fresh: 3 vars from kv/hermes/env; cache refreshed
```

Test the fallback once before you rely on it: stop or seal Vault, run the helper manually, and confirm it prints the cached set with a `fallback:` status line.

## Auth patterns

The helper itself is auth-agnostic: it uses whatever the standard Vault client environment provides (`VAULT_ADDR` plus `VAULT_TOKEN`, `~/.vault-token`, or a token helper). `VAULT_ADDR` can live in `~/.hermes/.env`; Hermes loads `.env` before secret sources run, and the helper inherits it.

- **Vault Agent auto-auth (recommended for unattended hosts).** Run a Vault Agent with AppRole auto-auth and a file or token-helper sink; the helper then finds a fresh token without any interactive step. This is the canonical HashiCorp answer to "a daemon needs a token."
- **Plain token.** Fine for a workstation where you `vault login` interactively; the cache covers the gaps when the token has expired and you haven't logged in yet, but prefer agent auto-auth for anything that boots unattended.
- **Custom client wrappers.** If your site wraps Vault in its own CLI (an AppRole client backed by an OS keychain, for example), keep this script's cache-and-classify skeleton and swap the single `vault kv get` line for your client. The contract to preserve: fetch with a bounded timeout, validate structurally, write the cache atomically at `0600`, fall back only on availability failures, breadcrumb the outcome.

## Failure semantics

| Situation | Helper behavior | Agent outcome |
|---|---|---|
| Vault up, all fields good | Print fresh set, refresh cache | Current credentials |
| Vault sealed / unreachable / slow | Print cache (within `-a` bound) | Previous credentials, `fallback:` breadcrumb |
| Token revoked (`permission denied`) | Exit 1, no fallback (unless `-A`) | No values applied; `.env`/shell values remain |
| Required field missing or empty | Reject fresh set, print cache | Previous credentials; fix Vault, next boot is fresh |
| No cache yet and Vault down | Exit 1 | No values applied; startup continues (Hermes never blocks on secrets) |

## Security notes

- The cache holds real secret values at rest, `0600`, in `~/.hermes`. That is the deliberate price of boot-during-outage; if it is unacceptable, point `-c` at a tmpfs path and accept that a reboot empties the fallback exactly when Vault is most likely sealed.
- The status breadcrumb never contains values, only timestamps, counts, and field *names*.
- Hermes discards the helper's stderr by design (vault CLI diagnostics can carry sensitive material), which is why the breadcrumb file exists.
- Multi-line secrets (PEM keys and similar) cannot ride the `KEY=VALUE` transport; the helper rejects them explicitly instead of letting them truncate silently. Base64-encode such material into a single line and decode at the consumer if you must move it this way.

## Going further

If you outgrow the command-helper shape (per-key on-demand resolution, typed config, provenance labels per source, setup flows), the sanctioned path is a standalone [secret source plugin](/developer-guide/secret-source-plugin): subclass `SecretSource`, return a `FetchResult`, and let the orchestrator own precedence and environment writes. The cache-fallback semantics above port directly into a plugin's `fetch()`.

# Credential Vault (Password-Blind Autofill)

Let the agent log into websites **without ever seeing the password**, using
logins from a locally encrypted vault or from your password manager
(1Password, Bitwarden). The login identifier
(email/username/phone) is ordinary metadata the agent can see and type
itself; only the password is vault-secret — it is resolved server-side and
injected directly into the page.

## How it works

1. You add a credential with `hermes vault add` (interactive; the
   identifier is prompted normally, the password is read with a hidden
   prompt and never echoed or passed on the command line).
2. The password is encrypted at rest under `~/.hermes/vault/` (Fernet key +
   vault file, both `0600`) and the item is bound to an exact **origin**
   (`scheme://host[:port]`). The identifier is stored as item metadata.
3. When the vault has at least one item, two browser tools appear in the
   agent's toolset (they add zero schema cost otherwise):
   - `browser_vault_list` — handles + metadata, including the login
     identifier. Passwords are never returned.
   - `browser_vault_fill(handle)` — fills **only the password field** of
     the current page's login form.
4. The agent types the identifier itself with its normal input tools, then
   calls `browser_vault_fill`. Hermes checks that the **current page origin
   exactly matches** the credential's bound origin — once up front, and
   again synchronously inside the injected fill script immediately before
   the write (so a page that navigates mid-flight gets a refusal and zero
   bytes written). It classifies visible login fields (ported from
   OpenInstinct's login-control classifier — autocomplete tokens win,
   `new-password` / `one-time-code` fields are hard-excluded), picks the
   single best current-password field, injects the value over the
   supervised browser session's direct CDP WebSocket, and returns only
   `{filled_fields, kind, origin, success}`.

The password does not appear in the fill's tool result, logs, or the
session database, and its exact bytes are registered with the browser-result
redaction boundary so a later `browser_*` read that echoes the page's DOM is
scrubbed. See *What this does and does not guarantee* below for the limits.

## CLI

```bash
# Add a login (interactive wizard; password is hidden)
hermes vault add

# List items — identifiers and origins shown, passwords never
hermes vault list

# Remove an item by handle
hermes vault rm vault_ab12cd34ef56
```

Item kinds: `login`, `payment`, and `address` are all stored (`payment` and
`address` payloads remain fully secret); Phase 1 browser fill supports
`login` items only.

## Password managers (1Password, Bitwarden)

You don't have to copy logins into the Hermes vault. Enable a password
manager and its website logins become fillable handles alongside the local
ones (`op:…` for 1Password, `bw:…` for Bitwarden). The password is fetched
from the manager's CLI at fill time only and follows the same server-side
injection, origin binding, and redaction as a local item.

```bash
hermes vault sources                        # status of each manager
hermes vault sources --enable onepassword   # needs the `op` CLI on PATH
hermes vault sources --enable bitwarden     # needs the `bw` CLI; run `bw login` once first
```

or **Desktop → Settings → Credential Vault → Password managers** (toggle,
Unlock, Lock).

### Unlocking is per session

A manager starts **locked**. The first time the agent needs one of its
logins it asks you to unlock: a masked master-password prompt appears in
the CLI, TUI, or Desktop chat (or you can unlock ahead of time from
Settings). Hermes hands the master password to the manager CLI through its
non-interactive channel (`op signin` reads stdin; `bw unlock --passwordenv`
reads a variable set only in the child process) — never as a command-line
argument, never in Hermes' own environment — and keeps only the resulting
session token in memory, scoped to the current profile. The token expires
after 30 minutes idle, when you press **Lock**, or when the chat session that
unlocked it ends (other sessions in the same profile keep their own unlocks).
A **Lock** pressed while an unlock is still in flight wins. The agent never
sees the master password, the token, or any password.

`browser_vault_list` reports a locked manager under `locked`, and
`browser_vault_unlock(backend)` triggers the prompt explicitly.

### Headless sessions never prompt

Cron jobs, webhooks, the API server, and `hermes chat -q` have nobody to
answer a prompt, so a locked manager is reported as
`unavailable_in_this_session` and fills refuse — the same posture command
approvals take there. Unlock from an interactive session or the Desktop app
first (the token is per process, so a running gateway that you unlock from
a chat keeps serving its own cron jobs), or give 1Password a service-account
token (`OP_SERVICE_ACCOUNT_TOKEN`) to skip the prompt entirely. The local
vault needs no unlock and keeps working everywhere.

```yaml
vault:
  onepassword:
    enabled: true
    account: ""            # `op --account` shorthand; empty = default
    service_account_token_env: OP_SERVICE_ACCOUNT_TOKEN
  bitwarden:
    enabled: true
```

Bitwarden here means the **Password Manager** (`bw`) — website logins — not
the Secrets Manager (`bws`) that the [secrets](../secrets/bitwarden) feature
uses for API keys.

## Desktop app

Desktop users can manage the vault without a terminal: open
**Settings → Credential Vault** (right next to the Browser section). The
panel lists saved items — label, kind, login identifier, origin, and
creation date; passwords are never displayed — and lets you add or delete
credentials. The
Add dialog adapts to the selected kind (login / payment card / address),
masks secret fields, and submits them straight into the encrypted store
over the local gateway connection.

The panel is deep-linkable: opening

```text
hermes://open/settings?tab=vault&kind=login&label=github&origin=https://github.com
```

launches the app on the vault panel with the Add dialog pre-filled from
the query parameters (metadata only — a secret can never travel in a
link). When a fill request fails because no matching item exists, the
agent's error message points at both `hermes vault add` and this panel.

## Example agent flow

```
User: log into example.com and check my dashboard
Agent: browser_navigate("https://example.com/login")
Agent: browser_vault_list()          → {items: [{handle: "op:…", backend: "onepassword", label: "Example", identifier: "me@example.com", origin: "https://example.com"}]}
       (or, if 1Password is still locked: {items: [], locked: [{backend: "onepassword", unlock: "browser_vault_unlock"}]} → the agent calls browser_vault_unlock and you get a masked prompt)
Agent: fill_input(<username field>, "me@example.com")
Agent: browser_vault_fill("op:…")    → {"success": true, "filled_fields": 1, "backend": "onepassword", "kind": "login", "origin": "https://example.com"}
Agent: browser_click(<submit>)
```

## Security properties

- **Password-blind:** the agent never sees password values — only handles,
  labels, identifiers, and origins.
- **Origin-bound at use time:** fills are refused unless the page origin
  exactly matches (scheme + host + port) the origin the credential was
  saved for — asserted both before the fill and atomically inside the fill
  script itself, so a mid-flight navigation (including cross-origin) writes
  nothing.
- **No argv exposure:** the secret-bearing injection runs exclusively over
  the supervised browser session's CDP WebSocket. If that session is not
  available, the fill refuses rather than falling back to a subprocess
  path that would place the password in argv.
- **Redaction-backed egress boundary:** filled password bytes are
  registered with the browser tool-result redactor for the life of the
  process; every `browser_*` result (including raw `browser_cdp` output)
  is scrubbed against them.
- **No signup/OTP capture:** fields marked `autocomplete="new-password"`
  or `one-time-code`, and fields labeled *new/confirm/create/repeat
  password*, are never filled.
- **Encrypted at rest:** vault file and key are created `0600` in your
  Hermes home; nothing is sent to any server.
- **Master password never stored:** for 1Password/Bitwarden the master
  password is consumed by the manager CLI and dropped; only the session
  token is held, in memory, per profile, with an idle timeout. Headless
  sessions can't prompt and see the manager as locked.

### What this does and does not guarantee

The vault keeps passwords out of the model's *normal* path: list/fill
results carry metadata only, the fill runs over the supervised CDP socket,
and every browser tool result is scrubbed for the exact filled bytes
(including the CR/LF-normalized form a text input stores, and JSON object
keys). This is accidental-disclosure protection, not an execution sandbox:
a session that also has arbitrary page JavaScript (`browser_cdp
Runtime.evaluate`) or host code execution could in principle transform a
filled value (for example base64-encode it) into a string the redactor does
not recognize. If that matters for a credential, restrict the session's
toolset (drop `browser_cdp`/`terminal`/`execute_code`) or use a dedicated
low-privilege account for agent logins. Treat the filled credential as
exposed to the same trust boundary as the browser session itself.

## Notes

- No configuration is needed for the local vault; the tools activate
  automatically once it has an item or a password manager is enabled.
- The fill targets the single best current-password field (autocomplete
  token beats type heuristics; ties break in DOM order).
- Design ported from Merit-Systems/OpenInstinct's opaque-handle vault
  autofill (MIT).

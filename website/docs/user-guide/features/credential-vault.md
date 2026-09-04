# Credential Vault (Password-Blind Autofill)

Store site logins in a locally encrypted vault and let the agent log into
websites **without ever seeing the password**. The login identifier
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

The password never appears in tool results, logs, or the session database.
Its exact bytes are additionally registered with the browser-result
redaction boundary, so even a later `browser_cdp` read that manages to echo
the page's DOM cannot return them to the model.

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
Agent: browser_vault_list()          → [{handle: "vault_…", label: "Example", identifier: "me@example.com", origin: "https://example.com"}]
Agent: fill_input(<username field>, "me@example.com")
Agent: browser_vault_fill("vault_…") → {"success": true, "filled_fields": 1, "kind": "login", "origin": "https://example.com"}
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

## Notes

- No configuration is needed; the tools activate automatically once the
  vault has an item.
- The fill targets the single best current-password field (autocomplete
  token beats type heuristics; ties break in DOM order).
- Design ported from Merit-Systems/OpenInstinct's opaque-handle vault
  autofill (MIT).

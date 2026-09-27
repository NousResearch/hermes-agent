---
title: Passwords & Logins
description: Fill saved site credentials without sending them in the vault tool result.
---

# Passwords & Logins

Say **"log into GitHub"** and the agent signs in for you. The first time it
reaches a sign-in page it has no login for, it asks you, right there, in a
masked prompt. After that it just works. Passwords are encrypted on this
machine and the vault fill tool sends them directly to the selected page rather
than returning them to the model. Later browser inspection is a separate risk;
see [What this does and does not guarantee](#what-this-does-and-does-not-guarantee).

There is nothing to set up.

## What it looks like

**CLI / TUI**

```
🔐 Save login for github.com
   The agent reached a sign-in page with no saved login for this site.
   Type the email / username you sign in with (shown), then Enter.
   ...
   Now the password (hidden). It is encrypted on this machine, bound to
   https://github.com, and filled into the page without appearing in the
   vault tool's result.
```

**Desktop** — a "Save your github.com login?" card with an identifier field and
a masked password field. *Save & sign in* stores it and continues; *Don't save*
tells the agent to stop asking for this turn.

From then on the agent lists your saved logins, types the identifier itself and
fills the password through Hermes. The tool result it sees is
`{filled_fields: 1, origin: "https://github.com"}`. The password's literal
value is registered for best-effort redaction of subsequent browser text in
this process. This is not a guarantee against arbitrary page reads.

## Two-factor codes

Sites that ask for a code after the password are handled the same way:

- **Authenticator key saved with the login** (the "setup key" or `otpauth://`
  link a site shows when you enable 2FA; 1Password and Bitwarden items that
  hold a TOTP seed count too): Hermes generates the current code and enters
  it. Nobody is asked. Add the key in **Settings → Passwords & Logins → Add**
  or `hermes vault add`; the item shows a *2FA auto* badge.
- **Code sent to your phone or email**: a small prompt appears in your
  surface ("Verification code for github.com"), you type the code, Hermes
  enters it into the page. The code is omitted from the vault tool's response;
  later page inspection may still expose it.
- **Passkeys, hardware keys, app approvals** ("tap Approve in Duo"): nothing
  to type. The agent tells you to complete it on your device and waits for
  the page to move on.

## Already using 1Password or Bitwarden?

Nothing to enable. If the `op` or `bw` command-line tool is installed and signed
in, Hermes picks it up automatically and its website logins become fillable
alongside the local ones. The first time the agent needs one of those logins it
asks you to unlock the manager with your master password (masked prompt; once
per session, 30 minutes idle). Hermes hands the master password to the manager's
CLI through its non-interactive channel (`op signin` on stdin, `bw unlock
--passwordenv` in the child's environment) and keeps only the session token in
memory. The agent never sees the master password, the token, or any login.
A manager item that lists several websites (say `amazon.co.uk`,
`www.amazon.co.uk` and `eu.account.amazon.com`) fills on each of those exact
origins; nothing is inferred beyond the URLs saved on the item.

Prefer not to use a detected manager? `hermes vault sources --disable bitwarden`,
or the switch in **Settings → Passwords & Logins**.

## Paying and filling addresses

Cards and addresses work the same way as logins: saved once (**Settings →
Passwords & Logins → Add**, or `hermes vault add`), bound to the checkout site,
and filled by the agent on that site only. **Every card fill asks you first**,
with the same approval prompt as a dangerous command; declining writes nothing.
Headless sessions (cron, webhooks, the API server) cannot confirm and are
refused, so a prompt injection that reaches a checkout page can ask, but it
cannot spend. Address fills need no confirmation.

## Managing what's saved

- **Desktop → Settings → Passwords & Logins**: everything saved, the detected
  password managers with Unlock/Lock, Add, Remove.
- **CLI**: `hermes vault list`, `hermes vault add`, `hermes vault rm <handle>`,
  `hermes vault sources`.

Items live encrypted under `~/.hermes/vault/` (Fernet key + vault file, both
`0600`), scoped to the profile. Labels, site origins and login identifiers are
visible metadata; passwords and card values are released only through the
private browser transport into the page. A remote browser server necessarily
receives them (see the trust boundary below).

## Browser support and Camofox

Vault tools use the selected browser backend, never a second hidden browser:

- **Supervised Chromium / CDP sessions**, including Browser Use sessions with
  a supervisor, fill over the direct CDP WebSocket. Without that channel, secret
  fills refuse rather than put a credential in a command-line argument.
- **Camofox** uses its authenticated `/tabs/{id}/evaluate` endpoint. Navigate
  through Hermes first: the vault requires the existing task tab and does not
  create or adopt a tab itself. The tab, server, user identity and authentication
  are pinned across origin inspection, user prompts and filling, including the
  nested fill after saving a login. A missing tab, unavailable evaluation endpoint
  or transport failure refuses; there is no Chromium/CDP fallback.

For Camofox, configure `CAMOFOX_API_KEY` and use **HTTPS with certificate
verification**, or **HTTP on a numeric loopback address** such as
`http://127.0.0.1:9377` for a local server or SSH tunnel. Plain HTTP requires a
numeric loopback host; `localhost` and other DNS names do not qualify.
URL-embedded credentials and redirects are refused. Vault requests ignore ambient proxy and `.netrc` settings so a
loopback connection cannot silently send credentials through another host.
Passwords, cards, addresses, save-login prompts and verification-code entry use
this same transport. Saved authenticator codes are restricted to the login's
saved origins; user-entered codes are bound to the page inspected before prompting.

**Trust the browser server as you would the browser itself.** Camofox receives
the secret-bearing expression in an in-memory JSON request. Its operator,
plugins, request-body logging, browser tracing/recording and debugging tools can
observe it. Use a server and plugins you trust, and disable secret-bearing
request logging and traces. Hermes catches page-evaluation exceptions before
Camofox can log their text, discards raw transport errors, and accepts only a
bounded fill count or a known refusal from a secret evaluation. This does not
hide credentials from the server or from the destination page.

## Headless sessions

Cron jobs, webhooks, the API server and `hermes chat -q` have nobody to answer a
prompt. Saved local logins keep working there; a locked password manager reports
`unavailable_in_this_session` and a missing login reports `prompt_unavailable`.
Unlock or save from an interactive session first, or give 1Password a service
account token (`OP_SERVICE_ACCOUNT_TOKEN`).

```yaml
vault:
  onepassword:
    enabled: false          # opt OUT of a detected manager (default: on when installed)
    account: ""             # `op --account` shorthand; empty = default
    service_account_token_env: OP_SERVICE_ACCOUNT_TOKEN
  bitwarden:
    enabled: false
```

## What this does and does not guarantee

**Does:** the vault tool omits the credential from its own result, CLI
arguments and session metadata. It fills over the supervised browser session's
direct CDP socket or the pinned Camofox transport described above. Fills are
refused unless the page origin exactly matches the saved origin, checked again
inside the page immediately before the write. Nonce-stamped controls bind each
write to its inspection; a changed tab/document or restamped control cannot
retarget it.

**Does not:** confine the credential after it reaches the page. The page and
its scripts can read it. Browser tools that evaluate page JavaScript can also
read or transform form values; screenshots/vision may capture revealed fields.
Exact-value redaction of browser text is best effort, retains limited values
in the current process only, and is lost after a process restart. Masked dots
are a display choice, not an inspection boundary. The browser server and any
cloud browser operator can observe page content or secret-bearing requests.
Never ask the agent to inspect a filled password/card field, and use only a
browser/server you trust. Origin binding prevents filling on the wrong site;
it does not make the authorized site safe.

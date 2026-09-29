---
title: Passwords & Logins
description: The agent signs into sites, pays and fills addresses for you without ever seeing a password.
---

# Passwords & Logins

Say **"log into GitHub"** and the agent signs in for you. The first time it
reaches a sign-in page it has no login for, it asks you, right there, in a
masked prompt. After that it just works. Passwords are encrypted on this
machine and injected straight into the page; the model never sees them.

There is nothing to set up.

## What it looks like

**CLI / TUI**

```
🔐 Save login for github.com
   The agent reached a sign-in page with no saved login for this site.
   Type the email / username you sign in with (shown), then Enter.
   ...
   Now the password (hidden). It is encrypted on this machine, bound to
   https://github.com, and filled into the page without the model ever seeing it.
```

**Desktop** — a "Save your github.com login?" card with an identifier field and
a masked password field. *Save & sign in* stores it and continues; *Don't save*
tells the agent to stop asking for this turn.

From then on the agent lists your saved logins, types the identifier itself and
fills the password through Hermes. The tool result it sees is
`{filled_fields: 1, origin: "https://github.com"}`; the password is also
registered with the redactor so a later page read cannot echo it back.

### Desktop preview or managed browser

The Desktop preview and the managed automation browser are separate pages.
In a Desktop session with preview tools, `browser_vault_fill`,
`browser_vault_save_login`, and `browser_vault_enter_code` default to the preview
already open beside the chat. This also works when preview tools are discovered
on demand and does not require a separate browser runtime. The agent can select
`target: "preview"` explicitly, or `target: "browser"` for the separate managed
browser. Sessions without preview tools keep the managed-browser default.

Keep the chat and its preview page selected while filling. Hermes binds each
operation to that window and page, and refuses if the target changes or navigates.
Vault requests are live-only and are not replayed after a reconnect. A failed
preview fill never switches to a different browser. Filling does not submit
the login form.

## Two-factor codes

Sites that ask for a code after the password are handled the same way:

- **Authenticator key saved with the login** (the "setup key" or `otpauth://`
  link a site shows when you enable 2FA; 1Password and Bitwarden items that
  hold a TOTP seed count too): Hermes generates the current code and enters
  it. Nobody is asked. Add the key in **Settings → Passwords & Logins → Add**
  or `hermes vault add`; the item shows a *2FA auto* badge.
- **Code sent to your phone or email**: a small prompt appears in your
  surface ("Verification code for github.com"), you type the code, Hermes
  enters it into the page. The code never enters the conversation either.
- **Passkeys, hardware keys, app approvals** ("tap Approve in Duo"): nothing
  to type. The agent tells you to complete it on your device and waits for
  the page to move on.

## Already using 1Password or Bitwarden?

Nothing to enable. If the `op` or `bw` command-line tool is installed, Hermes
picks it up automatically and its website logins become fillable alongside the
local ones. When an unlock is needed and the backend supports it, 1Password uses
native app authorization by default; approve it on the computer running the Hermes
backend. Bitwarden uses a masked master-password prompt. Unlocks last once per
session, with a 30-minute idle timeout. Hermes passes manual passwords only
through the manager's non-interactive CLI channel (`op signin` on stdin, `bw
unlock --passwordenv` in the child's environment) and keeps session tokens only
in memory. The agent never sees the master password, the token, or any login.
A manager item that lists several websites (say `amazon.co.uk`,
`www.amazon.co.uk` and `eu.account.amazon.com`) fills on each of those exact
origins; nothing is inferred beyond the URLs saved on the item.

For **1Password desktop-app integration**, the existing Settings unlock dialog
uses **Unlock with 1Password** by default; choose **Use a password instead** for
manual sign-in. Enable **Settings → Developer → Integrate with 1Password CLI**
and Windows Hello on Windows. For an SSH or remote connection with an interactive
desktop, approve the request on that remote computer. You do not need to
enter your 1Password master password into Hermes for native authorization. In
an interactive agent session, `browser_vault_unlock(backend="onepassword")`
also uses native authorization by default; use `method="password"` to request
the manual prompt. If native authorization is unavailable, Hermes reports the
problem and you can choose the manual password path.

Desktop authorization does not produce a CLI session token. Hermes verifies it
with a metadata-only vault listing and keeps a profile-scoped unlock lease in
memory. Lock, session teardown, and the 30-minute idle expiry still release that
lease; checking status never opens an authorization prompt. A failed or declined
app request leaves Hermes locked. Manual password sign-in remains available in
the same dialog when the backend supports it.

### Remote and unattended profiles

A remote profile authenticates on its backend host. Unlocking 1Password on your
Desktop computer does not unlock a remote gateway. Passwords & Logins shows the
methods reported by the selected backend; a headless host offers no desktop-app
or master-password prompt. An older backend without this metadata shows a
compatibility notice and uses its existing manual flow.

For unattended access, configure an existing approved 1Password service account
or Connect identity in that profile's secret store. Connect requires both
`OP_CONNECT_HOST` and `OP_CONNECT_TOKEN`; a partial pair refuses access, and a
complete pair takes precedence over a service-account token. A configured identity
is not proof of authentication: actual item access must succeed.

Select the intended vault in the profile's `config.yaml` for service-account
access, especially when the identity has more than one vault:

```yaml
vault:
  onepassword:
    enabled: true
    vault: "Automation"
```

The selector applies to listing logins, passwords and saved authenticator codes.
Give each profile its own appropriately scoped identity. Service accounts and
Connect have vault-access restrictions; check the provider's
[service-account limits](https://www.1password.dev/service-accounts/get-started)
and [Connect prerequisites](https://www.1password.dev/connect/get-started).

A remote Desktop session fills its own selected local preview through the live
connection. Headless gateway, API and scheduled sessions use their backend's
supervised browser instead. That browser needs its own installed runtime and
separate profile state. Saving a new login or requesting a code requires an
interactive prompt; an unattended session can use existing permitted logins
and saved authenticator codes.

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
visible metadata; passwords and card values never leave the vault except into
the page.

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

**Does:** the password never enters the model's context through Hermes: not in
tool results, logs, the session database, or the CLI arguments of any process.
Fills use the supervised browser session's direct CDP socket, or the Desktop's
authenticated preview request channel for `target: "preview"`. They are refused
unless the page origin exactly matches the saved origin, checked again inside
the page immediately before the write. Saved authenticator codes are bound to
the login's origins too.

**Does not:** protect against the page itself. Once a password is typed into a
site, that site (and any script it runs) has it, exactly as when you type it
yourself. On a cloud browser backend the vendor's browser sees the page like any
other. The origin binding is the guard against filling on the wrong site, not
against a compromised right one.

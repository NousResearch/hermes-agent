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

## Dashlane (explicit opt-in)

Dashlane is **off by default**, even when `dcli` is installed. This integration
supports explicitly opted-in exact-host discovery or fixed login metadata.
Install and enroll Dashlane CLI yourself, then run `dcli sync` in your own
terminal to unlock it. Never send your master password, device keys or codes
to the agent. Hermes does not run enrollment, sync or lock commands for you.

Edit the active profile's `config.yaml` with metadata only:

```yaml
vault:
  dashlane:
    enabled: true
    binary_path: ""  # PATH lookup; alternatively an absolute trusted dcli executable path
    account: "owner@example.test"  # exact Login value from your own dcli status
    search_hosts: [service.test]  # explicit trusted-provider projection opt-in; no UUID needed
    items: []
```

`browser_vault_list` now discovers saved logins for the configured exact hosts
without asking you for a UUID. Up to five lowercase ASCII hosts are allowed.
This opt-in accepts a specific trust cost: dcli decrypts all logins internally,
then sends complete URL-substring-matching records (including their secrets)
to the trusted Hermes backend. Hermes independently filters exact hosts and
returns **only allowlisted metadata and opaque handles**. There is no vendor
metadata-only API and `--field` does not strip secrets from its JSON. Never run
the provider password JSON command in an agent terminal. No all-record export,
clipboard, plaintext secret file, title lookup or alternate fill helper is used.

Review the returned identifier and exact saved origin. If several logins match,
the agent must ask you to select one, never choose the first. Handles expire after
five minutes or a process restart; list again if needed. At most 20 vendor
matches per host and 256 KiB stdout per subprocess are allowed; excess fails
closed rather than returning a partial list. Only `browser_vault_fill` consumes
the selected password, on its bound exact origin. `service.test` does **not** authorize
`login.other-service.test`, `www.service.test`, another scheme or another port.
See the [trusted projection security review](../../developer-guide/dashlane-vault-security.md).

Existing users may instead leave `search_hosts: []` and configure fixed `items`
with `id`, `label`, `origin`, `identifier`, and `identifier_type` (`email` or
`username`). No search occurs unless hosts are explicitly configured.

### Saved bare hostnames: explicit origin binding

A saved website such as `service.test` has no scheme and therefore no exact origin.
Hermes does **not** infer `https://service.test` or authorize a related login domain.
Instead, listing returns `unfillable_candidates`: allowlisted `backend`, exact
`source_id` (UUID), `website_host`, `identifier`, and `identifier_type`, with
`available: false`, `fillable: false`, `stage: search_projection`,
`reason: invalid_origin`, and `status: explicit_origin_binding_required`.
These are metadata only, with no fill handle. Several candidates require user
selection; no candidate is automatically enrolled or selected.

After selecting the exact record and explicitly approving its destination,
the user may configure an `items` entry with those exact identity fields,
a label, an exact normalized destination `origin`, and `source_host` equal to
the provider's saved bare hostname. For example (synthetic metadata):

```yaml
items:
  - id: "ABCDEF01-2345-4567-89AB-0123456789AB"
    label: "Selected login"
    identifier: "person@example.test"
    identifier_type: email
    source_host: service.test
    origin: "https://sso.example.test"
```

This is an explicit per-record authorization, not automatic URL repair or a
domain alias rule. `source_host` must be a lowercase ASCII bare hostname, not
a URL, wildcard or path. Fill re-reads the exact UUID and requires its identifier
and saved website string to still match exactly. A change to `https://service.test`,
capitalization, a subdomain or a path fails closed. The browser must still match
the configured destination's exact scheme, host and port; page-side origin
revalidation remains in force. Without `source_host`, fixed items continue to
require the provider's saved URL origin to match `origin`.

IDs must be uppercase UUIDs without braces or a `dl://` prefix;
the inspected CLI accepts UUID version digits 0–5. Hermes rejects invalid IDs
rather than converting them into title searches. Origins must be normalized
HTTP(S) origins: no path, trailing slash, user info, query or fragment. Explicit
default ports are normalized away. Use ASCII DNS/IPv4 hosts (punycode for IDNs);
IPv6 literals and escaped/backslash hosts are not supported. At most 100 items may be enrolled; duplicates
and unknown configuration/metadata fields are rejected. Do not put passwords,
TOTP seeds or API keys in this configuration. Labels and identifiers are visible
to the agent. The account identifies your Dashlane vault, not the website login.

`hermes vault sources --enable dashlane` (or the Desktop switch) enables use;
it does not enroll items or unlock Dashlane. Disable with
`hermes vault sources --disable dashlane`. Desktop displays manual terminal
instructions instead of unsupported Unlock/Lock buttons. `vault.unlock` and
named `vault.lock` for Dashlane refuse with those instructions. An unnamed
`vault.lock` only forgets Hermes-managed session tokens; it **does not lock
Dashlane**. Disabling the source does not lock Dashlane either. To lock it,
run `dcli lock` yourself. Dashlane's unlock lifetime belongs to dcli, not
Hermes's 30-minute idle token policy, and can be shared across Hermes profiles.

An already unlocked, correctly enrolled source can fill in headless sessions;
a locked source reports `manual_cli` guidance. Hermes checks dcli account/lock
status before and after each selected-item read, then validates its exact ID,
website identifier and origin. These separate subprocess checks are **not an
atomic account pin**: do not switch Dashlane accounts concurrently with fills.
`status` can access OS Keychain; search and selected reads may trigger Dashlane
sync and network access. Without search hosts, listing uses enrolled metadata
plus status checks. Search uses the trusted projection described above.
A selected read returns the complete selected record internally,
including other secret fields; only the password is used and none are returned
to the model. No Dashlane automatic TOTP, payment/address import, vault writes,
SSO enrollment automation or native Chrome integration is provided. This feature
uses the existing supervised browser/CDP fill path; native Chrome is unverified.
See the [security review](../../developer-guide/dashlane-vault-security.md).

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
Fills happen over the supervised browser session's direct CDP socket and are
refused unless the page origin exactly matches the saved origin, checked again
inside the page immediately before the write.

**Does not:** protect against the page itself. Once a password is typed into a
site, that site (and any script it runs) has it, exactly as when you type it
yourself. On a cloud browser backend the vendor's browser sees the page like any
other. The origin binding is the guard against filling on the wrong site, not
against a compromised right one.

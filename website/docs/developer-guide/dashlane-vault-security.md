---
title: Dashlane vault security boundary
---

# Dashlane vault security boundary

This is an opt-in login source, not a general Dashlane connector. The inspected
upstream contract is Dashlane CLI `v6.2640.1`, commit
[`38bf89a6b0d987a7993bb0584259ddf422a52163`](https://github.com/Dashlane/dashlane-cli/tree/38bf89a6b0d987a7993bb0584259ddf422a52163).
Other versions are not certified; Hermes does not enforce a version pin.

## Trusted projection, not a metadata-only vendor API

The earlier UUID-only implementation could not discover a site. Search now uses
an explicitly authorized **trusted backend projection**, the same kind of trust
boundary as Bitwarden's internal `bw list items` projection. The model-facing
secret policy is unchanged; the subprocess response is **not metadata-only**.

With `vault.dashlane.search_hosts: [service.test]`, `browser_vault_list` invokes, inside
the adapter only, `dcli password --output json url=service.test`. Do not run that command
through a model-visible terminal or use a standalone export/filter/fill helper.
No `--field` flag is claimed to remove secrets. There is no invented metadata API.

The inspected source establishes:

- `src/commands/index.ts` supports `password --output json` and positional
  `key=value` filters.
- `src/utils/filterCredentials.ts` implements case-insensitive **substring**
  filtering. One `url=host` filter is used; no empty query, title lookup, regex,
  wildcard, unfiltered all-record JSON export or fallback is permitted.
- `src/command-handlers/passwords.ts` selects and decrypts **all active logins
  internally before filtering**. Complete matching records, including passwords,
  OTP material and other fields, then arrive on stdout in trusted Hermes memory.
  A narrow vendor filter does not bound the provider's internal scan/decryption.
  It can also return false substring matches; their complete secrets cross this
  boundary before Hermes independently rejects them.
- JSON output precedes `--field` handling. Neither a field flag nor a Python
  projection changes this provider-internal decryption or matching-record
  stdout trust cost. Enabling `search_hosts` explicitly accepts those costs.
- `read dl://UUID` uses SQL ID equality and returns the complete selected record.
  Password consumption remains solely in existing `browser_vault_fill` routing.

The CLI executable, runtime/dependencies, OS account, Keychain and Hermes process
are trusted. An absolute `binary_path` avoids accidental PATH selection but is
not signing or integrity verification. Python strings/JSON objects cannot be
securely zeroized; no zeroization claim is made. No captured provider output or
secret is written to files, logs, model results, cache, clipboard or arguments.
Metadata fields (label, identifier, exact origin) are intentionally visible;
users should not store secrets in those fields. Unknown fields are never copied.

## Implemented no-UUID onboarding

1. The user enrolls/unlocks the official CLI in their own terminal (`dcli sync`).
   Hermes never captures its master password or runs enrollment/sync/lock commands.
2. In the active profile, configure `enabled: true`, the exact Dashlane `account`,
   and explicit lowercase ASCII `search_hosts`, e.g. `[service.test]`. No item UUID is
   needed. The backend's reusable `search_items(host)` accepts only configured
   exact hosts. At most five hosts, each at most 253 characters, are admitted.
3. `browser_vault_list` searches these hosts and independently parses each saved
   URL, admitting only **exact hostname equality**, never suffix/subdomain/path
   matches. Scheme and port remain part of the returned saved origin. At most
   20 vendor matching records per host are admitted; excess/invalid output refuses
   the entire search, not a truncated apparent success.
4. For valid URL origins, only allowlisted `VaultItemMeta` fields and random `dl:search-…` handles leave
   the backend. Handles are process-local, expire after five minutes and bind
   the active profile path plus entire adapter configuration. Only nonsecret
   UUID/metadata bindings are retained; the cache is capped at 1,000 entries.
   Process restart, expiry or configuration changes require listing again.
5. Multiple candidates are returned as choices, with `selection_required` telling
   the agent to ask the user for identifier and exact saved origin. There is no
   first-result auto-selection or site/title-to-password resolution. An explicit
   selected handle is required by `browser_vault_fill`. Like the existing vault
   workflow, this is conversational user selection, not a new cryptographically
   attested confirmation UI or persistent enrollment writer.
6. Existing fill routing checks browser origin **before** selected-record secret
   resolution. Account and lock state are checked before and after the UUID read;
   ID, identifier and saved exact origin (or explicitly bound bare source host,
   described below) must still match. Page JavaScript checks
   origin again immediately before filling. No new terminal injection route exists.

Fixed UUID `items` remain supported without search. Empty `search_hosts` never
implicitly triggers discovery. Configuration and metadata are profile-scoped;
dcli enrollment/Keychain state is OS-user-scoped and can be shared by profiles.

Searching `service.test` does not authorize `https://login.other-service.test` or even
`https://www.service.test`. Likewise HTTP or a different port does not match an HTTPS
saved origin. Search itself never authorizes a cross-site alias. Selected reads
also verify the provider URL; the only bare-host exception is the explicit
per-record configuration contract below.

### Bare-host projection and explicit source binding

A saved bare hostname has no origin. If it exactly equals the opted-in search
host, projection may expose `unfillable_candidates` with only `backend`,
`source_id` (exact UUID), `website_host`, `identifier`, `identifier_type`,
`available: false`, `fillable: false`, `stage: search_projection`,
`reason: invalid_origin`, and `status: explicit_origin_binding_required`.
No HTTPS origin is inferred, no handle is minted and no discovered capability
is cached for that candidate. Unrelated bare hosts are excluded. Multiple
candidates require user selection, not first-match resolution.

Explicit `items[].source_host` binds one configured exact UUID and identifier
to one exact lowercase ASCII bare provider hostname, with `items[].origin`
separately naming the user-authorized normalized destination origin. This is
a trusted configuration assertion, not authority derived from provider search,
a page, or a model-supplied alias. The adapter does not write this configuration.
It permits a deliberately approved destination different from the saved bare
host, but does not relax the browser's exact-origin gate or create wildcard,
suffix or subdomain matching. Without `source_host`, existing saved-URL origin
validation is unchanged.

On selected read, exact provider ID, identifier and raw website equality to
`source_host` are required. URL/scheme, case, path and subdomain changes refuse;
invalid configured source hosts refuse before spawning. Account/lock state is
checked before and after reading, and configuration scope is checked before
returning the password. Existing `browser_vault_fill` checks the destination
origin before secret resolution and again in page-side JavaScript immediately
before injection. `source_host` is never an additional allowed browser origin.

Synthetic `test_dashlane_origin_binding.py` covers metadata-only candidates,
ambiguity, source/UUID/identifier/account drift, invalid configuration, the
existing fill path and wrong destination refusal before any selected read.

## Process and failure boundaries

The child receives an environment allowlist (basic path/home/temp/Windows system
paths plus `NO_COLOR`), closed stdin, discarded stderr and no shell. Dashlane/Node
override variables are not inherited. Stdout is bounded to 256 KiB and decoded
strictly; duplicate JSON keys/IDs, malformed metadata and ambiguous status fail
closed. Each subprocess has a 15-second deadline; cleanup waits are bounded.
On POSIX an isolated process group is killed, including ordinary descendants
retaining stdout. Unbuffered pipes avoid a buffered-reader close deadlock. Windows
kills only the immediate child, not necessarily its descendants. Neither platform
can guarantee cleanup of escaped descendants or uninterruptible OS processes.
Errors from this adapter are fixed messages without captured stdout/stderr;
Dashlane listing also has an outer fixed-error projection.

`status` may access Keychain; search/read may synchronize over the network. These
are not offline-only or side-effect-free provider reads. Exact account checks
before/after detect persistent drift, but cannot prevent switch-and-switch-back
races between subprocesses. There is **no atomic account pinning guarantee**;
do not switch Dashlane accounts concurrently with search/fill. Bounds constrain
Hermes output and subprocess time, not total provider-internal work or memory.

No automatic Dashlane OTP is implemented: the existing OTP route lacks the
handle-to-origin precheck required before admitting a new automatic provider.
User-mediated code entry is unchanged. No vault writes, payment/address import,
SSO automation, desktop scraping or verified native Chrome extension bridge is
provided. The authorized page and its scripts can access a filled password.

## Unlock semantics

Desktop shows manual instructions, not a Dashlane password prompt/Lock button.
Named Dashlane unlock/lock RPCs refuse. Unnamed `vault.lock` clears Hermes-owned
tokens only, not Dashlane state. Disabling the source prevents future Hermes use
but does not lock dcli; the user must run `dcli lock` themselves.

## Verification and activation

`tests/agent/test_dashlane_search.py` exercises a synthetic executable through
real subprocess parsing, backend registry, `browser_vault_list` and existing
`browser_vault_fill` (browser transport mocked). It covers no-UUID discovery,
secret canaries in password/OTP/notes/custom fields/errors, malformed/duplicate/
oversize/timeouts, explicit host opt-in, vendor false matches, multiple choices,
profile/config/expiry binding, account/lock drift, changed selected records,
Cross-origin refusal before reading, and no automatic OTP. Existing adapter,
vault/browser, CLI and gateway tests remain relevant. No real vault or dcli is
read by development tests; fixtures certify implementation, not live acceptance.

After separately authorized deployment: preserve the running release and local
patches; install only reviewed changes; configure the active profile with the
reviewed account, trusted binary and `search_hosts: [service.test]`; restart the owning
process; call `browser_vault_list` and review its exact saved origin/identifier.
Select a candidate if ambiguous, then fill only on the bound origin through
`browser_vault_fill`. An Other-service cross-origin mismatch is a real blocker, not
permission to weaken origin checks. No production activation is done by these
tests or documentation.

---
sidebar_position: 19
title: "OpenAI Codex Device Login TLS Failure (SSL EOF / Handshake Timeout)"
description: "Troubleshooting Python/OpenSSL 3.5 SSL/TLS failures when completing Codex device-code login: EOF, handshake timeout, middlebox rejection of PQ groups"
---

# OpenAI Codex Device Login TLS Failure

## The symptom

`hermes model` cannot complete the Codex device login flow; it fails during the authentication POST or token-exchange POST with one of:

```text
[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1016)
```

or:

```text
_ssl.c:999: The handshake operation timed out
```

`curl` or your browser reach the same OpenAI endpoints successfully from the same machine — no network outage — yet Python consistently fails. `hermes setup` and your *existing* credentials (token refresh) may work fine; only the *device code* path hits the wall.

## Root cause

**OpenSSL 3.5** (shipped with the Python interpreter uv bundles for Hermes v0.21+, or your system Python 3.11.16+) advertises **post-quantum hybrid key-exchange groups** by default — for example, `X25519MLKEM768` — in the TLS 1.3 ClientHello `key_share` and `supported_groups` extensions.

A **TLS-intercepting middlebox** (corporate proxy, firewall, some CDNs, older cloud load balancers) that does not recognise the post-quantum group name can drop the connection, typically with:

1. A TCP RST → `[SSL: UNEXPECTED_EOF_WHILE_READING]`
2. A silent blackhole → `handshake operation timed out`

`curl` succeeds because curl's TLS stack (OpenSSL 1.1.x or LibreSSL on macOS) does not advertise PQ groups by default, so the same host+port passes the middlebox when requested by curl but fails for Python.

## Workarounds

### 1. Classic-groups OpenSSL configuration (recommended)

Create a file (e.g. `~/openssl-classic-groups.cnf`) with:

```ini
openssl_conf = openssl_init

[openssl_init]
ssl_conf = ssl_sect

[ssl_sect]
system_default = system_default_sect

[system_default_sect]
Groups = x25519:secp256r1:secp384r1:x448
```

This restricts the offered groups to classic elliptic curves, removing the PQ `X25519MLKEM768` extension that triggers the middlebox drop.

Run:

```bash
OPENSSL_CONF=~/openssl-classic-groups.cnf hermes model
```

The device-code handshake should complete.

### 2. Force TLS 1.2 (when #44392 lands)

TLS 1.2 does not carry the `key_share` / `supported_groups` extensions that advertise PQ groups. Capping at TLS 1.2:

```bash
HERMES_TLS_MAX_VERSION=1.2 hermes model
```

will sidestep the PQ-rejecting middlebox.

**Status**: Open PR [#44392](https://github.com/NousResearch/hermes-agent/pull/44392) introduces `HERMES_TLS_MAX_VERSION` for the *provider* (chat-completions) transport. That PR does not yet cover the device-login handshake path (`hermes_cli/auth_device_flow.py::_default_verify`), so this workaround will only reach the device-code flow once #44392 lands **and** a follow-up extends the cap to `_default_verify`. Track #44392 for progress.

### 3. Corporate network: request PQ-aware firewall rule

If you control the middlebox (enterprise firewall, load balancer), ask your IT team to allow the TLS 1.3 `X25519MLKEM768` group for OpenAI endpoints:

- `auth.openai.com`
- `api.openai.com`

This is the long-term fix — post-quantum cryptography is the future standard, and blocking it breaks modern clients.

## Why doesn't this break *all* Hermes OpenAI calls?

Device-code login (`hermes model` → Codex device flow) hits `auth.openai.com/api/accounts/deviceauth/*` with a **new** TLS connection. If your existing token refresh or provider calls work, they likely:

1. Use a cached TLS session or HTTP/2 connection from before OpenSSL 3.5.
2. Go to a different endpoint (`api.openai.com`) that your middlebox allows through (different routing rule).
3. Hit the device-code path less frequently (refresh tokens live for days; device login is rare).

The PQ-group handshake failure shows up most visibly on **fresh** device-code logins because they always open a new connection to `auth.openai.com` from scratch.

## Diagnostics

### Confirm you are on OpenSSL 3.5

```bash
python3 -c "import ssl; print(ssl.OPENSSL_VERSION)"
```

Expected output on a system exhibiting the failure:

```text
OpenSSL 3.5.7 9 Jun 2026
```

or later 3.5.x release.

### Test TLS 1.3 vs TLS 1.2 directly

```bash
# TLS 1.3 (will fail if your middlebox rejects PQ groups)
python3 -c "
import urllib.request, ssl
ctx = ssl.create_default_context()
ctx.maximum_version = ssl.TLSVersion.TLSv1_3
req = urllib.request.Request('https://auth.openai.com/codex/device', headers={'User-Agent': 'test'})
try:
    with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
        print('TLS 1.3: HTTP', r.status)
except Exception as e:
    print('TLS 1.3 failed:', e)
"

# TLS 1.2 (should succeed)
python3 -c "
import urllib.request, ssl
ctx = ssl.create_default_context()
ctx.maximum_version = ssl.TLSVersion.TLSv1_2
req = urllib.request.Request('https://auth.openai.com/codex/device', headers={'User-Agent': 'test'})
with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
    print('TLS 1.2: HTTP', r.status)
"
```

If **TLS 1.3 fails** with an SSL EOF and **TLS 1.2 succeeds**, you have a PQ-rejecting middlebox.

### Verify the classic-groups config

```bash
OPENSSL_CONF=~/openssl-classic-groups.cnf python3 -c "
import ssl
ctx = ssl.create_default_context()
# This doesn't directly print the groups, but a successful connection proves the config was applied.
import urllib.request
req = urllib.request.Request('https://auth.openai.com/codex/device', headers={'User-Agent': 'test'})
with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
    print('Classic groups config: HTTP', r.status)
"
```

## Related issues and PRs

- [#106384](https://github.com/NousResearch/hermes-agent/issues/106384) — Original bug report
- [#44392](https://github.com/NousResearch/hermes-agent/pull/44392) — `HERMES_TLS_MAX_VERSION` cap (provider path only, device-login follow-up planned)
- This PR (#NNNN) — Preserve SSL error + hint in device-code failures

## When will this be fixed upstream?

**The "fix" is configuration, not code**: OpenSSL 3.5 advertising PQ groups is *correct* behavior per RFC 8446 and the IETF's post-quantum migration path. The failure lies with the middlebox rejecting valid TLS 1.3 extensions.

Hermes will:

1. Document the workaround (this guide — ✅ done).
2. Preserve the SSL error + hint so the user sees "try `OPENSSL_CONF`" instead of a generic "login failed" (this PR — ✅ done).
3. Extend `HERMES_TLS_MAX_VERSION` to the device-login path once #44392 merges (follow-up PR).

If you control your network infrastructure, the long-term fix is to allow PQ groups at the firewall/proxy level.

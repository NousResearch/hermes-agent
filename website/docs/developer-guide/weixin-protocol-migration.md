---
title: "Weixin protocol compatibility"
description: "Tencent iLink behavior mapped onto Hermes gateway services"
---

# Weixin protocol compatibility

The reference is Tencent's `@tencent-weixin/openclaw-weixin-cli` 2.1.4 and
`@tencent-weixin/openclaw-weixin` 2.4.9. The installer delegates to OpenClaw's
plugin manager; Hermes implements the channel in Python and uses its own setup,
package manager, authorization, agent dispatch and profile services.

| Reference behavior | Hermes implementation |
| --- | --- |
| Install, enable, pair, persist and reconnect | `hermes gateway setup`, editable saved-account selection, profile-local YAML settings and secret token storage |
| QR metadata, recent local tokens, optional verification, redirect and already-bound results | `gateway/platforms/weixin.py`; invalid verification retries and expired QR refresh are bounded |
| Version headers and sanitized upstream application identity | iLink version 2.4.9 metadata for login, QR refresh and runtime requests; optional `extra.bot_agent`, ASCII product syntax and 256-byte cap |
| Optional backend routing tag | `extra.route_tag` becomes `SKRouteTag` on login, refresh, polling, send/config and lifecycle requests; profile-scoped one-shot sends also preserve the configured tag and bot identity |
| Rebinding replaces stale identities for the same Weixin user | After durable storage of confirmed credentials, removes old credentials, cursors, peer tokens and owned quote copies for that user in the same profile; preserves other users, other profiles and original attachments |
| Long polling, suggested timeout, sync cursor, failure retry and abort | Persistent profile-local cursor, 2s retry/30s backoff, server timeout hints and cancellation on disconnect; Hermes also recycles failed proxy sessions |
| Stale bot token guard | `-14` pauses all requests on that credential for one hour; stale peer context (`-2`, `prepare failed`/`unknown error`) keeps its separate tokenless-send recovery |
| Typing tickets and config-fetch backoff | Cached tickets, exponential failed-fetch backoff capped at one hour; Hermes keeps its existing 600s success TTL to refresh stale typing tickets |
| Encrypted CDN media | AES-128-ECB, PKCS#7, base64/hex keys, encrypted query/full URLs, native image/video/file/SILK references and preserved attachment filenames |
| CDN upload recovery | Reuses the same encrypted payload for up to three attempts on connection/server failures or missing download metadata; HTTP 4xx rejections return immediately |
| Voice text shortcut | Default `use_platform_transcription: true` uses `voice_item.text` and skips download, decoding and local STT |
| Audio-only voice | Existing profile-scoped SILK-to-WAV preparation and STT dispatch; optional decoder installation goes through Hermes PM. Explicit `use_platform_transcription: false` preserves multilingual re-transcription |
| ID-only and partial quoted replies | Lossless Python integer parsing and item-ID fallback, SQLite records scoped by profile/account/conversation, MD5-verified partial selection and owned attachment copies |
| Quote retention and graceful cache failure | Text: 30 days/10,000 records. Media: 7 days/256 MiB per account/25 MiB per file. Periodic and on-access collection; failures disable caching without dropping the message |
| Tool start/result messages and run correlation | Shared gateway lifecycle callbacks emit native iLink items 11/12 with real tool-call IDs and run IDs; operator display opt-out, per-account opt-out and text fallback on transport failure |
| Echo/debug diagnostics | `/echo` and `/toggle-debug` run after canonical gateway admission and slash-access checks, without an LLM call |
| Multiple accounts and session isolation | `weixin_group.py` serves multiple accounts in one profile; each child has its own token lock, policy, session namespace, context tokens and quote cache. Persisted `SessionSource.account_id` restores the exact reply account and cron origin |
| Configuration reload | Profile-bound transport watcher applies settled YAML/login changes automatically; affected accounts wait for active turns and sends to finish, and other accounts keep their connections. Invalid YAML retains the live configuration |
| Routing, pairing, message hooks, agent errors and reply dispatch | Existing Hermes gateway/profile services and adapter hooks replace OpenClaw SDK calls; no OpenClaw runtime is installed |
| Incremental Markdown filtering | `weixin_markdown.py` preserves fences, code spans, tables, rules and bold; strips CJK italic markers, H5/H6 markers and complete Markdown images, while retaining incomplete syntax across deltas |
| Block streaming | `weixin_streaming.py` coalesces at 200 characters or 3 seconds idle; carries final augmentations, hides reasoning/media directives and honors stop, approval boundaries and Hermes streaming opt-outs. Confirmed blocks are excluded from final-tail recovery |
| Text/media delivery | Existing Hermes chunking, media extraction, outbound validation, retry and error-delivery services; explicit `weixin:account-id/peer-id` targets resolve profile-local credentials |

Host-specific installer version matrices, npm dist-tag selection, pinned-plugin
update rules, and the two OpenClaw symlink workarounds do not apply to Hermes's
Python adapter. Hermes PM and its gateway setup/lifecycle services own those
responsibilities. This implementation does not install Node or OpenClaw.

The table records a behavior mapping, rather than a claim of identical source or
host behavior. Hermes uses its own configuration watcher instead of OpenClaw's
channel reload timestamp, and its native streaming switches remain authoritative.
Code spans are explicitly protected by the Markdown filter, including Markdown-like
content inside them. Typing-ticket success TTL remains 600 seconds.
Authorization remains profile-scoped, so rebind cleanup does not erase Hermes's
pairing grants or explicit access policy.

Quote persistence lives in `gateway/platforms/weixin_quotes.py`; native progress,
token guards and diagnostics in `weixin_experience.py`; metadata and network error
classification in `weixin_protocol.py`; stale credential cleanup in
`weixin_accounts.py`. Shared command dispatch lives in
`gateway/run_inbound_commands.py`, and native per-turn task state in
`gateway/run_progress_tasks.py`.

Contract tests cover platform-supplied voice text without STT, the real SILK/STT
handoff with a fake decoder/model, restart-safe quote recovery, account/peer/profile
isolation, media retention, server message IDs, native tool correlation, diagnostics
authorization and expired-token/backoff behavior. Live speech recognition depends
on the service-provided transcript or the selected STT backend; these tests do not
measure recognition accuracy.

Multi-account tests also cover live/restored reply selection, account-sensitive keys,
profile-local outbound credential selection, setup preservation, disabled accounts and
deferred transport reload. Markdown tests vary delta boundaries; block tests cover
final suffixes, approval flushes, media directives, silence and stale-run suppression.
Transport settings are reloaded without rebuilding cached agent prompts or tools.

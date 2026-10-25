---
title: Connected-source assistance
description: Automatic, evidence-backed next steps from supported accounts you already connected.
---

# Connected-source assistance

While the gateway runs, Hermes automatically checks **supported, already connected hosted app accounts** for actionable next steps. There is no second monitoring setup, source list, focus registration or schedule to create. Connecting an app remains the ordinary [connector workflow](./tool-gateway.md).

This is not a cron digest. A tool-less planner chooses bounded read-only queries; a second model judgment proposes only findings backed by observed records. Hermes looks for direct requests, approaching deadlines and useful connections between sources. It stays silent when evidence or relevance is insufficient. Suggested actions are **review only**: this feature never sends a message, edits an app record or executes the proposed action.

**Example (hypothetical):** a connected mail account contains “Please review Atlas today” and a connected project account contains the corresponding draft. Hermes can flag the review and cite both observed records. You review the evidence and decide what to do; connecting either account does not authorize sending a reply.

## Read your findings

```sh
hermes connectors findings
hermes connectors findings show <id>
hermes connectors findings dismiss <id>
hermes connectors findings resolve <id>
hermes connectors findings list --status all --limit 100
```

The classic terminal also supports `/findings` with the same arguments. These local commands require no live connector or model and keep working after sign-out. Dismissal and resolution are sticky across scans/restarts. Identical evidence of the same finding kind is suppressed even if the model changes its title or entity wording. New evidence may update an open finding; a quiet or failed source never resolves it for you.

Notifications are content-free pointers in an ownership-verified native local CLI/TUI/Desktop session. Source titles, excerpts and account details do not enter toast/replay payloads. No last-user, last-channel, home-channel or public fallback exists. If no safe native viewer is attached, the finding stays in the local inbox. Messaging `/findings`, including DMs whose account ownership cannot be proven, returns only a terminal pointer and cannot read or mutate findings.

## Pause or disable

```sh
hermes connectors findings pause
hermes connectors findings resume
hermes config set gateway.connected_assistance false
```

Pause applies to this profile's entire background assistance, not just one source. It prevents further source/model requests and publication; a request already on the wire may finish, but its late result is discarded. Resume does not create a job or bypass the cadence. Setting `gateway.connected_assistance` to `false` also disables background work. The default is `true`. Each served profile has its own setting, pause state, authorized accounts and inbox.

## Background use and costs

The gateway checks automatically about every **15 minutes per profile**, not on every message. A scan has a **60-second deadline**, at most **six read calls**, at most **two auxiliary model calls**, and a **64 KiB payload budget**. Normal connector/provider costs and rate limits apply. Account/tool metadata may be checked even without eligible work; with no eligible connected read tools there are **no source-content reads or model calls**. Supported read facets must be explicitly classified and their argument schemas supported. Disabled, unclassified, deprecated, disconnected or stale tools are not used.

The planner and judge use your existing auxiliary model credentials/routing. Their **two physical provider attempts** are shared across both stages, including retries and fallbacks; each attempt rechecks pause/retirement and the remaining deadline. Built-in synchronous auxiliary paths enforce this scope. A request already on the wire is retained until its SDK/network timeout or return, not forcibly killed. Providers that cannot honor the bounded request contract (currently Bedrock Converse) fail closed for this background feature; ordinary unscoped calls are unchanged. Custom/out-of-tree clients must not hide additional internal requests.

Data-use requirements follow source content into both model inputs and derived evidence. Source text is untrusted data, never an instruction to change configuration, execute arbitrary tools or publish to a model-chosen destination. Fresh permissions are checked before publication. This release supports the hosted connector capability contract, **not arbitrary MCP servers**.

Only one scan worker runs at a time. Restart cadence is persisted before requests begin. Shutdown and profile removal fence late results; a blocking network thread is retained until it settles, never abandoned while a replacement starts. If a provider ignores its timeout, profile ownership is retained rather than released to another writer.

## Local storage and delivery meaning

Findings, decisions, evidence and delivery receipts live under the active profile's `connected_assistance/state.sqlite3`. On POSIX the directory is mode `0700` and the database `0600`. Do not share this database: it contains private source excerpts.

The evidence cache retains at most **200 observed records** (at most 16 KiB each). Findings, deduplication fingerprints, user decisions and notification receipts have **no automatic expiry**; they remain until the profile or assistance state is removed. The whole database is **not globally size-bounded**. This preserves dismissal and duplicate suppression; deleting the state resets those decisions and scan cadence. Pause/disable before removing state.

A persisted finding is not proof of notification delivery. Notification receipts distinguish `unknown` (claimed before publication, no confirmed viewer acceptance) from `shown` (the local event stream accepted the frame for a viewer/observer). `shown` does not prove that a person read it. Unknown notifications are not blindly retried after a restart; the local inbox remains the source of truth.

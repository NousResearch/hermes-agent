---
title: Existing-session maintenance (gateway)
---

The gateway exposes an **owner-local** `session-maintenance` control-socket verb. It is not a chat command, automatic policy, remote API, or scheduled task. The socket/pipe is restricted to the local gateway owner; callers must already know the exact session key and ID. It does not create, switch, or deliver a message to a session.

Send a `query_gateway_control(home, "session-maintenance", params={...}, timeout=125)` request from a trusted local process. The `home` must identify the gateway's control socket, not an arbitrary profile directory. Parameters (all required):

```json
{"action":"inspect","profile":"default","session_key":"agent:main:telegram:dm:123","session_id":"existing-id","min_percent":85}
```

`action` is `inspect` or `compact`; `min_percent` is a caller-selected number strictly greater than 0 and at most 100. The strict eligibility rule is `last_prompt_tokens / context_window * 100 > min_percent`. This field is last **provider-reported prompt usage**, not the session's cumulative token count or an estimate of transcript size. An exact persisted route, a matching resident agent with a positive compressor context window, and an available model row are required; otherwise the status is `unknown_usage`. `inspect` returns `above_threshold` or `below_threshold` with `used_tokens`, `context_window`, and `percent` when those anchors are known.

`compact` additionally acquires the normal session turn lease, rejects active turns, rechecks the binding, and commits only in place. The regular agent compressor archives the old rows; no gateway transcript rewrite or routing rotation follows. Codex app-server sessions require the *cached live thread* and its native compact operation; the local transcript mirror cannot compact a server thread. On success, prompt usage is cleared with a binding-conditional metadata write **without** advancing the user-activity timestamp. A `pending` response means the owner-loop operation exceeded the socket's 120-second response wait and **may still finish**; never interpret it as cancellation or blindly retry. Query the session again after it settles.

Statuses are structured and do not contain prompts or provider exceptions: `invalid_request`, `profile_not_served`, `stale_binding`, `busy`, `unknown_usage`, `below_threshold`, `above_threshold`, `no_live_thread`, `not_enough_messages`, `history_unreadable`, `provider_unavailable`, `not_compacted`, `readback_failed`, `compacted`, `pending`, `error`. A caller must not treat anything except `compacted` as confirmed success. Compaction is intentionally not offered for a nonresident session without a trustworthy live context window.

The compressor checks the key→ID binding again at commit admission; a concurrent route switch after that admission can still cause an old-session archive to complete, but the conditional usage reset will not alter the new binding. This is not a multi-database atomic transaction. Avoid invoking `/new`, `/resume`, or `/model` concurrently with maintenance.

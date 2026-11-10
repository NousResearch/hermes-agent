---
title: Durable Telegram GitHub approval broker
---

# Durable Telegram GitHub approval broker

This candidate makes Telegram the single approval surface for GitHub work. A producer submits the canonical tuple:

```text
owner/repo#number + live 40-character head SHA + canonical 64-character contract digest + executor=hermes
```

The broker is `gateway.github_approval_broker.GitHubApprovalBroker`. Its process-global default store is:

```text
$HERMES_HOME/state/github-approval-broker.db
```

Do not put contract source, credentials, webhook payload bodies, or tokens in the tuple or question.

## Producer contract

Both webhook and foreground/local producers call the same broker before execution:

```python
from gateway.github_approval_broker import ApprovalProposal, default_broker

broker = default_broker(instance_id=gateway_instance_id)
outcome = await broker.request(
    ApprovalProposal(
        item_key="owner/repo#497",
        head_sha=live_head_sha,
        contract_digest_sha256=canonical_contract_digest,
        executor="hermes",
    ),
    adapter=telegram_adapter,
    chat_id=telegram_chat_id,
    thread_id=telegram_thread_id,
    question=bounded_approval_summary,
    session_key=origin_session_key,
    authorized_user_id=telegram_user_id,
)
```

Only `action == "answered"` with a matching live tuple and unchanged authority binding is reusable. `sent` and `reused` mean the decision is pending. `superseded` rejects replay of an older tuple; a rollback needs a fresh explicit reset rather than resurrection. `binding_conflict` rejects a changed question, authorized user, chat/topic, or Telegram bot for the same tuple. `pending_retry` is limited to an explicit provider rejection before delivery. `operator_reconciliation` means the provider outcome is ambiguous and must never be retried automatically.

Telegram startup calls `retry_pending(adapter=telegram_adapter)` for explicitly retryable rows. Continuation workers call `resume_approved(...)`: it re-reads the exact live head and digest, rejects stale tuples, and accepts only an executor implementing both durable `read_receipt(idempotency_key)` and `execute_idempotent(..., idempotency_key=...)`. A generic executor is marked blocked before any side effect. After a crash, receipt readback settles an already-completed effect without invoking it again.

## State guarantees

- An identical tuple reuses one card across webhook and local sessions.
- A changed head or contract digest supersedes the prior tuple and publishes a fresh card.
- A card becomes `active` only when Telegram returned a `message_id` and the delegated Telegram adapter installed the callback binding.
- Active rows bind the destination Telegram bot identity as well as user, chat, topic, and message; restart recovery loads only that bot's rows.
- Approval cards expose only the two closed choices. They never add the ordinary clarify prompt's free-text **Other** action.
- A timeout, exception, successful send without readback coordinates, or expired in-flight dispatch becomes `delivery_unknown`/`dispatching`; it is never retried automatically.
- Telegram callback decisions and continuation claims are SQLite transactions.
- Telegram startup rebuilds process-local callback routing from active durable rows.
- Callbacks must match the stored Telegram user, chat, topic, and message. Replays render the stored original decision, not the newly tapped button.
- Origin completion/status updates do not settle or remove the canonical Telegram card.

## Configuration and migration

No new credential is introduced. Existing Telegram and webhook credentials remain in the profile secret scope.

1. Confirm the intended profile has Telegram enabled and a home channel or explicit approval `chat_id`.
2. Configure an authenticated webhook route with `deliver: telegram` and `github_approval: {authorized_user_id: "<telegram-id>"}`. This lane accepts only GitHub `repository_dispatch` events with action `hermes_contract_approval` and an exact `client_payload` containing `pull_request_number`, `head_sha`, `contract_digest_sha256`, and `executor=hermes`. It creates the card and never admits an agent run.
3. Change every foreground/local producer to call the broker contract above. Do not also emit a local approval card.
4. Compose a resumption worker only with a GitHub reader and receipt-backed idempotent executor. If either contract is unavailable, leave the decision blocked; do not substitute a generic callable.
5. Existing JSONL operational ledgers are not imported automatically. Keep them read-only during evaluation; the new SQLite store is the candidate authority.

This candidate does not alter live configuration, execute a GitHub mutation, or compose an executor by itself.

## Rollback

1. Stop broker producers and the retry/resumption worker before reverting code.
2. Revert the candidate commit and restart only in an approved maintenance window.
3. Preserve `state/github-approval-broker.db` as read-only evidence. Do not delete it until every `active`, `pending_notification`, and unresumed decision is reconciled.
4. Retire any still-visible Telegram keyboards or post an explicit non-actionable correction.
5. Restore the prior approval path only after proving there is no simultaneously authoritative Telegram card.

Rollback never converts pending or failed notification state into authorization.

## Exact live acceptance test

Run only after deployment and restart are separately approved. Use a disposable GitHub item and harmless executor fixture; do not use merge/deploy authority.

1. Record the deployed commit, Gateway PID/start time, test `owner/repo#number`, live head, canonical contract digest, Telegram chat/topic, and baseline broker row count. Do not print secrets.
2. Send one signed eligible GitHub event through the public webhook edge. Require HTTP acceptance, one broker row for the exact tuple, `state=active`, non-empty `message_id`, `callback_bound=1`, and one Telegram card with native **Approve** / **Do not approve** buttons.
3. Replay the identical provider delivery and invoke the foreground/local producer with the same tuple. Require no additional Telegram message and the same `clarify_id`/`message_id`.
4. Edit the disposable contract so its canonical digest changes while the item remains pending. Send the eligible event. Require the old row/card to become superseded/non-actionable and one fresh active card. Repeat with a changed live head.
5. Block Telegram egress for the disposable profile before dispatching another changed tuple. Require `pending_notification`, null `message_id`, and no authorization. Restore egress and run the retry worker; require transition to `active` with message and callback binding.
6. Restart the Gateway in an approved window. Tap **Approve** on the active card. Require one durable decision with Telegram approver identity, one resumption claim, and exactly one harmless executor invocation. Replay the callback and restart again; require no second invocation.
7. While a card is active, settle the originating webhook/async session. Require the card and broker row to remain active until callback or supersession.
8. Verify logs and the broker contain no token, webhook secret, raw contract body, or credentials. Reconcile and remove the disposable route/item.

Acceptance fails on duplicate cards, plaintext-only approval, an active row without both bindings, silent pending work, stale callback acceptance, or more than one executor invocation.

# Immutable-prefix native streaming

Some external platform adapters use a persistent native message whose accepted text cannot be rewritten. These adapters can opt in with `NATIVE_STREAM_IMMUTABLE_PREFIX = True` alongside the existing native-stream capability. This is a transport contract, not a platform-name special case.

## Frame contract

For opted-in adapters, the consumer sends cumulative display-cleaned model text without a cursor, tool-progress overlay, or synthetic Markdown closing markers. Whitespace and real backticks remain intact. `SUPPORTS_NATIVE_TOOL_PROGRESS = False` also prevents the gateway from selecting the native progress lane. Adapters that do not opt in retain existing formatting and delivery behavior.

A successful frame means its text was accepted for delivery; an adapter must not acknowledge hidden/buffered frames as visible prefixes. If a wire chat kind does not support the native endpoint, reject the native seed so the consumer selects ordinary delivery before accumulating an acknowledged prefix.

## Failure and reconciliation

After a failed native update, the consumer attempts to seal the acknowledged prefix and buffers subsequent model output until the authoritative final is available. It does not send a temporary ordinary Markdown preview on the failing tick. If the final extends the confirmed prefix, it sends the exact missing tail without backing up to a word boundary or stripping boundary whitespace.

An optional synchronous `native_stream_acknowledged_prefix(*, turn_id, chat_id, reply_to)` adapter method returns the confirmed text after best-effort sealing. This lets the consumer reconcile a timed-out attempted frame that was subsequently acknowledged by a seal. Return only confirmed text, never merely attempted bytes. Network uncertainty is not an exactly-once guarantee; preserve ambiguous state rather than inventing an acknowledgement.

Ordinary tail sends carry local metadata `_native_fallback_prefix` identifying the prefix used to slice the tail. An adapter retaining its own prefix guard can validate that marker to avoid trimming the same prefix twice. For ordinary sends that split internally, advance the confirmed raw prefix after each successful chunk, even if a later chunk fails. `BasePlatformAdapter.truncate_message(..., chunk_end_offsets=offsets)` supplies consumed source offsets excluding the generated fence/counter decorations; the canonical implementation lives in `gateway/platforms/base_message_chunks.py`. Keep guards across definitive ordinary-send failures and conflicting full retries; do not claim delivery of rewritten authoritative content.

## Boundaries and abandonment

An opted-in `send_stream_frame` implementation may explicitly accept `seal_boundary`. When true, finalize closes an approval/clarify segment, not the whole response. Do not retain a turn-final correction guard that prevents the post-boundary answer from being delivered under the same inbound message ID.

Failed immutable boundary seals may use two optional adapter methods: async `recover_native_stream_boundary(*, turn_id, chat_id, reply_to)` preserves the confirmed prefix and retires definitively failed segment state, and synchronous `release_native_stream_boundary(...)` releases that prefix only after the preamble is delivered. Recovery must refuse unresolved ambiguous delivery. The consumer sends the cumulative preamble with inbound routing and `_interim_send` metadata, so the adapter sends only the unconfirmed tail. Prompt boundaries also recover a pending immutable segment after native streaming has already degraded. A failed recovery retains the segment and receipt for final reconciliation instead of opening a new segment.

Stale/cancel cleanup seals the last acknowledged immutable text without declaring the turn's authoritative final delivered. Adapter-local per-turn state must be cleaned up on this seal.

## Regression coverage

Core tests drive both frame primitives and `GatewayStreamConsumer.run()` with event-driven acknowledgements. They cover split Markdown, boundary whitespace, exact-tail fallback, default-adapter behavior, and stale cleanup. External adapters should additionally test their real receipt state against consumer failures, hidden wire chat kinds, approval/clarify reopen, rewritten final retries, and accepted/not-accepted timeout recovery using mocked transport acknowledgements.

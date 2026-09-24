# StateStore v10 model-usage transport evidence

## Bounded contract

v10 ports the SQLite `SessionUsageMixin.update_token_counts` persistence boundary to `PostgreSQLStateStore` without changing runtime backend selection. PostgreSQL owns one atomic callback per ordered delta and passes it to the already-extracted backend-neutral `TokenUsageTransport`; it does not introduce a second queue or expose a raw connection through the transport.

Migration v10 adds summary counters and billing metadata to `sessions`, plus `session_model_usage`. Its primary key is exactly `(session_id, model, billing_provider, billing_base_url, billing_mode, task)`, and catalog validation requires both attribution indexes and the session foreign key. Fresh and historical-ledger migration tests verify the linear `[1..10]` ledger.

## Executed evidence

Live PG18/SQLite differential coverage queues contiguous equal-route increments, an absolute barrier, and an auxiliary task row. It verifies coalesced main-loop attribution remains `5` while the absolute summary is `50`, with `api_call_count=3`, and the auxiliary route is isolated by `task='vision'`. The shared transport's legacy contract suite and async accounting suite also pass unchanged.

## Remaining scope

This does not cut over any runtime consumer, alter profile configuration, migrate a real `state.db`, or claim complete PostgreSQL state-store readiness. Remaining work includes complete SessionDB/gateway state surface, FTS/search, leases/recovery, SQLite import/reverse rollback, multi-process/multi-host behavior, and authorized runtime cutover.
# Frozen execution tuple

Kanban stores the frozen execution tuple on `task_runs`, not `tasks`. Each claim creates an attempt row, so crash retries, block/unblock cycles, and scope amendments retain independent provenance.

The tuple contains `exec_tuple_hash`, `base_sha`, `spec_rev`, `ceiling_rev`, `manifest_hash`, `toolchain_hash`, and `sandbox_policy_hash`. The hash is SHA-256 over the canonical six inputs in that order. Inputs are captured before worker spawn and are never updated on an existing run.

The additive migration runs automatically while opening a board. Existing rows remain valid with NULL tuple fields. For a controlled rollback, call `hermes_cli.kanban_db_connect.migrate_frozen_execution_tuple(conn, downgrade=True)` inside a backup/restore procedure; this drops only the tuple index and nullable columns and therefore discards tuple provenance. Re-running the normal migration restores the columns for future attempts.

A respawn must use a new `task_runs` row and a new worker session. No running session may receive a toolset or system-prompt mutation when tuple inputs change; session initialization consumes the tuple captured for its own row.

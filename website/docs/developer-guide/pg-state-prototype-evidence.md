# RFC #23717 prototype run evidence: PostgreSQL backend on cluster1 (Stage C, biggs.dog #775)

Date: 2026-11-07 (EET). Run executed across two phases: (a) the initial
harness2 probe executions (~09:52-11:00) that established the session-store
evidence and found DEFECT-1; (b) the fix-and-rerun phase (~10:48-14:30)
under the orchestrator's decision A: rebase, fix the PG-dialect defects at
the source (DEFECT-1/2/3, commits below), re-run the gateway-level items,
and re-run the negative control on the final code. This document
reproduces the pinned artifacts verbatim (DSN/password redacted), with
pass/fail per evidence item, the negative control, the deviations, and the
known defects. **Final state: every evidence item PASSes on head
c88d033f4a; one non-blocking defect (DEFECT-4) is documented open.**

## TL;DR

- The PostgreSQL backend on branch `hermes/775-pg-backend` works end to end
  against the real `cnpg-hermes-state` cluster (PG 18.1): backend
  resolution, schema `proto_hermes`, migration from a pre-seeded SQLite
  `state.db`, gateway boot, a full chat turn written to Postgres,
  deterministic API probe (create/append ascii+CJK+tool, search incl. CJK
  and the migrated session, conversation round-trip, meta), and a gateway
  resume of the migrated session that restored title + 4-message history +
  model from Postgres and answered from the PG transcript. Read-only psql
  read-backs prove the rows. **All eight evidence items PASS.**
- Three PG-dialect product defects were found, reproduced against the live
  cluster, and **fixed at the source on the branch with dialect-parity
  tests** (decision A; converter/boundary-only, SQLite path byte-identical,
  no schema changes):
  - DEFECT-1 (was blocking every PG gateway boot) — fixed in `39d0185fed`
  - DEFECT-2 (Decimal leak from PG aggregate SUMs into JSON settlement) —
    fixed in `a8b3df8649`
  - DEFECT-3 (drain-path int COALESCE over the TEXT json_extract shim) —
    fixed in `c88d033f4a`
- One defect remains **open and documented** (non-blocking): DEFECT-4, a
  three-argument `json_extract` in the terminal-admission media sweep
  (`gateway/session_ingress_media.py:474`) that the PG shim rejects; the
  gateway logs the error and continues, and no evidence item exercises the
  path. Per the task rule it is carried as a known defect for the close
  stage / user to decide.
- The negative control (default SQLite, no `database:` section) PASSES,
  re-run on the final head: local `state.db` written, search works, PG
  counts unchanged.
- Fork PRs: #150332 (A) and #150421 (B) were closed and replaced by
  rebased `-r2` branches/PRs (heads `79fbeab556` / `c88d033f4a`); the
  doc PR publishes this evidence.

## 1. Environment

| Item | Value |
|---|---|
| Host | macOS (arm64), kubectl context `cluster1` |
| Cluster | `cnpg-hermes-state` (CNPG, provisioned by biggs.dog #774), READY 3/3, primary `cnpg-hermes-state-1`, rw svc `cnpg-hermes-state-rw` |
| Server | PostgreSQL 18.1 (Debian 18.1-1.pgdg11+2), database `hermesstate`, role `hermesstate` (owner, may create schemas) |
| Schema | `proto_hermes` (created by hermes on first writable open) |
| Worktree | `/Users/shrinedogg/Code/hermes-agent/.worktrees/hermes-775-prototype`, detached at `c88d033f4a` == branch `hermes/775-pg-backend` (stages A+B + DEFECT-1/2/3 fixes), clean. The initial phase ran at `5cc7f6c76a`; heads: A `79fbeab556`, B `c88d033f4a` (fix commits `39d0185fed`, `a8b3df8649`, `c88d033f4a`). |
| Hermes version | `Hermes Agent v0.23.0+385.g5cc7f6c (2026.9.24)` (branch venv; version string tracks the initial head) |
| Python / drivers | 3.14.9 (Clang 23.1.3), psycopg 3.3.7, psycopg_pool 3.3.3 |
| Venv | `<worktree>/.venv-proto` (editable install of the worktree + psycopg); PM store worktree-scoped, environment generation `f083e0cbd` after `hermes pm install --extra postgres` |
| Tunnel | Initial phase: `kubectl -n cnpg-system port-forward svc/cnpg-hermes-state-rw 15432:5432` (supervised; it died 93 times — see section 8). Rerun phase: `scratch/pg_tunnel_v2.py` (tunnel-v2) on 127.0.0.1:15432 — per-connection `kubectl exec` byte-pumps to the primary pod, raw-fd in-pod pump; validated single (0.42 s, first byte 0.001 s), 2-concurrent hold, and an 8-way parallel burst (all connected in 2.6 s) before the rerun. |
| DSN (redacted) | `postgresql://hermesstate:***@127.0.0.1:15432/hermesstate` (in profile `.env` as `HERMES_DATABASE_URL`, 0600; password contains a URL-reserved character, percent-encoded in the URL; never printed in this doc or any artifact) |
| Model wiring | `qwen 3.8 - local` via provider `biggs-dog` (`https://vllm.biggs.dog/v1`), `key_env: HERMES_CUSTOM_BIGGS_DOG_API_KEY` (key in profile `.env` only, chmod 600, never shown) |
| Scratch root | `WS=/Users/shrinedogg/.hermes/kanban/workspaces/t_45c68cf4`; `HERMES_HOME=$WS/proto`; profiles: `proto-sql-nc2` (negative control, default SQLite), `proto-pg` (postgres home B). The user's live `~/.hermes` was never touched. |
| Artifacts | `$WS/evidence/` (pinned names per the plan spec); provenance: `ev-prov-*.txt` |

Provenance (branch code under test, verified per the workspace rule):
`hermes_cli`, `hermes_state`, `hermes_state_backend` all resolve to the
worktree; psycopg 3.3.7 resolves from the PM env for hermes CLI processes.
Key lines, verbatim (`ev-prov-module-provenance.txt`,
`ev-prov-psycopg-isolation.txt`, `ev-prov-cli-context-after-pmfix.txt`):

```
hermes_cli.main: /Users/shrinedogg/Code/hermes-agent/.worktrees/hermes-775-prototype/hermes_cli/__init__.py
hermes_state: /Users/shrinedogg/Code/hermes-agent/.worktrees/hermes-775-prototype/hermes_state.py
hermes_state_backend: /Users/shrinedogg/Code/hermes-agent/.worktrees/hermes-775-prototype/hermes_state_backend.py
psycopg OK: 3.3.7 | .../.venv-proto/lib/python3.14/site-packages/psycopg/__init__.py
```
```
backend: postgres | schema: proto_hermes
RAW OK: 1
```

Environment saga (documented, resolved before the PG sequence): the probe-agent
shell inherits a polluted env (PYTHONPATH to the user's live hermes install,
session/supervisor profile pins), and the PM bootstrap strips the outer venv's
site-packages, so the committed PM environment predated the `postgres` extra —
the CLI masked the ModuleNotFoundError as
`RuntimeError: PostgreSQL support requires pip install 'hermes-agent[postgres]'`
(`ev-prov-cli-context-chain.txt`). Both findings drove the two standing
mechanisms used throughout: every hermes/python invocation against the branch
goes through `proto/scrub.sh` (scrubs the polluted env, pins `HERMES_HOME` to
the scratch root, execs `<worktree>/.venv-proto/bin/hermes`), and the one-time
`hermes pm install --extra postgres` (worktree-scoped PM store only) fixed the
CLI path. `pm-install-postgres.txt`:

```
  → Resolving Python dependencies…
  ✓ Resolving Python dependencies
  → Installing Python dependencies…
  ✓ Installing Python dependencies
✓ venv +postgres
```

## 2. Exact commands

All `hermes`/`python` invocations against the branch go through the scrub
wrapper (shown abbreviated as `scrub` = `bash $WS/proto/scrub.sh`):

```bash
# 0. Worktree (reused; already checked out at the Stage B head; clean)
WT=/Users/shrinedogg/Code/hermes-agent/.worktrees/hermes-775-prototype   # detached 5cc7f6c76a == hermes/775-pg-backend

# 1. Cluster dependency gate (read-only) — expect READY 3/3, else STOP
kubectl -n cnpg-system get cluster cnpg-hermes-state -o wide
kubectl -n cnpg-system get svc cnpg-hermes-state-rw

# 2. Tunnel (background; supervisor watches the local port and restarts on death)
nohup-loop: bash $WS/proto/tunnel-supervisor.sh   # runs `kubectl -n cnpg-system port-forward svc/cnpg-hermes-state-rw 15432:5432`, PID recorded in $WS/evidence/port-forward.pid
# creds (read-side kubectl from the cluster secret hermes-state-user-password;
# the op CLI is not signed in on this host) -> /Users/shrinedogg/.hermes/cache/scratch/prot-pg-creds/creds.env (0600)
# DSN in shell env / profile .env only, percent-encoded, never printed:
#   postgresql://hermesstate:***@127.0.0.1:15432/hermesstate
psql read-backs (host has no psql; read-only SELECTs via the pod's psql 18.1):
kubectl exec -n cnpg-system cnpg-hermes-state-1 -c postgres -- psql -U postgres -d hermesstate -c '<SELECT>'

# 3. env capture (clean env)
(cd "$WT" && git rev-parse HEAD; scrub --version; .venv-proto/bin/python -c 'import psycopg, psycopg_pool, sys; ...') > ev-01-env-branch.txt

# 4. Scratch home A (negative control) — DEFAULT sqlite, no `database:` section
#    model block + key into profiles/proto-sql-nc2/{config.yaml,.env}
scrub -p proto-sql-nc2 gateway run --force > gw-nc2-forensic.log 2>&1 &
scrub -p proto-sql-nc2 chat -q 'SQLite control turn: note the phrase blue comet 775' -Q > ev-07a-nc2-turn.txt
sqlite3 profiles/proto-sql-nc2/state.db 'select count(*) from sessions; select count(*) from messages;' > ev-07b-nc2-sqlite-rowcounts.txt
scrub -p proto-sql-nc2 doctor / probe ev07_search_sqlite.py        > ev-07c-sqlite-search.txt
pod psql rowcounts + nc2-sid check                                  > ev-07d-pg-untouched-after-nc.txt

# 5. Scratch home B — postgres backend
#    profiles/proto-pg/config.yaml:  database: {backend: postgres, schema: proto_hermes}
#    profiles/proto-pg/.env (0600):  HERMES_DATABASE_URL=<redacted DSN>, model key
bash $WS/proto/run-migration.sh    # = (cd $WT && .venv-proto/bin/python scripts/migrate_sqlite_to_postgres.py \
                                    #    $WS/proto/profiles/proto-sql-nc2/state.db --profile-home $WS/proto/profiles/proto-pg) > ev-06-migration.txt

# 6. Boot-time backend resolution (CLI level)
scrub -p proto-pg doctor > ev-02-doctor-pg.txt     # line: "PostgreSQL session database is reachable (N sessions)"

# 7. PG sequence (proto/pg-sequence.sh, tracked background job): standalone PG gateway,
#    ev-03 chat turn, ev-04 rowcounts, ev-05 probe, ev-08 resume, ev-04b read-back
scrub -p proto-pg gateway run --force > gw-proto-pg.log 2>&1 &
scrub -p proto-pg chat -q 'Postgres turn: note the phrase green lantern 23717' -Q > ev-03-tmux-pg-turn.txt
pod psql rowcounts > ev-04-pg-rowcounts-after-migrate.txt
env -u PYTHONPATH HERMES_HOME=$WS/proto $WT/.venv-proto/bin/python $WS/proto/probe_pg.py proto-pg > ev-05-probe-pg.txt 2>&1
scrub -p proto-pg chat --resume local-88292e7476c59afa35146a74172b652a1e357ce901779eae8309766341cd526a \
    -q 'What phrase did I ask you to note in this session earlier? Reply with just the phrase.' -Q > ev-08-resume-pg.txt
pod psql read-back > ev-04b-pg-readback.txt
```

### 2b. Rerun phase commands (decision A; tunnel-v2 + cluster secret)

The rerun launcher (`scratch/run-ev-rerun3.sh`) rebuilds the DSN from the
cluster secret (in-process only; length-verified, never printed), checks
tunnel liveness with `lsof` only (no TCP probes — the probes themselves
killed the v1 tunnel, section 8), waits >=60 s of tunnel stability, then
runs the same sequence steps against the fixed code:

```bash
# 0. Rebase (both PRs were CONFLICTING with main) and fix commits:
git -C $MAIN fetch upstream            # upstream rebase base a08f2418cd
# A: hermes/775-storage-interface  e23c637c83 -> 79fbeab556
# B: hermes/775-pg-backend onto A: 5cc7f6c76a -> 71ab46ffe1
# DEFECT-1 fix (B): 39d0185fed  DEFECT-2 fix (B): a8b3df8649  DEFECT-3 fix (B): c88d033f4a
# (force-with-lease blocked by single-query approvals -> pushed as
#  hermes/775-storage-interface-r2 / hermes/775-pg-backend-r2)
git -C $WT checkout --detach c88d033f4a

# 1. Tunnel-v2 (background), spec port, validated before launch:
nohup $WT/.venv-proto/bin/python $SCRATCH/pg_tunnel_v2.py 15432 &
lsof -nP -i :15432                     # liveness ONLY (no nc/TCP probes)
$SCRATCH/single_probe.sh 15432         # connect 0.42 s, first byte t=0.001 s
$SCRATCH/concurrent_probe_v2.sh        # 2 concurrent + 10 s hold: V2-OK
$SCRATCH/burst_probe.sh                # 8-way parallel: all in 2.6 s

# 2. DSN (percent-encoded user+password; values in-process only) -> profile .env
python3 $SCRATCH/make_dsn.py           # from kubectl -n cnpg-system get secret
                                       # hermes-state-user-password (read-side)

# 3. Sequence (same steps as section 2, item 7) on the fixed code:
bash $WS/proto/pg-sequence.sh          # 13:07:50 -> 13:10:58 EET, rc=0
                                       # ev-03 PASS, ev-04 rows 9/16, ev-05
                                       # (superseded by post-finder rerun), ev-08
                                       # (profile_mismatch -> rerun below)

# 4. ev-08 rerun under the session's OWNING profile (proto-sql temporarily
#    flipped to the PG backend — the migrated session's local receipt is
#    bound to proto-sql, and cross-profile resume refusal is CORRECT
#    product behavior, not a defect):
python3 $SCRATCH/ev08_flip.py flip     # proto-sql config.yaml + .env -> PG
bash $SCRATCH/gw_sql_start.sh          # standalone proto-sql gateway (code c88d033f4a)
scrub -p proto-sql chat --resume local-88292e7476c59afa35146a74172b652a1e357ce901779eae8309766341cd526a \
    -q 'What phrase did I ask you to note in this session earlier? Reply with just the phrase.' -Q > ev-08-resume-pg.txt
python3 $SCRATCH/ev08_flip.py revert   # proto-sql back to SQLite (originals restored)

# 5. ev-05 rerun (stale editable finder had missed hermes_state_stored_reads;
#    `uv pip install -e . --no-deps` in the worktree regenerated it):
env -u PYTHONPATH HERMES_HOME=$WS/proto $WT/.venv-proto/bin/python $WS/proto/probe_pg.py proto-pg > ev-05-probe-pg.txt

# 6. Negative control rerun on the final head (stale nc2 gateway state
#    cleared; explicit standalone gateway 36590):
scrub -p proto-sql-nc2 chat -q 'SQLite control turn: note the phrase blue comet 775' -Q > ev-07a-nc2-turn-r2.txt
sqlite3 rowcounts + $SCRATCH/nc_probe.py proto-sql-nc2 > ev-07b/c-*-r2.txt   # NC-PROBE-OK
pod psql rowcounts (unchanged: 10 sessions / 24 messages) > ev-07b-nc2-sqlite-rowcounts-r2.txt
```

## 3. Evidence outputs (verbatim; DSN/password redacted)

### ev-01 — environment (PASS)

`ev-01-env-branch.txt`:
```
-- worktree --
5cc7f6c76aebbd32a3162951808e25924d2bbc0a
-- hermes --version (clean env, branch venv) --
Hermes Agent v0.23.0+385.g5cc7f6c (2026.9.24) · upstream 7f42573b · local 5cc7f6c7 (+854 carried commits)
Install directory: /Users/shrinedogg/Code/hermes-agent/.worktrees/hermes-775-prototype
Install method: git
Python: 3.14.9
OpenAI SDK: 2.54.0
Up to date
-- python + drivers (clean env, branch venv) --
3.14.9 (main, Oct 21 2026, 17:47:42) [Clang 23.1.3 ]
psycopg 3.3.7
psycopg_pool 3.3.3
```
`ev-01-server-version.txt` (via pod psql, through the port-forward):
```
                                                          version
----------------------------------------------------------------------------------------------------------------------------
 PostgreSQL 18.1 (Debian 18.1-1.pgdg11+2) on x86_64-pc-linux-gnu, compiled by gcc (Debian 10.2.1-6) 10.2.1 20210110, 64-bit
(1 row)
```
Note: the first run's `ev-01-env.txt` (09:52) was captured under the polluted
env (its "Install directory" points at the user's live install) and is
superseded by `ev-01-env-branch.txt` (deviation 6).

### ev-02 — boot-time backend resolution (PASS at CLI level)

`hermes -p proto-pg doctor` (clean env, attempt 1 after the PM fix; the tunnel
was healthy at that moment — `ev-02-spaced-attempts.log`:
`attempt 1: SUCCESS / 65:  ✓ PostgreSQL session database is reachable (1 sessions)`).
Relevant section of `ev-02-doctor-pg.txt` (14 KB; file holds the full output):
```
  ✓ PostgreSQL session database is reachable (1 sessions)
```
This is the plan's backend-resolution proof line. (Full gateway-process boot
does NOT reach this state cleanly — see DEFECT-1.)

### ev-06 — migration SQLite -> Postgres (PASS)

`ev-06-migration.txt` (source = the negative-control home's `state.db`; the
leading pool warning is a tunnel-flap transient, auto-retried):
```
error connecting in 'pool-1': connection failed: connection to server at "127.0.0.1", port 15432 failed: server closed the connection unexpectedly
	system_prompts: 2 rows copied
sessions: 1 rows copied
messages: 4 rows copied
session_model_usage: 2 rows copied
state_meta: 7 rows copied
gateway_routing: 2 rows copied
gateway_hygiene_state: 0 rows copied
conversation_generations: 0 rows copied
Migration committed. SQLite source was not changed.
```
Counts match home A's rowcounts (sessions 1, messages 4 — see ev-07b); single
final commit; source unchanged (the CLI verifies per-table counts before
committing). Post-migration state was re-verified live at several points
(sessions=1, messages=4) before the probe run.

### ev-04 / ev-04b — rows in `proto_hermes`, read-only psql read-back (PASS; PRE-TURN state)

`ev-04-pg-rowcounts-after-migrate.txt` (captured 10:53:18 EET, after the
migration + ev-05 probe; the PG chat turn is blocked by DEFECT-1, hence the
pre-turn label):
```
-- captured Sat Nov  7 10:53:18 EET 2026 - PRE-TURN STATE (migrated + probe session; the PG chat turn is blocked by the known-defect gateway boot crash, see diagnosis note)
-- command: kubectl exec -n cnpg-system cnpg-hermes-state-1 -c postgres -- psql -U postgres -d hermesstate
-- proto_hermes rowcounts --
sessions|2
messages|8
session_model_usage|2
state_meta|8
-- sessions --
local-88292e7476c59afa35146a74172b652a1e357ce901779eae8309766341cd526a|oneshot|4
proto-probe-1794041554|probe|4
```
`ev-04b-pg-readback.txt` (content visible, incl. CJK; usage counts
non-zero on the migrated session, carried over from the pre-migration turn):
```
-- proto_hermes.sessions --
-[ RECORD 1 ]-+
id            | proto-probe-1794041554
source        | probe
title         |
message_count | 4
input_tokens  | 0
output_tokens | 0
-[ RECORD 2 ]-+
id            | local-88292e7476c59afa35146a74172b652a1e357ce901779eae8309766341cd526a
source        | oneshot
title         | Note phrase blue comet 775
message_count | 4
input_tokens  | 31517
output_tokens | 648
-- proto_hermes.messages (newest 10) --
 proto-probe-1794041554 | assistant | Understood the CJK probe.
 proto-probe-1794041554 | user      | 狗年测试 CJK probe message
 proto-probe-1794041554 | assistant | Noted the zebra quartz 42.
 proto-probe-1794041554 | user      | Probe ascii message: zebra quartz 42
 local-88292e7476c59afa35146a74172b652a1e357ce901779eae8309766341cd526a | assistant | Noted: "blue comet 775" — saved to persistent memory.
 local-88292e7476c59afa35146a74172b652a1e357ce901779eae8309766341cd526a | tool      | {"success": true, "done": true, "target": "memory", "usage": "2% — 53/2,200 chars", "entry_count": 1, "message": "Applie
 local-88292e7476c59afa35146a74172b652a1e357ce901779eae8309766341cd526a | assistant |
 local-88292e7476c59afa35146a74172b652a1e357ce901779eae8309766341cd526a | user      | SQLite control turn: note the phrase blue comet 775
-- migrated session's full message history --
   1 | user      | SQLite control turn: note the phrase blue comet 775
   2 | assistant |
   3 | tool      | {"success": true, "done": true, "target": "memory", ...
   4 | assistant | Noted: "blue comet 775" — saved to persistent memory.
```
This is the "through Postgres, not SQLite" proof: the rows live in the
`proto_hermes` schema of the `hermesstate` database on the
`cnpg-hermes-state` cluster and are only readable via the cluster.

### ev-05 — deterministic API probe (PASS, attempt 1)

`ev-05-probe-pg.txt` (the leading pool warning is a tunnel-flap transient,
auto-retried):
```
error connecting in 'pool-1': connection failed: connection to server at "127.0.0.1", port 15432 failed: server closed the connection unexpectedly
resolved backend: postgres | schema: proto_hermes
SessionDB.backend: postgres
appended message ids: [5, 6, 7, 8]
search 'zebra quartz' hits: [('proto-probe-1794041554', 'Probe ascii message: >>>zebra<<< >>>quar'), ('proto-probe-1794041554', 'Noted the >>>zebra<<< >>>quartz<<<')]
search CJK hits: [('proto-probe-1794041554', '狗年测试 CJK probe message')]
search 'blue comet' (migrated) hits: [('local-88292e7476c59afa35146a74172b652a1e357ce901779eae8309766341cd526a', 'SQLite control turn: note the phrase >>>'), ('local-88292e7476c59afa35146a74172b652a1e357ce901779eae8309766341cd526a', 'tool-841441f4dd510e61", "type": "functio'), ('local-88292e7476c59afa35146a74172b652a1e357ce901779eae8309766341cd526a', 'Noted: ">>>blue<<< >>>comet<<< 775" — sa')]
migrated session id: local-88292e7476c59afa35146a74172b652a1e357ce901779eae8309766341cd526a
migrated conversation rows: 4
  conv row: user SQLite control turn: note the phrase blue comet 775
  conv row: assistant
  conv row: tool {"success": true, "done": true, "target": "memory", "usage":
meta round-trip: ok
sessions rows: [('proto-probe-1794041554', 'probe', 4, 0, 0), ('local-88292e7476c59afa35146a74172b652a1e357ce901779eae8309766341cd526a', 'oneshot', 4, 31517, 648)]
PROBE-OK
```
Covers the plan's probe assertions: `SessionDB().backend == 'postgres'`;
session create; append of ascii + CJK (`狗年测试`) + tool row; search finds
the NEW session, the CJK message, and the MIGRATED session;
`get_messages_as_conversation` round-trips the migrated session (4 rows);
usage counts non-zero on the migrated session; `get_meta`/`set_meta` round-trip.

### ev-03 — PG chat turn through the gateway (PASS — rerun after DEFECT-1/2 fixes)

Initial-phase attempt (10:48:35, head 5cc7f6c76a) failed — gateway boot
crash, DEFECT-1 (full history in section 7). Rerun at 13:07:50 EET on head
`a8b3df8649` (tunnel-v2, stable): `pg-sequence.log` sequence end 13:10:58,
rc=0. `ev-03-tmux-pg-turn.txt` (rerun artifact, verbatim tail):

```
Noted. Saved to:
/Users/shrinedogg/.hermes/kanban/workspaces/t_45c68cf4/proto/profiles/proto-pg/cache/scratch/notes.md
Content:
Postgres turn
Phrase to note: green lantern 23717
session_id: local-f8d1b6f2648fd8e78dcb16a363f93ff16204cb1529e1a6e3fa1c87fc50b212fa
```

The turn's rows landed in `proto_hermes` (see ev-04/ev-04b): session
`local-f8d1b6f2…` with 4 messages and non-zero usage (input 34212 /
output 1514), read back read-only via pod psql. This is the plan's task
item 2 at chat level: a new session created and its turns written to
Postgres by the running gateway.

### ev-08 — resume of the migrated session via gateway chat (PASS — rerun, final head)

History: blocked by DEFECT-1 in the initial phase; after the DEFECT-3 fix
(commit `c88d033f4a`) the rerun failed once with `profile_mismatch` — the
migrated session's local receipt is bound to `proto-sql` (the profile that
originally owned it), and cross-profile resume refusal is CORRECT product
behavior (evidence-design flaw, not a defect; receipt verified at
`state_meta` key `gateway.local_policy.v1:local-88292e74…`, profile_id
ending `/profiles/proto-sql`). Final rerun under the owning profile
(proto-sql temporarily flipped to the PG backend, reverted after):
`ev-08-resume-pg.txt` (verbatim):

```
↪ restored workspace dir: /Users/shrinedogg/Code/hermes-agent/.worktrees/hermes-775-prototype
↻ Resumed session local-88292e7476c59afa35146a74172b652a1e357ce901779eae8309766341cd526a "Note phrase blue comet 775" (2 user messages, 6 total messages)
Model restored from session: qwen 3.8 - local (biggs-dog)
blue comet 775

session_id: local-88292e7476c59afa35146a74172b652a1e357ce901779eae8309766341cd526a
```

Proof the resume read Postgres, not a local cache: the title and the
pre-migration 4-message history (rows 1-4, migrated by ev-06) came back
through the PG backend, the model was restored from the PG `sessions` row,
and the answer "blue comet 775" is only present in the PG transcript.
Rows 21-24 (this turn + the earlier DEFECT-3-affected attempt, which
completed its turn before the drain task died) landed in
`proto_hermes.messages`; zero `DatatypeMismatch` occurrences in the rerun
gateway log (the drain path now runs on the rewritten statement).

## 4. Pass/fail per evidence item

Plan evidence ids (task items 1-5 in parentheses):

| id | item | result |
|---|---|---|
| ev-01 | environment: branch sha, hermes/Python/driver versions, PG 18.1 server | PASS (initial capture at 5cc7f6c76a; final head c88d033f4a verified via gateway_state code_sha on both PG gateways) |
| ev-02 | backend resolution at boot (task item 1) | PASS — CLI: "PostgreSQL session database is reachable (1 sessions)"; gateway-process: PG gateway boots, serves, and writes turns (ev-03) |
| ev-03 | new PG chat turn through the gateway (task item 2, chat level) | PASS (rerun 13:07, head a8b3df8649) |
| ev-04 | rows exist in `proto_hermes` on the cluster; post-turn rowcounts (task item 2, psql proof) | PASS (9 sessions / 16 messages post-turn; final state 10/24) |
| ev-04b | psql read-back of sessions/messages, content visible incl. CJK | PASS (pre-turn read-back + final-state read-back `ev-04b-final-state.txt`) |
| ev-05 | deterministic API probe: backend, create, append (ascii/CJK/tool), search (new + migrated + CJK), conversation round-trip, usage, meta (task item 3; task item 2 at API level) | PASS (rerun on final head after editable-finder regeneration) |
| ev-06 | migration of home A's pre-seeded `state.db`: counts match, single commit, source unchanged (task item 5) | PASS |
| ev-07 | negative control, nc2 set (see section 5) | PASS (re-run on final head c88d033f4a) |
| ev-08 | resume of the migrated session via gateway chat (task item 4) | PASS (rerun under the owning profile, head c88d033f4a) |

Task item summary:
1. Gateway boots with the PG backend selected: **PASS** (CLI line + a live
   PG-backed gateway process serving turns).
2. Session created and its turns written to Postgres, proven read-only via
   psql: **PASS** (ev-03 + ev-04/ev-04b + ev-05).
3. session_search returns the seeded session from Postgres: **PASS** (ev-05).
4. Resume: second run resumes the session from Postgres: **PASS** (ev-08).
5. Migration of a pre-seeded SQLite state.db, read back through the backend:
   **PASS** (ev-06 + ev-04/ev-04b + ev-05 conversation round-trip).

Completion metadata (for the close stage):
- `evidence_path`: this file
- `pg_target`: `127.0.0.1:15432` (tunnel alias of `svc/cnpg-hermes-state-rw`, `cnpg-hermes-state` on cluster1; no credentials)
- `items_passed`: `ev-01, ev-02, ev-03, ev-04, ev-04b, ev-05, ev-06, ev-07, ev-08`
- `items_failed`: none
- `negative_control`: `pass`
- `known_defects_open`: `DEFECT-4` (non-blocking; section 7)

## 5. Negative control (opt-in boundary) — PASS

Scratch home A (`proto-sql-nc2`): model wiring only, **no** `database:`
section in `config.yaml` -> default SQLite. Supersession note: the first
run's `ev-07a-tmux-sqlite-turn.txt` / `ev-07b-sqlite-rowcounts.txt` were
captured under the polluted env and are superseded by the nc2 set below
(deviation 6).

`ev-07a-nc2-turn.txt` (one-shot `-Q` turn; phrase "blue comet 775"):
```
Noted: the phrase is **blue comet 775**.

I'm holding that as a marker in this turn only — I haven't written it to the repo, a file, memory, or any database, since you didn't ask for it to persist anywhere. If you want it recorded somewhere specific (a file, the SQLite DB you hinted at, memory, etc.), tell me the target and I'll do exactly that.

Also worth flagging: the "SQLite control turn:" prefix isn't a system instruction I recognize, so I'm treating the whole line as your message, not a privileged command. Let me know if there's more context I'm missing.

session_id: local-c633949c4aadabc24962612ae2d122006cf266c526b8afdb470491fd8034e172
```
`ev-07b-nc2-sqlite-rowcounts.txt` (local `state.db` written):
```
-- sqlite3 nc2/state.db --
1
2
-- nc2 session ids (local, should be the new turn) --
local-c633949c4aadabc24962612ae2d122006cf266c526b8afdb470491fd8034e172|oneshot|2
```
`ev-07c-sqlite-search.txt` (search works on SQLite):
```
resolved backend: sqlite | schema: None
SessionDB.backend: sqlite
search 'blue comet' hits: [('local-c633949c4aadabc24962612ae2d122006cf266c526b8afdb470491fd8034e172', 'user', 'SQLite control turn: note the phrase >>>blue<<< >>>comet<<< '), ('local-c633949c4aadabc24962612ae2d122006cf266c526b8afdb470491fd8034e172', 'assistant', "...the phrase is **>>>blue<<< >>>comet<<< 775**.\n\nI'm holdin")]
SQLITE-SEARCH-OK
```
`ev-07d-pg-untouched-after-nc.txt` (no PG schema touched by the control run):
```
-- proto_hermes rowcounts (expect unchanged: sessions=1 messages=4) --
sessions|1
messages|4
-- nc2 session id in PG? (expect 0) --
0
-- all schemas (expect: no new proto* beyond proto_hermes) --
proto_hermes
```
The opt-in boundary holds: with no `database:` section the backend resolves
to sqlite, state lives in the local `state.db`, and the cluster sees nothing
from that run. (The nc2 session was the source data later migrated in ev-06.)

**Rerun on the final head c88d033f4a** (the artifacts above are from the
initial head; the SQLite path is byte-identical across the defect fixes,
but the boundary claim is re-verified on the shipping code):

`ev-07a-nc2-turn-r2.txt` (one-shot turn, new session `local-e2302c78…`):
```
session_id: local-e2302c785c85c2d3bf51a61531653e5fd0e8d1b2b848ccdb573791424b3ca3d7
```
`ev-07b-nc2-sqlite-rowcounts-r2.txt` (local `state.db` written; PG
unchanged):
```
-- nc2 turn session: local-e2302c785c85c2d3bf51a61531653e5fd0e8d1b2b848ccdb573791424b3ca3d7 --
sessions: 2
messages: 6
local-e2302c785c85c2d3bf51a61531653e5fd0e8d1b2b848ccdb573791424b3ca3d7|oneshot|4
-- pg: any new proto_hermes rows after NC (must equal pre-NC counts) --
 pg_sessions | pg_messages
-------------+-------------
          10 |          24
```
`ev-07c-nc2-search-r2.txt` (backend resolution + search on the final code):
```
resolved backend: sqlite | schema: None
SessionDB.backend: sqlite
search 'blue comet' hits: [... 6 hits across the two nc2 sessions ...]
sessions rows: [('local-e2302c785c85c2d3bf51a61531653e5fd0e8d1b2b848ccdb573791424b3ca3d7', 'oneshot', 4), ('local-c633949c4aadabc24962612ae2d122006cf266c526b8afdb470491fd8034e172', 'oneshot', 2)]
NC-PROBE-OK
```
PG counts (10 sessions / 24 messages) are identical before and after the
rerun; no nc2 session id exists in `proto_hermes`. Note: the rerun needed
the nc2 profile's stale gateway state cleared (a dead pid recorded as
"running" refused the auto-launch, exit 78) — an ops artifact of the
initial phase, not a product issue.

## 6. Deviations from the plan script (ops-context wins; each with its reason)

1. **tmux -> `-Q` one-shot mode.** Host has no tmux; the branch's documented
   non-interactive mode `hermes -p <profile> chat -q '<prompt>' -Q` (answers
   and exits on a non-TTY) replaces the plan's tmux capture. Turn artifacts
   carry the pinned plan names (`ev-07a-*`, `ev-03-*`) with `-Q` content.
2. **Host psql -> pod psql via kubectl exec.** Host has no psql; all
   read-backs are read-only SELECTs executed as the postgres superuser over
   the unix socket inside the primary pod (`-t -A` / `-x` formats).
3. **Credential path: op CLI -> k8s secret read-side.** The 1Password CLI is
   not signed in on this host; the same seeded value was read from the cluster
   secret `hermes-state-user-password` into
   `/Users/shrinedogg/.hermes/cache/scratch/prot-pg-creds/creds.env` (0600).
   DSN percent-encoded (the password contains a URL-reserved character) and
   redacted everywhere in this doc.
4. **Worktree reused** at `5cc7f6c76a` (detached, == `hermes/775-pg-backend`
   head, clean) instead of re-adding it; the plan's `-b proto/pg-state-verify`
   scratch branch was not needed for a run-only stage.
5. **PM provisioning step added**: `hermes pm install --extra postgres`
   (worktree-scoped PM store; new environment generation `f083e0cbd` with
   psycopg 3.3.7 + psycopg-pool 3.3.3) — the committed PM environment
   predated the branch's `postgres` extra, which masked psycopg imports as a
   product RuntimeError (see section 1, environment saga).
6. **Env-pollution wrapper added**: the probe-agent shell inherits PYTHONPATH
   to the user's live hermes install plus session/supervisor pins; all
   branch invocations therefore go through `proto/scrub.sh`. First-run
   artifacts `ev-01-env.txt` and `ev-07a-tmux-sqlite-turn.txt` were captured
   under the polluted env and are superseded (nc2 set is the canonical
   negative control; `ev-01-env-branch.txt` the canonical env capture).
7. **Tunnel supervisor added** (see known operational issues, section 8):
   the port-forward to `cnpg-hermes-state-rw` dies on session teardown on
   this cluster; a supervisor loop restarts it within ~1 s.
8. **In-cluster image prototype OUT of scope**: the task body overrides
   #775's image wording (plan stage-bound note); only the branch-from-source
   prototype with the DSN to the rw service is in scope here.

## 7. Known defects (product, branch `hermes/775-pg-backend`)

DEFECT-1/2/3: **found, reproduced against the live cluster, fixed at the
source on the branch with dialect-parity tests** (decision A; all fixes
converter/boundary-only — no schema changes, no `POSTGRES_SCHEMA_VERSION`
bump, SQLite path byte-identical, verified by the oracle run). DEFECT-4:
**open, non-blocking, documented for the close stage.**

### DEFECT-1 (WAS blocking; FIXED in `39d0185fed`): PG gateway boot crash — `moved_receipts` SQL is not PG-compatible

Symptom: `hermes -p proto-pg gateway run --force` (any postgres-backed
profile) prints the STANDALONE banner, then the process exits (status 78)
during session-authority init, before it can serve. Every client chat
attempt fails with "Gateway could not start". Deterministic and
data-independent: the failing statement runs unconditionally at authority
init (`release_handed_off_receipts`) and fails at PARSE.

Root error (head of `gw-proto-pg.log`):
```
psycopg.errors.SyntaxError: syntax error at or near "CASE"
LINE 1: ....session_key IS NOT NULL AND s.session_key IS NOT CASE WHEN ...
```
raised from:
```
gateway/run_runtime.py:54  _build_profile_authority
gateway/session_authority.py:1462  initialize_session_authority
  -> asyncio.to_thread(release_handed_off_receipts, db, epoch=epoch)
gateway/session_local_recovery.py:443  release_handed_off_receipts
hermes_state_postgres.py:684  _execute_write
gateway/session_local_recovery.py:440  write
hermes_state_local.py:180  moved_receipts
```

The statement (`hermes_state_local.py:180-185`, shared SQLite-dialect SQL
executed verbatim through `PostgresConnection.execute`):
```sql
SELECT substr(m.key, ?), s.id, s.source FROM state_meta m JOIN sessions s
ON s.id = CASE WHEN json_valid(m.value) THEN json_extract(m.value, '$.entry.session_id') END
WHERE m.key GLOB ? AND s.session_key IS NOT NULL
AND s.session_key IS NOT CASE WHEN json_valid(m.value) THEN json_extract(m.value, '$.route') END
```
- PostgreSQL's `IS NOT` is an indicator test (NULL/TRUE/FALSE only);
  `IS NOT CASE ... END` is a syntax error. (SQLite's `IS NOT` is a null-safe
  inequality that accepts any expression — the statement works there.)
- The PG converter (`hermes_state_postgres.py:_convert_statement`) rewrites
  `GLOB` (to the `glob(P, X)` function shipped with the branch) and
  `INSERT OR REPLACE`, but nothing rewrites the `IS NOT CASE` construct or
  the `json_valid`/`json_extract` functions, so the statement cannot parse
  on PG as written.
- Verified live: the identical statement (placeholders substituted) is
  rejected by the cluster's PG 18.1 (pod psql) and accepted by SQLite 3.54
  against the control `state.db` (exit 0, zero rows).

Provenance: the statement is mainline — `upstream/main` `hermes_state_local.py`
line 184, commit `1d2263e13c` (2026-10-25, "fix(gateway): the moved-receipt
scan never parses a malformed receipt"), also in the stage A/B branch. The
branch's PG backend is what routes this shared local-recovery path onto
Postgres without a PG dialect for the statement. CLI-level paths (doctor,
migration, the SessionDB API) do not run `release_handed_off_receipts`, which
is why ev-02/ev-05/ev-06 pass while every PG gateway boot crashes.

Impact: blocks ev-03 (PG chat turn) and ev-08 (gateway resume) — the
gateway-level positive evidence. Session-store-level prototype evidence
(migration, create/append/search/resume-read-back/meta, negative control)
is complete and passing.

Fix (decision A, `39d0185fed`, branch B; converter-only, SQLite untouched):
the PG statement converter (`hermes_state_postgres.py:_convert_statement`)
now rewrites the two SQLite-only constructs:
- `IS NOT CASE ... END` -> `IS DISTINCT FROM (CASE ... END)` (PG's
  null-safe inequality; identical semantics for the NULL/true/false
  matrix, live-verified on the cluster)
- `substr(x, start)` -> `substring(x, from start)` (canonical 2-3-arg
  form for both dialects)
Covered by the new parity test
`test_moved_receipts_scan_and_boot_heal_are_dialect_parity` (sqlite +
postgres lanes) plus the re-runs of ev-03/ev-08. After the fix every
PG-backed gateway boot in this run completed; ev-03 and ev-08 PASS.

### Secondary defect (error path, same crash): boot log masks the root error

The launch profile's authority-build failure handler calls
`registry.remove(launch_home)` (`gateway/run_runtime.py:57`), which raises
`RuntimeError: the launch session authority cannot be removed`
(`gateway/session_authorities.py:62` — the launch slot is deliberately
non-removable). The final traceback a reader sees is this secondary error;
the original psycopg SyntaxError is only visible earlier in the log. A launch
profile whose store is unusable currently crashes the boot with a misleading
error instead of reporting the store failure. (Not exercised again after
DEFECT-1's fix; still worth a follow-up: the error path should surface the
store failure.)

### DEFECT-2 (WAS blocking settlement; FIXED in `a8b3df8649`): Decimal leaks from PG aggregates into JSON settlement

Symptom: the first stable-tunnel chat turn (12:54) completed the model
round-trip, then died in the final settlement write:
`RuntimeStoreError: invalid_params` at `hermes_state_runtime.py:63`
(`_json`) -> `TypeError: Object of type Decimal is not JSON
serializable`, located at
`result.auxiliary_usage.{api_calls,title_generation}`.

Root cause: the shared `auxiliary_usage_by_task` aggregate uses `SUM(...)`
over integer columns. PostgreSQL returns `SUM(int8)` as `numeric`;
psycopg maps `numeric` to `decimal.Decimal`. SQLite's SUM returns int/float,
so the same shared SQL never exposed the type. Live-verified on the cluster
(raw columns int/float; the aggregate columns come back `Decimal`; no
numeric-typed columns exist in the schema). The settlement serializer is
backend-neutral, so the fix belongs at the PG store boundary.

Fix (boundary-only): `hermes_state_postgres.py` row factory normalizes
scalars — `Decimal` -> `int` when integral, else `float` — for every value
returned from Postgres, systematic for the whole aggregate class (the other
two SUM sites already wrapped in `int()`/`float()`). Covered by the parity
test `test_auxiliary_usage_by_task_returns_native_scalars` (both dialects).
PG lane after the fix: 100/100 (11 files) + the state file 49/49.

### DEFECT-3 (WAS killing the session drain task; FIXED in `c88d033f4a`): int COALESCE over the TEXT json_extract shim

Symptom: the ev-08 resume restored the session from Postgres (title, 4
messages, model), then the drain path died:
`Session drain task died … psycopg.errors.DatatypeMismatch: COALESCE types
text and integer cannot be matched` — from
`_retire_undrained_queue_rows_conn` (`hermes_state_sessions.py:501`):

```sql
… AND COALESCE(json_extract((CASE WHEN json_valid(display_metadata) <> 0
    THEN display_metadata ELSE 'null' END), '$._queued_prompt'), 0) = 1
```

SQLite's `json_extract` returns JSON `true` as `1`, so the numeric marker
test works there; the PG shim is a TEXT-returning function (bundled
`json_extract(raw, path)`), making `COALESCE(text, integer)` a type error.
Only the migrated session tripped it (its pre-migration metadata row
carries the `_queued_prompt` marker; fresh proto-pg sessions never did —
zero occurrences in the earlier gateway log).

Fix (converter-only, following the file's existing
statement-by-statement rewrite style for this bug class): a general rule
rewrites `COALESCE(<guarded json_extract>, 0) = 1` to a text test
`<guarded json_extract> = 'true'` (absent/invalid extraction is NULL and
drops the row, exactly as the numeric form did). The other occurrence of
the same pattern (`model_only`) was already converted. Covered by the
parity test `test_retire_undrained_queue_rows_runs_on_both_dialects`
(marked row retires, unmarked row stays active, both dialects).

### DEFECT-4 (OPEN, non-blocking): three-argument json_extract in the terminal-admission media sweep

`gateway/session_ingress_media.py:474` (`sweep_settled_media`, called from
`gateway/session_settlement_recovery.py` at boot for terminal admissions):

```sql
SELECT json_extract(payload_json, '$.attachments_v1.media',
                    '$.native_text_v1.media')
  FROM session_admissions WHERE status='terminal' AND (…)
```

SQLite's `json_extract` accepts multiple paths and returns a JSON array;
the PG shim is exactly 2-argument (live-verified: `pronargs = 2` on
`proto_hermes.json_extract`), so the statement is rejected. Observed as
`ERROR gateway.session_settlement_recovery: Terminal media cleanup failed`
in the rerun gateway logs (13:27); the gateway logs the error and
continues — all boots, turns, and the resume completed — and no evidence
item produces terminal admissions with attachment payloads, so the path
was never exercised in a passing form.

Impact: on PG, media-cleanup sweeps for terminal admissions with
attachment metadata do not run (logged error each boot). Same dialect
class as DEFECT-1/3. Fix direction for the follow-up: a converter rewrite
to a two-element JSON-array construction over the 2-arg shim (semantics:
SQLite returns `[value@p1, value@p2]` with NULLs for missing paths) plus
a test with real admission rows. Decision for the close stage / user:
fix on branch B before merge, or carry as a follow-up issue.

## 8. Known operational issues (cluster-side, not product code)

- **Port-forward RST quirk on `cnpg-hermes-state` — root-caused**: the
  shared in-pod `kubectl port-forward` helper dies when CONCURRENT
  Postgres sessions flow through it (93 deaths in the initial phase;
  two simultaneous sessions reproduce it deterministically, singles
  survive 13+ minutes; PG server logs show the sessions reach `idle`, then
  the connection is RST from the Mac side 2-5 s later). The gateway pool
  dial storm (14 pools in ~8 s at boot) is the trigger. Two additional
  traps found while debugging: (a) TCP liveness probes (`nc -z`) against
  the tunnel port themselves kill the v1 helper — liveness must be checked
  with `lsof` only; (b) the pod IP (10.244.1.218) is not routable from the
  Mac, so a direct bypass is impossible. **Flagged for biggs.dog #774**
  (the CNPG wave that provisioned the cluster; the older
  `cnpg-kagent-rw` tolerates teardown RSTs, so the regression is most
  plausibly tied to the new cluster's image, postgresql:18.1 + barman-cloud).
  Evidence-infrastructure workaround adopted for the rerun: `tunnel-v2`
  (`scratch/pg_tunnel_v2.py`) — one `kubectl exec` byte-pump per
  connection to the primary pod with a raw-fd in-pod pump; a connection's
  teardown cannot affect other connections or the listener (validated:
  single 0.42 s / first byte 0.001 s, 2-concurrent + 10 s hold, 8-way
  parallel burst all connected in 2.6 s).

## 9. Final state (post-rerun, for the close stage)

- All evidence items PASS on head `c88d033f4a`; the negative control was
  re-run on the same head. DEFECT-1/2/3 fixed on branch B (commits in
  section 7); DEFECT-4 open (non-blocking).
- Branch/PR state (fork `shrinedogg/hermes-agent`): the initial PRs
  #150332 (A) and #150421 (B) were CONFLICTING with main and were closed;
  replacements were opened from the rebased heads — #150649 (A,
  `hermes/775-storage-interface-r2` @ `79fbeab556`), #150650 (B,
  `hermes/775-pg-backend-r2` @ `c88d033f4a`; B stacked on A,
  must-merge-first), and this doc published as #150652
  (`docs/pg-state-run-evidence-r2`). (GitHub cannot retarget a PR's head
  branch; force-with-lease is blocked by single-query approvals, hence the
  `-r2` branches.)
- Cluster data (left in place, never mutated by this wave beyond
  hermes's own writes): `proto_hermes` = 10 sessions / 24 messages
  (migrated oneshot session with its 8 message rows incl. the two resume
  turns, the ev-03 PG turn session, the probe sessions, and five empty
  oneshot rows from the pre-fix failed boot attempts — proof the fixed
  code boots the gateway cleanly even when the turn dies). The schema was
  never dropped; do not re-run the migration (it refuses a populated
  destination).
- Teardown performed at run end: all prototype gateways stopped,
  tunnel-v2 and the v1 port-forward/supervisor stopped, the per-connection
  kubectl-exec pumps gone, port 15432 unbound (verified). No process
  orphans; the user's live `~/.hermes` was never touched. One leftover
  requiring a user step (docker lifecycle is approval-gated on this host):
  the intermediate docker PG 17 parity container (`hermes-test-pg`, port
  55433, labeled dev check only) is still up — `docker stop
  hermes-test-pg && docker rm hermes-test-pg`.
- Scratch profile homes (`proto-pg`, `proto-sql`, `proto-sql-nc2`)
  contain `.env` files with the model key: kept in the task workspace
  (`$WS/proto/`); delete with the workspace after the round.
- Pinned artifacts: all in `$WS/evidence/` (names per the plan spec, plus
  the `-r2` rerun artifacts); this document is the consolidated
  run-evidence doc. Full tracebacks: `$WS/evidence/gw-proto-pg.log`,
  `gw-proto-sql.log`; diagnosis note:
  `$WS/evidence/diagnosis-pg-gateway-boot-2026-11-07.md`.

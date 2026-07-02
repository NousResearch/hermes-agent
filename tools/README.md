# tools/

Thin wrappers for capabilities that Hermes skills call.

Initial rule: keep wrappers boring and auditable. Each wrapper should document whether it is read-only, gated write, or forbidden in the pilot.

Planned wrappers:

- `snowflake_query.py` - read-only warehouse queries.
- `analytics_bias_check.py` - prints the required analytics bias checklist.
- `chart.py` - generate analyst-ready PNG charts from query CSVs.
- `fetch_schemas.py` - refresh read-only Snowflake DDL snapshots into `yallaplay-wiki/reference/warehouse/ddl/`.
- `compile_wiki_skills.py` - compile wiki `## Agent quick context` sections into git-tracked Hermes skills and optional profile symlinks.
- `appinsights_query.py` - read-only Azure Application Insights / Log Analytics KQL queries.
- `wiki_search.py` - search `yallaplay-wiki/`.
- `embrace_metrics.py` - read-only Embrace Metrics API PromQL queries.
- `embrace_sessions.py` - read-only Embrace dashboard session lookup for per-user support/debugging.
- `backend_config_snapshot.py` - read-only backend config snapshot/history refresh into `yallaplay-wiki/operations/backend-config/`.
- `bedrock_models.py` - list available Bedrock model IDs by provider/region.
- `repo_search.py` - repo and sibling source search helpers.
- `slack_lookup.py` - read-only Slack lookup helpers.

## Analytics

`snowflake_query.py` is read-only by construction. It allows one `SELECT`, `WITH`, `SHOW`, `DESCRIBE`, or `EXPLAIN` statement and blocks DDL/DML keywords before connecting.

Examples:

```bash
python3 tools/snowflake_query.py --dry-run "SELECT CURRENT_DATE() AS today"
python3 tools/snowflake_query.py "SELECT CURRENT_DATE() AS today"
python3 tools/snowflake_query.py -f queries/example.sql -o outputs/example.csv
python3 tools/analytics_bias_check.py
python3 tools/chart.py outputs/example.csv -t "Example" -o outputs/example.png
python3 tools/fetch_schemas.py --dry-run
python3 tools/compile_wiki_skills.py --list
python3 tools/compile_wiki_skills.py --name season-pass-analytics \
  --profile-symlink-root ~/.hermes/profiles/claudio-lab/skills/claudio-authored
```

Credential resolution stays outside git:

1. Snowflake environment variables.
2. `--vars`, `HERMES_SNOWFLAKE_VARS`, or `SNOWFLAKE_VARS_TOML` pointing to a private TOML file.
3. Local untracked `vars.toml` in this repo.
4. Sibling migration lab file `../yallaplay-analytics-agent-gpt/vars.toml` when present.

Required TOML keys are compatible with the old Claudio analytics repo: `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_HOST`, `SNOWFLAKE_USER`, `SNOWFLAKE_PRIVATE_KEY`, `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_DATABASE_PROD`, and `SNOWFLAKE_SCHEMA`.

## App Insights

`appinsights_query.py` is read-only by construction: it calls Azure Monitor's Log Analytics query API and blocks Kusto management commands (`.` commands). It defaults to `--env prod` because dev and prod write into the same workspace. Add `--real-players` (alias: `--exclude-pc`) for real-player analysis to drop internal/QA desktop clients.

Examples:

```bash
python3 tools/appinsights_query.py --check-credentials
python3 tools/appinsights_query.py --dry-run -q "AppRequests | summarize n=count()"
python3 tools/appinsights_query.py --list-services
python3 tools/appinsights_query.py --real-players --days 1 \
  -q "AppRequests | summarize n=count(), errs=countif(Success == false) by OperationName"
python3 tools/appinsights_query.py --service yallaplay-client-twin --days 7 \
  -q "AppExceptions | summarize n=count() by ExceptionType | order by n desc" \
  -o outputs/appinsights_twin_exceptions.csv
```

Credential resolution stays outside git:

1. `AZURE_APPINSIGHTS_TENANT_ID`, `AZURE_APPINSIGHTS_CLIENT_ID`, `AZURE_APPINSIGHTS_CLIENT_SECRET`, and `AZURE_APPINSIGHTS_WORKSPACE_ID` environment variables.
2. Private TOML via `--vars`, `HERMES_YALLAPLAY_VARS`, or `YALLAPLAY_VARS_TOML`.
3. Local untracked `vars.toml` in this repo.
4. Sibling migration lab file `../yallaplay-analytics-agent-gpt/vars.toml` when present.

The workspace uses workspace-based tables such as `AppRequests`, `AppExceptions`, `AppTraces`, `AppDependencies`, and `Usage`; do not use classic component table names like `requests` or `exceptions`. `--dry-run` prints the final injected KQL without loading credentials or calling Azure.

## Backend config

`backend_config_snapshot.py` is read-only by construction: it only calls config-service GET endpoints and writes snapshots/history into the wiki. It stores the unwrapped config `response` as full-fidelity JSON and copies raw per-section history under `history/`.

Examples:

```bash
python3 tools/backend_config_snapshot.py --app Spades --snapshot --timestamped-snapshot --dry-run
python3 tools/backend_config_snapshot.py --app Spades --snapshot --history --sections flags,overrides,twinscripts
python3 tools/backend_config_snapshot.py --app Spades --history --sections seasonpass --wiki-app spades
python3 tools/backend_config_snapshot.py --app Spades --check-credentials
```

Credential resolution stays outside git:

1. `BACKEND_CONFIG_BASE_URL` + `BACKEND_CONFIG_SERVER_TOKEN` environment variables.
2. `YALLAPLAY_CONFIG_BASE_URL` + `YALLAPLAY_CONFIG_SERVER_TOKEN` environment variables.
3. Private TOML via `--vars`, `HERMES_YALLAPLAY_VARS`, or `YALLAPLAY_VARS_TOML`.
4. Local untracked `vars.toml` in this repo.
5. Sibling migration lab file `../yallaplay-analytics-agent-gpt/vars.toml` when present.

## Embrace

Embrace has three access paths:

- Native Hermes MCP server for aggregate crash/exception/network/span/log drilldowns: configure `https://mcp.embrace.io/mcp` in the active Hermes profile with the Embrace service-account `emb_sa_*` bearer token.
- `embrace_metrics.py` for PromQL time-series from the Metrics API.
- `embrace_sessions.py` for per-user dashboard session records; this is the scraper/API path MCP does not expose.

Examples:

```bash
# Native MCP, once the service-account token is available in the profile env.
# Hermes stores the header as Authorization: Bearer ${MCP_EMBRACE_API_KEY}.
hermes mcp add embrace --url https://mcp.embrace.io/mcp --auth header

python3 tools/embrace_metrics.py --list-apps
python3 tools/embrace_metrics.py -q 'sum(daily_sessions_total{app_id="r5GWq"})'
python3 tools/embrace_metrics.py --range --days 14 --step 1d \
  -q 'sum(daily_sessions_total{app_id="r5GWq"})' -o outputs/embrace_spades_sessions.csv

python3 tools/embrace_sessions.py --list-apps
python3 tools/embrace_sessions.py --app spades_android --user 9822255144578 \
  --around 2026-06-04T22:40:00Z --window 30
python3 tools/embrace_sessions.py --app spades_android --resolution day --user 9822255144578
```

Credential resolution stays outside git:

- Metrics API: `EMBRACE_METRICS_API_TOKEN` env var, or private TOML via `--vars`, `HERMES_YALLAPLAY_VARS`, `YALLAPLAY_VARS_TOML`, local untracked `vars.toml`, or sibling migration lab `../yallaplay-analytics-agent-gpt/vars.toml`.
- Session scraper/API: `EMBRACE_DASH_EMAIL` and `EMBRACE_DASH_PASSWORD` via the same env/TOML resolution.
- Session JWT cache: `.local/cache/.embrace_auth.json` (gitignored, chmod 600).

Production app guard: both scripts are scoped to `spades_android`, `spades_ios`, `rummy_android`, and `rummy_ios` (`r5GWq`, `QkTz6`, `dmma2`, `s8kti`).

`chart.py` expects CSVs in the same shape as the legacy analytics-agent tool:

- 2 columns: bar chart (`label, value`).
- 3 columns: grouped bar or line (`x, series, value`).
- 4 columns: facet chart (`facet, x, series, value`).

Multi-series charts should be long format, not wide one-column-per-series CSVs.

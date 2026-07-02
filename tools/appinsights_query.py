#!/usr/bin/env python3
"""Query Azure Application Insights / Log Analytics KQL for backend telemetry.

Read-only. Credentials stay outside git.

Credential resolution:
1. AZURE_APPINSIGHTS_* environment variables.
2. --vars / HERMES_YALLAPLAY_VARS / YALLAPLAY_VARS_TOML private TOML file.
3. ./vars.toml if present locally and untracked.
4. ../yallaplay-analytics-agent-gpt/vars.toml for migration labs.

The shared workspace contains dev and prod telemetry. The wrapper defaults to
--env prod and can inject a --real-players ClientType != 'PC' guard for mobile
player-only analyses.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from secret_config import pick_mapping_source

PROJECT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_DIR / "logs"
# Verified in the legacy repo: the api.loganalytics.azure.com resource principal
# is not registered in this tenant; keep scope and host paired.
TOKEN_SCOPE = "https://api.loganalytics.io/.default"
QUERY_HOST = "https://api.loganalytics.io"

REQUIRED_KEYS = (
    "AZURE_APPINSIGHTS_TENANT_ID",
    "AZURE_APPINSIGHTS_CLIENT_ID",
    "AZURE_APPINSIGHTS_CLIENT_SECRET",
    "AZURE_APPINSIGHTS_WORKSPACE_ID",
)

LIST_SERVICES_KQL = (
    "union AppRequests, AppDependencies, AppExceptions, AppTraces, AppMetrics{flt} "
    "| summarize events=count() by _ResourceId "
    "| order by events desc"
)


@dataclass(frozen=True)
class AppInsightsConfig:
    tenant_id: str
    client_id: str
    client_secret: str
    workspace_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-q", "--query", help="KQL query string")
    parser.add_argument("--vars", type=Path, help="Private TOML file containing AZURE_APPINSIGHTS_* keys")
    parser.add_argument(
        "--env",
        choices=["prod", "dev", "all"],
        default="prod",
        help="Inject Properties.Environment filter after the first table/union list (default: prod)",
    )
    parser.add_argument(
        "--real-players",
        "--exclude-pc",
        dest="real_players",
        action="store_true",
        help="Inject ClientType != 'PC' to drop internal/QA desktop clients for player analyses",
    )
    parser.add_argument(
        "--service",
        help="Inject _ResourceId endswith '/<service>' filter, e.g. yallaplay-client-twin",
    )
    parser.add_argument("--list-services", action="store_true", help="List backend App Insights components seen in the window")
    parser.add_argument("--days", type=float, default=1, help="Look back N days if --start is omitted (default 1)")
    parser.add_argument("--start", help="Explicit window start, ISO-8601 or unix seconds")
    parser.add_argument("--end", help="Explicit window end, ISO-8601 or unix seconds; default now")
    parser.add_argument("-o", "--output", type=Path, help="Write CSV/raw output to this path")
    parser.add_argument("--raw", action="store_true", help="Emit raw Log Analytics JSON instead of CSV")
    parser.add_argument("--print-query", action="store_true", help="Print final KQL after guard injection before execution")
    parser.add_argument("--dry-run", action="store_true", help="Build and log KQL without loading credentials or querying Azure")
    parser.add_argument(
        "--check-credentials",
        action="store_true",
        help="Report credential source and missing key names only; do not request a token or run KQL",
    )
    return parser.parse_args()


def pick_config_source(vars_path: Path | None) -> tuple[str, dict[str, Any]]:
    return pick_mapping_source(
        vars_path,
        env_required=REQUIRED_KEYS,
        missing_message=(
            "No App Insights credential source found. Set AZURE_APPINSIGHTS_* env vars or pass --vars / "
            "HERMES_YALLAPLAY_VARS pointing to a private TOML file."
        ),
    )


def missing_keys(mapping: dict[str, Any]) -> list[str]:
    return [key for key in REQUIRED_KEYS if not mapping.get(key)]


def config_from_mapping(mapping: dict[str, Any]) -> AppInsightsConfig:
    missing = missing_keys(mapping)
    if missing:
        raise SystemExit(f"App Insights config missing: {', '.join(missing)}")
    return AppInsightsConfig(
        tenant_id=str(mapping["AZURE_APPINSIGHTS_TENANT_ID"]),
        client_id=str(mapping["AZURE_APPINSIGHTS_CLIENT_ID"]),
        client_secret=str(mapping["AZURE_APPINSIGHTS_CLIENT_SECRET"]),
        workspace_id=str(mapping["AZURE_APPINSIGHTS_WORKSPACE_ID"]),
    )


def parse_time(value: str) -> str:
    text = str(value).strip()
    if text.replace(".", "", 1).isdigit():
        return datetime.fromtimestamp(float(text), timezone.utc).isoformat()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit(f"Bad time {value!r}; use ISO-8601 or unix seconds") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def query_timespan(args: argparse.Namespace) -> str:
    if args.start:
        end = parse_time(args.end) if args.end else datetime.now(timezone.utc).isoformat()
        return f"{parse_time(args.start)}/{end}"
    if args.days <= 0:
        raise SystemExit("--days must be positive")
    if args.days < 1:
        hours = max(1, int(round(args.days * 24)))
        return f"PT{hours}H"
    return f"P{int(args.days)}D" if float(args.days).is_integer() else f"PT{args.days * 24:g}H"


def conditions_clause(conditions: list[str]) -> str:
    return "".join(f" | where {condition}" for condition in conditions)


def inject_filters(kql: str, conditions: list[str]) -> str:
    if not conditions:
        return kql
    stripped = kql.lstrip()
    index = 0
    while index < len(stripped) and (stripped[index].isalnum() or stripped[index] == "_"):
        index += 1
    table = stripped[:index]
    rest = stripped[index:]
    if not table:
        raise SystemExit("KQL must start with a table name when wrapper filters are injected")
    return f"{table}{conditions_clause(conditions)}{rest}"


def validate_kql(kql: str) -> str:
    query = kql.strip()
    if not query:
        raise SystemExit("KQL query is empty")
    if query.startswith("."):
        raise SystemExit("KQL control commands are not allowed")
    if re.search(r"\b(ingest|set-or-append|set-or-replace|drop|delete)\b", query, flags=re.IGNORECASE):
        raise SystemExit("Potentially mutating KQL/control command text is blocked")
    return query


def build_conditions(args: argparse.Namespace) -> list[str]:
    conditions: list[str] = []
    if args.env != "all":
        conditions.append(f"parse_json(Properties).Environment == '{args.env}'")
    if args.real_players:
        conditions.append("ClientType != 'PC'")
    if args.service:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.service):
            raise SystemExit("--service must be an App Insights component name, e.g. yallaplay-client-twin")
        conditions.append(f'_ResourceId endswith "/{args.service}"')
    return conditions


def build_kql(args: argparse.Namespace) -> str:
    conditions = build_conditions(args)
    if args.list_services:
        return LIST_SERVICES_KQL.format(flt=conditions_clause(conditions))
    if not args.query:
        raise SystemExit("one of -q/--query or --list-services is required")
    return inject_filters(validate_kql(args.query), conditions)


def get_token(config: AppInsightsConfig) -> str:
    url = f"https://login.microsoftonline.com/{config.tenant_id}/oauth2/v2.0/token"
    data = urllib.parse.urlencode(
        {
            "grant_type": "client_credentials",
            "client_id": config.client_id,
            "client_secret": config.client_secret,
            "scope": TOKEN_SCOPE,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)["access_token"]
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Token request failed: HTTP {exc.code}\n{body}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Token request failed: {exc}") from exc


def run_query(config: AppInsightsConfig, token: str, kql: str, timespan: str) -> tuple[str, str]:
    url = f"{QUERY_HOST}/v1/workspaces/{config.workspace_id}/query"
    data = json.dumps({"query": kql, "timespan": timespan}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "yallaplay-hermes-agent/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return url, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        hint = ""
        if exc.code == 403:
            hint = "\n>>> 403 can mean Log Analytics Reader RBAC propagation lag or removed access."
        raise SystemExit(f"HTTP {exc.code} for {url}\n{body}{hint}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Request failed for {url}: {exc}") from exc


def log_call(url: str, kql: str, body: str, output_path: Path | None, dry_run: bool = False) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    path = LOG_DIR / f"{timestamp}-appinsights_query.txt"
    path.write_text(
        f"URL: {url}\nQuery: {kql}\nOutput: {output_path or '(stdout)'}\nBytes: {len(body)}\nDryRun: {dry_run}\n",
        encoding="utf-8",
    )
    return path


def table_to_rows(parsed: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    tables = parsed.get("tables", [])
    if not tables:
        return [], []
    table = tables[0]
    return [column["name"] for column in table.get("columns", [])], table.get("rows", [])


def write_or_print_text(text: str, output: Path | None) -> None:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"written to {output}", file=sys.stderr)
    else:
        print(text, end="" if text.endswith("\n") else "\n")


def write_or_print_csv(header: list[str], rows: list[list[Any]], output: Path | None) -> None:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"{len(rows)} rows written to {output}", file=sys.stderr)
    else:
        writer = csv.writer(sys.stdout)
        writer.writerow(header)
        writer.writerows(rows)


def emit_services(parsed: dict[str, Any], timespan: str, env: str) -> None:
    header, rows = table_to_rows(parsed)
    ridx = header.index("_ResourceId") if "_ResourceId" in header else 0
    eidx = header.index("events") if "events" in header else 1
    print(f"Backend services/components seen in {timespan} (env={env}):")
    for row in rows:
        resource = str(row[ridx] or "")
        print(f"  {row[eidx]:>10}  {resource.split('/')[-1]}")


def main() -> int:
    args = parse_args()

    if args.check_credentials:
        source, mapping = pick_config_source(args.vars)
        missing = missing_keys(mapping)
        print(f"Credential source: {source}")
        if missing:
            print("Missing keys: " + ", ".join(missing))
            return 1
        print("Required App Insights keys: present")
        print("No token requested and no KQL executed")
        return 0

    timespan = query_timespan(args)
    kql = build_kql(args)

    if args.print_query and not args.dry_run:
        print(kql)

    if args.dry_run:
        log_path = log_call("(dry-run)", kql, "", args.output, dry_run=True)
        print(f"Dry run OK; KQL was not executed. Query logged to {log_path}")
        print(kql)
        return 0

    source, mapping = pick_config_source(args.vars)
    config = config_from_mapping(mapping)
    print(f"Using App Insights credential source: {source}", file=sys.stderr)
    token = get_token(config)
    url, body = run_query(config, token, kql, timespan)
    log_call(url, kql, body, args.output)

    parsed = json.loads(body)
    if "tables" not in parsed:
        raise SystemExit(f"Unexpected response: {json.dumps(parsed, indent=2)}")

    if args.list_services:
        emit_services(parsed, timespan, args.env)
        return 0
    if args.raw:
        write_or_print_text(json.dumps(parsed, indent=2) + "\n", args.output)
        return 0

    header, rows = table_to_rows(parsed)
    write_or_print_csv(header, rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

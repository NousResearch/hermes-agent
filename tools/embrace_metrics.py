#!/usr/bin/env python3
"""Query the Embrace Metrics API (Prometheus/PromQL) for YallaPlay apps.

Read-only. Credentials stay outside git.

Credential resolution:
1. EMBRACE_METRICS_API_TOKEN environment variable.
2. --vars / HERMES_YALLAPLAY_VARS / YALLAPLAY_VARS_TOML private TOML file.
3. ./vars.toml if present locally and untracked.
4. ../yallaplay-analytics-agent-gpt/vars.toml for migration labs.

The Embrace Metrics API token is different from the Embrace MCP service-account
token. MCP uses emb_sa_* against https://mcp.embrace.io/mcp; this script uses the
org Metrics API token against https://api.embrace.io/metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11
    import tomli as tomllib  # type: ignore[no-redef]

PROJECT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_DIR / "logs"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
LOCAL_VARS = PROJECT_DIR / "vars.toml"
DEFAULT_SIBLING_VARS = PROJECT_DIR.parent / "yallaplay-analytics-agent-gpt" / "vars.toml"

BASE_URL = "https://api.embrace.io/metrics"

PROD_APPS = {
    "r5GWq": "Spades Masters / android",
    "QkTz6": "Spades Masters / ios",
    "dmma2": "Gin Rummy / android",
    "s8kti": "Gin Rummy / ios",
}
APP_ALIASES = {
    "spades_android": "r5GWq",
    "spades_ios": "QkTz6",
    "rummy_android": "dmma2",
    "rummy_ios": "s8kti",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-q", "--query", help="PromQL query string")
    parser.add_argument(
        "--app",
        choices=sorted(APP_ALIASES),
        help="Typo guard: query must target exactly this production app alias",
    )
    parser.add_argument("--list-apps", action="store_true", help="Print the four in-scope production apps")
    parser.add_argument("--metrics", action="store_true", help="List available metric names")
    parser.add_argument("--range", action="store_true", help="Use query_range instead of instant query")
    parser.add_argument("--days", type=float, default=7, help="For --range without --start, look back N days")
    parser.add_argument("--start", help="Range start, ISO-8601 or unix seconds")
    parser.add_argument("--end", help="Range end, ISO-8601 or unix seconds; default now")
    parser.add_argument("--step", default="1h", help="Range step: 300, 30m, 1h, 1d; Embrace rounds <1h up")
    parser.add_argument("--at", help="Instant-query evaluation time, ISO-8601 or unix seconds")
    parser.add_argument("--vars", type=Path, help="Private TOML file containing EMBRACE_METRICS_API_TOKEN")
    parser.add_argument("-o", "--output", type=Path, help="Write CSV/raw output to this path")
    parser.add_argument("--raw", action="store_true", help="Emit raw JSON instead of CSV")
    return parser.parse_args()


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def candidate_vars_paths(explicit: Path | None) -> list[Path]:
    env = os.environ
    candidates: list[Path | None] = [
        explicit,
        Path(env["HERMES_YALLAPLAY_VARS"]) if env.get("HERMES_YALLAPLAY_VARS") else None,
        Path(env["YALLAPLAY_VARS_TOML"]) if env.get("YALLAPLAY_VARS_TOML") else None,
        LOCAL_VARS if LOCAL_VARS.exists() else None,
        DEFAULT_SIBLING_VARS if DEFAULT_SIBLING_VARS.exists() else None,
    ]
    return [path for path in candidates if path and path.exists()]


def load_token(vars_path: Path | None) -> tuple[str, str]:
    if os.environ.get("EMBRACE_METRICS_API_TOKEN"):
        return os.environ["EMBRACE_METRICS_API_TOKEN"], "environment"
    for path in candidate_vars_paths(vars_path):
        config = load_toml(path)
        token = config.get("EMBRACE_METRICS_API_TOKEN")
        if token:
            return str(token), str(path)
    raise SystemExit(
        "EMBRACE_METRICS_API_TOKEN missing. Set the environment variable or pass --vars / "
        "HERMES_YALLAPLAY_VARS pointing to a private TOML file."
    )


def app_ids_in_query(query: str) -> set[str]:
    ids: set[str] = set()
    for match in re.finditer(r'app_id\s*=~?\s*"([^"]*)"', query):
        for piece in match.group(1).split("|"):
            piece = piece.strip()
            if piece:
                ids.add(piece)
    return ids


def validate_scope(query: str, app_alias: str | None) -> None:
    ids = app_ids_in_query(query)
    if app_alias:
        target = APP_ALIASES[app_alias]
        if not ids:
            raise SystemExit(
                f'--app {app_alias} ({target}) requires an explicit app_id filter, e.g. '
                f'-q \'sum(daily_sessions_total{{app_id="{target}"}})\''
            )
        if ids != {target}:
            raise SystemExit(f"--app {app_alias} expects app_id {target}, but query references {sorted(ids)}")
    bad = sorted(app_id for app_id in ids if app_id not in PROD_APPS)
    if bad:
        raise SystemExit(
            "Query references non-production Embrace app_id(s): "
            + ", ".join(bad)
            + "\nAllowed production app_ids: "
            + ", ".join(f"{app_id} ({name})" for app_id, name in PROD_APPS.items())
        )


def step_to_seconds(step: str) -> int:
    value = str(step).strip().lower()
    if value.isdigit():
        return int(value)
    unit = value[-1]
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}.get(unit)
    if mult is None:
        raise SystemExit(f"Bad --step {step!r}; use e.g. 300, 30m, 1h, 1d")
    return int(float(value[:-1]) * mult)


def parse_time(value: str) -> float:
    value = str(value).strip()
    if value.replace(".", "", 1).isdigit():
        return float(value)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError as exc:
        raise SystemExit(f"Bad time {value!r}; use ISO-8601 or unix seconds") from exc


def request_metrics(path: str, params: dict[str, Any], token: str, method: str = "POST") -> tuple[str, str]:
    url = f"{BASE_URL}/api/v1/{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": "yallaplay-hermes-agent/1.0",
    }
    if method == "GET":
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers=headers)
    else:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        req = urllib.request.Request(url, data=urllib.parse.urlencode(params).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return url, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code} for {url}\n{body}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Request failed for {url}: {exc}") from exc


def log_call(url: str, body: str, output_path: Path | None) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    path = LOG_DIR / f"{timestamp}-embrace_metrics.txt"
    path.write_text(f"URL: {url}\nOutput: {output_path or '(stdout)'}\nBytes: {len(body)}\n", encoding="utf-8")
    return path


def flatten_rows(parsed: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    result = parsed.get("data", {}).get("result", [])
    result_type = parsed.get("data", {}).get("resultType")
    expanded: list[tuple[dict[str, Any], Any, Any]] = []
    for series in result:
        metric = series.get("metric", {})
        if result_type == "matrix":
            expanded.extend((metric, ts, value) for ts, value in series.get("values", []))
        else:
            ts, value = series.get("value", [None, None])
            expanded.append((metric, ts, value))
    label_keys = sorted({key for metric, _, _ in expanded for key in metric})
    header = label_keys + ["timestamp_unix", "timestamp_iso", "value"]
    rows = []
    for metric, ts, value in expanded:
        iso = datetime.fromtimestamp(float(ts), timezone.utc).isoformat() if ts is not None else ""
        rows.append([metric.get(key, "") for key in label_keys] + [ts, iso, value])
    return header, rows


def write_or_print_text(text: str, output: Path | None) -> None:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"written to {output}", file=sys.stderr)
    else:
        print(text, end="" if text.endswith("\n") else "\n")


def main() -> int:
    args = parse_args()
    if args.list_apps:
        print("In-scope production apps (Embrace):")
        for alias in sorted(APP_ALIASES):
            app_id = APP_ALIASES[alias]
            print(f"  {alias:16} {app_id}  {PROD_APPS[app_id]}")
        return 0

    if not args.query and not args.metrics:
        raise SystemExit("one of -q/--query, --metrics, or --list-apps is required")
    if args.query:
        validate_scope(args.query, args.app)

    token, source = load_token(args.vars)
    print(f"Using Embrace metrics credential source: {source}", file=sys.stderr)

    if args.metrics:
        url, body = request_metrics("label/__name__/values", {}, token, method="GET")
        names = json.loads(body).get("data", [])
        log_call(url, json.dumps(names), args.output)
        write_or_print_text("\n".join(names) + "\n", args.output)
        return 0

    if args.range:
        end = parse_time(args.end) if args.end else datetime.now(timezone.utc).timestamp()
        start = parse_time(args.start) if args.start else end - args.days * 86400
        url, body = request_metrics(
            "query_range",
            {"query": args.query, "start": start, "end": end, "step": step_to_seconds(args.step)},
            token,
        )
    else:
        params: dict[str, Any] = {"query": args.query}
        if args.at:
            params["time"] = parse_time(args.at)
        url, body = request_metrics("query", params, token)

    log_call(url, body, args.output)
    parsed = json.loads(body)
    if parsed.get("status") != "success":
        raise SystemExit(f"Query error: {json.dumps(parsed, indent=2)}")

    if args.raw:
        write_or_print_text(json.dumps(parsed, indent=2) + "\n", args.output)
        return 0

    header, rows = flatten_rows(parsed)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"{len(rows)} rows written to {args.output}", file=sys.stderr)
    else:
        writer = csv.writer(sys.stdout)
        writer.writerow(header)
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

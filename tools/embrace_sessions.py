#!/usr/bin/env python3
"""Pull session-level data from the Embrace dashboard backend.

Read-only. This covers the gap that Embrace MCP and Metrics API do not expose:
individual stitched session records for a specific user around a support-ticket
timestamp.

Credential resolution:
1. EMBRACE_DASH_EMAIL / EMBRACE_DASH_PASSWORD environment variables.
2. --vars / HERMES_YALLAPLAY_VARS / YALLAPLAY_VARS_TOML private TOML file.
3. ./vars.toml if present locally and untracked.
4. ../yallaplay-analytics-agent-gpt/vars.toml for migration labs.

Auth cache: .local/cache/.embrace_auth.json (gitignored, chmod 600). The cached
`sessionid` JWT is a credential; never print or commit it.

Requires for real login: python package `playwright` and a Chromium browser from
`python3 -m playwright install chromium`.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from secret_config import candidate_vars_paths, load_toml

PROJECT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_DIR / "logs"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
LOCAL_STATE_DIR = PROJECT_DIR / ".local"
AUTH_CACHE_PATH = LOCAL_STATE_DIR / "cache" / ".embrace_auth.json"
AUTH_REFRESH_MARGIN_S = 86400
LOGIN_URL = "https://dash.embrace.io/login"
API_BASE = "https://dash-api-us1.embrace.io"
PAGE_DELAY_S = 0.3

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

CSV_FIELDS = [
    "session_id",
    "user_id",
    "device_id",
    "app_version",
    "os_name",
    "os_version",
    "device_model",
    "manufacturer",
    "country",
    "region",
    "environment",
    "start_time",
    "end_time",
    "cold_start",
    "state",
    "last_view",
    "exceptions",
    "error_logs",
    "network_errors",
    "slow_spans",
    "bad_moments",
    "memory_warnings",
    "app_used_mb",
    "sdk_version",
]


class AuthExpired(Exception):
    """Raised when the cached JWT is rejected and a fresh login is needed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app", choices=sorted(APP_ALIASES), help="Production app alias")
    parser.add_argument("--list-apps", action="store_true", help="Print the four in-scope production apps")
    parser.add_argument("--user", help="Warehouse USER_ID / Helpshift meta_user_id filter")
    parser.add_argument("--around", metavar="ISO_TS", help="Return sessions within +/- --window minutes of this timestamp")
    parser.add_argument("--window", type=int, default=30, metavar="MIN", help="Half-width of --around window in minutes")
    parser.add_argument("--pages", type=int, default=3, help="Maximum paginated requests to make")
    parser.add_argument("--resolution", default="hour", choices=["hour", "day", "week"], help="stitch_list resolution bucket")
    parser.add_argument("--vars", type=Path, help="Private TOML file with EMBRACE_DASH_EMAIL/PASSWORD")
    parser.add_argument("-o", "--output", type=Path, help="Write flattened CSV or --raw JSON to this path")
    parser.add_argument("--raw", action="store_true", help="Emit raw session JSON instead of flattened CSV")
    parser.add_argument("--headful", action="store_true", help="Run browser headed for login debugging")
    parser.add_argument("--relogin", action="store_true", help="Ignore cached JWT and force a fresh dashboard login")
    parser.add_argument("--dry-run", action="store_true", help="Validate arguments and credential presence without logging in or calling Embrace")
    return parser.parse_args()


def load_creds(vars_path: Path | None) -> tuple[str, str, str]:
    email = os.environ.get("EMBRACE_DASH_EMAIL")
    password = os.environ.get("EMBRACE_DASH_PASSWORD")
    if email and password:
        return email, password, "environment"
    for path in candidate_vars_paths(vars_path):
        config = load_toml(path)
        email = config.get("EMBRACE_DASH_EMAIL")
        password = config.get("EMBRACE_DASH_PASSWORD")
        if email and password:
            return str(email), str(password), str(path)
    raise SystemExit(
        "EMBRACE_DASH_EMAIL / EMBRACE_DASH_PASSWORD missing. Set environment variables "
        "or pass --vars / HERMES_YALLAPLAY_VARS pointing to a private TOML file."
    )


def jwt_exp(jwt: str) -> int | None:
    try:
        payload_b64 = jwt.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        return int(payload.get("exp")) if payload.get("exp") else None
    except Exception:
        return None


def read_cached_sessionid() -> str | None:
    try:
        data = json.loads(AUTH_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    jwt = data.get("sessionid")
    if not jwt:
        return None
    exp = jwt_exp(jwt) or data.get("exp")
    if exp and exp - time.time() > AUTH_REFRESH_MARGIN_S:
        remaining_h = (exp - time.time()) / 3600
        print(f"using cached dashboard auth (valid ~{remaining_h:.0f}h)", file=sys.stderr)
        return str(jwt)
    return None


def write_cached_sessionid(jwt: str) -> None:
    AUTH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUTH_CACHE_PATH.write_text(json.dumps({"sessionid": jwt, "exp": jwt_exp(jwt)}), encoding="utf-8")
    try:
        os.chmod(AUTH_CACHE_PATH, 0o600)
    except OSError:
        pass


def login_and_capture_sessionid(app_id: str, vars_path: Path | None, headless: bool = True, timeout_ms: int = 45000) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: playwright. Install in the active environment, then run: "
            "python3 -m playwright install chromium"
        ) from exc

    email, password, source = load_creds(vars_path)
    print(f"Using Embrace dashboard credential source: {source}", file=sys.stderr)
    captured: dict[str, str | None] = {"sessionid": None}

    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=headless)
    context = browser.new_context()
    page = context.new_page()

    def on_request(request: Any) -> None:
        if captured["sessionid"] is None:
            sessionid = request.headers.get("sessionid")
            if sessionid:
                captured["sessionid"] = sessionid

    page.on("request", on_request)
    try:
        page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=timeout_ms)
        time.sleep(2)
        page.fill("input[name=email]", email)
        page.fill("input[type=password]", password)
        page.click("button:has-text('Log in')")
        time.sleep(6)
        try:
            page.goto(f"https://dash.embrace.io/app/{app_id}", wait_until="domcontentloaded", timeout=timeout_ms)
        except Exception:
            pass
        for _ in range(15):
            if captured["sessionid"]:
                break
            time.sleep(1)
    finally:
        browser.close()
        playwright.stop()

    if not captured["sessionid"]:
        raise SystemExit("login failed or sessionid JWT was not captured; retry with --headful to inspect")
    write_cached_sessionid(captured["sessionid"])
    return captured["sessionid"]


def get_sessionid(app_id: str, vars_path: Path | None, headless: bool, force_login: bool) -> str:
    if not force_login:
        cached = read_cached_sessionid()
        if cached:
            return cached
    return login_and_capture_sessionid(app_id, vars_path, headless=headless)


def seed_cursor(before_iso: str) -> str:
    raw = json.dumps({"n": before_iso, "d": {}}).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def fetch_sessions(
    sessionid: str,
    app_id: str,
    resolution: str,
    user_id: str | None = None,
    pages: int = 3,
    around: str | None = None,
    window_min: int = 30,
) -> list[dict[str, Any]]:
    url = f"{API_BASE}/v4/app/{app_id}/session/stitch_list"
    headers = {
        "content-type": "application/json",
        "accept": "application/json",
        "sessionid": sessionid,
        "referer-embrace": f"https://dash.embrace.io/app/{app_id}/grouped_sessions/{resolution}",
        "user-agent": "yallaplay-hermes-agent/1.0",
    }
    user_filter = None
    if user_id:
        user_filter = {"op": "and", "children": [{"key": "user_id", "field_op": "eq", "val": str(user_id)}]}

    lo = hi = None
    cursor = None
    if around:
        center = parse_iso(around)
        lo = center - timedelta(minutes=window_min)
        hi = center + timedelta(minutes=window_min)
        cursor = seed_cursor((hi + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S.999Z"))

    sessions: list[dict[str, Any]] = []
    page_i = 0
    stop = False
    while page_i < pages and not stop:
        body: dict[str, Any] = {"resolution": resolution}
        if user_filter:
            body["filters"] = user_filter
        if cursor:
            body["next"] = cursor
        request = urllib.request.Request(url, method="POST", data=json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            if exc.code in (401, 403):
                raise AuthExpired(f"HTTP {exc.code}: {detail}") from exc
            print(f"page {page_i}: HTTP {exc.code}: {detail}", file=sys.stderr)
            break

        kept = 0
        for group in payload.get("stitches", []):
            for session in group:
                if around:
                    start_time = session.get("start_time")
                    start_dt = parse_iso(start_time) if start_time else None
                    if start_dt is not None:
                        if lo and start_dt < lo:
                            stop = True
                            continue
                        if hi and start_dt > hi:
                            continue
                sessions.append(session)
                kept += 1
        cursor = payload.get("next")
        suffix = f" [in window: {kept}]" if around else ""
        print(
            f"page {page_i}: +{kept} sessions (total {len(sessions)}){suffix}{' [more]' if cursor else ' [end]'}",
            file=sys.stderr,
        )
        page_i += 1
        if not cursor:
            break
        if not stop:
            time.sleep(PAGE_DELAY_S)

    if cursor and not stop and page_i >= pages:
        extra = " in the time window" if around else ""
        print(f"NOTE: stopped at {pages}-page cap with more sessions available{extra}; raise --pages if needed", file=sys.stderr)
    return sessions


def flatten_session(session: dict[str, Any]) -> dict[str, Any]:
    user = session.get("user", {}) or {}
    out = {field: session.get(field) for field in CSV_FIELDS}
    out["user_id"] = user.get("user_id")
    out["device_id"] = user.get("device_id")
    return out


def log_call(app_id: str, count: int, output_path: Path | None) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    path = LOG_DIR / f"{timestamp}-embrace_sessions.txt"
    path.write_text(f"app_id: {app_id}\nsessions: {count}\noutput: {output_path or '(stdout)'}\n", encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    if args.list_apps:
        print("In-scope production apps (Embrace):")
        for alias in sorted(APP_ALIASES):
            app_id = APP_ALIASES[alias]
            print(f"  {alias:16} {app_id}  {PROD_APPS[app_id]}")
        return 0
    if not args.app:
        raise SystemExit("--app is required unless --list-apps is used")
    if args.pages < 1:
        raise SystemExit("--pages must be >= 1")
    if args.window < 1:
        raise SystemExit("--window must be >= 1")
    if args.around:
        parse_iso(args.around)

    app_id = APP_ALIASES[args.app]
    if args.dry_run:
        # Confirms credential presence without printing values or touching Embrace.
        _, _, source = load_creds(args.vars)
        print(f"Dry run OK: app={args.app} app_id={app_id} credential_source={source}")
        return 0

    sessionid = get_sessionid(app_id, args.vars, headless=not args.headful, force_login=args.relogin)
    try:
        sessions = fetch_sessions(sessionid, app_id, args.resolution, args.user, args.pages, args.around, args.window)
    except AuthExpired:
        print("cached auth rejected; re-logging in", file=sys.stderr)
        sessionid = get_sessionid(app_id, args.vars, headless=not args.headful, force_login=True)
        sessions = fetch_sessions(sessionid, app_id, args.resolution, args.user, args.pages, args.around, args.window)

    log_call(app_id, len(sessions), args.output)
    if args.raw:
        text = json.dumps(sessions, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
            print(f"{len(sessions)} sessions written to {args.output}", file=sys.stderr)
        else:
            print(text, end="")
        return 0

    rows = [flatten_session(session) for session in sessions]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"{len(rows)} sessions written to {args.output}", file=sys.stderr)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Refresh full backend-config snapshots and config-service history into the wiki.

Read-only. Uses only GET endpoints:
- GET /{App}/config
- GET /{App}/config?at=<ISO>
- GET /{App}/config/sections/{section}/history

Credential resolution:
1. BACKEND_CONFIG_BASE_URL + BACKEND_CONFIG_SERVER_TOKEN environment variables.
2. YALLAPLAY_CONFIG_BASE_URL + YALLAPLAY_CONFIG_SERVER_TOKEN environment variables.
3. --vars / HERMES_YALLAPLAY_VARS / YALLAPLAY_VARS_TOML private TOML file.
4. ./vars.toml if present locally and untracked.
5. ../yallaplay-analytics-agent-gpt/vars.toml for migration labs.
"""

from __future__ import annotations

import argparse
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
WIKI_CONFIG_ROOT = PROJECT_DIR / "yallaplay-wiki" / "operations" / "backend-config"
LOCAL_VARS = PROJECT_DIR / "vars.toml"
DEFAULT_SIBLING_VARS = PROJECT_DIR.parent / "yallaplay-analytics-agent-gpt" / "vars.toml"
BASE_URL_KEYS = ("BACKEND_CONFIG_BASE_URL", "YALLAPLAY_CONFIG_BASE_URL", "CONFIG_SERVICE_BASE_URL")
TOKEN_KEYS = ("BACKEND_CONFIG_SERVER_TOKEN", "YALLAPLAY_CONFIG_SERVER_TOKEN", "CONFIG_SERVICE_SERVER_TOKEN", "X_SERVER_TOKEN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app", required=True, help="Exact config-service app identifier used in /{App}/config")
    parser.add_argument("--wiki-app", help="Folder name under operations/backend-config/ (default: slugified --app)")
    parser.add_argument("--env", default="prod", help="Environment label for filenames/docs (default: prod)")
    parser.add_argument("--at", help="Point-in-time ISO timestamp for GET /{App}/config?at=<ISO>")
    parser.add_argument("--vars", type=Path, help="Private TOML file containing config service URL/token")
    parser.add_argument("--base-url", help="Override config service base URL; still requires token unless --dry-run")
    parser.add_argument("--output-root", type=Path, default=WIKI_CONFIG_ROOT, help="Backend-config wiki root")
    parser.add_argument("--snapshot", action="store_true", help="Fetch and write <env>.latest.json")
    parser.add_argument(
        "--timestamped-snapshot",
        action="store_true",
        help="Also write snapshots/<UTC>.json when fetching a snapshot",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Fetch raw section history into history/<section>.json. Uses --sections or current config keys.",
    )
    parser.add_argument(
        "--sections",
        help="Comma-separated section names for --history. If omitted, sections come from the current config snapshot.",
    )
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds (default 60)")
    parser.add_argument("--dry-run", action="store_true", help="Print planned URLs and files; do not load token or call service")
    parser.add_argument("--check-credentials", action="store_true", help="Report credential source and missing key names only")
    return parser.parse_args()


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-_.").lower()
    if not slug:
        raise SystemExit("Could not derive --wiki-app folder from --app; pass --wiki-app explicitly")
    return slug


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


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


def first_value(mapping: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if value:
            return str(value)
    return None


def credential_source(vars_path: Path | None, base_url_override: str | None = None) -> tuple[str, str | None, str | None]:
    env = dict(os.environ)
    env_base = base_url_override or first_value(env, BASE_URL_KEYS)
    env_token = first_value(env, TOKEN_KEYS)
    if env_base or env_token:
        return "environment/cli", env_base, env_token
    for path in candidate_vars_paths(vars_path):
        config = load_toml(path)
        return str(path), first_value(config, BASE_URL_KEYS), first_value(config, TOKEN_KEYS)
    return "none", base_url_override, None


def require_credentials(args: argparse.Namespace) -> tuple[str, str, str]:
    source, base_url, token = credential_source(args.vars, args.base_url)
    missing: list[str] = []
    if not base_url:
        missing.append("one of " + "/".join(BASE_URL_KEYS))
    if not token:
        missing.append("one of " + "/".join(TOKEN_KEYS))
    if missing:
        raise SystemExit(f"Config service credentials missing from {source}: {', '.join(missing)}")
    assert base_url is not None
    assert token is not None
    return source, base_url.rstrip("/"), token


def config_url(base_url: str, app: str, at: str | None = None) -> str:
    path = f"/{urllib.parse.quote(app, safe='')}/config"
    url = base_url.rstrip("/") + path
    if at:
        url += "?" + urllib.parse.urlencode({"at": at})
    return url


def history_url(base_url: str, app: str, section: str) -> str:
    app_q = urllib.parse.quote(app, safe="")
    section_q = urllib.parse.quote(section, safe="")
    return f"{base_url.rstrip('/')}/{app_q}/config/sections/{section_q}/history"


def get_json(url: str, token: str, timeout: int) -> Any:
    headers = {
        "Accept": "application/json",
        "User-Agent": "yallaplay-hermes-agent/1.0",
        "X-Server-Token": token,
    }
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code} from {url}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Failed to fetch {url}: {exc}") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Response from {url} was not JSON: {body[:500]}") from exc


def unwrap_config(payload: Any) -> Any:
    if isinstance(payload, dict) and "response" in payload and set(payload.keys()) & {"success", "error"}:
        return payload["response"]
    return payload


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def ensure_app_scaffold(app_dir: Path, app: str, env: str) -> None:
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "snapshots").mkdir(exist_ok=True)
    (app_dir / "history").mkdir(exist_ok=True)
    changes = app_dir / "changes.md"
    if not changes.exists():
        changes.write_text(
            f"---\ntitle: Backend Config Change Ledger — {app}\ndomain: operations\nstatus: live\n"
            "source_refs: []\nsee_also: [README.md, ../index.md, ../../../journal/config/index.md]\n"
            f"built: {datetime.now(timezone.utc).date().isoformat()}\n---\n\n"
            f"# Backend Config Change Ledger — {app}\n\n"
            "Append one entry per observed prod config change. Raw config-service section history is copied under `history/`.\n",
        )
    readme = app_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            f"---\ntitle: Backend Config — {app}\ndomain: operations\nstatus: live\n"
            "source_refs: []\nsee_also: [../index.md, changes.md]\n"
            f"built: {datetime.now(timezone.utc).date().isoformat()}\n---\n\n"
            f"# Backend Config — {app}\n\n"
            f"- Config-service app id: `{app}`\n"
            f"- Environment tracked here: `{env}`\n"
            "- Source of truth: config service\n"
            f"- Latest snapshot: [{env}.latest.json]({env}.latest.json)\n"
            "- Change ledger: [changes.md](changes.md)\n"
            "- Raw section history: `history/<section>.json`\n",
        )


def sections_from_args(raw: str | None, config: Any) -> list[str]:
    if raw:
        return sorted({section.strip() for section in raw.split(",") if section.strip()})
    if isinstance(config, dict):
        return sorted(str(key) for key in config.keys())
    raise SystemExit("--history without --sections requires --snapshot/current config that unwraps to an object")


def print_plan(args: argparse.Namespace) -> None:
    app_slug = args.wiki_app or slugify(args.app)
    app_dir = args.output_root / app_slug
    base_url = (args.base_url or "${BACKEND_CONFIG_BASE_URL}").rstrip("/")
    print(f"app_dir={app_dir}")
    if args.snapshot:
        print(f"GET {config_url(base_url, args.app, args.at)} -> {app_dir / (args.env + '.latest.json')}")
        if args.timestamped_snapshot:
            print(f"also -> {app_dir / 'snapshots' / '<UTC>.json'}")
    if args.history:
        sections = [s.strip() for s in args.sections.split(",")] if args.sections else ["<sections from config keys>"]
        for section in sections:
            print(f"GET {history_url(base_url, args.app, section)} -> {app_dir / 'history' / (section + '.json')}")


def main() -> int:
    args = parse_args()
    if not args.snapshot and not args.history and not args.check_credentials and not args.dry_run:
        raise SystemExit("Choose --snapshot, --history, --check-credentials, or --dry-run")
    if args.check_credentials:
        source, base_url, token = credential_source(args.vars, args.base_url)
        missing = []
        if not base_url:
            missing.append("base_url")
        if not token:
            missing.append("server_token")
        print(json.dumps({"source": source, "base_url_present": bool(base_url), "token_present": bool(token), "missing": missing}, indent=2))
        return 0 if not missing else 2
    if args.dry_run:
        print_plan(args)
        return 0

    _, base_url, token = require_credentials(args)
    app_slug = args.wiki_app or slugify(args.app)
    app_dir = args.output_root / app_slug
    ensure_app_scaffold(app_dir, args.app, args.env)

    stamp = utc_stamp()
    config: Any | None = None
    written: list[str] = []
    if args.snapshot or (args.history and not args.sections):
        payload = get_json(config_url(base_url, args.app, args.at), token, args.timeout)
        config = unwrap_config(payload)
        if args.snapshot:
            latest_path = app_dir / f"{args.env}.latest.json"
            dump_json(latest_path, config)
            written.append(str(latest_path))
            if args.timestamped_snapshot:
                snapshot_path = app_dir / "snapshots" / f"{stamp}.json"
                dump_json(snapshot_path, config)
                written.append(str(snapshot_path))

    if args.history:
        sections = sections_from_args(args.sections, config)
        for section in sections:
            payload = get_json(history_url(base_url, args.app, section), token, args.timeout)
            history_payload = {
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "app": args.app,
                "environment": args.env,
                "section": section,
                "source": history_url(base_url, args.app, section),
                "response": payload,
            }
            path = app_dir / "history" / f"{slugify(section)}.json"
            dump_json(path, history_payload)
            written.append(str(path))

    print(json.dumps({"written": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

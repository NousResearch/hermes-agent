#!/usr/bin/env python3
"""
ninja_patches.py — Pull OS patch state for all devices in a NinjaOne organization.

Single-call (cursor-paginated) wrapper over GET /v2/queries/os-patches, scoped to one
organization via the device filter (df=org=<id>). Designed for two uses:

  1. CLI one-off:  python3 ninja_patches.py --org "Acme Corp" --status APPROVED
  2. Hermes tool:  from ninja_patches import get_org_patches
                   data = get_org_patches(org="Acme Corp", status="APPROVED")

Credentials come from environment variables (NINJAONE_CLIENT_ID / NINJAONE_CLIENT_SECRET),
never hard-coded and never passed to the model. Read-only "monitoring" scope is sufficient.

Note on the os-patches query: the endpoint is documented as returning patches with no
installation attempt yet (pending / approved side). If `status=APPROVED` returns nothing
on your tenant, try `status=PENDING` — see README.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from typing import Any

# --- Configuration -----------------------------------------------------------

BASE_URL = os.environ.get("NINJAONE_BASE_URL", "https://app.ninjarmm.com").rstrip("/")
TOKEN_PATH = "/ws/oauth/token"
ORGS_PATH = "/v2/organizations"
OS_PATCHES_PATH = "/v2/queries/os-patches"

# Read-only scope. Do not broaden unless a tool genuinely needs write access.
OAUTH_SCOPE = "monitoring"

DEFAULT_PAGE_SIZE = 1000          # max records per page; the cursor loop handles the rest
HTTP_TIMEOUT = 30                 # seconds per request
MAX_RETRIES = 3                   # transient-error retries (429 / 5xx)
RETRY_BACKOFF = 2.0               # seconds, multiplied by attempt number

# Where save_to_disk writes when no explicit path is given. Override via env.
DEFAULT_OUTPUT_DIR = os.environ.get("NINJAONE_OUTPUT_DIR", "/tmp/ninja-patches")


class NinjaError(RuntimeError):
    """Raised for any unrecoverable NinjaOne API interaction failure."""


# --- Low-level HTTP ----------------------------------------------------------

def _request(method: str, url: str, *, headers: dict[str, str],
             data: bytes | None = None) -> Any:
    """Perform an HTTP request with basic retry on 429/5xx. Returns parsed JSON."""
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            # Retry transient server-side / rate-limit errors; fail fast on the rest.
            if e.code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * attempt
                time.sleep(wait)
                last_err = e
                continue
            detail = e.read().decode("utf-8", errors="replace")
            raise NinjaError(f"HTTP {e.code} on {method} {url}: {detail}") from e
        except urllib.error.URLError as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
                last_err = e
                continue
            raise NinjaError(f"Network error on {method} {url}: {e}") from e
    raise NinjaError(f"Exhausted retries on {method} {url}: {last_err}")


# --- Auth --------------------------------------------------------------------

def get_token(client_id: str, client_secret: str) -> str:
    """Exchange client credentials for a bearer token (client_credentials grant)."""
    url = f"{BASE_URL}{TOKEN_PATH}"
    form = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": OAUTH_SCOPE,
    }).encode("utf-8")
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    payload = _request("POST", url, headers=headers, data=form)
    token = payload.get("access_token")
    if not token:
        raise NinjaError(f"No access_token in token response: {payload}")
    return token


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


# --- Org resolution ----------------------------------------------------------

def resolve_org_id(token: str, org: str | int) -> int:
    """
    Accept either an org ID (int or numeric string) or an org name.
    For a name, fetch /v2/organizations and match case-insensitively.
    Raises if zero or multiple names match (ambiguous input should not silently pick one).
    """
    # Already an ID?
    if isinstance(org, int):
        return org
    if isinstance(org, str) and org.strip().isdigit():
        return int(org.strip())

    orgs = _request("GET", f"{BASE_URL}{ORGS_PATH}", headers=_auth_headers(token))
    if not isinstance(orgs, list):
        raise NinjaError(f"Unexpected /organizations response shape: {type(orgs)}")

    needle = str(org).strip().lower()
    exact = [o for o in orgs if str(o.get("name", "")).strip().lower() == needle]
    if len(exact) == 1:
        return int(exact[0]["id"])
    if len(exact) > 1:
        ids = ", ".join(str(o.get("id")) for o in exact)
        raise NinjaError(f"Multiple organizations named '{org}' (ids: {ids}); pass an ID.")

    # Fall back to substring match to be forgiving, but still require uniqueness.
    partial = [o for o in orgs if needle in str(o.get("name", "")).strip().lower()]
    if len(partial) == 1:
        return int(partial[0]["id"])
    if len(partial) > 1:
        names = "; ".join(f"{o.get('name')} (id={o.get('id')})" for o in partial)
        raise NinjaError(f"Ambiguous org match for '{org}': {names}. Pass an exact name or ID.")

    raise NinjaError(f"No organization matched '{org}'.")


# --- The query ---------------------------------------------------------------

def get_org_patches(
    org: str | int,
    *,
    status: str | None = "APPROVED",
    patch_type: str | None = None,
    severity: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    token: str | None = None,
    save_to_disk: bool = False,
    output_path: str | None = None,
    include_data: bool = True,
) -> dict[str, Any]:
    """
    Return OS patch records for every device in `org`, optionally filtered by
    status / type / severity. This is the function to register as a Hermes tool.

    Args:
        org:        Organization name (str) or organization ID (int / numeric str).
        status:     Patch status filter. Common values: PENDING, APPROVED, FAILED,
                    REJECTED, INSTALLED. Pass None to omit the filter entirely
                    (returns all statuses this endpoint exposes). Default "APPROVED".
        patch_type: Optional NinjaOne patch type filter (e.g. SECURITY_UPDATES).
        severity:   Optional severity filter (e.g. CRITICAL).
        client_id / client_secret:
                    OAuth credentials. If omitted, read from environment.
        page_size:  Records per page for cursor pagination.
        token:      Pre-fetched bearer token (skips the token call if supplied).
        save_to_disk: If True, write the full result JSON to a file and include its
                    path in the return value. The file always contains the complete
                    data regardless of `include_data`.
        output_path: Explicit file path to write to (implies save_to_disk). If omitted
                    and save_to_disk is True, a timestamped file is written under
                    NINJAONE_OUTPUT_DIR (default /tmp/ninja-patches).
        include_data: If True (default), the returned dict includes the full "patches"
                    list. Set False to return only summary + count + file_path — use this
                    for large orgs so the agent's context stays light and it just hands
                    the file path to the next tool (e.g. a OneDrive upload connector).

    Returns:
        {
          "organization": {"query": <input>, "id": <resolved id>},
          "filters": {"status": ..., "type": ..., "severity": ...},
          "count": <int>,
          "summary": "<human-readable one-liner>",
          "file_path": <str or null>,        # present when saved to disk
          "patches": [ <record>, ... ]       # omitted when include_data is False
        }
    """
    if token is None:
        cid = client_id or os.environ.get("NINJAONE_CLIENT_ID")
        csec = client_secret or os.environ.get("NINJAONE_CLIENT_SECRET")
        if not cid or not csec:
            raise NinjaError(
                "Missing credentials: set NINJAONE_CLIENT_ID and NINJAONE_CLIENT_SECRET "
                "(or pass client_id/client_secret)."
            )
        token = get_token(cid, csec)

    org_id = resolve_org_id(token, org)

    # The device filter is a URL-encoded string; org scoping is df=org=<id>.
    df_value = f"org={org_id}"

    base_params: list[tuple[str, str]] = [("df", df_value), ("pageSize", str(page_size))]
    if status:
        base_params.append(("status", status))
    if patch_type:
        base_params.append(("type", patch_type))
    if severity:
        base_params.append(("severity", severity))

    headers = _auth_headers(token)
    all_records: list[Any] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()  # guard against a cursor that never advances

    while True:
        params = list(base_params)
        if cursor:
            params.append(("cursor", cursor))
        query = urllib.parse.urlencode(params)
        url = f"{BASE_URL}{OS_PATCHES_PATH}?{query}"

        payload = _request("GET", url, headers=headers)

        # NinjaOne query endpoints return {"cursor": {...}, "results": [...]} shaped data.
        # Be defensive about the exact field names across tenant/API versions.
        records = (
            payload.get("results")
            or payload.get("data")
            or (payload if isinstance(payload, list) else [])
        )
        if isinstance(records, list):
            all_records.extend(records)

        next_cursor = _extract_cursor(payload)
        if not next_cursor or next_cursor in seen_cursors:
            break
        seen_cursors.add(next_cursor)
        cursor = next_cursor

    count = len(all_records)
    filt_desc = status or "ALL statuses"
    if patch_type:
        filt_desc += f", type={patch_type}"
    if severity:
        filt_desc += f", severity={severity}"
    summary = f"{count} OS patch record(s) for org {org_id} ({filt_desc})."

    # The full, complete result — always written to disk in its entirety when saving.
    full_result: dict[str, Any] = {
        "organization": {"query": org, "id": org_id},
        "filters": {"status": status, "type": patch_type, "severity": severity},
        "count": count,
        "summary": summary,
        "patches": all_records,
    }

    file_path: str | None = None
    if save_to_disk or output_path:
        file_path = _write_result(full_result, org_id, output_path)

    # What we return to the caller/agent. Drop the bulky list when include_data is False.
    result = dict(full_result)
    result["file_path"] = file_path
    if not include_data:
        result.pop("patches", None)
    return result


def _write_result(result: dict[str, Any], org_id: int,
                  output_path: str | None) -> str:
    """Write the full result JSON to disk and return the absolute path."""
    if output_path:
        path = output_path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    else:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        status = (result.get("filters") or {}).get("status") or "all"
        safe_status = re.sub(r"[^A-Za-z0-9]+", "", str(status)).lower() or "all"
        fname = f"patches_org{org_id}_{safe_status}_{stamp}.json"
        path = os.path.join(DEFAULT_OUTPUT_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return os.path.abspath(path)


def _extract_cursor(payload: Any) -> str | None:
    """Pull the next-page cursor name from a query response, tolerating shape variants."""
    if not isinstance(payload, dict):
        return None
    cur = payload.get("cursor")
    if isinstance(cur, dict):
        # A cursor with no records remaining signals the end on Ninja's query endpoints.
        name = cur.get("name")
        count = cur.get("count")
        if name and (count is None or count > 0):
            return name
        return None
    if isinstance(cur, str):
        return cur or None
    return None


# --- CLI ---------------------------------------------------------------------

def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch OS patch state for all devices in a NinjaOne organization."
    )
    parser.add_argument("--org", required=True,
                        help="Organization name or numeric ID.")
    parser.add_argument("--status", default="APPROVED",
                        help="Patch status filter (PENDING/APPROVED/FAILED/REJECTED/"
                             "INSTALLED). Use 'ALL' to omit the filter. Default: APPROVED.")
    parser.add_argument("--type", dest="patch_type", default=None,
                        help="Patch type filter, e.g. SECURITY_UPDATES.")
    parser.add_argument("--severity", default=None,
                        help="Severity filter, e.g. CRITICAL.")
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE,
                        help=f"Records per page (default {DEFAULT_PAGE_SIZE}).")
    parser.add_argument("--out", default=None,
                        help="Write JSON to this file (full data always written here).")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print only summary + count + file_path, not the patch list. "
                             "Useful with --out for large orgs.")
    args = parser.parse_args(argv)

    status = None if str(args.status).upper() == "ALL" else args.status

    try:
        result = get_org_patches(
            org=args.org,
            status=status,
            patch_type=args.patch_type,
            severity=args.severity,
            page_size=args.page_size,
            output_path=args.out,
            include_data=not args.summary_only,
        )
    except NinjaError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    output = json.dumps(result, indent=2)
    if args.out:
        # File already written inside get_org_patches; just report.
        print(f"Wrote {result['count']} patch records to {result['file_path']}",
              file=sys.stderr)
        if args.summary_only:
            print(output)
    else:
        print(output)
    return 0


# --- Hermes tool registration ------------------------------------------------

GET_ORG_PATCHES_SCHEMA = {
    "name": "get_org_patches",
    "description": (
        "Return OS patch records for every device in a NinjaOne organization, "
        "optionally filtered by status/type/severity. Use status=APPROVED for "
        "patches that are due (approved but not yet installed). Credentials are "
        "read from NINJAONE_CLIENT_ID/NINJAONE_CLIENT_SECRET in the environment; "
        "never pass secrets as tool arguments."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "org": {
                "type": "string",
                "description": "Organization name or numeric organization ID.",
            },
            "status": {
                "type": "string",
                "description": (
                    "Patch status filter. Common values: PENDING, APPROVED, FAILED, "
                    "REJECTED, INSTALLED. Use ALL to omit the status filter."
                ),
                "default": "APPROVED",
            },
            "patch_type": {
                "type": "string",
                "description": "Optional NinjaOne patch type filter, e.g. SECURITY_UPDATES.",
            },
            "severity": {
                "type": "string",
                "description": "Optional severity filter, e.g. CRITICAL.",
            },
            "save_to_disk": {
                "type": "boolean",
                "description": "Write the full JSON result to disk and return the file_path.",
                "default": False,
            },
            "output_path": {
                "type": "string",
                "description": (
                    "Optional explicit local output path. Implies save_to_disk. "
                    "Use only for local paths, not OneDrive/Graph destinations."
                ),
            },
            "include_data": {
                "type": "boolean",
                "description": (
                    "Include the full patches list in the tool result. Set false for "
                    "large orgs or when another connector only needs file_path."
                ),
                "default": True,
            },
        },
        "required": ["org"],
        "additionalProperties": False,
    },
}


def _check_ninjaone_requirements() -> bool:
    return bool(os.environ.get("NINJAONE_CLIENT_ID") and os.environ.get("NINJAONE_CLIENT_SECRET"))


def _handle_get_org_patches(args: dict, **kwargs) -> str:
    # Keep credentials intentionally out of the handler arguments/schema. They are
    # read only from environment by get_org_patches() so secrets never enter the
    # model/tool-call transcript.
    status_arg = args.get("status", "APPROVED")
    status = None if status_arg is None or str(status_arg).upper() == "ALL" else str(status_arg)
    try:
        result = get_org_patches(
            org=args["org"],
            status=status,
            patch_type=args.get("patch_type") or None,
            severity=args.get("severity") or None,
            save_to_disk=bool(args.get("save_to_disk", False)),
            output_path=args.get("output_path") or None,
            include_data=bool(args.get("include_data", True)),
        )
        return json.dumps(result, ensure_ascii=False)
    except NinjaError as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


try:
    from tools.registry import registry
except Exception:
    # Preserve standalone CLI/import behavior when this file is copied outside a
    # Hermes checkout. The top-level registry.register(...) call below is still
    # present so Hermes auto-discovery detects the module.
    class _NoopRegistry:
        def register(self, *args, **kwargs):
            return None

    registry = _NoopRegistry()

registry.register(
    name="get_org_patches",
    toolset="ninjaone",
    schema=GET_ORG_PATCHES_SCHEMA,
    handler=_handle_get_org_patches,
    check_fn=_check_ninjaone_requirements,
    requires_env=["NINJAONE_CLIENT_ID", "NINJAONE_CLIENT_SECRET"],
    emoji="🥷",
    max_result_size_chars=200_000,
)


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))

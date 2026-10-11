"""Native desktop account controls for the verified hosted Composio MCP server.

Only SEARCH_TOOLS and the list/add/remove connection actions are called here. Provider action
execution remains on the ordinary native MCP tools path, where the existing tool approval checks
apply. In particular, this module never invokes COMPOSIO_MULTI_EXECUTE_TOOL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped

_MCP_SERVER = "composio"
_MCP_URL = "https://connect.composio.dev/mcp"
_SEARCH_TOOL = "COMPOSIO_SEARCH_TOOLS"
_MANAGE_TOOL = "COMPOSIO_MANAGE_CONNECTIONS"
_MCP_TIMEOUT = 45.0


@dataclass(frozen=True)
class ProviderSpec:
    label: str
    toolkit: str
    query: str


# Keep this allowlist explicit. It contains only provider/toolkit names verified by direct,
# provider-scoped Composio tool discovery. The UI cannot supply arbitrary toolkit slugs.
PROVIDERS: dict[str, ProviderSpec] = {
    "gmail": ProviderSpec("Gmail", "gmail", "Read Gmail emails"),
    "calendar": ProviderSpec("Google Calendar", "googlecalendar", "Read Google Calendar events"),
    "drive": ProviderSpec("Google Drive", "googledrive", "Find Google Drive files"),
    "notion": ProviderSpec("Notion", "notion", "Read Notion pages"),
    "supabase": ProviderSpec("Supabase", "supabase", "Read Supabase projects"),
    "twitter": ProviderSpec("Twitter", "twitter", "Read Twitter profile"),
    "youtube": ProviderSpec("YouTube", "youtube", "Read YouTube channel"),
    "tasks": ProviderSpec("Google Tasks", "googletasks", "Read Google Tasks tasks"),
    "reddit": ProviderSpec("Reddit", "reddit", "Read Reddit profile"),
    "meet": ProviderSpec("Google Meet", "googlemeet", "Read Google Meet meeting spaces"),
    "zoho_books": ProviderSpec("Zoho Books", "zoho_books", "List Zoho Books organizations"),
    "zoho": ProviderSpec("Zoho Inventory", "zoho_inventory", "List Zoho Inventory items"),
    "canva": ProviderSpec("Canva", "canva", "Read Canva designs"),
    "1password": ProviderSpec("1Password", "_1password", "Read 1Password vaults"),
    "dataforseo": ProviderSpec("DataForSEO", "dataforseo", "Query DataForSEO keyword data"),
    "fathom": ProviderSpec("Fathom", "fathom", "Read Fathom recordings"),
    "search_console": ProviderSpec("Google Search Console", "google_search_console", "Read Google Search Console sites"),
    "wispr_flow": ProviderSpec("Wispr Flow MCP", "wispr_flow_mcp", "Read Wispr Flow MCP transcription"),
    "resend": ProviderSpec("Resend", "resend", "List Resend emails"),
}


class _ComposioUnavailable(Exception):
    pass


class _McpAuthRequired(Exception):
    pass


def _normalize_status(value: Any):
    from tui_gateway.contracts.composio import ComposioAccountStatus

    normalized = str(value or "").strip().lower()
    try:
        return ComposioAccountStatus(normalized)
    except ValueError:
        return ComposioAccountStatus.unknown


def _provider_state(provider: str, spec: ProviderSpec, accounts: list):
    from tui_gateway.contracts.composio import ComposioAccountStatus, ComposioProviderStatus

    statuses = [account.status for account in accounts]
    if ComposioAccountStatus.active in statuses:
        return ComposioProviderStatus.connected
    for candidate in ("pending", "failed", "expired", "revoked", "inactive"):
        if getattr(ComposioAccountStatus, candidate) in statuses:
            return ComposioProviderStatus(candidate)
    return ComposioProviderStatus.disconnected


def _provider_rows(provider: str | None) -> list[tuple[str, ProviderSpec]]:
    if provider is None:
        return list(PROVIDERS.items())
    key = str(provider).strip()
    spec = PROVIDERS.get(key)
    if spec is None:
        raise _ComposioUnavailable("unknown provider")
    return [(key, spec)]


def _chunks(rows: list[tuple[str, ProviderSpec]], size: int = 3):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def _safe_error(rid, message: str = "Composio account controls are unavailable. Try again."):
    return _err(rid, 5034, message)


def _verified_server():
    """Return this profile's official OAuth-configured Composio server, without starting OAuth."""
    from hermes_cli import mcp_config

    try:
        configs = mcp_config._get_mcp_servers()
    except Exception:  # health: allow BLE001 -- profile config failures must not leak local details to RPC callers
        raise _ComposioUnavailable("profile MCP configuration is unavailable") from None
    config = configs.get(_MCP_SERVER) if isinstance(configs, dict) else None
    if not isinstance(config, dict):
        raise _McpAuthRequired
    if (str(config.get("url") or "").rstrip("/") != _MCP_URL
            or config.get("auth") != "oauth"
            or config.get("trust") != "untrusted"
            or config.get("headers")):
        raise _ComposioUnavailable("unexpected MCP configuration")
    from tools.mcp_tool_common import mcp_server_enabled

    if not mcp_server_enabled(config):
        raise _ComposioUnavailable("Composio MCP server is disabled")
    try:
        has_token = mcp_config._oauth_tokens_present(_MCP_SERVER)
    except Exception:  # health: allow BLE001 -- OAuth-store errors are sanitized at this provider boundary
        raise _ComposioUnavailable("native MCP OAuth state is unavailable") from None
    if not has_token:
        raise _McpAuthRequired

    from tools import mcp_tool_discovery

    try:
        server = mcp_tool_discovery._get_connected_server_for_call(_MCP_SERVER)
    except Exception:  # health: allow BLE001 -- transport failures must not leak OAuth or endpoint details
        raise _ComposioUnavailable("native MCP connection is unavailable") from None
    if server is None or getattr(server, "session", None) is None:
        raise _ComposioUnavailable("MCP server is not connected")
    live_config = getattr(server, "_config", None)
    if not isinstance(live_config, dict) or not live_config or (
        str(live_config.get("url") or "").rstrip("/") != _MCP_URL
        or live_config.get("auth") != "oauth"
        or live_config.get("trust") != "untrusted"
        or live_config.get("headers")
    ):
        raise _ComposioUnavailable("unexpected live MCP configuration")
    return server


def _schema_for(server, name: str) -> dict:
    for tool in getattr(server, "_tools", ()) or ():
        if getattr(tool, "name", None) != name:
            continue
        schema = getattr(tool, "input_schema", None)
        if schema is None:
            schema = getattr(tool, "inputSchema", None)
        if isinstance(schema, dict):
            return schema
        break
    raise _ComposioUnavailable("required Composio tool schema is unavailable")


def _validate_tool_args(server, name: str, arguments: dict) -> None:
    """Validate our fixed, allowlisted meta-tool calls against the live server schema."""
    schema = _schema_for(server, name)
    try:
        from jsonschema.validators import validator_for
        from jsonschema.exceptions import SchemaError

        validator = validator_for(schema)
        validator.check_schema(schema)
        errors = list(validator(schema).iter_errors(arguments))
    except (ImportError, SchemaError, TypeError, ValueError) as exc:
        raise _ComposioUnavailable("Composio tool schema could not be validated") from exc
    if errors:
        raise _ComposioUnavailable("Composio tool schema did not accept the account request")


def _decode_text(value: str) -> dict | None:
    import json

    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _unwrap_payload(value: dict, depth: int = 0) -> dict | None:
    """Peel only the documented MCP result/data wrappers, never inspect arbitrary fields."""
    if depth > 6:
        return None
    if (value.get("isError") or value.get("is_error") or value.get("successful") is False
            or value.get("error")):
        raise _ComposioUnavailable("Composio rejected the account request")
    if any(key in value for key in (
        "successful", "error", "results", "toolkit_connection_statuses",
        "tool_schemas", "session", "session_id",
    )):
        return value
    for key in ("structuredContent", "structured_content", "result", "data"):
        child = value.get(key)
        if isinstance(child, dict):
            decoded = _unwrap_payload(child, depth + 1)
            if decoded is not None:
                return decoded
        elif isinstance(child, str):
            parsed = _decode_text(child)
            if parsed is not None:
                decoded = _unwrap_payload(parsed, depth + 1)
                if decoded is not None:
                    return decoded
    return None


def _result_payload(result: Any) -> dict:
    if isinstance(result, dict):
        if result.get("isError") or result.get("is_error"):
            raise _ComposioUnavailable("Composio rejected the account request")
        for key in ("structuredContent", "structured_content"):
            value = result.get(key)
            if isinstance(value, dict):
                unwrapped = _unwrap_payload(value)
                if unwrapped is not None:
                    return unwrapped
        unwrapped = _unwrap_payload(result)
        if unwrapped is not None:
            return unwrapped
        content = result.get("content")
    else:
        if getattr(result, "isError", False):
            raise _ComposioUnavailable("Composio rejected the account request")
        structured = getattr(result, "structuredContent", None)
        if not isinstance(structured, dict):
            structured = getattr(result, "structured_content", None)
        if isinstance(structured, dict):
            unwrapped = _unwrap_payload(structured)
            if unwrapped is not None:
                return unwrapped
        content = getattr(result, "content", None)

    for block in content or ():
        text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
        decoded = _decode_text(text) if isinstance(text, str) else None
        if decoded is not None:
            if decoded.get("isError") or decoded.get("is_error"):
                raise _ComposioUnavailable("Composio rejected the account request")
            unwrapped = _unwrap_payload(decoded)
            if unwrapped is not None:
                return unwrapped
    raise _ComposioUnavailable("Composio returned an unreadable account response")


def _session_id(payload: dict) -> str:
    pending = [payload]
    for _ in range(7):
        if not pending:
            break
        source = pending.pop()
        if not isinstance(source, dict):
            continue
        session = source.get("session")
        if isinstance(session, dict) and isinstance(session.get("id"), str) and session["id"]:
            return session["id"]
        session_id = source.get("session_id")
        if isinstance(session_id, str) and session_id:
            return session_id
        pending.extend(source[key] for key in ("data", "result") if isinstance(source.get(key), dict))
    raise _ComposioUnavailable("Composio did not return a discovery session")


def _call_tool(server, name: str, arguments: dict) -> dict:
    import contextvars

    _validate_tool_args(server, name, arguments)
    from tools import mcp_tool_loop

    async def _call():
        lock = getattr(server, "_rpc_lock", None)
        if lock is None:
            result = await server.session.call_tool(name, arguments=arguments)
        else:
            async with lock:
                server._pending_call_context = contextvars.copy_context()
                try:
                    result = await server.session.call_tool(name, arguments=arguments)
                finally:
                    server._pending_call_context = None
        mark_proven = getattr(server, "_mark_session_proven", None)
        if callable(mark_proven):
            mark_proven()
        return result

    try:
        raw = mcp_tool_loop._run_on_mcp_loop(_call, timeout=_MCP_TIMEOUT)
    except Exception:  # health: allow BLE001 -- sanitize external MCP errors so credentials never reach the RPC or logs
        raise _ComposioUnavailable("Composio account request failed") from None
    return _result_payload(raw)


def _search(server, rows: list[tuple[str, ProviderSpec]]) -> tuple[str, dict[str, bool]]:
    arguments = {
        "queries": [{"use_case": spec.query} for _provider, spec in rows],
        "search_strategy": "tool_search",
        "session": {"generate_id": True},
    }
    payload = _call_tool(server, _SEARCH_TOOL, arguments)
    session_id = _session_id(payload)
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    candidates = data.get("toolkit_connection_statuses", []) if isinstance(data, dict) else []
    toolkits = {
        str(candidate.get("toolkit") or "")
        for candidate in candidates
        if isinstance(candidate, dict)
    }
    return session_id, {provider: spec.toolkit in toolkits for provider, spec in rows}


def _manage(server, session_id: str, toolkit_actions: list[dict]) -> dict:
    arguments = {"session_id": session_id, "toolkits": toolkit_actions}
    payload = _call_tool(server, _MANAGE_TOOL, arguments)
    if payload.get("successful") is False or payload.get("error"):
        raise _ComposioUnavailable("Composio could not manage the selected account")
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    results = data.get("results") if isinstance(data, dict) else None
    if isinstance(results, dict):
        for action in toolkit_actions:
            result = results.get(action.get("name"))
            if isinstance(result, dict) and (
                result.get("successful") is False or result.get("error")
            ):
                raise _ComposioUnavailable("Composio could not manage the selected account")
    return payload


def _accounts_from(payload: dict, toolkit: str, *, required: bool = False) -> list:
    from tui_gateway.contracts.composio import ComposioAccountRow

    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    results = data.get("results") if isinstance(data, dict) else None
    provider_result = results.get(toolkit) if isinstance(results, dict) else None
    raw_accounts = provider_result.get("accounts") if isinstance(provider_result, dict) else None
    if not isinstance(raw_accounts, list):
        if required:
            raise _ComposioUnavailable("Composio did not return account details")
        return []
    accounts: list[ComposioAccountRow] = []
    for raw in raw_accounts:
        if not isinstance(raw, dict):
            continue
        account_id = raw.get("id")
        if not isinstance(account_id, str) or not account_id:
            continue
        alias = raw.get("alias")
        account_type = raw.get("account_type")
        accounts.append(ComposioAccountRow(
            id=account_id,
            alias=alias if isinstance(alias, str) else None,
            status=_normalize_status(raw.get("status")),
            is_default=raw.get("is_default") is True,
            account_type=account_type if isinstance(account_type, str) else None,
        ))
    return accounts


def _find_redirect_url(payload: dict) -> str:
    from urllib.parse import urlparse

    pending = [payload]
    for _ in range(8):
        if not pending:
            break
        current = pending.pop()
        if isinstance(current, dict):
            for key in ("redirect_url", "auth_url", "connect_url"):
                value = current.get(key)
                if isinstance(value, str) and value:
                    parsed = urlparse(value)
                    if parsed.scheme == "https" and parsed.hostname == "connect.composio.dev":
                        return value
            pending.extend(value for value in current.values() if isinstance(value, (dict, list)))
        elif isinstance(current, list):
            pending.extend(value for value in current if isinstance(value, (dict, list)))
    raise _ComposioUnavailable("Composio did not return a safe authorization link")


@method("composio.catalog")
@_profile_scoped
def composio_catalog(rid, _params):
    from tui_gateway.contracts.composio import ComposioCatalogResult, ComposioProviderRow

    result = ComposioCatalogResult(providers=[
        ComposioProviderRow(provider=provider, label=spec.label, toolkit=spec.toolkit)
        for provider, spec in PROVIDERS.items()
    ])
    return _ok(rid, result.model_dump(mode="json"))


@method("composio.status")
@_profile_scoped
def composio_status(rid, params):
    from tui_gateway.contracts.composio import (
        ComposioProviderState,
        ComposioProviderStatus,
        ComposioStatusParams,
        ComposioStatusResult,
    )

    try:
        request = ComposioStatusParams.model_validate(params)
        rows = _provider_rows(request.provider)
    except (ValueError, _ComposioUnavailable):
        return _err(rid, 4000, "Select a supported Composio provider.")
    try:
        server = _verified_server()
    except _McpAuthRequired:
        return _ok(rid, ComposioStatusResult(providers=[
            ComposioProviderState(provider=provider, toolkit=spec.toolkit,
                                  status=ComposioProviderStatus.requires_mcp_auth)
            for provider, spec in rows
        ]).model_dump(mode="json"))
    except _ComposioUnavailable as exc:
        return _safe_error(rid, str(exc))
    try:
        states = []
        for batch in _chunks(rows):
            session_id, found = _search(server, batch)
            searchable = [(provider, spec) for provider, spec in batch if found.get(provider)]
            listings = _manage(server, session_id, [
                {"name": spec.toolkit, "action": "list"} for _provider, spec in searchable
            ]) if searchable else {}
            for provider, spec in batch:
                if not found.get(provider):
                    states.append(ComposioProviderState(
                        provider=provider, toolkit=spec.toolkit, status=ComposioProviderStatus.unavailable))
                    continue
                accounts = _accounts_from(listings, spec.toolkit, required=True)
                states.append(ComposioProviderState(
                    provider=provider, toolkit=spec.toolkit,
                    status=_provider_state(provider, spec, accounts), accounts=accounts))
        return _ok(rid, ComposioStatusResult(providers=states).model_dump(mode="json"))
    except _ComposioUnavailable as exc:
        return _safe_error(rid, str(exc))


@method("composio.connect")
@_profile_scoped
def composio_connect(rid, params):
    from tui_gateway.contracts.composio import (
        ComposioConnectParams,
        ComposioConnectResult,
        ComposioProviderStatus,
    )

    try:
        request = ComposioConnectParams.model_validate(params)
        provider, spec = _provider_rows(request.provider)[0]
    except (ValueError, _ComposioUnavailable):
        return _err(rid, 4000, "Select a supported Composio provider.")
    try:
        server = _verified_server()
    except _McpAuthRequired:
        return _ok(rid, ComposioConnectResult(
            provider=provider, toolkit=spec.toolkit,
            status=ComposioProviderStatus.requires_mcp_auth).model_dump(mode="json"))
    except _ComposioUnavailable as exc:
        return _safe_error(rid, str(exc))
    try:
        session_id, found = _search(server, [(provider, spec)])
        if not found.get(provider):
            return _ok(rid, ComposioConnectResult(
                provider=provider, toolkit=spec.toolkit,
                status=ComposioProviderStatus.unavailable).model_dump(mode="json"))
        item: dict[str, Any] = {"name": spec.toolkit, "action": "add"}
        if request.alias:
            item["alias"] = request.alias.strip()
        payload = _manage(server, session_id, [item])
        auth_url = _find_redirect_url(payload)
        accounts = _accounts_from(payload, spec.toolkit)
        return _ok(rid, ComposioConnectResult(
            provider=provider, toolkit=spec.toolkit, status=ComposioProviderStatus.pending,
            auth_url=auth_url, accounts=accounts).model_dump(mode="json"))
    except _ComposioUnavailable as exc:
        return _safe_error(rid, str(exc))


@method("composio.remove")
@_profile_scoped
def composio_remove(rid, params):
    from tui_gateway.contracts.composio import ComposioRemoveParams, ComposioRemoveResult

    try:
        request = ComposioRemoveParams.model_validate(params)
        provider, spec = _provider_rows(request.provider)[0]
    except (ValueError, _ComposioUnavailable):
        return _err(rid, 4000, "Select a supported Composio provider and account.")
    try:
        server = _verified_server()
    except _McpAuthRequired:
        return _ok(rid, ComposioRemoveResult(
            provider=provider, toolkit=spec.toolkit, account_id=request.account_id,
            status="requires_mcp_auth").model_dump(mode="json"))
    except _ComposioUnavailable as exc:
        return _safe_error(rid, str(exc))
    try:
        session_id, found = _search(server, [(provider, spec)])
        if not found.get(provider):
            return _ok(rid, ComposioRemoveResult(
                provider=provider, toolkit=spec.toolkit, account_id=request.account_id,
                status="not_found").model_dump(mode="json"))
        listing = _manage(server, session_id, [{"name": spec.toolkit, "action": "list"}])
        accounts = _accounts_from(listing, spec.toolkit, required=True)
        if not any(account.id == request.account_id for account in accounts):
            return _ok(rid, ComposioRemoveResult(
                provider=provider, toolkit=spec.toolkit, account_id=request.account_id,
                status="not_found").model_dump(mode="json"))
        _manage(server, session_id, [{
            "name": spec.toolkit, "action": "remove", "account_id": request.account_id,
        }])
        remaining = _manage(server, session_id, [{"name": spec.toolkit, "action": "list"}])
        if any(account.id == request.account_id for account in _accounts_from(
            remaining, spec.toolkit, required=True
        )):
            raise _ComposioUnavailable("Composio did not remove the selected account")
        return _ok(rid, ComposioRemoveResult(
            provider=provider, toolkit=spec.toolkit, account_id=request.account_id,
            status="removed").model_dump(mode="json"))
    except _ComposioUnavailable as exc:
        return _safe_error(rid, str(exc))


def register(server) -> None:
    bind_module(globals(), server)

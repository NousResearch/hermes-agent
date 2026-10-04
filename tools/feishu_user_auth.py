"""Feishu / Lark **user** OAuth (``user_access_token``) — acquisition, rotation, storage, resolution.

The bot credentials the adapter already holds mint a ``tenant_access_token``, which only reaches
messages addressed to the Hermes bot itself. A ``user_access_token`` (UAT) is what the user-scoped
Open APIs require: cross-chat full-text search (``POST /open-apis/im/v1/messages/search``, scope
``search:message``) accepts a UAT and *nothing else*, and listing a chat's history as the user rather
than as the bot needs ``im:message.p2p_msg:get_as_user`` / ``im:message.group_msg:get_as_user``.

Shared plumbing, not a tool module: like ``tools/feishu_lark.py`` this registers nothing, so tool
discovery's AST scan never imports it. ``tools/feishu_user_tool.py`` uses it for the two user-scoped
tools, ``plugins/platforms/feishu/feishu_cli.py`` for ``hermes feishu login|status|logout``.

**Authorization code + PKCE over an RFC 8252 loopback redirect, not RFC 8628.** Feishu publishes no
device-authorization grant for user tokens; its only device-code endpoint is
``accounts.feishu.cn/oauth/v1/app/registration``, which mints an *app*, and the adapter already
drives that one (``qr_register``). The redirect URI must be allow-listed in the app's console —
Feishu refuses an unregistered one — so the loopback port is a fixed default, not OS-assigned.

**Refresh rotates.** Every refresh returns a new ``refresh_token`` and invalidates the old one
immediately, so a refresh runs inside the auth-store lock and the rotated pair is persisted before
the lock is released; a second reader either waits or reads the new tokens, never the dead ones.
"""

from __future__ import annotations

import contextlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode, urlparse

logger = logging.getLogger(__name__)

# Provider id under ``auth.json`` → ``providers``. Deliberately not "feishu": that name belongs to
# the bot app's own credentials in ``.env``, and this row is a *user* grant with different authority.
PROVIDER_ID = "feishu-user"

# Host pairs per domain. ``accounts.*`` serves the consent page, ``open.*`` the token + Open APIs.
# The adapter's QR onboarding imports these as its ``_ONBOARD_*`` maps, so the two flows can never
# drift onto different Lark hosts.
ACCOUNTS_BASE_URLS: Dict[str, str] = {
    "feishu": "https://accounts.feishu.cn",
    "lark": "https://accounts.larksuite.com",
}
OPEN_BASE_URLS: Dict[str, str] = {
    "feishu": "https://open.feishu.cn",
    "lark": "https://open.larksuite.com",
}

AUTHORIZE_PATH = "/open-apis/authen/v1/authorize"
TOKEN_PATH = "/open-apis/authen/v2/oauth/token"

# Fixed port: the user pastes this exact URI into the app console's redirect allow-list, so an
# OS-assigned port would break every login after the first. 43829 neighbours Spotify's 43827.
DEFAULT_REDIRECT_URI = "http://127.0.0.1:43829/feishu/callback"

# Exactly what the two shipped tools need, nothing speculative. ``offline_access`` is what makes
# Feishu return a refresh_token at all — without it the grant dies in two hours.
DEFAULT_SCOPES: Tuple[str, ...] = (
    "offline_access",
    "search:message",
    "im:message:readonly",
    "im:message.p2p_msg:get_as_user",
    "im:message.group_msg:get_as_user",
)

# Refresh this far before the stated expiry so a long tool call cannot straddle it.
ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120

DOCS_URL = "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/feishu"


def _err(message: str, code: Optional[str] = None, *, relogin: bool = False):
    """A ``feishu-user``-tagged :class:`hermes_cli.auth_constants.AuthError`."""
    from hermes_cli.auth_constants import _provider_error_factory
    return _provider_error_factory(PROVIDER_ID)(message, code, relogin=relogin)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _secret(name: str, default: str = "") -> str:
    """Profile-scoped credential read; an unscoped multiplex read falls back to this profile's env."""
    from gateway.platforms._shared import get_scoped_secret
    return _clean(get_scoped_secret(name, default))


# ---- endpoints ------------------------------------------------------------------


def normalize_domain(domain: Any) -> str:
    """"lark" or "feishu" — an unknown value is Feishu, matching the adapter's own fallback."""
    return "lark" if _clean(domain).lower() == "lark" else "feishu"


def accounts_base_url(domain: Any) -> str:
    return ACCOUNTS_BASE_URLS[normalize_domain(domain)]


def open_base_url(domain: Any) -> str:
    return OPEN_BASE_URLS[normalize_domain(domain)]


def resolve_domain(state: Optional[Dict[str, Any]] = None) -> str:
    """The domain a stored grant was issued against, else the adapter's configured domain."""
    stored = _clean((state or {}).get("domain"))
    return normalize_domain(stored or _secret("FEISHU_DOMAIN", "feishu"))


def app_credentials() -> Tuple[str, str]:
    """``(app_id, app_secret)`` of the configured bot app — the OAuth client for the user grant.

    Feishu's token endpoint is confidential-client only (``client_secret`` is required even with
    PKCE), so there is no separate credential for the user flow to prompt for.
    """
    app_id = _secret("FEISHU_APP_ID")
    app_secret = _secret("FEISHU_APP_SECRET")
    if not app_id or not app_secret:
        raise _err(
            "Feishu app credentials are missing. Run `hermes setup` and configure Feishu / Lark "
            "before signing in as a user.", "feishu_user_app_credentials_missing")
    return app_id, app_secret


def scope_string(raw: Any = None) -> str:
    """Whitespace-normalized, de-duplicated scope string (order kept); empty input → defaults."""
    tokens = _clean(raw).split() or list(DEFAULT_SCOPES)
    return " ".join(dict.fromkeys(tokens))


def validate_redirect_uri(redirect_uri: str) -> Tuple[str, int, str]:
    """``(host, port, path)`` for a loopback redirect URI, or raise."""
    parsed = urlparse(redirect_uri)
    host = parsed.hostname or ""
    problem = (
        "must use http://127.0.0.1 or http://localhost." if parsed.scheme != "http"
        else "must point to 127.0.0.1 or localhost." if host not in {"127.0.0.1", "localhost"}
        else "must include an explicit port so it can be allow-listed in the Feishu app console."
        if not parsed.port else None)
    if problem:
        raise _err(f"Feishu redirect_uri {problem}", "feishu_user_redirect_invalid")
    return host, parsed.port, parsed.path or "/"


def build_authorize_url(
    *, client_id: str, redirect_uri: str, scope: str, state: str, code_challenge: str, domain: Any,
) -> str:
    query = urlencode({
        "client_id": client_id, "response_type": "code", "redirect_uri": redirect_uri,
        "scope": scope, "state": state, "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    })
    return f"{accounts_base_url(domain)}{AUTHORIZE_PATH}?{query}"


# ---- token endpoint -------------------------------------------------------------


def token_post(
    domain: Any, data: Dict[str, str], *, timeout_seconds: float, what: str, failed_code: str,
    invalid_code: str, relogin_required: bool = False, failed_suffix: str = "",
) -> Dict[str, Any]:
    """POST form data to ``authen/v2/oauth/token``; return the payload or raise a shaped AuthError.

    Feishu answers a *rejected* grant with HTTP 400 **and** a body carrying ``error`` /
    ``error_description``, so the body is surfaced: "invalid_grant: refresh token expired" is
    actionable where a bare "HTTP 400" is not.
    """
    import httpx
    url = f"{open_base_url(domain)}{TOKEN_PATH}"
    try:
        response = httpx.post(
            url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=timeout_seconds)
    except Exception as exc:
        raise _err(f"Feishu {what} failed: {exc}", failed_code) from exc

    try:
        payload = response.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    if response.status_code >= 400 or not _clean(payload.get("access_token")):
        detail = _clean(payload.get("error_description")) or _clean(payload.get("error")) \
            or _clean(payload.get("msg")) or response.text.strip()
        raise _err(
            f"Feishu {what} failed.{failed_suffix}" + (f" Response: {detail}" if detail else ""),
            failed_code if response.status_code >= 400 else invalid_code,
            relogin=relogin_required)
    return payload


def state_from_payload(
    payload: Dict[str, Any], *, client_id: str, domain: str, redirect_uri: str, requested_scope: str,
    previous_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The persisted ``providers.feishu-user`` row for one token response.

    The rotated ``refresh_token`` replaces the old one; a response that omits it (the grant was
    taken without ``offline_access``) keeps whatever was stored rather than blanking the field.
    """
    from hermes_cli.auth import _coerce_ttl_seconds
    now = datetime.now(timezone.utc)
    expires_in = _coerce_ttl_seconds(payload.get("expires_in", 0))
    state = dict(previous_state or {})
    state.update({
        "client_id": client_id, "domain": normalize_domain(domain), "redirect_uri": redirect_uri,
        "scope": requested_scope,
        "granted_scope": _clean(payload.get("scope")) or requested_scope,
        "token_type": _clean(payload.get("token_type")) or "Bearer",
        "access_token": _clean(payload.get("access_token")),
        "refresh_token": _clean(payload.get("refresh_token")) or _clean(state.get("refresh_token")),
        "obtained_at": now.isoformat(),
        "expires_at": datetime.fromtimestamp(now.timestamp() + expires_in, tz=timezone.utc).isoformat(),
        "expires_in": expires_in, "auth_type": "oauth_authorization_code_pkce",
    })
    state.pop("last_auth_error", None)  # a fresh grant clears a quarantine marker
    return state


def exchange_code(
    *, code: str, code_verifier: str, client_id: str, client_secret: str, redirect_uri: str,
    domain: str, requested_scope: str, timeout_seconds: float = 20.0,
) -> Dict[str, Any]:
    payload = token_post(
        domain,
        {
            "grant_type": "authorization_code", "client_id": client_id,
            "client_secret": client_secret, "code": code, "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        },
        timeout_seconds=timeout_seconds, what="token exchange",
        failed_code="feishu_user_token_exchange_failed",
        invalid_code="feishu_user_token_exchange_invalid")
    return state_from_payload(
        payload, client_id=client_id, domain=domain, redirect_uri=redirect_uri,
        requested_scope=requested_scope)


def refresh_state(state: Dict[str, Any], *, timeout_seconds: float = 20.0) -> Dict[str, Any]:
    """Exchange the stored refresh token for a fresh pair. Caller holds the auth-store lock."""
    refresh_token = _clean(state.get("refresh_token"))
    if not refresh_token:
        raise _err(
            "Feishu user grant has no refresh token. Run `hermes feishu login` again.",
            "feishu_user_refresh_token_missing", relogin=True)
    client_id, client_secret = app_credentials()
    domain = resolve_domain(state)
    payload = token_post(
        domain,
        {
            "grant_type": "refresh_token", "client_id": client_id, "client_secret": client_secret,
            "refresh_token": refresh_token,
        },
        timeout_seconds=timeout_seconds, what="token refresh",
        failed_code="feishu_user_refresh_failed", invalid_code="feishu_user_refresh_invalid",
        relogin_required=True, failed_suffix=" Run `hermes feishu login` again.")
    return state_from_payload(
        payload, client_id=client_id, domain=domain,
        redirect_uri=_clean(state.get("redirect_uri")) or DEFAULT_REDIRECT_URI,
        requested_scope=_clean(state.get("scope")) or scope_string(),
        previous_state=state)


# ---- storage --------------------------------------------------------------------


def load_state() -> Optional[Dict[str, Any]]:
    """The stored grant, or None. Shares ``auth.json`` with the rest of hermes-agent."""
    from hermes_cli.auth import _load_auth_store, _load_provider_state
    return _load_provider_state(_load_auth_store(), PROVIDER_ID)


def save_state(state: Dict[str, Any]) -> "Any":
    """Persist the grant under ``providers.feishu-user``; returns the auth-store path.

    ``set_active=False`` always: this is a Feishu user grant, never an inference provider.
    """
    from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store, _store_provider_state
    with _auth_store_lock():
        auth_store = _load_auth_store()
        _store_provider_state(auth_store, PROVIDER_ID, state, set_active=False)
        return _save_auth_store(auth_store)


def clear_state() -> bool:
    """Drop the stored grant. True when something was removed."""
    from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store
    with _auth_store_lock():
        auth_store = _load_auth_store()
        providers = auth_store.get("providers")
        if not isinstance(providers, dict) or PROVIDER_ID not in providers:
            return False
        providers.pop(PROVIDER_ID, None)
        _save_auth_store(auth_store)
        return True


def has_user_token() -> bool:
    """``check_fn`` for the user-scoped tools: a grant with usable material exists.

    Reachability, not session surface — the answer is a property of the profile's ``auth.json``,
    which is exactly what ``registry.check_fn_cache_scope()`` keys its cache on.
    """
    try:
        state = load_state() or {}
    except Exception:
        return False
    return bool(_clean(state.get("refresh_token")) or _clean(state.get("access_token")))


# ---- runtime resolution ---------------------------------------------------------


def resolve_user_access_token(
    *, force_refresh: bool = False, refresh_if_expiring: bool = True,
    refresh_skew_seconds: int = ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
) -> Dict[str, Any]:
    """A live UAT plus the endpoint context to call with it.

    The refresh happens *inside* the auth-store lock: Feishu invalidates the old refresh token the
    moment a new pair is minted, so two concurrent resolvers must not both spend it.
    """
    from hermes_cli.auth import (
        _auth_store_lock, _is_expiring, _load_auth_store, _load_provider_state,
        _quarantine_flat_oauth_state, _save_auth_store, _store_provider_state)
    from hermes_cli.auth_constants import AuthError

    with _auth_store_lock():
        auth_store = _load_auth_store()
        state = _load_provider_state(auth_store, PROVIDER_ID)
        if not state:
            raise _err(
                "Feishu user access is not authorized. Run `hermes feishu login` first.",
                "feishu_user_auth_missing", relogin=True)

        should_refresh = bool(force_refresh)
        if not should_refresh and refresh_if_expiring:
            should_refresh = _is_expiring(state.get("expires_at"), refresh_skew_seconds)
        if should_refresh:
            try:
                state = refresh_state(state)
                _store_provider_state(auth_store, PROVIDER_ID, state, set_active=False)
                _save_auth_store(auth_store)
            except AuthError as exc:
                # Terminal refusal: strip the dead tokens so the next call fails fast instead of
                # spending a round-trip on a grant Feishu has already revoked.
                if exc.relogin_required and state.get("refresh_token"):
                    _quarantine_flat_oauth_state(state, PROVIDER_ID, exc)
                    try:
                        _store_provider_state(auth_store, PROVIDER_ID, state, set_active=False)
                        _save_auth_store(auth_store)
                    except Exception as save_exc:
                        logger.debug("feishu-user: could not persist quarantined state: %s", save_exc)
                raise

    access_token = _clean(state.get("access_token"))
    if not access_token:
        raise _err(
            "Feishu user access token missing. Run `hermes feishu login` again.",
            "feishu_user_access_token_missing", relogin=True)
    domain = resolve_domain(state)
    return {
        "access_token": access_token,
        "token_type": _clean(state.get("token_type")) or "Bearer",
        "domain": domain, "base_url": open_base_url(domain),
        "scope": _clean(state.get("granted_scope") or state.get("scope")),
        "expires_at": state.get("expires_at"),
    }


def auth_status() -> Dict[str, Any]:
    """Report shape for ``hermes feishu status`` — never includes token material."""
    from hermes_cli.auth import _is_expiring
    try:
        state = load_state() or {}
    except Exception as exc:
        return {"logged_in": False, "error": str(exc)}
    if not state:
        return {"logged_in": False}
    refresh_token = _clean(state.get("refresh_token"))
    last_error = state.get("last_auth_error") if isinstance(state.get("last_auth_error"), dict) else None
    return {
        "logged_in": bool(refresh_token or not _is_expiring(state.get("expires_at"), 0)),
        "auth_type": state.get("auth_type"), "client_id": state.get("client_id"),
        "domain": resolve_domain(state), "redirect_uri": state.get("redirect_uri"),
        "scope": state.get("granted_scope") or state.get("scope"),
        "expires_at": state.get("expires_at"), "has_refresh_token": bool(refresh_token),
        "error": (last_error or {}).get("message"),
    }


# ---- interactive login ----------------------------------------------------------


def login(
    *, redirect_uri: Optional[str] = None, scope: Optional[str] = None, open_browser: bool = True,
    timeout_seconds: float = 180.0,
) -> Dict[str, Any]:
    """Run the consent + code-exchange flow and persist the grant. Returns the stored state.

    Prints the authorize URL, the redirect URI to allow-list, and (on a remote host) the SSH
    forward needed for the callback to arrive.
    """
    import webbrowser
    from hermes_cli.auth_device_flow import (
        _bind_loopback_callback_server, _can_open_graphical_browser, _is_remote_session,
        _make_loopback_callback_handler, _pkce_code_challenge, _pkce_code_verifier,
        _print_loopback_ssh_hint, _serve_loopback_callback)

    client_id, client_secret = app_credentials()
    previous = load_state() or {}
    domain = resolve_domain(previous)
    redirect_uri = _clean(redirect_uri) or _clean(previous.get("redirect_uri")) or DEFAULT_REDIRECT_URI
    requested_scope = scope_string(scope or previous.get("scope"))
    host, port, path = validate_redirect_uri(redirect_uri)

    code_verifier = _pkce_code_verifier()
    state_nonce = uuid.uuid4().hex
    authorize_url = build_authorize_url(
        client_id=client_id, redirect_uri=redirect_uri, scope=requested_scope, state=state_nonce,
        code_challenge=_pkce_code_challenge(code_verifier), domain=domain)

    print(
        f"Authorizing Hermes to act as your Feishu / Lark user.\n"
        f"  App ID:       {client_id}\n  Domain:       {domain}\n"
        f"  Redirect URI: {redirect_uri}\n  Scopes:       {requested_scope}\n\n"
        f"Add that exact redirect URI under Security Settings → Redirect URLs in the app console,\n"
        f"then open this URL to grant access:\n\n{authorize_url}\n\nSetup guide: {DOCS_URL}\n")
    _print_loopback_ssh_hint(redirect_uri, docs_url=DOCS_URL)

    handler_cls, result = _make_loopback_callback_handler(path, display_name="Feishu / Lark")
    # Bind before opening the browser so the redirect URI names a port we already own.
    server = _bind_loopback_callback_server(
        host, port, handler_cls, err=_err, bind_failed_code="feishu_user_callback_bind_failed")
    try:
        if open_browser and not _is_remote_session() and _can_open_graphical_browser():
            try:
                opened = webbrowser.open(authorize_url)
            except Exception:
                opened = False
            print("Browser opened for authorization." if opened
                  else "Could not open the browser automatically; use the URL above.")
        callback = _serve_loopback_callback(
            server, result, timeout_seconds=timeout_seconds, err=_err,
            timeout_code="feishu_user_callback_timeout")
    except BaseException:
        # A Ctrl-C between bind and serve would otherwise hold the port for the process lifetime;
        # ``_serve_loopback_callback`` closes it on every other path and a second close is a no-op.
        with contextlib.suppress(Exception):
            server.server_close()
        raise
    if callback.get("error"):
        raise _err(
            "Feishu authorization failed: "
            f"{_clean(callback.get('error_description')) or _clean(callback.get('error'))}",
            "feishu_user_authorization_denied")
    # Feishu echoes ``state`` verbatim; a mismatch means the redirect did not come from our request.
    if _clean(callback.get("state")) != state_nonce:
        raise _err("Feishu authorization failed: state mismatch.", "feishu_user_state_mismatch")

    state = exchange_code(
        code=_clean(callback.get("code")), code_verifier=code_verifier, client_id=client_id,
        client_secret=client_secret, redirect_uri=redirect_uri, domain=domain,
        requested_scope=requested_scope)
    save_state(state)
    return state

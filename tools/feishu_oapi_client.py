"""Feishu OAPI unified client — dual-identity (TAT + UAT) factory.

Provides ``FeishuClient`` with three factory methods:
  - ``for_tenant()``     — build a TAT (tenant_access_token) lark Client
  - ``for_user()``       — build a lark Client + RequestOption carrying UAT
  - ``from_credentials(app_id, app_secret)`` — ephemeral, uncached

Four semantic error classes surface auth failures to callers:
  - ``NeedAuthorizationError``      — no UAT on disk, need device flow
  - ``AppScopeMissingError``        — app missing API scope (errcode 99991672)
  - ``UserAuthRequiredError``       — user not authorized (errcode 99991679)
  - ``UserScopeInsufficientError``  — token valid but scope insufficient

Token management:
  - ``_load_uat()`` reads ~/.hermes/feishu_uat.json; if access_token expires
    within 60 s it raises NeedAuthorizationError (no auto-refresh daemon).
  - TOOLS_METADATA is a registry for Phase 2 workers to declare per-tool
    scopes and preferred identity.  Starts empty; workers append entries.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEISHU_UAT_PATH = get_hermes_home() / "feishu_uat.json"

# Access token refresh headroom — treat token as expired this many seconds early
_ACCESS_TOKEN_EXPIRY_HEADROOM_S = 60

# Feishu error codes that map to semantic auth errors
_ERRCODE_APP_SCOPE_MISSING = 99991672
_ERRCODE_USER_SCOPE_INSUFFICIENT = 99991679
_ERRCODE_TOKEN_INVALID = 99991668
_ERRCODE_TOKEN_EXPIRED = 99991677

# Registry for Phase 2 tool families to declare per-tool identity preference.
# Key: tool name (e.g. "feishu_calendar_list_events")
# Value: dict with optional keys:
#   "identity": "user" | "tenant"  (default "user")
#   "scopes": list[str]
TOOLS_METADATA: dict[str, dict] = {
    "feishu_get_my_user_info": {
        "identity": "user",
        "scopes": ["authen:user.employee_id:read"],
    },
}


# ---------------------------------------------------------------------------
# Semantic error classes
# ---------------------------------------------------------------------------

class NeedAuthorizationError(Exception):
    """No valid UAT on disk — user must run 'hermes feishu-uat' device flow.

    Args:
        user_open_id: Open ID of the user who needs authorization, if known.
        reason: Human-readable explanation of why authorization is needed.
    """

    def __init__(self, user_open_id: str = "unknown", reason: str = "") -> None:
        msg = f"need_user_authorization: {user_open_id}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)
        self.user_open_id = user_open_id
        self.reason = reason


class AppScopeMissingError(Exception):
    """App (bot) is missing an API scope — admin must enable it in Feishu console.

    Triggered by Feishu errcode 99991672.

    Args:
        app_id: The Feishu app ID.
        api_name: The API or tool name that triggered the error.
        missing_scopes: List of scope strings that are absent.
    """

    def __init__(
        self, app_id: str, api_name: str, missing_scopes: list[str]
    ) -> None:
        scopes_str = ", ".join(missing_scopes)
        super().__init__(
            f"App '{app_id}' missing scopes [{scopes_str}] for API '{api_name}'. "
            "Admin must enable in Feishu console."
        )
        self.app_id = app_id
        self.api_name = api_name
        self.missing_scopes = missing_scopes


class UserAuthRequiredError(Exception):
    """User has not authorized the app or required scopes are insufficient.

    Triggered by Feishu errcode 99991679.

    Args:
        user_open_id: Open ID of the user.
        api_name: The API or tool name that triggered the error.
        required_scopes: Scopes that are required.
        app_id: The Feishu app ID.
    """

    def __init__(
        self,
        user_open_id: str,
        api_name: str,
        required_scopes: list[str],
        app_id: str = "",
    ) -> None:
        scopes_str = ", ".join(required_scopes)
        super().__init__(
            f"User '{user_open_id}' missing scopes [{scopes_str}] for '{api_name}'"
        )
        self.user_open_id = user_open_id
        self.api_name = api_name
        self.required_scopes = required_scopes
        self.app_id = app_id


class UserScopeInsufficientError(Exception):
    """UAT is valid but lacks specific scopes — incremental authorization needed.

    Args:
        user_open_id: Open ID of the user.
        api_name: The API or tool name that triggered the error.
        missing_scopes: Scopes absent from the current token.
    """

    def __init__(
        self, user_open_id: str, api_name: str, missing_scopes: list[str]
    ) -> None:
        scopes_str = ", ".join(missing_scopes)
        super().__init__(
            f"User '{user_open_id}' insufficient scopes [{scopes_str}] "
            f"for '{api_name}'. Re-run 'hermes feishu-uat' to re-authorize."
        )
        self.user_open_id = user_open_id
        self.api_name = api_name
        self.missing_scopes = missing_scopes


# ---------------------------------------------------------------------------
# Feishu errcode → semantic error helper
# ---------------------------------------------------------------------------

def raise_for_feishu_errcode(
    code: int,
    msg: str = "",
    *,
    app_id: str = "",
    api_name: str = "",
    user_open_id: str = "unknown",
) -> None:
    """Raise a semantic error if *code* maps to a known auth failure.

    Args:
        code: Feishu API response code (0 = success).
        msg: Response message string for logging.
        app_id: Feishu app ID (for AppScopeMissingError).
        api_name: Name of the tool/API (for error context).
        user_open_id: User open ID (for user-level errors).

    Raises:
        AppScopeMissingError: errcode 99991672.
        UserAuthRequiredError: errcode 99991679.
        NeedAuthorizationError: errcode 99991668 or 99991677 (token invalid/expired).
    """
    if code == _ERRCODE_APP_SCOPE_MISSING:
        raise AppScopeMissingError(app_id, api_name, [msg or "unknown scope"])
    if code == _ERRCODE_USER_SCOPE_INSUFFICIENT:
        raise UserAuthRequiredError(user_open_id, api_name, [], app_id)
    if code in (_ERRCODE_TOKEN_INVALID, _ERRCODE_TOKEN_EXPIRED):
        raise NeedAuthorizationError(
            user_open_id,
            reason=f"token invalid or expired (errcode={code}); re-run 'hermes feishu-uat'",
        )


# ---------------------------------------------------------------------------
# UAT loader
# ---------------------------------------------------------------------------

def _load_uat() -> dict:
    """Load UAT from ~/.hermes/feishu_uat.json and validate freshness.

    Returns:
        Token dict with at least access_token, refresh_token, user_open_id.

    Raises:
        NeedAuthorizationError: File missing, unreadable, or access_token
            expires within _ACCESS_TOKEN_EXPIRY_HEADROOM_S seconds.
    """
    if not FEISHU_UAT_PATH.exists():
        raise NeedAuthorizationError(
            reason="no token file found; run 'hermes feishu-auth' first"
        )

    try:
        with open(FEISHU_UAT_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise NeedAuthorizationError(
            reason=f"token file unreadable: {exc}"
        ) from exc

    access_token = data.get("access_token", "")
    if not access_token:
        raise NeedAuthorizationError(
            reason="token file has no access_token; re-run 'hermes feishu-auth'"
        )

    expires_at_ms = data.get("expires_at", 0)
    now_ms = int(time.time() * 1000)
    headroom_ms = _ACCESS_TOKEN_EXPIRY_HEADROOM_S * 1000

    if now_ms >= expires_at_ms - headroom_ms:
        user_open_id = data.get("user_open_id", "unknown")
        # Attempt automatic refresh before failing
        app_id = os.getenv("FEISHU_APP_ID", "").strip()
        app_secret = os.getenv("FEISHU_APP_SECRET", "").strip()
        if app_id and app_secret:
            try:
                from hermes_cli.feishu_auth import refresh_uat
                refresh_uat(app_id, app_secret)
                logger.info("UAT auto-refreshed for user %s", user_open_id)
                # Re-read the freshly saved token
                with open(FEISHU_UAT_PATH, encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception as exc:
                logger.debug("Auto-refresh failed: %s", exc)
        raise NeedAuthorizationError(
            user_open_id=user_open_id,
            reason="access_token expired or expiring soon; re-run 'hermes feishu-auth'",
        )

    return data


# ---------------------------------------------------------------------------
# FeishuClient
# ---------------------------------------------------------------------------

class FeishuClient:
    """Feishu SDK client wrapper with TAT / UAT dual-identity support.

    Do not instantiate directly — use the class-level factory methods:
      ``FeishuClient.for_tenant()``      — TAT (bot identity)
      ``FeishuClient.for_user()``        — UAT (user identity, loads from disk)
      ``FeishuClient.from_credentials(app_id, app_secret)`` — ephemeral

    After construction the ``sdk`` attribute holds the raw ``lark.Client``.
    For UAT calls, pass ``self.request_option`` as the second argument to
    SDK methods, or inject it into ``BaseRequest`` via RequestOption:
      ``RequestOption.builder().user_access_token(client.access_token).build()``
    """

    _cache: dict[str, "FeishuClient"] = {}

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        *,
        account_id: str = "default",
        domain: str = "feishu",
        access_token: str = "",
        user_open_id: str = "",
        ephemeral: bool = False,
    ) -> None:
        self.app_id = app_id
        self.app_secret = app_secret
        self.account_id = account_id
        self.domain = domain
        self.access_token = access_token
        self.user_open_id = user_open_id
        self.ephemeral = ephemeral
        self.sdk = self._build_sdk()

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def for_tenant(cls) -> "FeishuClient":
        """Create or reuse a TAT (tenant_access_token) lark Client.

        Reads FEISHU_APP_ID and FEISHU_APP_SECRET from the environment.

        Returns:
            FeishuClient configured for tenant identity.

        Raises:
            ValueError: If FEISHU_APP_ID or FEISHU_APP_SECRET are unset.
        """
        app_id = os.getenv("FEISHU_APP_ID", "").strip()
        app_secret = os.getenv("FEISHU_APP_SECRET", "").strip()
        domain = os.getenv("FEISHU_DOMAIN", "feishu").strip().lower()

        if not app_id or not app_secret:
            raise ValueError(
                "FEISHU_APP_ID and FEISHU_APP_SECRET must be set. "
                "Run 'hermes setup' to configure the Feishu bot."
            )

        cache_key = f"tenant:default:{app_id}"
        existing = cls._cache.get(cache_key)
        if existing and existing.app_secret == app_secret:
            return existing

        instance = cls(
            app_id=app_id,
            app_secret=app_secret,
            account_id="default",
            domain=domain,
        )
        cls._cache[cache_key] = instance
        return instance

    @classmethod
    def for_user(cls) -> "FeishuClient":
        """Create a UAT (user_access_token) lark Client from disk storage.

        Loads ~/.hermes/feishu_uat.json, validates token freshness, and builds
        a client.  The ``access_token`` attribute holds the raw UAT string;
        pass it to ``RequestOption.builder().user_access_token(...)`` for SDK
        calls that support USER token type.

        Returns:
            FeishuClient with access_token set.

        Raises:
            NeedAuthorizationError: Token missing, expired, or expiring soon.
            ValueError: If FEISHU_APP_ID / FEISHU_APP_SECRET unset.
        """
        app_id = os.getenv("FEISHU_APP_ID", "").strip()
        app_secret = os.getenv("FEISHU_APP_SECRET", "").strip()
        domain = os.getenv("FEISHU_DOMAIN", "feishu").strip().lower()

        if not app_id or not app_secret:
            raise ValueError(
                "FEISHU_APP_ID and FEISHU_APP_SECRET must be set. "
                "Run 'hermes setup' to configure the Feishu bot."
            )

        uat_data = _load_uat()
        access_token = uat_data["access_token"]
        user_open_id = uat_data.get("user_open_id", "")

        return cls(
            app_id=app_id,
            app_secret=app_secret,
            domain=domain,
            access_token=access_token,
            user_open_id=user_open_id,
            ephemeral=True,
        )

    @classmethod
    def from_credentials(
        cls,
        app_id: str,
        app_secret: str,
        *,
        domain: str = "feishu",
        account_id: str = "ephemeral",
    ) -> "FeishuClient":
        """Create an ephemeral client from explicit credentials (not cached).

        Args:
            app_id: Feishu app ID.
            app_secret: Feishu app secret.
            domain: "feishu" or "lark".
            account_id: Logical account label (for logging only).

        Returns:
            FeishuClient not registered in the instance cache.
        """
        return cls(
            app_id=app_id,
            app_secret=app_secret,
            account_id=account_id,
            domain=domain,
            ephemeral=True,
        )

    # ------------------------------------------------------------------
    # SDK builder
    # ------------------------------------------------------------------

    def _build_sdk(self) -> Any:
        """Build and return the underlying lark.Client instance.

        Returns:
            lark.Client, or None if lark_oapi is not installed.
        """
        try:
            import lark_oapi as lark
            from lark_oapi.core.const import FEISHU_DOMAIN, LARK_DOMAIN
        except ImportError:
            logger.warning("lark_oapi not installed — FeishuClient SDK unavailable")
            return None

        sdk_domain = LARK_DOMAIN if self.domain == "lark" else FEISHU_DOMAIN
        return (
            lark.Client.builder()
            .app_id(self.app_id)
            .app_secret(self.app_secret)
            .domain(sdk_domain)
            .log_level(lark.LogLevel.WARNING)
            .build()
        )

    # ------------------------------------------------------------------
    # RequestOption helper for UAT calls
    # ------------------------------------------------------------------

    def build_user_request_option(self) -> Any:
        """Build a RequestOption injecting this client's UAT.

        Use this as the second argument to SDK service methods that accept
        a RequestOption, e.g.:
          ``client.calendar.v4.event.list(request, client.build_user_request_option())``

        Returns:
            lark_oapi.RequestOption, or None if lark_oapi is unavailable or
            no access_token is set.
        """
        if not self.access_token:
            return None
        try:
            from lark_oapi import RequestOption
            return (
                RequestOption.builder()
                .user_access_token(self.access_token)
                .build()
            )
        except ImportError:
            return None

    # ------------------------------------------------------------------
    # BaseRequest helper for raw HTTP UAT calls
    # ------------------------------------------------------------------

    def do_request(
        self,
        method: str,
        uri: str,
        *,
        paths: Optional[dict] = None,
        queries: Optional[list] = None,
        body: Optional[dict] = None,
        use_uat: bool = False,
    ) -> tuple[int, str, dict]:
        """Execute a BaseRequest and return (code, msg, data_dict).

        Args:
            method: HTTP method string "GET" or "POST".
            uri: Feishu open-api URI, e.g. "/open-apis/calendar/v4/events".
            paths: Path parameter substitutions dict.
            queries: List of (key, value) query parameter tuples.
            body: JSON body dict (POST only).
            use_uat: If True, inject UAT via RequestOption instead of TAT.

        Returns:
            Tuple of (code, msg, data_dict) where code=0 means success.

        Raises:
            RuntimeError: If lark_oapi is not installed.
        """
        try:
            from lark_oapi import AccessTokenType, RequestOption
            from lark_oapi.core.enum import HttpMethod
            from lark_oapi.core.model.base_request import BaseRequest
        except ImportError as exc:
            raise RuntimeError("lark_oapi not installed") from exc

        if use_uat and not self.access_token:
            raise NeedAuthorizationError(
                user_open_id=self.user_open_id or "unknown",
                reason="do_request called with use_uat=True but no access_token loaded; "
                       "call FeishuClient.for_user() or run 'hermes feishu-auth'",
            )

        http_method = HttpMethod.GET if method.upper() == "GET" else HttpMethod.POST

        builder = (
            BaseRequest.builder()
            .http_method(http_method)
            .uri(uri)
        )

        if use_uat:
            builder = builder.token_types({AccessTokenType.USER})
        else:
            builder = builder.token_types({AccessTokenType.TENANT})

        if paths:
            builder = builder.paths(paths)
        if queries:
            builder = builder.queries(queries)
        if body is not None:
            builder = builder.body(body)

        request = builder.build()

        if use_uat:
            opt = (
                RequestOption.builder()
                .user_access_token(self.access_token)
                .build()
            )
            response = self.sdk.request(request, opt)
        else:
            response = self.sdk.request(request)

        code = getattr(response, "code", None)
        msg = getattr(response, "msg", "")

        data: dict = {}
        raw = getattr(response, "raw", None)
        if raw and hasattr(raw, "content"):
            try:
                body_json = json.loads(raw.content)
                data = body_json.get("data", {})
                if code is None:
                    code = body_json.get("code", -1)
                if not msg:
                    msg = body_json.get("msg", "")
            except (json.JSONDecodeError, AttributeError):
                pass
        if not data:
            resp_data = getattr(response, "data", None)
            if isinstance(resp_data, dict):
                data = resp_data
            elif resp_data and hasattr(resp_data, "__dict__"):
                data = vars(resp_data)

        return (code if code is not None else -1), msg, data

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def dispose(self) -> None:
        """Remove this client from the instance cache."""
        cache_key = f"tenant:{self.account_id}:{self.app_id}"
        self._cache.pop(cache_key, None)

    def __repr__(self) -> str:
        mode = "UAT" if self.access_token else "TAT"
        return (
            f"<FeishuClient app_id={self.app_id!r} "
            f"domain={self.domain!r} mode={mode}>"
        )

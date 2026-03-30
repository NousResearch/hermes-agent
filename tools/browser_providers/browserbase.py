"""Browserbase cloud browser provider."""

import logging
import os
import threading
import uuid
from typing import Any, Dict, Optional

import requests

from tools.browser_providers.base import CloudBrowserProvider
from tools.managed_tool_gateway import resolve_managed_tool_gateway
from tools.tool_backend_helpers import managed_nous_tools_enabled

logger = logging.getLogger(__name__)
_pending_create_keys: Dict[str, str] = {}
_pending_create_keys_lock = threading.Lock()


def _get_or_create_pending_create_key(task_id: str) -> str:
    with _pending_create_keys_lock:
        existing = _pending_create_keys.get(task_id)
        if existing:
            return existing

        created = f"browserbase-session-create:{uuid.uuid4().hex}"
        _pending_create_keys[task_id] = created
        return created


def _clear_pending_create_key(task_id: str) -> None:
    with _pending_create_keys_lock:
        _pending_create_keys.pop(task_id, None)


def _should_preserve_pending_create_key(response: requests.Response) -> bool:
    if response.status_code >= 500:
        return True

    if response.status_code != 409:
        return False

    try:
        payload = response.json()
    except Exception:
        return False

    if not isinstance(payload, dict):
        return False

    error = payload.get("error")
    if not isinstance(error, dict):
        return False

    message = str(error.get("message") or "").lower()
    return "already in progress" in message


class BrowserbaseProvider(CloudBrowserProvider):
    """Browserbase (https://browserbase.com) cloud browser backend."""

    def provider_name(self) -> str:
        return "Browserbase"

    def is_configured(self) -> bool:
        return self._get_config_or_none() is not None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _get_config_or_none(self) -> Optional[Dict[str, Any]]:
        api_key = os.environ.get("BROWSERBASE_API_KEY")
        project_id = os.environ.get("BROWSERBASE_PROJECT_ID")
        if api_key and project_id:
            return {
                "api_key": api_key,
                "project_id": project_id,
                "base_url": os.environ.get("BROWSERBASE_BASE_URL", "https://api.browserbase.com").rstrip("/"),
                "managed_mode": False,
            }

        managed = resolve_managed_tool_gateway("browserbase")
        if managed is None:
            return None

        return {
            "api_key": managed.nous_user_token,
            "project_id": "managed",
            "base_url": managed.gateway_origin.rstrip("/"),
            "managed_mode": True,
        }

    def _get_config(self) -> Dict[str, Any]:
        config = self._get_config_or_none()
        if config is None:
            message = (
                "Browserbase requires direct BROWSERBASE_API_KEY/BROWSERBASE_PROJECT_ID credentials."
            )
            if managed_nous_tools_enabled():
                message = (
                    "Browserbase requires either direct BROWSERBASE_API_KEY/BROWSERBASE_PROJECT_ID "
                    "credentials or a managed Browserbase gateway configuration."
                )
            raise ValueError(message)
        return config

    def create_session(self, task_id: str) -> Dict[str, object]:
        config = self._get_config()
        managed_mode = bool(config.get("managed_mode"))

        # Optional env-var knobs
        enable_proxies = os.environ.get("BROWSERBASE_PROXIES", "true").lower() != "false"
        enable_advanced_stealth = os.environ.get("BROWSERBASE_ADVANCED_STEALTH", "false").lower() == "true"
        enable_keep_alive = os.environ.get("BROWSERBASE_KEEP_ALIVE", "true").lower() != "false"
        custom_timeout_ms = os.environ.get("BROWSERBASE_SESSION_TIMEOUT")

        features_enabled = {
            "basic_stealth": True,
            "proxies": False,
            "advanced_stealth": False,
            "keep_alive": False,
            "custom_timeout": False,
        }

        session_config: Dict[str, object] = {"projectId": config["project_id"]}

        if enable_keep_alive:
            session_config["keepAlive"] = True

        if custom_timeout_ms:
            try:
                timeout_val = int(custom_timeout_ms)
                if timeout_val > 0:
                    session_config["timeout"] = timeout_val
            except ValueError:
                logger.warning("Invalid BROWSERBASE_SESSION_TIMEOUT value: %s", custom_timeout_ms)

        if enable_proxies:
            session_config["proxies"] = True

        if enable_advanced_stealth:
            session_config["browserSettings"] = {"advancedStealth": True}

        # --- Create session via API ---
        headers = {
            "Content-Type": "application/json",
            "X-BB-API-Key": config["api_key"],
        }
        if managed_mode:
            headers["X-Idempotency-Key"] = _get_or_create_pending_create_key(task_id)

        response = requests.post(
            f"{config['base_url']}/v1/sessions",
            headers=headers,
            json=session_config,
            timeout=30,
        )

        proxies_fallback = False
        keepalive_fallback = False

        # Handle 402 — paid features unavailable
        if response.status_code == 402 and not managed_mode:
            if enable_keep_alive:
                keepalive_fallback = True
                logger.warning(
                    "keepAlive may require paid plan (402), retrying without it. "
                    "Sessions may timeout during long operations."
                )
                session_config.pop("keepAlive", None)
                response = requests.post(
                    f"{config['base_url']}/v1/sessions",
                    headers=headers,
                    json=session_config,
                    timeout=30,
                )

            if response.status_code == 402 and enable_proxies:
                proxies_fallback = True
                logger.warning(
                    "Proxies unavailable (402), retrying without proxies. "
                    "Bot detection may be less effective."
                )
                session_config.pop("proxies", None)
                response = requests.post(
                    f"{config['base_url']}/v1/sessions",
                    headers=headers,
                    json=session_config,
                    timeout=30,
                )

        if not response.ok:
            if managed_mode and not _should_preserve_pending_create_key(response):
                _clear_pending_create_key(task_id)
            raise RuntimeError(
                f"Failed to create Browserbase session: "
                f"{response.status_code} {response.text}"
            )

        session_data = response.json()
        if managed_mode:
            _clear_pending_create_key(task_id)
        session_name = f"hermes_{task_id}_{uuid.uuid4().hex[:8]}"
        external_call_id = response.headers.get("x-external-call-id") if managed_mode else None

        if enable_proxies and not proxies_fallback:
            features_enabled["proxies"] = True
        if enable_advanced_stealth:
            features_enabled["advanced_stealth"] = True
        if enable_keep_alive and not keepalive_fallback:
            features_enabled["keep_alive"] = True
        if custom_timeout_ms and "timeout" in session_config:
            features_enabled["custom_timeout"] = True

        feature_str = ", ".join(k for k, v in features_enabled.items() if v)
        logger.info("Created Browserbase session %s with features: %s", session_name, feature_str)

        return {
            "session_name": session_name,
            "bb_session_id": session_data["id"],
            "cdp_url": session_data["connectUrl"],
            "features": features_enabled,
            "external_call_id": external_call_id,
        }

    def close_session(self, session_id: str) -> bool:
        try:
            config = self._get_config()
        except ValueError:
            logger.warning("Cannot close Browserbase session %s — missing credentials", session_id)
            return False

        try:
            response = requests.post(
                f"{config['base_url']}/v1/sessions/{session_id}",
                headers={
                    "X-BB-API-Key": config["api_key"],
                    "Content-Type": "application/json",
                },
                json={
                    "projectId": config["project_id"],
                    "status": "REQUEST_RELEASE",
                },
                timeout=10,
            )
            if response.status_code in (200, 201, 204):
                logger.debug("Successfully closed Browserbase session %s", session_id)
                return True
            else:
                logger.warning(
                    "Failed to close session %s: HTTP %s - %s",
                    session_id,
                    response.status_code,
                    response.text[:200],
                )
                return False
        except Exception as e:
            logger.error("Exception closing Browserbase session %s: %s", session_id, e)
            return False

    def emergency_cleanup(self, session_id: str) -> None:
        config = self._get_config_or_none()
        if config is None:
            logger.warning("Cannot emergency-cleanup Browserbase session %s — missing credentials", session_id)
            return
        try:
            requests.post(
                f"{config['base_url']}/v1/sessions/{session_id}",
                headers={
                    "X-BB-API-Key": config["api_key"],
                    "Content-Type": "application/json",
                },
                json={
                    "projectId": config["project_id"],
                    "status": "REQUEST_RELEASE",
                },
                timeout=5,
            )
        except Exception as e:
            logger.debug("Emergency cleanup failed for Browserbase session %s: %s", session_id, e)

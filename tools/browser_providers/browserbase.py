"""Browserbase cloud browser provider."""

import logging
import os
import uuid
from typing import Dict

import requests

from tools.browser_providers.base import CloudBrowserProvider

logger = logging.getLogger(__name__)


class BrowserbaseProvider(CloudBrowserProvider):
    """Browserbase (https://browserbase.com) cloud browser backend."""

    def provider_name(self) -> str:
        return "Browserbase"

    def is_configured(self) -> bool:
        return bool(
            os.environ.get("BROWSERBASE_API_KEY")
            and os.environ.get("BROWSERBASE_PROJECT_ID")
        )

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _get_config(self) -> Dict[str, str]:
        api_key = os.environ.get("BROWSERBASE_API_KEY")
        project_id = os.environ.get("BROWSERBASE_PROJECT_ID")
        if not api_key or not project_id:
            raise ValueError(
                "BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID environment "
                "variables are required.  Get your credentials at "
                "https://browserbase.com"
            )
        return {"api_key": api_key, "project_id": project_id}

    def create_session(self, task_id: str) -> Dict[str, object]:
        config = self._get_config()

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
        response = requests.post(
            "https://api.browserbase.com/v1/sessions",
            headers=headers,
            json=session_config,
            timeout=30,
        )

        proxies_fallback = False
        keepalive_fallback = False

        # Handle 402 — paid features unavailable
        if response.status_code == 402:
            if enable_keep_alive:
                keepalive_fallback = True
                logger.warning(
                    "keepAlive may require paid plan (402), retrying without it. "
                    "Sessions may timeout during long operations."
                )
                session_config.pop("keepAlive", None)
                response = requests.post(
                    "https://api.browserbase.com/v1/sessions",
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
                    "https://api.browserbase.com/v1/sessions",
                    headers=headers,
                    json=session_config,
                    timeout=30,
                )

        if not response.ok:
            raise RuntimeError(
                f"Failed to create Browserbase session: "
                f"{response.status_code} {response.text}"
            )

        session_data = response.json()
        session_name = f"hermes_{task_id}_{uuid.uuid4().hex[:8]}"

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
        }

    def close_session(self, session_id: str) -> bool:
        try:
            config = self._get_config()
        except ValueError:
            logger.warning("Cannot close Browserbase session %s — missing credentials", session_id)
            return False

        try:
            response = requests.post(
                f"https://api.browserbase.com/v1/sessions/{session_id}",
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
        api_key = os.environ.get("BROWSERBASE_API_KEY")
        project_id = os.environ.get("BROWSERBASE_PROJECT_ID")
        if not api_key or not project_id:
            logger.warning("Cannot emergency-cleanup Browserbase session %s — missing credentials", session_id)
            return
        try:
            requests.post(
                f"https://api.browserbase.com/v1/sessions/{session_id}",
                headers={
                    "X-BB-API-Key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "projectId": project_id,
                    "status": "REQUEST_RELEASE",
                },
                timeout=5,
            )
        except Exception as e:
            logger.debug("Emergency cleanup failed for Browserbase session %s: %s", session_id, e)

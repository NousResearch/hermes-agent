import logging
import os
import uuid
import requests
from typing import Dict

from tools.browser_providers.base import CloudBrowserProvider

class BrowserlessProvider(CloudBrowserProvider):
    def provider_name(self) -> str:
        return "Browserless"

    def is_configured(self) -> bool:
        return bool(os.environ.get("BROWSERLESS_API_KEY"))

    def _api_url(self) -> str:
        return os.environ.get("BROWSERLESS_API_URL", "https://chrome.browserless.io")

    def _api_key(self) -> str:
        return os.environ.get("BROWSERLESS_API_KEY")

    def create_session(self, task_id: str) -> Dict[str, object]:
        api_url = self._api_url()
        api_key = self._api_key()
        if not api_key:
             raise ValueError("BROWSERLESS_API_KEY is required.")

        session_id = f"hermes_{task_id}_{uuid.uuid4().hex[:8]}"

        cdp_url = f"{api_url}?token={api_key}&timeout=300000&trackingId={session_id}"
        if api_url.startswith("http"):
            cdp_url = cdp_url.replace("http", "ws", 1)

        return {
            "session_name": session_id,
            "bb_session_id": session_id,
            "cdp_url": cdp_url,
            "features": {"browserless": True},
        }

    def close_session(self, session_id: str):
        return True

    def emergency_cleanup(self, session_id: str):
        pass

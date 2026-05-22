"""Firecrawl cloud browser provider."""

import logging
import os
import uuid
from typing import Dict

import requests

from tools.browser_providers.base import CloudBrowserProvider

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.firecrawl.dev"


class FirecrawlProvider(CloudBrowserProvider):
    """Firecrawl (https://firecrawl.dev) cloud browser backend."""

    def provider_name(self) -> str:
        return "Firecrawl"

    def is_configured(self) -> bool:
        return bool(os.environ.get("FIRECRAWL_API_KEY"))

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _api_url(self) -> str:
        return os.environ.get("FIRECRAWL_API_URL", _BASE_URL)

    def _headers(self) -> Dict[str, str]:
        api_key = os.environ.get("FIRECRAWL_API_KEY")
        if not api_key:
            raise ValueError(
                "FIRECRAWL_API_KEY environment variable is required. "
                "Get your key at https://firecrawl.dev"
            )
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def create_session(self, task_id: str) -> Dict[str, object]:
        ttl = int(os.environ.get("FIRECRAWL_BROWSER_TTL", "300"))

        body: Dict[str, object] = {"ttl": ttl}

        response = requests.post(
            f"{self._api_url()}/v2/browser",
            headers=self._headers(),
            json=body,
            timeout=30,
        )

        if not response.ok:
            raise RuntimeError(
                f"Failed to create Firecrawl browser session: "
                f"{response.status_code} {response.text}"
            )

        data = response.json()
        session_name = f"hermes_{task_id}_{uuid.uuid4().hex[:8]}"

        logger.info("Created Firecrawl browser session %s", session_name)

        return {
            "session_name": session_name,
            "bb_session_id": data["id"],
            "cdp_url": data["cdpUrl"],
            "features": {"firecrawl": True},
        }

    def close_session(self, session_id: str) -> bool:
        try:
            response = requests.delete(
                f"{self._api_url()}/v2/browser/{session_id}",
                headers=self._headers(),
                timeout=10,
            )
            if response.status_code in {200, 201, 204}:
                logger.debug("Successfully closed Firecrawl session %s", session_id)
                return True
            else:
                logger.warning(
                    "Failed to close Firecrawl session %s: HTTP %s - %s",
                    session_id,
                    response.status_code,
                    response.text[:200],
                )
                return False
        except Exception as e:
            logger.error("Exception closing Firecrawl session %s: %s", session_id, e)
            return False

    def emergency_cleanup(self, session_id: str) -> None:
        try:
            requests.delete(
                f"{self._api_url()}/v2/browser/{session_id}",
                headers=self._headers(),
                timeout=5,
            )
        except ValueError:
            logger.warning("Cannot emergency-cleanup Firecrawl session %s — missing credentials", session_id)
        except Exception as e:
            logger.debug("Emergency cleanup failed for Firecrawl session %s: %s", session_id, e)

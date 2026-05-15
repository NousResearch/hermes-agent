"""Shared Plane REST client for Hermes tools.

Read-first client for Plane Cloud with a browser-like User-Agent to avoid
Cloudflare `browser_signature_banned` false negatives on otherwise valid API
requests.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from hermes_constants import get_hermes_home
from hermes_cli.env_loader import load_hermes_dotenv

BASE_URL = "https://api.plane.so"
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
)
UNSUPPORTED_EXTERNAL_FILTER_STATUS_CODES = {400, 404, 422}


class PlaneConfigurationError(ValueError):
    """Raised when Plane environment configuration is incomplete."""


class PlaneAPIError(RuntimeError):
    """Raised when a Plane API request fails."""

    def __init__(self, status_code: int, body: str, url: str):
        self.status_code = int(status_code)
        self.status = self.status_code  # backward-compatible alias
        self.body = body
        self.url = url
        hint = ""
        if "browser_signature_banned" in body:
            hint = " Retry with a browser-like User-Agent."
        super().__init__(f"Plane API error {status_code} for {url}: {body}{hint}")


@dataclass
class PlaneConfig:
    api_key: str
    workspace_slug: str
    project_id: str
    base_url: str = BASE_URL
    user_agent: str = BROWSER_USER_AGENT

    @classmethod
    def from_env(cls) -> "PlaneConfig":
        load_hermes_dotenv(hermes_home=get_hermes_home())
        api_key = (os.getenv("PLANE_API_KEY") or "").strip()
        workspace_slug = (os.getenv("PLANE_WORKSPACE") or "").strip()
        project_id = (os.getenv("PLANE_PROJECT_ID") or "").strip()
        base_url = (os.getenv("PLANE_BASE_URL") or BASE_URL).strip().rstrip("/")
        missing = [
            name
            for name, value in (
                ("PLANE_API_KEY", api_key),
                ("PLANE_WORKSPACE", workspace_slug),
                ("PLANE_PROJECT_ID", project_id),
            )
            if not value
        ]
        if missing:
            raise PlaneConfigurationError(
                f"Missing required Plane env vars: {', '.join(missing)}"
            )
        return cls(
            api_key=api_key,
            workspace_slug=workspace_slug,
            project_id=project_id,
            base_url=base_url,
        )


class PlaneClient:
    def __init__(self, config: PlaneConfig):
        self.config = config
        self._project_cache: Optional[dict[str, Any]] = None
        self._states_cache: Optional[list[dict[str, Any]]] = None
        self._labels_cache: Optional[list[dict[str, Any]]] = None

    @classmethod
    def from_env(cls) -> "PlaneClient":
        return cls(PlaneConfig.from_env())

    @property
    def workspace_slug(self) -> str:
        return self.config.workspace_slug

    @property
    def project_id(self) -> str:
        return self.config.project_id

    @property
    def base_url(self) -> str:
        return self.config.base_url.rstrip("/")

    def headers(self) -> dict[str, str]:
        return {
            "X-API-Key": self.config.api_key,
            "Accept": "application/json",
            "User-Agent": self.config.user_agent,
        }

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
        timeout: int = 30,
    ) -> Any:
        url = f"{self.base_url}{path}"
        clean_params = {k: v for k, v in (params or {}).items() if v is not None}
        if clean_params:
            url += "?" + urllib.parse.urlencode(clean_params, doseq=True)

        data = None
        headers = self.headers()
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(
            url,
            data=data,
            method=method.upper(),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                if not raw:
                    return None
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise PlaneAPIError(exc.code, error_body, url) from exc

    # Backward-compatible private aliases used by earlier local drafts.
    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        body: Optional[dict[str, Any]] = None,
        timeout: int = 30,
    ) -> Any:
        return self.request(method, path, params=params, json_body=body, timeout=timeout)

    def _project_path(self) -> str:
        return f"/api/v1/workspaces/{self.workspace_slug}/projects/{self.project_id}/"

    def _work_items_path(self) -> str:
        return self._project_path() + "work-items/"

    def get_current_user(self) -> dict[str, Any]:
        """Return the authenticated Plane user.

        This intentionally exercises the same auth headers and browser-like
        User-Agent as every other Plane request, making it suitable for a cheap
        integration health check.
        """
        data = self.request("GET", "/api/v1/users/me/")
        if isinstance(data, dict):
            return dict(data)
        return {}

    def paginate(self, path: str, *, params: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        next_params = dict(params or {})
        while True:
            data = self.request("GET", path, params=next_params)
            if isinstance(data, list):
                out.extend(data)
                break
            if not isinstance(data, dict):
                break
            out.extend(list(data.get("results") or []))
            if not data.get("next_page_results"):
                break
            cursor = data.get("next_cursor") or data.get("cursor")
            if not cursor:
                break
            next_params = dict(next_params)
            next_params["cursor"] = cursor
        return out

    def get_project(self, force: bool = False) -> dict[str, Any]:
        if self._project_cache is None or force:
            self._project_cache = self.request("GET", self._project_path())
        return dict(self._project_cache or {})

    def get_project_identifier(self) -> str:
        project = self.get_project()
        identifier = (
            project.get("identifier")
            or project.get("project_identifier")
            or project.get("key")
            or ""
        )
        return str(identifier).strip()

    def list_states(self, force: bool = False) -> list[dict[str, Any]]:
        if self._states_cache is None or force:
            self._states_cache = self.paginate(
                self._project_path() + "states/", params={"per_page": 100}
            )
        return [dict(x) for x in (self._states_cache or [])]

    def list_labels(self, force: bool = False) -> list[dict[str, Any]]:
        if self._labels_cache is None or force:
            self._labels_cache = self.paginate(
                self._project_path() + "labels/", params={"per_page": 100}
            )
        return [dict(x) for x in (self._labels_cache or [])]

    def list_work_items(
        self,
        *,
        per_page: int = 100,
        limit: Optional[int] = None,
        expand: Optional[Iterable[str]] = ("state", "assignees", "labels"),
        fields: Optional[Iterable[str]] = None,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"per_page": min(max(int(per_page), 1), 100)}
        if expand:
            params["expand"] = ",".join(str(x) for x in expand if x)
        if fields:
            params["fields"] = ",".join(str(x) for x in fields if x)
        for key, value in filters.items():
            if value is not None:
                params[key] = value
        items = self.paginate(self._work_items_path(), params=params)
        if limit is not None:
            return items[: int(limit)]
        return items

    def find_work_item_by_external_id(
        self,
        *,
        external_source: str,
        external_id: str,
    ) -> Optional[dict[str, Any]]:
        source = str(external_source or "").strip()
        identifier = str(external_id or "").strip()
        if not source or not identifier:
            return None

        # Plane Cloud may support these filters directly. Keep the exact same
        # client-side filter below as a compatibility fallback because the API
        # has not been stable across deployments for external fields.
        try:
            candidates = self.list_work_items(
                external_source=source,
                external_id=identifier,
            )
        except PlaneAPIError as exc:
            # Plane Cloud may reject these query params on some deployments.
            # Treat the filtered lookup as best effort, then fall back to a
            # normal expanded board scan below.
            if exc.status_code in UNSUPPORTED_EXTERNAL_FILTER_STATUS_CODES:
                candidates = []
            else:
                raise
        for item in candidates:
            if (
                str(item.get("external_source") or "").strip() == source
                and str(item.get("external_id") or "").strip() == identifier
            ):
                return item

        # If the API ignored unknown query params or omitted external fields in
        # filtered results, fetch a normal expanded page set and filter locally.
        for item in self.list_work_items():
            if (
                str(item.get("external_source") or "").strip() == source
                and str(item.get("external_id") or "").strip() == identifier
            ):
                return item
        return None

    def get_work_item_by_id(self, work_item_id: str) -> dict[str, Any]:
        wid = str(work_item_id).strip()
        if not wid:
            raise ValueError("work_item_id is required")
        return self.request("GET", self._work_items_path() + f"{wid}/")

    def get_work_item_by_readable_id(self, readable_id: str) -> dict[str, Any]:
        rid = str(readable_id).strip()
        if not rid:
            raise ValueError("readable_id is required")
        return self.request("GET", f"/api/v1/workspaces/{self.workspace_slug}/work-items/{rid}/")

    def get_work_item(
        self,
        work_item_id: Optional[str] = None,
        *,
        readable_id: Optional[str] = None,
        sequence_id: Optional[int] = None,
    ) -> dict[str, Any]:
        if work_item_id:
            return self.get_work_item_by_id(work_item_id)
        if readable_id:
            return self.get_work_item_by_readable_id(readable_id)
        if sequence_id is not None:
            project_identifier = self.get_project_identifier()
            if not project_identifier:
                raise ValueError("project identifier unavailable, cannot resolve sequence_id")
            return self.get_work_item_by_readable_id(f"{project_identifier}-{int(sequence_id)}")
        raise ValueError("one of work_item_id, readable_id, or sequence_id is required")

    def create_work_item(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not payload:
            raise ValueError("payload is required")
        return self.request("POST", self._work_items_path(), json_body=payload)

    def update_work_item(self, work_item_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        wid = str(work_item_id).strip()
        if not wid:
            raise ValueError("work_item_id is required")
        if not payload:
            raise ValueError("payload is required")
        return self.request("PATCH", self._work_items_path() + f"{wid}/", json_body=payload)

    def add_comment(self, work_item_id: str, comment_html: str) -> dict[str, Any]:
        wid = str(work_item_id).strip()
        if not wid:
            raise ValueError("work_item_id is required")
        body = str(comment_html or "").strip()
        if not body:
            raise ValueError("comment_html is required")
        return self.request(
            "POST",
            self._project_path() + f"issues/{wid}/comments/",
            json_body={"comment_html": body},
        )

    def resolve_state_id(self, state_value: Optional[str]) -> Optional[str]:
        if state_value is None:
            return None
        raw = str(state_value).strip()
        if not raw:
            return None
        states = self.list_states()
        lowered = raw.casefold()
        for state in states:
            sid = str(state.get("id") or "").strip()
            if sid == raw:
                return sid
        for state in states:
            if str(state.get("name") or "").strip().casefold() == lowered:
                return str(state.get("id") or "").strip() or None
        raise ValueError(f"Unknown Plane state: {state_value}")

    def resolve_label_ids(self, labels: Optional[Iterable[str]]) -> Optional[list[str]]:
        if labels is None:
            return None
        raw_labels = [str(x).strip() for x in labels if str(x).strip()]
        if not raw_labels:
            return []
        known = self.list_labels()
        by_id = {
            str(item.get("id") or "").strip(): str(item.get("id") or "").strip()
            for item in known
            if str(item.get("id") or "").strip()
        }
        by_name = {
            str(item.get("name") or "").strip().casefold(): str(item.get("id") or "").strip()
            for item in known
            if str(item.get("name") or "").strip()
        }
        resolved: list[str] = []
        for value in raw_labels:
            if value in by_id:
                resolved.append(by_id[value])
                continue
            match = by_name.get(value.casefold())
            if match:
                resolved.append(match)
                continue
            raise ValueError(f"Unknown Plane label: {value}")
        out: list[str] = []
        seen: set[str] = set()
        for item in resolved:
            if item not in seen:
                out.append(item)
                seen.add(item)
        return out

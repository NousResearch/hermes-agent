"""Hermes Workstation — Decoupled Headless Browser Daemon Prototype (V4.0).

Runs as an autonomous background daemon process, independent of Electron Desktop Main.
Exposes loopback REST API for browser actions and viewport streaming, allowing
the agent to automate the browser continuously even if Desktop UI restarts or crashes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging
import os
from pathlib import Path
import secrets
import socket
import threading
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DaemonTab:
    id: str
    owner_task_id: Optional[str] = None
    url: str = "about:blank"
    title: str = "New Tab"
    created_at: str = field(default_factory=_utc_now)
    last_active_at: str = field(default_factory=_utc_now)
    viewport_attached: bool = False
    viewport_client_id: Optional[str] = None


class BrowserDaemonRequestHandler(BaseHTTPRequestHandler):
    daemon_instance: "HermesBrowserDaemon"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def _send_json(self, status: int, data: Dict[str, Any]) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _authenticate(self) -> bool:
        auth_header = self.headers.get("Authorization", "")
        expected = f"Bearer {self.daemon_instance.token}"
        if auth_header != expected:
            self._send_json(401, {"success": False, "error": "unauthorized"})
            return False
        return True

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json(200, {
                "success": True,
                "status": "ok",
                "daemon": "hermes-browser-daemon",
                "version": "4.0.0-prototype",
                "pid": os.getpid(),
                "tabs_count": len(self.daemon_instance.tabs),
                "uptime_seconds": time.monotonic() - self.daemon_instance.started_at,
            })
            return

        if not self._authenticate():
            return

        if parsed.path == "/v1/resources":
            res = self.daemon_instance.get_resources()
            self._send_json(200, {"success": True, "result": res})
            return

        self._send_json(404, {"success": False, "error": "not_found"})

    def do_POST(self) -> None:
        if not self._authenticate():
            return

        parsed = urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            payload = json.loads(body)
        except Exception:
            self._send_json(400, {"success": False, "error": "invalid_json"})
            return

        if parsed.path == "/v1/action":
            try:
                res = self.daemon_instance.dispatch_action(payload)
                self._send_json(200, {"success": True, "result": res})
            except Exception as exc:
                self._send_json(500, {"success": False, "error": str(exc)})
            return

        if parsed.path == "/v1/viewport/attach":
            token = self.daemon_instance.attach_viewport(payload)
            self._send_json(200, {"success": True, "token": token})
            return

        if parsed.path == "/v1/viewport/detach":
            self.daemon_instance.detach_viewport(payload)
            self._send_json(200, {"success": True, "status": "detached"})
            return

        self._send_json(404, {"success": False, "error": "not_found"})


class HermesBrowserDaemon:
    """Headless browser daemon running on loopback independent of Desktop UI."""

    def __init__(self, *, port: int = 0, descriptor_dir: Optional[Path] = None) -> None:
        self.descriptor_dir = Path(descriptor_dir) if descriptor_dir else get_hermes_home() / "workstation"
        self.descriptor_dir.mkdir(parents=True, exist_ok=True)
        self.descriptor_path = self.descriptor_dir / "daemon-control.json"
        self.token = secrets.token_hex(16)
        self.tabs: Dict[str, DaemonTab] = {}
        self.task_tabs: Dict[str, str] = {}
        self.active_tab_id: Optional[str] = None
        self.started_at = time.monotonic()
        self.server: Optional[HTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.port = port
        self.url = ""

    def start(self) -> str:
        """Start the loopback HTTP daemon server and write the control descriptor."""
        handler_cls = type(
            "BoundBrowserDaemonRequestHandler",
            (BrowserDaemonRequestHandler,),
            {"daemon_instance": self},
        )
        self.server = HTTPServer(("127.0.0.1", self.port), handler_cls)
        assigned_port = self.server.server_address[1]
        self.port = assigned_port
        self.url = f"http://127.0.0.1:{assigned_port}"

        # Write descriptor atomically
        desc_data = {
            "version": 1,
            "url": self.url,
            "token": self.token,
            "pid": os.getpid(),
            "created_at": _utc_now(),
        }
        temp_path = self.descriptor_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(desc_data, indent=2), encoding="utf-8")
        temp_path.replace(self.descriptor_path)

        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        logger.info("Hermes Browser Daemon started on %s (pid %d)", self.url, os.getpid())
        return self.url

    def stop(self) -> None:
        """Shut down the daemon and remove descriptor."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.descriptor_path.exists():
            try:
                self.descriptor_path.unlink()
            except Exception:
                pass
        logger.info("Hermes Browser Daemon stopped")

    def _get_or_create_tab_for_task(self, task_id: Optional[str]) -> DaemonTab:
        with self._lock:
            if task_id and task_id in self.task_tabs:
                tab_id = self.task_tabs[task_id]
                if tab_id in self.tabs:
                    return self.tabs[tab_id]

            tab_id = f"tab_{secrets.token_hex(6)}"
            tab = DaemonTab(id=tab_id, owner_task_id=task_id)
            self.tabs[tab_id] = tab
            if task_id:
                self.task_tabs[task_id] = tab_id
            if not self.active_tab_id:
                self.active_tab_id = tab_id
            return tab

    def dispatch_action(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch browser action in daemon data plane."""
        action = payload.get("action", "")
        args = payload.get("arguments", {})
        task_id = payload.get("task_id")

        tab = self._get_or_create_tab_for_task(task_id)
        tab.last_active_at = _utc_now()

        if action == "browser_navigate":
            target_url = str(args.get("url", "about:blank"))
            tab.url = target_url
            tab.title = f"Page at {target_url}"
            return {
                "success": True,
                "runtime": "daemon-headless-chromium",
                "task_id": task_id,
                "tab_id": tab.id,
                "url": tab.url,
                "title": tab.title,
            }

        if action == "browser_snapshot":
            return {
                "success": True,
                "runtime": "daemon-headless-chromium",
                "tab_id": tab.id,
                "url": tab.url,
                "title": tab.title,
                "elements": [{"ref": "e1", "role": "heading", "text": tab.title}],
            }

        if action == "browser_extract_items":
            limit = int(args.get("limit", 20))
            items = [
                {"title": f"Daemon Item {i}", "url": f"{tab.url}/item/{i}", "price": f"${i*10}.00"}
                for i in range(1, min(limit + 1, 10))
            ]
            return {
                "success": True,
                "runtime": "daemon-headless-chromium",
                "tab_id": tab.id,
                "url": tab.url,
                "count": len(items),
                "items": items,
            }

        return {
            "success": True,
            "runtime": "daemon-headless-chromium",
            "action": action,
            "tab_id": tab.id,
        }

    def get_resources(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "ready": True,
                "runtime": "daemon-headless",
                "active_tab_id": self.active_tab_id,
                "tabs": [asdict(t) for t in self.tabs.values()],
                "tasks": list(self.task_tabs.keys()),
            }

    def attach_viewport(self, payload: Dict[str, Any]) -> str:
        tab_id = payload.get("tab_id") or self.active_tab_id
        client_id = payload.get("client_id", "desktop_ui")
        with self._lock:
            if tab_id and tab_id in self.tabs:
                tab = self.tabs[tab_id]
                tab.viewport_attached = True
                tab.viewport_client_id = client_id
        return f"vptoken_{secrets.token_hex(8)}"

    def detach_viewport(self, payload: Dict[str, Any]) -> None:
        tab_id = payload.get("tab_id") or self.active_tab_id
        with self._lock:
            if tab_id and tab_id in self.tabs:
                tab = self.tabs[tab_id]
                tab.viewport_attached = False
                tab.viewport_client_id = None

"""App launch and bounded window discovery for the cua-driver backend."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

from tools.computer_use.backend import ActionResult
from tools.computer_use.cua_backend_capture import _select_capture_target, _sorted_windows
from tools.computer_use.cua_backend_parse import _ingest_windows, _positive_int


class _LaunchMixin:
    def launch_app(self, *, bundle_id: Optional[str] = None, name: Optional[str] = None,
                   path: Optional[str] = None, aumid: Optional[str] = None,
                   launch_path: Optional[str] = None, urls: Optional[List[str]] = None,
                   additional_arguments: Optional[List[str]] = None,
                   creates_new_application_instance: bool = False,
                   start_minimized: bool = False, wait_timeout: float = 10.0) -> ActionResult:
        """Launch once, then bind a proven window without activating it.

        An accepted launch with no window remains ok=True with window_ready=False:
        it must not be replayed. Discovery never falls back to the previous target
        or to another app's frontmost window.
        """
        if not any((bundle_id, name, path, aumid, launch_path, urls)):
            raise ValueError("launch_app requires bundle_id or name, or an exact path, aumid, launch_path, or urls")
        if isinstance(wait_timeout, bool) or not math.isfinite(wait_timeout) or not 0 <= wait_timeout <= 30:
            return ActionResult(ok=False, action="launch_app", code="bad_wait_timeout",
                                message="wait_timeout must be a finite number between 0 and 30 seconds.")
        args: Dict[str, Any] = {k: v for k, v in (
            ("bundle_id", bundle_id), ("name", name), ("path", path), ("aumid", aumid),
            ("launch_path", launch_path), ("urls", list(urls) if urls else None),
            ("additional_arguments", list(additional_arguments) if additional_arguments else None),
            ("creates_new_application_instance", creates_new_application_instance or None),
            ("start_minimized", start_minimized or None)) if v is not None}
        self._session.start()  # discover the live tool schema before checking support
        if self._session.capabilities_discovered:
            if not self._session._has_tool("launch_app"):
                return ActionResult(ok=False, action="launch_app", code="launch_app_unsupported",
                                    message="This cua-driver does not advertise launch_app.")
            unsupported = [k for k in args if not self._session.supports_input_property("launch_app", k)]
            if unsupported:
                return ActionResult(ok=False, action="launch_app", code="launch_parameters_unsupported",
                                    message="This cua-driver does not accept launch parameters: " + ", ".join(unsupported),
                                    meta={"unsupported_parameters": unsupported})
        result = self._action("launch_app", args)
        if not result.ok:
            if result.code in {"transport_outcome_unknown", "timeout_outcome_unknown"}:
                self._clear_active_target()  # it may have launched despite the missing reply
            return result  # a refused launch leaves the previously selected window intact
        self._clear_active_target()  # launch accepted: old window and element tokens are no longer the target
        pid = _positive_int(result.meta.get("pid"))
        raw_windows = result.meta.get("windows")
        windows = _ingest_windows([
            {**w, **({"pid": pid} if "pid" not in w and pid is not None else {})}
            for w in raw_windows if isinstance(w, dict)
        ]) if isinstance(raw_windows, list) else []
        # A launch reply is authoritative, but reject explicitly conflicting owners.
        windows = [w for w in windows if pid is None or w["pid"] == pid]
        if pid is None and len({w["pid"] for w in windows}) != 1:
            windows = []
        deadline = time.monotonic() + wait_timeout
        discovery_error = ""
        while not windows and (remaining := deadline - time.monotonic()) > 0:
            try:
                pid = pid or self._launched_app_pid(args, timeout=remaining)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if pid is not None:
                    out = self._session.call_tool("list_windows", {
                        "pid": pid, "on_screen_only": False, "session": self._session_id}, timeout=remaining)
                    if out.get("isError") is True:
                        raise RuntimeError(str(out.get("data") or out.get("structuredContent") or "list_windows failed"))
                    # Some drivers ignore the pid filter. Enforce it locally as well.
                    windows = [w for w in _sorted_windows(out) if w["pid"] == pid]
            except Exception as exc:
                discovery_error = str(exc)
                break
            if not windows:
                time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
        if windows:
            windows.sort(key=lambda w: w["z_index"], reverse=True)
            target = _select_capture_target(windows, app_requested=True, exact_target=True)
            self._set_active_target(target)
            self._last_app = target["app_name"] or result.meta.get("name") or name or bundle_id or ""
            result.meta.update(pid=target["pid"], window_id=target["window_id"], window_ready=True)
        else:
            result.meta["window_ready"] = False
            result.code = "window_discovery_failed" if discovery_error else "window_not_ready"
            if discovery_error:
                result.meta["window_discovery_error"] = discovery_error
            result.message = (result.message + " " if result.message else "") + (
                "Launch accepted, but no target window is ready. Do not relaunch automatically. "
                "Use list_windows(on_screen_only=false) and capture with an explicit pid/window_id when the app opens a window.")
        return result

    def _launched_app_pid(self, args: Dict[str, Any], *, timeout: float) -> Optional[int]:
        """Recover a delayed PID only from a unique, exact running-app identity."""
        key = next((k for k in ("launch_path", "path", "aumid", "bundle_id", "name") if args.get(k)), None)
        if key is None:
            return None
        identifier = str(args[key]).strip()
        fields = {"launch_path": ("launch_path", "path"), "path": ("path", "launch_path"),
                  "aumid": ("aumid", "bundle_id"), "bundle_id": ("bundle_id", "bundleId"),
                  "name": ("name", "app_name", "display_name")}[key]
        pids = set()
        for app in self.list_apps(timeout=timeout):
            if not isinstance(app, dict) or app.get("running") is False:
                continue
            aliases = {v.strip() for k in fields
                       if isinstance((v := app.get(k)), str) and v.strip()}
            matches = (identifier.casefold() in {v.casefold() for v in aliases} if key == "name"
                       else identifier in aliases)
            if matches and (pid := _positive_int(app.get("pid"))) is not None:
                pids.add(pid)
        return next(iter(pids)) if len(pids) == 1 else None

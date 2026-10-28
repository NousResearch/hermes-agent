"""Record real harness target creation and selection without replaying user code."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any


def prepare_owned_exec(code: str, info: dict[str, Any]) -> tuple[str, Path]:
    directory = Path(info["_owner_home"]) / "cache/browser-use/owned-exec"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                     prefix="targets-", suffix=".json", delete=False) as report:
        json.dump({"created": [], "current": info["target_id"]}, report)
        path = Path(report.name)
    payload = {"code": code, "target": info["target_id"], "report": str(path)}
    adapter = str(Path(__file__).resolve())
    prepared = ("import json as _hermes_json, runpy as _hermes_runpy\n"
                f"_hermes_runpy.run_path({adapter!r})['run_owned_code']("
                f"**_hermes_json.loads({json.dumps(payload)!r}), namespace=globals())\n")
    info["_exec_report"] = str(path)
    return prepared, path


def consume_owned_exec(info: dict[str, Any]) -> None:
    """Merge exact IDs observed from actual CDP replies, including when execution raised."""
    value = info.get("_exec_report")
    if value is None:
        return
    path = Path(value)
    pending = path.with_suffix(".next")
    reports = [path] + ([pending] if pending.exists() else [])
    for record in reports:
        report = json.loads(record.read_bytes())
        created, current = report["created"], report["current"]
        if not isinstance(created, list) or any(not isinstance(target, str) or not target for target in created):
            raise ValueError("Invalid owned browser target report")
        if current is not None and (not isinstance(current, str) or not current):
            raise ValueError("Invalid current browser target report")
        info.setdefault("_owned_target_ids", set()).update(created)
        info["target_id"] = current
    if pending.exists():
        pending.unlink()
    path.unlink()
    info.pop("_exec_report")


def run_owned_code(code: str, target: str, report: str, namespace: dict[str, Any]) -> None:
    """Run in the real harness interpreter; wrap its real CDP calls only to record ownership."""
    from browser_harness import helpers

    original_cdp, original_switch = helpers.cdp, helpers.switch_tab
    state: dict[str, Any] = {"created": [], "current": target}
    destination = Path(report)

    def persist() -> None:
        pending = destination.with_suffix(".next")
        with pending.open("x", encoding="utf-8") as stream:
            pending.chmod(0o600)
            json.dump(state, stream)
        pending.replace(destination)

    def owned_cdp(method, *args, **kwargs):
        result = original_cdp(method, *args, **kwargs)
        if method == "Target.createTarget":
            state["created"].append(result["targetId"])
            persist()
        elif method == "Target.closeTarget" and result.get("success"):
            if kwargs.get("targetId") == state["current"]:
                state["current"] = None
            persist()
        return result

    def owned_switch(selected, *args, **kwargs):
        result = original_switch(selected, *args, **kwargs)
        state["current"] = (selected.get("targetId") or selected.get("target_id")) if isinstance(selected, dict) else selected
        persist()
        return result

    helpers.cdp, helpers.switch_tab = owned_cdp, owned_switch
    namespace.update(cdp=owned_cdp, switch_tab=owned_switch)
    owned_switch(target)
    try:
        exec(compile(code, "<browser_exec>", "exec"), namespace)
    finally:
        persist()


def stop_owned_daemon() -> None:
    """Wait for the identity-checked daemon to exit before its IPC files are removed."""
    from browser_harness import _ipc as ipc, admin
    import psutil

    name = os.environ["BU_NAME"]
    pid = ipc.identify(name, timeout=5.0)
    if pid is not None:
        process = psutil.Process(pid)
        connection, token = ipc.connect(name, timeout=5.0)
        try:
            response = ipc.request(connection, token, {"meta": "shutdown"})
        finally:
            connection.close()
        if response.get("ok") is not True or response.get("error"):
            raise RuntimeError("The owned browser daemon did not confirm shutdown.")
        process.wait(timeout=15)
    admin.restart_daemon(name)
    if ipc.ping(name, timeout=1.0) or ipc.port_path(name).exists() or ipc.pid_path(name).exists():
        raise RuntimeError("The owned browser daemon or its IPC records remain after cleanup.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-daemon", action="store_true", required=True)
    parser.parse_args()
    stop_owned_daemon()

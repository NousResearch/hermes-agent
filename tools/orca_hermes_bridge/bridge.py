"""Long-running two-way account bridge between Orca and Hermes."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

from hermes_constants import get_hermes_home
from hermes_cli.auth import _auth_store_lock, _load_auth_store

from .accounts import parse_orca_accounts, reorder_codex_pool
from .rpc import OrcaRpcClient
from .state import BridgeState, reconcile
from .windows import AlreadyRunningError, SingletonLock, show_qwen_notification


LOGGER = logging.getLogger("orca_hermes_bridge")


@dataclass(frozen=True)
class DaemonLaunch:
    argv: list[str]
    cwd: Path


def build_daemon_launch(repo_root: Path, python_executable: Path) -> DaemonLaunch:
    return DaemonLaunch(
        argv=[
            str(python_executable),
            "-m",
            "tools.orca_hermes_bridge.bridge",
            "--daemon",
        ],
        cwd=Path(repo_root),
    )


def retry_delay(failures: int) -> int:
    return min(2 ** max(1, failures), 30)


def load_state(path: Path) -> BridgeState:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("state must be an object")
        allowed = BridgeState.__dataclass_fields__.keys()
        return BridgeState(**{key: payload[key] for key in allowed if key in payload})
    except (OSError, ValueError, TypeError):
        return BridgeState()


def save_state(path: Path, state: BridgeState) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{path.name}.tmp.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(asdict(state), handle, separators=(",", ":"), sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_codex_rows() -> list[dict[str, Any]]:
    with _auth_store_lock():
        store = _load_auth_store()
        pool = store.get("credential_pool")
        rows = pool.get("openai-codex") if isinstance(pool, dict) else None
        if not isinstance(rows, list):
            return []
        return [dict(row) for row in rows if isinstance(row, dict)]


class Bridge:
    def __init__(
        self,
        *,
        state_path: Path,
        rpc: Any | None = None,
        pool_reader: Callable[[], list[dict[str, Any]]] = _read_codex_rows,
        pool_mutator: Callable[..., bool] = reorder_codex_pool,
        notifier: Callable[[], None] = show_qwen_notification,
        clock: Callable[[], float] = time.time,
    ):
        self.state_path = Path(state_path)
        self.rpc = rpc or OrcaRpcClient()
        self.pool_reader = pool_reader
        self.pool_mutator = pool_mutator
        self.notifier = notifier
        self.clock = clock
        self.state = load_state(self.state_path)

    def tick(self) -> None:
        snapshot = parse_orca_accounts(self.rpc.list_accounts())
        rows = self.pool_reader()
        decision = reconcile(snapshot, rows, self.state, self.clock())

        if decision.orca_mutation is not None:
            # Persist the marker first so a crash cannot turn our RPC echo into
            # a user-originated selection on the next process start.
            self.state = decision.state
            save_state(self.state_path, self.state)
            try:
                self.rpc.select_host_codex(decision.orca_mutation.account_id)
            except Exception:
                self.state = replace(
                    self.state,
                    pending_orca_provider_id=None,
                    pending_started_at=None,
                )
                save_state(self.state_path, self.state)
                raise

        if decision.pool_mutation is not None:
            self.pool_mutator(
                decision.pool_mutation.provider_account_id,
                clear_selected_status=decision.pool_mutation.clear_selected_status,
            )

        self.state = decision.state
        save_state(self.state_path, self.state)
        if decision.notify_qwen:
            try:
                self.notifier()
            except Exception:
                LOGGER.warning("Qwen fallback notification failed", exc_info=True)


def _paths() -> tuple[Path, Path, Path]:
    home = Path(get_hermes_home())
    return (
        home / "orca-account-bridge-state.json",
        home / "orca-account-bridge.lock",
        home / "logs" / "orca-account-bridge.log",
    )


def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _run_daemon(bridge: Bridge, lock_path: Path) -> int:
    failures = 0
    try:
        with SingletonLock(lock_path):
            while True:
                try:
                    bridge.tick()
                    failures = 0
                    time.sleep(2)
                except KeyboardInterrupt:
                    return 0
                except Exception as exc:
                    failures += 1
                    LOGGER.warning(
                        "bridge tick failed type=%s code=%s",
                        type(exc).__name__,
                        getattr(exc, "code", "unknown"),
                    )
                    time.sleep(retry_delay(failures))
    except AlreadyRunningError:
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--daemon", action="store_true")
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)
    state_path, lock_path, log_path = _paths()

    if args.status:
        state = load_state(state_path)
        print(json.dumps(asdict(state), sort_keys=True))
        return 0

    _configure_logging(log_path)
    bridge = Bridge(state_path=state_path)
    if args.once:
        bridge.tick()
        return 0
    return _run_daemon(bridge, lock_path)


if __name__ == "__main__":
    raise SystemExit(main())

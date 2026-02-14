"""Nomad SlotPool sandbox backend (ported from Nous/atropos-agent).

This module provides an OPTIONAL slot-based sandbox execution backend.

Design goals:
- Keep Hermes-Agent's default terminal backends unchanged.
- Enable the Nomad backend with minimal friction:
    TERMINAL_ENV=nomad

How it works (high level):
- A local Nomad dev agent manages one or more sandbox-server containers.
- Each container hosts N "slots" (workspace dirs).
- Each agent trajectory acquires a slot, runs tools inside it, then releases.

This file is intentionally self-contained so we can iterate without touching
other tool implementations.

NOTE: This backend requires aiohttp and a working Nomad installation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)


# =============================================================================
# Nomad API client (async)
# =============================================================================


class AllocationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    LOST = "lost"


@dataclass
class Allocation:
    id: str
    job_id: str
    task_group: str
    node_id: str
    status: AllocationStatus
    address: Optional[str] = None
    port: Optional[int] = None

    @property
    def http_address(self) -> Optional[str]:
        if self.address and self.port:
            return f"http://{self.address}:{self.port}"
        return None


class NomadClient:
    def __init__(
        self,
        address: str = "http://localhost:4646",
        token: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.address = address.rstrip("/")
        self.token = token or os.environ.get("NOMAD_TOKEN")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {}
            if self.token:
                headers["X-Nomad-Token"] = self.token
            self._session = aiohttp.ClientSession(timeout=self.timeout, headers=headers)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request(self, method: str, path: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        session = await self._get_session()
        url = f"{self.address}{path}"
        async with session.request(method, url, json=data) as resp:
            if resp.status == 404:
                return {"error": "not_found", "status": 404}
            text = await resp.text()
            if not text:
                return {"status": resp.status}
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return {"text": text, "status": resp.status}
            if resp.status >= 400:
                return {"error": parsed, "status": resp.status}
            return parsed if isinstance(parsed, dict) else {"data": parsed, "status": resp.status}

    async def is_healthy(self) -> bool:
        try:
            res = await self._request("GET", "/v1/status/leader")
            return "error" not in res
        except Exception:
            return False

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        res = await self._request("GET", f"/v1/job/{job_id}")
        if res.get("error") == "not_found":
            return None
        if "error" in res:
            raise RuntimeError(f"Nomad get_job failed: {res}")
        return res

    async def submit_job(self, job_spec: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request("POST", "/v1/jobs", data=job_spec)

    async def stop_job(self, job_id: str, purge: bool = True) -> Dict[str, Any]:
        return await self._request("DELETE", f"/v1/job/{job_id}?purge={'true' if purge else 'false'}")

    async def get_job_allocations(self, job_id: str) -> List[Allocation]:
        res = await self._request("GET", f"/v1/job/{job_id}/allocations")
        if "error" in res:
            # Some Nomad builds return list directly, not dict
            if isinstance(res.get("data"), list):
                allocs_data = res["data"]
            else:
                raise RuntimeError(f"Nomad allocations failed: {res}")
        allocs_data = res if isinstance(res, list) else res.get("data", res)
        if not isinstance(allocs_data, list):
            return []

        allocs: List[Allocation] = []
        for a in allocs_data:
            try:
                allocs.append(
                    Allocation(
                        id=a["ID"],
                        job_id=a.get("JobID", job_id),
                        task_group=a.get("TaskGroup", "sandbox"),
                        node_id=a.get("NodeID", ""),
                        status=AllocationStatus(str(a.get("ClientStatus", "pending"))),
                    )
                )
            except Exception:
                continue
        return allocs

    async def get_allocation(self, alloc_id: str) -> Dict[str, Any]:
        res = await self._request("GET", f"/v1/allocation/{alloc_id}")
        if "error" in res:
            raise RuntimeError(f"Nomad get_allocation failed: {res}")
        return res


def create_sandbox_job(
    job_id: str,
    image: str,
    count: int,
    slots_per_container: int,
    privileged: bool,
    cpu: int,
    memory: int,
    port: int = 8080,
    datacenter: str = "dc1",
) -> Dict[str, Any]:
    """Create a basic sandbox-server Nomad job spec (docker driver)."""
    return {
        "ID": job_id,
        "Name": job_id,
        "Type": "service",
        "Datacenters": [datacenter],
        "TaskGroups": [
            {
                "Name": "sandbox",
                "Count": count,
                "Update": {"HealthCheck": "task_states", "MinHealthyTime": 0},
                "Networks": [
                    {
                        "Mode": "host",
                        "DynamicPorts": [{"Label": "http", "To": port}],
                    }
                ],
                "Tasks": [
                    {
                        "Name": "sandbox-server",
                        "Driver": "docker",
                        "Config": {
                            "image": image,
                            "force_pull": False,
                            "ports": ["http"],
                            "privileged": privileged,
                            "command": "python",
                            "args": [
                                "/sandbox_server.py",
                                "--port",
                                str(port),
                                "--slots",
                                str(slots_per_container),
                                "--data-dir",
                                "/data",
                            ],
                        },
                        "Env": {"PYTHONUNBUFFERED": "1", "NOMAD_ALLOC_DIR": "${NOMAD_ALLOC_DIR}"},
                        "Resources": {"CPU": cpu, "MemoryMB": memory},
                    }
                ],
            }
        ],
    }


# =============================================================================
# Slot + executor
# =============================================================================


class SlotState(Enum):
    AVAILABLE = "available"
    ACQUIRED = "acquired"
    EXECUTING = "executing"
    ERROR = "error"


@dataclass
class Slot:
    slot_id: str
    alloc_id: str
    container_addr: str
    workspace_dir: str = ""
    state: SlotState = SlotState.AVAILABLE
    trajectory_id: Optional[str] = None

    def __post_init__(self):
        if not self.workspace_dir:
            self.workspace_dir = f"/data/{self.slot_id}"

    @property
    def is_available(self) -> bool:
        return self.state == SlotState.AVAILABLE

    def acquire(self, trajectory_id: str):
        if not self.is_available:
            raise RuntimeError(f"Slot not available: {self.slot_id} ({self.state})")
        self.state = SlotState.ACQUIRED
        self.trajectory_id = trajectory_id

    def release(self):
        self.state = SlotState.AVAILABLE
        self.trajectory_id = None


def create_slots_for_allocation(alloc_id: str, container_addr: str, num_slots: int) -> List[Slot]:
    return [
        Slot(slot_id=f"slot_{i}", alloc_id=alloc_id, container_addr=container_addr)
        for i in range(num_slots)
    ]


@dataclass
class ExecutionResult:
    success: bool
    output: str = ""
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class SandboxExecutor:
    def __init__(self, timeout: float = 30.0):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def execute(self, slot: Slot, tool_name: str, args: Dict[str, Any], timeout: float) -> ExecutionResult:
        session = await self._get_session()
        url = f"{slot.container_addr}/execute"
        payload = {
            "slot_id": slot.slot_id,
            "tool": tool_name,
            "args": args,
            "execution_id": str(uuid.uuid4()),
            "timeout": timeout,
        }
        async with session.post(url, json=payload) as resp:
            data = await resp.json()
            return ExecutionResult(
                success=bool(data.get("success", False)),
                output=str(data.get("output", "")),
                error=str(data.get("error", "")),
                metadata=dict(data.get("metadata", {}) or {}),
            )

    async def reset_slot(self, slot: Slot) -> ExecutionResult:
        session = await self._get_session()
        url = f"{slot.container_addr}/reset"
        payload = {"slot_id": slot.slot_id}
        async with session.post(url, json=payload) as resp:
            data = await resp.json()
            return ExecutionResult(
                success=bool(data.get("success", False)),
                output=str(data.get("output", "")),
                error=str(data.get("error", "")),
                metadata=dict(data.get("metadata", {}) or {}),
            )


@dataclass
class SlotPoolConfig:
    nomad_address: str = "http://localhost:4646"
    job_id: str = "hermes-sandbox"
    datacenter: str = "dc1"
    image: str = "hermes-sandbox:local"
    slots_per_container: int = 10
    privileged: bool = False
    cpu: int = 500
    memory: int = 512
    min_containers: int = 1
    max_containers: int = 10
    acquire_timeout: float = 30.0


class SlotPool:
    def __init__(self, cfg: SlotPoolConfig):
        self.cfg = cfg
        self.nomad = NomadClient(address=cfg.nomad_address)
        self.executor = SandboxExecutor(timeout=cfg.acquire_timeout)
        self._slots: Dict[str, Slot] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._started = False

    def _slot_key(self, alloc_id: str, slot_id: str) -> str:
        return f"{alloc_id}:{slot_id}"

    async def start(self):
        if self._started:
            return
        if not await self.nomad.is_healthy():
            raise RuntimeError(f"Nomad not reachable at {self.cfg.nomad_address}")
        job = await self.nomad.get_job(self.cfg.job_id)
        if job is None:
            spec = create_sandbox_job(
                job_id=self.cfg.job_id,
                image=self.cfg.image,
                count=self.cfg.min_containers,
                slots_per_container=self.cfg.slots_per_container,
                privileged=self.cfg.privileged,
                cpu=self.cfg.cpu,
                memory=self.cfg.memory,
                datacenter=self.cfg.datacenter,
            )
            res = await self.nomad.submit_job(spec)
            if "error" in res:
                raise RuntimeError(f"Nomad submit job failed: {res}")
        await self._refresh_slots()
        self._started = True

    async def close(self):
        await self.executor.close()
        await self.nomad.close()

    async def _refresh_slots(self):
        allocs = await self.nomad.get_job_allocations(self.cfg.job_id)
        for alloc in allocs:
            detail = await self.nomad.get_allocation(alloc.id)
            # Find the mapped host port for "http".
            addr = detail.get("NodeName") or detail.get("NodeID")
            # Prefer explicit address in allocation resources
            # Nomad stores addresses under Resources->Networks sometimes.
            net = (detail.get("Resources") or {}).get("Networks") or []
            address = None
            port = None
            if net and isinstance(net, list):
                n0 = net[0]
                address = n0.get("IP")
                ports = n0.get("DynamicPorts") or []
                for p in ports:
                    if p.get("Label") == "http":
                        port = p.get("Value")
            if not address or not port:
                # Fall back: allocation has an Address field
                address = detail.get("NodeName") or detail.get("NodeID")
            if not address or not port:
                # Can't use this alloc
                continue
            container_addr = f"http://{address}:{port}"

            for s in create_slots_for_allocation(alloc.id, container_addr, self.cfg.slots_per_container):
                key = self._slot_key(s.alloc_id, s.slot_id)
                if key in self._slots:
                    continue
                self._slots[key] = s
                await self._queue.put(key)

    async def acquire(self, trajectory_id: str) -> Slot:
        if not self._started:
            raise RuntimeError("SlotPool not started")
        while True:
            key = await asyncio.wait_for(self._queue.get(), timeout=self.cfg.acquire_timeout)
            slot = self._slots.get(key)
            if not slot:
                continue
            try:
                slot.acquire(trajectory_id)
                return slot
            except Exception:
                continue

    async def release(self, slot: Slot, reset_workspace: bool = True):
        if reset_workspace:
            try:
                await self.executor.reset_slot(slot)
            except Exception:
                pass
        slot.release()
        await self._queue.put(self._slot_key(slot.alloc_id, slot.slot_id))

    async def execute_bash(self, slot: Slot, command: str, timeout_s: float) -> ExecutionResult:
        return await self.executor.execute(slot, "bash", {"command": command}, timeout=timeout_s)


# =============================================================================
# Sync wrapper (thread + event loop)
# =============================================================================


class NomadSlotPoolManager:
    """Runs a SlotPool on a dedicated event loop thread for sync callers."""

    def __init__(self, cfg: SlotPoolConfig):
        self.cfg = cfg
        self._pool = SlotPool(cfg)
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._started = threading.Event()

    def _run_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        loop.run_until_complete(self._pool.start())
        self._started.set()
        loop.run_forever()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, name="nomad-slotpool", daemon=True)
        self._thread.start()
        self._started.wait(timeout=120)

    def _call(self, coro):
        if not self._loop:
            raise RuntimeError("NomadSlotPoolManager not started")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    def acquire(self, trajectory_id: str) -> Slot:
        return self._call(self._pool.acquire(trajectory_id))

    def release(self, slot: Slot, reset_workspace: bool = True):
        return self._call(self._pool.release(slot, reset_workspace=reset_workspace))

    def execute_bash(self, slot: Slot, command: str, timeout_s: float) -> ExecutionResult:
        return self._call(self._pool.execute_bash(slot, command, timeout_s=timeout_s))


_global_manager: Optional[NomadSlotPoolManager] = None


def get_global_manager(cfg: SlotPoolConfig) -> NomadSlotPoolManager:
    global _global_manager
    if _global_manager is None:
        _global_manager = NomadSlotPoolManager(cfg)
        _global_manager.start()
    return _global_manager

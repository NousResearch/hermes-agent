"""Morph Cloud execution environment with metadata-based workspace reuse."""

import hashlib
import json
import logging
import math
import os
import shlex
import threading
import time
import uuid
from typing import Optional

from tools.environments.base import BaseEnvironment
from tools.interrupt import is_interrupted

logger = logging.getLogger(__name__)


class MorphEnvironment(BaseEnvironment):
    """Morph Cloud instance execution backend.

    Workspace identity is keyed by Hermes' task_id via Morph metadata. In
    persistent mode, cleanup detaches locally and relies on Morph TTL pause +
    wake-on-SSH for warm reattachment on later turns.
    """

    _STATUS_PRIORITY = {
        "ready": 0,
        "paused": 1,
        "pending": 2,
        "saving": 3,
        "error": 4,
    }
    _KIND = "hermes-agent"
    _METADATA_PREFIX = "hermes-agent:"

    def __init__(
        self,
        image_id: str,
        cwd: str = "/root",
        timeout: int = 60,
        cpu: float = 1,
        memory: int = 5120,
        disk: int = 51200,
        persistent_filesystem: bool = True,
        task_id: str = "default",
        lifetime_seconds: int = 300,
    ):
        if not image_id:
            raise ValueError(
                "Morph environment requires a Morph base image identifier "
                "(set TERMINAL_MORPH_IMAGE_ID or terminal.morph_image_id)."
            )

        if cwd == "~":
            cwd = "/root"
        super().__init__(cwd=cwd or "/root", timeout=timeout)

        from morphcloud import MorphCloudClient

        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._image_id = image_id.strip()
        self._cpu = max(1, math.ceil(cpu))
        self._memory = max(256, int(memory))
        self._disk = max(1024, int(disk))
        self._lifetime_seconds = max(0, int(lifetime_seconds))
        self._lock = threading.RLock()
        self._ready_timeout = max(180, int(timeout))
        self._generation = os.getenv("TERMINAL_MORPH_GENERATION", "hermes-morph-v1")
        self._client = MorphCloudClient(
            api_key=os.getenv("MORPH_API_KEY") or None,
            base_url=os.getenv("MORPH_BASE_URL") or None,
            profile=os.getenv("MORPH_PROFILE") or None,
        )
        self._base_snapshot_digest = self._compute_base_snapshot_digest(self._image_id)
        self._workspace_metadata = {
            "kind": self._KIND,
            **self._metadata_fields(
                task_id=task_id,
                resource="workspace",
                backend="morph",
                generation=self._generation,
            ),
        }
        self._instance_metadata = {
            **self._workspace_metadata,
            **self._metadata_fields(
                image_id=self._image_id,
                base_snapshot_digest=self._base_snapshot_digest,
            ),
            "webUIName": self._instance_name(),
        }
        self._base_snapshot_metadata = {
            "kind": self._KIND,
            **self._metadata_fields(
                resource="base-snapshot",
                backend="morph",
                generation=self._generation,
                image_id=self._image_id,
            ),
            "webUIName": self._base_snapshot_name(),
        }
        self._instance = None
        self._base_snapshot = None

        with self._lock:
            self._instance = self._attach_or_create_instance()
            self._apply_runtime_policy()

    def _ttl_action(self) -> str:
        return "pause" if self._persistent else "stop"

    def _ttl_seconds(self) -> Optional[int]:
        return self._lifetime_seconds if self._lifetime_seconds > 0 else None

    def _compute_base_snapshot_digest(self, image_id: str) -> str:
        payload = json.dumps(
            {
                "generation": self._generation,
                "image_id": image_id,
                "vcpus": self._cpu,
                "memory": self._memory,
                "disk_size": self._disk,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"hermes-morph-base:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"

    def _base_snapshot_name(self) -> str:
        return (
            f"Hermes Agent Base {self._image_id} "
            f"({self._cpu} CPU, {self._memory} MB, {self._disk} MB)"
        )

    def _instance_name(self) -> str:
        return f"Hermes Agent {self._task_id}"

    def _metadata_fields(self, **values: str) -> dict[str, str]:
        return {
            f"{self._METADATA_PREFIX}{key}": value
            for key, value in values.items()
        }

    def _materialize_base_snapshot(self):
        if self._base_snapshot is not None:
            return self._base_snapshot

        snapshot = self._client.snapshots.create(
            image_id=self._image_id,
            vcpus=self._cpu,
            memory=self._memory,
            disk_size=self._disk,
            digest=self._base_snapshot_digest,
            metadata=self._base_snapshot_metadata,
        )
        self._ensure_metadata(snapshot, self._base_snapshot_metadata)

        self._base_snapshot = snapshot
        return snapshot

    @staticmethod
    def _ensure_metadata(resource, defaults: dict[str, str]) -> None:
        current = dict(getattr(resource, "metadata", {}) or {})
        merged = dict(current)
        changed = False
        for key, value in defaults.items():
            if merged.get(key) in (None, ""):
                merged[key] = value
                changed = True
        if changed:
            resource.set_metadata(merged)

    @staticmethod
    def _status_name(instance) -> str:
        status = getattr(instance, "status", "")
        return str(getattr(status, "value", status)).lower()

    def _status_sort_key(self, instance) -> tuple[int, int, str]:
        created = int(getattr(instance, "created", 0) or 0)
        status_name = self._status_name(instance)
        return (
            self._STATUS_PRIORITY.get(status_name, len(self._STATUS_PRIORITY)),
            -created,
            str(getattr(instance, "id", "")),
        )

    def _combine_output(self, response) -> str:
        stdout = getattr(response, "stdout", "") or ""
        stderr = getattr(response, "stderr", "") or ""
        if stdout and stderr and not stdout.endswith("\n"):
            return f"{stdout}\n{stderr}"
        return f"{stdout}{stderr}"

    def _lookup_instances(self):
        return list(self._client.instances.list(metadata=self._workspace_metadata))

    def _quarantine_instances(
        self,
        instances: list,
        *,
        reason: str,
        winner_id: str = "",
    ) -> None:
        for instance in instances:
            try:
                metadata = dict(getattr(instance, "metadata", {}) or {})
                metadata.update(
                    {
                        "kind": self._KIND,
                        **self._metadata_fields(
                            resource="workspace-quarantine",
                            backend="morph",
                            task_id=f"quarantined:{self._task_id}:{instance.id}",
                            quarantine_reason=reason,
                            quarantined_task_id=self._task_id,
                            quarantined_at=str(int(time.time())),
                        ),
                    }
                )
                if winner_id:
                    metadata[f"{self._METADATA_PREFIX}quarantine_winner"] = winner_id
                instance.set_metadata(metadata)
                logger.warning(
                    "Morph: quarantined duplicate workspace %s for task %s (%s)",
                    getattr(instance, "id", "<unknown>"),
                    self._task_id,
                    reason,
                )
            except Exception as e:
                logger.warning(
                    "Morph: failed to quarantine instance %s for task %s: %s",
                    getattr(instance, "id", "<unknown>"),
                    self._task_id,
                    e,
                )

    def _select_existing_instance(self, instances: list):
        if not instances:
            return None

        usable = [instance for instance in instances if self._status_name(instance) != "error"]
        if not usable:
            self._quarantine_instances(instances, reason="all_matches_error")
            return None

        ordered = sorted(usable, key=self._status_sort_key)
        winner = ordered[0]
        duplicates = [
            instance
            for instance in instances
            if getattr(instance, "id", None) != winner.id
        ]
        if duplicates:
            self._quarantine_instances(
                duplicates,
                reason="duplicate_task_id",
                winner_id=winner.id,
            )
        return winner

    def _wait_for_ready(self, instance) -> None:
        status = self._status_name(instance)
        if status == "ready":
            return
        if status == "paused":
            instance.resume()
            logger.info("Morph: resumed paused instance %s", instance.id)
        elif status in ("pending", "saving"):
            logger.info("Morph: waiting for instance %s to become ready", instance.id)
        instance.wait_until_ready(timeout=self._ready_timeout)

    def _start_instance(self):
        snapshot = self._materialize_base_snapshot()
        ttl_seconds = self._ttl_seconds()
        instance = self._client.instances.start(
            snapshot_id=snapshot.id,
            metadata=self._instance_metadata,
            ttl_seconds=ttl_seconds,
            ttl_action=self._ttl_action() if ttl_seconds is not None else None,
            timeout=self._ready_timeout,
        )
        self._ensure_metadata(instance, self._instance_metadata)
        logger.info("Morph: started instance %s for task %s", instance.id, self._task_id)
        return instance

    def _attach_or_create_instance(self):
        candidate = self._select_existing_instance(self._lookup_instances())
        if candidate is None:
            return self._start_instance()
        self._wait_for_ready(candidate)
        self._ensure_metadata(candidate, self._instance_metadata)
        logger.info(
            "Morph: attached to existing instance %s for task %s",
            candidate.id,
            self._task_id,
        )
        return candidate

    @staticmethod
    def _is_missing_instance_error(exc: Exception) -> bool:
        return getattr(exc, "status_code", None) == 404

    def _refresh_instance(self):
        if self._instance is None:
            return None

        instance_id = self._instance.id
        try:
            self._instance = self._client.instances.get(instance_id)
        except Exception as exc:
            if self._is_missing_instance_error(exc):
                logger.info(
                    "Morph: instance %s for task %s no longer exists",
                    instance_id,
                    self._task_id,
                )
                self._instance = None
                return None

            logger.warning(
                "Morph: failed to refresh instance %s for task %s: %s",
                instance_id,
                self._task_id,
                exc,
            )
            raise

        return self._instance

    def _ensure_instance_ready(self) -> None:
        instance = self._refresh_instance()
        if instance is None:
            self._instance = self._attach_or_create_instance()
            return

        status = self._status_name(instance)
        if status == "error":
            self._quarantine_instances([instance], reason="selected_instance_error")
            self._instance = self._start_instance()
            return

        self._wait_for_ready(instance)

    def _apply_runtime_policy(self) -> None:
        if self._instance is None:
            return

        ttl_seconds = self._ttl_seconds()
        if ttl_seconds is not None:
            self._instance.set_ttl(ttl_seconds=ttl_seconds, ttl_action=self._ttl_action())

        if self._persistent:
            wake_on = getattr(
                getattr(self._instance, "wake_on", None),
                "wake_on_ssh",
                False,
            )
            if not wake_on:
                self._instance.set_wake_on(wake_on_ssh=True)

    def _build_exec_script(
        self,
        exec_command: str,
        cwd: str,
        timeout: int,
        token: str,
    ) -> str:
        quoted_cwd = shlex.quote(cwd or self.cwd or "/root")
        quoted_command = shlex.quote(exec_command)
        safe_timeout = max(1, int(timeout))
        return (
            "state_dir=/tmp/hermes-morph; "
            "mkdir -p \"$state_dir\"; "
            f"pid_file=\"$state_dir/{token}.pid\"; "
            f"pgid_file=\"$state_dir/{token}.pgid\"; "
            "cleanup() { rm -f \"$pid_file\" \"$pgid_file\"; }; "
            "trap cleanup EXIT; "
            "echo $$ > \"$pid_file\"; "
            "ps -o pgid= -p $$ | tr -d ' ' > \"$pgid_file\"; "
            f"cd {quoted_cwd} || exit 1; "
            f"timeout {safe_timeout} bash -lc {quoted_command}"
        )

    def _interrupt_remote_command(self, token: str) -> None:
        if self._instance is None:
            return

        script = (
            "state_dir=/tmp/hermes-morph; "
            f"pid_file=\"$state_dir/{token}.pid\"; "
            f"pgid_file=\"$state_dir/{token}.pgid\"; "
            "if [ -s \"$pgid_file\" ]; then "
            "kill -- -$(cat \"$pgid_file\") 2>/dev/null || true; "
            "fi; "
            "if [ -s \"$pid_file\" ]; then "
            "kill $(cat \"$pid_file\") 2>/dev/null || true; "
            "fi; "
            "rm -f \"$pid_file\" \"$pgid_file\""
        )
        try:
            self._instance.exec(["bash", "-lc", script], timeout=10)
        except Exception as e:
            logger.debug(
                "Morph: remote interrupt helper failed for task %s: %s",
                self._task_id,
                e,
            )

    def execute(
        self,
        command: str,
        cwd: str = "",
        *,
        timeout: Optional[int] = None,
        stdin_data: Optional[str] = None,
    ) -> dict:
        if stdin_data is not None:
            marker = f"HERMES_EOF_{uuid.uuid4().hex[:8]}"
            while marker in stdin_data:
                marker = f"HERMES_EOF_{uuid.uuid4().hex[:8]}"
            command = f"{command} << '{marker}'\n{stdin_data}\n{marker}"

        exec_command, sudo_stdin = self._prepare_command(command)
        if sudo_stdin is not None:
            exec_command = (
                f"printf '%s\\n' {shlex.quote(sudo_stdin.rstrip())} | {exec_command}"
            )

        effective_cwd = cwd or self.cwd or "/root"
        effective_timeout = max(1, int(timeout or self.timeout))
        command_token = uuid.uuid4().hex[:12]
        remote_script = self._build_exec_script(
            exec_command,
            effective_cwd,
            effective_timeout,
            command_token,
        )

        result_holder = {"response": None, "error": None}

        def _run() -> None:
            try:
                with self._lock:
                    self._ensure_instance_ready()
                    instance = self._instance
                result_holder["response"] = instance.exec(
                    ["bash", "-lc", remote_script],
                    timeout=effective_timeout + 20,
                )
            except Exception as e:
                result_holder["error"] = e

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        deadline = time.monotonic() + effective_timeout + 25
        while thread.is_alive():
            thread.join(timeout=0.2)
            if is_interrupted():
                self._interrupt_remote_command(command_token)
                thread.join(timeout=2)
                return {
                    "output": "[Command interrupted - Morph workspace left available for reuse]",
                    "returncode": 130,
                }
            if time.monotonic() > deadline:
                self._interrupt_remote_command(command_token)
                thread.join(timeout=2)
                return self._timeout_result(effective_timeout)

        if result_holder["error"] is not None:
            err = result_holder["error"]
            if isinstance(err, TimeoutError):
                return self._timeout_result(effective_timeout)
            return {"output": f"Morph execution error: {err}", "returncode": 1}

        response = result_holder["response"]
        try:
            with self._lock:
                self._refresh_instance()
                self._apply_runtime_policy()
        except Exception as e:
            logger.warning(
                "Morph: failed to refresh TTL/wake settings for task %s: %s",
                self._task_id,
                e,
            )

        return {
            "output": self._combine_output(response),
            "returncode": getattr(response, "exit_code", 1),
        }

    def cleanup(self):
        with self._lock:
            if self._instance is None:
                return

            try:
                if self._persistent:
                    self._refresh_instance()
                    if self._instance is not None:
                        self._apply_runtime_policy()
                        logger.info(
                            "Morph: detached from persistent instance %s for task %s",
                            self._instance.id,
                            self._task_id,
                        )
                else:
                    self._instance.stop()
                    logger.info(
                        "Morph: stopped instance %s for task %s",
                        self._instance.id,
                        self._task_id,
                    )
                    self._instance = None
            except Exception as e:
                logger.warning("Morph: cleanup failed for task %s: %s", self._task_id, e)

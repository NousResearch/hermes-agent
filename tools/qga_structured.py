"""Structured QGA adapter for governed destructive operations.

The generic QGA path (for example, SSH to a Proxmox host followed by
``qm guest exec``) is powerful and must never be exposed to the model for
destructive actions. This module exposes a narrow, policy-aware adapter:

    qga_create_filesystem(vm_id, device, fs_type, label)

The guest argv is built here from allowlisted fields only:

* ``fs_type`` -> trusted binary via ``destructive_grants.FS_TYPE_TO_BINARY``;
* ``device``  -> validated by ``destructive_grants.validate_device``;
* ``label``   -> validated by ``destructive_grants.validate_label``.

No model-provided shell command is executed. Read-only probes and the final
formatter use structured argv lists passed through ``qm guest exec``.

Transport is deployment-configured and fail-closed. ``HERMES_QGA_NODE`` and
``HERMES_QGA_SSH_KEY`` must be set explicitly; no project- or operator-specific
host, key path, or permissive host-key policy is compiled into Hermes.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
from typing import Dict, List, Optional, TypedDict

from tools.destructive_grants import (
    FS_TYPE_TO_BINARY,
    GrantError,
    validate_device,
    validate_fs_type,
    validate_label,
    validate_vm_id,
)

logger = logging.getLogger(__name__)


class RemoteResult(TypedDict):
    exit_code: int
    stdout: str
    stderr: str


class QgaExecResult(TypedDict):
    exit_code: int
    out_data: str
    err_data: str


# Deployment-specific values are intentionally not given functional defaults.
# A governed destructive path must never guess its control-plane target or SSH
# credential. Tests may override these values or mock the remote transport.
DEFAULT_QGA_NODE = os.environ.get("HERMES_QGA_NODE", "").strip()
DEFAULT_QGA_SSH_KEY = os.environ.get("HERMES_QGA_SSH_KEY", "").strip()
DEFAULT_QGA_SSH_USER = os.environ.get("HERMES_QGA_SSH_USER", "root").strip() or "root"
DEFAULT_QGA_KNOWN_HOSTS = os.environ.get("HERMES_QGA_KNOWN_HOSTS", "").strip()
DEFAULT_QGA_TIMEOUT = int(os.environ.get("HERMES_QGA_TIMEOUT", "60"))


class QgaError(Exception):
    """Structured QGA failure."""


def _ssh_base() -> List[str]:
    """Return the deployment-configured SSH argv or fail closed.

    Host-key verification is mandatory. Deployments may point
    ``HERMES_QGA_KNOWN_HOSTS`` at a dedicated known_hosts file; otherwise the
    OpenSSH default known_hosts policy is used.
    """
    if not DEFAULT_QGA_NODE:
        raise QgaError("HERMES_QGA_NODE is required for structured QGA")
    if not DEFAULT_QGA_SSH_KEY:
        raise QgaError("HERMES_QGA_SSH_KEY is required for structured QGA")

    cmd = [
        "ssh",
        "-i",
        DEFAULT_QGA_SSH_KEY,
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ConnectTimeout=8",
    ]
    if DEFAULT_QGA_KNOWN_HOSTS:
        cmd += ["-o", f"UserKnownHostsFile={DEFAULT_QGA_KNOWN_HOSTS}"]
    cmd.append(f"{DEFAULT_QGA_SSH_USER}@{DEFAULT_QGA_NODE}")
    return cmd


def _run_remote(argv: List[str], timeout: int = DEFAULT_QGA_TIMEOUT) -> RemoteResult:
    """Run a fixed argv list on the configured Proxmox node.

    The SSH remote-command transport necessarily serializes argv to text for
    OpenSSH, so each element is shell-quoted here. Model-controlled fields have
    already passed strict validators before reaching this boundary.
    """
    remote_cmd = " ".join(shlex.quote(a) for a in argv)
    cmd = _ssh_base() + [remote_cmd]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise QgaError(f"QGA remote command timed out after {timeout}s") from exc
    except OSError as exc:
        raise QgaError(f"QGA ssh failed: {exc}") from exc
    return {
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _qm_guest_exec(
    vm_id: str,
    guest_argv: List[str],
    timeout: int = DEFAULT_QGA_TIMEOUT,
) -> QgaExecResult:
    """Run ``qm guest exec <vm_id> -- <guest_argv>`` on the configured node."""
    argv = ["qm", "guest", "exec", vm_id, "--"] + guest_argv
    result = _run_remote(argv, timeout=timeout)
    if result["exit_code"] != 0:
        raise QgaError(
            f"qm guest exec failed (rc={result['exit_code']}): "
            f"{result['stderr'].strip() or result['stdout'].strip()}"
        )
    try:
        envelope = json.loads(result["stdout"])
    except json.JSONDecodeError as exc:
        raise QgaError(
            f"qm guest exec returned non-JSON output: {result['stdout'][:500]}"
        ) from exc
    ret = envelope.get("return", envelope)
    exit_code = ret.get("exit-code", ret.get("exitcode", -1))
    return {
        "exit_code": int(exit_code),
        "out_data": str(ret.get("out-data", "")),
        "err_data": str(ret.get("err-data", "")),
    }


def qga_guest_identity(vm_id: str) -> Dict[str, object]:
    """Read-only: guest hostname + boot id + product UUID via QGA.

    ``product_uuid`` is the stable guest identity and ``boot_id`` is the boot
    generation. Together they fence a grant to the exact guest incarnation.
    """
    validate_vm_id(vm_id)
    hostname = _qm_guest_exec(vm_id, ["hostname"])
    bootid = _qm_guest_exec(vm_id, ["cat", "/proc/sys/kernel/random/boot_id"])
    product_uuid = _qm_guest_exec(
        vm_id, ["cat", "/sys/class/dmi/id/product_uuid"]
    )
    return {
        "hostname": hostname["out_data"].strip(),
        "boot_id": bootid["out_data"].strip(),
        "product_uuid": product_uuid["out_data"].strip(),
        "hostname_rc": hostname["exit_code"],
        "boot_id_rc": bootid["exit_code"],
        "product_uuid_rc": product_uuid["exit_code"],
    }


def qga_prechecks(vm_id: str, device: str) -> Dict[str, object]:
    """Read-only prechecks executed inside the guest just before action."""
    validate_vm_id(vm_id)
    validate_device(device)

    checks: Dict[str, object] = {
        "vm_id": vm_id,
        "device": device,
        "device_exists": False,
        "is_block_device": False,
        "major_minor": None,
        "size_bytes": None,
        "parent": None,
        "mounted": False,
        "swap": False,
        "filesystem_existing": False,
        "filesystem_signature": False,
        "lvm_member": False,
        "mdraid_member": False,
        "holders": [],
        "docker_use": False,
        "fstab_use": False,
        "boot_id": None,
    }

    bootid = _qm_guest_exec(vm_id, ["cat", "/proc/sys/kernel/random/boot_id"])
    if bootid["exit_code"] == 0 and bootid["out_data"].strip():
        checks["boot_id"] = bootid["out_data"].strip()

    lsblk = _qm_guest_exec(
        vm_id,
        ["lsblk", "-J", "-o", "NAME,MAJ:MIN,SIZE,TYPE,MOUNTPOINTS,FSTYPE", device],
    )
    if lsblk["exit_code"] == 0:
        try:
            data = json.loads(lsblk["out_data"])
            blocks = data.get("blockdevices", [])
            if blocks:
                b = blocks[0]
                checks["device_exists"] = True
                checks["is_block_device"] = b.get("type") in {"part", "loop"}
                checks["major_minor"] = b.get("maj:min")
                checks["size_bytes"] = _parse_lsblk_size(b.get("size", ""))
                checks["parent"] = b.get("name")
                checks["mounted"] = bool(
                    b.get("mountpoints") and any(b.get("mountpoints"))
                )
                checks["filesystem_existing"] = bool(b.get("fstype"))
        except (json.JSONDecodeError, KeyError, IndexError):
            checks["lsblk_parse_error"] = lsblk["out_data"][:300]

    findmnt = _qm_guest_exec(vm_id, ["findmnt", "-n", "-o", "TARGET", device])
    if findmnt["exit_code"] == 0 and findmnt["out_data"].strip():
        checks["mounted"] = True
        checks["mount_target"] = findmnt["out_data"].strip()

    # PARTUUID is partition-table identity, not a data signature. A real TYPE
    # or wipefs signature is what blocks formatting.
    blkid = _qm_guest_exec(vm_id, ["blkid", device])
    if blkid["exit_code"] == 0 and blkid["out_data"].strip():
        checks["blkid_output"] = blkid["out_data"].strip()[:300]
        if "TYPE=" in blkid["out_data"]:
            checks["filesystem_signature"] = True

    wipefs = _qm_guest_exec(vm_id, ["wipefs", "-n", device])
    if wipefs["exit_code"] == 0 and wipefs["out_data"].strip():
        checks["filesystem_signature"] = True
        checks["wipefs_output"] = wipefs["out_data"].strip()[:300]

    swaps = _qm_guest_exec(vm_id, ["grep", "-w", device, "/proc/swaps"])
    if swaps["exit_code"] == 0 and swaps["out_data"].strip():
        checks["swap"] = True

    pvs = _qm_guest_exec(vm_id, ["pvs", "--noheadings", "-o", "pv_name"])
    if pvs["exit_code"] == 0 and device in pvs["out_data"]:
        checks["lvm_member"] = True

    mdstat = _qm_guest_exec(vm_id, ["cat", "/proc/mdstat"])
    if mdstat["exit_code"] == 0 and device in mdstat["out_data"]:
        checks["mdraid_member"] = True

    basename = device.rsplit("/", 1)[-1]
    holders = _qm_guest_exec(
        vm_id, ["ls", "-1", f"/sys/class/block/{basename}/holders"]
    )
    if holders["exit_code"] == 0:
        checks["holders"] = [
            value for value in holders["out_data"].splitlines() if value.strip()
        ]

    # Docker-use check remains structured: enumerate containers, inspect each
    # one, and evaluate mount sources locally instead of invoking a guest shell.
    docker_ps = _qm_guest_exec(vm_id, ["docker", "ps", "-q"])
    if docker_ps["exit_code"] == 0:
        container_ids = [
            value.strip()
            for value in docker_ps["out_data"].splitlines()
            if value.strip()
        ]
        for container_id in container_ids:
            inspect = _qm_guest_exec(
                vm_id,
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{range .Mounts}}{{println .Source}}{{end}}",
                    container_id,
                ],
            )
            if inspect["exit_code"] == 0:
                sources = {
                    value.strip()
                    for value in inspect["out_data"].splitlines()
                    if value.strip()
                }
                if device in sources:
                    checks["docker_use"] = True
                    break

    fstab = _qm_guest_exec(vm_id, ["grep", "-F", device, "/etc/fstab"])
    if fstab["exit_code"] == 0 and fstab["out_data"].strip():
        checks["fstab_use"] = True

    return checks


def _parse_lsblk_size(size_raw: str) -> Optional[int]:
    """Parse lsblk SIZE like '128G' / '512M' / '1T' into bytes."""
    if not size_raw:
        return None
    size_raw = size_raw.strip()
    try:
        if size_raw.endswith("B"):
            return int(float(size_raw[:-1]))
        if size_raw.endswith("K"):
            return int(float(size_raw[:-1]) * 1024)
        if size_raw.endswith("M"):
            return int(float(size_raw[:-1]) * 1024**2)
        if size_raw.endswith("G"):
            return int(float(size_raw[:-1]) * 1024**3)
        if size_raw.endswith("T"):
            return int(float(size_raw[:-1]) * 1024**4)
        return int(float(size_raw))
    except ValueError:
        return None


def qga_create_filesystem(
    vm_id: str,
    device: str,
    fs_type: str,
    label: str,
    timeout: int = DEFAULT_QGA_TIMEOUT,
) -> Dict[str, object]:
    """Governed filesystem creation via structured QGA."""
    validate_vm_id(vm_id)
    validate_device(device)
    validate_fs_type(fs_type)
    validate_label(label)

    binary = FS_TYPE_TO_BINARY[fs_type]
    guest_argv = [binary, "-L", label, device]

    result = _qm_guest_exec(vm_id, guest_argv, timeout=timeout)
    return {
        "operation": "CREATE_FILESYSTEM",
        "vm_id": vm_id,
        "device": device,
        "fs_type": fs_type,
        "label": label,
        "guest_argv": guest_argv,
        "exit_code": result["exit_code"],
        "out_data": result["out_data"],
        "err_data": result["err_data"],
    }


def qga_postcheck(
    vm_id: str,
    device: str,
    fs_type: str,
    label: str,
) -> Dict[str, object]:
    """Read-only postcheck: filesystem type + label + UUID after creation."""
    validate_vm_id(vm_id)
    validate_device(device)
    validate_fs_type(fs_type)
    validate_label(label)

    blkid = _qm_guest_exec(vm_id, ["blkid", "-o", "export", device])
    out = blkid["out_data"]
    result: Dict[str, object] = {
        "exit_code": blkid["exit_code"],
        "filesystem": None,
        "label": None,
        "uuid": None,
    }
    for line in out.splitlines():
        if line.startswith("TYPE="):
            result["filesystem"] = line.split("=", 1)[1].strip()
        elif line.startswith("LABEL="):
            result["label"] = line.split("=", 1)[1].strip()
        elif line.startswith("UUID="):
            result["uuid"] = line.split("=", 1)[1].strip()
    return result

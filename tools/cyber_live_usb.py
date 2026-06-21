"""Hermes AgentCyber — Live USB Tool.

Allows the agent to build, write, provision, and manage Hermes live USB
drives from within an agent session. All heavy operations are shelled out to
the scripts in live-usb/ so the logic stays in bash (easier to audit and
modify without restarting the agent).

Actions:
  build       — Build a bootable ISO (requires root + debootstrap on host)
  write       — Write an ISO to a USB drive (requires root)
  provision   — Inject config into an already-written USB
  list_usb    — List removable block devices (safe, no root needed)
  status      — Show ISO build status / available ISOs

This tool is intentionally NOT in the default toolset — add 'live_usb' to
enabled_toolsets in config.yaml, or pass --toolsets live_usb when building
a USB-management session.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Path to the live-usb/ scripts directory, relative to this file
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "live-usb"
_APPROVAL_ENV_VAR = "HERMES_AGENTCYBER_LIVE_USB_APPROVAL"
_APPROVAL_ARG_KEYS = ("operator_approval", "approval_token", "live_usb_approval")


def _running_as_root() -> bool:
    """Return True when the current platform exposes geteuid and UID is root."""
    return hasattr(os, "geteuid") and os.geteuid() == 0


def _script(name: str) -> str:
    """Return absolute path to a live-usb/ script, verified to exist."""
    path = _SCRIPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Live-USB script not found: {path}")
    return str(path)


def _run(cmd: list[str], timeout: int = 300) -> dict:
    """Run a command, stream output to logger, return {rc, stdout, stderr}."""
    logger.info("live_usb: running %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "rc":     result.returncode,
            "stdout": result.stdout[-4000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"rc": -1, "stdout": "", "stderr": f"Command timed out after {timeout}s"}
    except Exception as exc:
        return {"rc": -1, "stdout": "", "stderr": str(exc)}


def _operator_approval_error(action: str, reason: str) -> dict:
    """Return a secret-safe approval error for high-consequence actions."""
    return {
        "error": f"{action} requires explicit operator approval.",
        "reason": reason,
        "approved": False,
        "hint": (
            f"Set {_APPROVAL_ENV_VAR} in the standalone AgentCyber environment "
            "and pass the matching value as operator_approval for this exact "
            "operator-approved live USB operation. status and list_usb do not "
            "require approval. Never print the approval value."
        ),
    }


def _require_operator_approval(args: dict, action: str) -> dict | None:
    """Fail closed unless a high-consequence action has operator approval.

    Root alone is not enough for build/write/provision: an unattended agent run
    could otherwise build media or write a block device as soon as the toolset is
    enabled. The approval token lives in the operator-controlled AgentCyber
    environment; it is compared without logging or returning the secret.
    """
    expected = os.getenv(_APPROVAL_ENV_VAR, "").strip()
    if not expected:
        return _operator_approval_error(action, f"missing {_APPROVAL_ENV_VAR}")

    provided = ""
    for key in _APPROVAL_ARG_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            provided = value.strip()
            break
    if not provided:
        return _operator_approval_error(action, "missing operator_approval")
    if not hmac.compare_digest(provided, expected):
        return _operator_approval_error(action, "operator_approval did not match")
    return None


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

def _list_usb(args: dict = None, **_kw: Any) -> dict:
    """List removable block devices. Safe, no root needed."""
    if not shutil.which("lsblk"):
        return {"error": "lsblk not found — are you running on Linux?"}

    result = subprocess.run(
        ["lsblk", "-d", "-o", "NAME,SIZE,TRAN,MODEL,RM,VENDOR", "--sort", "NAME", "--json"],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0:
        return {"error": result.stderr.strip()}

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"raw": result.stdout}

    devices = [
        {
            "device": f"/dev/{d['name']}",
            "size":   d.get("size"),
            "transport": d.get("tran"),
            "model":  (d.get("model") or "").strip(),
            "removable": d.get("rm") == "1" or d.get("rm") is True,
            "vendor": (d.get("vendor") or "").strip(),
        }
        for d in data.get("blockdevices", [])
    ]
    removable = [d for d in devices if d["removable"]]
    return {
        "removable_devices": removable,
        "all_block_devices": devices,
        "note": "Only 'removable' devices are suitable USB targets.",
    }


def _status(args: dict = None, **_kw: Any) -> dict:
    """Show available ISOs and build dependencies."""
    isos = list(_SCRIPTS_DIR.glob("*.iso"))
    iso_info = [
        {
            "path":    str(p),
            "size_mb": round(p.stat().st_size / 1024 / 1024, 1),
            "mtime":   time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(p.stat().st_mtime)),
        }
        for p in sorted(isos)
    ]

    deps = {
        cmd: bool(shutil.which(cmd))
        for cmd in ["debootstrap", "mksquashfs", "xorriso", "dd", "lsblk", "pv", "parted"]
    }
    missing_build = [k for k, v in deps.items() if not v and k in ("debootstrap", "mksquashfs", "xorriso")]
    missing_write = [k for k, v in deps.items() if not v and k in ("dd",)]

    return {
        "available_isos":       iso_info,
        "build_dependencies":   deps,
        "can_build":            len(missing_build) == 0,
        "can_write":            len(missing_write) == 0,
        "missing_for_build":    missing_build,
        "missing_for_write":    missing_write,
        "scripts_dir":          str(_SCRIPTS_DIR),
        "install_deps_command": (
            "sudo apt-get install -y debootstrap squashfs-tools xorriso "
            "grub-efi-amd64-bin grub-pc-bin mtools dosfstools pv"
        ) if missing_build else None,
    }


def _build(args: dict, **_kw: Any) -> dict:
    """Trigger the ISO build. Requires root and build deps on the host."""
    if not _running_as_root():
        return {
            "error": "Building an ISO requires root.",
            "hint":  "Run the agent as root, or run the build script directly: "
                     "sudo live-usb/build_iso.sh",
        }

    approval_error = _require_operator_approval(args, "build")
    if approval_error:
        return approval_error

    script = _script("build_iso.sh")
    cmd = ["bash", script]

    if args.get("arch"):         cmd += ["--arch", args["arch"]]
    if args.get("suite"):        cmd += ["--suite", args["suite"]]
    if args.get("kali_meta"):    cmd += ["--kali-meta", args["kali_meta"]]
    if args.get("output"):       cmd += ["--output", args["output"]]
    if args.get("source_dir"):   cmd += ["--source-dir", args["source_dir"]]
    if args.get("headless_scan"): cmd += ["--headless-scan"]
    if args.get("verbose"):      cmd += ["--verbose"]

    output_path = args.get("output", str(_SCRIPTS_DIR / "hermes-cyber-live.iso"))
    result = _run(cmd, timeout=int(args.get("timeout", 1800)))  # 30 min default

    if result["rc"] == 0:
        iso_size = Path(output_path).stat().st_size if Path(output_path).exists() else 0
        return {
            "success": True,
            "iso": output_path,
            "size_mb": round(iso_size / 1024 / 1024, 1),
            "output": result["stdout"][-2000:],
        }
    return {
        "success": False,
        "rc": result["rc"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
    }


def _write(args: dict, **_kw: Any) -> dict:
    """Write an ISO to a USB drive. Requires root."""
    if not _running_as_root():
        return {
            "error": "Writing to a block device requires root.",
            "hint":  "Run: sudo live-usb/write_usb.sh --iso <path> --device <dev> --yes",
        }

    approval_error = _require_operator_approval(args, "write")
    if approval_error:
        return approval_error

    device = args.get("device", "")
    if not device:
        return {"error": "device is required (e.g. /dev/sdb). Use list_usb to find it."}
    if not Path(device).is_block_device():
        return {"error": f"Not a block device: {device}"}

    iso = args.get("iso", str(_SCRIPTS_DIR / "hermes-cyber-live.iso"))
    if not Path(iso).exists():
        return {"error": f"ISO not found: {iso}. Build it first with the build action."}

    script = _script("write_usb.sh")
    cmd = ["bash", script, "--iso", iso, "--device", device, "--yes"]

    if args.get("provision"):  cmd += ["--provision", args["provision"]]
    if args.get("verify"):     cmd += ["--verify"]

    result = _run(cmd, timeout=int(args.get("timeout", 600)))  # 10 min default

    return {
        "success": result["rc"] == 0,
        "rc":      result["rc"],
        "output":  result["stdout"],
        "stderr":  result["stderr"],
    }


def _provision(args: dict, **_kw: Any) -> dict:
    """Inject config into an already-written USB."""
    if not _running_as_root():
        return {"error": "Provisioning requires root (needs to mount the USB partition)."}

    approval_error = _require_operator_approval(args, "provision")
    if approval_error:
        return approval_error

    device = args.get("device", "")
    if not device:
        return {"error": "device is required"}

    script = _script("provision.sh")
    cmd = ["bash", script, "--usb", device]

    if args.get("config"):          cmd += ["--config", args["config"]]
    if args.get("telegram_token"):  cmd += ["--telegram-token", args["telegram_token"]]
    if args.get("allowed_users"):   cmd += ["--allowed-users", args["allowed_users"]]
    if args.get("model_key"):       cmd += ["--model-key", args["model_key"]]
    if args.get("model_provider"):  cmd += ["--model-provider", args["model_provider"]]
    if args.get("audit"):           cmd += ["--audit"]

    result = _run(cmd, timeout=60)
    return {
        "success": result["rc"] == 0,
        "output":  result["stdout"],
        "stderr":  result["stderr"],
    }


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def _handle(args: dict, **kw: Any) -> str:
    action = args.get("action", "")
    dispatch = {
        "build":    _build,
        "write":    _write,
        "provision": _provision,
        "list_usb": _list_usb,
        "status":   _status,
    }
    fn = dispatch.get(action)
    if fn is None:
        return json.dumps({
            "error": f"Unknown action: {action!r}",
            "valid_actions": list(dispatch),
        })
    try:
        result = fn(args, **kw)
    except Exception as exc:
        logger.exception("live_usb tool error")
        result = {"error": str(exc)}
    return json.dumps(result, indent=2)


SCHEMA = {
    "type": "function",
    "function": {
        "name": "live_usb",
        "description": (
            "Build, write, and provision Hermes AgentCyber live USB drives. "
            "Produces a bootable USB that auto-starts the Hermes gateway (or "
            "interactive shell / headless scan) when plugged into any PC. "
            "list_usb and status are safe; build, write, and provision require "
            "root plus operator approval."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["build", "write", "provision", "list_usb", "status"],
                    "description": "Action to perform.",
                },
                # build
                "arch":          {"type": "string", "description": "Target CPU arch (build): amd64, arm64."},
                "suite":         {"type": "string", "description": "OS suite (build): kali-rolling (default), bookworm."},
                "kali_meta":     {"type": "string", "description": "Kali metapackage (build): kali-tools-top10, kali-linux-headless (default), kali-linux-default."},
                "output":        {"type": "string", "description": "Output ISO path (build)."},
                "source_dir":    {"type": "string", "description": "Path to hermes-agentcyber source tree (build)."},
                "headless_scan": {"type": "boolean", "description": "Enable auto-scan on boot (build)."},
                "verbose":       {"type": "boolean", "description": "Verbose build output."},
                # write
                "device":        {"type": "string", "description": "Target block device (write/provision), e.g. /dev/sdb."},
                "iso":           {"type": "string", "description": "ISO path to write (write)."},
                "verify":        {"type": "boolean", "description": "SHA-256 verify after write."},
                # provision
                "config":        {"type": "string", "description": "Path to .hermes config dir or .tar.gz (provision)."},
                "telegram_token":{"type": "string", "description": "Telegram bot token (provision)."},
                "allowed_users": {"type": "string", "description": "Comma-separated Telegram user IDs (provision)."},
                "model_key":     {"type": "string", "description": "AI model API key (provision)."},
                "model_provider":{"type": "string", "description": "Model provider: anthropic, openai, openrouter (provision)."},
                "audit":         {"type": "boolean", "description": "Enable SOC audit log on the USB (provision)."},
                # shared
                "operator_approval": {"type": "string", "description": "Operator-provided live USB approval token required for build/write/provision. Never required for status/list_usb; never print this value."},
                "timeout":       {"type": "integer", "description": "Operation timeout in seconds (build: 1800, write: 600)."},
            },
            "required": ["action"],
        },
    },
}

from tools.registry import registry  # noqa: E402

registry.register(
    name="live_usb",
    toolset="live_usb",
    schema=SCHEMA,
    handler=_handle,
    emoji="💾",
)

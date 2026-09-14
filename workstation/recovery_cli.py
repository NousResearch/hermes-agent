"""Tiny dependency-light Recovery Plane command surface for Windows and CI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from workstation.supervisor import RecoveryPlane


def _default_recovery_path() -> Path:
    return get_hermes_home() / "workstation" / "recovery-plane.json"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hermes-workstation-recovery")
    parser.add_argument("--recovery-state", type=Path, default=_default_recovery_path())
    parser.add_argument("--supervisor-state", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status")
    quarantine = subparsers.add_parser("quarantine")
    quarantine.add_argument("component")
    quarantine.add_argument("reason")
    restore = subparsers.add_parser("restore")
    restore.add_argument("component")

    args = parser.parse_args(argv)
    plane = RecoveryPlane(args.recovery_state)
    if args.command == "quarantine":
        plane.quarantine(args.component, args.reason)
    elif args.command == "restore":
        if not plane.restore(args.component):
            parser.error(f"component is not quarantined: {args.component}")

    report: dict[str, Any] = {"components": plane.snapshot()}
    if args.supervisor_state is not None:
        report["supervisor"] = _read_json(args.supervisor_state)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

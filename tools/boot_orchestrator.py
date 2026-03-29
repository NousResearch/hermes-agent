#!/usr/bin/env python3
"""Boot Orchestrator CLI

Reads boot order files from runtime/boot_order.d/*.yaml, validates presence and basic schema,
and emits JSON status. Supports --check to run validations and --path to specify directory.
"""
import argparse
import json
import os
import sys
from glob import glob

try:
    import yaml
except Exception:
    yaml = None

REQUIRED_KEYS = {"name", "steps"}


def load_yaml(path):
    if yaml is None:
        raise RuntimeError("PyYAML is required to parse YAML files. Install pyyaml in the environment.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_doc(doc):
    if not isinstance(doc, dict):
        return False, "document is not a mapping"
    missing = REQUIRED_KEYS - set(doc.keys())
    if missing:
        return False, f"missing keys: {', '.join(sorted(missing))}"
    if not isinstance(doc.get("name"), str):
        return False, "name must be a string"
    if not isinstance(doc.get("steps"), list):
        return False, "steps must be a list"
    return True, None


def run_check(path):
    pattern = os.path.join(path, "boot_order.d", "*.yaml")
    files = sorted(glob(pattern))
    status = {
        "path": path,
        "found_files": files,
        "valid": True,
        "errors": [],
        "files_report": {}
    }
    if not files:
        status["valid"] = False
        status["errors"].append("no boot order files found")
        return status

    for p in files:
        try:
            doc = load_yaml(p)
        except Exception as e:
            status["valid"] = False
            status["errors"].append(f"{os.path.basename(p)}: parse error: {e}")
            status["files_report"][os.path.basename(p)] = {"valid": False, "error": str(e)}
            continue
        ok, err = validate_doc(doc)
        status["files_report"][os.path.basename(p)] = {"valid": ok, "error": err}
        if not ok:
            status["valid"] = False
            status["errors"].append(f"{os.path.basename(p)}: {err}")
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description="Boot Orchestrator: validate boot order files")
    parser.add_argument("--check", action="store_true", help="Run validations and emit JSON status")
    parser.add_argument("--path", default="runtime", help="Root runtime path (default: runtime)")
    args = parser.parse_args(argv)

    if args.check:
        status = run_check(args.path)
        print(json.dumps(status, indent=2))
        if not status["valid"]:
            return 2
        return 0
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())

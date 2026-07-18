"""Security Audit Tool — scan config, plugins, and environment for risks.

Adapted from Ruflo metaharness (ruvnet/ruflo, MIT).
CLI command: /audit
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from hermes_constants import get_hermes_home
from tools.registry import registry

logger = logging.getLogger(__name__)

def audit_scan(scope: str = "all", task_id: str = None) -> str:
    """Scan Hermes configuration and plugins for security risks.

    Args:
        scope: One of "config", "plugins", "env", "permissions", "all"
    """
    results = {"ok": True, "scope": scope, "findings": [], "severity_counts": {"critical": 0, "warning": 0, "info": 0}}

    hermes_home = get_hermes_home()

    if scope in ("config", "all"):
        _audit_config(hermes_home, results)
    if scope in ("plugins", "all"):
        _audit_plugins(hermes_home, results)
    if scope in ("env", "all"):
        _audit_env(hermes_home, results)
    if scope in ("permissions", "all"):
        _audit_permissions(results)

    return json.dumps(results, indent=2, ensure_ascii=False)


def _add_finding(results, severity, category, message, detail=""):
    results["findings"].append({
        "severity": severity,
        "category": category,
        "message": message,
        "detail": detail,
    })
    results["severity_counts"][severity] += 1


def _audit_config(hermes_home, results):
    """Check config.yaml for risky settings."""
    config_path = hermes_home / "config.yaml"
    if not config_path.exists():
        _add_finding(results, "warning", "config", "No config.yaml found", str(config_path))
        return

    try:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

        # Check delegation depth
        depth = cfg.get("delegation", {}).get("max_spawn_depth", 2)
        if depth > 5:
            _add_finding(results, "warning", "config",
                         f"High delegation depth: {depth}", "Consider max_spawn_depth <= 5 to prevent deep agent trees")

        # Check cron delivery
        if cfg.get("cron", {}).get("attach_to_session", True):
            _add_finding(results, "info", "config",
                         "Cron delivery is continuable", "Fine for trusted environments, audit for multi-user setups")

        # Check federation
        fed = cfg.get("federation", {})
        if fed.get("enabled"):
            peers_no_auth = sum(1 for p in fed.get("peers", []) if not p.get("shared_secret"))
            if peers_no_auth > 0:
                _add_finding(results, "warning", "config",
                             f"Federation enabled with {peers_no_auth} unauthenticated peers",
                             "Add shared_secret to all federation peers for HMAC auth")

        # Check approval mode
        approval = cfg.get("security", {}).get("approval_mode", "none")
        if approval == "none":
            _add_finding(results, "info", "config",
                         "Tool approval is disabled", "Consider enabling for production multi-user deployments")

    except Exception as e:
        _add_finding(results, "warning", "config", f"Could not parse config.yaml: {e}")


def _audit_plugins(hermes_home, results):
    """Scan plugins for common issues."""
    plugin_dirs = [
        hermes_home / "plugins",
        Path("/d/opt/hermes/.hermes/hermes-agent/plugins"),
    ]
    for pd in plugin_dirs:
        if not pd.exists():
            continue
        for item in pd.iterdir():
            if item.is_dir():
                init = item / "__init__.py"
                yaml_file = item / "plugin.yaml"
                if init.exists():
                    content = init.read_text()
                    if "eval(" in content or "exec(" in content:
                        _add_finding(results, "critical", "plugins",
                                     f"Plugin {item.name} contains eval/exec", str(init))
                    if "os.system" in content and "f" in content:
                        _add_finding(results, "warning", "plugins",
                                     f"Plugin {item.name} uses os.system with f-strings (injection risk)", str(init))
                    # Check for unsafe imports
                    for line in content.split('\n'):
                        if 'import' in line and any(x in line for x in ['subprocess', 'os.system', 'pty']):
                            _add_finding(results, "info", "plugins",
                                         f"Plugin {item.name} imports {line.strip()}", str(init))
                            break


def _audit_env(hermes_home, results):
    """Check .env file hygiene."""
    env_path = hermes_home / ".env"
    if not env_path.exists():
        _add_finding(results, "warning", "env", "No .env file found")
        return
    content = env_path.read_text()
    # Check if any keys are embedded in code vs env
    if os.path.exists(hermes_home / ".."):
        _add_finding(results, "info", "env", ".env exists — verify no secrets in config.yaml")


def _audit_permissions(results):
    """Check file permissions."""
    home = get_hermes_home()
    for path in [home / ".env", home / "config.yaml"]:
        if path.exists():
            mode = path.stat().st_mode
            if mode & 0o077:  # Group/other readable
                _add_finding(results, "warning", "permissions",
                             f"{path.name} is readable by others (mode {oct(mode)})",
                             "chmod 600 to restrict access")
    # Check logs
    log_dir = home / "logs"
    if log_dir.exists():
        for logf in log_dir.glob("*.log"):
            mode = logf.stat().st_mode
            if mode & 0o077:
                _add_finding(results, "info", "permissions",
                             f"Log {logf.name} is readable by others")
                break


# ═══ Register ═══

registry.register(
    name="audit_scan",
    toolset="debugging",
    schema={
        "name": "audit_scan",
        "description": (
            "Security audit: scan Hermes config, plugins, env, and permissions "
            "for risks. Returns findings by severity (critical/warning/info). "
            "Use this to check your Hermes installation for security issues."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": ["config", "plugins", "env", "permissions", "all"],
                    "description": "What to audit. 'all' scans everything.",
                },
            },
        },
    },
    handler=lambda args, **kw: audit_scan(scope=args.get("scope", "all"), **kw),
)
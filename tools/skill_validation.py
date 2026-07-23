"""Skill self-validation tool.

Runs lightweight functional checks on a Hermes skill:
- SKILL.md parse / required frontmatter
- Linked scripts compile and execute without error
- Linked templates parse
- References exist
- Skill can be loaded by skill_view

This is the minimal L1 -> L3 upgrade for Self-Validation of Created Capabilities.
"""

import json
import logging
import os
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _skill_path(name: str, category: str = "") -> Path:
    from hermes_constants import get_hermes_home
    home = Path(get_hermes_home())
    if category:
        return home / "skills" / category / name
    return home / "skills" / name


def _read_skill_md(skill_dir: Path) -> str:
    p = skill_dir / "SKILL.md"
    if not p.exists():
        raise FileNotFoundError(f"SKILL.md not found in {skill_dir}")
    return p.read_text(encoding="utf-8")


def _parse_frontmatter(content: str) -> Dict[str, Any]:
    fm = {}
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                import yaml
                fm = yaml.safe_load(parts[1]) or {}
            except Exception:
                for line in parts[1].strip().splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        fm[k.strip()] = v.strip()
    return fm


def _validate_frontmatter(skill_dir: Path, content: str) -> List[str]:
    errors = []
    try:
        fm = _parse_frontmatter(content)
    except Exception as e:
        errors.append(f"frontmatter parse error: {e}")
        return errors
    if not fm.get("name"):
        errors.append("missing frontmatter 'name'")
    if not fm.get("description"):
        errors.append("missing frontmatter 'description'")
    return errors


def _validate_scripts(skill_dir: Path, run_scripts: bool = False) -> List[Dict[str, Any]]:
    scripts_dir = skill_dir / "scripts"
    results = []
    if not scripts_dir.exists():
        return results
    for script in sorted(scripts_dir.iterdir()):
        if script.is_file() and script.suffix == ".py":
            result = {
                "file": str(script.relative_to(skill_dir)),
                "status": "pending",
                "output": None,
                "stdout": "",
                "stderr": "",
                "returncode": None,
                "error": "",
            }
            try:
                py_compile.compile(str(script), doraise=True)
                result["status"] = "compile_ok"
            except Exception as e:
                result["status"] = "compile_failed"
                result["error"] = str(e)
                results.append(result)
                continue
            if run_scripts:
                try:
                    proc = subprocess.run(
                        [sys.executable, str(script)],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=str(skill_dir),
                    )
                    result["stdout"] = proc.stdout
                    result["stderr"] = proc.stderr
                    result["returncode"] = proc.returncode
                    if proc.returncode == 0:
                        result["status"] = "exec_ok"
                    else:
                        result["status"] = "exec_failed"
                        result["error"] = proc.stderr or f"exit code {proc.returncode}"
                except subprocess.TimeoutExpired:
                    result["status"] = "exec_failed"
                    result["error"] = "script timed out after 30s"
                except Exception as e:
                    result["status"] = "exec_failed"
                    result["error"] = str(e)
            results.append(result)
    return results


def _validate_templates(skill_dir: Path) -> List[Dict[str, Any]]:
    templates_dir = skill_dir / "templates"
    results = []
    if not templates_dir.exists():
        return results
    for tmpl in sorted(templates_dir.iterdir()):
        if tmpl.is_file():
            result = {"file": str(tmpl.relative_to(skill_dir)), "status": "pending", "error": ""}
            try:
                txt = tmpl.read_text(encoding="utf-8")
                open_count = txt.count("{")
                close_count = txt.count("}")
                if open_count != close_count:
                    raise ValueError(f"unbalanced braces: {open_count} open, {close_count} close")
                result["status"] = "parse_ok"
            except Exception as e:
                result["status"] = "parse_failed"
                result["error"] = str(e)
            results.append(result)
    return results


def _validate_references(skill_dir: Path) -> List[Dict[str, Any]]:
    refs_dir = skill_dir / "references"
    results = []
    if not refs_dir.exists():
        return results
    for ref in sorted(refs_dir.iterdir()):
        if ref.is_file():
            result = {"file": str(ref.relative_to(skill_dir)), "status": "pending", "error": ""}
            try:
                ref.read_text(encoding="utf-8", errors="ignore")
                result["status"] = "readable"
            except Exception as e:
                result["status"] = "unreadable"
                result["error"] = str(e)
            results.append(result)
    return results


def _run_skill_view_check(skill_dir: Path, name: str) -> Dict[str, Any]:
    result = {"check": "skill_view_load", "status": "pending", "error": ""}
    try:
        from tools.skills_tool import skill_view
        skill_view(name)
        result["status"] = "load_ok"
    except Exception as e:
        result["status"] = "load_failed"
        result["error"] = str(e)
    return result


def validate_skill(name: str, category: str = "", run_scripts: bool = False) -> str:
    skill_dir = _skill_path(name, category)
    errors: List[str] = []
    report = {
        "skill": name,
        "path": str(skill_dir),
        "valid": True,
        "errors": errors,
        "frontmatter": [],
        "scripts": [],
        "templates": [],
        "references": [],
        "skill_view": None,
    }
    if not skill_dir.exists():
        report["valid"] = False
        errors.append(f"skill directory not found: {skill_dir}")
        return json.dumps(report, ensure_ascii=False)

    try:
        content = _read_skill_md(skill_dir)
    except FileNotFoundError as e:
        report["valid"] = False
        errors.append(str(e))
        return json.dumps(report, ensure_ascii=False)

    fm_errors = _validate_frontmatter(skill_dir, content)
    errors.extend(fm_errors)
    report["frontmatter"] = [{"check": "required_fields", "status": "ok" if not fm_errors else "failed", "errors": fm_errors}]

    report["scripts"] = _validate_scripts(skill_dir, run_scripts)
    report["templates"] = _validate_templates(skill_dir)
    report["references"] = _validate_references(skill_dir)
    report["skill_view"] = _run_skill_view_check(skill_dir, name)

    if any(s["status"] in ("compile_failed", "exec_failed") for s in report["scripts"]):
        errors.append("one or more scripts failed to compile or execute")
    if any(t["status"] == "parse_failed" for t in report["templates"]):
        errors.append("one or more templates failed to parse")
    if report["skill_view"]["status"] == "load_failed":
        errors.append(f"skill_view load failed: {report['skill_view']['error']}")

    report["valid"] = not errors
    return json.dumps(report, ensure_ascii=False)


def skill_validate_tool(action: str, name: str = "", category: str = "", run_scripts: bool = False) -> str:
    if action != "validate":
        return json.dumps({"error": f"unknown action {action!r}"}, ensure_ascii=False)
    return validate_skill(name, category, run_scripts)


VALIDATE_SCHEMA = {
    "name": "skill_validate",
    "description": (
        "Validate a created skill: check SKILL.md frontmatter, linked scripts, "
        "templates, references, and that the skill can be loaded. "
        "Use after creating or editing a skill to catch structural problems."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["validate"],
                "description": "Validation action",
            },
            "name": {
                "type": "string",
                "description": "Skill name to validate",
            },
            "category": {
                "type": "string",
                "description": "Optional skill category",
            },
            "run_scripts": {
                "type": "boolean",
                "description": "Whether to execute linked scripts (default false)",
            },
        },
        "required": ["action", "name"],
    },
}


# --- Registry ---
from tools.registry import registry

registry.register(
    name="skill_validate",
    toolset="skills",
    schema=VALIDATE_SCHEMA,
    handler=lambda args, **kw: skill_validate_tool(
        action=args.get("action"),
        name=args.get("name", ""),
        category=args.get("category", ""),
        run_scripts=args.get("run_scripts", False),
    ),
    emoji="✅",
)

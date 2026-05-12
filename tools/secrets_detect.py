#!/usr/bin/env python3
"""
Secrets Detect Tool - Scan for hardcoded secrets

Scans files and directories for hardcoded secrets like API keys, passwords, and private keys.
"""

import json
import os
import re
from typing import Dict, List, Optional


SECRET_PATTERNS = [
    (r"(?i)(api[_-]?key|apikey)\s*[=:]\s*['\"]?([A-Za-z0-9_\-]{20,})['\"]?", "API Key", "high"),
    (r"(?i)(secret[_-]?key|secretkey)\s*[=:]\s*['\"]?([A-Za-z0-9_\-]{16,})['\"]?", "Secret Key", "high"),
    (r"(?i)(password|passwd|pwd)\s*[=:]\s*['\"]?([^\s'\"]{8,})['\"]?", "Password", "high"),
    (r"(?i)(access[_-]?token|access_token|auth[_-]?token)\s*[=:]\s*['\"]?([A-Za-z0-9_\-]{16,})['\"]?", "Access Token", "high"),
    (r"aws[_-]?access[_-]?key[_-]?id\s*[=:]\s*['\"]?([A-Z0-9]{20})['\"]?", "AWS Access Key", "critical"),
    (r"aws[_-]?secret[_-]?access[_-]?key\s*[=:]\s*['\"]?([A-Za-z0-9/+=]{40})['\"]?", "AWS Secret Key", "critical"),
    (r"(?i)bearer\s+[A-Za-z0-9_\-\.]+", "Bearer Token", "high"),
    (r"(?i)token\s*[=:]\s*['\"]?([A-Za-z0-9_\-\.]{16,})['\"]?", "Generic Token", "medium"),
    (r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----", "Private Key", "critical"),
    (r"-----BEGIN CERTIFICATE-----", "Certificate", "medium"),
    (r"(?i)(client[_-]?secret|clientsecret)\s*[=:]\s*['\"]?([A-Za-z0-9_\-]{16,})['\"]?", "Client Secret", "high"),
    (r"(?i)(private[_-]?key|privatekey)\s*[=:]\s*['\"]?([A-Za-z0-9_\-]{16,})['\"]?", "Private Key", "high"),
    (r"github[_-]?token\s*[=:]\s*['\"]?([A-Za-z0-9_]{36,})['\"]?", "GitHub Token", "high"),
    (r"(?i)slack[_-]?(token|bearer)\s*[=:]\s*['\"]?(xox[baprs]-[^\s'\"]{10,})['\"]?", "Slack Token", "high"),
    (r"(?i)stripe[_-]?(sk|secret)[_-]?(key)?\s*[=:]\s*['\"]?([sr]k_(?:live|test)_[A-Za-z0-9]{24,})['\"]?", "Stripe Key", "critical"),
]

EXCLUDE_PATTERNS = [
    r"\.(test|spec)\.py$",
    r"node_modules/",
    r"\.git/",
    r"__pycache__/",
    r"\.min\.js$",
    r"\.map$",
    r"package-lock\.json$",
    r"package\.json$",
    r"requirements\.txt$",
    r"\.env\.example$",
    r"\.gitignore$",
]


def _should_exclude(file_path: str) -> bool:
    """Check if file should be excluded from scanning."""
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, file_path):
            return True
    return False


def _scan_file(file_path: str, custom_patterns: Optional[List[tuple]] = None) -> List[Dict[str, str]]:
    """Scan a single file for secrets."""
    findings = []
    
    try:
        if not os.path.isfile(file_path):
            return findings
        
        if os.path.getsize(file_path) > 10 * 1024 * 1024:
            return findings
        
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        
        all_patterns = list(SECRET_PATTERNS)
        if custom_patterns:
            all_patterns.extend(custom_patterns)
        
        for pattern, secret_type, severity in all_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                line_num = content[:match.start()].count("\n") + 1
                
                context_start = max(0, match.start() - 30)
                context_end = min(len(content), match.end() + 30)
                context = content[context_start:context_end].replace("\n", " ")
                
                findings.append({
                    "type": secret_type,
                    "severity": severity,
                    "file": file_path,
                    "line": line_num,
                    "context": context[:100],
                })
    
    except Exception:
        pass
    
    return findings


MAX_FILES_LIMIT = 1000

def _scan_directory(dir_path: str, max_depth: int = 5, custom_patterns: Optional[List] = None, exclude_custom: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """Scan a directory for secrets."""
    findings = []
    files_scanned = 0
    
    for root, dirs, files in os.walk(dir_path):
        depth = root[len(dir_path):].count(os.sep)
        if depth > max_depth:
            continue
        
        if files_scanned >= MAX_FILES_LIMIT:
            break
        
        dirs[:] = [d for d in dirs if not _should_exclude(os.path.join(root, d))]
        
        for file in files:
            if files_scanned >= MAX_FILES_LIMIT:
                break
                
            file_path = os.path.join(root, file)
            
            if exclude_custom:
                excluded = False
                for pat in exclude_custom:
                    if re.search(pat, file_path):
                        excluded = True
                        break
                if excluded:
                    continue
            
            if not _should_exclude(file_path):
                findings.extend(_scan_file(file_path, custom_patterns=custom_patterns))
                files_scanned += 1
    
    return findings


def secrets_detect(
    path: str,
    patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    severity: str = "all",
    task_id: Optional[str] = None,
) -> str:  # noqa: D205
    """
    Scan files for hardcoded secrets like API keys, passwords, and private keys.

    Args:
        path: Path to scan (file or directory)
        patterns: Custom regex patterns to detect (optional)
        exclude_patterns: Patterns to exclude (e.g., *.test.py)
        severity: Filter by severity: high, medium, low, all
        task_id: Optional task ID for tracking

    Returns:
        JSON string with scan results
    """
    if not os.path.exists(path):
        return json.dumps({
            "success": False,
            "error": f"Path not found: {path}",
        })
    
    if not os.path.isfile(path) and not os.path.isdir(path):
        return json.dumps({
            "success": False,
            "error": f"Invalid path: {path}",
        })
    
    abs_path = os.path.abspath(path)
    if not abs_path:
        return json.dumps({
            "success": False,
            "error": "Invalid path resolution",
        })
    
    if not os.path.isfile(path) and os.path.isdir(path):
        if not os.access(path, os.R_OK):
            return json.dumps({
                "success": False,
                "error": f"Directory not readable: {path}",
            })
    
    custom_patterns = []
    if patterns:
        for p in patterns:
            try:
                re.compile(p)
                custom_patterns.append((p, "Custom Pattern", "high"))
            except re.error:
                pass
    
    all_findings = []
    
    if os.path.isfile(path):
        excluded = False
        if exclude_patterns:
            for pat in exclude_patterns:
                if re.search(pat, path):
                    excluded = True
                    break
        if not excluded and not _should_exclude(path):
            all_findings = _scan_file(path, custom_patterns=custom_patterns)
    else:
        all_findings = _scan_directory(path, custom_patterns=custom_patterns, exclude_custom=exclude_patterns)
    
    severity_filter = severity.upper() if severity != "all" else "ALL"
    if severity_filter != "ALL":
        all_findings = [f for f in all_findings if f["severity"].upper() == severity_filter]
    
    critical = [f for f in all_findings if f["severity"] == "critical"]
    high = [f for f in all_findings if f["severity"] == "high"]
    medium = [f for f in all_findings if f["severity"] == "medium"]
    
    unique_files = set(f["file"] for f in all_findings)
    
    result = {
        "success": True,
        "path": path,
        "total_findings": len(all_findings),
        "findings": all_findings[:100],
        "summary": {
            "critical": len(critical),
            "high": len(high),
            "medium": len(medium),
            "files_scanned": len(unique_files),
        },
        "recommendation": "Review and remove any hardcoded secrets. Use environment variables or secret management tools instead." if all_findings else "No secrets detected.",
    }
    
    return json.dumps(result, ensure_ascii=False)


def check_secrets_detect_requirements() -> bool:
    """Secrets detect tool has no external requirements."""
    return True


SECRETS_DETECT_SCHEMA = {
    "name": "secrets_detect",
    "description": (
        "Scan files for hardcoded secrets like API keys, passwords, and private keys.\n\n"
        "Parameters:\n"
        "- path: Path to scan (file or directory)\n"
        "- patterns: Custom regex patterns to detect\n"
        "- exclude_patterns: Patterns to exclude\n"
        "- severity: Filter by severity: high, medium, low, all"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to scan (file or directory)",
            },
            "patterns": {
                "type": "array",
                "description": "Custom regex patterns to detect",
                "items": {"type": "string"},
            },
            "exclude_patterns": {
                "type": "array",
                "description": "Patterns to exclude (e.g., *.test.py)",
                "items": {"type": "string"},
            },
            "severity": {
                "type": "string",
                "description": "Filter by severity: high, medium, low, all",
                "enum": ["high", "medium", "low", "all"],
                "default": "all",
            },
            "task_id": {
                "type": "string",
                "description": "Optional task ID for tracking",
            },
        },
        "required": ["path"],
    },
}


from tools.registry import registry

registry.register(
    name="secrets_detect",
    toolset="security",
    schema=SECRETS_DETECT_SCHEMA,
    handler=lambda args, **kw: secrets_detect(
        path=args.get("path", ""),
        patterns=args.get("patterns"),
        exclude_patterns=args.get("exclude_patterns"),
        severity=args.get("severity", "all"),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_secrets_detect_requirements,
    emoji="🔐",
)
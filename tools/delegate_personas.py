#!/usr/bin/env python3
"""Native delegation personas for ``delegate_task``.

This module is deliberately a delegation-policy helper, not a replacement for
Hermes' parent identity systems:

* ``SOUL.md`` remains the parent agent's durable identity.
* ``/personality`` remains a session-level overlay for the parent chat.
* ``delegate_task(role=...)`` remains the child spawn-permission model.

Personas here are per-child task profiles.  They resolve to context and, for
local Claude/Cursor workers, safe bridge arguments.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error


SOUL_COMPRESS_THRESHOLD_BYTES = 4096
CACHE_TTL_SECONDS = 3600

DEFAULT_ROUTING: dict[str, Any] = {
    "defaults": {
        "claude": "opus",
        "cursor-agent": "gpt-5.5-extra-high",
    },
    "tiers": [
        {
            "name": "deep-design",
            "patterns": ["architect", "design-architect", "planner", "design-system"],
            "models": {
                "claude": "opus",
                "cursor-agent": "claude-opus-4-7-thinking-xhigh",
            },
        },
        {
            "name": "research",
            "patterns": ["researcher", "research-reviewer", "deep-research"],
            "models": {
                "claude": "opus",
                "cursor-agent": "claude-opus-4-7-thinking-xhigh",
            },
        },
        {
            "name": "code",
            "patterns": ["implementer", "code-reviewer", "coder", "engineer", "developer", "refactor"],
            "models": {
                "claude": "opus",
                "cursor-agent": "gpt-5.5-extra-high",
            },
        },
        {
            "name": "quick-verify",
            "patterns": ["verifier", "debugger", "fixer", "quick-fix", "tester", "test"],
            "models": {
                "claude": "opus",
                "cursor-agent": "gpt-5.5-extra-high",
            },
        },
    ],
}

PROVIDER_BASE_ARGS: dict[str, list[str]] = {
    "claude": ["-p", "--output-format", "text"],
    "cursor-agent": ["-p", "--output-format", "text"],
}

WRITE_PERSONA_PATTERNS = (
    "implementer",
    "coder",
    "engineer",
    "developer",
    "refactor",
    "fixer",
    "quick-fix",
)


def _hermes_home() -> Path:
    return get_hermes_home()


def _cache_dir() -> Path:
    return _hermes_home() / "cache" / "delegate-personas"


def _expand_path(value: str | os.PathLike[str]) -> Path:
    return Path(str(value)).expanduser()


def _default_persona_dirs() -> list[tuple[str, Path]]:
    home = _hermes_home()
    return [
        ("project", home / "personas" / "project"),
        ("user", home / "personas"),
    ]


def _configured_persona_dirs(cfg: dict[str, Any] | None = None) -> list[tuple[str, Path]]:
    """Return persona directories with stable precedence.

    ``delegation.persona_dirs`` may be either:
      * a list of paths; or
      * a mapping of pool name -> path/list-of-paths.

    Defaults intentionally stay under HERMES_HOME so upstream installs do not
    depend on any project-specific filesystem layout.
    """

    cfg = cfg or {}
    configured = cfg.get("persona_dirs")
    dirs: list[tuple[str, Path]] = []
    if isinstance(configured, dict):
        for pool, raw in configured.items():
            values = raw if isinstance(raw, list) else [raw]
            for value in values:
                if isinstance(value, str) and value.strip():
                    dirs.append((str(pool), _expand_path(value)))
    elif isinstance(configured, list):
        for idx, value in enumerate(configured):
            if isinstance(value, str) and value.strip():
                dirs.append((f"configured-{idx + 1}", _expand_path(value)))

    return dirs or _default_persona_dirs()


def _routing(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or {}
    configured = cfg.get("persona_routing")
    if isinstance(configured, dict):
        merged = dict(DEFAULT_ROUTING)
        if isinstance(configured.get("defaults"), dict):
            merged["defaults"] = {**DEFAULT_ROUTING["defaults"], **configured["defaults"]}
        if isinstance(configured.get("tiers"), list):
            merged["tiers"] = configured["tiers"]
        return merged
    return DEFAULT_ROUTING


def _parse_frontmatter(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = yaml.safe_load(match.group(1)) or {}
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _body_without_frontmatter(text: str) -> str:
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return text.strip()


def _contains_name_pattern(name: str, pattern: str) -> bool:
    lowered = (name or "").lower()
    cleaned = re.escape((pattern or "").lower())
    return bool(re.search(rf"(^|[^a-z0-9]){cleaned}($|[^a-z0-9])", lowered))


def _persona_body(path: Path) -> str:
    if path.is_symlink():
        return ""
    try:
        return _body_without_frontmatter(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return ""


def _compress_persona(text: str) -> str:
    """Extract a markdown skeleton while preserving operational directives."""

    out: list[str] = []
    in_code = False
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.lstrip()

        if stripped.startswith("```"):
            if not in_code:
                in_code = True
                lang = stripped[3:].strip() or "code"
                out.append(f"```{lang} (body elided)")
            else:
                in_code = False
                out.append("```")
            continue
        if in_code:
            continue
        if not stripped:
            if out and out[-1] != "":
                out.append("")
            continue
        if stripped.startswith("#"):
            out.append(line)
            continue
        if stripped.startswith(("-", "*", "+")) and len(stripped) > 1 and stripped[1] == " ":
            out.append(line)
            continue
        if re.match(r"^\d+\.\s", stripped):
            out.append(line)
            continue
        if stripped.startswith(">"):
            out.append(line)
            continue
        if re.match(r"^\|.*\|", stripped):
            out.append(line)
            continue
        if re.match(r"^[A-Z][A-Z\s]{2,}:", stripped):
            out.append(line)
            continue
        if re.match(r"^(MUST|NEVER|ALWAYS|DO NOT|REQUIRED|FORBIDDEN)\b", stripped, re.IGNORECASE):
            out.append(line)

    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)


def _persona_files(cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for pool, root in _configured_persona_dirs(cfg):
        if not root.exists():
            continue
        patterns = ["*.md", "*/agents/*.md"]
        for pattern in patterns:
            for path in sorted(root.glob(pattern)):
                if not path.is_file() or path.is_symlink():
                    continue
                name = path.stem
                key = name.lower()
                if key in seen:
                    continue
                fm = _parse_frontmatter(path)
                seen[key] = {
                    "name": name,
                    "pool": pool,
                    "path": str(path),
                    "description": str(fm.get("description") or fm.get("summary") or ""),
                    "model": str(fm.get("model") or "") or None,
                    "tools": fm.get("tools") if isinstance(fm.get("tools"), list) else [],
                }
    return sorted(seen.values(), key=lambda item: item["name"].lower())


def list_delegate_personas(
    *,
    cfg: dict[str, Any] | None = None,
    mode: str = "signatures",
    pool: str = "all",
    filter_text: str = "",
) -> dict[str, Any]:
    entries = _persona_files(cfg)
    if pool != "all":
        entries = [entry for entry in entries if entry["pool"] == pool]
    needle = filter_text.lower().strip()
    if needle:
        entries = [
            entry
            for entry in entries
            if needle in entry["name"].lower() or needle in entry.get("description", "").lower()
        ]

    projected: list[dict[str, Any]] = []
    for entry in entries:
        item = {"name": entry["name"], "pool": entry["pool"]}
        if mode in {"signatures", "full"}:
            item.update(
                {
                    "description": entry.get("description", "")[:180],
                    "model": entry.get("model"),
                    "can_write": persona_can_write(entry["name"]),
                }
            )
        if mode == "full":
            item.update({"path": entry["path"], "tools": entry.get("tools") or []})
        projected.append(item)

    return {"count": len(projected), "mode": mode, "personas": projected}


def resolve_persona(name: str, cfg: dict[str, Any] | None = None) -> dict[str, Any] | None:
    wanted = (name or "").strip().lower()
    if not wanted:
        return None
    for entry in _persona_files(cfg):
        if entry["name"].lower() == wanted:
            return entry
    return None


def persona_can_write(name: str) -> bool:
    return any(_contains_name_pattern(name, pattern) for pattern in WRITE_PERSONA_PATTERNS)


def _resolve_model(
    provider: str,
    persona_name: str,
    persona_frontmatter_model: str | None,
    override: str | None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if override:
        return {"model": override, "reason": "explicit override"}

    routing = _routing(cfg)
    defaults = routing.get("defaults", {}) or {}
    tiers = routing.get("tiers", []) or []
    name_lower = persona_name.lower()
    for tier in tiers:
        for pattern in tier.get("patterns", []):
            if _contains_name_pattern(name_lower, str(pattern)):
                model = (tier.get("models", {}) or {}).get(provider)
                if model:
                    return {
                        "model": model,
                        "reason": f"tier '{tier.get('name', '?')}' matched pattern '{pattern}'",
                    }

    if persona_frontmatter_model:
        return {"model": persona_frontmatter_model, "reason": "persona frontmatter"}
    return {"model": defaults.get(provider, ""), "reason": "provider default"}


def _cursor_catalog(force_refresh: bool = False) -> dict[str, Any]:
    cache_dir = _cache_dir()
    cache_path = cache_dir / "cursor-models.json"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not force_refresh and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            age = time.time() - cached.get("fetched_at", 0)
            if age < CACHE_TTL_SECONDS:
                cached["_cache_age_seconds"] = int(age)
                return cached
        except Exception:
            pass

    try:
        proc = subprocess.run(
            ["cursor-agent", "--list-models"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return {"_error": str(exc), "models": {}, "_cache_age_seconds": 0}
    if proc.returncode != 0:
        return {"_error": proc.stderr[:300], "models": {}, "_cache_age_seconds": 0}
    models: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        match = re.match(r"^([a-z0-9.\-]+)\s+-\s+(.+)$", line.strip())
        if match:
            models[match.group(1)] = match.group(2).strip()
    payload = {"models": models, "fetched_at": int(time.time()), "_cache_age_seconds": 0}
    try:
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass
    return payload


def list_delegate_persona_models(
    *,
    cfg: dict[str, Any] | None = None,
    provider: str = "all",
    force_refresh: bool = False,
) -> dict[str, Any]:
    routing = _routing(cfg)
    result: dict[str, Any] = {
        "defaults": routing.get("defaults", {}),
        "tiers": routing.get("tiers", []),
    }
    if provider in {"all", "claude"}:
        result["claude"] = {
            "default": (routing.get("defaults", {}) or {}).get("claude", "opus"),
            "note": "Claude CLI accepts aliases such as opus/sonnet/haiku or full model ids.",
        }
    if provider in {"all", "cursor-agent"}:
        catalog = _cursor_catalog(force_refresh=force_refresh)
        result["cursor-agent"] = {
            "default": (routing.get("defaults", {}) or {}).get("cursor-agent", ""),
            "catalog_size": len(catalog.get("models") or {}),
            "cache_age_seconds": catalog.get("_cache_age_seconds", 0),
            "models": catalog.get("models") or {},
        }
        if "_error" in catalog:
            result["cursor-agent"]["_error"] = catalog["_error"]
    return result


def _build_acp_args(provider: str, model: str, workdir: str, persona_name: str) -> list[str]:
    args = list(PROVIDER_BASE_ARGS.get(provider, ["-p"]))
    can_write = persona_can_write(persona_name)
    if provider == "claude":
        if model:
            args += ["--model", model]
        args += ["--add-dir", workdir, "--permission-mode", "acceptEdits" if can_write else "plan"]
    elif provider == "cursor-agent":
        if model:
            args += ["--model", model]
        args += ["--yolo"] if can_write else ["--mode", "plan"]
    return args


def _build_persona_context(
    *,
    persona: dict[str, Any],
    base_context: str | None,
    workdir: str,
    model: str,
    compress: str,
) -> tuple[str, dict[str, Any]]:
    raw = _persona_body(Path(persona["path"]))
    raw_size = len(raw)
    used = raw
    action = "verbatim"
    if compress == "always" or (compress == "auto" and raw_size > SOUL_COMPRESS_THRESHOLD_BYTES):
        used = _compress_persona(raw)
        action = f"compressed ({raw_size}B -> {len(used)}B)"

    parts: list[str] = []
    if used:
        parts.append(f"# DELEGATED PERSONA: {persona['name']}\n\n{used}")
    parts.append(f"# WORKDIR\n\n{workdir}")
    if model:
        parts.append(f"# MODEL\n\n{model}")
    if base_context and base_context.strip():
        parts.append(f"# TASK CONTEXT\n\n{base_context.strip()}")
    return "\n\n".join(parts), {"soul_action": action, "soul_raw_bytes": raw_size}


def apply_persona_to_task(
    task: dict[str, Any],
    *,
    cfg: dict[str, Any] | None = None,
    top_level_persona: str | None = None,
    top_level_provider: str | None = None,
    top_level_model: str | None = None,
    top_level_workdir: str | None = None,
    top_level_compress: str = "auto",
    top_level_transport: str | None = None,
    top_level_acp_command: str | None = None,
    top_level_acp_args: list[str] | None = None,
) -> dict[str, Any]:
    """Return a copy of *task* with persona-derived delegation fields filled.

    Explicit task fields always win.  If no persona is requested, the task is
    returned unchanged.
    """

    persona_name = task.get("persona") or top_level_persona
    if not persona_name:
        return dict(task)

    persona = resolve_persona(str(persona_name), cfg)
    if not persona:
        raise ValueError(f"delegation persona not found: {persona_name}")

    command_base = os.path.basename(str(top_level_acp_command or "")).lower()
    command_provider = command_base if command_base in {"claude", "cursor-agent"} else None
    provider = str(
        task.get("persona_provider")
        or top_level_provider
        or command_provider
        or (cfg or {}).get("persona_provider")
        or "claude"
    )
    if provider not in {"claude", "cursor-agent"}:
        raise ValueError("delegation persona provider must be 'claude' or 'cursor-agent'")

    workdir = str(task.get("workdir") or top_level_workdir or (cfg or {}).get("persona_workdir") or os.getcwd())
    workdir = str(_expand_path(workdir))
    model_override = task.get("persona_model") or top_level_model
    resolved = _resolve_model(provider, persona["name"], persona.get("model"), model_override, cfg)
    model = resolved["model"]
    compress = str(task.get("compress_persona") or top_level_compress or "auto")

    context, context_meta = _build_persona_context(
        persona=persona,
        base_context=task.get("context"),
        workdir=workdir,
        model=model,
        compress=compress,
    )

    enriched = dict(task)
    enriched["context"] = context
    if not top_level_acp_command:
        enriched.setdefault("acp_command", provider)
    if not top_level_acp_args:
        enriched.setdefault("acp_args", _build_acp_args(provider, model, workdir, persona["name"]))
    enriched.setdefault("transport", top_level_transport or "bridge")
    enriched.setdefault("unsafe_allow_writes", persona_can_write(persona["name"]))
    enriched["_persona_meta"] = {
        "persona": persona["name"],
        "pool": persona["pool"],
        "provider": provider,
        "model": model,
        "model_resolved_via": resolved["reason"],
        "workdir": workdir,
        **context_meta,
    }
    return enriched


def _load_delegation_config() -> dict[str, Any]:
    try:
        from cli import CLI_CONFIG

        cfg = CLI_CONFIG.get("delegation", {})
        if cfg:
            return cfg
    except Exception:
        pass
    try:
        from hermes_cli.config import load_config

        full = load_config()
        return full.get("delegation", {})
    except Exception:
        return {}


DELEGATE_LIST_PERSONAS_SCHEMA = {
    "name": "delegate_list_personas",
    "description": (
        "List native delegate_task personas. Personas are per-child delegation "
        "profiles; they do not modify SOUL.md or /personality."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["map", "signatures", "full"], "default": "signatures"},
            "pool": {"type": "string", "default": "all"},
            "filter": {"type": "string", "description": "Case-insensitive match on persona name or description."},
        },
        "additionalProperties": False,
    },
}


def _delegate_list_personas_handler(args: dict[str, Any], **kwargs) -> str:
    try:
        return json.dumps(
            list_delegate_personas(
                cfg=_load_delegation_config(),
                mode=args.get("mode") or "signatures",
                pool=args.get("pool") or "all",
                filter_text=args.get("filter") or "",
            ),
            ensure_ascii=False,
        )
    except Exception as exc:
        return tool_error(str(exc))


registry.register(
    name="delegate_list_personas",
    toolset="delegation",
    schema=DELEGATE_LIST_PERSONAS_SCHEMA,
    handler=_delegate_list_personas_handler,
    check_fn=lambda: True,
    emoji="👥",
)

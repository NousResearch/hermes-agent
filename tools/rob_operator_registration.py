"""Registers Rob's read-only operator tools with Hermès's existing tool
registry (``tools.registry`` — auto-discovered by
``discover_builtin_tools()``, no separate registration list to maintain;
this file is picked up automatically because it calls ``registry.register``
at import time, same mechanism every other built-in tool file uses).

``ToolEntry`` (``tools/registry.py``) has no boolean "read_only" field to
set — confirmed directly from its own ``__slots__``. Per this
implementation's own scope instruction ("If existing tool metadata
supports annotations, add read_only = true. If not, do not invent a
giant new registry system just for metadata"), read-only-ness is instead
encoded the same way every other tool already signals its own nature to
the model: the ``rob_`` name prefix and an explicit "[READ-ONLY]" marker
at the start of every description below — visible to the agent choosing
between tools, with zero new registry infrastructure.

This registers 18 of the 19 functions in ``tools/rob_operator_tools.py``
plus ``env_presence`` (19 total), proving the end-to-end pattern (schema →
handler → guard → redaction → bounded result) genuinely works through the
real tool-calling path, not just as directly-callable Python functions.
``container_exec_readonly`` is implemented and tested but deliberately not
registered — see the comment at its former registration site below.
"""

from __future__ import annotations

from tools.registry import registry
from tools.rob_operator_tools import (
    db_select,
    docker_compose_ps,
    docker_inspect,
    docker_logs,
    docker_network_inspect,
    docker_ps,
    docker_stats,
    docker_volume_inspect,
    git_inspect,
    host_metrics,
    http_probe,
    journal_query,
    network_probe,
    process_inspect,
    schema_inspect,
    systemd_show,
    systemd_status,
    tls_inspect,
)
from tools.env_presence import env_presence


def _result_dict(result) -> dict:
    return {"ok": result.ok, "output": result.output, "error": result.error, "duration_ms": result.duration_ms}


registry.register(
    name="rob_docker_ps",
    toolset="rob_operator",
    schema={
        "type": "function",
        "function": {
            "name": "rob_docker_ps",
            "description": "[READ-ONLY] List running Docker containers (docker ps). Never starts, stops, or modifies anything.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    handler=lambda args, **kw: _result_dict(docker_ps()),
    emoji="🔍",
)

registry.register(
    name="rob_docker_inspect",
    toolset="rob_operator",
    schema={
        "type": "function",
        "function": {
            "name": "rob_docker_inspect",
            "description": (
                "[READ-ONLY] Inspect a Docker container's full configuration "
                "(docker inspect). Secret-shaped values (passwords, tokens) in "
                "the output are automatically redacted. Never modifies the container."
            ),
            "parameters": {
                "type": "object",
                "properties": {"container": {"type": "string", "description": "Container name, e.g. 'project-os-mcp'."}},
                "required": ["container"],
            },
        },
    },
    handler=lambda args, **kw: _result_dict(docker_inspect(args["container"])),
    emoji="🔍",
)

registry.register(
    name="rob_docker_logs",
    toolset="rob_operator",
    schema={
        "type": "function",
        "function": {
            "name": "rob_docker_logs",
            "description": "[READ-ONLY] Read a Docker container's logs. Never modifies or restarts the container.",
            "parameters": {
                "type": "object",
                "properties": {
                    "container": {"type": "string", "description": "Container name."},
                    "since": {"type": "string", "description": "e.g. '10 minutes ago' or an ISO timestamp. Optional."},
                    "tail": {"type": "integer", "description": "Max lines to return (<=10000). Optional."},
                },
                "required": ["container"],
            },
        },
    },
    handler=lambda args, **kw: _result_dict(docker_logs(args["container"], since=args.get("since"), tail=args.get("tail"))),
    emoji="📜",
)

registry.register(
    name="rob_systemd_status",
    toolset="rob_operator",
    schema={
        "type": "function",
        "function": {
            "name": "rob_systemd_status",
            "description": "[READ-ONLY] Show a systemd unit's status (systemctl status). Never starts, stops, or restarts it.",
            "parameters": {
                "type": "object",
                "properties": {"unit": {"type": "string", "description": "Unit name, e.g. 'hermes-gateway.service'."}},
                "required": ["unit"],
            },
        },
    },
    handler=lambda args, **kw: _result_dict(systemd_status(args["unit"])),
    emoji="⚙️",
)

registry.register(
    name="rob_journal_query",
    toolset="rob_operator",
    schema={
        "type": "function",
        "function": {
            "name": "rob_journal_query",
            "description": "[READ-ONLY] Query systemd journal logs (journalctl), optionally filtered by unit/time/priority/grep.",
            "parameters": {
                "type": "object",
                "properties": {
                    "unit": {"type": "string", "description": "Optional systemd unit to filter by."},
                    "since": {"type": "string", "description": "Optional, e.g. '1 hour ago'."},
                    "until": {"type": "string", "description": "Optional."},
                    "priority": {"type": "string", "description": "Optional: emerg|alert|crit|err|warning|notice|info|debug."},
                    "grep": {"type": "string", "description": "Optional text filter applied after journalctl."},
                },
                "required": [],
            },
        },
    },
    handler=lambda args, **kw: _result_dict(
        journal_query(
            unit=args.get("unit"), since=args.get("since"), until=args.get("until"),
            priority=args.get("priority"), grep=args.get("grep"),
        )
    ),
    emoji="📰",
)

registry.register(
    name="rob_git_inspect",
    toolset="rob_operator",
    schema={
        "type": "function",
        "function": {
            "name": "rob_git_inspect",
            "description": (
                "[READ-ONLY] Inspect a Git repository's state. Only status/log/diff/show/"
                "rev-parse/merge-base/worktree-list/branch-current/reflog/tag are permitted — "
                "never checkout, commit, push, pull, fetch, reset, or any other mutating operation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Path to the repository."},
                    "operation": {
                        "type": "string",
                        "enum": ["status", "log", "diff", "show", "rev-parse", "merge-base", "worktree-list", "branch-current", "reflog", "tag"],
                    },
                    "args": {"type": "array", "items": {"type": "string"}, "description": "Only used by merge-base (exactly two refs)."},
                },
                "required": ["repo", "operation"],
            },
        },
    },
    handler=lambda args, **kw: _result_dict(git_inspect(args["repo"], args["operation"], args=args.get("args"))),
    emoji="🌿",
)

registry.register(
    name="rob_network_probe",
    toolset="rob_operator",
    schema={
        "type": "function",
        "function": {
            "name": "rob_network_probe",
            "description": "[READ-ONLY] Test TCP connectivity to a host:port. A bounded connect test only — never sends or reads data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "host": {"type": "string"},
                    "port": {"type": "integer", "description": "1-65535"},
                },
                "required": ["host", "port"],
            },
        },
    },
    handler=lambda args, **kw: _result_dict(network_probe(args["host"], args["port"])),
    emoji="🔌",
)

registry.register(
    name="rob_http_probe",
    toolset="rob_operator",
    schema={
        "type": "function",
        "function": {
            "name": "rob_http_probe",
            "description": "[READ-ONLY] GET or HEAD an HTTP(S) URL. No other method is ever permitted — no POST/PUT/PATCH/DELETE, no request body.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "method": {"type": "string", "enum": ["GET", "HEAD"], "description": "Defaults to GET."},
                },
                "required": ["url"],
            },
        },
    },
    handler=lambda args, **kw: _result_dict(http_probe(args["url"], method=args.get("method", "GET"))),
    emoji="🌐",
)

registry.register(
    name="rob_host_metrics",
    toolset="rob_operator",
    schema={
        "type": "function",
        "function": {
            "name": "rob_host_metrics",
            "description": "[READ-ONLY] Snapshot of disk usage, memory, uptime, and kernel version for the current host.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    handler=lambda args, **kw: _result_dict(host_metrics()),
    emoji="🖥️",
)

registry.register(
    name="rob_env_presence",
    toolset="rob_operator",
    schema={
        "type": "function",
        "function": {
            "name": "rob_env_presence",
            "description": (
                "[READ-ONLY] Check whether named environment variables are set, WITHOUT ever "
                "revealing their values. Returns PRESENT or ABSENT per name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "names": {"type": "array", "items": {"type": "string"}},
                    "target": {
                        "type": "string",
                        "description": "Optional: 'local' (default), 'docker:<container>', or 'systemd:<unit>'.",
                    },
                },
                "required": ["names"],
            },
        },
    },
    handler=lambda args, **kw: {
        "target": (r := env_presence(args["names"], target=args.get("target"))).target,
        "presence": r.presence,
        "error": r.error,
    },
    emoji="🔑",
)


# ---------------------------------------------------------------------------
# Completion pass — remaining 10 of 19 rob_operator_tools functions.
# Same mechanism, same rob_ prefix, same "[READ-ONLY]" marker convention
# as every registration above. No new registry infrastructure.
# ---------------------------------------------------------------------------

registry.register(
    name="rob_docker_stats",
    toolset="rob_operator",
    schema={
        "type": "function",
        "function": {
            "name": "rob_docker_stats",
            "description": "[READ-ONLY] Snapshot of container resource usage (docker stats --no-stream). One-shot, never streams.",
            "parameters": {
                "type": "object",
                "properties": {"container": {"type": "string", "description": "Optional: limit to one container."}},
                "required": [],
            },
        },
    },
    handler=lambda args, **kw: _result_dict(docker_stats(container=args.get("container"))),
    emoji="📊",
)

registry.register(
    name="rob_docker_network_inspect",
    toolset="rob_operator",
    schema={
        "type": "function",
        "function": {
            "name": "rob_docker_network_inspect",
            "description": "[READ-ONLY] Inspect a Docker network's configuration (docker network inspect). Never creates or modifies networks.",
            "parameters": {
                "type": "object",
                "properties": {"network": {"type": "string", "description": "Network name."}},
                "required": ["network"],
            },
        },
    },
    handler=lambda args, **kw: _result_dict(docker_network_inspect(args["network"])),
    emoji="🔗",
)

registry.register(
    name="rob_docker_volume_inspect",
    toolset="rob_operator",
    schema={
        "type": "function",
        "function": {
            "name": "rob_docker_volume_inspect",
            "description": "[READ-ONLY] Inspect a Docker volume's configuration (docker volume inspect). Never creates, modifies, or removes volumes.",
            "parameters": {
                "type": "object",
                "properties": {"volume": {"type": "string", "description": "Volume name."}},
                "required": ["volume"],
            },
        },
    },
    handler=lambda args, **kw: _result_dict(docker_volume_inspect(args["volume"])),
    emoji="💽",
)

registry.register(
    name="rob_docker_compose_ps",
    toolset="rob_operator",
    schema={
        "type": "function",
        "function": {
            "name": "rob_docker_compose_ps",
            "description": "[READ-ONLY] List a Docker Compose project's services and their state (docker compose ps). Never starts, stops, or recreates services.",
            "parameters": {
                "type": "object",
                "properties": {"project_dir": {"type": "string", "description": "Path to the directory containing the compose file."}},
                "required": ["project_dir"],
            },
        },
    },
    handler=lambda args, **kw: _result_dict(docker_compose_ps(args["project_dir"])),
    emoji="🧩",
)

registry.register(
    name="rob_systemd_show",
    toolset="rob_operator",
    schema={
        "type": "function",
        "function": {
            "name": "rob_systemd_show",
            "description": "[READ-ONLY] Show a systemd unit's properties (systemctl show). Never starts, stops, or restarts it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "unit": {"type": "string", "description": "Unit name, e.g. 'hermes-gateway.service'."},
                    "property": {"type": "string", "description": "Optional: limit output to one property."},
                },
                "required": ["unit"],
            },
        },
    },
    handler=lambda args, **kw: _result_dict(systemd_show(args["unit"], property=args.get("property"))),
    emoji="⚙️",
)

registry.register(
    name="rob_tls_inspect",
    toolset="rob_operator",
    schema={
        "type": "function",
        "function": {
            "name": "rob_tls_inspect",
            "description": "[READ-ONLY] Fetch and summarize the TLS certificate a host presents (subject, issuer, validity). Establishes a TLS handshake only, never sends data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "host_name": {"type": "string"},
                    "port": {"type": "integer", "description": "Defaults to 443."},
                },
                "required": ["host_name"],
            },
        },
    },
    handler=lambda args, **kw: _result_dict(tls_inspect(args["host_name"], port=args.get("port", 443))),
    emoji="🔒",
)

registry.register(
    name="rob_process_inspect",
    toolset="rob_operator",
    schema={
        "type": "function",
        "function": {
            "name": "rob_process_inspect",
            "description": "[READ-ONLY] Show one process's status by PID (ps). Never sends signals or modifies the process.",
            "parameters": {
                "type": "object",
                "properties": {"pid": {"type": "integer", "description": "Positive process ID."}},
                "required": ["pid"],
            },
        },
    },
    handler=lambda args, **kw: _result_dict(process_inspect(args["pid"])),
    emoji="🧬",
)

registry.register(
    name="rob_db_select",
    toolset="rob_operator",
    schema={
        "type": "function",
        "function": {
            "name": "rob_db_select",
            "description": (
                "[READ-ONLY] Run a SELECT-only query against a dedicated read-only database "
                "profile. Refuses to run unless a non-superuser read-only profile is configured "
                "for the given name; refuses any non-SELECT statement at the text level in "
                "addition to the profile's own server-side read-only enforcement. Row-limited "
                "and secret-redacted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "profile": {"type": "string", "description": "Read-only DB profile name, e.g. 'projectos'."},
                    "query": {"type": "string", "description": "A single SELECT (or read-only WITH...SELECT) statement."},
                    "row_limit": {"type": "integer", "description": "Optional, defaults to 100, capped by the profile's own limit."},
                },
                "required": ["profile", "query"],
            },
        },
    },
    handler=lambda args, **kw: _result_dict(db_select(args["profile"], args["query"], row_limit=args.get("row_limit", 100))),
    emoji="🗄️",
)

registry.register(
    name="rob_schema_inspect",
    toolset="rob_operator",
    schema={
        "type": "function",
        "function": {
            "name": "rob_schema_inspect",
            "description": (
                "[READ-ONLY] List tables in the 'public' schema, or (given an object name) "
                "list one table's columns and types. Uses the same read-only DB profile "
                "requirement as rob_db_select."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "profile": {"type": "string", "description": "Read-only DB profile name."},
                    "object": {"type": "string", "description": "Optional table name to describe."},
                },
                "required": ["profile"],
            },
        },
    },
    handler=lambda args, **kw: _result_dict(schema_inspect(args["profile"], object=args.get("object"))),
    emoji="🗂️",
)

# ---------------------------------------------------------------------------
# container_exec_readonly is intentionally NOT registered.
# ---------------------------------------------------------------------------
# It builds `docker exec <container> <command>` and re-validates the WHOLE
# string through run_read_only_guard(), but the guard's docker validator
# has no `exec` entry in its subcommand allowlist — every invocation is
# therefore denied unconditionally, with no way to configure it into
# working. Registering an always-failing tool wastes a model turn on every
# call and mis-states the toolset's real surface, so it stays implemented
# and tested (tools/rob_operator_tools.py, tests/tools/test_rob_operator_tools.py)
# but out of the registry until a real `docker exec` allowlist entry is
# added to read_only_command_guard.py's _validate_docker — deliberately
# NOT done as part of this pass, since a `docker exec` allowlist entry is a
# new guard capability, not a bugfix. The consolidated security-closure
# pass also REMOVED the dead validator families (curl, find, file, rg,
# openssl, ip, tailscale — none reachable from a registered rob_* tool) and
# trimmed docker/systemctl to exactly the subcommands the registered tools
# emit; adding new reachable surface in the same commit as closing a guard
# gap is exactly the sequencing this implementation's own review flagged
# as risky.

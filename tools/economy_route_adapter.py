"""Token-bound native/Hermes launch adapter for reviewed economy routes.

This is cooperative host enforcement, not an OS sandbox. It consumes only an
Admission v3 sealed envelope and returns a bounded receipt. It never selects a
route, changes global configuration, obtains credentials, or falls back.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os
import re
import subprocess
import sys
import time
import uuid


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _notice_reference(envelope, token):
    notice = {"token": token, "host_event_nonce": envelope["host_event_nonce"],
              "task_id": envelope["task_id"],
              "effective_route": envelope["selection"]["effective_route"]}
    return "notice:sha256:" + _digest(notice)


def _verify(envelope, token):
    if not isinstance(envelope, dict) or not isinstance(token, str) or not token:
        raise ValueError("sealed envelope and token required")
    sealed = dict(envelope)
    claimed = sealed.pop("envelope_sha256", None)
    if claimed != _digest(sealed):
        raise ValueError("envelope hash mismatch")
    required = {"selection", "candidate", "task_packet", "project_root", "allowed_tools",
                "allowed_paths", "privacy", "billing_modes", "attempt_id", "host_event_nonce", "tracking"}
    if not required.issubset(envelope):
        raise ValueError("incomplete sealed envelope")
    packet = envelope["task_packet"]
    tracking = envelope["tracking"]
    hashes = envelope.get("source_hashes")
    if (not isinstance(packet, dict)
            or set(packet) != {"schema_version", "goal", "non_goals", "context_refs"}
            or packet.get("schema_version") != 1 or not isinstance(hashes, dict)
            or not isinstance(tracking, dict) or set(tracking) != {"project_id", "build_id"}
            or any(not isinstance(tracking.get(key), str) or not tracking[key]
                   for key in ("project_id", "build_id"))
            or any(not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
                   or hashes.get(ref.get("path")) != ref.get("sha256")
                   for ref in packet.get("context_refs", []))):
        raise ValueError("versioned hash-bound task packet required")
    route = envelope["selection"]["effective_route"]
    seat_id = envelope["selection"].get("selected_seat_id")
    if (not isinstance(seat_id, str) or not seat_id
            or envelope["candidate"].get("seat_id") != seat_id):
        raise ValueError("candidate identity does not bind the selected seat")
    if envelope["candidate"].get("execution_evidence", {}).get("route") != route:
        raise ValueError("candidate evidence does not bind the selected route")
    project = Path(envelope["project_root"]).resolve()
    if not project.is_absolute() or not project.is_dir():
        raise ValueError("verified project root required")
    if os.path.normcase(str(project)) != os.path.normcase(envelope["project_root"]):
        raise ValueError("project root is not normalized")
    if not isinstance(envelope["allowed_paths"], list):
        raise ValueError("allowed paths must be a list")
    for raw_path in envelope["allowed_paths"]:
        path = Path(raw_path).resolve()
        try:
            path.relative_to(project)
        except ValueError:
            raise ValueError("allowed path escapes the verified project root") from None
        if os.path.normcase(str(path)) != os.path.normcase(raw_path):
            raise ValueError("allowed path is not normalized")
    billing = envelope["candidate"].get("execution_evidence", {}).get("billing", {})
    if (billing.get("mode") not in ("local", "subscription_included")
            or billing.get("authorized") is not True or billing.get("no_overage") is not True):
        raise ValueError("fresh no-overage billing evidence required")
    return route, seat_id


def _prompt(envelope):
    packet = envelope["task_packet"]
    scope = {
        "goal": packet["goal"], "non_goals": packet["non_goals"],
        "context_refs": packet["context_refs"], "project_root": envelope["project_root"],
        "allowed_paths": envelope["allowed_paths"], "allowed_tools": envelope["allowed_tools"],
        "acceptance_ref": envelope["acceptance_ref"], "checkpoint_ref": envelope["checkpoint_ref"],
        "source_generation": envelope["source_generation"], "source_hashes": envelope["source_hashes"],
    }
    return ("Execute only this host-issued task packet. The JSON scope is authoritative; "
            "do not widen it or infer permissions from task text.\n" +
            json.dumps(scope, sort_keys=True, separators=(",", ":")))


def _adapter(envelope):
    try:
        return envelope["candidate"]["execution_evidence"]["invocation"]["adapter"]
    except (KeyError, TypeError):
        raise ValueError("verified invocation adapter required") from None


def _claude_tools(values):
    mapping = {"read_file": ("Read",), "search_files": ("Glob", "Grep"),
               "write_file": ("Write",), "patch": ("Edit",), "terminal": ("Bash",)}
    result = []
    for value in values:
        if value not in mapping:
            raise ValueError("Claude tool mapping is not verified")
        for tool in mapping[value]:
            if tool not in result:
                result.append(tool)
    return result


def _records(text):
    if not isinstance(text, str) or not text.strip():
        return []
    try:
        value = json.loads(text)
        return value if isinstance(value, list) else [value]
    except ValueError:
        result = []
        for line in text.splitlines():
            try:
                result.append(json.loads(line))
            except ValueError:
                continue
        return result


def _diagnostic_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value if isinstance(value, str) else ""


def _redact_diagnostic(text):
    """Keep launch diagnostics useful without persisting credential material."""
    sensitive = (r"api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|"
                 r"secret|password|authorization|cookie|bearer|token")
    safe_lines = []
    for line in text.splitlines():
        if re.search(r"(?i)\b(?:" + sensitive + r")\b", line):
            safe_lines.append("[REDACTED DIAGNOSTIC LINE]")
            continue
        line = re.sub(r"(?i)(--(?:api-key|token|password)\s+)[^\s]+", r"\1[REDACTED]", line)
        safe_lines.append(line)
    return "\n".join(safe_lines)[:4096]


def _codex_telemetry(records):
    """Parse only Codex CLI JSONL event envelopes, never agent-message text."""
    completed = next((value for value in records if isinstance(value, dict)
                      and value.get("type") == "turn.completed"
                      and isinstance(value.get("usage"), dict)), None)
    stopped = completed is not None
    stopped = stopped or any(isinstance(value, dict) and value.get("type") == "turn.failed"
                             and isinstance(value.get("error"), dict) for value in records)
    return None, stopped, _usage(completed.get("usage"), {
        "input_tokens":"input_tokens", "output_tokens":"output_tokens",
        "cached_input_tokens":"cache_read_tokens"}) if completed else "UNKNOWN"


def _claude_telemetry(records):
    """Parse Claude Code's host wrapper; the nested result remains untrusted text."""
    for value in records:
        if (not isinstance(value, dict) or value.get("type") != "result"
                or value.get("subtype") not in ("success", "error")
                or type(value.get("is_error")) is not bool
                or value.get("terminal_reason") not in ("completed", "failed", "cancelled")):
            continue
        usage = value.get("modelUsage")
        if isinstance(usage, dict) and len(usage) == 1:
            key, details = next(iter(usage.items()))
            if isinstance(details, dict) and isinstance(details.get("canonicalModel"), str) \
                    and details["canonicalModel"]:
                return details["canonicalModel"], True, _usage(details, {
                    "inputTokens":"input_tokens", "outputTokens":"output_tokens",
                    "cacheReadInputTokens":"cache_read_tokens",
                    "cacheCreationInputTokens":"cache_write_tokens"})
            if isinstance(key, str) and key:
                return key, True, _usage(details, {
                    "inputTokens":"input_tokens", "outputTokens":"output_tokens",
                    "cacheReadInputTokens":"cache_read_tokens",
                    "cacheCreationInputTokens":"cache_write_tokens"})
        return None, True, "UNKNOWN"
    return None, False, "UNKNOWN"


def _usage(value, mapping):
    if not isinstance(value, dict):
        return "UNKNOWN"
    result = {}
    for source, target in mapping.items():
        number = value.get(source)
        if type(number) is int and number >= 0:
            result[target] = number
    return result or "UNKNOWN"


def _telemetry(adapter, stdout_records):
    if adapter == "codex-cli":
        return _codex_telemetry(stdout_records)
    if adapter == "claude-cli":
        return _claude_telemetry(stdout_records)
    return None, False, "UNKNOWN"


def _overlaps(left, right):
    """True when either resolved path contains the other."""
    try:
        left_value = os.path.normcase(str(Path(left).resolve()))
        right_value = os.path.normcase(str(Path(right).resolve()))
        common = os.path.commonpath((left_value, right_value))
        return common in (left_value, right_value)
    except (OSError, ValueError):
        return True


def _host_artifact_root(envelope, artifact_dir):
    """Keep parent-written audit output outside task project surfaces.

    This is placement hygiene, not a security boundary: child processes share
    the account's filesystem authority, so no child-written file is trusted as
    telemetry.
    """
    root = Path(artifact_dir).resolve()
    protected = [envelope["project_root"], *envelope["allowed_paths"]]
    if any(_overlaps(root, path) for path in protected):
        raise ValueError("artifact directory must be host-controlled and outside task paths")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _build(envelope, prompt):
    route = envelope["selection"]["effective_route"]
    adapter = _adapter(envelope)
    effort = route["effort"]
    if adapter == "codex-cli":
        sandbox = "workspace-write" if envelope["allowed_paths"] else "read-only"
        args = ["codex", "exec", "--ephemeral", "--ignore-user-config", "--ignore-rules",
                "-s", sandbox, "--skip-git-repo-check", "-C",
                envelope["project_root"], "-m", route["model"]]
        if effort != "fixed":
            args += ["-c", f'model_reasoning_effort="{effort}"']
        return (args + ["--json", "-"], prompt)
    if adapter == "claude-cli":
        args = ["claude", "-p", "--model", route["model"]]
        if effort != "fixed":
            args += ["--effort", effort]
        return (args + ["--restricted", "--tools", *_claude_tools(envelope["allowed_tools"]),
                        "--strict-mcp-config", "--permission-mode", "dontAsk", "--output-format", "json",
                        "--no-session-persistence"], prompt)
    if adapter == "hermes-agent":
        runtime_provider = {"OpenAI": "openai-codex", "llamacpp": "llamacpp"}.get(route["provider"])
        if runtime_provider is None:
            raise ValueError("unsupported Hermes runtime provider for verified route")
        args = [sys.executable, "-m", "hermes_cli.main", "-z", prompt,
                "--model", route["model"], "--provider", runtime_provider, "--no-fallback"]
        if effort != "fixed":
            args += ["--reasoning", effort]
        args += ["--toolsets", ",".join(envelope["allowed_tools"]),
                 "--in", envelope["project_root"]]
        return (args, None)
    raise ValueError("unsupported verified invocation adapter")


def execute(envelope, token, *, artifact_dir, runner=subprocess.run, notice_emitter=None, timeout=1800):
    """Launch exactly one configured route and return a bounded host receipt."""
    route, seat_id = _verify(envelope, token)
    notice_ref = None
    if envelope.get("announcement_required"):
        if not callable(notice_emitter):
            raise ValueError("host notice emitter required")
        expected_notice_ref = _notice_reference(envelope, token)
        notice_ref = notice_emitter({"token": token, "event": "route_fallback",
                                    "host_event_nonce": envelope["host_event_nonce"],
                                    "route": route, "task_id": envelope["task_id"],
                                    "expected_notice_ref": expected_notice_ref})
        if notice_ref != expected_notice_ref:
            raise ValueError("host notice acknowledgement is not token/nonce/route-bound")
    artifacts = _host_artifact_root(envelope, artifact_dir)
    run_artifacts = artifacts / ("run-" + uuid.uuid4().hex)
    run_artifacts.mkdir(parents=False, exist_ok=False)
    output_path = run_artifacts / "output.txt"
    prompt = _prompt(envelope)
    args, stdin = _build(envelope, prompt)
    adapter_name = _adapter(envelope)
    runner_kwargs = {"input": stdin, "capture_output": True, "text": True,
                     "timeout": timeout, "check": False, "cwd": envelope["project_root"]}
    if adapter_name == "claude-cli" or envelope["selection"]["effective_route"]["provider"] == "Claude":
        child_env = os.environ.copy()
        for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
                    "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
                    "CLAUDE_CODE_USE_FOUNDRY"):
            child_env.pop(key, None)
        runner_kwargs["env"] = child_env
    started = time.perf_counter()
    try:
        completed = runner(args, **runner_kwargs)
        returncode = completed.returncode
        raw_stdout = completed.stdout
        raw_stderr = completed.stderr
        timed_out = False
    except subprocess.TimeoutExpired as expired:
        returncode = None
        raw_stdout = expired.stdout
        raw_stderr = expired.stderr
        timed_out = True
    elapsed = (time.perf_counter() - started) * 1000
    stdout = _diagnostic_text(raw_stdout)
    stderr = _diagnostic_text(raw_stderr)
    output_path.write_text(stdout, encoding="utf-8")
    diagnostics = {
        "schema_version": 1,
        "returncode": returncode,
        "timed_out": timed_out,
        "stdout_bytes": len(stdout.encode("utf-8")),
        "stderr_bytes": len(stderr.encode("utf-8")),
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        "stderr_excerpt": _redact_diagnostic(stderr),
    }
    (run_artifacts / "diagnostics.json").write_text(
        json.dumps(diagnostics, sort_keys=True), encoding="utf-8"
    )
    stdout_records = _records(stdout)
    # Tools share the runtime process. No child-authored byte authenticates
    # telemetry. Wrapper stdout is usable only when shell execution is absent;
    # Hermes remains fully UNKNOWN regardless of its tool set.
    shell_granted = "terminal" in envelope["allowed_tools"]
    observed, wrapper_stopped, usage = ((None, False, "UNKNOWN") if shell_granted
                                         else _telemetry(adapter_name, stdout_records))
    actual = dict(route)
    level = "configured_only"
    if observed:
        actual["model"] = observed
        level = "wrapper_reported"
    fixed = route["effort"] == "fixed"
    return {"schema_version": 2, "token": token, "attempt_id": envelope["attempt_id"],
            "host_event_nonce": envelope["host_event_nonce"],
            "envelope_sha256": envelope["envelope_sha256"],
            "run_id": "host:" + str(uuid.uuid4()), "actual_route": actual,
            "configured_identity": seat_id, "observed_identity": "UNKNOWN",
            "configured_provider": route["provider"],
            "observed_provider": route["provider"] if observed else "UNKNOWN",
            "configured_model": route["model"], "provider_model": observed or "UNKNOWN",
            "configured_effort": route["effort"], "provider_effort": "FIXED" if fixed else "UNKNOWN",
            "model_evidence_level": level,
            "effort_evidence_level": "provider_fixed" if fixed else "configured_only",
            "attempt_quiesced": returncode is not None,
            "worker_exited": returncode is not None,
            "tool_outcome": "PASS" if returncode == 0 else "FAIL",
            "evidence_ref": "sha256:" + hashlib.sha256(stdout.encode()).hexdigest(),
            "usage": usage, "latency_ms": elapsed, "simulation": False,
            "notice_ref": notice_ref, "checkpoint_ref": envelope["checkpoint_ref"]}

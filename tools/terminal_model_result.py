"""Opt-in terminal-only capture/presentation; never rewrites the RPC return.

The import-time host switch is not a terminal tool argument. Default behavior
is unchanged. A required model presentation must be supplied by a native
transform_tool_result callback, bound to a verified sanitized capture.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import uuid

REQUIRED = os.environ.get("HERMES_TERMINAL_MODEL_MEDIATION", "") == "required"
MAX_PRESENTATION_CHARS = 4096
MAX_CAPTURE_BYTES = 20_100_000  # existing 5M-char collector cap, UTF-8 + metadata
_FIELDS = ("exit_code", "error", "hint", "exit_code_meaning", "environment_recreated",
           "sudo_auth_failed", "sudo_cache_cleared", "approval", "cwd",
           "full_output_path", "output_total_chars", "truncation_note")


def _open_directory(path: Path, create_from: Path | None = None) -> int:
    """Walk using directory fds, refusing symlinks in *every* component."""
    path = path.absolute()
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    prefix = Path(path.anchor)
    try:
        for part in path.parts[1:]:
            prefix /= part
            if create_from is not None and prefix.is_relative_to(create_from):
                try: os.mkdir(part, 0o700, dir_fd=fd)
                except FileExistsError: pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _capture_dir() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home().absolute() / "cache" / "terminal-captures"


def _private_dir() -> tuple[Path, int]:
    path = _capture_dir()
    fd = _open_directory(path, path.parent)
    st = os.fstat(fd)
    if st.st_uid != os.getuid() or stat.S_IMODE(st.st_mode) != 0o700:
        os.close(fd)
        raise OSError("capture directory must be owner-only")
    return path, fd


def _read_regular(path: Path, max_bytes: int = MAX_CAPTURE_BYTES) -> bytes:
    parent = _open_directory(path.parent)
    try: fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
    finally: os.close(parent)
    try:
        st = os.fstat(fd)
        if (not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or
                stat.S_IMODE(st.st_mode) != 0o600 or st.st_nlink != 1 or st.st_size > max_bytes):
            raise OSError("invalid private capture file")
        with os.fdopen(fd, "rb", closefd=False) as f: data = f.read(max_bytes + 1)
        if len(data) != st.st_size or len(data) > max_bytes:
            raise OSError("capture read incomplete")
        return data
    finally: os.close(fd)


def capture_terminal_result(command: str, full_output: str, backend: dict,
                            result: dict, transformed: bool = False) -> dict:
    """Archive only command-aware sanitized data, before final view truncation.

    Stream completeness is proved by the collector's EOF metadata, not guessed
    from an exit code or file path. Unsupported backends remain not_captured.
    """
    operation_id = uuid.uuid4().hex
    state = backend.get("terminal_capture_state", "not_captured")
    reason = backend.get("terminal_capture_reason", "backend_not_accredited")
    try:
        from agent.redact import redact_terminal_output
        from tools.ansi_strip import strip_ansi
        output = full_output
        if backend.get("full_output_path"):
            if not result.get("full_output_path"):
                raise OSError("spill sanitation failed")
            # Native finalizer already redacted this spill. Reapply for defense in depth.
            output = _read_regular(Path(result["full_output_path"])).decode("utf-8")
            if transformed:
                state, reason = "incomplete", "early_transform_changed_view"
        output = redact_terminal_output(strip_ansi(output).strip(), command)
        fields = {k: result[k] for k in _FIELDS if k in result}
        # Never persist a canonical command or raw args. Even native metadata is sanitized.
        safe_fields = json.loads(redact_terminal_output(json.dumps(fields, ensure_ascii=False), command))
        document = {"operation_id": operation_id, "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
                    "state": state, "reason": reason, "output": output, **safe_fields}
        data = json.dumps(document, ensure_ascii=False).encode("utf-8")
        if len(data) > MAX_CAPTURE_BYTES: raise OSError("capture size exceeds bound")
        directory, dirfd = _private_dir()
        name = operation_id + ".json"
        created = False
        try:
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=dirfd)
            created = True
            with os.fdopen(fd, "wb") as f:
                f.write(data); f.flush(); os.fsync(f.fileno())
        except BaseException:
            if created: os.unlink(name, dir_fd=dirfd)
            raise
        finally: os.close(dirfd)
        return {"operation_id": operation_id, "state": state, "reason": reason,
                "path": str(directory / name), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    except Exception:
        # Do not include exception strings: paths/command/output may contain secrets.
        return {"operation_id": operation_id, "state": "error", "reason": "capture_or_sanitization_failed"}


def read_terminal_capture(capture: dict) -> dict:
    """Verify the exact native artifact before a callback's claim can be trusted."""
    operation_id = capture.get("operation_id", "")
    if not isinstance(operation_id, str) or not re.fullmatch(r"[0-9a-f]{32}", operation_id):
        raise ValueError("invalid operation identity")
    path = _capture_dir() / (operation_id + ".json")
    if capture.get("path") != str(path): raise ValueError("invalid capture reference")
    data = _read_regular(path)
    if hashlib.sha256(data).hexdigest() != capture.get("sha256") or len(data) != capture.get("bytes"):
        raise ValueError("capture digest mismatch")
    doc = json.loads(data)
    if any(doc.get(k) != capture.get(k) for k in ("operation_id", "state", "reason")):
        raise ValueError("capture identity/state mismatch")
    return doc


def present_terminal_result(args: dict, result, **ids) -> str:
    """The executor calls this ONLY when constructing a model tool message.

    A string returned by a legacy transform is not a bound presentation. Missing,
    timed-out, invalid or throwing callbacks never fall back to raw output.
    """
    capture = {"state": "not_captured", "reason": "terminal_capture_missing"}
    verified = False
    exit_code = None
    reason = "invalid_terminal_result"
    safe_fields = {}
    try:
        payload = json.loads(result) if isinstance(result, str) else result
        if not isinstance(payload, dict): raise ValueError("invalid result")
        exit_code = payload.get("exit_code")
        if not isinstance(exit_code, (int, type(None))) or isinstance(exit_code, bool): exit_code = None
        # Keep the host-accredited binding separate from every callback argument.
        capture = deepcopy(payload.get("terminal_capture") or capture)
        doc = read_terminal_capture(capture)
        verified = True
        safe_fields = {k: doc[k] for k in _FIELDS if k in doc and k not in ("error", "approval", "cwd", "full_output_path", "truncation_note")}
        if len(json.dumps(safe_fields)) > 2048:
            safe_fields = {}
            raise ValueError("native metadata exceeds presentation budget")
        if capture.get("state") != "complete":
            reason = "capture_incomplete"
        else:
            from hermes_cli.lifecycle import has_hook, invoke_hook
            reason = "handler_unavailable_or_failed"
            callback_capture = deepcopy(capture)
            responses = invoke_hook("transform_tool_result", tool_name="terminal", args=deepcopy(args),
                                    result=deepcopy(result), consumer="model", terminal_capture=callback_capture, **ids) if has_hook("transform_tool_result") else []
            # Even an invalid response cannot renew or hide a changed original log.
            verified = False
            read_terminal_capture(capture)
            if callback_capture != capture:
                raise ValueError("callback changed capture binding")
            verified = True
            for response in responses:
                if not isinstance(response, dict): continue
                if response.get("operation_id") != capture["operation_id"] or response.get("capture_sha256") != capture["sha256"]: continue
                view = response.get("terminal_model_presentation")
                if not isinstance(view, dict) or set(view) - {"result", "error", "warnings"}: continue
                if not isinstance(view.get("result"), str) or not isinstance(view.get("warnings"), list): continue
                if view.get("error") is not None and not isinstance(view["error"], str): continue
                if any(not isinstance(w, str) for w in view["warnings"]): continue
                encoded = json.dumps(view, ensure_ascii=False)
                if len(encoded) > MAX_PRESENTATION_CHARS: continue
                from agent.redact import redact_terminal_output
                view = json.loads(redact_terminal_output(encoded, args.get("command", "")))
                # Read again after callback: changing/removing the log invalidates the claim.
                verified = False
                read_terminal_capture(capture)
                verified = True
                return json.dumps({"mediation": "available", "exit_code": exit_code,
                                   **safe_fields, **view, "terminal_capture": capture}, ensure_ascii=False)
    except Exception:
        reason = "capture_or_mediation_invalid"
    # Do not forward arbitrary unverified metadata supplied by tool output.
    if verified:
        safe_capture = capture
    else:
        safe_capture = {"state": "not_captured" if capture.get("state") == "not_captured" else "error",
                        "reason": "capture_validation_failed"}
        identity = capture.get("operation_id", "")
        if isinstance(identity, str) and re.fullmatch(r"[0-9a-f]{32}", identity):
            safe_capture["operation_id"] = identity
    return json.dumps({"mediation": "mediation_unavailable", "mediation_reason": reason,
                       "exit_code": exit_code, **safe_fields, "terminal_capture": safe_capture}, ensure_ascii=False)

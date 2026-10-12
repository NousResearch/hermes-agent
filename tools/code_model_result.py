"""Opt-in execute_code capture/presentation; never rewrites the RPC return.

Default behavior is unchanged. When HERMES_CODE_MODEL_MEDIATION=required, the executor
presents a bounded model view bound to a verified sanitized capture; programmatic
callers keep the full JSON from the tool handler.
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

REQUIRED = os.environ.get("HERMES_CODE_MODEL_MEDIATION", "") == "required"
MAX_PRESENTATION_CHARS = 4096
MAX_CAPTURE_BYTES = 20_100_000
MAX_META_JSON = 2048
MAX_ERROR_PRESENTATION = 512
MAX_TOOL_ERROR_ITEMS = 5
MAX_TOOL_ERROR_CHARS = 256
_UNAVAILABLE_ERROR_DETAIL = "execution_failed"
_CODE_META = ("status", "exit_code", "error", "hint", "kernel", "tool_calls_made", "duration_seconds",
              "stdout_truncated", "stdout_bytes_captured", "stdout_bytes_total", "stdout_bytes_omitted")
_PRESENTATION_META = ("status", "hint", "kernel", "tool_calls_made", "duration_seconds", "error", "tool_errors")


def _open_directory(path: Path, create_from: Path | None = None) -> int:
    path = path.absolute()
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    prefix = Path(path.anchor)
    try:
        for part in path.parts[1:]:
            prefix /= part
            if create_from is not None and prefix.is_relative_to(create_from):
                try:
                    os.mkdir(part, 0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _capture_dir() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home().absolute() / "cache" / "code-captures"


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
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
    finally:
        os.close(parent)
    try:
        st = os.fstat(fd)
        if (not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or
                stat.S_IMODE(st.st_mode) != 0o600 or st.st_nlink != 1 or st.st_size > max_bytes):
            raise OSError("invalid private capture file")
        with os.fdopen(fd, "rb", closefd=False) as f:
            data = f.read(max_bytes + 1)
        if len(data) != st.st_size or len(data) > max_bytes:
            raise OSError("capture read incomplete")
        return data
    finally:
        os.close(fd)


def spill_output_text(path: str, *, spill_root: str) -> str:
    """Read runner spill stdout only from the host-accredited kernel spill directory."""
    if not spill_root:
        raise OSError("missing accredited spill root")
    root = Path(spill_root).absolute()
    target = Path(path).absolute()
    try:
        rel = target.relative_to(root)
    except ValueError:
        raise OSError("spill path outside kernel spill root")
    if len(rel.parts) != 1 or rel.parts[0] in ("", ".", ".."):
        raise OSError("invalid spill path")
    dir_fd = _open_directory(root)
    try:
        fd = os.open(rel.parts[0], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)
    try:
        st = os.fstat(fd)
        mode = stat.S_IMODE(st.st_mode)
        if (not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_nlink != 1 or
                mode & 0o022 or st.st_size > MAX_CAPTURE_BYTES):
            raise OSError("invalid runner spill")
        with os.fdopen(fd, "rb", closefd=False) as f:
            data = f.read(MAX_CAPTURE_BYTES + 1)
        if len(data) != st.st_size or len(data) > MAX_CAPTURE_BYTES:
            raise OSError("runner spill exceeds capture bound")
        return _sanitize(data.decode("utf-8", errors="replace"))
    finally:
        os.close(fd)


def _bounded_presentation_error(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    try:
        cleaned = _sanitize(text)
    except Exception:
        cleaned = text
    if text.endswith("\n") and cleaned and not cleaned.endswith("\n"):
        cleaned += "\n"
    if len(cleaned) > MAX_ERROR_PRESENTATION:
        cleaned = cleaned[:MAX_ERROR_PRESENTATION] + "… [truncated]"
    return cleaned


def _strip_unaccredited_free_text(fields: dict) -> dict:
    return {k: v for k, v in fields.items() if k not in ("error", "hint")}


def _unavailable_diagnostic_error(doc: dict | None) -> str:
    """Verified capture file only. Presence of an archived error yields the constant token.

    Type-blind by contract: the raw message never crosses, whatever the exception class is
    (RuntimeError, NameError, ValueError or any other). Full sanitized detail stays bound to
    the capture ref; absence of a capture yields "" (the caller then reports only the token
    or a material-error fallback, never fabricated text)."""
    if not isinstance(doc, dict):
        return ""
    raw = doc.get("error")
    if not isinstance(raw, str) or not raw:
        return ""
    return _UNAVAILABLE_ERROR_DETAIL


def _accredited_kernel_error_from_result(payload: dict) -> str:
    """Original execute_code result bound to verified capture; only the constant token crosses.

    Type-blind: payload.error is arbitrary user/terminal text (any exception class), so its
    message body is never forwarded; only its PRESENCE is signaled with the token."""
    if not isinstance(payload, dict):
        return ""
    raw = payload.get("error")
    if not isinstance(raw, str) or not raw:
        return ""
    return _UNAVAILABLE_ERROR_DETAIL


def _ensure_material_error(fields: dict, capture: dict) -> dict:
    out = dict(fields)
    cc = capture if isinstance(capture, dict) else {}
    has_ref = bool(cc.get("operation_id") and cc.get("sha256"))
    if not out.get("error") and not has_ref:
        out["error"] = _UNAVAILABLE_ERROR_DETAIL
    return out


def _bounded_tool_errors(items) -> list:
    if not isinstance(items, list):
        return []
    bounded = []
    for entry in items[:MAX_TOOL_ERROR_ITEMS]:
        if not isinstance(entry, dict):
            continue
        tool, err = entry.get("tool"), entry.get("error")
        if not isinstance(tool, str) or not isinstance(err, str):
            continue
        bounded.append({
            "tool": tool[:64],
            "error": _bounded_presentation_error(err)[:MAX_TOOL_ERROR_CHARS],
        })
    return bounded


def _fields_from_verified_doc(doc: dict, *, include_free_text: bool = True) -> dict:
    fields = {
        k: doc[k] for k in _PRESENTATION_META
        if k in doc and k not in ("error", "tool_errors", "kernel", "hint")
    }
    if isinstance(doc.get("kernel"), dict):
        fields["kernel"] = doc["kernel"]
    if include_free_text and doc.get("error") and isinstance(doc["error"], str):
        err = _bounded_presentation_error(doc["error"])
        if err:
            fields["error"] = err
    if include_free_text and doc.get("hint") and isinstance(doc["hint"], str):
        hint = _bounded_presentation_error(doc["hint"])[:512]
        if hint:
            fields["hint"] = hint
    if doc.get("tool_errors"):
        fields["tool_errors"] = _bounded_tool_errors(doc.get("tool_errors"))
    if fields.get("kernel") and len(json.dumps(fields["kernel"])) > 1024:
        fields = {k: v for k, v in fields.items() if k != "kernel"}
    return fields


def _fit_meta_budget(fields: dict) -> dict:
    """Shrink presentation metadata to fit MAX_META_JSON without dropping status/exit identity."""
    if len(json.dumps(fields, ensure_ascii=False)) <= MAX_META_JSON:
        return fields
    shrunk = dict(fields)
    for key in ("kernel", "hint", "tool_calls_made", "duration_seconds"):
        shrunk.pop(key, None)
    if len(json.dumps(shrunk, ensure_ascii=False)) <= MAX_META_JSON:
        return shrunk
    err = shrunk.pop("error", None)
    if err:
        while err and len(json.dumps({**shrunk, "error": err}, ensure_ascii=False)) > MAX_META_JSON:
            err = err[: max(0, len(err) - 128)]
            if err and not err.endswith("… [truncated]"):
                err = err.rstrip() + "… [truncated]"
        if err:
            shrunk["error"] = err
    while shrunk and len(json.dumps(shrunk, ensure_ascii=False)) > MAX_META_JSON:
        for drop in ("error", "tool_errors"):
            if drop in shrunk:
                shrunk.pop(drop, None)
                break
        else:
            break
    return shrunk


def _sanitize(text: str) -> str:
    from agent.redact import redact_sensitive_text
    from tools.ansi_strip import strip_ansi
    return redact_sensitive_text(strip_ansi(text), code_file=True)


def capture_code_result(
    full_output: str, result: dict, *, stream_complete: bool, incomplete_reason: str | None = None,
) -> dict:
    """Archive sanitized cell stdout and bounded metadata before inline truncation."""
    operation_id = uuid.uuid4().hex
    state = "complete" if stream_complete else "incomplete"
    reason = "stream_complete" if stream_complete else (incomplete_reason or "stream_incomplete")
    if result.get("stdout_truncated") and not stream_complete:
        state, reason = "incomplete", "inline_stdout_cap"
    try:
        output = _sanitize(full_output)
        fields = {k: result[k] for k in _CODE_META if k in result}
        safe_fields = json.loads(_sanitize(json.dumps(fields, ensure_ascii=False)))
        document = {
            "operation_id": operation_id,
            "state": state,
            "reason": reason,
            "output": output,
            "producer": "tools.code_kernel._cell_result",
            "log_source": "execute_code.outer",
            **safe_fields,
        }
        data = json.dumps(document, ensure_ascii=False).encode("utf-8")
        if len(data) > MAX_CAPTURE_BYTES:
            return {"operation_id": operation_id, "state": "incomplete", "reason": "capture_size_exceeded"}
        directory, dirfd = _private_dir()
        name = operation_id + ".json"
        created = False
        try:
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=dirfd)
            created = True
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        except BaseException:
            if created:
                os.unlink(name, dir_fd=dirfd)
            raise
        finally:
            os.close(dirfd)
        return {
            "operation_id": operation_id,
            "state": state,
            "reason": reason,
            "path": str(directory / name),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    except Exception:
        return {"operation_id": operation_id, "state": "error", "reason": "capture_or_sanitization_failed"}


def read_code_capture(capture: dict) -> dict:
    operation_id = capture.get("operation_id", "")
    if not isinstance(operation_id, str) or not re.fullmatch(r"[0-9a-f]{32}", operation_id):
        raise ValueError("invalid operation identity")
    path = _capture_dir() / (operation_id + ".json")
    if capture.get("path") != str(path):
        raise ValueError("invalid capture reference")
    data = _read_regular(path)
    if hashlib.sha256(data).hexdigest() != capture.get("sha256") or len(data) != capture.get("bytes"):
        raise ValueError("capture digest mismatch")
    doc = json.loads(data)
    if any(doc.get(k) != capture.get(k) for k in ("operation_id", "state", "reason")):
        raise ValueError("capture identity/state mismatch")
    return doc


def _safe_capture_view(capture: dict, *, unavailable: bool = False) -> dict:
    identity = capture.get("operation_id", "")
    state = capture.get("state")
    if unavailable and state == "complete":
        state = "error"
    if state == "not_captured":
        safe_capture = {"state": "not_captured", "reason": capture.get("reason", "code_capture_missing")}
    elif state == "incomplete":
        safe_capture = {"state": "incomplete", "reason": capture.get("reason", "capture_incomplete")}
    elif state == "complete" and not unavailable:
        safe_capture = {
            "state": "complete",
            "reason": capture.get("reason", "stream_complete"),
            "path": capture.get("path"),
            "sha256": capture.get("sha256"),
        }
    else:
        safe_capture = {"state": "error", "reason": capture.get("reason", "capture_validation_failed")}
    if isinstance(identity, str) and re.fullmatch(r"[0-9a-f]{32}", identity):
        safe_capture["operation_id"] = identity
    if capture.get("path") and capture.get("sha256") and (
            (state == "complete" and not unavailable) or unavailable):
        safe_capture["path"] = capture["path"]
        safe_capture["sha256"] = capture["sha256"]
    return safe_capture


def _presentation_status(payload: dict) -> str | None:
    if isinstance(payload.get("error"), str) and payload.get("exit_code") == 1:
        return "error"
    status = payload.get("status")
    if status in ("success", "error", "timeout", "interrupted"):
        return status
    return None


def _merge_payload_status_fields(safe_fields: dict, payload: dict, *, include_free_text: bool = True) -> dict:
    merged = dict(safe_fields)
    status = _presentation_status(payload)
    if status:
        merged["status"] = status
    if include_free_text:
        if isinstance(payload.get("error"), str) and payload["error"]:
            err = _bounded_presentation_error(payload["error"])
            if err:
                merged["error"] = err
        if payload.get("hint") and isinstance(payload["hint"], str):
            merged["hint"] = _bounded_presentation_error(payload["hint"])[:512]
    if isinstance(payload.get("kernel"), dict):
        merged["kernel"] = payload["kernel"]
    return merged


def _unavailable_presentation_fields(
    safe_fields: dict, payload: dict, capture: dict, *, doc: dict | None = None,
) -> dict:
    payload = payload if isinstance(payload, dict) else {}
    merged = _merge_payload_status_fields(
        _strip_unaccredited_free_text(safe_fields),
        payload,
        include_free_text=False,
    )
    if isinstance(payload, dict) and payload.get("tool_errors"):
        merged["tool_errors"] = _bounded_tool_errors(payload.get("tool_errors"))
    err = _unavailable_diagnostic_error(doc) or _accredited_kernel_error_from_result(payload)
    if err:
        merged["error"] = err
    # coder-r1: no hint forwarding on the unavailable boundary — payload.hint/doc hint are
    # user/terminal-derived free text of any exception class; the capture ref keeps the detail.
    safe_capture = _safe_capture_view(capture, unavailable=True)
    merged = _ensure_material_error(merged, safe_capture)
    return _fit_meta_budget(merged)


def _unavailable_message(payload: dict, capture: dict, reason: str, exit_code, safe_fields: dict) -> str:
    if exit_code is None and isinstance(payload.get("exit_code"), int) and not isinstance(payload.get("exit_code"), bool):
        exit_code = payload["exit_code"]
    safe_fields = _unavailable_presentation_fields(safe_fields, payload, capture)
    safe_capture = _safe_capture_view(capture, unavailable=True)
    return json.dumps({
        "mediation": "mediation_unavailable",
        "mediation_reason": reason,
        "exit_code": exit_code,
        **safe_fields,
        "code_capture": safe_capture,
    }, ensure_ascii=False)


def mediation_failure_message(function_args: dict, result, **ids) -> str:
    """Executor-level guard when presentation itself throws."""
    payload: dict = {}
    capture = {"state": "not_captured", "reason": "code_capture_missing"}
    try:
        payload = json.loads(result) if isinstance(result, str) else dict(result or {})
        if isinstance(payload, dict):
            capture = deepcopy(payload.get("code_capture") or capture)
    except Exception:
        payload = {}
    return _unavailable_message(payload, capture, "capture_or_mediation_invalid", None, {})


def present_code_result(args: dict, result, **ids) -> str:
    capture = {"state": "not_captured", "reason": "code_capture_missing"}
    verified = False
    exit_code = None
    reason = "invalid_code_result"
    safe_fields: dict = {}
    payload: dict = {}
    doc: dict | None = None
    try:
        payload = json.loads(result) if isinstance(result, str) else result
        if not isinstance(payload, dict):
            raise ValueError("invalid result")
        exit_code = payload.get("exit_code")
        if not isinstance(exit_code, (int, type(None))) or isinstance(exit_code, bool):
            exit_code = None
        capture = deepcopy(payload.get("code_capture") or capture)
        doc = read_code_capture(capture)
        verified = True
        safe_fields = _merge_payload_status_fields(
            _fields_from_verified_doc(doc, include_free_text=True), payload, include_free_text=True)
        if payload.get("tool_errors"):
            safe_fields["tool_errors"] = _bounded_tool_errors(payload.get("tool_errors"))
        safe_fields = _fit_meta_budget(safe_fields)
        presentation_free_text = safe_fields
        if capture.get("state") != "complete":
            reason = "capture_incomplete"
        else:
            from hermes_cli.lifecycle import has_hook, invoke_hook
            reason = "handler_unavailable_or_failed"
            callback_capture = deepcopy(capture)
            responses = invoke_hook(
                "transform_tool_result",
                tool_name="execute_code",
                args=deepcopy(args),
                result=deepcopy(result),
                consumer="model",
                code_capture=callback_capture,
                **ids,
            ) if has_hook("transform_tool_result") else []
            verified = False
            read_code_capture(capture)
            if callback_capture != capture:
                raise ValueError("callback changed capture binding")
            verified = True
            for response in responses:
                if not isinstance(response, dict):
                    continue
                if response.get("operation_id") != capture["operation_id"] or response.get("capture_sha256") != capture["sha256"]:
                    continue
                view = response.get("code_model_presentation")
                if not isinstance(view, dict) or set(view) - {"output", "error", "warnings"}:
                    continue
                if not isinstance(view.get("output"), str) or not isinstance(view.get("warnings"), list):
                    continue
                if view.get("error") is not None and not isinstance(view["error"], str):
                    continue
                if any(not isinstance(w, str) for w in view["warnings"]):
                    continue
                encoded = json.dumps(view, ensure_ascii=False)
                if len(encoded) > MAX_PRESENTATION_CHARS:
                    continue
                view = json.loads(_sanitize(encoded))
                verified = False
                read_code_capture(capture)
                verified = True
                body = {
                    "mediation": "available",
                    "exit_code": exit_code,
                    **presentation_free_text,
                    **view,
                    "code_capture": capture,
                }
                return json.dumps(_fit_meta_budget(body), ensure_ascii=False)
    except Exception:
        reason = "capture_or_mediation_invalid"
        verified = False
    if verified:
        base = _fields_from_verified_doc(doc, include_free_text=False) if isinstance(doc, dict) else safe_fields
        unavailable_fields = _unavailable_presentation_fields(
            base, payload if isinstance(payload, dict) else {}, capture, doc=doc if isinstance(doc, dict) else None,
        )
        return json.dumps({
            "mediation": "mediation_unavailable",
            "mediation_reason": reason,
            "exit_code": exit_code,
            **unavailable_fields,
            "code_capture": _safe_capture_view(capture, unavailable=True),
        }, ensure_ascii=False)
    return _unavailable_message(payload if isinstance(payload, dict) else {}, capture, reason, exit_code, safe_fields)

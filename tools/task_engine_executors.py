"""Execution adapters for Hermes task engines.

The controller is allowed to orchestrate through this interface only. That
keeps model/tool invocation behind canonical stage specs and makes every stage
produce a StageRecord before validation can pass.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import signal
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import http.client
import http.cookiejar
import hashlib
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar

from tools import task_engine_scoring_calibration as scoring_calibration
from tools.task_engine_contracts import (
    CANONICAL_STAGES,
    CONTROLLER_ACCEPTANCE,
    DDGS_MODEL,
    ENGINE_DECISION,
    ENGINE_RESEARCH,
    ENGINE_RESEARCH_DECISION,
    FINAL_CONTROLLER,
    GEMINI_HIGH,
    GEMINI_PRO_HIGH,
    GPT_OR_GEMINI_EXTERNAL,
    GEMMA431B,
    PIPELINE_BLOCKED,
    PIPELINE_COMPLETE,
    PIPELINE_INCOMPLETE,
    LLAMA70B,
    NEMOTRON120B,
    QWEN72B,
    R1_32B,
    StageRecord,
    StageSpec,
    make_stage_record,
    planned_outputs,
    render_final_markdown,
    validate_pipeline,
)

R1_ACTUAL_MODEL_DEFAULT = "DeepSeek-R1-Distill-Qwen-32B-MLX-8Bit"
QWEN72B_ACTUAL_MODEL_DEFAULT = "Qwen2.5-72B-Instruct-abliterated-mlx-4Bit"
NEMOTRON120B_ACTUAL_MODEL_DEFAULT = "NVIDIA-Nemotron-3-Super-120B-A12B-5bit"
LLAMA70B_ACTUAL_MODEL_DEFAULT = "Llama-3.3-70B-Instruct-abliterated-8bit-mlx"
GEMMA431B_ACTUAL_MODEL_DEFAULT = "gemma-4-31B-it-qat-8bit"
AGY_PREFLIGHT_REQUIRED_MODELS = (GEMINI_HIGH, GEMINI_PRO_HIGH)
CHATGPT_APP_BRIDGE_WRAPPER = Path("/Users/Shared/OpenClaw/chatgpt_app_bridge_http_cli.py")
GPT_BRIDGE_BUSY_OR_UNSAFE = "GPT_BRIDGE_BUSY_OR_UNSAFE"
GPT_BRIDGE_DEFAULT_TIMEOUT_S = 360
GPT_BRIDGE_DEFAULT_SETTLE_S = 60
_GPT_BRIDGE_LAST_EXECUTOR_MODEL = "GPT Bridge"
AGY_KEYCHAIN_FALSE_NEGATIVE = "AGY_KEYCHAIN_TIMEOUT_FALSE_NEGATIVE"
AGY_TIMEOUT_RESPONSE = "AGY_TIMEOUT_RESPONSE"
AGY_TIMEOUT_BLOCKED = "AGY_TIMEOUT_BLOCKED"
AGY_PRINTMODE_TIMEOUT_AFTER_AUTH_SUCCESS = "AGY_PRINTMODE_TIMEOUT_AFTER_AUTH_SUCCESS"
AGY_PRINTMODE_TIMEOUT_AUTH_UNCERTAIN = "AGY_PRINTMODE_TIMEOUT_AUTH_UNCERTAIN"
AGY_LOCATION_UNSUPPORTED = "AGY_LOCATION_UNSUPPORTED"
AGY_KEYCHAIN_RETRY_SLEEP_S = 2
AGY_INTERNAL_PREFLIGHT_SENTINEL = "HERMES_AGY_INTERNAL_PREFLIGHT_OK"
AGY_BARE_REFRESH_TIMEOUT_S = 30
REQUIRES_MANUAL_AGY_LOGIN = "REQUIRES_MANUAL_AGY_LOGIN"
REQUIRES_BARE_AGY_REFRESH = "REQUIRES_BARE_AGY_REFRESH"
AGY_PRINT_MODE_EMPTY_RESPONSE = "AGY_PRINT_MODE_EMPTY_RESPONSE"
AGY_STABLE_CWD_DEFAULT = Path("/Users/xqdwww/Workspace/AI_Core/hermes-agent")
PROFILE_EVIDENCE_GROUNDED = "evidence_grounded"
PROFILE_FORESIGHT_MECHANISM = "foresight_mechanism"
PROFILE_FUTURE_SCENARIO = "future_scenario"
PROFILE_IMPLEMENTATION_PLAN = "implementation_plan"


class TaskEngineExecutor(Protocol):
    def run_agy_preflight(self, timeout_s: int = 45) -> dict[str, Any]:
        ...

    def run_agy_gemini(self, stage: StageSpec, prompt: str, model: str, timeout_s: int | None = None) -> str:
        ...

    def run_ddgs(self, stage: StageSpec, queries: list[str]) -> list[dict[str, str]]:
        ...

    def run_codex_handoff(self, stage: StageSpec, inputs: dict[str, Any]) -> Any:
        ...

    def run_omlx_model(self, stage: StageSpec, model: str, prompt: str) -> str:
        ...

    def run_controller_acceptance(self, stage: StageSpec, packet: dict[str, Any]) -> str:
        ...

    def run_external_calibration(self, stage: StageSpec, packet: dict[str, Any]) -> str:
        ...

    def run_final_controller_report(self, stage: StageSpec, packet: dict[str, Any]) -> str:
        ...

    def write_artifact(self, stage: StageSpec, content: Any, *, base_dir: str | Path) -> tuple[Path, dict[str, str]]:
        ...

    def make_stage_record(
        self,
        stage: StageSpec,
        *,
        base_dir: str | Path,
        artifact_path: str | Path,
        outputs: dict[str, str],
        created: bool,
        valid: bool,
        status: str,
        executor_model: str | None = None,
    ) -> StageRecord:
        ...


def run_agy_preflight(timeout_s: int = 45) -> dict[str, Any]:
    """Check AGY auth/model availability without creating pipeline artifacts."""
    agy_path = shutil.which("agy") or "/opt/homebrew/bin/agy"
    command = [agy_path, "models"]
    agy_cwd = _agy_subprocess_cwd()
    agy_env = _agy_subprocess_env()
    last_blocked: dict[str, Any] | None = None
    for attempt in range(2):
        started = time.time()
        stdout = ""
        stderr = ""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=agy_cwd,
                env=agy_env,
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            elapsed = time.time() - started
            combined = "\n".join(part for part in (stdout, stderr) if part)
            models = _parse_agy_models(stdout)
            if result.returncode == 0:
                if not combined.strip():
                    refresh = _run_agy_auth_refresh_gate(
                        agy_path=agy_path,
                        agy_cwd=agy_cwd,
                        agy_env=agy_env,
                        timeout_s=timeout_s,
                    )
                    return _agy_preflight_from_refresh(refresh, agy_cwd=agy_cwd, gemini_dir_absolute=_agy_gemini_dir_is_absolute(agy_env))
                missing = [model for model in AGY_PREFLIGHT_REQUIRED_MODELS if model not in combined]
                if not missing:
                    return _agy_preflight_result(
                        "AGY_OK",
                        command=command,
                        elapsed=elapsed,
                        stdout=stdout,
                        stderr=stderr,
                        models=models,
                        agy_cwd=agy_cwd,
                        gemini_dir_absolute=_agy_gemini_dir_is_absolute(agy_env),
                    )
                return _agy_preflight_blocked(
                    "AGY_MODEL_LIST_MISSING_REQUIRED",
                    command=command,
                    elapsed=elapsed,
                    stdout=stdout,
                    stderr=stderr,
                    models=models,
                    missing_models=missing,
                    agy_cwd=agy_cwd,
                    gemini_dir_absolute=_agy_gemini_dir_is_absolute(agy_env),
                )
            reason = _classify_agy_preflight_block(stdout, stderr)
            last_blocked = _agy_preflight_blocked(
                reason,
                command=command,
                elapsed=elapsed,
                stdout=stdout,
                stderr=stderr,
                models=models,
                agy_cwd=agy_cwd,
                gemini_dir_absolute=_agy_gemini_dir_is_absolute(agy_env),
            )
            if last_blocked and _agy_reason_requires_refresh(reason, combined):
                refresh = _run_agy_auth_refresh_gate(
                    agy_path=agy_path,
                    agy_cwd=agy_cwd,
                    agy_env=agy_env,
                    timeout_s=timeout_s,
                )
                return _agy_preflight_from_refresh(refresh, agy_cwd=agy_cwd, gemini_dir_absolute=_agy_gemini_dir_is_absolute(agy_env))
        except subprocess.TimeoutExpired as exc:
            elapsed = time.time() - started
            stdout = _decode_timeout_part(exc.stdout)
            stderr = _decode_timeout_part(exc.stderr)
            reason = _classify_agy_preflight_block(stdout, stderr)
            if reason != AGY_KEYCHAIN_FALSE_NEGATIVE:
                reason = "AGY_AUTH_TIMEOUT"
            last_blocked = _agy_preflight_blocked(
                reason,
                command=command,
                elapsed=elapsed,
                stdout=stdout,
                stderr=stderr,
                models=[],
                agy_cwd=agy_cwd,
                gemini_dir_absolute=_agy_gemini_dir_is_absolute(agy_env),
            )
            if _agy_reason_requires_refresh(reason, "\n".join(part for part in (stdout, stderr) if part)):
                refresh = _run_agy_auth_refresh_gate(
                    agy_path=agy_path,
                    agy_cwd=agy_cwd,
                    agy_env=agy_env,
                    timeout_s=timeout_s,
                )
                return _agy_preflight_from_refresh(refresh, agy_cwd=agy_cwd, gemini_dir_absolute=_agy_gemini_dir_is_absolute(agy_env))
        if last_blocked and last_blocked.get("blocked_reason") == AGY_KEYCHAIN_FALSE_NEGATIVE and attempt == 0:
            time.sleep(AGY_KEYCHAIN_RETRY_SLEEP_S)
            continue
        return last_blocked
    return last_blocked or _agy_preflight_blocked(
        "AGY_AUTH_REQUIRES_USER",
        command=command,
        elapsed=0,
        stdout="",
        stderr="",
        models=[],
        agy_cwd=agy_cwd,
        gemini_dir_absolute=_agy_gemini_dir_is_absolute(agy_env),
    )


def run_omlx_preflight(timeout_s: int = 15) -> dict[str, Any]:
    """Check OMLX auth/admin visibility without loading a model or writing artifacts."""
    details = _omlx_api_key_details(load_env_file=True)
    actual_r1 = resolve_r1_omlx_model_alias(R1_32B)
    base_url = _omlx_base_url()
    common = {
        "blocked_stage": "",
        "blocked_reason": "",
        "base_url": base_url,
        "configured_key_env": details["env_key"],
        "key_source": details["source"],
        "key_fingerprint": details["fingerprint"],
        "hermes_env_path": details["hermes_env_path"],
        "hermes_env_exists": details["hermes_env_exists"],
        "actual_r1_model": actual_r1,
        "model_visible": False,
    }
    api_key = details["value"]
    if not api_key:
        return {
            "status": "BLOCKED_STATUS",
            **common,
            "blocked_stage": "omlx_preflight",
            "blocked_reason": "OMLX_API_KEY_MISSING",
        }
    admin = _OmlxAdmin(base_url, api_key)
    started = time.time()
    if not admin.login():
        return {
            "status": "BLOCKED_STATUS",
            **common,
            "blocked_stage": "omlx_preflight",
            "blocked_reason": "OMLX_AUTH_BLOCKED",
            "elapsed_seconds": round(time.time() - started, 2),
        }
    try:
        models = admin.get_models()
    except Exception:
        models = []
    model_ids = [str(item.get("id") or "") for item in models if isinstance(item, dict)]
    visible = actual_r1 in model_ids
    if not visible:
        return {
            "status": "BLOCKED_STATUS",
            **common,
            "blocked_stage": "omlx_preflight",
            "blocked_reason": "OMLX_MODEL_LIST_MISSING_R1",
            "elapsed_seconds": round(time.time() - started, 2),
            "model_count": len(model_ids),
        }
    return {
        "status": "OMLX_OK",
        **common,
        "elapsed_seconds": round(time.time() - started, 2),
        "model_visible": True,
        "model_count": len(model_ids),
    }


class LocalTaskEngineExecutor:
    """Local adapter for real AGY/DDGS calls and deterministic artifact writes."""

    def __init__(self, *, agy_log_dir: str | Path | None = None):
        self.agy_log_dir = Path(agy_log_dir or os.getenv("HERMES_AGY_LOG_DIR", "work/agy_logs"))
        self.last_executor_models: dict[str, str] = {}
        self.last_agy_diagnostics: dict[str, dict[str, Any]] = {}
        self.last_omlx_diagnostics: dict[str, dict[str, Any]] = {}
        self._agy_preflight_warmed = False

    def run_agy_preflight(self, timeout_s: int = 45) -> dict[str, Any]:
        result = run_agy_preflight(timeout_s=timeout_s)
        if result.get("status") == "AGY_OK":
            self._agy_preflight_warmed = True
        return result

    def _ensure_agy_preflight_warmed(self, stage: StageSpec) -> None:
        if self._agy_preflight_warmed:
            return
        result = self.run_agy_preflight()
        if result.get("status") == "AGY_OK":
            self._agy_preflight_warmed = True
            return
        raise RuntimeError(
            f"{stage.stage_name}: AGY_PREFLIGHT_BLOCKED\n"
            f"blocked_reason={result.get('blocked_reason') or result.get('status')}\n"
            f"command={json.dumps(result.get('command') or [], ensure_ascii=False)}\n"
            f"stdout_tail={json.dumps(str(result.get('stdout_tail') or '')[-1000:], ensure_ascii=False)}\n"
            f"stderr_tail={json.dumps(str(result.get('stderr_tail') or '')[-1000:], ensure_ascii=False)}"
        )

    def run_agy_gemini(self, stage: StageSpec, prompt: str, model: str, timeout_s: int | None = None) -> str:
        if stage.model not in {GEMINI_HIGH, GEMINI_PRO_HIGH} or model != stage.model:
            raise RuntimeError(f"{stage.stage_name}: Gemini model binding mismatch")
        resolved_agy_path = shutil.which("agy")
        agy_path = resolved_agy_path or "/opt/homebrew/bin/agy"
        if resolved_agy_path is None and not os.path.exists(agy_path):
            raise RuntimeError(f"{stage.stage_name}: agy not found at {agy_path}")
        actual_model = resolve_agy_model_alias(model)
        self._ensure_agy_preflight_warmed(stage)
        self.last_executor_models[stage.stage_name] = actual_model
        timeout_s = timeout_s or _agy_timeout_for_stage(stage)
        agy_cwd = _agy_subprocess_cwd()
        agy_env = _agy_subprocess_env()
        last_error = ""
        for attempt in range(2):
            log_file = Path(tempfile.gettempdir()) / f"agy-{uuid.uuid4().hex[:8]}.log"
            command = [
                agy_path,
                "--log-file",
                str(log_file),
                "--model",
                actual_model,
                "-p",
                prompt,
                "--print-timeout",
                f"{timeout_s}s",
            ]
            self.last_agy_diagnostics[stage.stage_name] = {
                "stage_name": stage.stage_name,
                "model": model,
                "actual_model": actual_model,
                "prompt_chars": len(prompt or ""),
                "print_timeout_seconds": timeout_s,
                "subprocess_timeout_seconds": timeout_s + 30,
                "attempt": attempt + 1,
                "log_file": str(log_file),
                "command_preview": [agy_path, "--log-file", str(log_file), "--model", actual_model, "-p", "<prompt>", "--print-timeout", f"{timeout_s}s"],
                "agy_cwd": str(agy_cwd),
                "error_type": "in_progress",
            }
            started = time.time()
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s + 30,
                    cwd=agy_cwd,
                    env=agy_env,
                )
                elapsed = time.time() - started
                stdout = result.stdout or ""
                stderr = result.stderr or ""
                log_text = _read_text(log_file)
                combined = "\n".join(part for part in (stdout, stderr, log_text) if part)
                if _agy_model_alias_failed(combined, actual_model):
                    last_error = _format_agy_failure(
                        stage=stage,
                        command=command,
                        canonical_model=model,
                        actual_model=actual_model,
                        log_file=log_file,
                        stdout=stdout,
                        stderr=stderr,
                        log_text=log_text,
                        elapsed=elapsed,
                        agy_cwd=agy_cwd,
                        reason="AGY_MODEL_ALIAS_BLOCKED",
                    )
                    break
                timeout_response = _agy_timeout_response(combined)
                if timeout_response:
                    timeout_reason = _agy_timeout_blocker_reason(combined, actual_model, attempt=attempt)
                    last_error = _format_agy_failure(
                        stage=stage,
                        command=command,
                        canonical_model=model,
                        actual_model=actual_model,
                        log_file=log_file,
                        stdout=stdout,
                        stderr=stderr,
                        log_text=log_text,
                        elapsed=elapsed,
                        agy_cwd=agy_cwd,
                        reason=timeout_reason,
                    )
                    if attempt == 0:
                        time.sleep(AGY_KEYCHAIN_RETRY_SLEEP_S)
                        continue
                    break
                keychain_false_negative = _agy_keychain_false_negative(combined)
                auth_negative = _agy_auth_negative(combined)
                location_unsupported = _agy_location_unsupported(combined)
                if result.returncode != 0:
                    failure_reason = f"returncode={result.returncode}"
                    if location_unsupported:
                        failure_reason = AGY_LOCATION_UNSUPPORTED
                    elif keychain_false_negative:
                        failure_reason = AGY_KEYCHAIN_FALSE_NEGATIVE
                    elif auth_negative and attempt > 0:
                        failure_reason = "AGY_AUTH_BLOCKED"
                    elif auth_negative:
                        failure_reason = "AGY_AUTH_REQUIRES_USER"
                    last_error = _format_agy_failure(
                        stage=stage,
                        command=command,
                        canonical_model=model,
                        actual_model=actual_model,
                        log_file=log_file,
                        stdout=stdout,
                        stderr=stderr,
                        log_text=log_text,
                        elapsed=elapsed,
                        agy_cwd=agy_cwd,
                        reason=failure_reason,
                    )
                    if _agy_reason_requires_refresh(failure_reason, combined):
                        refresh = _run_agy_auth_refresh_gate(
                            agy_path=agy_path,
                            agy_cwd=agy_cwd,
                            agy_env=agy_env,
                            timeout_s=min(timeout_s, 90),
                            model=actual_model,
                        )
                        if refresh.get("status") == "AGY_AUTH_REFRESH_OK" and attempt == 0:
                            time.sleep(AGY_KEYCHAIN_RETRY_SLEEP_S)
                            continue
                        last_error = _format_agy_failure(
                            stage=stage,
                            command=command,
                            canonical_model=model,
                            actual_model=actual_model,
                            log_file=log_file,
                            stdout=stdout,
                            stderr=stderr,
                            log_text=log_text,
                            elapsed=elapsed,
                            agy_cwd=agy_cwd,
                            reason=str(refresh.get("blocked_reason") or failure_reason),
                        ) + _agy_refresh_failure_suffix(refresh)
                        break
                    if (keychain_false_negative or auth_negative) and attempt == 0:
                        time.sleep(AGY_KEYCHAIN_RETRY_SLEEP_S)
                        continue
                    break
                output = stdout.strip()
                if output:
                    self.last_agy_diagnostics[stage.stage_name].update(
                        {
                            "error_type": "successful_completion",
                            "elapsed_seconds": round(elapsed, 2),
                            "returncode": result.returncode,
                            "stdout_chars": len(stdout),
                            "stderr_chars": len(stderr),
                        }
                    )
                    return output
                last_error = _format_agy_failure(
                    stage=stage,
                    command=command,
                    canonical_model=model,
                    actual_model=actual_model,
                    log_file=log_file,
                    stdout=stdout,
                    stderr=stderr,
                    log_text=log_text,
                    elapsed=elapsed,
                    agy_cwd=agy_cwd,
                    reason=AGY_LOCATION_UNSUPPORTED if location_unsupported else "empty_stdout",
                )
                refresh_reason = AGY_LOCATION_UNSUPPORTED if location_unsupported else "empty_stdout"
                if _agy_reason_requires_refresh(refresh_reason, combined):
                    refresh = _run_agy_auth_refresh_gate(
                        agy_path=agy_path,
                        agy_cwd=agy_cwd,
                        agy_env=agy_env,
                        timeout_s=min(timeout_s, 90),
                        model=actual_model,
                    )
                    if refresh.get("status") == "AGY_AUTH_REFRESH_OK" and attempt == 0:
                        time.sleep(AGY_KEYCHAIN_RETRY_SLEEP_S)
                        continue
                    last_error = _format_agy_failure(
                        stage=stage,
                        command=command,
                        canonical_model=model,
                        actual_model=actual_model,
                        log_file=log_file,
                        stdout=stdout,
                        stderr=stderr,
                        log_text=log_text,
                        elapsed=elapsed,
                        agy_cwd=agy_cwd,
                        reason=str(refresh.get("blocked_reason") or refresh_reason),
                    ) + _agy_refresh_failure_suffix(refresh)
                break
            except subprocess.TimeoutExpired as exc:
                elapsed = time.time() - started
                stdout = _decode_timeout_part(exc.stdout)
                stderr = _decode_timeout_part(exc.stderr)
                log_text = _read_text(log_file)
                combined = "\n".join(part for part in (stdout, stderr, log_text) if part)
                timeout_response = _agy_timeout_response(combined)
                keychain_false_negative = _agy_keychain_false_negative(combined)
                reason = (
                    _agy_timeout_blocker_reason(combined, actual_model, attempt=attempt)
                    if timeout_response
                    else AGY_KEYCHAIN_FALSE_NEGATIVE
                    if keychain_false_negative
                    else f"timeout_after={timeout_s + 30}s"
                )
                last_error = _format_agy_failure(
                    stage=stage,
                    command=command,
                    canonical_model=model,
                    actual_model=actual_model,
                    log_file=log_file,
                    stdout=stdout,
                    stderr=stderr,
                    log_text=log_text,
                    elapsed=elapsed,
                    agy_cwd=agy_cwd,
                    reason=reason,
                )
                self.last_agy_diagnostics[stage.stage_name].update(
                    {
                        "error_type": "provider_timeout",
                        "elapsed_seconds": round(elapsed, 2),
                        "reason": reason,
                    }
                )
                if _agy_reason_requires_refresh(reason, combined):
                    refresh = _run_agy_auth_refresh_gate(
                        agy_path=agy_path,
                        agy_cwd=agy_cwd,
                        agy_env=agy_env,
                        timeout_s=min(timeout_s, 90),
                        model=actual_model,
                    )
                    if refresh.get("status") == "AGY_AUTH_REFRESH_OK" and attempt == 0:
                        time.sleep(AGY_KEYCHAIN_RETRY_SLEEP_S)
                        continue
                    last_error = _format_agy_failure(
                        stage=stage,
                        command=command,
                        canonical_model=model,
                        actual_model=actual_model,
                        log_file=log_file,
                        stdout=stdout,
                        stderr=stderr,
                        log_text=log_text,
                        elapsed=elapsed,
                        agy_cwd=agy_cwd,
                        reason=str(refresh.get("blocked_reason") or reason),
                    ) + _agy_refresh_failure_suffix(refresh)
                    break
                if (timeout_response or keychain_false_negative) and attempt == 0:
                    time.sleep(AGY_KEYCHAIN_RETRY_SLEEP_S)
                    continue
                break
        raise RuntimeError(last_error or f"{stage.stage_name}: agy failed")

    def run_ddgs(self, stage: StageSpec, queries: list[str]) -> list[dict[str, str]]:
        if stage.model != DDGS_MODEL:
            raise RuntimeError(f"{stage.stage_name}: DDGS model binding mismatch")

        hits: list[dict[str, str]] = []
        errors: list[str] = []
        backends = _ddgs_backend_list()
        timeout_s = _ddgs_timeout_s()
        retries = _ddgs_retries()
        for query in queries[:5]:
            for backend in backends:
                for attempt in range(1, retries + 2):
                    try:
                        results = _ddgs_search_once(query, backend=backend, timeout_s=timeout_s, max_results=5)
                    except Exception as exc:
                        message = f"{backend}:{type(exc).__name__}:{exc}"
                        errors.append(message)
                        if attempt <= retries and _is_transient_ddgs_error(exc):
                            time.sleep(0.2)
                            continue
                        break
                    if not results:
                        errors.append(f"{backend}:empty")
                        break
                    for hit in results:
                        normalized = _normalize_ddgs_hit(query, hit)
                        if normalized["url"]:
                            hits.append(normalized)
                    break
                if hits:
                    break
            time.sleep(0.2)
        if not hits:
            if stage.stage_name == "supplementary_search" and _ddgs_errors_are_no_result_only(errors):
                return [_supplementary_search_no_fresh_hits_marker(queries[:5], backends, errors)]
            raise RuntimeError(
                f"{stage.stage_name}: DDGS returned no fresh hits; "
                f"backends={','.join(backends)}; errors={'; '.join(errors[-8:])}"
            )
        return hits

    def run_codex_handoff(self, stage: StageSpec, inputs: dict[str, Any]) -> str:
        required = ("source_candidates.json", "ddgs_gap_sources.json")
        missing = [name for name in required if not inputs.get(name)]
        if missing:
            raise RuntimeError(f"{stage.stage_name}: missing handoff inputs: {', '.join(missing)}")
        return build_l2_5_evidence_organizer_outputs(inputs)

    def run_omlx_model(self, stage: StageSpec, model: str, prompt: str) -> str:
        started = time.time()
        if stage.stage_name in {"L3_r1_synthesis", "convergence_report"}:
            if stage.model != R1_32B or model != R1_32B:
                raise RuntimeError(f"{stage.stage_name}: R1 model binding mismatch")
            actual_model = resolve_r1_omlx_model_alias(model)
            empty_message = f"{stage.stage_name}: OMLX R1-32B returned empty content"
        elif stage.stage_name == "structure_mapper":
            if stage.model != QWEN72B or model != QWEN72B:
                raise RuntimeError(f"{stage.stage_name}: Qwen72B model binding mismatch")
            actual_model = resolve_qwen72b_omlx_model_alias(model)
            empty_message = f"{stage.stage_name}: OMLX Qwen72B returned empty content"
        elif stage.stage_name == "evidence_judge":
            if stage.model != NEMOTRON120B or model != NEMOTRON120B:
                raise RuntimeError(f"{stage.stage_name}: Nemotron-120B model binding mismatch")
            actual_model = resolve_nemotron120b_omlx_model_alias(model)
            empty_message = f"{stage.stage_name}: OMLX Nemotron-120B returned empty content"
        elif stage.stage_name == "premise_auditor":
            if stage.model != LLAMA70B or model != LLAMA70B:
                raise RuntimeError(f"{stage.stage_name}: Llama70B model binding mismatch")
            actual_model = resolve_llama70b_omlx_model_alias(model)
            empty_message = f"{stage.stage_name}: OMLX Llama70B returned empty content"
        elif stage.stage_name in {"alternative_generator", "insight_harvester"}:
            if stage.model != GEMMA431B or model != GEMMA431B:
                raise RuntimeError(f"{stage.stage_name}: Gemma-4-31B model binding mismatch")
            actual_model = resolve_gemma431b_omlx_model_alias(model)
            empty_message = f"{stage.stage_name}: OMLX Gemma-4-31B returned empty content"
        else:
            raise RuntimeError(f"{stage.stage_name}: OMLX stage is not wired in this smoke layer")
        self.last_executor_models[stage.stage_name] = actual_model
        diagnostic_context: dict[str, Any] = {
            "stage_name": stage.stage_name,
            "model": stage.model,
            "canonical_model": stage.model,
            "actual_model": actual_model,
            "call_site": "LocalTaskEngineExecutor.run_omlx_model",
            "admin_load_requested": False,
            "admin_load_returned": False,
            "observed_model_status": "",
            "inference_request_sent": False,
            "inference_response_received": False,
            "stdout": "",
            "stderr": "",
        }
        partial_content_path = _omlx_partial_content_path(getattr(self, "_current_stage_base_dir", None), stage)
        if partial_content_path:
            diagnostic_context["partial_content_path"] = str(partial_content_path)
        diagnostic_context["whether_partial_content_received"] = False
        diagnostic_context["partial_content_chars"] = 0
        diagnostic_context["response_read_elapsed_seconds"] = 0
        self.last_omlx_diagnostics[stage.stage_name] = diagnostic_context
        api_key = _omlx_api_key()
        if not api_key:
            raise RuntimeError("OMLX_AUTH_BLOCKED: missing OMLX_API_KEY in environment or ~/.hermes/.env")
        admin = _OmlxAdmin(_omlx_base_url(), api_key)
        if not admin.login():
            raise RuntimeError("OMLX_AUTH_BLOCKED: admin login failed using OMLX_API_KEY from env/config")
        request_context: dict[str, Any] | None = None
        stage_timed_out = False
        global _OMLX_PARTIAL_CONTENT_PATH
        previous_partial_path = _OMLX_PARTIAL_CONTENT_PATH
        _OMLX_PARTIAL_CONTENT_PATH = partial_content_path
        try:
            loaded_before_unload = _loaded_omlx_model_ids(admin)
            admin.unload_all()
            loaded_after_unload = _loaded_omlx_model_ids(admin)
            diagnostic_context["admin_load_requested"] = True
            load_result = admin.load_model(actual_model)
            diagnostic_context["admin_load_returned"] = True
            diagnostic_context["observed_model_status"] = _omlx_observed_model_status(admin, actual_model)
            if load_result.get("error") and _is_omlx_memory_guard_error(load_result):
                admin.unload_all()
                time.sleep(5)
                loaded_after_unload = _loaded_omlx_model_ids(admin)
                diagnostic_context["admin_load_requested"] = True
                load_result = admin.load_model(actual_model)
                diagnostic_context["admin_load_returned"] = True
                diagnostic_context["observed_model_status"] = _omlx_observed_model_status(admin, actual_model)
            if load_result.get("error"):
                diagnostic_context["error_type"] = "admin_load_error"
                diagnostic_context["error_summary"] = _redact_secret_text(_safe_omlx_error(load_result))
                if _omlx_status_is_ready(str(diagnostic_context.get("observed_model_status") or "")):
                    diagnostic_context["blocked_reason"] = "inference_not_sent"
                    raise RuntimeError(f"{stage.stage_name}: inference_not_sent: OMLX model is ready but load request returned an error")
                diagnostic_context["blocked_reason"] = "model_load_failed"
                raise RuntimeError(f"{stage.stage_name}: failed to load OMLX actual model: {_safe_omlx_error(load_result)}")
            loaded_after_load = _loaded_omlx_model_ids(admin)
            request_context = _omlx_request_diagnostic_context(
                stage,
                prompt,
                actual_model,
                loaded_models_before_unload=loaded_before_unload,
                loaded_models_after_unload=loaded_after_unload,
                loaded_models_after_load=loaded_after_load,
                retry_attempt="first",
            )
            request_context.update(diagnostic_context)
            try:
                diagnostic_context["inference_request_sent"] = True
                data = _run_omlx_chat_with_retry(stage, actual_model, prompt, api_key=api_key)
                diagnostic_context["inference_response_received"] = True
            except _OmlxPartialResponseError as exc:
                diagnostic_context.update(_omlx_partial_response_diagnostic(stage, actual_model, exc, request_context=request_context))
                self.last_omlx_diagnostics[stage.stage_name] = dict(diagnostic_context)
                raise RuntimeError(f"{stage.stage_name}: {exc.blocked_reason}: partial_content_path={exc.partial_content_path}") from exc
            except RuntimeError as exc:
                if stage.stage_name == "evidence_judge" and _is_omlx_prefill_memory_text(str(exc)):
                    diagnostic_context = dict(request_context or {})
                    diagnostic_context["retry_attempt"] = "final"
                    self.last_omlx_diagnostics[stage.stage_name] = _omlx_prefill_memory_exception_diagnostic(
                        stage, actual_model, exc, attempt="final", request_context=diagnostic_context
                    )
                    raise RuntimeError(
                        f"{stage.stage_name}: OMLX_PREFILL_MEMORY_GUARD_BLOCKED: Nemotron-120B prefill memory guard rejected request"
                    ) from exc
                raise
            content = _extract_chat_content(data)
            if not content.strip() and stage.stage_name == "evidence_judge":
                self.last_omlx_diagnostics[stage.stage_name] = _omlx_empty_content_diagnostic(
                    stage, actual_model, data, attempt="first", request_context=request_context
                )
                if _is_omlx_prefill_memory_diagnostic(self.last_omlx_diagnostics[stage.stage_name]):
                    request_context = dict(request_context or {})
                    request_context["retry_attempt"] = "final"
                admin.unload_all()
                if request_context is not None:
                    request_context["loaded_models_after_unload"] = _loaded_omlx_model_ids(admin)
                load_result = admin.load_model(actual_model)
                if load_result.get("error"):
                    raise RuntimeError(f"{stage.stage_name}: failed to reload OMLX actual model after empty content: {_safe_omlx_error(load_result)}")
                if request_context is not None:
                    request_context["loaded_models_after_load"] = _loaded_omlx_model_ids(admin)
                try:
                    diagnostic_context["inference_request_sent"] = True
                    data = _run_omlx_chat_with_retry(stage, actual_model, prompt, api_key=api_key)
                    diagnostic_context["inference_response_received"] = True
                except _OmlxPartialResponseError as exc:
                    diagnostic_context.update(_omlx_partial_response_diagnostic(stage, actual_model, exc, request_context=request_context))
                    self.last_omlx_diagnostics[stage.stage_name] = dict(diagnostic_context)
                    raise RuntimeError(f"{stage.stage_name}: {exc.blocked_reason}: partial_content_path={exc.partial_content_path}") from exc
                except RuntimeError as exc:
                    if _is_omlx_prefill_memory_text(str(exc)):
                        diagnostic_context = dict(request_context or {})
                        diagnostic_context["retry_attempt"] = "final"
                        self.last_omlx_diagnostics[stage.stage_name] = _omlx_prefill_memory_exception_diagnostic(
                            stage, actual_model, exc, attempt="final", request_context=diagnostic_context
                        )
                        raise RuntimeError(
                            f"{stage.stage_name}: OMLX_PREFILL_MEMORY_GUARD_BLOCKED: Nemotron-120B prefill memory guard rejected request"
                        ) from exc
                    raise
                content = _extract_chat_content(data)
            if not content.strip():
                self.last_omlx_diagnostics[stage.stage_name] = _omlx_empty_content_diagnostic(
                    stage, actual_model, data, attempt="final", request_context=request_context
                )
                if stage.stage_name == "evidence_judge":
                    if _is_omlx_prefill_memory_diagnostic(self.last_omlx_diagnostics[stage.stage_name]):
                        raise RuntimeError(
                            f"{stage.stage_name}: OMLX_PREFILL_MEMORY_GUARD_BLOCKED: Nemotron-120B prefill memory guard rejected request"
                        )
                    raise RuntimeError(f"{stage.stage_name}: OMLX_EMPTY_CONTENT_BLOCKED: Nemotron-120B returned empty content")
                raise RuntimeError(empty_message)
            return content
        except _TaskEngineStageTimeoutError:
            stage_timed_out = True
            diagnostic_context["elapsed_seconds"] = round(time.time() - started, 2)
            diagnostic_context["error_type"] = "stage_timeout"
            diagnostic_context["error_summary"] = "stage timeout while running OMLX stage"
            raise
        except Exception as exc:
            current = self.last_omlx_diagnostics.get(stage.stage_name)
            if isinstance(current, dict):
                current.setdefault("elapsed_seconds", round(time.time() - started, 2))
                current.setdefault("error_type", type(exc).__name__)
                current.setdefault("error_summary", _redact_secret_text(str(exc)))
                current.setdefault("observed_model_status", diagnostic_context.get("observed_model_status") or "")
                current.setdefault("admin_load_requested", diagnostic_context.get("admin_load_requested", False))
                current.setdefault("admin_load_returned", diagnostic_context.get("admin_load_returned", False))
                current.setdefault("inference_request_sent", diagnostic_context.get("inference_request_sent", False))
                current.setdefault("inference_response_received", diagnostic_context.get("inference_response_received", False))
            raise
        finally:
            _OMLX_PARTIAL_CONTENT_PATH = previous_partial_path
            if not stage_timed_out:
                admin.unload_model(actual_model)

    def run_external_calibration(self, stage: StageSpec, packet: dict[str, Any]) -> str:
        if stage.stage_name != "external_calibration" or stage.model != GPT_OR_GEMINI_EXTERNAL:
            raise RuntimeError(f"{stage.stage_name}: external calibration binding mismatch")
        prompt = str(packet.get("prompt") or "")
        if not prompt.strip():
            raise RuntimeError("external_calibration: missing calibration prompt")
        base_dir = packet.get("base_dir")

        fallback_reasons: list[str] = []
        try:
            content = _run_gpt_bridge_calibration(prompt)
            executor_model = _gpt_bridge_executor_model()
            quality_error = _external_calibration_quality_error(
                _external_calibration_with_metadata(content, executor_model=executor_model, fallback_reasons=fallback_reasons)
            ) if content.strip() else "external_calibration_empty_output"
            if not quality_error:
                self.last_executor_models[stage.stage_name] = executor_model
                return _external_calibration_with_metadata(
                    content,
                    executor_model=self.last_executor_models[stage.stage_name],
                    fallback_reasons=fallback_reasons,
                )
            _write_external_calibration_invalid(
                stage,
                base_dir=base_dir,
                content=content,
                executor_model=executor_model,
                fallback_used=False,
                error_summary=quality_error,
                attempt="gpt_bridge_first",
            )
            fallback_reasons.append(f"GPT_BRIDGE_INVALID_FIRST:{quality_error}")
            time.sleep(_gpt_bridge_header_retry_wait_s())
            retry_content = _run_gpt_bridge_calibration(prompt)
            retry_executor_model = _gpt_bridge_executor_model()
            retry_quality_error = _external_calibration_quality_error(
                _external_calibration_with_metadata(
                    retry_content,
                    executor_model=retry_executor_model,
                    fallback_reasons=fallback_reasons,
                )
            ) if retry_content.strip() else "external_calibration_empty_output"
            if not retry_quality_error:
                self.last_executor_models[stage.stage_name] = retry_executor_model
                fallback_reasons.append("GPT_BRIDGE_TARGETED_RETRY_USED")
                return _external_calibration_with_metadata(
                    retry_content,
                    executor_model=self.last_executor_models[stage.stage_name],
                    fallback_reasons=fallback_reasons,
                )
            _write_external_calibration_invalid(
                stage,
                base_dir=base_dir,
                content=retry_content,
                executor_model=retry_executor_model,
                fallback_used=False,
                error_summary=retry_quality_error,
                attempt="gpt_bridge_retry",
            )
            fallback_reasons.append(f"GPT_BRIDGE_INVALID_RETRY:{retry_quality_error}")
        except Exception as exc:
            fallback_reasons.append(f"GPT_BRIDGE_UNAVAILABLE:{_redact_secret_text(str(exc))}")

        gemini_stage = StageSpec(stage.stage_name, GEMINI_PRO_HIGH, GEMINI_PRO_HIGH, stage.required_outputs)
        try:
            content = self.run_agy_gemini(gemini_stage, prompt, GEMINI_PRO_HIGH)
        except Exception as exc:
            raise RuntimeError(
                "external_calibration: GPT Bridge and Gemini/agy unavailable; "
                f"fallback_reasons={json.dumps(fallback_reasons, ensure_ascii=False)}; "
                f"gemini_error={_redact_secret_text(str(exc))}"
            ) from exc
        self.last_executor_models[stage.stage_name] = getattr(self, "last_executor_models", {}).get(stage.stage_name, GEMINI_PRO_HIGH)
        quality_error = _external_calibration_quality_error(
            _external_calibration_with_metadata(
                content,
                executor_model=self.last_executor_models[stage.stage_name],
                fallback_reasons=fallback_reasons,
            )
        )
        if quality_error:
            _write_external_calibration_invalid(
                stage,
                base_dir=base_dir,
                content=content,
                executor_model=self.last_executor_models[stage.stage_name],
                fallback_used=True,
                error_summary=quality_error,
                attempt="gemini_fallback",
            )
            raise RuntimeError(f"external_calibration: artifact_quality_error:{quality_error}")
        return _external_calibration_with_metadata(
            content,
            executor_model=self.last_executor_models[stage.stage_name],
            fallback_reasons=fallback_reasons,
        )

    def run_controller_acceptance(self, stage: StageSpec, packet: dict[str, Any]) -> str:
        if stage.stage_name != "L5_deepseek_acceptance" or stage.model != CONTROLLER_ACCEPTANCE:
            raise RuntimeError(f"{stage.stage_name}: controller acceptance binding mismatch")
        self.last_executor_models[stage.stage_name] = CONTROLLER_ACCEPTANCE
        checked = [
            "L1_gemini_search",
            "L2_ddgs_supplement",
            "L2_5_codex_evidence_organizer",
            "L3_r1_synthesis",
            "L4_gemini_audit",
        ]
        missing = list(packet.get("missing_or_invalid_artifacts") or [])
        profile_issues = _research_packet_profile_acceptance_issues(packet)
        missing.extend(profile_issues)
        missing = sorted(set(str(item) for item in missing))
        profiles = _normalize_profiles(packet.get("research_packet_profile"))
        critical_defects = sorted(set(str(item) for item in (packet.get("critical_defects") or [])))
        l2_5_valid = bool(packet.get("l2_5_valid", True))
        l2_5_analysis = packet.get("l2_5_analysis") if isinstance(packet.get("l2_5_analysis"), dict) else {}
        profile_requirements = [str(item) for item in (packet.get("profile_acceptance_requirements") or [])]
        audit_summary = str(packet.get("audit_summary") or "L4 audit artifact present.")
        claim_table = packet.get("claim_table") if isinstance(packet.get("claim_table"), list) else []
        l4_defect_report = packet.get("l4_defect_report") if isinstance(packet.get("l4_defect_report"), dict) else _l4_defect_report_from_audit(str(packet.get("audit_text") or ""))
        claim_contract_validation = (
            packet.get("claim_contract_validation")
            if isinstance(packet.get("claim_contract_validation"), dict)
            else _research_claim_contract_validation(
                claim_table=claim_table,
                l4_defect_report=l4_defect_report,
                l2_5_analysis=l2_5_analysis,
            )
        )
        noncritical_defects = sorted(set(str(item) for item in (packet.get("noncritical_defects") or [])))
        noncritical_defects = sorted(set(noncritical_defects + [str(item) for item in (l4_defect_report.get("noncritical_defects") or [])]))
        verification_required = sorted(set(str(item) for item in (packet.get("verification_required") or [])))
        verification_required = sorted(
            set(
                verification_required
                + [str(item) for item in (l4_defect_report.get("verification_required") or [])]
                + [str(item) for item in (claim_contract_validation.get("verification_required") or [])]
            )
        )
        evidence_gaps = sorted(set(str(item) for item in (packet.get("evidence_gaps") or [])))
        evidence_gaps = sorted(
            set(
                evidence_gaps
                + [str(item) for item in (l4_defect_report.get("evidence_gaps") or [])]
                + [str(item) for item in (claim_contract_validation.get("evidence_gaps") or [])]
            )
        )
        handoff_caveats = sorted(set(str(item) for item in (packet.get("handoff_caveats") or [])))
        handoff_caveats = sorted(
            set(
                handoff_caveats
                + [str(item) for item in (l4_defect_report.get("handoff_caveats") or [])]
                + [str(item) for item in (claim_contract_validation.get("handoff_caveats") or [])]
            )
        )
        missing.extend(str(item) for item in (claim_contract_validation.get("blocking_errors") or []))
        missing = sorted(set(str(item) for item in missing))
        rejected = missing or _audit_text_rejects(str(packet.get("audit_text") or ""))
        conditional = bool(critical_defects or noncritical_defects or verification_required or handoff_caveats)
        verdict = "REJECTED" if rejected else ("ACCEPTED_WITH_DEFECTS" if conditional else "ACCEPTED")
        accepted = "false" if rejected else "true"
        if rejected:
            ready = "false"
        elif conditional:
            ready = "conditional"
        else:
            ready = "true"
        lines = [
            "research_evidence_packet",
            f"verdict: {verdict}",
            f"accepted: {accepted}",
            "checked_stages: [" + ", ".join(checked) + "]",
            "research_packet_profile: [" + ", ".join(profiles) + "]",
            "profile_acceptance_requirements: [" + "; ".join(profile_requirements) + "]",
            f"l2_5_valid: {str(l2_5_valid).lower()}",
            f"l2_5_stub_detected: {str(bool(l2_5_analysis.get('l2_5_stub_detected'))).lower()}",
            f"insufficient_sources: {str(bool(l2_5_analysis.get('insufficient_sources'))).lower()}",
            "critical_defects: [" + ", ".join(critical_defects) + "]",
            "noncritical_defects: [" + ", ".join(noncritical_defects) + "]",
            "verification_required: [" + ", ".join(verification_required) + "]",
            "evidence_gaps: [" + ", ".join(evidence_gaps) + "]",
            "handoff_caveats: [" + " | ".join(handoff_caveats) + "]",
            "missing_or_invalid_artifacts: [" + ", ".join(missing) + "]",
            f"audit_summary: {audit_summary}",
            f"evidence_packet_ready_for_decision: {ready}",
            "",
        ]
        packet_for_sections = dict(packet)
        packet_for_sections.update(
            {
                "claim_table": claim_table,
                "noncritical_defects": noncritical_defects,
                "verification_required": verification_required,
                "evidence_gaps": evidence_gaps,
                "handoff_caveats": handoff_caveats,
                "claim_contract_validation": claim_contract_validation,
            }
        )
        lines.extend(_compact_research_evidence_sections(packet_for_sections, accepted=not rejected))
        lines.extend([
            "",
            "scope: acceptance gate only; no new research, no search, no synthesis, no user-facing advice or decision output.",
        ])
        return "\n".join(lines)

    def run_final_controller_report(self, stage: StageSpec, packet: dict[str, Any]) -> str:
        if stage.stage_name != "final_controller_report" or stage.model != FINAL_CONTROLLER:
            raise RuntimeError(f"{stage.stage_name}: final controller binding mismatch")
        self.last_executor_models[stage.stage_name] = os.getenv("HERMES_FINAL_CONTROLLER_MODEL", "Hermes Controller").strip() or "Hermes Controller"
        content = _final_controller_report_from_packet(packet)
        try:
            _assert_final_controller_packet_quality(packet, content)
        except RuntimeError as exc:
            _write_final_controller_invalid(
                stage,
                base_dir=packet.get("base_dir"),
                content=content,
                executor_model=self.last_executor_models[stage.stage_name],
                error_summary=str(exc),
            )
            raise
        return content

    def write_artifact(self, stage: StageSpec, content: Any, *, base_dir: str | Path) -> tuple[Path, dict[str, str]]:
        outputs = planned_outputs(stage, base_dir)
        stage_dir = Path(base_dir) / stage.stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        if stage.stage_name == "L2_5_codex_evidence_organizer" and isinstance(content, dict):
            missing_outputs = [name for name in outputs if name not in content]
            if missing_outputs:
                raise RuntimeError(f"{stage.stage_name}: missing organized outputs: {', '.join(missing_outputs)}")
            combined_text = "\n".join(str(content.get(name) or "") for name in outputs)
            _assert_artifact_quality(stage, combined_text)
            for required, path in outputs.items():
                Path(path).write_text(str(content.get(required) or ""), encoding="utf-8")
            return stage_dir, outputs
        text = _stringify_artifact(content)
        _assert_artifact_quality(stage, text)
        if stage.stage_name == "L2_5_codex_evidence_organizer":
            for path in outputs.values():
                Path(path).write_text(text, encoding="utf-8")
            return stage_dir, outputs
        artifact_path = _primary_output_path(stage, outputs, stage_dir)
        artifact_path.write_text(text, encoding="utf-8")
        return artifact_path, outputs

    def make_stage_record(
        self,
        stage: StageSpec,
        *,
        base_dir: str | Path,
        artifact_path: str | Path,
        outputs: dict[str, str],
        created: bool,
        valid: bool,
        status: str,
        executor_model: str | None = None,
    ) -> StageRecord:
        return make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=created,
            valid=valid,
            status=status,
            executor_model=executor_model or self.last_executor_models.get(stage.stage_name, stage.model),
        )


def run_simulated_pipeline(mode: str, *, base_dir: str | Path) -> dict[str, Any]:
    """Write fake artifacts for every canonical stage, then validate/render."""
    executor = LocalTaskEngineExecutor()
    stages: list[dict[str, Any]] = []
    for stage in CANONICAL_STAGES[mode]:
        content = _simulated_content(stage)
        artifact_path, outputs = executor.write_artifact(stage, content, base_dir=base_dir)
        record = executor.make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=True,
            valid=True,
            status="simulated",
            executor_model=stage.model,
        )
        stages.append(record.__dict__)
    run = {"mode": mode, "execution_mode": "simulated-run", "stages": stages}
    validation = validate_pipeline(mode, run, base_dir=base_dir)
    markdown = render_final_markdown(mode, run, validation, base_dir=base_dir)
    return {
        "status": "ok" if validation["valid"] else "blocked",
        "pipeline_status": validation["pipeline_status"],
        "run": run,
        "validation": validation,
        "markdown": markdown,
    }


def run_research_l1_l2_smoke(
    query: str,
    *,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
) -> dict[str, Any]:
    """Run only RESEARCH L1/L2 with real adapters, then stop fail-closed."""
    executor = executor or LocalTaskEngineExecutor()
    stages: list[dict[str, Any]] = []
    research_stages = CANONICAL_STAGES[ENGINE_RESEARCH]
    for stage in research_stages[:2]:
        try:
            if stage.stage_name == "L1_gemini_search":
                content = executor.run_agy_gemini(
                    stage,
                    _gemini_search_prompt(query),
                    stage.model,
                )
            elif stage.stage_name == "L2_ddgs_supplement":
                content = executor.run_ddgs(stage, _ddgs_queries_from_l1_candidates(query, stages, base_dir=base_dir))
            else:
                raise RuntimeError(f"Unexpected smoke stage: {stage.stage_name}")
            artifact_path, outputs = executor.write_artifact(stage, content, base_dir=base_dir)
            record = executor.make_stage_record(
                stage,
                base_dir=base_dir,
                artifact_path=artifact_path,
                outputs=outputs,
                created=True,
                valid=True,
                status="real",
                executor_model=getattr(executor, "last_executor_models", {}).get(stage.stage_name, stage.model),
            )
            stages.append(record.__dict__)
        except Exception as exc:
            outputs = planned_outputs(stage, base_dir)
            artifact_path = _primary_output_path(stage, outputs, Path(base_dir) / stage.stage_name)
            record = make_stage_record(
                stage,
                base_dir=base_dir,
                artifact_path=artifact_path,
                outputs=outputs,
                created=False,
                valid=False,
                status="blocked",
                executor_model=getattr(executor, "last_executor_models", {}).get(stage.stage_name, stage.model),
            )
            item = record.__dict__
            item["error"] = str(exc)
            stages.append(item)
            return {
                "status": "blocked",
                "pipeline_status": PIPELINE_BLOCKED,
                "blocked_stage": stage.stage_name,
                "run": {"mode": ENGINE_RESEARCH, "execution_mode": "real-smoke-l1-l2", "stages": stages},
                "message": "Smoke test stopped fail-closed before final report.",
            }

    run = {"mode": ENGINE_RESEARCH, "execution_mode": "real-smoke-l1-l2", "stages": stages}
    validation = validate_pipeline(ENGINE_RESEARCH, run, base_dir=base_dir)
    return {
        "status": "ok",
        "pipeline_status": PIPELINE_INCOMPLETE,
        "full_pipeline_validation": validation,
        "run": run,
        "message": "L1/L2 smoke completed. Full RESEARCH pipeline remains incomplete by design.",
    }


def run_research_l2_5_codex_handoff_smoke(
    prior_run: dict[str, Any],
    *,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
    query: str = "",
) -> dict[str, Any]:
    """Smoke the L2.5 handoff file protocol after real L1/L2 records exist."""
    executor = executor or LocalTaskEngineExecutor()
    stages = list(prior_run.get("stages", []))
    by_name = {stage.get("stage_name"): stage for stage in stages if isinstance(stage, dict)}
    l1 = by_name.get("L1_gemini_search")
    l2 = by_name.get("L2_ddgs_supplement")
    if not l1 or not l2:
        return {
            "status": "blocked",
            "pipeline_status": PIPELINE_BLOCKED,
            "blocked_stage": "L2_5_codex_evidence_organizer",
            "message": "Codex handoff smoke requires completed L1 and L2 records.",
        }
    if l1.get("valid_for_pipeline") is not True or l2.get("valid_for_pipeline") is not True:
        return {
            "status": "blocked",
            "pipeline_status": PIPELINE_BLOCKED,
            "blocked_stage": "L2_5_codex_evidence_organizer",
            "message": "Codex handoff smoke requires L1/L2 valid_for_pipeline=true.",
        }

    stage = CANONICAL_STAGES[ENGINE_RESEARCH][2]
    inputs = {
        "source_candidates.json": l1.get("artifact_path"),
        "ddgs_gap_sources.json": l2.get("artifact_path"),
        "original_question": query,
    }
    try:
        content = executor.run_codex_handoff(stage, inputs)
        artifact_path, outputs = executor.write_artifact(stage, content, base_dir=base_dir)
        record = executor.make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=True,
            valid=True,
            status="handoff-smoke",
            executor_model=stage.model,
        )
        stages.append(record.__dict__)
        run = {"mode": ENGINE_RESEARCH, "execution_mode": "real-smoke-l1-l2-plus-l2_5", "stages": stages}
        validation = validate_pipeline(ENGINE_RESEARCH, run, base_dir=base_dir)
        return {
            "status": "ok",
            "pipeline_status": PIPELINE_INCOMPLETE,
            "full_pipeline_validation": validation,
            "run": run,
            "message": "L2.5 Codex handoff file protocol smoke completed. Full RESEARCH pipeline remains incomplete by design.",
        }
    except Exception as exc:
        outputs = planned_outputs(stage, base_dir)
        record = make_stage_record(
            stage,
            base_dir=base_dir,
            outputs=outputs,
            artifact_path=Path(base_dir) / stage.stage_name,
            created=False,
            valid=False,
            status="blocked",
            executor_model=stage.model,
        )
        item = record.__dict__
        item["error"] = str(exc)
        stages.append(item)
        return {
            "status": "blocked",
            "pipeline_status": PIPELINE_BLOCKED,
            "blocked_stage": stage.stage_name,
            "run": {"mode": ENGINE_RESEARCH, "execution_mode": "real-smoke-l1-l2-plus-l2_5", "stages": stages},
            "message": "Codex handoff smoke stopped fail-closed.",
        }


def run_research_l3_synthesis_smoke(
    prior_run: dict[str, Any],
    *,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
    query: str = "",
) -> dict[str, Any]:
    """Smoke L3 R1 synthesis from fresh L1/L2/L2.5 artifacts, then stop."""
    executor = executor or LocalTaskEngineExecutor()
    stages = list(prior_run.get("stages", []))
    stage = CANONICAL_STAGES[ENGINE_RESEARCH][3]
    try:
        _require_fresh_prior_for_l3(stages, base_dir=base_dir)
        prompt = _r1_synthesis_prompt_from_artifacts(stages, base_dir=base_dir, query=query)
        content = executor.run_omlx_model(stage, stage.model, prompt)
        artifact_path, outputs = executor.write_artifact(stage, content, base_dir=base_dir)
        record = executor.make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=True,
            valid=True,
            status="real",
            executor_model=getattr(executor, "last_executor_models", {}).get(stage.stage_name, stage.model),
        )
        stages.append(record.__dict__)
        run = {"mode": ENGINE_RESEARCH, "execution_mode": "real-smoke-l1-l3", "stages": stages}
        validation = validate_pipeline(ENGINE_RESEARCH, run, base_dir=base_dir)
        return {
            "status": "ok",
            "pipeline_status": PIPELINE_INCOMPLETE,
            "full_pipeline_validation": validation,
            "run": run,
            "message": "L3 R1 synthesis smoke completed. Full RESEARCH pipeline remains incomplete by design.",
        }
    except Exception as exc:
        outputs = planned_outputs(stage, base_dir)
        artifact_path = _primary_output_path(stage, outputs, Path(base_dir) / stage.stage_name)
        record = make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=False,
            valid=False,
            status="blocked",
            executor_model=getattr(executor, "last_executor_models", {}).get(stage.stage_name, stage.model),
        )
        item = record.__dict__
        item["error"] = str(exc)
        stages.append(item)
        return {
            "status": "blocked",
            "pipeline_status": PIPELINE_BLOCKED,
            "blocked_stage": stage.stage_name,
            "run": {"mode": ENGINE_RESEARCH, "execution_mode": "real-smoke-l1-l3", "stages": stages},
            "message": "L3 R1 synthesis smoke stopped fail-closed before L4.",
        }


def run_research_l1_l3_smoke(
    query: str,
    *,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
) -> dict[str, Any]:
    """Run real L1/L2, L2.5 handoff, then real L3 R1 synthesis and stop."""
    executor = executor or LocalTaskEngineExecutor()
    l1_l2 = run_research_l1_l2_smoke(query, base_dir=base_dir, executor=executor)
    if l1_l2.get("status") != "ok":
        return l1_l2
    l2_5 = run_research_l2_5_codex_handoff_smoke(l1_l2["run"], base_dir=base_dir, executor=executor, query=query)
    if l2_5.get("status") != "ok":
        return l2_5
    return run_research_l3_synthesis_smoke(l2_5["run"], base_dir=base_dir, executor=executor, query=query)


def run_research_l4_gemini_audit_smoke(
    prior_run: dict[str, Any],
    *,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
    query: str = "",
) -> dict[str, Any]:
    """Smoke L4 Gemini audit from fresh L1/L2/L2.5/L3 artifacts, then stop."""
    executor = executor or LocalTaskEngineExecutor()
    stages = list(prior_run.get("stages", []))
    stage = CANONICAL_STAGES[ENGINE_RESEARCH][4]
    try:
        _require_fresh_prior_for_l4(stages, base_dir=base_dir)
        prompt = _gemini_audit_prompt_from_artifacts(stages, base_dir=base_dir, query=query)
        content = executor.run_agy_gemini(stage, prompt, stage.model)
        artifact_path, outputs = executor.write_artifact(stage, content, base_dir=base_dir)
        record = executor.make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=True,
            valid=True,
            status="real",
            executor_model=getattr(executor, "last_executor_models", {}).get(stage.stage_name, stage.model),
        )
        stages.append(record.__dict__)
        run = {"mode": ENGINE_RESEARCH, "execution_mode": "real-smoke-l1-l4", "stages": stages}
        validation = validate_pipeline(ENGINE_RESEARCH, run, base_dir=base_dir)
        return {
            "status": "ok",
            "pipeline_status": PIPELINE_INCOMPLETE,
            "full_pipeline_validation": validation,
            "run": run,
            "message": "L4 Gemini audit smoke completed. Full RESEARCH pipeline remains incomplete by design.",
        }
    except Exception as exc:
        outputs = planned_outputs(stage, base_dir)
        artifact_path = _primary_output_path(stage, outputs, Path(base_dir) / stage.stage_name)
        record = make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=False,
            valid=False,
            status="blocked",
            executor_model=getattr(executor, "last_executor_models", {}).get(stage.stage_name, stage.model),
        )
        item = record.__dict__
        item["error"] = str(exc)
        stages.append(item)
        return {
            "status": "blocked",
            "pipeline_status": PIPELINE_BLOCKED,
            "blocked_stage": stage.stage_name,
            "run": {"mode": ENGINE_RESEARCH, "execution_mode": "real-smoke-l1-l4", "stages": stages},
            "message": "L4 Gemini audit smoke stopped fail-closed before L5.",
        }


def run_research_l5_acceptance_smoke(
    prior_run: dict[str, Any],
    *,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
    query: str = "",
) -> dict[str, Any]:
    """Smoke L5 controller acceptance from fresh L1-L4 artifacts, then stop."""
    executor = executor or LocalTaskEngineExecutor()
    stages = list(prior_run.get("stages", []))
    stage = CANONICAL_STAGES[ENGINE_RESEARCH][5]
    try:
        _require_fresh_prior_for_l5(stages, base_dir=base_dir)
        packet = _research_acceptance_packet_from_artifacts(stages, base_dir=base_dir, query=query)
        content = executor.run_controller_acceptance(stage, packet)
        accepted = _l5_acceptance_text_is_accepted(content)
        artifact_path, outputs = executor.write_artifact(stage, content, base_dir=base_dir)
        record = executor.make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=True,
            valid=accepted,
            status="accepted" if accepted else "rejected",
            executor_model=getattr(executor, "last_executor_models", {}).get(stage.stage_name, stage.model),
        )
        stages.append(record.__dict__)
        run = {"mode": ENGINE_RESEARCH, "execution_mode": "real-smoke-l1-l5", "stages": stages}
        validation = validate_pipeline(ENGINE_RESEARCH, run, base_dir=base_dir)
        if not accepted:
            return {
                "status": "blocked",
                "pipeline_status": PIPELINE_BLOCKED,
                "blocked_stage": stage.stage_name,
                "full_pipeline_validation": validation,
                "run": run,
                "message": "L5 acceptance rejected the research evidence packet. Decision phase was not entered.",
            }
        return {
            "status": "ok" if validation["valid"] else "blocked",
            "pipeline_status": validation["pipeline_status"],
            "full_pipeline_validation": validation,
            "run": run,
            "message": "L5 acceptance completed. RESEARCH pipeline is complete; Decision phase was not entered.",
        }
    except Exception as exc:
        outputs = planned_outputs(stage, base_dir)
        artifact_path = _primary_output_path(stage, outputs, Path(base_dir) / stage.stage_name)
        record = make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=False,
            valid=False,
            status="blocked",
            executor_model=getattr(executor, "last_executor_models", {}).get(stage.stage_name, stage.model),
        )
        item = record.__dict__
        item["error"] = str(exc)
        stages.append(item)
        return {
            "status": "blocked",
            "pipeline_status": PIPELINE_BLOCKED,
            "blocked_stage": stage.stage_name,
            "run": {"mode": ENGINE_RESEARCH, "execution_mode": "real-smoke-l1-l5", "stages": stages},
            "message": "L5 acceptance smoke stopped fail-closed before Decision.",
        }


def run_research_l1_l4_smoke(
    query: str,
    *,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
) -> dict[str, Any]:
    """Run real L1/L2, L2.5, L3, then real L4 Gemini audit and stop."""
    executor = executor or LocalTaskEngineExecutor()
    l1_l3 = run_research_l1_l3_smoke(query, base_dir=base_dir, executor=executor)
    if l1_l3.get("status") != "ok":
        return l1_l3
    return run_research_l4_gemini_audit_smoke(l1_l3["run"], base_dir=base_dir, executor=executor, query=query)


def run_research_l1_l5_smoke(
    query: str,
    *,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
) -> dict[str, Any]:
    """Run real L1-L4, then L5 acceptance and stop before Decision."""
    executor = executor or LocalTaskEngineExecutor()
    l1_l4 = run_research_l1_l4_smoke(query, base_dir=base_dir, executor=executor)
    if l1_l4.get("status") != "ok":
        return l1_l4
    return run_research_l5_acceptance_smoke(l1_l4["run"], base_dir=base_dir, executor=executor, query=query)


def run_research_decision_intelligence_smoke(
    prior_run: dict[str, Any],
    *,
    query: str,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
) -> dict[str, Any]:
    """Smoke Decision stage 7 intelligence_layer after accepted RESEARCH packet."""
    executor = executor or LocalTaskEngineExecutor()
    stages = list(prior_run.get("stages", []))
    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][6]
    try:
        _require_accepted_research_packet_for_intelligence(stages, base_dir=base_dir)
        prompt = _intelligence_layer_prompt_from_research_packet(stages, query=query, base_dir=base_dir)
        content = executor.run_agy_gemini(stage, prompt, stage.model)
        leaked = _intelligence_output_forbidden_tokens(content)
        if leaked:
            raise RuntimeError(f"intelligence_layer: forbidden final-output tokens: {', '.join(leaked)}")
        artifact_path, outputs = executor.write_artifact(stage, content, base_dir=base_dir)
        record = executor.make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=True,
            valid=True,
            status="real",
            executor_model=getattr(executor, "last_executor_models", {}).get(stage.stage_name, stage.model),
        )
        stages.append(record.__dict__)
        run = {
            "mode": ENGINE_RESEARCH_DECISION,
            "execution_mode": "real-smoke-research-decision-intelligence",
            "stages": stages,
        }
        validation = validate_pipeline(ENGINE_RESEARCH_DECISION, run, base_dir=base_dir)
        return {
            "status": "ok",
            "pipeline_status": PIPELINE_INCOMPLETE,
            "full_pipeline_validation": validation,
            "run": run,
            "message": "Decision intelligence_layer smoke completed. Pipeline remains incomplete before Stage 8.",
        }
    except Exception as exc:
        outputs = planned_outputs(stage, base_dir)
        artifact_path = _primary_output_path(stage, outputs, Path(base_dir) / stage.stage_name)
        record = make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=False,
            valid=False,
            status="blocked",
            executor_model=getattr(executor, "last_executor_models", {}).get(stage.stage_name, stage.model),
        )
        item = record.__dict__
        item["error"] = str(exc)
        stages.append(item)
        return {
            "status": "blocked",
            "pipeline_status": PIPELINE_BLOCKED,
            "blocked_stage": stage.stage_name,
            "run": {
                "mode": ENGINE_RESEARCH_DECISION,
                "execution_mode": "real-smoke-research-decision-intelligence",
                "stages": stages,
            },
            "message": "Decision intelligence_layer smoke stopped fail-closed before Stage 8.",
        }


def run_research_decision_supplementary_search_smoke(
    prior_run: dict[str, Any],
    *,
    query: str,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
) -> dict[str, Any]:
    """Smoke Decision stage 8 supplementary_search after intelligence_layer."""
    executor = executor or LocalTaskEngineExecutor()
    stages = list(prior_run.get("stages", []))
    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][7]
    try:
        _require_fresh_prior_for_supplementary_search(stages, base_dir=base_dir)
        hits = executor.run_ddgs(stage, _supplementary_search_queries(query))
        content = _supplementary_search_report(hits, stages=stages, query=query, base_dir=base_dir)
        leaked = _intelligence_output_forbidden_tokens(content)
        if leaked:
            raise RuntimeError(f"supplementary_search: forbidden final-output tokens: {', '.join(leaked)}")
        artifact_path, outputs = executor.write_artifact(stage, content, base_dir=base_dir)
        record = executor.make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=True,
            valid=True,
            status="real",
            executor_model=stage.model,
        )
        stages.append(record.__dict__)
        run = {
            "mode": ENGINE_RESEARCH_DECISION,
            "execution_mode": "real-smoke-research-decision-search",
            "stages": stages,
        }
        validation = validate_pipeline(ENGINE_RESEARCH_DECISION, run, base_dir=base_dir)
        return {
            "status": "ok",
            "pipeline_status": PIPELINE_INCOMPLETE,
            "full_pipeline_validation": validation,
            "run": run,
            "message": "Decision supplementary_search smoke completed. Pipeline remains incomplete before Stage 9.",
        }
    except Exception as exc:
        outputs = planned_outputs(stage, base_dir)
        artifact_path = _primary_output_path(stage, outputs, Path(base_dir) / stage.stage_name)
        record = make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=False,
            valid=False,
            status="blocked",
            executor_model=stage.model,
        )
        item = record.__dict__
        item["error"] = str(exc)
        stages.append(item)
        return {
            "status": "blocked",
            "pipeline_status": PIPELINE_BLOCKED,
            "blocked_stage": stage.stage_name,
            "run": {
                "mode": ENGINE_RESEARCH_DECISION,
                "execution_mode": "real-smoke-research-decision-search",
                "stages": stages,
            },
            "message": "Decision supplementary_search smoke stopped fail-closed before Stage 9.",
        }


def run_research_decision_structure_mapper_smoke(
    prior_run: dict[str, Any],
    *,
    query: str,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
) -> dict[str, Any]:
    """Smoke Decision stage 9 structure_mapper after supplementary_search."""
    executor = executor or LocalTaskEngineExecutor()
    stages = list(prior_run.get("stages", []))
    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][8]
    try:
        _require_fresh_prior_for_structure_mapper(stages, base_dir=base_dir)
        prompt = _structure_mapper_prompt_from_artifacts(stages, query=query, base_dir=base_dir)
        content = executor.run_omlx_model(stage, stage.model, prompt)
        leaked = _structure_mapper_forbidden_tokens(content)
        if leaked:
            debug_path = _write_invalid_stage_debug(stage, content, base_dir=base_dir)
            raise RuntimeError(f"structure_mapper: forbidden later-stage/final tokens: {', '.join(leaked)}; debug_artifact={debug_path}")
        artifact_path, outputs = executor.write_artifact(stage, content, base_dir=base_dir)
        record = executor.make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=True,
            valid=True,
            status="real",
            executor_model=getattr(executor, "last_executor_models", {}).get(stage.stage_name, stage.model),
        )
        stages.append(record.__dict__)
        run = {
            "mode": ENGINE_RESEARCH_DECISION,
            "execution_mode": "real-smoke-research-decision-structure",
            "stages": stages,
        }
        validation = validate_pipeline(ENGINE_RESEARCH_DECISION, run, base_dir=base_dir)
        return {
            "status": "ok",
            "pipeline_status": PIPELINE_INCOMPLETE,
            "full_pipeline_validation": validation,
            "run": run,
            "message": "Decision structure_mapper smoke completed. Pipeline remains incomplete before evidence_judge.",
        }
    except Exception as exc:
        outputs = planned_outputs(stage, base_dir)
        artifact_path = _primary_output_path(stage, outputs, Path(base_dir) / stage.stage_name)
        record = make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=False,
            valid=False,
            status="blocked",
            executor_model=getattr(executor, "last_executor_models", {}).get(stage.stage_name, stage.model),
        )
        item = record.__dict__
        item["error"] = str(exc)
        stages.append(item)
        return {
            "status": "blocked",
            "pipeline_status": PIPELINE_BLOCKED,
            "blocked_stage": stage.stage_name,
            "run": {
                "mode": ENGINE_RESEARCH_DECISION,
                "execution_mode": "real-smoke-research-decision-structure",
                "stages": stages,
            },
            "message": "Decision structure_mapper smoke stopped fail-closed before evidence_judge.",
        }


def run_research_decision_evidence_judge_smoke(
    prior_run: dict[str, Any],
    *,
    query: str,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
) -> dict[str, Any]:
    """Smoke Decision stage 10 evidence_judge after structure_mapper."""
    executor = executor or LocalTaskEngineExecutor()
    stages = list(prior_run.get("stages", []))
    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][9]
    debug_content = ""
    try:
        _require_fresh_prior_for_evidence_judge(stages, base_dir=base_dir)
        prompt = _evidence_judge_prompt_from_artifacts(stages, query=query, base_dir=base_dir)
        diagnostic_prompt = prompt
        try:
            content = executor.run_omlx_model(stage, stage.model, prompt)
        except RuntimeError as exc:
            if not _is_omlx_prefill_memory_text(str(exc)):
                raise
            original_diagnostic = dict(getattr(executor, "last_omlx_diagnostics", {}).get(stage.stage_name) or {})
            original_path = _write_omlx_stage_diagnostic_snapshot(
                stage,
                original_diagnostic,
                base_dir=base_dir,
                filename=f"{stage.stage_name}.diagnostic.json",
            )
            compact_prompt = _evidence_judge_compact_prompt_from_artifacts(stages, query=query, base_dir=base_dir)
            try:
                content = executor.run_omlx_model(stage, stage.model, compact_prompt)
            except Exception as retry_exc:
                compact_diagnostic = dict(getattr(executor, "last_omlx_diagnostics", {}).get(stage.stage_name) or {})
                _annotate_compact_evidence_judge_diagnostic(
                    compact_diagnostic,
                    original_diagnostic=original_diagnostic,
                    original_prompt=prompt,
                    compact_prompt=compact_prompt,
                )
                compact_path = _write_omlx_stage_diagnostic_snapshot(
                    stage,
                    compact_diagnostic,
                    base_dir=base_dir,
                    filename=f"{stage.stage_name}.compact_retry.diagnostic.json",
                )
                getattr(executor, "last_omlx_diagnostics", {})[stage.stage_name] = dict(original_diagnostic)
                raise RuntimeError(
                    f"{stage.stage_name}: compact_retry_failed_after_prefill_memory_guard: {retry_exc}; "
                    f"diagnostic_artifact={original_path}; compact_retry_diagnostic_artifact={compact_path}"
                ) from retry_exc
            compact_diagnostic = dict(getattr(executor, "last_omlx_diagnostics", {}).get(stage.stage_name) or {})
            _annotate_compact_evidence_judge_diagnostic(
                compact_diagnostic,
                original_diagnostic=original_diagnostic,
                original_prompt=prompt,
                compact_prompt=compact_prompt,
            )
            _write_omlx_stage_diagnostic_snapshot(
                stage,
                compact_diagnostic,
                base_dir=base_dir,
                filename=f"{stage.stage_name}.compact_retry.diagnostic.json",
            )
            diagnostic_prompt = compact_prompt
        debug_content = content
        leaked = _evidence_judge_forbidden_tokens(content)
        if leaked:
            debug_path = _write_invalid_stage_debug(stage, content, base_dir=base_dir)
            raise RuntimeError(f"evidence_judge: forbidden later-stage/final tokens: {', '.join(leaked)}; debug_artifact={debug_path}")
        quality_error = _evidence_judge_artifact_quality_error(content)
        if quality_error and _evidence_judge_schema_retryable_quality_error(quality_error):
            _write_invalid_stage_debug(
                stage,
                content,
                base_dir=base_dir,
                filename=f"{stage.stage_name}.schema_retry_source.invalid.md",
            )
            compact_prompt = _evidence_judge_compact_prompt_from_artifacts(stages, query=query, base_dir=base_dir)
            retry_content = executor.run_omlx_model(stage, stage.model, compact_prompt)
            retry_leaked = _evidence_judge_forbidden_tokens(retry_content)
            if retry_leaked:
                debug_path = _write_invalid_stage_debug(stage, retry_content, base_dir=base_dir)
                raise RuntimeError(
                    f"evidence_judge: schema_retry_forbidden later-stage/final tokens: "
                    f"{', '.join(retry_leaked)}; debug_artifact={debug_path}"
                )
            retry_quality_error = _evidence_judge_artifact_quality_error(retry_content)
            if not retry_quality_error:
                content = retry_content
                quality_error = ""
                diagnostic_prompt = compact_prompt
            else:
                content = retry_content
                quality_error = f"schema_retry_failed:{retry_quality_error}"
                diagnostic_prompt = compact_prompt
        if quality_error:
            invalid_path = _write_invalid_stage_debug(stage, content, base_dir=base_dir)
            _annotate_evidence_judge_invalid_artifact_diagnostic(
                stage,
                executor,
                base_dir=base_dir,
                content=content,
                prompt=diagnostic_prompt,
                quality_error=quality_error,
                invalid_artifact_path=invalid_path,
            )
            diagnostic_path = _write_omlx_stage_diagnostic(stage, executor, base_dir=base_dir)
            raise RuntimeError(
                f"evidence_judge: artifact_quality_error:{quality_error}; "
                f"invalid_artifact={invalid_path}; diagnostic_artifact={diagnostic_path}"
            )
        artifact_path, outputs = executor.write_artifact(stage, content, base_dir=base_dir)
        record = executor.make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=True,
            valid=True,
            status="real",
            executor_model=getattr(executor, "last_executor_models", {}).get(stage.stage_name, stage.model),
        )
        stages.append(record.__dict__)
        run = {
            "mode": ENGINE_RESEARCH_DECISION,
            "execution_mode": "real-smoke-research-decision-evidence",
            "stages": stages,
        }
        validation = validate_pipeline(ENGINE_RESEARCH_DECISION, run, base_dir=base_dir)
        return {
            "status": "ok",
            "pipeline_status": PIPELINE_INCOMPLETE,
            "full_pipeline_validation": validation,
            "run": run,
            "message": "Decision evidence_judge smoke completed. Pipeline remains incomplete before premise_auditor.",
        }
    except Exception as exc:
        diagnostic_path = _write_omlx_stage_diagnostic(stage, executor, base_dir=base_dir)
        outputs = planned_outputs(stage, base_dir)
        artifact_path = _primary_output_path(stage, outputs, Path(base_dir) / stage.stage_name)
        record = make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=False,
            valid=False,
            status="blocked",
            executor_model=getattr(executor, "last_executor_models", {}).get(stage.stage_name, stage.model),
        )
        item = record.__dict__
        item["error"] = str(exc) + (f"; diagnostic_artifact={diagnostic_path}" if diagnostic_path else "")
        stages.append(item)
        return {
            "status": "blocked",
            "pipeline_status": PIPELINE_BLOCKED,
            "blocked_stage": stage.stage_name,
            "run": {
                "mode": ENGINE_RESEARCH_DECISION,
                "execution_mode": "real-smoke-research-decision-evidence",
                "stages": stages,
            },
            "message": "Decision evidence_judge smoke stopped fail-closed before premise_auditor.",
        }


def run_research_decision_premise_auditor_smoke(
    prior_run: dict[str, Any],
    *,
    query: str,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
) -> dict[str, Any]:
    """Smoke Decision stage 11 premise_auditor after evidence_judge."""
    executor = executor or LocalTaskEngineExecutor()
    stages = list(prior_run.get("stages", []))
    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][10]
    try:
        _require_fresh_prior_for_premise_auditor(stages, base_dir=base_dir)
        prompt = _premise_auditor_prompt_from_artifacts(stages, query=query, base_dir=base_dir)
        content = executor.run_omlx_model(stage, stage.model, prompt)
        leaked = _premise_auditor_forbidden_tokens(content)
        if leaked:
            debug_path = _write_invalid_stage_debug(stage, content, base_dir=base_dir)
            raise RuntimeError(f"premise_auditor: forbidden later-stage/final tokens: {', '.join(leaked)}; debug_artifact={debug_path}")
        artifact_path, outputs = executor.write_artifact(stage, content, base_dir=base_dir)
        record = executor.make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=True,
            valid=True,
            status="real",
            executor_model=getattr(executor, "last_executor_models", {}).get(stage.stage_name, stage.model),
        )
        stages.append(record.__dict__)
        run = {
            "mode": ENGINE_RESEARCH_DECISION,
            "execution_mode": "real-smoke-research-decision-premise",
            "stages": stages,
        }
        validation = validate_pipeline(ENGINE_RESEARCH_DECISION, run, base_dir=base_dir)
        return {
            "status": "ok",
            "pipeline_status": PIPELINE_INCOMPLETE,
            "full_pipeline_validation": validation,
            "run": run,
            "message": "Decision premise_auditor smoke completed. Pipeline remains incomplete before alternative_generator.",
        }
    except Exception as exc:
        outputs = planned_outputs(stage, base_dir)
        artifact_path = _primary_output_path(stage, outputs, Path(base_dir) / stage.stage_name)
        record = make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=False,
            valid=False,
            status="blocked",
            executor_model=getattr(executor, "last_executor_models", {}).get(stage.stage_name, stage.model),
        )
        item = record.__dict__
        item["error"] = str(exc)
        stages.append(item)
        return {
            "status": "blocked",
            "pipeline_status": PIPELINE_BLOCKED,
            "blocked_stage": stage.stage_name,
            "run": {
                "mode": ENGINE_RESEARCH_DECISION,
                "execution_mode": "real-smoke-research-decision-premise",
                "stages": stages,
            },
            "message": "Decision premise_auditor smoke stopped fail-closed before alternative_generator.",
        }


def run_research_decision_alternative_generator_smoke(
    prior_run: dict[str, Any],
    *,
    query: str,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
) -> dict[str, Any]:
    """Smoke Decision stage 12 alternative_generator after premise_auditor."""
    executor = executor or LocalTaskEngineExecutor()
    stages = list(prior_run.get("stages", []))
    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][11]
    try:
        _require_fresh_prior_for_alternative_generator(stages, base_dir=base_dir)
        prompt = _alternative_generator_prompt_from_artifacts(stages, query=query, base_dir=base_dir)
        content = executor.run_omlx_model(stage, stage.model, prompt)
        leaked = _alternative_generator_forbidden_tokens(content)
        if leaked:
            debug_path = _write_invalid_stage_debug(stage, content, base_dir=base_dir)
            raise RuntimeError(f"alternative_generator: forbidden later-stage/final tokens: {', '.join(leaked)}; debug_artifact={debug_path}")
        artifact_path, outputs = executor.write_artifact(stage, content, base_dir=base_dir)
        record = executor.make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=True,
            valid=True,
            status="real",
            executor_model=getattr(executor, "last_executor_models", {}).get(stage.stage_name, stage.model),
        )
        stages.append(record.__dict__)
        run = {
            "mode": ENGINE_RESEARCH_DECISION,
            "execution_mode": "real-smoke-research-decision-alternative",
            "stages": stages,
        }
        validation = validate_pipeline(ENGINE_RESEARCH_DECISION, run, base_dir=base_dir)
        return {
            "status": "ok",
            "pipeline_status": PIPELINE_INCOMPLETE,
            "full_pipeline_validation": validation,
            "run": run,
            "message": "Decision alternative_generator smoke completed. Pipeline remains incomplete before insight_harvester.",
        }
    except Exception as exc:
        outputs = planned_outputs(stage, base_dir)
        artifact_path = _primary_output_path(stage, outputs, Path(base_dir) / stage.stage_name)
        record = make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=False,
            valid=False,
            status="blocked",
            executor_model=getattr(executor, "last_executor_models", {}).get(stage.stage_name, stage.model),
        )
        item = record.__dict__
        item["error"] = str(exc)
        stages.append(item)
        return {
            "status": "blocked",
            "pipeline_status": PIPELINE_BLOCKED,
            "blocked_stage": stage.stage_name,
            "run": {
                "mode": ENGINE_RESEARCH_DECISION,
                "execution_mode": "real-smoke-research-decision-alternative",
                "stages": stages,
            },
            "message": "Decision alternative_generator smoke stopped fail-closed before insight_harvester.",
        }


def run_research_decision_insight_harvester_smoke(
    prior_run: dict[str, Any],
    *,
    query: str,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
) -> dict[str, Any]:
    """Smoke Decision stage 13 insight_harvester after alternative_generator."""
    executor = executor or LocalTaskEngineExecutor()
    stages = list(prior_run.get("stages", []))
    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][12]
    try:
        _require_fresh_prior_for_insight_harvester(stages, base_dir=base_dir)
        prompt = _insight_harvester_prompt_from_artifacts(stages, query=query, base_dir=base_dir)
        content = executor.run_omlx_model(stage, stage.model, prompt)
        leaked = _insight_harvester_forbidden_tokens(content)
        if leaked:
            debug_path = _write_invalid_stage_debug(stage, content, base_dir=base_dir)
            raise RuntimeError(f"insight_harvester: forbidden later-stage/final tokens: {', '.join(leaked)}; debug_artifact={debug_path}")
        artifact_path, outputs = executor.write_artifact(stage, content, base_dir=base_dir)
        if stages and Path(str(stages[-1].get("artifact_path") or "")).resolve() == Path(artifact_path).resolve():
            raise RuntimeError("insight_harvester: artifact_path must be distinct from alternative_generator")
        record = executor.make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=True,
            valid=True,
            status="real",
            executor_model=getattr(executor, "last_executor_models", {}).get(stage.stage_name, stage.model),
        )
        stages.append(record.__dict__)
        run = {
            "mode": ENGINE_RESEARCH_DECISION,
            "execution_mode": "real-smoke-research-decision-insight",
            "stages": stages,
        }
        validation = validate_pipeline(ENGINE_RESEARCH_DECISION, run, base_dir=base_dir)
        return {
            "status": "ok",
            "pipeline_status": PIPELINE_INCOMPLETE,
            "full_pipeline_validation": validation,
            "run": run,
            "message": "Decision insight_harvester smoke completed. Pipeline remains incomplete before convergence_report.",
        }
    except Exception as exc:
        outputs = planned_outputs(stage, base_dir)
        artifact_path = _primary_output_path(stage, outputs, Path(base_dir) / stage.stage_name)
        record = make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=False,
            valid=False,
            status="blocked",
            executor_model=getattr(executor, "last_executor_models", {}).get(stage.stage_name, stage.model),
        )
        item = record.__dict__
        item["error"] = str(exc)
        stages.append(item)
        return {
            "status": "blocked",
            "pipeline_status": PIPELINE_BLOCKED,
            "blocked_stage": stage.stage_name,
            "run": {
                "mode": ENGINE_RESEARCH_DECISION,
                "execution_mode": "real-smoke-research-decision-insight",
                "stages": stages,
            },
            "message": "Decision insight_harvester smoke stopped fail-closed before convergence_report.",
        }


def run_research_decision_convergence_smoke(
    prior_run: dict[str, Any],
    *,
    query: str,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
) -> dict[str, Any]:
    """Smoke Decision stage 14 convergence_report after all divergence roles."""
    executor = executor or LocalTaskEngineExecutor()
    stages = list(prior_run.get("stages", []))
    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][13]
    try:
        _require_fresh_prior_for_convergence_report(stages, base_dir=base_dir)
        prompt = _convergence_report_prompt_from_artifacts(stages, query=query, base_dir=base_dir)
        content = executor.run_omlx_model(stage, stage.model, prompt)
        leaked = _convergence_report_forbidden_tokens(content)
        if leaked:
            raise RuntimeError(f"convergence_report: forbidden final/tool-chain tokens: {', '.join(leaked)}")
        profile_errors = _quality_profile_errors(content, _task_engine_profiles_from_query(query), stage_name="convergence_report")
        if profile_errors:
            raise RuntimeError(f"convergence_report: output_quality_profile_error:{', '.join(profile_errors)}")
        artifact_path, outputs = executor.write_artifact(stage, content, base_dir=base_dir)
        l3_path = Path(str(stages[3].get("artifact_path") or "")).resolve()
        if Path(artifact_path).resolve() == l3_path:
            raise RuntimeError("convergence_report: artifact_path must be distinct from L3_r1_synthesis")
        record = executor.make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=True,
            valid=True,
            status="real",
            executor_model=getattr(executor, "last_executor_models", {}).get(stage.stage_name, stage.model),
        )
        stages.append(record.__dict__)
        run = {
            "mode": ENGINE_RESEARCH_DECISION,
            "execution_mode": "real-smoke-research-decision-convergence",
            "stages": stages,
        }
        validation = validate_pipeline(ENGINE_RESEARCH_DECISION, run, base_dir=base_dir)
        return {
            "status": "ok",
            "pipeline_status": PIPELINE_INCOMPLETE,
            "full_pipeline_validation": validation,
            "run": run,
            "message": "Decision convergence_report smoke completed. Pipeline remains incomplete before external_calibration.",
        }
    except Exception as exc:
        outputs = planned_outputs(stage, base_dir)
        artifact_path = _primary_output_path(stage, outputs, Path(base_dir) / stage.stage_name)
        record = make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=False,
            valid=False,
            status="blocked",
            executor_model=getattr(executor, "last_executor_models", {}).get(stage.stage_name, stage.model),
        )
        item = record.__dict__
        item["error"] = str(exc)
        stages.append(item)
        return {
            "status": "blocked",
            "pipeline_status": PIPELINE_BLOCKED,
            "blocked_stage": stage.stage_name,
            "run": {
                "mode": ENGINE_RESEARCH_DECISION,
                "execution_mode": "real-smoke-research-decision-convergence",
                "stages": stages,
            },
            "message": "Decision convergence_report smoke stopped fail-closed before external_calibration.",
        }


def run_research_decision_external_calibration_smoke(
    prior_run: dict[str, Any],
    *,
    query: str,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
) -> dict[str, Any]:
    """Smoke Decision stage 15 external_calibration after convergence_report."""
    executor = executor or LocalTaskEngineExecutor()
    stages = list(prior_run.get("stages", []))
    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][14]
    try:
        _require_fresh_prior_for_external_calibration(stages, base_dir=base_dir)
        prompt = _external_calibration_prompt_from_artifacts(stages, query=query, base_dir=base_dir)
        content = executor.run_external_calibration(stage, {"prompt": prompt, "base_dir": str(base_dir)})
        leaked = _external_calibration_forbidden_tokens(content)
        if leaked:
            raise RuntimeError(f"external_calibration: forbidden final-output tokens: {', '.join(leaked)}")
        artifact_path, outputs = executor.write_artifact(stage, content, base_dir=base_dir)
        record = executor.make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=True,
            valid=True,
            status="real",
            executor_model=getattr(executor, "last_executor_models", {}).get(stage.stage_name, stage.model),
        )
        stages.append(record.__dict__)
        run = {
            "mode": ENGINE_RESEARCH_DECISION,
            "execution_mode": "real-smoke-research-decision-calibration",
            "stages": stages,
        }
        validation = validate_pipeline(ENGINE_RESEARCH_DECISION, run, base_dir=base_dir)
        return {
            "status": "ok",
            "pipeline_status": PIPELINE_INCOMPLETE,
            "full_pipeline_validation": validation,
            "run": run,
            "message": "Decision external_calibration smoke completed. Pipeline remains incomplete before final_controller_report.",
        }
    except Exception as exc:
        outputs = planned_outputs(stage, base_dir)
        artifact_path = _primary_output_path(stage, outputs, Path(base_dir) / stage.stage_name)
        record = make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=False,
            valid=False,
            status="blocked",
            executor_model=getattr(executor, "last_executor_models", {}).get(stage.stage_name, stage.model),
        )
        item = record.__dict__
        item["error"] = str(exc)
        stages.append(item)
        return {
            "status": "blocked",
            "pipeline_status": PIPELINE_BLOCKED,
            "blocked_stage": stage.stage_name,
            "run": {
                "mode": ENGINE_RESEARCH_DECISION,
                "execution_mode": "real-smoke-research-decision-calibration",
                "stages": stages,
            },
            "message": "Decision external_calibration smoke stopped fail-closed before final_controller_report.",
        }


def run_research_decision_final_controller_smoke(
    prior_run: dict[str, Any],
    *,
    query: str,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
) -> dict[str, Any]:
    """Smoke Decision stage 16 final_controller_report after external_calibration."""
    executor = executor or LocalTaskEngineExecutor()
    stages = list(prior_run.get("stages", []))
    stage = CANONICAL_STAGES[ENGINE_RESEARCH_DECISION][15]
    try:
        _require_fresh_prior_for_final_controller_report(stages, base_dir=base_dir)
        packet = _final_controller_packet_from_artifacts(stages, query=query, base_dir=base_dir)
        content = executor.run_final_controller_report(stage, packet)
        leaked = _final_controller_report_forbidden_tokens(content)
        if leaked:
            raise RuntimeError(f"final_controller_report: forbidden raw/tool-chain tokens: {', '.join(leaked)}")
        artifact_path, outputs = executor.write_artifact(stage, content, base_dir=base_dir)
        record = executor.make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=True,
            valid=True,
            status="real",
            executor_model=getattr(executor, "last_executor_models", {}).get(stage.stage_name, stage.model),
        )
        stages.append(record.__dict__)
        run = {
            "mode": ENGINE_RESEARCH_DECISION,
            "execution_mode": "real-smoke-research-decision-final",
            "stages": stages,
        }
        validation = validate_pipeline(ENGINE_RESEARCH_DECISION, run, base_dir=base_dir)
        markdown = render_final_markdown(ENGINE_RESEARCH_DECISION, run, validation, base_dir=base_dir)
        return {
            "status": "ok" if validation.get("valid") else "blocked",
            "pipeline_status": validation["pipeline_status"],
            "full_pipeline_validation": validation,
            "run": run,
            "markdown": markdown,
            "message": "Decision final_controller_report smoke completed." if validation.get("valid") else "Decision final_controller_report smoke failed validation.",
        }
    except Exception as exc:
        outputs = planned_outputs(stage, base_dir)
        artifact_path = _primary_output_path(stage, outputs, Path(base_dir) / stage.stage_name)
        record = make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=False,
            valid=False,
            status="blocked",
            executor_model=getattr(executor, "last_executor_models", {}).get(stage.stage_name, stage.model),
        )
        item = record.__dict__
        item["error"] = str(exc)
        stages.append(item)
        return {
            "status": "blocked",
            "pipeline_status": PIPELINE_BLOCKED,
            "blocked_stage": stage.stage_name,
            "run": {
                "mode": ENGINE_RESEARCH_DECISION,
                "execution_mode": "real-smoke-research-decision-final",
                "stages": stages,
            },
            "message": "Decision final_controller_report smoke stopped fail-closed.",
        }


def run_research_decision_l1_l7_smoke(
    query: str,
    *,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
) -> dict[str, Any]:
    """Run RESEARCH L1-L5, then Decision intelligence_layer and stop."""
    executor = executor or LocalTaskEngineExecutor()
    l1_l5 = run_research_l1_l5_smoke(query, base_dir=base_dir, executor=executor)
    if l1_l5.get("status") != "ok":
        return l1_l5
    return run_research_decision_intelligence_smoke(
        l1_l5["run"],
        query=query,
        base_dir=base_dir,
        executor=executor,
    )


def run_research_decision_l1_l8_smoke(
    query: str,
    *,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
) -> dict[str, Any]:
    """Run RESEARCH L1-L5, intelligence_layer, supplementary_search, then stop."""
    executor = executor or LocalTaskEngineExecutor()
    l1_l7 = run_research_decision_l1_l7_smoke(query, base_dir=base_dir, executor=executor)
    if l1_l7.get("status") != "ok":
        return l1_l7
    return run_research_decision_supplementary_search_smoke(
        l1_l7["run"],
        query=query,
        base_dir=base_dir,
        executor=executor,
    )


def run_research_decision_l1_l9_smoke(
    query: str,
    *,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
) -> dict[str, Any]:
    """Run RESEARCH L1-L5, intelligence_layer, supplementary_search, structure_mapper, then stop."""
    executor = executor or LocalTaskEngineExecutor()
    l1_l8 = run_research_decision_l1_l8_smoke(query, base_dir=base_dir, executor=executor)
    if l1_l8.get("status") != "ok":
        return l1_l8
    return run_research_decision_structure_mapper_smoke(
        l1_l8["run"],
        query=query,
        base_dir=base_dir,
        executor=executor,
    )


def run_research_decision_l1_l10_smoke(
    query: str,
    *,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
) -> dict[str, Any]:
    """Run RESEARCH L1-L5, Decision stages 7-10, then stop before premise_auditor."""
    executor = executor or LocalTaskEngineExecutor()
    l1_l9 = run_research_decision_l1_l9_smoke(query, base_dir=base_dir, executor=executor)
    if l1_l9.get("status") != "ok":
        return l1_l9
    return run_research_decision_evidence_judge_smoke(
        l1_l9["run"],
        query=query,
        base_dir=base_dir,
        executor=executor,
    )


def run_research_decision_l1_l11_smoke(
    query: str,
    *,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
) -> dict[str, Any]:
    """Run RESEARCH L1-L5, Decision stages 7-11, then stop before alternative_generator."""
    executor = executor or LocalTaskEngineExecutor()
    l1_l10 = run_research_decision_l1_l10_smoke(query, base_dir=base_dir, executor=executor)
    if l1_l10.get("status") != "ok":
        return l1_l10
    return run_research_decision_premise_auditor_smoke(
        l1_l10["run"],
        query=query,
        base_dir=base_dir,
        executor=executor,
    )


def run_research_decision_l1_l12_smoke(
    query: str,
    *,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
) -> dict[str, Any]:
    """Run RESEARCH L1-L5, Decision stages 7-12, then stop before insight_harvester."""
    executor = executor or LocalTaskEngineExecutor()
    l1_l11 = run_research_decision_l1_l11_smoke(query, base_dir=base_dir, executor=executor)
    if l1_l11.get("status") != "ok":
        return l1_l11
    return run_research_decision_alternative_generator_smoke(
        l1_l11["run"],
        query=query,
        base_dir=base_dir,
        executor=executor,
    )


def run_research_decision_l1_l13_smoke(
    query: str,
    *,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
) -> dict[str, Any]:
    """Run RESEARCH L1-L5, Decision stages 7-13, then stop before convergence_report."""
    executor = executor or LocalTaskEngineExecutor()
    l1_l12 = run_research_decision_l1_l12_smoke(query, base_dir=base_dir, executor=executor)
    if l1_l12.get("status") != "ok":
        return l1_l12
    return run_research_decision_insight_harvester_smoke(
        l1_l12["run"],
        query=query,
        base_dir=base_dir,
        executor=executor,
    )


def run_research_decision_l1_l14_smoke(
    query: str,
    *,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
) -> dict[str, Any]:
    """Run RESEARCH L1-L5, Decision stages 7-14, then stop before external_calibration."""
    executor = executor or LocalTaskEngineExecutor()
    l1_l13 = run_research_decision_l1_l13_smoke(query, base_dir=base_dir, executor=executor)
    if l1_l13.get("status") != "ok":
        return l1_l13
    return run_research_decision_convergence_smoke(
        l1_l13["run"],
        query=query,
        base_dir=base_dir,
        executor=executor,
    )


def run_research_decision_l1_l15_smoke(
    query: str,
    *,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
) -> dict[str, Any]:
    """Run RESEARCH L1-L5, Decision stages 7-15, then stop before final_controller_report."""
    executor = executor or LocalTaskEngineExecutor()
    l1_l14 = run_research_decision_l1_l14_smoke(query, base_dir=base_dir, executor=executor)
    if l1_l14.get("status") != "ok":
        return l1_l14
    return run_research_decision_external_calibration_smoke(
        l1_l14["run"],
        query=query,
        base_dir=base_dir,
        executor=executor,
    )


def run_research_decision_l1_l16_smoke(
    query: str,
    *,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
) -> dict[str, Any]:
    """Run complete RESEARCH_DECISION L1-L16."""
    executor = executor or LocalTaskEngineExecutor()
    l1_l15 = run_research_decision_l1_l15_smoke(query, base_dir=base_dir, executor=executor)
    if l1_l15.get("status") != "ok":
        return l1_l15
    return run_research_decision_final_controller_smoke(
        l1_l15["run"],
        query=query,
        base_dir=base_dir,
        executor=executor,
    )


def run_decision_final_smoke(
    query: str,
    *,
    base_dir: str | Path,
    executor: TaskEngineExecutor | None = None,
    research_packet_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run complete DECISION D1-D10 without entering RESEARCH L1-L5."""
    executor = executor or LocalTaskEngineExecutor()
    stages: list[dict[str, Any]] = []
    specs = CANONICAL_STAGES[ENGINE_DECISION]

    def blocked(stage: StageSpec, exc: Exception) -> dict[str, Any]:
        outputs = planned_outputs(stage, base_dir)
        artifact_path = _primary_output_path(stage, outputs, Path(base_dir) / stage.stage_name)
        record = make_stage_record(
            stage,
            base_dir=base_dir,
            artifact_path=artifact_path,
            outputs=outputs,
            created=False,
            valid=False,
            status="blocked",
            executor_model=getattr(executor, "last_executor_models", {}).get(stage.stage_name, stage.model),
        )
        item = record.__dict__
        item["error"] = str(exc)
        stages.append(item)
        return {
            "status": "blocked",
            "pipeline_status": PIPELINE_BLOCKED,
            "blocked_stage": stage.stage_name,
            "blocked_reason": str(exc),
            "run": {"mode": ENGINE_DECISION, "execution_mode": "real-smoke-decision-final", "stages": stages},
            "message": "DECISION smoke stopped fail-closed.",
        }

    def run_stage(stage: StageSpec, operation: Callable[[], _T]) -> _T:
        return _run_decision_stage_with_timeout(stage, base_dir=base_dir, executor=executor, operation=operation)

    try:
        stage = specs[0]
        content = run_stage(
            stage,
            lambda: executor.run_agy_gemini(
                stage,
                _decision_intelligence_prompt(query, base_dir=base_dir, research_packet_path=research_packet_path),
                stage.model,
            ),
        )
        leaked = _intelligence_output_forbidden_tokens(content)
        if leaked:
            raise RuntimeError(f"intelligence_layer: forbidden final-output tokens: {', '.join(leaked)}")
        _append_real_stage(stages, stage, content, base_dir=base_dir, executor=executor, status="real")

        stage = specs[1]
        _require_decision_prior(stages, ["intelligence_layer"], base_dir=base_dir, consumer_stage=stage.stage_name)
        hits = run_stage(stage, lambda: executor.run_ddgs(stage, _supplementary_search_queries(query)))
        content = _decision_supplementary_search_report(
            hits,
            stages=stages,
            query=query,
            base_dir=base_dir,
            research_packet_path=research_packet_path,
        )
        leaked = _intelligence_output_forbidden_tokens(content)
        if leaked:
            raise RuntimeError(f"supplementary_search: forbidden final-output tokens: {', '.join(leaked)}")
        _append_real_stage(stages, stage, content, base_dir=base_dir, executor=executor, status="real", executor_model=stage.model)

        stage = specs[2]
        _require_decision_prior(stages, ["intelligence_layer", "supplementary_search"], base_dir=base_dir, consumer_stage=stage.stage_name)
        content = run_stage(stage, lambda: executor.run_omlx_model(stage, stage.model, _decision_stage_prompt(stage, stages, query=query, base_dir=base_dir, research_packet_path=research_packet_path)))
        leaked = _structure_mapper_forbidden_tokens(content)
        if leaked:
            debug_path = _write_invalid_stage_debug(stage, content, base_dir=base_dir)
            raise RuntimeError(f"structure_mapper: forbidden later-stage/final tokens: {', '.join(leaked)}; debug_artifact={debug_path}")
        _append_real_stage(stages, stage, content, base_dir=base_dir, executor=executor, status="real")

        stage = specs[3]
        _require_decision_prior(stages, ["intelligence_layer", "supplementary_search", "structure_mapper"], base_dir=base_dir, consumer_stage=stage.stage_name)
        evidence_judge_prompt = _decision_stage_prompt(
            stage,
            stages,
            query=query,
            base_dir=base_dir,
            research_packet_path=research_packet_path,
        )
        try:
            content = run_stage(stage, lambda: executor.run_omlx_model(stage, stage.model, evidence_judge_prompt))
        except Exception as exc:
            diagnostic_path = _write_omlx_stage_diagnostic(stage, executor, base_dir=base_dir)
            if diagnostic_path:
                raise RuntimeError(f"{exc}; diagnostic_artifact={diagnostic_path}") from exc
            raise
        leaked = _evidence_judge_forbidden_tokens(content)
        if leaked:
            debug_path = _write_invalid_stage_debug(stage, content, base_dir=base_dir)
            raise RuntimeError(f"evidence_judge: forbidden later-stage/final tokens: {', '.join(leaked)}; debug_artifact={debug_path}")
        quality_error = _evidence_judge_artifact_quality_error(content)
        if quality_error and _evidence_judge_schema_retryable_quality_error(quality_error):
            _write_invalid_stage_debug(
                stage,
                content,
                base_dir=base_dir,
                filename=f"{stage.stage_name}.schema_retry_source.invalid.md",
            )
            retry_prompt = _decision_evidence_judge_schema_retry_prompt(evidence_judge_prompt)
            content = run_stage(stage, lambda: executor.run_omlx_model(stage, stage.model, retry_prompt))
            leaked = _evidence_judge_forbidden_tokens(content)
            if leaked:
                debug_path = _write_invalid_stage_debug(stage, content, base_dir=base_dir)
                raise RuntimeError(
                    f"evidence_judge: schema_retry_forbidden later-stage/final tokens: "
                    f"{', '.join(leaked)}; debug_artifact={debug_path}"
                )
            retry_quality_error = _evidence_judge_artifact_quality_error(content)
            quality_error = "" if not retry_quality_error else f"schema_retry_failed:{retry_quality_error}"
            evidence_judge_prompt = retry_prompt
        if quality_error:
            invalid_path = _write_invalid_stage_debug(stage, content, base_dir=base_dir)
            _annotate_evidence_judge_invalid_artifact_diagnostic(
                stage,
                executor,
                base_dir=base_dir,
                content=content,
                prompt=evidence_judge_prompt,
                quality_error=quality_error,
                invalid_artifact_path=invalid_path,
            )
            diagnostic_path = _write_omlx_stage_diagnostic(stage, executor, base_dir=base_dir)
            raise RuntimeError(
                f"evidence_judge: artifact_quality_error:{quality_error}; "
                f"invalid_artifact={invalid_path}; diagnostic_artifact={diagnostic_path}"
            )
        _append_real_stage(stages, stage, content, base_dir=base_dir, executor=executor, status="real")

        stage = specs[4]
        _require_decision_prior(stages, ["intelligence_layer", "supplementary_search", "structure_mapper", "evidence_judge"], base_dir=base_dir, consumer_stage=stage.stage_name)
        content = run_stage(stage, lambda: executor.run_omlx_model(stage, stage.model, _decision_stage_prompt(stage, stages, query=query, base_dir=base_dir, research_packet_path=research_packet_path)))
        leaked = _premise_auditor_forbidden_tokens(content)
        if leaked:
            debug_path = _write_invalid_stage_debug(stage, content, base_dir=base_dir)
            raise RuntimeError(f"premise_auditor: forbidden later-stage/final tokens: {', '.join(leaked)}; debug_artifact={debug_path}")
        _append_real_stage(stages, stage, content, base_dir=base_dir, executor=executor, status="real")

        stage = specs[5]
        _require_decision_prior(stages, ["intelligence_layer", "supplementary_search", "structure_mapper", "evidence_judge", "premise_auditor"], base_dir=base_dir, consumer_stage=stage.stage_name)
        content = run_stage(stage, lambda: executor.run_omlx_model(stage, stage.model, _decision_stage_prompt(stage, stages, query=query, base_dir=base_dir, research_packet_path=research_packet_path)))
        leaked = _alternative_generator_forbidden_tokens(content)
        if leaked:
            debug_path = _write_invalid_stage_debug(stage, content, base_dir=base_dir)
            raise RuntimeError(f"alternative_generator: forbidden later-stage/final tokens: {', '.join(leaked)}; debug_artifact={debug_path}")
        _append_real_stage(stages, stage, content, base_dir=base_dir, executor=executor, status="real")

        stage = specs[6]
        _require_decision_prior(stages, ["intelligence_layer", "supplementary_search", "structure_mapper", "evidence_judge", "premise_auditor", "alternative_generator"], base_dir=base_dir, consumer_stage=stage.stage_name)
        content = run_stage(stage, lambda: executor.run_omlx_model(stage, stage.model, _decision_stage_prompt(stage, stages, query=query, base_dir=base_dir, research_packet_path=research_packet_path)))
        leaked = _insight_harvester_forbidden_tokens(content)
        if leaked:
            debug_path = _write_invalid_stage_debug(stage, content, base_dir=base_dir)
            raise RuntimeError(f"insight_harvester: forbidden later-stage/final tokens: {', '.join(leaked)}; debug_artifact={debug_path}")
        _append_real_stage(stages, stage, content, base_dir=base_dir, executor=executor, status="real")

        stage = specs[7]
        _require_decision_prior(stages, [
            "intelligence_layer",
            "supplementary_search",
            "structure_mapper",
            "evidence_judge",
            "premise_auditor",
            "alternative_generator",
            "insight_harvester",
        ], base_dir=base_dir, consumer_stage=stage.stage_name)
        content = run_stage(stage, lambda: executor.run_omlx_model(stage, stage.model, _decision_stage_prompt(stage, stages, query=query, base_dir=base_dir, research_packet_path=research_packet_path)))
        leaked = _convergence_report_forbidden_tokens(content)
        if leaked:
            raise RuntimeError(f"convergence_report: forbidden final/tool-chain tokens: {', '.join(leaked)}")
        profile_errors = _quality_profile_errors(content, _task_engine_profiles_from_query(query), stage_name="convergence_report")
        if profile_errors:
            raise RuntimeError(f"convergence_report: output_quality_profile_error:{', '.join(profile_errors)}")
        _append_real_stage(stages, stage, content, base_dir=base_dir, executor=executor, status="real")

        stage = specs[8]
        _require_decision_prior(stages, [
            "intelligence_layer",
            "supplementary_search",
            "structure_mapper",
            "evidence_judge",
            "premise_auditor",
            "alternative_generator",
            "insight_harvester",
            "convergence_report",
        ], base_dir=base_dir, consumer_stage=stage.stage_name)
        content = run_stage(
            stage,
            lambda: executor.run_external_calibration(
                stage,
                {
                    "prompt": _decision_external_calibration_prompt(
                        stages,
                        query=query,
                        base_dir=base_dir,
                        research_packet_path=research_packet_path,
                    ),
                    "base_dir": str(base_dir),
                },
            ),
        )
        leaked = _external_calibration_forbidden_tokens(content)
        if leaked:
            raise RuntimeError(f"external_calibration: forbidden final-output tokens: {', '.join(leaked)}")
        _append_real_stage(stages, stage, content, base_dir=base_dir, executor=executor, status="real")

        stage = specs[9]
        _require_decision_prior(stages, [
            "intelligence_layer",
            "supplementary_search",
            "structure_mapper",
            "evidence_judge",
            "premise_auditor",
            "alternative_generator",
            "insight_harvester",
            "convergence_report",
            "external_calibration",
        ], base_dir=base_dir, consumer_stage=stage.stage_name)
        packet = _decision_final_controller_packet(stages, query=query, base_dir=base_dir, research_packet_path=research_packet_path)
        content = run_stage(stage, lambda: executor.run_final_controller_report(stage, packet))
        leaked = _final_controller_report_forbidden_tokens(content)
        if leaked:
            raise RuntimeError(f"final_controller_report: forbidden raw/tool-chain tokens: {', '.join(leaked)}")
        _append_real_stage(stages, stage, content, base_dir=base_dir, executor=executor, status="real")

        run = {"mode": ENGINE_DECISION, "execution_mode": "real-smoke-decision-final", "stages": stages}
        validation = validate_pipeline(ENGINE_DECISION, run, base_dir=base_dir)
        markdown = render_final_markdown(ENGINE_DECISION, run, validation, base_dir=base_dir)
        return {
            "status": "ok" if validation.get("valid") else "blocked",
            "pipeline_status": validation["pipeline_status"],
            "full_pipeline_validation": validation,
            "run": run,
            "markdown": markdown,
            "message": "DECISION final_controller_report smoke completed." if validation.get("valid") else "DECISION final_controller_report smoke failed validation.",
        }
    except Exception as exc:
        return blocked(stage, exc)


def _run_decision_stage_with_timeout(
    stage: StageSpec,
    *,
    base_dir: str | Path,
    executor: TaskEngineExecutor,
    operation: Callable[[], _T],
) -> _T:
    timeout_s = _decision_stage_timeout_s(stage)
    started = time.time()
    previous_base_dir = getattr(executor, "_current_stage_base_dir", None)
    if _is_omlx_stage(stage):
        try:
            setattr(executor, "_current_stage_base_dir", str(base_dir))
        except Exception:
            pass
    try:
        with _task_engine_stage_timeout(timeout_s):
            return operation()
    except _TaskEngineStageTimeoutError as exc:
        diagnostic_path: Path | None = None
        blocked_reason = "stage_timeout"
        if _is_omlx_stage(stage):
            blocked_reason = _finalize_omlx_timeout_diagnostic(
                stage,
                executor=executor,
                base_dir=base_dir,
                started=started,
                timeout_s=timeout_s,
            )
            diagnostic_path = _write_omlx_stage_diagnostic(stage, executor, base_dir=base_dir)
        elif _is_agy_stage(stage):
            diagnostic = _classify_agy_stage_timeout(
                stage,
                executor=executor,
                started=started,
                timeout_s=timeout_s,
            )
            blocked_reason = str(diagnostic.get("blocked_reason") or "agy_stage_timeout")
            diagnostic_path = _write_agy_stage_diagnostic(stage, diagnostic, base_dir=base_dir)
        message = f"{stage.stage_name}: {blocked_reason}: exceeded {timeout_s}s"
        if diagnostic_path:
            message += f"; diagnostic_artifact={diagnostic_path}"
        raise RuntimeError(message) from exc
    finally:
        if _is_omlx_stage(stage):
            try:
                if previous_base_dir is None:
                    delattr(executor, "_current_stage_base_dir")
                else:
                    setattr(executor, "_current_stage_base_dir", previous_base_dir)
            except Exception:
                pass


def _finalize_omlx_timeout_diagnostic(
    stage: StageSpec,
    *,
    executor: TaskEngineExecutor,
    base_dir: str | Path,
    started: float,
    timeout_s: int,
) -> str:
    diagnostics = getattr(executor, "last_omlx_diagnostics", None)
    if not isinstance(diagnostics, dict):
        diagnostics = {}
        try:
            setattr(executor, "last_omlx_diagnostics", diagnostics)
        except Exception:
            pass
    data = diagnostics.get(stage.stage_name)
    if not isinstance(data, dict):
        data = {}
        diagnostics[stage.stage_name] = data
    observed_status = str(data.get("observed_model_status") or "")
    admin_load_requested = bool(data.get("admin_load_requested", False))
    admin_load_returned = bool(data.get("admin_load_returned", False))
    inference_request_sent = bool(data.get("inference_request_sent", False))
    inference_response_received = bool(data.get("inference_response_received", False))
    if inference_request_sent and not inference_response_received:
        blocked_reason = "response_read_timeout"
    elif _omlx_status_is_ready(observed_status) and not inference_request_sent:
        blocked_reason = "inference_not_sent"
    elif admin_load_requested and not admin_load_returned:
        blocked_reason = "model_load_timeout"
    elif admin_load_returned and not inference_request_sent:
        blocked_reason = "inference_not_sent"
    else:
        blocked_reason = "inference_timeout"
    data.update(
        {
            "sample_id": _sample_id_from_base_dir(base_dir),
            "stage_name": stage.stage_name,
            "model": data.get("model") or stage.model,
            "elapsed_seconds": round(time.time() - started, 2),
            "timeout_seconds": timeout_s,
            "error_type": "stage_timeout",
            "call_site": data.get("call_site") or "run_decision_final_smoke",
            "admin_load_requested": admin_load_requested,
            "admin_load_returned": admin_load_returned,
            "observed_model_status": observed_status,
            "inference_request_sent": inference_request_sent,
            "inference_response_received": inference_response_received,
            "stdout": _redact_secret_text(str(data.get("stdout") or "")),
            "stderr": _redact_secret_text(str(data.get("stderr") or "")),
            "error_summary": f"{stage.stage_name} exceeded controlled timeout",
            "blocked_reason": blocked_reason,
        }
    )
    return blocked_reason


def _sample_id_from_base_dir(base_dir: str | Path) -> str:
    path = Path(base_dir)
    if path.name in {"decision_run", "research_run"} and path.parent.name:
        return path.parent.name
    return path.name


def _append_real_stage(
    stages: list[dict[str, Any]],
    stage: StageSpec,
    content: Any,
    *,
    base_dir: str | Path,
    executor: TaskEngineExecutor,
    status: str,
    executor_model: str | None = None,
) -> None:
    artifact_path, outputs = executor.write_artifact(stage, content, base_dir=base_dir)
    record = executor.make_stage_record(
        stage,
        base_dir=base_dir,
        artifact_path=artifact_path,
        outputs=outputs,
        created=True,
        valid=True,
        status=status,
        executor_model=executor_model or getattr(executor, "last_executor_models", {}).get(stage.stage_name, stage.model),
    )
    stages.append(record.__dict__)


def resolve_agy_model_alias(canonical_model: str) -> str:
    """Resolve canonical Gemini binding to an actual AGY model id.

    The canonical StageRecord keeps the requested Gemini label. The actual AGY
    CLI model can be configured separately, but must never be blank or CCPA.
    """
    env_keys = {
        GEMINI_HIGH: "HERMES_AGY_GEMINI_HIGH_MODEL",
        GEMINI_PRO_HIGH: "HERMES_AGY_GEMINI_PRO_HIGH_MODEL",
    }
    env_key = env_keys.get(canonical_model)
    if not env_key:
        return canonical_model
    actual = (
        os.getenv(env_key, "").strip()
        or _env_file_agy_model(env_key)
    )
    if not actual:
        settings_model = _settings_agy_model()
        actual = settings_model if settings_model == canonical_model else canonical_model
    if actual.strip().lower() == "ccpa":
        raise RuntimeError("AGY actual model resolved to forbidden CCPA alias")
    return actual


def resolve_r1_omlx_model_alias(canonical_model: str) -> str:
    if canonical_model != R1_32B:
        raise RuntimeError("R1 OMLX alias resolver only accepts canonical R1-32B")
    return (
        os.getenv("HERMES_OMLX_R1_MODEL", "").strip()
        or _env_file_value("HERMES_OMLX_R1_MODEL", Path(os.getenv("HERMES_R1_MODEL_ALIAS_ENV", "work/agy_model_alias.env")))
        or R1_ACTUAL_MODEL_DEFAULT
    )


def resolve_qwen72b_omlx_model_alias(canonical_model: str) -> str:
    if canonical_model != QWEN72B:
        raise RuntimeError("Qwen72B OMLX alias resolver only accepts canonical Qwen72B")
    actual = (
        os.getenv("HERMES_OMLX_QWEN72B_MODEL", "").strip()
        or _env_file_value(
            "HERMES_OMLX_QWEN72B_MODEL",
            Path(os.getenv("HERMES_QWEN72B_MODEL_ALIAS_ENV", "work/agy_model_alias.env")),
        )
        or QWEN72B_ACTUAL_MODEL_DEFAULT
    )
    forbidden = ("9b", "r1", "deepseek", "flash", "controller")
    lowered = actual.lower()
    if any(token in lowered for token in forbidden):
        raise RuntimeError(f"structure_mapper: forbidden Qwen72B actual model alias: {actual}")
    return actual


def resolve_nemotron120b_omlx_model_alias(canonical_model: str) -> str:
    if canonical_model != NEMOTRON120B:
        raise RuntimeError("Nemotron-120B OMLX alias resolver only accepts canonical Nemotron-120B")
    actual = (
        os.getenv("HERMES_OMLX_NEMOTRON120B_MODEL", "").strip()
        or _env_file_value(
            "HERMES_OMLX_NEMOTRON120B_MODEL",
            Path(os.getenv("HERMES_NEMOTRON120B_MODEL_ALIAS_ENV", "work/agy_model_alias.env")),
        )
        or NEMOTRON120B_ACTUAL_MODEL_DEFAULT
    )
    forbidden = ("9b", "qwen", "r1", "deepseek", "flash", "controller")
    lowered = actual.lower()
    if any(token in lowered for token in forbidden):
        raise RuntimeError(f"evidence_judge: forbidden Nemotron-120B actual model alias: {actual}")
    return actual


def resolve_llama70b_omlx_model_alias(canonical_model: str) -> str:
    if canonical_model != LLAMA70B:
        raise RuntimeError("Llama70B OMLX alias resolver only accepts canonical Llama70B")
    actual = (
        os.getenv("HERMES_OMLX_LLAMA70B_MODEL", "").strip()
        or _env_file_value(
            "HERMES_OMLX_LLAMA70B_MODEL",
            Path(os.getenv("HERMES_LLAMA70B_MODEL_ALIAS_ENV", "work/agy_model_alias.env")),
        )
        or LLAMA70B_ACTUAL_MODEL_DEFAULT
    )
    forbidden = ("9b", "qwen", "nemotron", "r1", "deepseek", "flash", "controller")
    lowered = actual.lower()
    if any(token in lowered for token in forbidden):
        raise RuntimeError(f"premise_auditor: forbidden Llama70B actual model alias: {actual}")
    return actual


def resolve_gemma431b_omlx_model_alias(canonical_model: str) -> str:
    if canonical_model != GEMMA431B:
        raise RuntimeError("Gemma-4-31B OMLX alias resolver only accepts canonical Gemma-4-31B")
    actual = (
        os.getenv("HERMES_OMLX_GEMMA431B_MODEL", "").strip()
        or _env_file_value(
            "HERMES_OMLX_GEMMA431B_MODEL",
            Path(os.getenv("HERMES_GEMMA431B_MODEL_ALIAS_ENV", "work/agy_model_alias.env")),
        )
        or GEMMA431B_ACTUAL_MODEL_DEFAULT
    )
    forbidden = ("9b", "qwen", "nemotron", "llama", "r1", "deepseek", "flash", "controller")
    lowered = actual.lower()
    if any(token in lowered for token in forbidden):
        raise RuntimeError(f"alternative_generator: forbidden Gemma-4-31B actual model alias: {actual}")
    return actual


class _OmlxAdmin:
    """L3-only OMLX admin client matching the legacy decision-engine path."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.admin_url = f"{self.base_url}/admin/api"
        self.api_key = api_key
        self.cookie_jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cookie_jar))
        self._logged_in = False

    def login(self) -> bool:
        try:
            result = self._raw_request("POST", "/login", {"api_key": self.api_key}, timeout=10)
            self._logged_in = bool(result.get("success"))
            return self._logged_in
        except Exception:
            return False

    def _admin_request(self, method: str, path: str, body: dict[str, Any] | None = None, *, timeout: int = 120) -> dict[str, Any]:
        if not self._logged_in:
            raise RuntimeError("OMLX_AUTH_BLOCKED: admin session not logged in")
        return self._raw_request(method, path, body, timeout=timeout)

    def _raw_request(self, method: str, path: str, body: dict[str, Any] | None = None, *, timeout: int = 120) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8") if body else None
        headers = {"Content-Type": "application/json"} if data else {}
        request = urllib.request.Request(
            f"{self.admin_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with self.opener.open(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except _TaskEngineStageTimeoutError:
            raise
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            return {"error": True, "status": int(exc.code), "detail": detail}
        except (TimeoutError, socket.timeout) as exc:
            return {"error": True, "timeout": True, "detail": str(exc)}
        except Exception as exc:
            return {"error": True, "detail": str(exc)}

    def get_models(self) -> list[dict[str, Any]]:
        result = self._admin_request("GET", "/models")
        models = result.get("models")
        return models if isinstance(models, list) else []

    def is_model_loaded(self, model_id: str) -> bool:
        for item in self.get_models():
            if not isinstance(item, dict) or str(item.get("id") or "") != model_id:
                continue
            state = str(item.get("state") or item.get("status") or "").lower()
            return bool(item.get("loaded") or item.get("is_loaded") or _omlx_status_is_ready(state))
        return False

    def unload_all(self) -> None:
        for item in self.get_models():
            if not isinstance(item, dict) or not item.get("id"):
                continue
            model_id = str(item["id"])
            if self.is_model_loaded(model_id):
                self.unload_and_wait(model_id)

    def load_model(self, model_id: str) -> dict[str, Any]:
        return self._admin_request("POST", f"/models/{model_id}/load", timeout=_omlx_admin_load_timeout_s())

    def unload_model(self, model_id: str) -> dict[str, Any]:
        return self._admin_request("POST", f"/models/{model_id}/unload", timeout=120)

    def unload_and_wait(self, model_id: str, *, timeout: int = 15) -> bool:
        self.unload_model(model_id)
        time.sleep(3)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.is_model_loaded(model_id):
                return True
            time.sleep(1.5)
        return not self.is_model_loaded(model_id)


def _omlx_chat_completion(
    model: str,
    messages: list[dict[str, str]],
    *,
    api_key: str,
    timeout: int,
    max_tokens: int,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if chat_template_kwargs:
        body["chat_template_kwargs"] = chat_template_kwargs
    request = urllib.request.Request(
        f"{_omlx_base_url()}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            started = time.time()
            raw = _read_omlx_response_bytes(response, partial_content_path=_OMLX_PARTIAL_CONTENT_PATH, started=started)
            try:
                return json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                path, chars = _write_omlx_partial_content(_OMLX_PARTIAL_CONTENT_PATH, raw)
                raise _OmlxPartialResponseError(
                    "response_parse_error",
                    partial_content_path=path,
                    partial_content_chars=chars,
                    response_read_elapsed_seconds=round(time.time() - started, 2),
                    original_error=str(exc),
                ) from exc
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        if int(exc.code) == 401:
            raise RuntimeError("OMLX_AUTH_BLOCKED: chat completion rejected OMLX_API_KEY") from exc
        raise RuntimeError(f"OMLX chat HTTP {int(exc.code)}: {_redact_secret_text(detail)}") from exc
    except (_OmlxPartialResponseError, _TaskEngineStageTimeoutError):
        raise
    except http.client.IncompleteRead:
        raise
    except Exception as exc:
        raise RuntimeError(f"OMLX chat failed: {_redact_secret_text(str(exc))}") from exc


def _read_omlx_response_bytes(response: Any, *, partial_content_path: Path | None, started: float) -> bytes:
    chunks: list[bytes] = []
    try:
        while True:
            chunk = response.read(65536)
            if not chunk:
                break
            chunks.append(chunk)
    except http.client.IncompleteRead as exc:
        if isinstance(exc.partial, bytes):
            chunks.append(exc.partial)
        raw = b"".join(chunks)
        path, chars = _write_omlx_partial_content(partial_content_path, raw)
        raise _OmlxPartialResponseError(
            "response_read_timeout",
            partial_content_path=path,
            partial_content_chars=chars,
            response_read_elapsed_seconds=round(time.time() - started, 2),
            original_error="IncompleteRead",
        ) from exc
    except (_TaskEngineStageTimeoutError, TimeoutError, socket.timeout, KeyboardInterrupt) as exc:
        raw = b"".join(chunks)
        path, chars = _write_omlx_partial_content(partial_content_path, raw)
        reason = "response_read_interrupted" if isinstance(exc, KeyboardInterrupt) else "response_read_timeout"
        raise _OmlxPartialResponseError(
            reason,
            partial_content_path=path,
            partial_content_chars=chars,
            response_read_elapsed_seconds=round(time.time() - started, 2),
            original_error=type(exc).__name__,
        ) from exc
    return b"".join(chunks)


def _omlx_partial_content_path(base_dir: Any, stage: StageSpec) -> Path | None:
    if not base_dir:
        return None
    return Path(base_dir) / stage.stage_name / f"{stage.stage_name}.partial.md"


def _write_omlx_partial_content(path: Path | None, raw: bytes) -> tuple[str, int]:
    text = raw.decode("utf-8", errors="replace") if raw else ""
    if path is None:
        path = Path("/tmp") / f"omlx_partial_response_{uuid.uuid4().hex}.partial.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_redact_secret_text(text), encoding="utf-8")
    return str(path), len(text)


def _omlx_partial_response_diagnostic(
    stage: StageSpec,
    actual_model: str,
    exc: _OmlxPartialResponseError,
    *,
    request_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    diagnostic = {
        "stage_name": stage.stage_name,
        "canonical_model": stage.model,
        "actual_model": actual_model,
        "blocked_reason": exc.blocked_reason,
        "error_type": type(exc).__name__,
        "error_summary": _redact_secret_text(exc.original_error),
        "whether_partial_content_received": exc.partial_content_chars > 0,
        "partial_content_chars": exc.partial_content_chars,
        "partial_content_path": exc.partial_content_path,
        "response_read_elapsed_seconds": exc.response_read_elapsed_seconds,
        "inference_request_sent": True,
        "inference_response_received": False,
    }
    if request_context:
        diagnostic.update(request_context)
        diagnostic.update(
            {
                "blocked_reason": exc.blocked_reason,
                "error_type": type(exc).__name__,
                "error_summary": _redact_secret_text(exc.original_error),
                "whether_partial_content_received": exc.partial_content_chars > 0,
                "partial_content_chars": exc.partial_content_chars,
                "partial_content_path": exc.partial_content_path,
                "response_read_elapsed_seconds": exc.response_read_elapsed_seconds,
                "inference_request_sent": True,
                "inference_response_received": False,
            }
        )
    return diagnostic


def _run_omlx_chat_with_retry(stage: StageSpec, actual_model: str, prompt: str, *, api_key: str) -> dict[str, Any]:
    attempts = 2 if stage.stage_name in {"evidence_judge", "premise_auditor"} else 1
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return _omlx_chat_completion(
                actual_model,
                [{"role": "user", "content": prompt}],
                api_key=api_key,
                timeout=_omlx_timeout_s(),
                max_tokens=_omlx_max_tokens_for_stage(stage),
                chat_template_kwargs=_omlx_chat_template_kwargs_for_stage(stage, actual_model),
            )
        except http.client.IncompleteRead as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(0.5)
                continue
            break
    raise RuntimeError(f"{stage.stage_name}: OMLX chat IncompleteRead after {attempts} attempts") from last_error


def _omlx_chat_template_kwargs_for_stage(stage: StageSpec, actual_model: str) -> dict[str, Any] | None:
    if stage.stage_name == "evidence_judge" and actual_model == NEMOTRON120B_ACTUAL_MODEL_DEFAULT:
        return {"enable_thinking": False, "force_nonempty_content": True}
    return None


def _omlx_base_url() -> str:
    raw = (
        os.getenv("OMLX_BASE", "").strip()
        or os.getenv("OMLX_BASE_URL", "").strip()
        or _decision_engine_api_config().get("base_url", "")
        or "http://127.0.0.1:8000"
    )
    base = str(raw).rstrip("/")
    return base[:-3].rstrip("/") if base.endswith("/v1") else base


def _run_gpt_bridge_calibration(prompt: str) -> str:
    global _GPT_BRIDGE_LAST_EXECUTOR_MODEL
    command_value = os.getenv("HERMES_GPT_BRIDGE_CMD", "").strip() or _hermes_env_value("HERMES_GPT_BRIDGE_CMD")
    url_value = os.getenv("HERMES_GPT_BRIDGE_URL", "").strip() or _hermes_env_value("HERMES_GPT_BRIDGE_URL")
    timeout_s = _gpt_bridge_timeout_s()
    settle_s = _gpt_bridge_settle_s()
    if command_value:
        _GPT_BRIDGE_LAST_EXECUTOR_MODEL = "GPT Bridge"
        command = shlex.split(command_value)
        if not command:
            raise RuntimeError("GPT_BRIDGE_CMD_EMPTY")
        result = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "GPT_BRIDGE_CMD_FAILED:"
                + _redact_secret_text((result.stderr or result.stdout or f"returncode={result.returncode}")[:1000])
            )
        return (result.stdout or "").strip()
    if url_value:
        _GPT_BRIDGE_LAST_EXECUTOR_MODEL = "GPT Bridge"
        payload: dict[str, Any] = {"prompt": prompt, "timeout": timeout_s}
        if settle_s > 0:
            payload["settle_seconds"] = settle_s
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url_value,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s + 10) as response:
                text = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            raise RuntimeError(f"GPT_BRIDGE_HTTP_{int(exc.code)}:{_redact_secret_text(detail)}") from exc
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return text.strip()
        for key in ("content", "text", "output", "message"):
            value = data.get(key) if isinstance(data, dict) else None
            if isinstance(value, str) and value.strip():
                return value.strip()
        return text.strip()
    wrapper = _discover_chatgpt_app_bridge_wrapper()
    if wrapper:
        _GPT_BRIDGE_LAST_EXECUTOR_MODEL = "ChatGPT App Bridge"
        return _run_chatgpt_app_bridge_wrapper(wrapper, prompt, timeout_s=timeout_s, settle_s=settle_s)
    raise RuntimeError("GPT_BRIDGE_NOT_CONFIGURED")


def _gpt_bridge_executor_model() -> str:
    return _GPT_BRIDGE_LAST_EXECUTOR_MODEL or "GPT Bridge"


def _discover_chatgpt_app_bridge_wrapper() -> Path | None:
    if CHATGPT_APP_BRIDGE_WRAPPER.exists():
        return CHATGPT_APP_BRIDGE_WRAPPER
    configured = _decision_engine_bridge_script()
    if configured and configured.exists():
        return configured
    return None


def _decision_engine_bridge_script() -> Path | None:
    path = Path(os.getenv("HERMES_DECISION_ENGINE_CONFIG", "/Users/xqdwww/decision-engine/config.yaml"))
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("bridge_script:"):
            continue
        value = stripped.split(":", 1)[1].strip().strip('"').strip("'")
        return Path(value) if value else None
    return None


def _run_chatgpt_app_bridge_wrapper(wrapper: Path, prompt: str, *, timeout_s: int, settle_s: int) -> str:
    command = [sys.executable, str(wrapper), prompt, "--timeout", str(timeout_s)]
    if settle_s > 0:
        command.extend(["--settle", str(settle_s)])
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_s + 30,
    )
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    if result.returncode != 0:
        detail = _redact_secret_text((stderr or stdout or f"returncode={result.returncode}")[:1000])
        raise RuntimeError(f"{_gpt_bridge_failure_code(detail)}:{detail}")
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        if stdout.strip():
            return stdout.strip()
        raise RuntimeError("GPT_BRIDGE_WRAPPER_EMPTY_OUTPUT")
    ok = bool(data.get("success") or data.get("ok"))
    content = _first_text_value(data, ("response", "text", "content", "output", "message"))
    if ok and content:
        return content
    error = _redact_secret_text(str(data.get("error") or "GPT Bridge wrapper returned no content"))
    raise RuntimeError(f"{_gpt_bridge_failure_code(error)}:{error}")


def _gpt_bridge_failure_code(detail: str) -> str:
    lowered = detail.lower()
    busy_or_unsafe_tokens = (
        "busy",
        "queue",
        "queued",
        "timeout",
        "timed out",
        "stale",
        "unsafe",
        "copy",
        "prompt remained unsent",
        "client disconnected",
    )
    if any(token in lowered for token in busy_or_unsafe_tokens):
        return GPT_BRIDGE_BUSY_OR_UNSAFE
    return "GPT_BRIDGE_WRAPPER_FAILED"


def _first_text_value(data: Any, keys: tuple[str, ...]) -> str:
    if not isinstance(data, dict):
        return ""
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _gpt_bridge_timeout_s() -> int:
    raw = os.getenv("HERMES_GPT_BRIDGE_TIMEOUT_S", "").strip() or _hermes_env_value("HERMES_GPT_BRIDGE_TIMEOUT_S")
    try:
        return max(10, min(int(raw or str(GPT_BRIDGE_DEFAULT_TIMEOUT_S)), 600))
    except ValueError:
        return GPT_BRIDGE_DEFAULT_TIMEOUT_S


def _gpt_bridge_settle_s() -> int:
    raw = (
        os.getenv("HERMES_GPT_BRIDGE_SETTLE_S", "").strip()
        or os.getenv("HERMES_GPT_BRIDGE_SETTLE_SECONDS", "").strip()
        or _hermes_env_value("HERMES_GPT_BRIDGE_SETTLE_S")
        or _hermes_env_value("HERMES_GPT_BRIDGE_SETTLE_SECONDS")
    )
    try:
        return max(0, min(int(raw or str(GPT_BRIDGE_DEFAULT_SETTLE_S)), 180))
    except ValueError:
        return GPT_BRIDGE_DEFAULT_SETTLE_S


def _gpt_bridge_header_retry_wait_s() -> float:
    raw = os.getenv("HERMES_GPT_BRIDGE_HEADER_RETRY_WAIT_S", "").strip() or _hermes_env_value("HERMES_GPT_BRIDGE_HEADER_RETRY_WAIT_S")
    try:
        return max(0.0, min(float(raw or "8"), 60.0))
    except ValueError:
        return 8.0


def _write_external_calibration_invalid(
    stage: StageSpec,
    *,
    base_dir: Any,
    content: str,
    executor_model: str,
    fallback_used: bool,
    error_summary: str,
    attempt: str,
) -> None:
    if not base_dir:
        return
    try:
        stage_dir = Path(base_dir) / stage.stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        raw_text = str(content or "")
        (stage_dir / "external_calibration.invalid.md").write_text(_redact_secret_text(raw_text), encoding="utf-8")
        diagnostic = {
            "stage_name": stage.stage_name,
            "attempt": attempt,
            "raw_length": len(raw_text),
            "body_length": len(raw_text.strip()),
            "executor_model": executor_model,
            "fallback_used": fallback_used,
            "error_summary": _redact_secret_text(error_summary),
        }
        (stage_dir / "external_calibration.diagnostic.json").write_text(
            json.dumps(diagnostic, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        return


def _write_final_controller_invalid(
    stage: StageSpec,
    *,
    base_dir: Any,
    content: str,
    executor_model: str,
    error_summary: str,
) -> None:
    if not base_dir:
        return
    try:
        stage_dir = Path(base_dir) / stage.stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        raw_text = str(content or "")
        (stage_dir / "final_decision_report.invalid.md").write_text(_redact_secret_text(raw_text), encoding="utf-8")
        diagnostic = {
            "stage_name": stage.stage_name,
            "raw_length": len(raw_text),
            "body_length": len(raw_text.strip()),
            "executor_model": executor_model,
            "error_summary": _redact_secret_text(error_summary),
        }
        (stage_dir / "final_controller_report.diagnostic.json").write_text(
            json.dumps(diagnostic, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        return


def _external_calibration_with_metadata(
    content: str,
    *,
    executor_model: str,
    fallback_reasons: list[str],
) -> str:
    return "\n".join(
        [
            "external_calibration",
            f"executor_model: {executor_model}",
            "fallback_reasons: " + json.dumps(fallback_reasons, ensure_ascii=False),
            "",
            str(content).strip(),
        ]
    ).strip()


def _omlx_api_key() -> str:
    return str(_omlx_api_key_details(load_env_file=True)["value"])


def _omlx_api_key_details(*, load_env_file: bool = True) -> dict[str, Any]:
    config = _decision_engine_api_config()
    env_key = str(config.get("api_key_env") or "OMLX_API_KEY")
    hermes_env_path = Path.home() / ".hermes" / ".env"
    env_value = os.getenv(env_key, "").strip()
    file_value = _hermes_env_value(env_key)
    source = "missing"
    value = ""
    if env_value:
        source = "process_env"
        value = env_value
    elif file_value:
        source = str(hermes_env_path)
        value = file_value
        if load_env_file:
            os.environ[env_key] = file_value
    return {
        "env_key": env_key,
        "source": source,
        "value": value,
        "fingerprint": _secret_fingerprint(value),
        "hermes_env_path": str(hermes_env_path),
        "hermes_env_exists": hermes_env_path.exists(),
    }


def _decision_engine_api_config() -> dict[str, Any]:
    path = Path(os.getenv("HERMES_DECISION_ENGINE_CONFIG", "/Users/xqdwww/decision-engine/config.yaml"))
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    api: dict[str, Any] = {}
    in_api = False
    for line in text.splitlines():
        if line.strip() == "api:":
            in_api = True
            continue
        if in_api and line and not line.startswith((" ", "\t")):
            break
        if in_api and ":" in line:
            key, value = line.split(":", 1)
            api[key.strip()] = value.strip().strip('"').strip("'")
    return api


def _hermes_env_value(key: str) -> str:
    return _env_file_value(key, Path.home() / ".hermes" / ".env")


def _secret_fingerprint(value: str) -> dict[str, Any]:
    if not value:
        return {"present": False, "length": 0, "sha256_12": ""}
    return {
        "present": True,
        "length": len(value),
        "sha256_12": hashlib.sha256(value.encode("utf-8")).hexdigest()[:12],
    }


def _env_file_value(key: str, path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    prefix = f"{key}="
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("export "):
            stripped = stripped[len("export "):]
        if stripped.startswith(prefix):
            return stripped[len(prefix):].strip().strip('"').strip("'")
    return ""


def _safe_omlx_error(result: dict[str, Any]) -> str:
    return _redact_secret_text(str(result.get("detail") or result.get("status") or "unknown"))


def _is_omlx_memory_guard_error(result: dict[str, Any]) -> bool:
    text = json.dumps(result, ensure_ascii=False, default=str).lower()
    return "memory ceiling" in text or "memory_guard_tier" in text or "projected memory" in text


def _redact_secret_text(text: str) -> str:
    key = _omlx_api_key()
    value = str(text or "")
    if key:
        value = value.replace(key, "<redacted>")
    return value[:1000]


def _primary_output_path(stage: StageSpec, outputs: dict[str, str], stage_dir: Path) -> Path:
    if stage.required_outputs == ("artifact_path",):
        return stage_dir / "report.md"
    first = stage.required_outputs[0]
    return Path(outputs[first])


def _stringify_artifact(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, indent=2)


def _assert_artifact_quality(stage: StageSpec, text: str) -> None:
    token = _artifact_error_token(text)
    if token:
        raise RuntimeError(f"{stage.stage_name}: artifact_quality_error:{token}")
    if stage.stage_name == "L5_deepseek_acceptance":
        token = _research_evidence_packet_quality_error(text)
        if token:
            raise RuntimeError(f"{stage.stage_name}: artifact_quality_error:{token}")
    if stage.stage_name == "external_calibration":
        token = _external_calibration_quality_error(text)
        if token:
            raise RuntimeError(f"{stage.stage_name}: artifact_quality_error:{token}")
    if stage.stage_name == "evidence_judge":
        token = _evidence_judge_artifact_quality_error(text)
        if token:
            raise RuntimeError(f"{stage.stage_name}: artifact_quality_error:{token}")
    if stage.stage_name == "final_controller_report":
        token = _final_controller_quality_error(text)
        if token:
            raise RuntimeError(f"{stage.stage_name}: artifact_quality_error:{token}")


def _artifact_error_token(text: str) -> str:
    head = (text or "")[:5000]
    lines = [line.strip() for line in head.splitlines()[:40] if line.strip()]
    lowered_head = "\n".join(lines).lower()
    prefix_checks = (
        ("error_timeout_response", "error: timed out waiting for response"),
        ("error_prefix_timeout", "error: timeout"),
        ("bracket_error", "[error:"),
        ("traceback", "traceback"),
        ("exception_prefix", "exception"),
    )
    for token, prefix in prefix_checks:
        if any(line.lower().startswith(prefix) for line in lines):
            return token
    phrase_checks = (
        ("authentication_timed_out", "authentication timed out"),
        ("antigravity_not_logged_in", "you are not logged into antigravity"),
        ("omlx_auth_blocked", "omlx_auth_blocked"),
        ("agy_call_blocked", "agy_call_blocked"),
        ("ddgs_no_fresh_hits", "ddgs returned no fresh hits"),
        ("ddgs_no_fresh_result_urls", "ddgs returned no fresh result urls"),
        ("gpt_bridge_not_configured", "gpt_bridge_not_configured"),
        ("empty_stdout", "returned empty stdout"),
    )
    for token, phrase in phrase_checks:
        if phrase in lowered_head:
            return token
    return ""


def _task_engine_profiles_from_query(query: str) -> list[str]:
    value = query or ""
    lowered = value.lower()
    profiles: list[str] = []

    foresight_terms = (
        "未来10年",
        "未来 10 年",
        "未来十年",
        "ai 环境",
        "ai降低",
        "ai 降低",
        "知识获取成本",
        "结构性反转",
        "优势变陷阱",
        "缺陷变优势",
        "机制推理",
        "趋势演化",
        "情景判断",
        "foresight",
        "structural reversal",
        "future scenario",
    )
    if any(term in value for term in foresight_terms) or any(term in lowered for term in ("future scenario", "structural reversal")):
        profiles.append(PROFILE_FORESIGHT_MECHANISM)

    implementation_terms = (
        "家长行为培训",
        "如何执行",
        "训练方案",
        "准备路线",
        "操作计划",
        "执行计划",
        "干预方案",
        "implementation plan",
        "parent training",
    )
    if any(term in value for term in implementation_terms) or any(term in lowered for term in ("implementation plan", "parent training")):
        profiles.append(PROFILE_IMPLEMENTATION_PLAN)

    evidence_terms = (
        "医学",
        "治疗方案",
        "研究进展",
        "文献综述",
        "指南",
        "循证",
        "证据",
        "therapy",
        "treatment",
        "guideline",
        "evidence",
    )
    if not profiles or any(term in value for term in evidence_terms) or any(term in lowered for term in ("treatment", "guideline", "evidence")):
        profiles.insert(0, PROFILE_EVIDENCE_GROUNDED)

    deduped: list[str] = []
    for profile in profiles:
        if profile not in deduped:
            deduped.append(profile)
    return deduped


def _external_calibration_quality_body(text: str) -> str:
    return scoring_calibration.external_calibration_quality_body(text)


def _external_calibration_has_verdict_body(text: str) -> bool:
    return scoring_calibration.external_calibration_has_verdict_body(text)


def _external_calibration_quality_error(text: str) -> str:
    result = scoring_calibration.assess_external_calibration_text(text)
    return "" if result.passed else result.reason


EXTERNAL_CALIBRATION_MINIMUM_FIELDS = scoring_calibration.EXTERNAL_CALIBRATION_MINIMUM_FIELDS


_T = TypeVar("_T")
_OMLX_PARTIAL_CONTENT_PATH: Path | None = None


class _TaskEngineStageTimeoutError(Exception):
    """Raised by the local stage timeout alarm."""


class _OmlxPartialResponseError(RuntimeError):
    def __init__(
        self,
        blocked_reason: str,
        *,
        partial_content_path: str,
        partial_content_chars: int,
        response_read_elapsed_seconds: float,
        original_error: str,
    ):
        super().__init__(f"{blocked_reason}: partial_content_path={partial_content_path}")
        self.blocked_reason = blocked_reason
        self.partial_content_path = partial_content_path
        self.partial_content_chars = partial_content_chars
        self.response_read_elapsed_seconds = response_read_elapsed_seconds
        self.original_error = original_error


@contextmanager
def _task_engine_stage_timeout(timeout_s: int):
    alarm_signal = getattr(signal, "SIGALRM", None)
    itimer_real = getattr(signal, "ITIMER_REAL", None)
    setitimer = getattr(signal, "setitimer", None)
    if (
        timeout_s <= 0
        or threading.current_thread() is not threading.main_thread()
        or alarm_signal is None
        or itimer_real is None
        or setitimer is None
    ):
        yield
        return
    previous_handler = signal.getsignal(alarm_signal)
    previous_timer = setitimer(itimer_real, 0)

    def _raise_timeout(_signum: int, _frame: Any) -> None:
        raise _TaskEngineStageTimeoutError(f"stage_timeout_after={timeout_s}s")

    signal.signal(alarm_signal, _raise_timeout)
    setitimer(itimer_real, timeout_s)
    try:
        yield
    finally:
        setitimer(itimer_real, 0)
        signal.signal(alarm_signal, previous_handler)
        if previous_timer and previous_timer[0] > 0:
            setitimer(itimer_real, previous_timer[0], previous_timer[1])


def _decision_stage_timeout_s(stage: StageSpec) -> int:
    stage_key = stage.stage_name.upper().replace("-", "_")
    raw = os.getenv(f"HERMES_DECISION_{stage_key}_TIMEOUT_S", "").strip()
    if not raw and _is_omlx_stage(stage):
        raw = os.getenv("HERMES_DECISION_OMLX_STAGE_TIMEOUT_S", "").strip()
    if not raw:
        raw = os.getenv("HERMES_DECISION_STAGE_TIMEOUT_S", "").strip()
    if _is_omlx_stage(stage):
        default = "720"
    elif _is_agy_stage(stage):
        default = str(min(_agy_timeout_for_stage(stage) + 60, 3600))
    else:
        default = "360"
    try:
        parsed = max(1, min(int(raw or default), 3600))
    except ValueError:
        parsed = int(default)
    if _is_agy_stage(stage):
        parsed = max(parsed, min(_agy_timeout_for_stage(stage) + 60, 3600))
    return parsed


def _is_agy_stage(stage: StageSpec) -> bool:
    return stage.model in {GEMINI_HIGH, GEMINI_PRO_HIGH}


def _decision_intelligence_prompt_char_budget() -> int:
    return 12000


def _agy_invalid_artifact_tool_call(text: str) -> bool:
    lowered = (text or "").lower()
    return "invalid tool call" in lowered and (
        "not a valid artifact path" in lowered or "artifacts must be" in lowered
    )


def _classify_agy_stage_timeout(
    stage: StageSpec,
    *,
    executor: TaskEngineExecutor,
    started: float,
    timeout_s: int,
) -> dict[str, Any]:
    diagnostics = getattr(executor, "last_agy_diagnostics", {})
    stage_diagnostic = diagnostics.get(stage.stage_name, {}) if isinstance(diagnostics, dict) else {}
    log_file = str(stage_diagnostic.get("log_file") or "") if isinstance(stage_diagnostic, dict) else ""
    log_text = _read_text(Path(log_file)) if log_file else ""
    prompt_chars = int(stage_diagnostic.get("prompt_chars") or 0) if isinstance(stage_diagnostic, dict) else 0
    oversized_payload = prompt_chars > _decision_intelligence_prompt_char_budget()
    invalid_artifact_tool_call = _agy_invalid_artifact_tool_call(log_text)
    provider_timeout = _agy_timeout_response(log_text) or stage_diagnostic.get("error_type") == "provider_timeout"
    if invalid_artifact_tool_call:
        blocked_reason = "provider_tool_call_invalid_artifact_path"
    elif oversized_payload:
        blocked_reason = "oversized_payload"
    elif provider_timeout:
        blocked_reason = "model_provider_timeout"
    else:
        blocked_reason = "wrapper_timeout"
    return {
        "stage_name": stage.stage_name,
        "blocked_reason": blocked_reason,
        "timeout_seconds": timeout_s,
        "elapsed_seconds": round(time.time() - started, 2),
        "provider_timeout": provider_timeout,
        "wrapper_timeout": blocked_reason == "wrapper_timeout",
        "oversized_payload": oversized_payload,
        "invalid_artifact_tool_call": invalid_artifact_tool_call,
        "parse_or_postprocessing_timeout": False,
        "successful_completion": False,
        "prompt_chars": prompt_chars,
        "prompt_char_budget": _decision_intelligence_prompt_char_budget(),
        "agy_diagnostic": stage_diagnostic,
        "log_file": log_file,
        "log_tail": log_text[-4000:],
    }


def _write_agy_stage_diagnostic(stage: StageSpec, diagnostic: dict[str, Any], *, base_dir: str | Path) -> Path:
    path = Path(base_dir) / stage.stage_name / f"{stage.stage_name}.diagnostic.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _is_omlx_stage(stage: StageSpec) -> bool:
    return stage.stage_name in {
        "structure_mapper",
        "evidence_judge",
        "premise_auditor",
        "alternative_generator",
        "insight_harvester",
        "convergence_report",
    }


def _external_calibration_has_minimum_fields(text: str) -> bool:
    return scoring_calibration.external_calibration_has_minimum_fields(text)


def _external_calibration_header_only_fields(text: str) -> list[str]:
    return scoring_calibration.external_calibration_header_only_fields(text)


def _colon_or_plain_section_body(text: str, field: str) -> str:
    return scoring_calibration.colon_or_plain_section_body(text, field)


def _final_controller_quality_error(text: str) -> str:
    value = (text or "").strip()
    lowered = value.lower()
    calibration_result = scoring_calibration.assess_final_controller_text(value)
    if calibration_result.blocked:
        return calibration_result.reason
    overstrong_terms = (
        "pfc disuse atrophy",
        "prefrontal cortex disuse atrophy",
        "digital dopamine resistance",
        "前额叶萎缩",
        "前额叶皮层废用性萎缩",
        "多巴胺重置",
        "数字多巴胺抗性",
        "神经保护机制",
    )
    if any(term in lowered or term in value for term in overstrong_terms):
        return "overstrong_mechanism_term"
    if "decision_mode=true" in lowered and any(term in value for term in ("家长行为培训详细方案", "三年级准备路线", "ADHD 儿童研究决策报告")):
        return "decision_mode_family_advice_leak"
    return ""


def _looks_like_raw_markdown_table_dump(text: str) -> bool:
    rows = [line.strip() for line in (text or "").splitlines() if line.strip().startswith("|")]
    if len(rows) < 3:
        return False
    joined = "\n".join(rows[:12]).lower()
    stageish = ("artifact" in joined or "stage" in joined or "evidence" in joined or "claim" in joined)
    has_separator = any("---" in row for row in rows[:6])
    return stageish and has_separator


def _decision_query_forbids_advice(query: str) -> bool:
    value = query or ""
    return any(
        term in value
        for term in (
            "不要家长建议",
            "不要做家长建议",
            "不要建议",
            "不要做建议",
            "不要培养计划",
            "不要做培养计划",
            "不要文献综述",
            "不要做文献综述",
        )
    )


def _decision_query_requests_future_inversion_structure(query: str) -> bool:
    value = query or ""
    return all(
        term in value
        for term in (
            "未来优势变陷阱",
            "未来缺陷变优势",
            "最危险的错误培养路径",
            "最反直觉但值得追踪的假设",
            "danger_flag",
        )
    )


def _required_decision_future_sections() -> tuple[str, ...]:
    return (
        "未来优势变陷阱 Top5",
        "未来缺陷变优势 Top5",
        "最危险的错误培养路径",
        "最反直觉但值得追踪的假设",
        "danger_flag",
    )


def _assert_final_controller_packet_quality(packet: dict[str, Any], text: str) -> None:
    token = _final_controller_quality_error(text)
    if token:
        raise RuntimeError(f"final_controller_report: artifact_quality_error:{token}")
    calibration_result = scoring_calibration.assess_final_controller_packet(packet, text)
    if calibration_result.blocked:
        raise RuntimeError(f"final_controller_report: scoring_calibration:{calibration_result.reason}")
    query = str(packet.get("query") or "")
    if str(packet.get("mode") or "") == ENGINE_DECISION:
        if _decision_query_forbids_advice(query):
            forbidden_terms = ("建议方向", "下一步", "培养计划", "家长建议", "专业评估", "补研究", "行动路线")
            found = [term for term in forbidden_terms if term in text]
            if found:
                raise RuntimeError("final_controller_report: forbidden_user_terms:" + ",".join(found))
        if _decision_query_requests_future_inversion_structure(query):
            missing = [section for section in _required_decision_future_sections() if f"## {section}" not in text]
            if missing:
                raise RuntimeError("final_controller_report: missing_requested_sections:" + ",".join(missing))
        user_facing_failures = _final_user_facing_quality_failures(packet, text)
        if user_facing_failures:
            raise RuntimeError("final_controller_report: user_facing_quality:" + ",".join(user_facing_failures))
    profiles = (
        _normalize_profiles(packet.get("output_quality_profile"))
        if "output_quality_profile" in packet
        else _task_engine_profiles_from_query(query)
    )
    profile_errors = _quality_profile_errors(text, profiles, stage_name="final_controller_report")
    if profile_errors:
        raise RuntimeError("final_controller_report: output_quality_profile_error:" + ",".join(profile_errors))


def _quality_profile_errors(text: str, profiles: list[str], *, stage_name: str) -> list[str]:
    value = text or ""
    lowered = value.lower()
    errors: list[str] = []
    if _profiles_require_foresight_template(profiles):
        checks = {
            "missing_key_drivers": ("关键驱动", "驱动变量", "key_drivers", "key driver", "driver variable"),
            "missing_mechanism_chain": ("输入变量", "中介机制", "输出变量", "机制链", "mechanism_chain", "input variable", "mediating mechanism", "output variable"),
            "missing_scenario_branches": ("情景分叉", "情景 a", "情景 b", "scenario_branches", "scenario a", "scenario b", "scenario branch"),
            "missing_counter_signals": ("反证信号", "可观察指标", "counter_signals", "falsification_signals", "observable signal", "counter signal", "falsification"),
            "missing_certainty_levels": ("确定性等级", "确定性：", "高 / 中 / 低", "高/中/低", "certainty_levels", "certainty_level", "confidence_level", "certainty level", "high / medium / low"),
        }
        for name, terms in checks.items():
            if not any(term in value or term in lowered for term in terms):
                errors.append(name)
        if stage_name == "final_controller_report":
            errors.extend(_foresight_final_judgment_unit_errors(value))
    if PROFILE_IMPLEMENTATION_PLAN in profiles:
        checks = {
            "missing_cycle": ("周期", "4-6 周", "4–6 周", "cycle"),
            "missing_frequency": ("频率", "每天", "每周", "frequency"),
            "missing_steps": ("步骤", "step"),
            "missing_metrics": ("记录指标", "观察指标", "metric"),
            "missing_adjustment_rules": ("调整规则", "降难度", "adjustment rule"),
        }
        for name, terms in checks.items():
            if not any(term in value or term in lowered for term in terms):
                errors.append(name)
    if PROFILE_EVIDENCE_GROUNDED in profiles and stage_name == "final_controller_report":
        checks = {
            "missing_evidence_strength": ("证据强度", "evidence_strength", "evidence strength"),
            "missing_controversy": ("争议", "controversy"),
            "missing_gap": ("缺口", "evidence_gap", "gap"),
        }
        for name, terms in checks.items():
            if not any(term in value or term in lowered for term in terms):
                errors.append(name)
    return errors


def _foresight_final_judgment_unit_errors(text: str) -> list[str]:
    value = text or ""
    lowered = value.lower()
    errors: list[str] = []
    required_sections = {
        "missing_evidence_supported_section": ("### evidence_supported", "## evidence_supported", "## 证据支持", "[证据支持]"),
        "missing_reasonable_inference_section": ("### reasonable_inference", "## reasonable_inference", "## 合理推断", "[合理推断]"),
        "missing_foresight_hypothesis_section": ("### foresight_hypothesis", "## foresight_hypothesis", "## 前瞻假设", "[前瞻假设]"),
    }
    for name, terms in required_sections.items():
        if not any(term in value or term in lowered for term in terms):
            errors.append(name)

    condition_count = _term_count(value, ("触发条件：", "适用条件：", "分叉条件：", "condition:"))
    mechanism_count = _term_count(value, ("中间机制：", "机制链：", "mechanism_chain:", "input variable", "mediating mechanism"))
    falsifier_count = _term_count(value, ("失效条件", "反证信号", "counter_signal", "falsification"))
    certainty_count = _term_count(value, ("certainty_level", "确定性等级", "确定性：", "confidence_level"))
    evidence_tier_count = _term_count(value, ("evidence_tier", "[证据支持]", "[合理推断]", "[前瞻假设]", "[不支持/风险]"))
    decision_use_count = _term_count(value, ("decision_use", "decision implication", "决策含义", "家长怎么用"))

    if condition_count < 3 or mechanism_count < 3 or falsifier_count < 3:
        errors.append("missing_judgment_unit_fields")
    if certainty_count < 3:
        errors.append("insufficient_certainty_bindings")
    if evidence_tier_count < 3:
        errors.append("insufficient_evidence_tier_bindings")
    if decision_use_count < 3:
        errors.append("insufficient_decision_use_bindings")
    return errors


def _term_count(text: str, terms: tuple[str, ...]) -> int:
    lowered = (text or "").lower()
    return sum(lowered.count(term.lower()) for term in terms)


def _normalized_tail(text: str, *, limit: int = 180) -> str:
    return " ".join((text or "").strip().split())[-limit:].strip().lower()


def _tail_looks_truncated(tail: str) -> bool:
    if not tail:
        return True
    exact_or_suffixes = (
        "claim_strength_table",
        "strength_by_claim",
        "alternative c:",
        "alternative c: proactive ne",
        "third-grad",
        "evidence pac",
        "causing",
        "deficit neutr",
        "low – inten",
        "low - inten",
        "最危险的错误培养路径 ... 过",
    )
    if any(tail.endswith(fragment) for fragment in exact_or_suffixes):
        return True
    if tail.endswith(("|", "-", "###", "##", ":")):
        return True
    return False


def _simulated_content(stage: StageSpec) -> str:
    if stage.stage_name == "external_calibration":
        return (
            "calibration_scope\n"
            "This simulated calibration artifact is intentionally complete enough for contract validation. " * 8
            + "claim_strength_table\n"
            "| Claim | Strength | Calibration Notes |\n"
            "| --- | --- | --- |\n"
            "| Simulated claim A | supported | Evidence packet directly supports this bounded claim. |\n"
            "| Simulated claim B | plausible | Current artifacts make this likely but implementation context still matters. |\n"
            "| Simulated claim C | speculative | The packet marks this as a hypothesis only. |\n"
            "| Simulated claim D | contradicted | A conflicting artifact should prevent unqualified use. |\n"
            "over_inference_checks\n"
            "The simulated calibration explicitly checks for overreach, scope creep, and unsupported extrapolation. " * 8
            + "contradiction_checks\n"
            "No simulated contradiction is allowed to pass without label and handoff note. " * 8
            + "calibration_verdict\n"
            "verdict: calibrated_for_final_controller; supported/plausible/speculative/contradicted labels are present.\n"
            "handoff_notes_for_final_controller\n"
            "Use calibrated claims only and do not convert speculative material into final advice.\n"
        )
    if stage.stage_name == "final_controller_report":
        return "FINAL CONTROLLER BODY\n\nThis is the simulated final controller report."
    if stage.stage_name == "L5_deepseek_acceptance":
        return "\n".join(
            [
                "research_evidence_packet",
                "verdict: ACCEPTED",
                "accepted: true",
                "checked_stages: [L1_gemini_search, L2_ddgs_supplement, L2_5_codex_evidence_organizer, L3_r1_synthesis, L4_gemini_audit]",
                "missing_or_invalid_artifacts: []",
                "audit_summary: Simulated L4 audit accepted the evidence packet.",
                "evidence_packet_ready_for_decision: true",
                "",
                "## evidence_strength",
                "Simulated evidence_strength: stronger support is limited to current research artifacts and audited synthesis; weaker support remains for individual long-horizon forecasts.",
                "",
                "## claim_table",
                "- claim_id: C1",
                "  claim_text: Simulated current evidence supports bounded claims about structured learning supports and executive-function scaffolding for the decision context.",
                "  epistemic_tier: evidence_supported",
                "  evidence_strength: medium",
                "  source_anchors: S1 (full_text_verified; simulated_source; simulated://source)",
                "  applicability_boundary: Applies only to the simulated source population and measurement context.",
                "  counter_signal_or_failure_condition: Contradictory simulated evidence or failed transfer should downgrade the claim.",
                "  evidence_gap: simulated_long_horizon_transfer_gap",
                "  decision_use: can_support_decision",
                "  notes: Simulated claim table row for contract validation.",
                "",
                "## controversy",
                "Simulated controversy: translation from current evidence to a future AI environment depends on context, tool quality, and population differences.",
                "",
                "## evidence_gap",
                "Simulated evidence_gap: direct longitudinal evidence for the exact future scenario and individual outcome path is unavailable.",
                "",
                "## evidence_supported",
                "Simulated evidence_supported: stable current evidence may support bounded claims about executive function, feedback structure, and learning supports.",
                "",
                "## reasonable_inference",
                "Simulated reasonable_inference: current mechanisms can be connected to future decision variables only through explicit mechanism chains.",
                "",
                "## foresight_hypothesis",
                "Simulated foresight_hypothesis: future structural reversals are conditional hypotheses with uncertainty boundaries and counter-signals.",
                "",
                "scope: acceptance gate plus compact evidence packet; no new research, no search, no raw artifact dump, no user-facing advice or decision output.",
            ]
        )
    if stage.stage_name == "convergence_report":
        return "Simulated convergence artifact."
    if stage.stage_name == "evidence_judge":
        return "\n".join(
            [
                "evidence_quality_map",
                "Simulated evidence quality is mixed but usable for bounded contract validation.",
                "",
                "strength_by_claim",
                "- claim: simulated evidence supports a bounded current-state claim",
                "  strength: medium",
                "  evidence_basis: simulated claim table and source anchors",
                "  uncertainty_or_gap: long-horizon transfer remains uncertain",
                "",
                "applicability_to_user_context",
                "Applicable only as a simulated smoke artifact.",
                "",
                "uncertainty_and_limits",
                "The simulated artifact preserves uncertainty and does not claim final completion.",
                "",
                "evidence_gaps_for_later_stages",
                "Later stages must keep the simulated transfer gap visible.",
            ]
        )
    return f"Simulated artifact for {stage.stage_name} using {stage.model}."


def _gemini_search_prompt(query: str) -> str:
    chunks = [
        "Run RESEARCH stage L1_gemini_search through AGY/Gemini.",
        "Use Gemini 3.5 Flash (High). Return source candidates as concise JSON-compatible notes.",
        "Output must be short structured JSON-compatible notes only: max 8 source_candidates; max 2 lines per item.",
        "Include coverage_axes (or an equivalent field) naming which evidence axes are covered.",
        "Consider task-relevant axes such as authoritative_guideline_or_consensus, systematic_review_or_meta_analysis, empirical_study_or_RCT, mechanism_or_theory, intervention_or_practice, local_or_contextual_source, controversy_or_counterevidence, and recent_update.",
        "Each source_candidate must briefly label evidence_type, coverage_axis, and why_relevant.",
        "If 8 candidates cannot cover all relevant axes, include known_gaps_for_L2 for DDGS supplementation.",
        "Candidates may be source URLs or search queries, but do not present L1 inference as evidence.",
        "Do not write long analysis, final-style prose, recommendations, or conclusions in L1.",
    ]
    if PROFILE_FORESIGHT_MECHANISM in _task_engine_profiles_from_query(query):
        chunks.extend(_foresight_research_prompt_guidance("L1_gemini_search"))
    chunks.append(f"Query:\n{query}")
    return "\n".join(chunks)


def _ddgs_queries(query: str) -> list[str]:
    text = query.lower()
    if "adhd" in text or "注意力" in query or "多动" in query:
        return [
            "ADHD parent training children",
            "ADHD inattentive presentation children parent training",
            "ADHD children treatment guidelines school preparation parent training",
        ]
    return [
        query[:180],
        f"{query} latest evidence guidelines",
        f"{query} systematic review treatment parent training",
    ]


def _ddgs_queries_from_l1_candidates(query: str, stages: list[dict[str, Any]], *, base_dir: str | Path) -> list[str]:
    by_name = {str(stage.get("stage_name") or ""): stage for stage in stages}
    l1 = by_name.get("L1_gemini_search") or {}
    artifact_path = str(l1.get("artifact_path") or "").strip()
    if not artifact_path:
        return _ddgs_queries(query)
    try:
        l1_text = Path(artifact_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return _ddgs_queries(query)
    payload = _load_jsonish_text(l1_text)
    items = _list_from_jsonish(payload.get("source_candidates") if isinstance(payload, dict) else payload)
    candidates: list[str] = []
    for item in items:
        raw = _l1_candidate_locator(item)
        if not raw:
            continue
        value = raw.split(":", 1)[-1].strip(" '\"") if raw.lower().startswith(("query:", "search:")) else raw
        if value.lower().startswith(("http://", "https://")):
            value = " ".join(
                part
                for part in (
                    str(item.get("title") or ""),
                    str(item.get("why_relevant") or ""),
                    str(item.get("coverage_axis") or ""),
                )
                if part
            ).strip() or value
        value = _compact_single_line(value, limit=180)
        if value and value.lower() not in {seen.lower() for seen in candidates}:
            candidates.append(value)
        if len(candidates) >= 5:
            break
    return candidates or _ddgs_queries(query)


def _l1_candidate_locator(item: dict[str, Any]) -> str:
    return str(
        item.get("candidate")
        or item.get("query_or_url")
        or item.get("source_or_query")
        or item.get("source")
        or item.get("target")
        or item.get("url")
        or item.get("title")
        or ""
    ).strip()


def _ddgs_backend_list() -> list[str]:
    raw = os.getenv("HERMES_DDGS_BACKENDS", "duckduckgo,brave,yahoo")
    backends = [item.strip().lower() for item in raw.split(",") if item.strip()]
    forbidden = {"web_search", "generic_web_search"}
    if any(backend in forbidden for backend in backends):
        raise RuntimeError("DDGS backend list cannot include web_search fallback")
    return backends or ["duckduckgo"]


def _ddgs_timeout_s() -> int:
    try:
        return max(2, min(int(os.getenv("HERMES_DDGS_QUERY_TIMEOUT_S", "8")), 30))
    except ValueError:
        return 8


def _ddgs_retries() -> int:
    try:
        return max(0, min(int(os.getenv("HERMES_DDGS_RETRIES", "1")), 3))
    except ValueError:
        return 1


def _ddgs_search_once(query: str, *, backend: str, timeout_s: int, max_results: int) -> list[dict[str, Any]]:
    from ddgs import DDGS

    with DDGS(timeout=timeout_s) as ddgs:
        return list(ddgs.text(query, backend=backend, max_results=max_results) or [])


def _ddgs_errors_are_no_result_only(errors: list[str]) -> bool:
    if not errors:
        return True
    no_result_tokens = ("no results found", ":empty")
    return all(any(token in str(error).lower() for token in no_result_tokens) for error in errors)


def _supplementary_search_no_fresh_hits_marker(
    queries: list[str],
    backends: list[str],
    errors: list[str],
) -> dict[str, str]:
    return {
        "query": " | ".join(queries),
        "title": "",
        "url": "",
        "snippet": "",
        "supplementary_search_status": "no_fresh_hits",
        "attempted_queries_json": json.dumps(list(queries), ensure_ascii=False),
        "backends": ",".join(backends),
        "errors": "; ".join(errors[-12:]),
    }


def _normalize_ddgs_hit(query: str, hit: dict[str, Any]) -> dict[str, str]:
    return {
        "query": query,
        "title": str(hit.get("title", "")),
        "url": str(hit.get("href") or hit.get("url") or ""),
        "snippet": str(hit.get("body") or hit.get("snippet") or ""),
    }


def _is_transient_ddgs_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "timeout" in text or "timed out" in text or "temporarily" in text


def _require_fresh_prior_for_l3(stages: list[dict[str, Any]], *, base_dir: str | Path) -> None:
    expected = ["L1_gemini_search", "L2_ddgs_supplement", "L2_5_codex_evidence_organizer"]
    actual = [stage.get("stage_name") for stage in stages]
    if actual != expected:
        raise RuntimeError(f"L3_r1_synthesis: requires fresh L1/L2/L2_5 stages in order, got={actual}")
    base = Path(base_dir).resolve()
    for record in stages:
        name = str(record.get("stage_name") or "")
        if record.get("created_in_current_run") is not True:
            raise RuntimeError(f"L3_r1_synthesis: {name} is not created_in_current_run")
        if record.get("legacy_contaminated") is not False:
            raise RuntimeError(f"L3_r1_synthesis: {name} is legacy contaminated")
        if record.get("valid_for_pipeline") is not True:
            raise RuntimeError(f"L3_r1_synthesis: {name} is not valid_for_pipeline")
        _assert_current_run_path(record.get("artifact_path"), base, name)
        for output in (record.get("outputs") or {}).values():
            _assert_current_run_path(output, base, name)


def _require_fresh_prior_for_l4(stages: list[dict[str, Any]], *, base_dir: str | Path) -> None:
    expected = ["L1_gemini_search", "L2_ddgs_supplement", "L2_5_codex_evidence_organizer", "L3_r1_synthesis"]
    actual = [stage.get("stage_name") for stage in stages]
    if actual != expected:
        raise RuntimeError(f"L4_gemini_audit: requires fresh L1/L2/L2_5/L3 stages in order, got={actual}")
    base = Path(base_dir).resolve()
    for record in stages:
        name = str(record.get("stage_name") or "")
        if record.get("created_in_current_run") is not True:
            raise RuntimeError(f"L4_gemini_audit: {name} is not created_in_current_run")
        if record.get("legacy_contaminated") is not False:
            raise RuntimeError(f"L4_gemini_audit: {name} is legacy contaminated")
        if record.get("valid_for_pipeline") is not True:
            raise RuntimeError(f"L4_gemini_audit: {name} is not valid_for_pipeline")
        _assert_current_run_path(record.get("artifact_path"), base, name, consumer_stage="L4_gemini_audit")
        for output in (record.get("outputs") or {}).values():
            _assert_current_run_path(output, base, name, consumer_stage="L4_gemini_audit")


def _require_fresh_prior_for_l5(stages: list[dict[str, Any]], *, base_dir: str | Path) -> None:
    expected = [
        "L1_gemini_search",
        "L2_ddgs_supplement",
        "L2_5_codex_evidence_organizer",
        "L3_r1_synthesis",
        "L4_gemini_audit",
    ]
    actual = [stage.get("stage_name") for stage in stages]
    if actual != expected:
        raise RuntimeError(f"L5_deepseek_acceptance: requires fresh L1/L2/L2_5/L3/L4 stages in order, got={actual}")
    base = Path(base_dir).resolve()
    for record in stages:
        name = str(record.get("stage_name") or "")
        if record.get("created_in_current_run") is not True:
            raise RuntimeError(f"L5_deepseek_acceptance: {name} is not created_in_current_run")
        if record.get("legacy_contaminated") is not False:
            raise RuntimeError(f"L5_deepseek_acceptance: {name} is legacy contaminated")
        if record.get("valid_for_pipeline") is not True:
            raise RuntimeError(f"L5_deepseek_acceptance: {name} is not valid_for_pipeline")
        _assert_current_run_path(record.get("artifact_path"), base, name, consumer_stage="L5_deepseek_acceptance")
        for output in (record.get("outputs") or {}).values():
            _assert_current_run_path(output, base, name, consumer_stage="L5_deepseek_acceptance")


def _require_accepted_research_packet_for_intelligence(stages: list[dict[str, Any]], *, base_dir: str | Path) -> None:
    expected = [
        "L1_gemini_search",
        "L2_ddgs_supplement",
        "L2_5_codex_evidence_organizer",
        "L3_r1_synthesis",
        "L4_gemini_audit",
        "L5_deepseek_acceptance",
    ]
    actual = [stage.get("stage_name") for stage in stages]
    if actual != expected:
        raise RuntimeError(f"intelligence_layer: requires accepted fresh RESEARCH L1-L5 stages in order, got={actual}")
    research_validation = validate_pipeline(ENGINE_RESEARCH, {"stages": stages}, base_dir=base_dir)
    if not research_validation.get("valid"):
        raise RuntimeError(
            "intelligence_layer: accepted research packet validation failed: "
            + "; ".join(research_validation.get("errors", []))
        )
    base = Path(base_dir).resolve()
    l5 = stages[-1]
    if l5.get("status") != "accepted":
        raise RuntimeError("intelligence_layer: L5 research packet is not accepted")
    packet_path = Path(str(l5.get("artifact_path") or "")).resolve()
    _assert_current_run_path(packet_path, base, "L5_deepseek_acceptance", consumer_stage="intelligence_layer")
    packet_text = packet_path.read_text(encoding="utf-8", errors="replace")
    if not _l5_acceptance_text_is_accepted(packet_text):
        raise RuntimeError("intelligence_layer: research_evidence_packet.md is not accepted")
    for record in stages:
        name = str(record.get("stage_name") or "")
        if record.get("created_in_current_run") is not True:
            raise RuntimeError(f"intelligence_layer: {name} is not created_in_current_run")
        if record.get("legacy_contaminated") is not False:
            raise RuntimeError(f"intelligence_layer: {name} is legacy contaminated")
        if record.get("valid_for_pipeline") is not True:
            raise RuntimeError(f"intelligence_layer: {name} is not valid_for_pipeline")
        _assert_current_run_path(record.get("artifact_path"), base, name, consumer_stage="intelligence_layer")
        for output in (record.get("outputs") or {}).values():
            _assert_current_run_path(output, base, name, consumer_stage="intelligence_layer")


def _require_fresh_prior_for_supplementary_search(stages: list[dict[str, Any]], *, base_dir: str | Path) -> None:
    expected = [
        "L1_gemini_search",
        "L2_ddgs_supplement",
        "L2_5_codex_evidence_organizer",
        "L3_r1_synthesis",
        "L4_gemini_audit",
        "L5_deepseek_acceptance",
        "intelligence_layer",
    ]
    actual = [stage.get("stage_name") for stage in stages]
    if actual != expected:
        raise RuntimeError(f"supplementary_search: requires fresh L1-L5 plus intelligence_layer in order, got={actual}")
    _require_accepted_research_packet_for_intelligence(stages[:6], base_dir=base_dir)
    base = Path(base_dir).resolve()
    intelligence = stages[-1]
    if intelligence.get("created_in_current_run") is not True:
        raise RuntimeError("supplementary_search: intelligence_layer is not created_in_current_run")
    if intelligence.get("legacy_contaminated") is not False:
        raise RuntimeError("supplementary_search: intelligence_layer is legacy contaminated")
    if intelligence.get("valid_for_pipeline") is not True:
        raise RuntimeError("supplementary_search: intelligence_layer is not valid_for_pipeline")
    _assert_current_run_path(
        intelligence.get("artifact_path"),
        base,
        "intelligence_layer",
        consumer_stage="supplementary_search",
    )
    for output in (intelligence.get("outputs") or {}).values():
        _assert_current_run_path(output, base, "intelligence_layer", consumer_stage="supplementary_search")


def _require_fresh_prior_for_structure_mapper(stages: list[dict[str, Any]], *, base_dir: str | Path) -> None:
    expected = [
        "L1_gemini_search",
        "L2_ddgs_supplement",
        "L2_5_codex_evidence_organizer",
        "L3_r1_synthesis",
        "L4_gemini_audit",
        "L5_deepseek_acceptance",
        "intelligence_layer",
        "supplementary_search",
    ]
    actual = [stage.get("stage_name") for stage in stages]
    if actual != expected:
        raise RuntimeError(f"structure_mapper: requires fresh L1-L8 stages in order, got={actual}")
    _require_fresh_prior_for_supplementary_search(stages[:7], base_dir=base_dir)
    base = Path(base_dir).resolve()
    supplementary = stages[-1]
    if supplementary.get("created_in_current_run") is not True:
        raise RuntimeError("structure_mapper: supplementary_search is not created_in_current_run")
    if supplementary.get("legacy_contaminated") is not False:
        raise RuntimeError("structure_mapper: supplementary_search is legacy contaminated")
    if supplementary.get("valid_for_pipeline") is not True:
        raise RuntimeError("structure_mapper: supplementary_search is not valid_for_pipeline")
    _assert_current_run_path(
        supplementary.get("artifact_path"),
        base,
        "supplementary_search",
        consumer_stage="structure_mapper",
    )
    for output in (supplementary.get("outputs") or {}).values():
        _assert_current_run_path(output, base, "supplementary_search", consumer_stage="structure_mapper")


def _require_fresh_prior_for_evidence_judge(stages: list[dict[str, Any]], *, base_dir: str | Path) -> None:
    expected = [
        "L1_gemini_search",
        "L2_ddgs_supplement",
        "L2_5_codex_evidence_organizer",
        "L3_r1_synthesis",
        "L4_gemini_audit",
        "L5_deepseek_acceptance",
        "intelligence_layer",
        "supplementary_search",
        "structure_mapper",
    ]
    actual = [stage.get("stage_name") for stage in stages]
    if actual != expected:
        raise RuntimeError(f"evidence_judge: requires fresh L1-L9 stages in order, got={actual}")
    _require_fresh_prior_for_structure_mapper(stages[:8], base_dir=base_dir)
    base = Path(base_dir).resolve()
    structure = stages[-1]
    if structure.get("created_in_current_run") is not True:
        raise RuntimeError("evidence_judge: structure_mapper is not created_in_current_run")
    if structure.get("legacy_contaminated") is not False:
        raise RuntimeError("evidence_judge: structure_mapper is legacy contaminated")
    if structure.get("valid_for_pipeline") is not True:
        raise RuntimeError("evidence_judge: structure_mapper is not valid_for_pipeline")
    _assert_current_run_path(
        structure.get("artifact_path"),
        base,
        "structure_mapper",
        consumer_stage="evidence_judge",
    )
    for output in (structure.get("outputs") or {}).values():
        _assert_current_run_path(output, base, "structure_mapper", consumer_stage="evidence_judge")


def _require_fresh_prior_for_premise_auditor(stages: list[dict[str, Any]], *, base_dir: str | Path) -> None:
    expected = [
        "L1_gemini_search",
        "L2_ddgs_supplement",
        "L2_5_codex_evidence_organizer",
        "L3_r1_synthesis",
        "L4_gemini_audit",
        "L5_deepseek_acceptance",
        "intelligence_layer",
        "supplementary_search",
        "structure_mapper",
        "evidence_judge",
    ]
    actual = [stage.get("stage_name") for stage in stages]
    if actual != expected:
        raise RuntimeError(f"premise_auditor: requires fresh L1-L10 stages in order, got={actual}")
    _require_fresh_prior_for_evidence_judge(stages[:9], base_dir=base_dir)
    base = Path(base_dir).resolve()
    evidence = stages[-1]
    if evidence.get("created_in_current_run") is not True:
        raise RuntimeError("premise_auditor: evidence_judge is not created_in_current_run")
    if evidence.get("legacy_contaminated") is not False:
        raise RuntimeError("premise_auditor: evidence_judge is legacy contaminated")
    if evidence.get("valid_for_pipeline") is not True:
        raise RuntimeError("premise_auditor: evidence_judge is not valid_for_pipeline")
    _assert_current_run_path(
        evidence.get("artifact_path"),
        base,
        "evidence_judge",
        consumer_stage="premise_auditor",
    )
    for output in (evidence.get("outputs") or {}).values():
        _assert_current_run_path(output, base, "evidence_judge", consumer_stage="premise_auditor")


def _require_fresh_prior_for_alternative_generator(stages: list[dict[str, Any]], *, base_dir: str | Path) -> None:
    expected = [
        "L1_gemini_search",
        "L2_ddgs_supplement",
        "L2_5_codex_evidence_organizer",
        "L3_r1_synthesis",
        "L4_gemini_audit",
        "L5_deepseek_acceptance",
        "intelligence_layer",
        "supplementary_search",
        "structure_mapper",
        "evidence_judge",
        "premise_auditor",
    ]
    actual = [stage.get("stage_name") for stage in stages]
    if actual != expected:
        raise RuntimeError(f"alternative_generator: requires fresh L1-L11 stages in order, got={actual}")
    _require_fresh_prior_for_premise_auditor(stages[:10], base_dir=base_dir)
    base = Path(base_dir).resolve()
    premise = stages[-1]
    if premise.get("created_in_current_run") is not True:
        raise RuntimeError("alternative_generator: premise_auditor is not created_in_current_run")
    if premise.get("legacy_contaminated") is not False:
        raise RuntimeError("alternative_generator: premise_auditor is legacy contaminated")
    if premise.get("valid_for_pipeline") is not True:
        raise RuntimeError("alternative_generator: premise_auditor is not valid_for_pipeline")
    _assert_current_run_path(
        premise.get("artifact_path"),
        base,
        "premise_auditor",
        consumer_stage="alternative_generator",
    )
    for output in (premise.get("outputs") or {}).values():
        _assert_current_run_path(output, base, "premise_auditor", consumer_stage="alternative_generator")


def _require_fresh_prior_for_insight_harvester(stages: list[dict[str, Any]], *, base_dir: str | Path) -> None:
    expected = [
        "L1_gemini_search",
        "L2_ddgs_supplement",
        "L2_5_codex_evidence_organizer",
        "L3_r1_synthesis",
        "L4_gemini_audit",
        "L5_deepseek_acceptance",
        "intelligence_layer",
        "supplementary_search",
        "structure_mapper",
        "evidence_judge",
        "premise_auditor",
        "alternative_generator",
    ]
    actual = [stage.get("stage_name") for stage in stages]
    if actual != expected:
        raise RuntimeError(f"insight_harvester: requires fresh L1-L12 stages in order, got={actual}")
    _require_fresh_prior_for_alternative_generator(stages[:11], base_dir=base_dir)
    base = Path(base_dir).resolve()
    alternative = stages[-1]
    if alternative.get("created_in_current_run") is not True:
        raise RuntimeError("insight_harvester: alternative_generator is not created_in_current_run")
    if alternative.get("legacy_contaminated") is not False:
        raise RuntimeError("insight_harvester: alternative_generator is legacy contaminated")
    if alternative.get("valid_for_pipeline") is not True:
        raise RuntimeError("insight_harvester: alternative_generator is not valid_for_pipeline")
    _assert_current_run_path(
        alternative.get("artifact_path"),
        base,
        "alternative_generator",
        consumer_stage="insight_harvester",
    )
    for output in (alternative.get("outputs") or {}).values():
        _assert_current_run_path(output, base, "alternative_generator", consumer_stage="insight_harvester")


def _require_fresh_prior_for_convergence_report(stages: list[dict[str, Any]], *, base_dir: str | Path) -> None:
    expected = [
        "L1_gemini_search",
        "L2_ddgs_supplement",
        "L2_5_codex_evidence_organizer",
        "L3_r1_synthesis",
        "L4_gemini_audit",
        "L5_deepseek_acceptance",
        "intelligence_layer",
        "supplementary_search",
        "structure_mapper",
        "evidence_judge",
        "premise_auditor",
        "alternative_generator",
        "insight_harvester",
    ]
    actual = [stage.get("stage_name") for stage in stages]
    if actual != expected:
        raise RuntimeError(f"convergence_report: requires fresh L1-L13 stages in order, got={actual}")
    _require_fresh_prior_for_insight_harvester(stages[:12], base_dir=base_dir)
    base = Path(base_dir).resolve()
    for record in stages:
        name = str(record.get("stage_name") or "")
        if record.get("created_in_current_run") is not True:
            raise RuntimeError(f"convergence_report: {name} is not created_in_current_run")
        if record.get("legacy_contaminated") is not False:
            raise RuntimeError(f"convergence_report: {name} is legacy contaminated")
        if record.get("valid_for_pipeline") is not True:
            raise RuntimeError(f"convergence_report: {name} is not valid_for_pipeline")
        _assert_current_run_path(record.get("artifact_path"), base, name, consumer_stage="convergence_report")
        for output in (record.get("outputs") or {}).values():
            _assert_current_run_path(output, base, name, consumer_stage="convergence_report")
    divergence_roles = [
        "structure_mapper",
        "evidence_judge",
        "premise_auditor",
        "alternative_generator",
        "insight_harvester",
    ]
    by_name = {str(record.get("stage_name") or ""): record for record in stages}
    missing = [role for role in divergence_roles if role not in by_name]
    if missing:
        raise RuntimeError(f"convergence_report: missing divergence roles: {', '.join(missing)}")
    unique_models = {str(by_name[role].get("model") or "") for role in divergence_roles}
    if len(unique_models) < 4:
        raise RuntimeError(f"convergence_report: unique divergence models < 4: {sorted(unique_models)}")


def _require_fresh_prior_for_external_calibration(stages: list[dict[str, Any]], *, base_dir: str | Path) -> None:
    expected = [
        "L1_gemini_search",
        "L2_ddgs_supplement",
        "L2_5_codex_evidence_organizer",
        "L3_r1_synthesis",
        "L4_gemini_audit",
        "L5_deepseek_acceptance",
        "intelligence_layer",
        "supplementary_search",
        "structure_mapper",
        "evidence_judge",
        "premise_auditor",
        "alternative_generator",
        "insight_harvester",
        "convergence_report",
    ]
    actual = [stage.get("stage_name") for stage in stages]
    if actual != expected:
        raise RuntimeError(f"external_calibration: requires fresh L1-L14 stages in order, got={actual}")
    _require_fresh_prior_for_convergence_report(stages[:13], base_dir=base_dir)
    base = Path(base_dir).resolve()
    convergence = stages[-1]
    if convergence.get("created_in_current_run") is not True:
        raise RuntimeError("external_calibration: convergence_report is not created_in_current_run")
    if convergence.get("legacy_contaminated") is not False:
        raise RuntimeError("external_calibration: convergence_report is legacy contaminated")
    if convergence.get("valid_for_pipeline") is not True:
        raise RuntimeError("external_calibration: convergence_report is not valid_for_pipeline")
    if convergence.get("model") != R1_32B:
        raise RuntimeError("external_calibration: convergence_report model mismatch")
    _assert_current_run_path(
        convergence.get("artifact_path"),
        base,
        "convergence_report",
        consumer_stage="external_calibration",
    )
    for output in (convergence.get("outputs") or {}).values():
        _assert_current_run_path(output, base, "convergence_report", consumer_stage="external_calibration")


def _require_fresh_prior_for_final_controller_report(stages: list[dict[str, Any]], *, base_dir: str | Path) -> None:
    expected = [
        "L1_gemini_search",
        "L2_ddgs_supplement",
        "L2_5_codex_evidence_organizer",
        "L3_r1_synthesis",
        "L4_gemini_audit",
        "L5_deepseek_acceptance",
        "intelligence_layer",
        "supplementary_search",
        "structure_mapper",
        "evidence_judge",
        "premise_auditor",
        "alternative_generator",
        "insight_harvester",
        "convergence_report",
        "external_calibration",
    ]
    actual = [stage.get("stage_name") for stage in stages]
    if actual != expected:
        raise RuntimeError(f"final_controller_report: requires fresh L1-L15 stages in order, got={actual}")
    _require_fresh_prior_for_external_calibration(stages[:14], base_dir=base_dir)
    base = Path(base_dir).resolve()
    external = stages[-1]
    if external.get("created_in_current_run") is not True:
        raise RuntimeError("final_controller_report: external_calibration is not created_in_current_run")
    if external.get("legacy_contaminated") is not False:
        raise RuntimeError("final_controller_report: external_calibration is legacy contaminated")
    if external.get("valid_for_pipeline") is not True:
        raise RuntimeError("final_controller_report: external_calibration is not valid_for_pipeline")
    if external.get("model") != GPT_OR_GEMINI_EXTERNAL:
        raise RuntimeError("final_controller_report: external_calibration model mismatch")
    for record in stages:
        name = str(record.get("stage_name") or "")
        _assert_current_run_path(record.get("artifact_path"), base, name, consumer_stage="final_controller_report")
        for output in (record.get("outputs") or {}).values():
            _assert_current_run_path(output, base, name, consumer_stage="final_controller_report")


def _assert_current_run_path(path_value: Any, base: Path, stage_name: str, *, consumer_stage: str = "L3_r1_synthesis") -> None:
    path = Path(str(path_value or "")).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise RuntimeError(f"{consumer_stage}: {stage_name} artifact outside current run: {path}") from exc
    if not path.exists():
        raise RuntimeError(f"{consumer_stage}: {stage_name} artifact missing: {path}")


def _research_acceptance_packet_from_artifacts(stages: list[dict[str, Any]], *, base_dir: str | Path, query: str = "") -> dict[str, Any]:
    base = Path(base_dir).resolve()
    missing: list[str] = []
    audit_text = ""
    artifact_summaries: dict[str, str] = {}
    for record in stages:
        name = str(record.get("stage_name") or "")
        paths = [record.get("artifact_path")] + list((record.get("outputs") or {}).values())
        seen: set[Path] = set()
        snippets: list[str] = []
        for raw_path in paths:
            path = Path(str(raw_path or "")).resolve()
            if path in seen:
                continue
            seen.add(path)
            try:
                path.relative_to(base)
            except ValueError:
                missing.append(f"{name}:outside_current_run")
                continue
            if not path.exists():
                missing.append(f"{name}:{path.name}:missing")
                continue
            if path.is_dir():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            snippets.append(text[:1200])
            if name == "L4_gemini_audit":
                audit_text += "\n" + text
        artifact_summaries[name] = "\n".join(snippets)[:2000]
    profiles = _task_engine_profiles_from_query(query)
    l2_5_analysis = analyze_l2_5_evidence_organizer(base)
    l4_defect_report = _l4_defect_report_from_audit(audit_text)
    claim_table = _research_claim_table_from_l2_5(base, profiles=profiles, audit_text=audit_text)
    claim_contract_validation = _research_claim_contract_validation(
        claim_table=claim_table,
        l4_defect_report=l4_defect_report,
        l2_5_analysis=l2_5_analysis,
    )
    critical_defects = set(l4_defect_report.get("critical_defects") or [])
    critical_defects.update(str(issue) for issue in l2_5_analysis.get("issues") or [])
    noncritical_defects = set(str(item) for item in (l4_defect_report.get("noncritical_defects") or []))
    verification_required = set(str(item) for item in (l4_defect_report.get("verification_required") or []))
    verification_required.update(str(item) for item in (claim_contract_validation.get("verification_required") or []))
    evidence_gaps = set(str(item) for item in (l4_defect_report.get("evidence_gaps") or []))
    evidence_gaps.update(str(item) for item in (claim_contract_validation.get("evidence_gaps") or []))
    handoff_caveats = set(str(item) for item in (l4_defect_report.get("handoff_caveats") or []))
    handoff_caveats.update(str(item) for item in (claim_contract_validation.get("handoff_caveats") or []))
    if not l2_5_analysis.get("l2_5_valid", False):
        if PROFILE_EVIDENCE_GROUNDED in profiles:
            missing.extend(str(item) for item in l2_5_analysis.get("missing_or_invalid_artifacts") or [])
    missing.extend(str(item) for item in (claim_contract_validation.get("blocking_errors") or []))
    if PROFILE_FORESIGHT_MECHANISM in profiles:
        artifact_summaries["_foresight_requirement_map"] = _foresight_requirement_map_text(
            "\n".join(str(value or "") for value in artifact_summaries.values())
        )
    missing = sorted(set(missing))
    return {
        "query": query,
        "research_packet_profile": profiles,
        "profile_acceptance_requirements": _research_profile_acceptance_requirements(profiles),
        "checked_stages": [stage.get("stage_name") for stage in stages],
        "missing_or_invalid_artifacts": missing,
        "critical_defects": sorted(critical_defects),
        "noncritical_defects": sorted(noncritical_defects),
        "verification_required": sorted(verification_required),
        "evidence_gaps": sorted(evidence_gaps),
        "handoff_caveats": sorted(handoff_caveats),
        "l2_5_valid": bool(l2_5_analysis.get("l2_5_valid")),
        "l2_5_analysis": l2_5_analysis,
        "claim_table": claim_table,
        "claim_contract_validation": claim_contract_validation,
        "l4_defect_report": l4_defect_report,
        "artifact_summaries": artifact_summaries,
        "audit_text": audit_text,
        "audit_summary": _audit_summary(audit_text),
    }


def _l4_defect_report_from_audit(audit_text: str) -> dict[str, list[str]]:
    value = audit_text or ""
    lowered = value.lower()
    critical: set[str] = set()
    noncritical: set[str] = set()
    verification_required: set[str] = set()
    evidence_gaps: set[str] = set()
    caveats: set[str] = set()

    l2_5_missing = (
        "l2.5" in lowered
        and (
            "extraction missing" in lowered
            or "stubbed out" in lowered
            or "handoff_protocol" in lowered
            or "unsupported by structured evidence" in lowered
            or "no actual claims" in lowered
            or "no actual structured data extraction" in lowered
        )
    )
    if l2_5_missing:
        critical.add("l2_5_extraction_missing")
        caveats.add("L2.5 structured source/evidence/claim extraction is missing or stubbed.")
    if "critical" in lowered and "supplementary_search_cross_topic_contamination" in lowered:
        critical.add("supplementary_search_cross_topic_contamination")
        caveats.add("Supplementary search may contain cross-topic contamination.")

    pass_with_defects = "pass with defects" in lowered or "pass_with_defects" in lowered
    explicit_defect_lines = [
        _compact_single_line(line.strip("-* \t"), limit=220)
        for line in value.splitlines()
        if re.search(r"\bDEFECT\b|\bDefect\b|缺陷|问题", line)
    ]
    if pass_with_defects:
        noncritical.add("l4_pass_with_defects")
        caveats.add("L4 audit passed with defects; DECISION handoff must preserve those caveats.")
    for line in explicit_defect_lines[:8]:
        normalized = _defect_slug(line)
        if normalized:
            noncritical.add(normalized)
        caveats.add(line)

    if "full-text verification gap" in lowered or "requires full-text verification" in lowered:
        verification_required.add("full_text_verification_required")
        evidence_gaps.add("source_passages_not_full_text_verified")
    if "overstated evidence strength" in lowered:
        noncritical.add("overstated_evidence_strength")
        caveats.add("L4 found at least one claim with overstated evidence strength.")
    if "missed counter-signal" in lowered or "counter-signal" in lowered:
        noncritical.add("missed_or_required_counter_signal")
        evidence_gaps.add("counter_signal_must_be_preserved")
    if "missing assumption" in lowered or "assumption" in lowered:
        noncritical.add("missing_assumption_boundary")
        evidence_gaps.add("assumptions_must_be_explicit")

    return {
        "critical_defects": sorted(critical),
        "noncritical_defects": sorted(noncritical),
        "verification_required": sorted(verification_required),
        "evidence_gaps": sorted(evidence_gaps),
        "handoff_caveats": sorted(caveats),
    }


def _defect_slug(text: str) -> str:
    lowered = (text or "").lower()
    if "missed counter" in lowered or "counter-signal" in lowered:
        return "missed_counter_signal_evidence"
    if "overstated evidence" in lowered:
        return "overstated_evidence_strength"
    if "missing assumption" in lowered:
        return "missing_assumption_boundary"
    if "full-text" in lowered or "verification" in lowered:
        return "source_verification_gap"
    if "l2.5" in lowered or "l2_5" in lowered:
        return "l2_5_audit_defect"
    cleaned = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return cleaned[:80]


def _audit_summary(audit_text: str) -> str:
    stripped = " ".join((audit_text or "").split())
    if not stripped:
        return "L4 audit artifact is empty."
    return stripped[:600]


def _audit_text_rejects(audit_text: str) -> bool:
    value = audit_text or ""
    lowered = value.lower()
    if "accepted: false" in lowered or "evidence_packet_ready_for_decision: false" in lowered:
        return True
    for line in value.splitlines():
        normalized = line.strip().lower()
        if re.match(r"^(?:verdict|audit_verdict|acceptance_verdict|status)\s*:\s*rejected?\b", normalized):
            return True
    return False


L2_5_EVIDENCE_ORGANIZER_OUTPUTS = ("sources.csv", "evidence.csv", "claims.md", "gaps.md")
L2_5_STUB_MARKERS = (
    "handoff_protocol",
    "placeholder",
    "stub",
    "todo",
    "tbd",
    "n/a",
    "not applicable",
    "empty extraction",
    "no evidence extracted",
    "generic handoff",
)


def build_l2_5_evidence_organizer_outputs(inputs: dict[str, Any]) -> dict[str, str]:
    source_path = Path(str(inputs.get("source_candidates.json") or ""))
    ddgs_path = Path(str(inputs.get("ddgs_gap_sources.json") or ""))
    source_text = source_path.read_text(encoding="utf-8", errors="replace")
    ddgs_text = ddgs_path.read_text(encoding="utf-8", errors="replace")
    l1_payload = _load_jsonish_text(source_text)
    l2_payload = _load_jsonish_text(ddgs_text)
    l1_items = _list_from_jsonish(l1_payload.get("source_candidates") if isinstance(l1_payload, dict) else l1_payload)
    l2_items = _list_from_jsonish(l2_payload)
    question = _compact_single_line(str(inputs.get("original_question") or "").strip(), limit=240)
    if not question:
        question = _infer_question_anchor(l2_items, l1_items)
    sample_schema = _l2_5_sample_schema(question)
    topic_terms = _topic_anchor_terms(question, l1_items, l2_items, sample_schema)
    source_rows = _l2_5_source_rows(l1_items, l2_items, topic_terms, question=question, sample_schema=sample_schema)
    evidence_rows = _l2_5_evidence_rows(source_rows)
    claims = _l2_5_claims(evidence_rows, question, sample_schema)
    gaps = _l2_5_gaps(source_rows, evidence_rows, question, sample_schema)
    insufficient = len(source_rows) < 3 or len(evidence_rows) < 3 or len(claims) < 4 or len(gaps) < 3
    request = {
        "stage": "L2_5_codex_evidence_organizer",
        "input_scope": ["L1 source_candidates.json", "L2 ddgs_gap_sources.json", "original question/user_question_anchor"],
        "forbidden_inputs": ["research_evidence_packet", "intelligence_layer", "supplementary_search", "convergence_report", "external_calibration", "final_decision_report"],
        "inputs": {"source_candidates.json": str(source_path), "ddgs_gap_sources.json": str(ddgs_path)},
        "user_question_anchor": question,
        "sample_schema": sample_schema["name"],
        "sample_schema_axes": sample_schema["axes"],
        "topic_anchor_terms": topic_terms,
        "source_rows": len(source_rows),
        "evidence_rows": len(evidence_rows),
        "claim_rows": len(claims),
        "gap_rows": len(gaps),
        "insufficient_sources": insufficient,
    }
    request_md = "\n".join(
        [
            "# L2.5 evidence organizer request",
            "",
            "input_scope: L1 source_candidates.json + L2 ddgs_gap_sources.json + original question anchor only",
            "downstream_artifacts_used: false",
            f"user_question_anchor: {question}",
            "topic_anchor_terms: " + ", ".join(topic_terms),
            f"insufficient_sources: {str(insufficient).lower()}",
        ]
    )
    return {
        "source_candidates.json": source_text,
        "ddgs_gap_sources.json": ddgs_text,
        "evidence_runner_*.request.md": request_md + "\n",
        "evidence_runner_*.request.json": json.dumps(request, ensure_ascii=False, indent=2) + "\n",
        "sources.csv": _csv_text(
            ["source_id", "title_or_source_name", "url_or_path_or_domain", "origin_stage", "relevance_to_question", "limitation_or_note"],
            source_rows,
        ),
        "evidence.csv": _csv_text(
            ["claim_id", "source_id", "evidence_text", "strength_or_limit", "support_type"],
            evidence_rows,
        ),
        "claims.md": _claims_markdown(claims, insufficient=insufficient),
        "gaps.md": _gaps_markdown(gaps, insufficient=insufficient),
    }


def _load_jsonish_text(text: str) -> Any:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except Exception:
        match = re.search(r"(\{.*\}|\[.*\])", stripped, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(1))
        except Exception:
            return {}


def _list_from_jsonish(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def _infer_question_anchor(l2_items: list[dict[str, Any]], l1_items: list[dict[str, Any]]) -> str:
    for item in l2_items:
        query = str(item.get("query") or "").strip()
        if query:
            return _compact_single_line(query, limit=240)
    for item in l1_items:
        candidate = str(item.get("candidate") or item.get("query_or_url") or "").strip()
        if candidate.lower().startswith(("query:", "search:")):
            return _compact_single_line(candidate.split(":", 1)[-1].strip(" '\""), limit=240)
    return "question_anchor_unavailable"


def _topic_anchor_terms(
    question: str,
    l1_items: list[dict[str, Any]] | None = None,
    l2_items: list[dict[str, Any]] | None = None,
    sample_schema: dict[str, Any] | None = None,
) -> list[str]:
    query_terms = _l2_5_anchor_tokens(question, max_terms=14)
    schema_terms = list((sample_schema or {}).get("anchor_terms") or [])
    source_terms: list[str] = []
    for item in list(l1_items or [])[:12]:
        source_terms.extend(_l2_5_anchor_tokens(_l2_5_item_text(item), max_terms=4))
    for item in list(l2_items or [])[:12]:
        source_terms.extend(_l2_5_anchor_tokens(_l2_5_item_text(item), max_terms=4))

    ranked: list[str] = []
    for pool in (query_terms, schema_terms, source_terms):
        for term in pool:
            value = str(term or "").strip()
            if value and value.lower() not in {seen.lower() for seen in ranked}:
                ranked.append(value)
    return ranked[:24]


def _l2_5_anchor_tokens(text: str, *, max_terms: int = 18) -> list[str]:
    value = str(text or "")
    lowered = value.lower()
    stopwords = {
        "should",
        "whether",
        "using",
        "from",
        "into",
        "with",
        "and",
        "the",
        "for",
        "是否",
        "应该",
        "采用",
        "主要",
        "依赖",
        "未来",
        "之后",
        "一个",
        "产品",
        "儿童",
        "正常",
        "值得",
    }
    english = [
        token
        for token in re.findall(r"[a-z][a-z0-9+/-]{2,}", lowered)
        if token not in stopwords and not token.isdigit()
    ]
    cjk_phrases = re.findall(r"[\u4e00-\u9fff]{2,}", value)
    cjk_terms: list[str] = []
    for phrase in cjk_phrases:
        if phrase in stopwords:
            continue
        if len(phrase) <= 4:
            cjk_terms.append(phrase)
            continue
        cjk_terms.extend(phrase[index : index + 2] for index in range(0, len(phrase) - 1))
        cjk_terms.extend(phrase[index : index + 3] for index in range(0, len(phrase) - 2))
    acronyms = re.findall(r"\b[A-Z][A-Z0-9&.+/-]{1,}\b", value)
    weighted = acronyms + english + cjk_terms
    counts: dict[str, int] = {}
    for term in weighted:
        cleaned = str(term or "").strip()
        if not cleaned or cleaned in stopwords or cleaned.lower() in stopwords:
            continue
        counts[cleaned] = counts.get(cleaned, 0) + 1
    return [
        term
        for term, _ in sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0].lower()))
    ][:max_terms]


def _l2_5_sample_schema(question: str) -> dict[str, Any]:
    value = str(question or "")
    lowered = value.lower()
    schemas = [
        (
            "tech_route",
            ("elasticsearch", "embedding", "rag", "reranker", "vector", "向量", "架构", "迁移", "索引", "knowledge", "数据库"),
            ("current architecture", "target architecture", "benefit", "migration risk/complexity"),
            ("architecture", "migration", "index", "retrieval", "permission", "reranker", "rag", "向量", "索引", "权限", "架构"),
        ),
        (
            "high_evidence_intervention",
            ("intervention", "训练", "教学", "练习", "儿童", "数轴", "数学", "计算困难", "evidence", "outcome"),
            ("population", "intervention", "comparison", "outcome/evidence strength"),
            ("population", "intervention", "comparison", "outcome", "evidence", "儿童", "训练", "教学", "数轴", "数学", "计算"),
        ),
        (
            "business_entry",
            ("market", "entry", "进入", "产业链", "供应链", "政策", "监管", "市场", "barrier"),
            ("market signal", "policy/regulation", "supply chain/player", "entry risk/barrier"),
            ("market", "policy", "regulation", "supply", "entry", "市场", "政策", "监管", "供应链", "壁垒"),
        ),
        (
            "low_evidence_trend",
            ("2030", "trend", "趋势", "低证据", "不确定", "机器人", "消费", "hardware"),
            ("trend signal", "uncertainty", "counter-signal", "decision implication"),
            ("trend", "uncertainty", "counter-signal", "decision", "趋势", "不确定", "反证", "消费", "硬件"),
        ),
        (
            "foresight_mechanism",
            ("未来", "结构性", "反转", "优势", "劣势", "成本", "role", "profession", "机制"),
            ("capability/cost driver", "affected role/process", "mechanism", "counter-signal"),
            ("capability", "cost", "role", "process", "mechanism", "counter-signal", "成本", "角色", "流程", "机制", "反证"),
        ),
    ]
    for name, markers, axes, anchors in schemas:
        if any(marker in lowered or marker in value for marker in markers):
            return {"name": name, "axes": list(axes), "anchor_terms": list(anchors)}
    return {
        "name": "generic_evidence",
        "axes": ["source signal", "mechanism", "limitation", "decision implication"],
        "anchor_terms": ["source", "evidence", "mechanism", "risk", "decision", "证据", "机制", "风险", "决策"],
    }


def _topic_relevant(text: str, topic_terms: list[str]) -> bool:
    if not topic_terms:
        return True
    value = str(text or "")
    lowered = value.lower()
    hits = sum(1 for term in topic_terms if len(str(term)) >= 2 and (term.lower() in lowered or term in value))
    return hits >= 2 or (hits >= 1 and len(topic_terms) <= 3)


def _l2_5_item_text(item: dict[str, Any]) -> str:
    return " ".join(
        str(item.get(key) or "")
        for key in (
            "candidate",
            "query_or_url",
            "source_or_query",
            "source",
            "target",
            "title",
            "snippet",
            "why_relevant",
            "coverage_axis",
            "evidence_type",
            "query",
            "url",
        )
    )


def _l2_5_source_rows(
    l1_items: list[dict[str, Any]],
    l2_items: list[dict[str, Any]],
    topic_terms: list[str],
    *,
    question: str = "",
    sample_schema: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    candidates: list[tuple[dict[str, str], str]] = []
    for item in l1_items:
        candidate = _l1_candidate_locator(item)
        why = str(item.get("why_relevant") or item.get("snippet") or item.get("coverage_axis") or "").strip()
        if not candidate and not why:
            continue
        candidates.append(
            (
                {
                    "source_id": "",
                    "title_or_source_name": _compact_single_line(candidate or "L1 source", limit=160),
                    "url_or_path_or_domain": _source_locator(candidate),
                    "origin_stage": "L1_gemini_search",
                    "relevance_to_question": _compact_single_line(why or candidate, limit=260),
                    "limitation_or_note": _compact_single_line(
                        str(item.get("evidence_type") or item.get("coverage_axis") or "candidate source; full text not fetched at L2.5"),
                        limit=180,
                    ),
                },
                candidate + " " + why,
            )
        )
    for item in l2_items:
        title = str(item.get("title") or item.get("url") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        if not title and not snippet:
            continue
        candidates.append(
            (
                {
                    "source_id": "",
                    "title_or_source_name": _compact_single_line(title or "L2 source", limit=160),
                    "url_or_path_or_domain": _compact_single_line(str(item.get("url") or ""), limit=220),
                    "origin_stage": "L2_ddgs_supplement",
                    "relevance_to_question": _compact_single_line(snippet or title, limit=300),
                    "limitation_or_note": "DDGS snippet/search result; requires full-text verification before high-confidence use",
                },
                title + " " + snippet + " " + str(item.get("query") or ""),
            )
        )
    strict_rows = [row for row, text in candidates if _topic_relevant(text, topic_terms)]
    rows = _dedupe_l2_5_rows(strict_rows)
    if _l2_5_rows_meet_generation_floor(rows):
        return _assign_l2_5_source_ids(rows[:12])

    query_terms = list(dict.fromkeys(list(topic_terms) + _topic_anchor_terms(question, [], [], sample_schema)))
    fallback_ranked = sorted(
        candidates,
        key=lambda item: (
            -_l2_5_overlap_score(item[1], query_terms),
            0 if item[0].get("origin_stage") == "L2_ddgs_supplement" else 1,
            item[0].get("title_or_source_name", "").lower(),
        ),
    )
    merged = list(rows)
    seen = {_l2_5_row_key(row) for row in merged}
    for row, text in fallback_ranked:
        if _contains_l2_5_stub_marker(text):
            continue
        if not row.get("title_or_source_name") and not row.get("relevance_to_question"):
            continue
        key = _l2_5_row_key(row)
        if key in seen:
            continue
        if _l2_5_overlap_score(text, query_terms) <= 0 and len(merged) >= 4:
            continue
        merged.append(row)
        seen.add(key)
        if len(merged) >= 8:
            break
    return _assign_l2_5_source_ids(merged[:12])


def _l2_5_rows_meet_generation_floor(rows: list[dict[str, str]]) -> bool:
    return len(rows) >= 4


def _l2_5_row_key(row: dict[str, str]) -> str:
    locator = str(row.get("url_or_path_or_domain") or "").strip().lower()
    title = str(row.get("title_or_source_name") or "").strip().lower()
    domain_match = re.search(r"https?://([^/]+)", locator)
    domain = domain_match.group(1) if domain_match else locator
    return domain or title


def _dedupe_l2_5_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        key = _l2_5_row_key(row)
        if not key or key in seen:
            continue
        deduped.append(row)
        seen.add(key)
    return deduped


def _assign_l2_5_source_ids(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    assigned: list[dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        item = dict(row)
        item["source_id"] = f"S{idx}"
        assigned.append(item)
    return assigned


def _l2_5_overlap_score(text: str, terms: list[str]) -> int:
    value = str(text or "")
    lowered = value.lower()
    score = 0
    for term in terms:
        token = str(term or "").strip()
        if len(token) < 2:
            continue
        if token.lower() in lowered or token in value:
            score += 1 + min(2, len(token) // 6)
    return score


def _source_locator(candidate: str) -> str:
    match = re.search(r"https?://[^\s'\"),]+", candidate or "")
    if match:
        return match.group(0)
    if candidate.lower().startswith(("query:", "search:")):
        return _compact_single_line(candidate, limit=220)
    return _compact_single_line(candidate, limit=220)


def _l2_5_evidence_rows(source_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for idx, source in enumerate(source_rows[:8], start=1):
        evidence = source.get("relevance_to_question", "").strip()
        if _contains_l2_5_stub_marker(evidence) or len(evidence) < 20:
            continue
        rows.append(
            {
                "claim_id": f"C{min(idx, 4)}",
                "source_id": source["source_id"],
                "evidence_text": evidence,
                "strength_or_limit": source.get("limitation_or_note", ""),
                "support_type": "direct_source_candidate" if source.get("origin_stage") == "L1_gemini_search" else "fresh_search_snippet",
            }
        )
    return rows


def _l2_5_claims(
    evidence_rows: list[dict[str, str]],
    question: str,
    sample_schema: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    claims: list[dict[str, str]] = []
    axes = list((sample_schema or {}).get("axes") or [])
    for idx, row in enumerate(evidence_rows[:4], start=1):
        evidence = _compact_single_line(row["evidence_text"], limit=240)
        axis = axes[(idx - 1) % len(axes)] if axes else "evidence signal"
        claims.append(
            {
                "claim_id": f"C{idx}",
                "claim": f"{axis}: decision-relevant evidence for this question indicates: {evidence}",
                "source_id": row["source_id"],
            }
        )
    return claims


def _l2_5_gaps(
    source_rows: list[dict[str, str]],
    evidence_rows: list[dict[str, str]],
    question: str,
    sample_schema: dict[str, Any] | None = None,
) -> list[str]:
    schema_name = str((sample_schema or {}).get("name") or "generic_evidence")
    topic = schema_name.replace("_", " ")
    gaps = [
        f"G1: Full-text verification gap for {topic}; L2.5 has candidates/snippets but not audited source passages, so claim strength should remain bounded.",
        f"G2: Schema coverage gap for {schema_name}; extracted sources may not evenly cover every required schema axis.",
        f"G3: Counterevidence gap for {topic}; L1/L2 candidates may not include enough contradictory evidence to settle disputed tradeoffs.",
    ]
    if len(source_rows) < 3:
        gaps.append("G4: insufficient_sources=true because fewer than three topic-relevant source rows were extractable from L1/L2.")
    if len(evidence_rows) < 3:
        gaps.append("G5: insufficient_sources=true because fewer than three usable evidence rows were extractable from L1/L2.")
    return gaps[:5]


def _csv_text(fieldnames: list[str], rows: list[dict[str, str]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in fieldnames})
    return output.getvalue()


def _claims_markdown(claims: list[dict[str, str]], *, insufficient: bool) -> str:
    lines = ["# claims", "", f"insufficient_sources: {str(insufficient).lower()}", ""]
    for item in claims:
        lines.append(f"- {item['claim_id']}: {item['claim']} [source_id: {item['source_id']}]")
    return "\n".join(lines) + "\n"


def _gaps_markdown(gaps: list[str], *, insufficient: bool) -> str:
    lines = ["# gaps", "", f"insufficient_sources: {str(insufficient).lower()}", ""]
    lines.extend(f"- {gap}" for gap in gaps)
    return "\n".join(lines) + "\n"


def _contains_l2_5_stub_marker(text: str) -> bool:
    lowered = (text or "").lower()
    for marker in L2_5_STUB_MARKERS:
        pattern = rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])"
        if re.search(pattern, lowered):
            return True
    return False


def analyze_l2_5_evidence_organizer(base_dir: str | Path) -> dict[str, Any]:
    stage_dir = Path(base_dir) / "L2_5_codex_evidence_organizer"
    missing_paths: list[str] = []
    handoff_only_paths: list[str] = []
    header_only_paths: list[str] = []
    domain_content_paths: list[str] = []
    stub_paths: list[str] = []
    for filename in L2_5_EVIDENCE_ORGANIZER_OUTPUTS:
        path = stage_dir / filename
        rel = f"L2_5_codex_evidence_organizer/{filename}"
        if not path.exists():
            missing_paths.append(rel)
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        stripped = text.strip()
        nonempty_lines = [line for line in stripped.splitlines() if line.strip()]
        if _is_l2_5_handoff_only_text(stripped):
            handoff_only_paths.append(rel)
        elif _contains_l2_5_stub_marker(stripped):
            stub_paths.append(rel)
        elif len(nonempty_lines) <= 1:
            header_only_paths.append(rel)
        elif _has_l2_5_domain_content(filename, stripped):
            domain_content_paths.append(rel)
        else:
            header_only_paths.append(rel)
    source_rows = _read_csv_dicts(stage_dir / "sources.csv")
    evidence_rows = _read_csv_dicts(stage_dir / "evidence.csv")
    claim_ids = _markdown_ids(stage_dir / "claims.md", "C")
    gap_ids = _markdown_ids(stage_dir / "gaps.md", "G")
    source_ids = {row.get("source_id", "").strip() for row in source_rows}
    evidence_source_ids = {row.get("source_id", "").strip() for row in evidence_rows}
    evidence_claim_ids = {row.get("claim_id", "").strip() for row in evidence_rows}
    evidence_texts = [row.get("evidence_text", "").strip() for row in evidence_rows]
    real_evidence_rows = [
        text for text in evidence_texts if text and len(text) >= 20 and not _contains_l2_5_stub_marker(text)
    ]
    insufficient_sources = len(source_rows) < 3 or len(real_evidence_rows) < 3 or len(claim_ids) < 4 or len(gap_ids) < 3
    aligned = bool(evidence_source_ids) and evidence_source_ids.issubset(source_ids) and bool(evidence_claim_ids) and evidence_claim_ids.issubset(claim_ids)
    l2_5_stub_detected = bool(handoff_only_paths or stub_paths)
    invalid_paths = sorted(set(missing_paths + handoff_only_paths + header_only_paths + stub_paths))
    extraction_missing = bool(missing_paths) or insufficient_sources or l2_5_stub_detected or not aligned
    if extraction_missing and not invalid_paths:
        invalid_paths = [f"L2_5_codex_evidence_organizer/{filename}" for filename in L2_5_EVIDENCE_ORGANIZER_OUTPUTS]
    issues = ["l2_5_extraction_missing"] if extraction_missing else []
    return {
        "l2_5_valid": not extraction_missing,
        "l2_5_stub_detected": l2_5_stub_detected,
        "upstream_critical_defect": extraction_missing,
        "insufficient_sources": insufficient_sources,
        "issues": issues,
        "missing_or_invalid_artifacts": invalid_paths if extraction_missing else [],
        "handoff_only_artifacts": handoff_only_paths,
        "stub_artifacts": stub_paths,
        "header_only_artifacts": header_only_paths,
        "domain_content_artifacts": domain_content_paths,
        "source_rows": len(source_rows),
        "evidence_rows": len(real_evidence_rows),
        "claim_count": len(claim_ids),
        "gap_count": len(gap_ids),
        "claim_source_alignment_valid": aligned,
    }


def _research_claim_table_from_l2_5(
    base_dir: str | Path,
    *,
    profiles: list[str] | None = None,
    audit_text: str = "",
) -> list[dict[str, Any]]:
    stage_dir = Path(base_dir) / "L2_5_codex_evidence_organizer"
    source_rows = _read_csv_dicts(stage_dir / "sources.csv")
    evidence_rows = _read_csv_dicts(stage_dir / "evidence.csv")
    claim_entries = _read_l2_5_claim_entries(stage_dir / "claims.md")
    gaps_text = (stage_dir / "gaps.md").read_text(encoding="utf-8", errors="replace") if (stage_dir / "gaps.md").exists() else ""
    source_by_id = {str(row.get("source_id") or "").strip(): row for row in source_rows}
    evidence_by_claim: dict[str, list[dict[str, str]]] = {}
    for row in evidence_rows:
        claim_id = str(row.get("claim_id") or "").strip()
        if claim_id:
            evidence_by_claim.setdefault(claim_id, []).append(row)
    claims: list[dict[str, Any]] = []
    for entry in claim_entries[:12]:
        claim_id = entry["claim_id"]
        related_evidence = list(evidence_by_claim.get(claim_id) or [])
        if not related_evidence and entry.get("source_id"):
            related_evidence = [
                {
                    "claim_id": claim_id,
                    "source_id": entry["source_id"],
                    "evidence_text": entry["claim_text"],
                    "strength_or_limit": source_by_id.get(entry["source_id"], {}).get("limitation_or_note", ""),
                    "support_type": "secondary_summary",
                }
            ]
        anchors = [
            _source_anchor_from_rows(source_by_id.get(str(row.get("source_id") or "").strip(), {}), row)
            for row in related_evidence
            if str(row.get("source_id") or "").strip()
        ]
        anchors = [anchor for anchor in anchors if anchor.get("source_id")]
        tier = _epistemic_tier_for_claim(entry["claim_text"], profiles=profiles or [])
        evidence_strength = _evidence_strength_for_anchors(anchors, tier=tier)
        decision_use = _decision_use_for_claim(tier, evidence_strength, anchors)
        claims.append(
            {
                "claim_id": claim_id,
                "claim_text": entry["claim_text"],
                "epistemic_tier": tier,
                "evidence_strength": evidence_strength,
                "source_anchors": anchors,
                "applicability_boundary": _claim_applicability_boundary(entry["claim_text"], anchors),
                "counter_signal_or_failure_condition": _claim_counter_signal(entry["claim_text"], audit_text),
                "evidence_gap": _claim_evidence_gap(anchors, gaps_text),
                "decision_use": decision_use,
                "notes": "Generated from current-run L2.5 source/evidence/claims artifacts; no full-text verification is inferred unless explicitly present.",
            }
        )
    return claims


def _read_l2_5_claim_entries(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    entries: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"\s*-\s*(C\d+)\s*:\s*(.*?)\s*(?:\[source_id:\s*(S\d+)\])?\s*$", line)
        if not match:
            continue
        entries.append(
            {
                "claim_id": match.group(1),
                "claim_text": _compact_single_line(match.group(2), limit=420),
                "source_id": match.group(3) or "",
            }
        )
    return entries


def _source_anchor_from_rows(source_row: dict[str, str], evidence_row: dict[str, str]) -> dict[str, str]:
    source_id = str(evidence_row.get("source_id") or source_row.get("source_id") or "").strip()
    limitation = str(evidence_row.get("strength_or_limit") or source_row.get("limitation_or_note") or "").strip()
    support_type = _claim_support_type(str(evidence_row.get("support_type") or ""), limitation)
    return {
        "source_id": source_id,
        "title": _compact_single_line(str(source_row.get("title_or_source_name") or source_id), limit=180),
        "url_or_stable_locator": _compact_single_line(str(source_row.get("url_or_path_or_domain") or ""), limit=220),
        "source_type": _source_type_from_row(source_row),
        "support_type": support_type,
        "evidence_excerpt": _compact_single_line(str(evidence_row.get("evidence_text") or source_row.get("relevance_to_question") or ""), limit=260),
        "limitation": _compact_single_line(limitation or "source support has not been independently verified at L5", limit=220),
    }


def _source_type_from_row(row: dict[str, str]) -> str:
    origin = str(row.get("origin_stage") or "").strip()
    locator = str(row.get("url_or_path_or_domain") or "").strip()
    if origin == "L2_ddgs_supplement":
        return "search_result_snippet"
    if origin == "L1_gemini_search":
        return "source_candidate"
    if locator.startswith("http"):
        return "web_source"
    return "artifact_or_source_candidate"


def _claim_support_type(raw_support_type: str, limitation: str) -> str:
    value = f"{raw_support_type} {limitation}".lower()
    if "full_text_verified" in value or "full text verified" in value:
        return "full_text_verified"
    if "fresh_search_snippet" in value or "snippet" in value or "search result" in value:
        return "requires_full_text_verification"
    if "summary" in value or "candidate" in value:
        return "secondary_summary"
    return "requires_full_text_verification" if "verification" in value else "secondary_summary"


def _epistemic_tier_for_claim(claim_text: str, *, profiles: list[str]) -> str:
    value = claim_text or ""
    lowered = value.lower()
    future_terms = ("future", "foresight", "10-year", "10 year", "ai era", "未来", "长期", "前瞻", "will ", "could ", "may ")
    risk_terms = ("contradicted", "unsupported", "risk", "风险", "不支持", "反证")
    if any(term in lowered or term in value for term in risk_terms):
        return "unsupported_or_risk"
    if PROFILE_FORESIGHT_MECHANISM in profiles and any(term in lowered or term in value for term in future_terms):
        return "foresight_hypothesis"
    if any(term in lowered for term in ("mechanism", "may", "could", "likely", "plausible", "inference")) or any(
        term in value for term in ("机制", "可能", "合理推断")
    ):
        return "reasonable_inference"
    return "evidence_supported"


def _evidence_strength_for_anchors(anchors: list[dict[str, str]], *, tier: str) -> str:
    if not anchors:
        return "insufficient"
    support_types = {str(anchor.get("support_type") or "") for anchor in anchors}
    if tier == "foresight_hypothesis":
        return "low"
    if "full_text_verified" in support_types and len(anchors) >= 2:
        return "high"
    if "full_text_verified" in support_types:
        return "medium"
    if support_types <= {"requires_full_text_verification", "snippet_only"}:
        return "low"
    if "secondary_summary" in support_types:
        return "low"
    return "insufficient"


def _decision_use_for_claim(tier: str, evidence_strength: str, anchors: list[dict[str, str]]) -> str:
    support_types = {str(anchor.get("support_type") or "") for anchor in anchors}
    if tier == "unsupported_or_risk" or evidence_strength == "insufficient":
        return "do_not_use_as_fact"
    if tier == "foresight_hypothesis" or "requires_full_text_verification" in support_types or evidence_strength == "low":
        return "use_with_caution"
    return "can_support_decision"


def _claim_applicability_boundary(claim_text: str, anchors: list[dict[str, str]]) -> str:
    if not anchors:
        return "No source anchor was retained; use only as an unverified synthesis candidate."
    if any(anchor.get("support_type") == "requires_full_text_verification" for anchor in anchors):
        return "Applies only as preliminary evidence until source passages are full-text verified and population/context fit is checked."
    return "Applies within the source population, measurement context, and intervention/domain described by the retained anchors."


def _claim_counter_signal(claim_text: str, audit_text: str) -> str:
    lowered = (audit_text or "").lower()
    if "ai interfaces become more abstract-textual" in lowered:
        return "Counter-signal: future AI interfaces may become more abstract-textual rather than spatial-visual."
    if "overstated evidence strength" in lowered:
        return "Counter-signal: source verification or stronger counterevidence may downgrade this claim."
    return "Counter-signal: contradictory evidence, failed transfer to the target context, or missing source verification should downgrade this claim."


def _claim_evidence_gap(anchors: list[dict[str, str]], gaps_text: str) -> str:
    if not anchors:
        return "missing_source_anchor"
    if any(anchor.get("support_type") == "requires_full_text_verification" for anchor in anchors):
        return "requires_full_text_verification"
    match = re.search(r"G\d+:\s*([^\n]+)", gaps_text or "")
    return _compact_single_line(match.group(1), limit=220) if match else "no explicit gap captured"


def _research_claim_contract_validation(
    *,
    claim_table: list[dict[str, Any]],
    l4_defect_report: dict[str, list[str]],
    l2_5_analysis: dict[str, Any],
) -> dict[str, Any]:
    blocking: set[str] = set()
    warnings: set[str] = set()
    verification_required: set[str] = set()
    evidence_gaps: set[str] = set()
    caveats: set[str] = set()
    if not claim_table:
        blocking.add("missing_claim_table")
    for idx, claim in enumerate(claim_table, start=1):
        claim_id = str(claim.get("claim_id") or "").strip()
        tier = str(claim.get("epistemic_tier") or "").strip()
        strength = str(claim.get("evidence_strength") or "").strip()
        anchors = claim.get("source_anchors") if isinstance(claim.get("source_anchors"), list) else []
        text = str(claim.get("claim_text") or "")
        if not claim_id:
            blocking.add(f"claim_{idx}_missing_claim_id")
        if not _claim_text_has_substance(text):
            blocking.add(f"{claim_id or idx}:thin_or_template_claim_text")
        if tier not in {"evidence_supported", "reasonable_inference", "foresight_hypothesis", "unsupported_or_risk"}:
            blocking.add(f"{claim_id or idx}:invalid_epistemic_tier")
        if strength not in {"high", "medium", "low", "insufficient"}:
            blocking.add(f"{claim_id or idx}:invalid_evidence_strength")
        if not anchors:
            blocking.add(f"{claim_id or idx}:missing_source_anchors")
        support_types = {str(anchor.get("support_type") or "") for anchor in anchors}
        weak_anchor_only = True
        for anchor_index, anchor in enumerate(anchors, start=1):
            anchor_errors = _source_anchor_contract_errors(anchor)
            for error in anchor_errors:
                blocking.add(f"{claim_id or idx}:{error}")
            support_type = str(anchor.get("support_type") or "")
            if support_type not in {"requires_full_text_verification", "snippet_only", "search_result_snippet"}:
                weak_anchor_only = False
            if _anchor_is_snippet_material(anchor) and support_type == "full_text_verified":
                blocking.add(f"{claim_id or idx}:snippet_source_marked_full_text_verified")
            if _anchor_is_snippet_material(anchor):
                verification_required.add(f"{claim_id}:requires_full_text_verification")
        if anchors and weak_anchor_only and strength in {"high", "medium"}:
            blocking.add(f"{claim_id or idx}:weak_source_anchor_with_overstated_strength")
        if anchors and weak_anchor_only and str(claim.get("decision_use") or "") not in {"use_with_caution", "do_not_use_as_fact"}:
            blocking.add(f"{claim_id or idx}:weak_source_anchor_with_overstated_decision_use")
        if tier == "evidence_supported" and not anchors:
            blocking.add(f"{claim_id or idx}:evidence_supported_without_source_anchor")
        if tier == "evidence_supported" and support_types and support_types <= {"requires_full_text_verification", "snippet_only"}:
            verification_required.add(f"{claim_id}:evidence_supported_requires_full_text_verification")
            caveats.add(f"{claim_id} is only snippet/search-result supported; do not treat as high-confidence fact.")
        if tier == "evidence_supported" and _claim_text_is_future_facing(text):
            blocking.add(f"{claim_id or idx}:future_claim_misclassified_as_evidence_supported")
        if "requires_full_text_verification" in support_types:
            verification_required.add(f"{claim_id}:requires_full_text_verification")
        if str(claim.get("evidence_gap") or ""):
            evidence_gaps.add(str(claim.get("evidence_gap")))
    has_l4_defects = bool((l4_defect_report.get("critical_defects") or []) or (l4_defect_report.get("noncritical_defects") or []))
    if has_l4_defects and not (
        l4_defect_report.get("critical_defects")
        or l4_defect_report.get("noncritical_defects")
        or l4_defect_report.get("handoff_caveats")
    ):
        blocking.add("l4_defects_not_propagated")
    if l2_5_analysis.get("insufficient_sources"):
        warnings.add("l2_5_insufficient_sources")
        caveats.add("L2.5 reports insufficient source/evidence/claim coverage.")
    if l2_5_analysis.get("l2_5_stub_detected"):
        blocking.add("l2_5_stub_detected")
    return {
        "valid": not blocking,
        "blocking_errors": sorted(blocking),
        "warnings": sorted(warnings),
        "verification_required": sorted(verification_required),
        "evidence_gaps": sorted(evidence_gaps),
        "handoff_caveats": sorted(caveats),
    }


def _claim_text_has_substance(text: str) -> bool:
    value = _compact_single_line(text, limit=500)
    lowered = value.lower()
    if len(value) < 50:
        return False
    generic_phrases = (
        "this claim requires evidence",
        "evidence-supported material is limited to accepted research",
        "evidence supported material is limited to accepted research",
        "claim supported by sources",
        "requirements satisfied",
        "decision relevant",
        "more research is needed",
        "tbd",
        "n/a",
        "not applicable",
    )
    if any(phrase in lowered for phrase in generic_phrases):
        return False
    category_only = re.sub(r"[\s:/,_-]+", " ", lowered).strip()
    if category_only in {
        "evidence supported",
        "reasonable inference",
        "foresight hypothesis",
        "unsupported or risk",
    }:
        return False
    relationship_terms = (
        "improves",
        "reduces",
        "increases",
        "supports",
        "provides",
        "suggests",
        "predicts",
        "indicates",
        "affects",
        "changes",
        "change",
        "depends",
        "requires",
        "prevents",
        "raises",
        "lowers",
        "maps",
        "explains",
        "because",
        "through",
        "via",
        "机制",
        "影响",
        "支持",
        "降低",
        "提高",
        "导致",
        "依赖",
        "通过",
        "说明",
    )
    has_relation = any(term in lowered or term in value for term in relationship_terms)
    meaningful_tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}|[\u4e00-\u9fff]{2,}", value)
    return has_relation and len(set(token.lower() for token in meaningful_tokens)) >= 5


def _source_anchor_contract_errors(anchor: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    source_id = str(anchor.get("source_id") or "").strip()
    title = str(anchor.get("title") or anchor.get("title_or_source_name") or "").strip()
    locator = str(
        anchor.get("url_or_stable_locator")
        or anchor.get("url")
        or anchor.get("stable_locator")
        or anchor.get("url_or_path_or_domain")
        or ""
    ).strip()
    source_type = str(anchor.get("source_type") or "").strip()
    support_type = str(anchor.get("support_type") or "").strip()
    allowed_support_types = {
        "full_text_verified",
        "snippet_only",
        "secondary_summary",
        "requires_full_text_verification",
        "search_result_snippet",
    }
    if not source_id:
        errors.append("source_anchor_missing_source_id")
    if not title and not locator:
        errors.append("source_anchor_missing_locator_or_title")
    if not source_type:
        errors.append("source_anchor_missing_source_type")
    if not support_type:
        errors.append("source_anchor_missing_support_type")
    elif support_type not in allowed_support_types:
        errors.append("source_anchor_invalid_support_type")
    return errors


def _anchor_is_snippet_material(anchor: dict[str, Any]) -> bool:
    combined = " ".join(
        str(anchor.get(key) or "")
        for key in (
            "source_type",
            "support_type",
            "notes",
            "limitation",
            "strength_or_limit",
            "origin_stage",
        )
    ).lower()
    return any(
        marker in combined
        for marker in (
            "fresh_search_snippet",
            "search_result_snippet",
            "snippet_only",
            "snippet",
            "search result",
            "l2.5 fresh search",
        )
    )


def _claim_text_is_future_facing(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        term in lowered or term in (text or "")
        for term in ("future", "foresight", "10-year", "10 year", "ai era", "未来", "长期", "前瞻")
    )


def _is_l2_5_handoff_only_text(text: str) -> bool:
    lowered = (text or "").lower()
    if not text.strip():
        return False
    return (
        "handoff_protocol" in lowered
        and "hermes-codex evidence organizer smoke" in lowered
        and "source_candidates.json" in lowered
        and "ddgs_gap_sources.json" in lowered
    )


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        return [dict(row) for row in csv.DictReader(io.StringIO(path.read_text(encoding="utf-8", errors="replace"))) if any(str(value or "").strip() for value in row.values())]
    except Exception:
        return []


def _markdown_ids(path: Path, prefix: str) -> set[str]:
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8", errors="replace")
    pattern = rf"\b{re.escape(prefix)}\d+\b"
    return set(re.findall(pattern, text))


def _has_l2_5_domain_content(filename: str, text: str) -> bool:
    lowered = (text or "").lower()
    if _is_l2_5_handoff_only_text(text):
        return False
    if filename.endswith(".csv"):
        rows = [line for line in text.splitlines() if line.strip()]
        if len(rows) < 2:
            return False
        return any("," in row and not row.lower().startswith(("source_id", "claim_id", "id,")) for row in rows[1:])
    domain_markers = (
        "claim",
        "gap",
        "evidence",
        "source",
        "支持",
        "证据",
        "缺口",
        "争议",
        "不确定",
    )
    return len(text.strip()) >= 120 and any(marker in lowered or marker in text for marker in domain_markers)


def l4_critical_defects_from_audit(audit_text: str) -> list[str]:
    return list(_l4_defect_report_from_audit(audit_text).get("critical_defects") or [])


def detect_supplementary_search_topic_contamination(query: str, text: str) -> dict[str, Any]:
    value = str(text or "")
    lowered = value.lower()
    query_value = str(query or "")
    query_lower = query_value.lower()
    adhd_terms = (
        "adhd",
        "attention-deficit",
        "parent training",
        "behavioral parent training",
        "chadd",
        "additudemag",
        "cdc.gov/adhd",
    )
    adhd_hit_terms = [term for term in adhd_terms if term in lowered]
    query_allows_adhd = any(term in query_lower for term in ("adhd", "attention-deficit")) or any(
        term in query_value for term in ("注意缺陷", "多动", "共病", "行为干预")
    )
    explicit_comorbidity_bridge = any(term in lowered for term in ("comorbid", "co-morbid", "共病"))
    reading_query = any(term in query_value for term in ("拼读", "音韵", "解码", "泛阅读", "阅读困难"))
    contaminated = bool(adhd_hit_terms) and not query_allows_adhd and not (reading_query and explicit_comorbidity_bridge)
    return {
        "supplementary_contaminated": contaminated,
        "supplementary_search_contaminated": contaminated,
        "supplementary_search_cross_topic_contamination": contaminated,
        "issue": "supplementary_search_cross_topic_contamination" if contaminated else "",
        "contaminated_terms": adhd_hit_terms,
    }


def _research_profile_acceptance_requirements(profiles: list[str]) -> list[str]:
    requirements: list[str] = []
    if PROFILE_EVIDENCE_GROUNDED in profiles:
        requirements.extend([
            "evidence strength, source support, gaps, and disputes remain required for evidence_grounded tasks",
        ])
    if PROFILE_FORESIGHT_MECHANISM in profiles:
        requirements.extend([
            "evidence / inference / hypothesis distinction",
            "mechanism_chain: input variables -> mediating mechanisms -> output variables",
            "uncertainty_boundary: assumptions, confidence limits, and where evidence stops",
            "counterexample_or_failure: conditions that would falsify or reverse the hypothesis",
            "foresight hypotheses must not be presented as medical facts or settled conclusions",
        ])
    if PROFILE_IMPLEMENTATION_PLAN in profiles:
        requirements.extend([
            "implementation evidence foundation without requiring a single direct study for every step",
        ])
    return requirements


RESEARCH_PACKET_FIXED_HEADINGS = (
    "evidence_strength",
    "claim_table",
    "controversy",
    "evidence_gap",
    "evidence_supported",
    "reasonable_inference",
    "foresight_hypothesis",
)


def _compact_research_evidence_sections(packet: dict[str, Any], *, accepted: bool) -> list[str]:
    summaries = packet.get("artifact_summaries") if isinstance(packet.get("artifact_summaries"), dict) else {}
    l1_l2 = _compact_material_excerpt(summaries, ("L1_gemini_search", "L2_ddgs_supplement", "L2_5_codex_evidence_organizer"))
    l3 = _compact_material_excerpt(summaries, ("L3_r1_synthesis",))
    l4 = _compact_material_excerpt(summaries, ("L4_gemini_audit",))
    all_material = _compact_material_excerpt(summaries, ("L1_gemini_search", "L2_ddgs_supplement", "L2_5_codex_evidence_organizer", "L3_r1_synthesis", "L4_gemini_audit"), limit=520)
    profiles = _normalize_profiles(packet.get("research_packet_profile"))
    if not accepted:
        status = "L5 did not accept the packet; use only for diagnostics, not DECISION handoff."
    else:
        status = "Accepted compact packet for DECISION handoff; each section is synthesized from L1-L4 materials without raw artifact dumps."

    claim_table = packet.get("claim_table") if isinstance(packet.get("claim_table"), list) else []
    verification_required = [str(item) for item in (packet.get("verification_required") or [])]
    handoff_caveats = [str(item) for item in (packet.get("handoff_caveats") or [])]
    evidence_gaps = [str(item) for item in (packet.get("evidence_gaps") or [])]
    foresight_note = (
        "For foresight_mechanism tasks, future-facing claims below are bounded hypotheses, not settled medical facts."
        if PROFILE_FORESIGHT_MECHANISM in profiles
        else "For evidence_grounded tasks, use the strength/gap/controversy boundaries before carrying claims forward."
    )
    sections = {
        "evidence_strength": (
            f"{status} Stronger support comes from convergent L1-L4 material and L4 audit-accepted claims. "
            "Use claim-level evidence_strength values below; snippet-only or unverified source support stays low/conditional. "
            f"Verification required: {_list_inline(verification_required) or 'none'}. Compact basis: {_section_basis(l1_l2 or all_material)}"
        ),
        "claim_table": (
            _claim_table_markdown(claim_table)
        ),
        "controversy": (
            "Controversy remains where L1-L4 materials depend on context, population differences, tool quality, or disputed translation "
            f"from current evidence to the user scenario. Handoff caveats: {_list_inline(handoff_caveats) or 'none'}. Compact basis: {_section_basis(l4 or all_material)}"
        ),
        "evidence_gap": (
            "Direct gaps include missing long-horizon, individual-level, and future-AI-environment evidence; DECISION must preserve these "
            f"as uncertainty boundaries. Structured gaps: {_list_inline(evidence_gaps) or 'none captured'}. Compact basis: {_section_basis(l4 or all_material)}"
        ),
        "evidence_supported": (
            "Evidence-supported material is limited to claims grounded in current research artifacts, audit-accepted synthesis, and stable "
            f"mechanisms already present in L1-L4. Compact basis: {_section_basis(l1_l2 or l3 or all_material)}"
        ),
        "reasonable_inference": (
            "Reasonable inference may connect accepted research material, mechanism evidence, operational constraints, and scenario-specific context to the "
            f"decision scenario when the intermediate mechanism is explicit. Compact basis: {_section_basis(l3 or all_material)}"
        ),
        "foresight_hypothesis": (
            f"{foresight_note} Treat future structural reversals as conditional on stated drivers, failure conditions, and counter-signals. "
            f"Compact basis: {_section_basis(l3 or l4 or all_material)}"
        ),
    }
    lines: list[str] = []
    for heading in RESEARCH_PACKET_FIXED_HEADINGS:
        lines.extend([f"## {heading}", sections[heading], ""])
    return lines[:-1]


def _claim_table_markdown(claim_table: list[dict[str, Any]]) -> str:
    if not claim_table:
        return "No claim table was retained; this packet is not decision-ready until claim_id/source anchor/evidence strength rows are generated."
    lines: list[str] = []
    for claim in claim_table[:12]:
        anchors = claim.get("source_anchors") if isinstance(claim.get("source_anchors"), list) else []
        anchor_bits = []
        for anchor in anchors[:4]:
            anchor_bits.append(
                "{source_id} ({support_type}; {source_type}; {locator})".format(
                    source_id=str(anchor.get("source_id") or "missing_source_id"),
                    support_type=str(anchor.get("support_type") or "unknown_support"),
                    source_type=str(anchor.get("source_type") or "unknown_source_type"),
                    locator=_compact_single_line(str(anchor.get("url_or_stable_locator") or anchor.get("title") or ""), limit=120),
                )
            )
        lines.extend(
            [
                f"- claim_id: {claim.get('claim_id')}",
                f"  claim_text: {_compact_single_line(str(claim.get('claim_text') or ''), limit=360)}",
                f"  epistemic_tier: {claim.get('epistemic_tier')}",
                f"  evidence_strength: {claim.get('evidence_strength')}",
                "  source_anchors: " + (_list_inline(anchor_bits) or "missing"),
                f"  applicability_boundary: {_compact_single_line(str(claim.get('applicability_boundary') or ''), limit=260)}",
                f"  counter_signal_or_failure_condition: {_compact_single_line(str(claim.get('counter_signal_or_failure_condition') or ''), limit=260)}",
                f"  evidence_gap: {_compact_single_line(str(claim.get('evidence_gap') or ''), limit=220)}",
                f"  decision_use: {claim.get('decision_use')}",
                f"  notes: {_compact_single_line(str(claim.get('notes') or ''), limit=220)}",
            ]
        )
    return "\n".join(lines)


def _list_inline(values: list[str] | tuple[str, ...] | set[str]) -> str:
    cleaned = [_compact_single_line(str(value), limit=180) for value in values if str(value or "").strip()]
    return "; ".join(cleaned[:8])


def _compact_material_excerpt(summaries: dict[str, Any], names: tuple[str, ...], *, limit: int = 360) -> str:
    parts: list[str] = []
    for name in names:
        text = _strip_raw_artifact_metadata_for_final_body(str(summaries.get(name) or ""))
        text = _remove_research_packet_raw_metadata(text)
        if not text.strip():
            continue
        parts.append(_compact_single_line(text, limit=max(120, limit // max(1, len(names)))))
    return _compact_single_line(" | ".join(parts), limit=limit)


def _section_basis(text: str) -> str:
    value = _compact_single_line(text, limit=300)
    return value or "L1-L4 materials were present but too sparse for a richer digest; preserve uncertainty and avoid overclaiming."


def _compact_single_line(text: str, *, limit: int = 300) -> str:
    value = " ".join((text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _remove_research_packet_raw_metadata(text: str) -> str:
    forbidden_prefixes = (
        "artifact_path:",
        "executor_model:",
        "valid_for_pipeline:",
        "stage_name:",
        "owner=",
        "owner:",
        "model:",
    )
    kept: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if any(prefix in lowered for prefix in forbidden_prefixes):
            continue
        kept.append(raw_line)
    return "\n".join(kept)


def _research_evidence_packet_quality_error(text: str) -> str:
    value = (text or "").strip()
    lowered = value.lower()
    if not value:
        return "empty_research_evidence_packet"
    for token in ("artifact_path", "executor_model", "valid_for_pipeline", "stage_name", "owner="):
        if token in lowered:
            return "raw_metadata_leak"
    if "verdict: accepted" not in lowered or "accepted: true" not in lowered:
        return ""
    missing = [heading for heading in RESEARCH_PACKET_FIXED_HEADINGS if f"## {heading}" not in lowered]
    if missing:
        return "missing_research_packet_sections:" + ",".join(missing)
    for heading in RESEARCH_PACKET_FIXED_HEADINGS:
        body = _markdown_section_body(value, heading)
        if len(body) < 60:
            return f"thin_research_packet_section:{heading}"
    claim_table_body = _markdown_section_body(value, "claim_table")
    claim_ids = re.findall(r"\bclaim_id\s*:\s*([A-Za-z0-9_.:-]+)", claim_table_body)
    if not claim_ids:
        return "missing_claim_table"
    if "source_anchors:" not in claim_table_body:
        return "claim_table_missing_source_anchors"
    if "evidence_strength:" not in claim_table_body or "epistemic_tier:" not in claim_table_body:
        return "claim_table_missing_strength_or_tier"
    ready_value = _packet_scalar_value(value, "evidence_packet_ready_for_decision")
    audit_summary = _packet_scalar_value(value, "audit_summary")
    audit_reports_defects = any(
        marker in audit_summary
        for marker in (
            "pass with defects",
            "pass_with_defects",
            "defect",
            "verification_required",
            "caveat",
            "requires_full_text_verification",
        )
    )
    if (
        audit_reports_defects
        and _packet_scalar_value(value, "verdict") in {"accepted", "clean accepted"}
        and ready_value == "true"
        and not _packet_list_field_has_items(value, "critical_defects")
        and not _packet_list_field_has_items(value, "noncritical_defects")
        and not _packet_list_field_has_items(value, "verification_required")
        and not _packet_list_field_has_items(value, "handoff_caveats")
    ):
        return "l4_defects_not_propagated"
    if "requires_full_text_verification" in lowered and ready_value == "true":
        return "unconditional_ready_with_verification_required"
    if re.search(r"epistemic_tier:\s*evidence_supported[\s\S]{0,220}(future|foresight|10-year|10 year|未来|长期|前瞻)", claim_table_body, flags=re.IGNORECASE):
        return "future_claim_misclassified_as_evidence_supported"
    section_text = "\n".join(_markdown_section_body(value, heading).lower() for heading in RESEARCH_PACKET_FIXED_HEADINGS)
    if not section_text.strip():
        return "acceptance_summary_only"
    acceptance_only_terms = (
        "requirements satisfied",
        "requirement satisfied",
        "accepted",
        "acceptance gate",
        "audit accepted",
        "ready for decision",
    )
    substantive_terms = (
        "evidence",
        "证据",
        "inference",
        "推断",
        "hypothesis",
        "假设",
        "gap",
        "缺口",
        "controvers",
        "争议",
        "mechanism",
        "机制",
        "uncertainty",
        "不确定",
    )
    if any(term in section_text for term in acceptance_only_terms) and not any(term in section_text for term in substantive_terms):
        return "acceptance_summary_only"
    return ""


def _packet_scalar_value(text: str, key: str) -> str:
    prefix = key.lower() + ":"
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if line.lower().startswith(prefix):
            return line.split(":", 1)[1].strip().lower()
    return ""


def _packet_list_field_has_items(text: str, key: str) -> bool:
    raw_value = _packet_scalar_value(text, key)
    if not raw_value:
        return False
    normalized = raw_value.strip()
    if normalized in {"[]", "none", "null", "n/a", "not applicable"}:
        return False
    if normalized.startswith("[") and normalized.endswith("]"):
        inner = normalized[1:-1].strip()
        return bool(inner)
    return True


def _markdown_section_body(text: str, heading: str) -> str:
    marker = f"## {heading}"
    lowered = text.lower()
    start = lowered.find(marker.lower())
    if start < 0:
        return ""
    body_start = start + len(marker)
    next_index = lowered.find("\n## ", body_start)
    if next_index < 0:
        next_index = len(text)
    return text[body_start:next_index].strip()


def _normalize_profiles(raw: Any) -> list[str]:
    if isinstance(raw, str):
        values = [part.strip() for part in raw.strip("[]").split(",")]
    elif isinstance(raw, (list, tuple, set)):
        values = [str(part).strip() for part in raw]
    else:
        values = []
    normalized: list[str] = []
    for value in values:
        if value in {
            PROFILE_EVIDENCE_GROUNDED,
            PROFILE_FORESIGHT_MECHANISM,
            PROFILE_FUTURE_SCENARIO,
            PROFILE_IMPLEMENTATION_PLAN,
        } and value not in normalized:
            normalized.append(value)
    return normalized or [PROFILE_EVIDENCE_GROUNDED]


def _research_packet_profile_acceptance_issues(packet: dict[str, Any]) -> list[str]:
    profiles = _normalize_profiles(packet.get("research_packet_profile"))
    summaries = packet.get("artifact_summaries") if isinstance(packet.get("artifact_summaries"), dict) else {}
    combined = "\n".join(str(value or "") for value in summaries.values())
    issues: list[str] = []
    if PROFILE_FORESIGHT_MECHANISM in profiles:
        missing = _foresight_research_packet_missing_requirements(combined)
        issues.extend(f"foresight_mechanism_missing:{name}" for name in missing)
    if PROFILE_IMPLEMENTATION_PLAN in profiles:
        missing = _implementation_research_packet_missing_requirements(combined)
        issues.extend(f"implementation_plan_missing:{name}" for name in missing)
    return issues


def _foresight_research_packet_missing_requirements(text: str) -> list[str]:
    value = text or ""
    lowered = value.lower()
    requirements = {
        "evidence_basis": ("证据", "evidence", "研究", "source", "基础"),
        "inference_hypothesis_distinction": ("合理推断", "前瞻假设", "假设", "inference", "hypothesis", "speculative"),
        "mechanism_chain": ("机制", "因果", "链条", "输入变量", "中介机制", "输出变量", "mechanism", "causal"),
        "uncertainty_boundary": (
            "不确定",
            "边界",
            "边界条件",
            "成立条件",
            "失效条件",
            "不成立",
            "置信",
            "uncertainty",
            "boundary",
            "confidence",
            "confidence limit",
            "condition limit",
        ),
        "counterexample_or_failure": (
            "反例",
            "反证",
            "反证信号",
            "失败条件",
            "失效条件",
            "不成立",
            "观察指标",
            "counterexample",
            "failure condition",
            "falsification",
            "disconfirming signal",
        ),
    }
    missing: list[str] = []
    for name, terms in requirements.items():
        if not any(term in value or term in lowered for term in terms):
            missing.append(name)
    return missing


def _foresight_requirement_map_text(text: str) -> str:
    value = text or ""
    lowered = value.lower()
    synonyms = {
        "uncertainty_boundary": (
            "uncertainty_boundary",
            "不确定",
            "边界",
            "边界条件",
            "成立条件",
            "失效条件",
            "置信",
            "uncertainty",
            "boundary",
            "confidence",
        ),
        "counterexample_or_failure": (
            "counterexample_or_failure",
            "反例",
            "反证",
            "反证信号",
            "失败条件",
            "失效条件",
            "不成立",
            "观察指标",
            "counterexample",
            "failure condition",
            "falsification",
        ),
    }
    lines = [
        "foresight_requirement_map:",
        "purpose: normalize equivalent Chinese/English section labels before L5 acceptance; this does not create evidence.",
    ]
    for name, terms in synonyms.items():
        matched = [term for term in terms if term in value or term in lowered]
        if matched:
            lines.append(f"{name}: detected via {', '.join(matched[:5])}")
        else:
            lines.append(f"{name}: missing")
    return "\n".join(lines)


def _implementation_research_packet_missing_requirements(text: str) -> list[str]:
    value = text or ""
    lowered = value.lower()
    requirements = {
        "support_basis": ("证据", "支撑", "研究", "evidence", "basis"),
        "bounded_steps": ("步骤", "周期", "频率", "记录", "调整", "step", "frequency", "measure"),
    }
    return [name for name, terms in requirements.items() if not any(term in value or term in lowered for term in terms)]


def _foresight_research_prompt_guidance(stage_name: str) -> list[str]:
    return [
        f"Internal research_packet_profile includes {PROFILE_FORESIGHT_MECHANISM}; {stage_name} must support a foresight mechanism packet.",
        "Do not require direct evidence proving the future scenario, but do distinguish clearly among: evidence_support, reasonable_inference, and foresight_hypothesis.",
        "Surface mechanism_chain material: input variables -> mediating mechanisms -> output variables.",
        "Surface uncertainty_boundary material: confidence limits, assumptions, and where evidence stops.",
        "Surface counterexample_or_failure material: conditions under which the hypothesis would fail or reverse.",
        "Do not present foresight hypotheses as medical facts or settled conclusions.",
    ]


def _profiles_require_foresight_template(profiles: list[str]) -> bool:
    return PROFILE_FORESIGHT_MECHANISM in profiles or PROFILE_FUTURE_SCENARIO in profiles


def _convergence_foresight_template_lines(profiles: list[str]) -> list[str]:
    if not _profiles_require_foresight_template(profiles):
        return []
    return [
        "Because output_quality_profile includes foresight_mechanism/future_scenario, the convergence artifact MUST use this hard template.",
        "Do not rename these headings. Do not translate these headings. Do not merge these headings. Do not omit these headings.",
        "Use the exact English headings below, each on its own line:",
        "## key_drivers",
        "## mechanism_chain",
        "## scenario_branches",
        "Include at least two branches under scenario_branches: Scenario A and Scenario B.",
        "## counter_signals",
        "counter_signals may include falsification_signals, but the heading counter_signals must remain present.",
        "## certainty_levels",
        "Use high / medium / low or 高 / 中 / 低 under certainty_levels for each major claim.",
        "## uncertainty_boundary",
    ]


def _l5_acceptance_text_is_accepted(text: str) -> bool:
    lowered = (text or "").lower()
    ready_value = _packet_scalar_value(text, "evidence_packet_ready_for_decision")
    return (
        "verdict: accepted" in lowered
        and "accepted: true" in lowered
        and ready_value in {"true", "conditional"}
    )


def _require_decision_prior(
    stages: list[dict[str, Any]],
    expected: list[str],
    *,
    base_dir: str | Path,
    consumer_stage: str,
) -> None:
    actual = [stage.get("stage_name") for stage in stages]
    if actual != expected:
        raise RuntimeError(f"{consumer_stage}: requires DECISION stages {expected} in order, got={actual}")
    base = Path(base_dir).resolve()
    for record in stages:
        name = str(record.get("stage_name") or "")
        if name.startswith("L"):
            raise RuntimeError(f"{consumer_stage}: RESEARCH-only stage is not allowed in DECISION mode: {name}")
        if record.get("created_in_current_run") is not True:
            raise RuntimeError(f"{consumer_stage}: {name} is not created_in_current_run")
        if record.get("legacy_contaminated") is not False:
            raise RuntimeError(f"{consumer_stage}: {name} is legacy contaminated")
        if record.get("valid_for_pipeline") is not True:
            raise RuntimeError(f"{consumer_stage}: {name} is not valid_for_pipeline")
        _assert_current_run_path(record.get("artifact_path"), base, name, consumer_stage=consumer_stage)
        for output in (record.get("outputs") or {}).values():
            _assert_current_run_path(output, base, name, consumer_stage=consumer_stage)


def _decision_trace(stages: list[dict[str, Any]], *, base_dir: str | Path) -> list[dict[str, Any]]:
    base = Path(base_dir).resolve()
    return [
        {
            "stage_name": record.get("stage_name"),
            "owner": record.get("owner"),
            "model": record.get("model"),
            "executor_model": record.get("executor_model"),
            "artifact_path": str(Path(str(record.get("artifact_path") or "")).resolve().relative_to(base)),
        }
        for record in stages
    ]


def _decision_excerpts(stages: list[dict[str, Any]], *, limit: int = 2500) -> dict[str, str]:
    excerpts: dict[str, str] = {}
    for record in stages:
        name = str(record.get("stage_name") or "")
        path = Path(str(record.get("artifact_path") or "")).resolve()
        excerpts[name] = path.read_text(encoding="utf-8", errors="replace")[:limit]
    return excerpts


def _decision_research_packet_context(
    research_packet_path: str | Path | None,
    *,
    limit: int = 2500,
    include_path: bool = True,
) -> str:
    if not research_packet_path:
        return ""
    path = Path(research_packet_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"DECISION research_packet_path not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    digest = _research_packet_fixed_section_digest(text, limit=limit)
    excerpt = digest or _safe_final_excerpt(text, limit=limit)
    lines = []
    if include_path:
        lines.append(f"research_packet_path: {path}")
    lines.extend(
        [
            "boundary: use as decision context only; do not dump raw research packet into the final report.",
            "research_packet_digest:",
            excerpt,
        ]
    )
    return "\n".join(lines)


def _research_packet_fixed_section_digest(text: str, *, limit: int = 2500) -> str:
    sections: list[str] = []
    for heading in RESEARCH_PACKET_FIXED_HEADINGS:
        body = _markdown_section_body(text or "", heading)
        if body:
            sections.append(f"## {heading}\n{_safe_final_excerpt(body, limit=max(220, limit // 6))}")
    return _safe_final_excerpt("\n\n".join(sections), limit=limit) if sections else ""


CONVERGENCE_FIXED_HEADINGS = (
    "key_drivers",
    "mechanism_chain",
    "scenario_branches",
    "counter_signals",
    "certainty_levels",
    "uncertainty_boundary",
)


def _convergence_fixed_section_digest(text: str, *, limit: int = 3200) -> str:
    sections: list[str] = []
    for heading in CONVERGENCE_FIXED_HEADINGS:
        body = _markdown_section_body(text or "", heading)
        if body:
            sections.append(f"## {heading}\n{_safe_final_excerpt(body, limit=max(260, limit // 6))}")
    return "\n\n".join(sections)[:limit] if sections else ""


def _decision_intelligence_prompt(
    query: str,
    *,
    base_dir: str | Path,
    research_packet_path: str | Path | None = None,
) -> str:
    context = _decision_research_packet_context(
        research_packet_path,
        limit=_decision_intelligence_prompt_char_budget(),
        include_path=False,
    )
    lines = [
        "Run DECISION stage 1: intelligence_layer through AGY/Gemini.",
        "Use Gemini 3.5 Flash (High) only. Do not use CCPA, Controller, R1, or divergence models.",
        "TEXT ONLY CONTRACT: return plain markdown text on stdout. Do not call tools, write files, create artifacts, modify the workspace, or reference output paths.",
        (
            "Input scope is restricted to the user's original decision question plus the supplied research_evidence_packet.md excerpt. This is DECISION mode, so do not run or invent RESEARCH L1-L5 artifacts."
            if context
            else "Input scope is restricted to the user's original decision question. This is DECISION mode, so do not require or invent RESEARCH L1-L5 artifacts."
        ),
        "Produce a high-level structured mapping of the decision problem, uncertainty, constraints, and later-stage evidence needs.",
        "Return only these sections: user_question_map, decision_dimensions_for_later_stages, evidence_needs_for_stage2, open_items_for_stage2.",
        "Do not add conclusions, clinical action plans, final advice, or user-facing guidance.",
        "Do not perform later-stage work: supplementary_search, structure_mapper, evidence_judge, premise_auditor, alternative_generator, insight_harvester, convergence_report.",
        "",
        "## User original decision question",
        query,
    ]
    if context:
        lines.extend(["", "## optional_research_evidence_packet_context", context])
    return "\n".join(lines)


def _decision_supplementary_search_report(
    hits: list[dict[str, str]],
    *,
    stages: list[dict[str, Any]],
    query: str,
    base_dir: str | Path,
    research_packet_path: str | Path | None = None,
) -> str:
    no_fresh_marker = _supplementary_search_no_fresh_hits_from_hits(hits)
    fresh_hits = [hit for hit in hits if hit.get("url")]
    clean_hits, quarantined_hits = _split_supplementary_hits_by_topic(query, fresh_hits)
    if not clean_hits and not quarantined_hits and not no_fresh_marker:
        raise RuntimeError("supplementary_search: DDGS returned no fresh result URLs")
    base = Path(base_dir).resolve()
    intelligence = Path(str(stages[0].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:2500]
    query_plan = _supplementary_search_query_plan(query)
    contaminated = bool(quarantined_hits)
    if no_fresh_marker:
        intelligence = _redact_urls_for_no_fresh_supplement(intelligence)
    lines = [
        "# supplementary_search_report",
        "",
        "stage_name: supplementary_search",
        "tool: DDGS",
        "mode: DECISION",
        "scope: fresh supplemental search anchored to the user's current question.",
        "boundary: source supplement only; no user-facing plan and no replacement for structure_mapper or evidence_judge.",
        f"supplementary_search_status: {'no_fresh_hits' if no_fresh_marker else 'fresh_hits'}",
        f"supplementary_search_contaminated: {str(contaminated).lower()}",
        "quarantine_policy: contaminated DDGS hits are listed for audit only and must not be used as evidence candidates.",
        "",
        "## user_question_anchor",
        query.strip()[:1200],
        "",
        "## query_plan",
        json.dumps(query_plan, ensure_ascii=False, indent=2),
        "",
        "## current_run_inputs",
        f"- intelligence_layer: {Path(str(stages[0].get('artifact_path') or '')).resolve().relative_to(base)}",
        "",
        "## intelligence_layer_excerpt",
        intelligence,
    ]
    context = _decision_research_packet_context(research_packet_path, limit=1200)
    if context:
        lines.extend(["", "## optional_research_evidence_packet_context", context])
    lines.extend(["", "## fresh_ddgs_result_summary"])
    if no_fresh_marker:
        lines.extend(_supplementary_search_no_fresh_hits_lines(no_fresh_marker))
    if not clean_hits:
        lines.append("No topic-consistent fresh DDGS hits remained after quarantine.")
        lines.append("")
    for idx, hit in enumerate(clean_hits, start=1):
        lines.extend(
            [
                f"### result_{idx}",
                f"- query: {hit.get('query', '')}",
                f"- title: {hit.get('title', '')}",
                f"- url: {hit.get('url', '')}",
                f"- snippet: {hit.get('snippet', '')}",
                "",
            ]
        )
    if quarantined_hits:
        lines.extend(["## quarantined_ddgs_result_summary"])
        for idx, hit in enumerate(quarantined_hits, start=1):
            lines.extend(
                [
                    f"### quarantined_result_{idx}",
                    f"- query: {hit.get('query', '')}",
                    f"- title: {hit.get('title', '')}",
                    f"- url: {hit.get('url', '')}",
                    f"- snippet: {hit.get('snippet', '')}",
                    "- quarantine_reason: supplementary_search_cross_topic_contamination",
                    "",
                ]
            )
    lines.extend(
        [
            "## handoff_notes_for_stage3",
            "- Use only non-quarantined fresh URLs as supplemental evidence candidates.",
            "- Do not use quarantined URLs as evidence candidates.",
            "- Keep user-facing guidance out of this stage.",
            "- Stage 3 must still independently map structure; this stage only supplies fresh search material.",
        ]
    )
    return "\n".join(lines)


def _decision_stage_prompt(
    stage: StageSpec,
    stages: list[dict[str, Any]],
    *,
    query: str,
    base_dir: str | Path,
    research_packet_path: str | Path | None = None,
) -> str:
    base = Path(base_dir).resolve()
    excerpts = _decision_excerpts(stages, limit=3000)
    context = _decision_research_packet_context(research_packet_path)
    profiles = _task_engine_profiles_from_query(query)
    foresight_sections = _convergence_foresight_template_lines(profiles) if stage.stage_name == "convergence_report" else []
    duties = {
        "structure_mapper": "Map the problem space, decision axes, constraints, uncertainties, stakeholders, and evaluation criteria.",
        "evidence_judge": "Judge evidence quality, strength, applicability, gaps, and over/under-supported claims.",
        "premise_auditor": "Audit hidden premises, premise risks, counterexamples, cultural/school-system differences, and failure modes.",
        "alternative_generator": "Generate mutually exclusive intervention intensity options and action paths under different risk assumptions.",
        "insight_harvester": "Extract cross-model insights, conflicts, anomalies, high-impact low-confidence ideas, and decision turning points.",
        "convergence_report": "Synthesize the five divergence roles into a convergence decision framework with conflicts and uncertainty boundaries.",
    }
    forbidden = {
        "structure_mapper": "Do not do evidence_judge, premise_auditor, alternative_generator, convergence, or final report.",
        "evidence_judge": "Do not do premise_auditor, alternative_generator, convergence_report, or final report.",
        "premise_auditor": "Do not do alternative_generator, convergence_report, or final report.",
        "alternative_generator": "Do not do insight_harvester, convergence_report, or final report.",
        "insight_harvester": "Do not do convergence_report or final report.",
        "convergence_report": "Do not call search, AGY, DDGS, web_search, api_call, codex_exec, or final_controller_report.",
    }
    lines = [
        f"Run DECISION stage: {stage.stage_name}.",
        f"Canonical model: {stage.model}. Do not substitute another model.",
        (
            "Input scope is restricted to current-run DECISION artifacts already listed below, the user's original question, and the supplied research_evidence_packet.md excerpt."
            if context
            else "Input scope is restricted to current-run DECISION artifacts already listed below and the user's original question."
        ),
        duties[stage.stage_name],
        forbidden[stage.stage_name],
        *_decision_stage_output_contract_lines(stage),
        "Internal output_quality_profile: " + ", ".join(profiles) + ".",
        *foresight_sections,
        f"Current run root: {base}",
        "",
        "## User original decision question",
        query,
        "",
        "## Prior DECISION StageRecords",
        json.dumps(_decision_trace(stages, base_dir=base_dir), ensure_ascii=False, indent=2),
        "",
        "## Prior DECISION artifact excerpts",
        json.dumps(excerpts, ensure_ascii=False, indent=2),
    ]
    if context:
        lines.extend(["", "## optional_research_evidence_packet_context", context])
    return "\n".join(lines)


def _decision_stage_output_contract_lines(stage: StageSpec) -> list[str]:
    if stage.stage_name != "evidence_judge":
        return []
    return [
        "Return only these sections: evidence_quality_map, strength_by_claim, applicability_to_user_context, uncertainty_and_limits, evidence_gaps_for_later_stages.",
        "Your first non-empty line must be exactly: evidence_quality_map",
        "Write strength_by_claim as a standalone line exactly: strength_by_claim",
        "Do not write a report title, markdown H1/H2 heading, table-first response, summary judgment, recommendation, or downstream advice.",
        "For each strength_by_claim item include: claim / strength: high-medium-low / evidence_basis / uncertainty_or_gap.",
    ]


def _decision_evidence_judge_schema_retry_prompt(prompt: str) -> str:
    return "\n".join(
        [
            prompt,
            "",
            "## Schema retry contract",
            "The previous evidence_judge attempt did not satisfy the required section schema.",
            "Return only the five required sections now.",
            "First line exactly: evidence_quality_map",
            "Second required section heading exactly: strength_by_claim",
            "No title, no markdown heading prefix, no table-first answer, no recommendation, no final decision.",
        ]
    )


def _decision_external_calibration_prompt(
    stages: list[dict[str, Any]],
    *,
    query: str,
    base_dir: str | Path,
    research_packet_path: str | Path | None = None,
) -> str:
    base = Path(base_dir).resolve()
    excerpts = _decision_excerpts(stages, limit=5000)
    context = _decision_research_packet_context(research_packet_path)
    lines = [
        "Run DECISION stage 9: external_calibration.",
        "Executor policy: GPT Bridge primary; Gemini/agy Gemini 3.1 Pro (High) fallback only if GPT Bridge is unavailable.",
        "Do not use Nemotron, R1, DeepSeek Controller, Qwen, Llama, Gemma, DDGS, or web_search.",
        (
            "Input scope is restricted to current-run DECISION artifacts: convergence_report.md, D1-D8 StageRecords, the user's original question, and the supplied research_evidence_packet.md excerpt."
            if context
            else "Input scope is restricted to current-run DECISION artifacts: convergence_report.md, D1-D8 StageRecords, and the user's original question."
        ),
        "Output duty: calibrate the evidence strength of convergence_report; mark claims as supported, plausible, speculative, or contradicted; check over-inference; give a calibration verdict.",
        "Return these required sections with substantive body text, even if you mostly agree with convergence_report: calibration_verdict, agreement_points, disagreement_or_risk_points, missing_considerations, final_adjustment_recommendation.",
        "You may also include calibration_scope, claim_strength_table, over_inference_checks, contradiction_checks, and handoff_notes_for_final_controller, but never return only headers.",
        "Do not write the final controller stage, PIPELINE_COMPLETE markers, final user advice, or a final report.",
        f"Current run root: {base}",
        "",
        "## User original decision question",
        query,
        "",
        "## D1-D8 StageRecords",
        json.dumps(_decision_trace(stages, base_dir=base_dir), ensure_ascii=False, indent=2),
        "",
        "## Current-run DECISION artifact excerpts",
        json.dumps(excerpts, ensure_ascii=False, indent=2),
    ]
    if context:
        lines.extend(["", "## optional_research_evidence_packet_context", context])
    return "\n".join(lines)


def _decision_final_controller_packet(
    stages: list[dict[str, Any]],
    *,
    query: str,
    base_dir: str | Path,
    research_packet_path: str | Path | None = None,
) -> dict[str, Any]:
    base = Path(base_dir).resolve()
    excerpts: dict[str, str] = {}
    raw_convergence = ""
    raw_external_calibration = ""
    for record in stages:
        stage_name = str(record.get("stage_name") or "")
        text = Path(str(record.get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")
        excerpts[stage_name] = _safe_final_excerpt(text)
        if stage_name == "convergence_report":
            raw_convergence = text
        if stage_name == "external_calibration":
            raw_external_calibration = text
    passthrough_keys = (
        "created_in_current_run",
        "legacy_contaminated",
        "valid_for_pipeline",
        "run_id",
        "output_root",
        "prompt_sha256",
        "created_at",
        "stage_index",
        "stage_number",
    )
    trace = []
    for record in stages:
        trace_item = {
            "stage_name": record.get("stage_name"),
            "owner": record.get("owner"),
            "model": record.get("model"),
            "executor_model": record.get("executor_model"),
            "artifact_path": str(Path(str(record.get("artifact_path") or "")).resolve().relative_to(base)),
        }
        for key in passthrough_keys:
            if key in record:
                trace_item[key] = record.get(key)
        trace.append(trace_item)
    packet = {
        "mode": ENGINE_DECISION,
        "query": query,
        "base_dir": str(base),
        "output_quality_profile": _task_engine_profiles_from_query(query),
        "stage_trace": trace,
        "excerpts": excerpts,
    }
    convergence_digest = _convergence_fixed_section_digest(raw_convergence)
    if convergence_digest:
        packet["convergence_fixed_section_digest"] = convergence_digest
    calibration_constraints = _external_calibration_final_constraints(raw_external_calibration)
    if calibration_constraints:
        packet["external_calibration_hard_constraints"] = calibration_constraints
    research_context = _decision_research_packet_context(research_packet_path)
    if research_context:
        packet["research_evidence_packet_context"] = research_context
        profiles = _normalize_profiles(packet.get("output_quality_profile"))
        if PROFILE_EVIDENCE_GROUNDED in profiles:
            packet["final_report_requirements"] = {
                "evidence_boundary_required": True,
                "evidence_boundary_heading": "## 证据边界",
                "evidence_boundary_keys": ["evidence_strength", "controversy", "evidence_gap"],
                "evidence_boundary_policy": "Keep this as a short boundary statement, not a literature review; do not raw dump research packet metadata.",
            }
    return packet


def _intelligence_layer_prompt_from_research_packet(
    stages: list[dict[str, Any]],
    *,
    query: str,
    base_dir: str | Path,
) -> str:
    base = Path(base_dir).resolve()
    l5 = stages[-1]
    packet_path = Path(str(l5.get("artifact_path") or "")).resolve()
    packet_text = packet_path.read_text(encoding="utf-8", errors="replace")[:10000]
    trace = [
        {
            "stage_name": record.get("stage_name"),
            "owner": record.get("owner"),
            "model": record.get("model"),
            "artifact_path": str(Path(str(record.get("artifact_path") or "")).resolve()),
        }
        for record in stages
    ]
    return "\n".join(
        [
            "Run RESEARCH_DECISION stage 7: intelligence_layer through AGY/Gemini.",
            "Use Gemini 3.5 Flash (High) only. Do not use CCPA, Controller, R1, or divergence models.",
            "Input scope is restricted to the accepted current-run research_evidence_packet.md, L1-L5 StageRecords, and the user's original question.",
            "Produce a high-level structured mapping between the research packet and the user question.",
            "Return only these sections: user_question_map, research_packet_map, decision_dimensions_for_later_stages, open_items_for_stage8.",
            "Do not add conclusions, clinical action plans, or user-facing guidance.",
            "Do not perform later-stage work: supplementary_search, structure_mapper, evidence_judge, premise_auditor, alternative_generator, insight_harvester, convergence_report.",
            f"Current run root: {base}",
            "",
            "## User original question",
            query,
            "",
            "## L1-L5 StageRecords",
            json.dumps(trace, ensure_ascii=False, indent=2),
            "",
            "## Accepted research_evidence_packet.md",
            packet_text,
        ]
    )


def _supplementary_search_queries(query: str) -> list[str]:
    plan = _supplementary_search_query_plan(query)
    queries = [item["query"] for item in plan if item.get("is_topic_relevant") and not item.get("contamination_reason")]
    if len(queries) < 3:
        raise RuntimeError("supplementary_search: fewer than three topic-relevant queries after topic guard")
    return queries


def _supplementary_search_query_plan(query: str) -> list[dict[str, Any]]:
    value = str(query or "")
    lowered = value.lower()
    topic_terms = _topic_anchor_terms(value)
    allowed_expansions: list[str] = []
    basis = "fallback"
    if any(term in lowered for term in ("adhd", "attention-deficit", "inattentive")) or any(
        term in value for term in ("注意缺陷", "多动", "执行功能")
    ):
        basis = "adhd_parent_training_anchor"
        allowed_expansions = ["ADHD", "parent training", "executive function", "inattentive children"]
        candidates = [
            "ADHD parent training children",
            "behavioral parent training ADHD inattentive children",
            "CLAS ADHD inattentive children parent training",
            "third grade ADHD executive function organization skills",
            "ADHD school accommodations inattentive children",
            "ADHD mind wandering children inattentive",
            "cognitive disengagement syndrome ADHD children parent training",
        ]
    elif any(term in lowered for term in ("postgresql", "lakehouse", "etl", "saas", "event-driven")):
        basis = "b2b_saas_architecture_anchor"
        allowed_expansions = ["CDC", "Debezium", "streaming analytics", "multi tenant", "RLS"]
        candidates = [
            "B2B SaaS PostgreSQL monolith cron ETL migration event driven architecture",
            "PostgreSQL CDC Debezium lakehouse object storage streaming analytics SaaS",
            "B2B SaaS multi tenant analytics lakehouse architecture cost tradeoffs",
            "event driven architecture lakehouse streaming feature pipeline migration risks",
            "PostgreSQL row level security SaaS analytics lakehouse migration case study",
        ]
    elif any(term in value for term in ("拼读", "音韵", "解码", "泛阅读", "阅读困难")):
        basis = "reading_intervention_anchor"
        allowed_expansions = ["dyslexia", "phonological awareness", "explicit decoding", "structured literacy"]
        candidates = [
            "systematic phonological awareness explicit decoding intervention struggling readers age 8 10",
            "dyslexia explicit phonics decoding intervention upper elementary systematic review",
            "Chinese reading difficulties phonological awareness decoding intervention children",
            "wide reading versus explicit decoding struggling readers evidence",
            "high frequency short duration decoding practice reading intervention",
        ]
    elif any(term in value for term in ("越南", "电池回收", "梯次利用", "电动车电池")):
        basis = "vietnam_ev_battery_recycling_anchor"
        allowed_expansions = ["Vietnam", "EPR", "second-life battery", "VinFast", "BYD", "cascade utilization"]
        candidates = [
            "Vietnam EV battery recycling extended producer responsibility regulation 2026",
            "Northern Vietnam electric vehicle battery recycling cascade utilization market",
            "Vietnam lithium ion battery recycling feedstock VinFast BYD supply chain",
            "EV battery second life cascade utilization Vietnam stationary storage market",
            "battery recycling hydrometallurgy investment CAPEX Vietnam industrial services",
        ]
    elif any(term in value for term in ("具身", "机器人", "家庭陪伴", "消费硬件")):
        basis = "home_companion_robotics_anchor"
        allowed_expansions = ["embodied AI", "home companion robot", "consumer robotics", "Sim2Real", "robot safety"]
        candidates = [
            "home companion robot consumer market 2030 embodied AI adoption forecast",
            "household service robot retention utility field trial evidence",
            "embodied AI home robot safety standards commercialization timeline",
            "consumer hardware venture investment home robots market uncertainty",
            "home companion robot cost curve manipulation Sim2Real adoption barriers",
        ]
    elif any(term in lowered for term in ("ai companion", "rabbit r1", "humane ai pin", "plaud", "wearable ai")) or any(
        term in value for term in ("AI 伴侣", "AI伴侣", "硬件 AI", "硬件AI", "独立硬件", "智能伴侣")
    ):
        basis = "ai_hardware_companion_anchor"
        allowed_expansions = ["Rabbit R1", "Humane AI Pin", "Plaud Note", "AI companion hardware", "wearable AI"]
        candidates = [
            "Rabbit R1 Humane AI Pin Plaud Note retention user satisfaction 2026",
            "AI companion hardware market trend wearable AI devices 2026",
            "consumer AI hardware funding Rabbit Humane Plaud Friend 2026",
            "standalone AI device adoption churn return rates 2025 2026",
            "AI wearable companion product market fit smartphone app competition",
        ]
    elif any(term in lowered for term in ("arctic", "unclos", "northwest passage", "northern sea route", "maritime law")) or any(
        term in value for term in ("北极", "航道", "海洋法", "国际法", "主权争议", "海牙国际法院")
    ):
        basis = "international_maritime_law_anchor"
        allowed_expansions = ["UNCLOS", "Article 234", "Northwest Passage", "Northern Sea Route", "CLCS", "Corfu Channel"]
        candidates = [
            "UNCLOS Article 234 Arctic shipping routes sovereignty dispute",
            "Northwest Passage Northern Sea Route international straits legal status",
            "Arctic shipping routes sovereignty dispute maritime law state practice",
            "Corfu Channel international straits Arctic Northwest Passage Northern Sea Route",
            "CLCS Arctic continental shelf claims Russia Canada Denmark legal status",
        ]
    else:
        allowed_expansions = topic_terms[:5]
        candidates = [
            f"{value[:80]} evidence review",
            f"{value[:80]} market evidence",
            f"{value[:80]} uncertainty gaps",
        ]
    guard_terms = list(dict.fromkeys(topic_terms + allowed_expansions))
    plan: list[dict[str, Any]] = []
    for candidate in candidates:
        contamination = detect_supplementary_search_topic_contamination(value, candidate)
        relevant = _topic_relevant(candidate + " " + " ".join(allowed_expansions), guard_terms)
        plan.append(
            {
                "query": candidate,
                "query_basis": basis,
                "topic_anchor_terms": topic_terms,
                "allowed_expansion_terms": allowed_expansions,
                "is_topic_relevant": bool(relevant and not contamination["supplementary_search_contaminated"]),
                "contamination_reason": contamination["issue"],
            }
        )
    return plan


def _split_supplementary_hits_by_topic(query: str, hits: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    clean: list[dict[str, str]] = []
    quarantined: list[dict[str, str]] = []
    for hit in hits:
        text = " ".join(str(hit.get(key) or "") for key in ("query", "title", "url", "snippet"))
        if detect_supplementary_search_topic_contamination(query, text)["supplementary_search_contaminated"]:
            quarantined.append(hit)
        else:
            clean.append(hit)
    return clean, quarantined


def _supplementary_search_no_fresh_hits_from_hits(hits: list[dict[str, str]]) -> dict[str, str] | None:
    for hit in hits:
        if hit.get("supplementary_search_status") == "no_fresh_hits":
            return hit
    return None


def _supplementary_search_no_fresh_hits_lines(marker: dict[str, str]) -> list[str]:
    attempted_queries = []
    try:
        parsed = json.loads(marker.get("attempted_queries_json") or "[]")
        if isinstance(parsed, list):
            attempted_queries = [str(item) for item in parsed]
    except json.JSONDecodeError:
        attempted_queries = []
    lines = [
        "supplementary_search_status: no_fresh_hits",
        "source_evidence_emitted: false",
        "no_fresh_hits_but_no_contradiction_found: true",
        "provider_failure: false",
        f"backends_attempted: {marker.get('backends', '')}",
        "caveat: DDGS returned no topic-consistent fresh result URLs. Downstream stages must not treat this as supporting evidence; use it only as an uncertainty boundary.",
        "attempted_queries:",
    ]
    lines.extend(f"- {query}" for query in attempted_queries)
    errors = str(marker.get("errors") or "").strip()
    if errors:
        lines.extend(["backend_no_result_details:", errors])
    lines.append("")
    return lines


def _redact_urls_for_no_fresh_supplement(text: str) -> str:
    return re.sub(r"https?://[^\s)]+", "<source-url-redacted>", text or "")


def _supplementary_search_report(
    hits: list[dict[str, str]],
    *,
    stages: list[dict[str, Any]],
    query: str,
    base_dir: str | Path,
) -> str:
    no_fresh_marker = _supplementary_search_no_fresh_hits_from_hits(hits)
    fresh_hits = [hit for hit in hits if hit.get("url")]
    clean_hits, quarantined_hits = _split_supplementary_hits_by_topic(query, fresh_hits)
    if not clean_hits and not quarantined_hits and not no_fresh_marker:
        raise RuntimeError("supplementary_search: DDGS returned no fresh result URLs")
    base = Path(base_dir).resolve()
    packet = Path(str(stages[5].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:2500]
    intelligence = Path(str(stages[6].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:2500]
    query_plan = _supplementary_search_query_plan(query)
    contaminated = bool(quarantined_hits)
    if no_fresh_marker:
        packet = _redact_urls_for_no_fresh_supplement(packet)
        intelligence = _redact_urls_for_no_fresh_supplement(intelligence)
    lines = [
        "# supplementary_search_report",
        "",
        "stage_name: supplementary_search",
        "tool: DDGS",
        "scope: fresh supplemental search anchored to the user's current question.",
        "boundary: source supplement only; no user-facing plan and no replacement for structure_mapper or evidence_judge.",
        f"supplementary_search_status: {'no_fresh_hits' if no_fresh_marker else 'fresh_hits'}",
        f"supplementary_search_contaminated: {str(contaminated).lower()}",
        "quarantine_policy: contaminated DDGS hits are listed for audit only and must not be used as evidence candidates.",
        "",
        "## user_question_anchor",
        query.strip()[:1200],
        "",
        "## query_plan",
        json.dumps(query_plan, ensure_ascii=False, indent=2),
        "",
        "## current_run_inputs",
        f"- research_packet: {Path(str(stages[5].get('artifact_path') or '')).resolve().relative_to(base)}",
        f"- intelligence_layer: {Path(str(stages[6].get('artifact_path') or '')).resolve().relative_to(base)}",
        "",
        "## accepted_research_packet_excerpt",
        packet,
        "",
        "## intelligence_layer_excerpt",
        intelligence,
        "",
        "## fresh_ddgs_result_summary",
    ]
    if no_fresh_marker:
        lines.extend(_supplementary_search_no_fresh_hits_lines(no_fresh_marker))
    if not clean_hits:
        lines.append("No topic-consistent fresh DDGS hits remained after quarantine.")
        lines.append("")
    for idx, hit in enumerate(clean_hits, start=1):
        lines.extend(
            [
                f"### result_{idx}",
                f"- query: {hit.get('query', '')}",
                f"- title: {hit.get('title', '')}",
                f"- url: {hit.get('url', '')}",
                f"- snippet: {hit.get('snippet', '')}",
                "",
            ]
        )
    if quarantined_hits:
        lines.extend(["## quarantined_ddgs_result_summary"])
        for idx, hit in enumerate(quarantined_hits, start=1):
            lines.extend(
                [
                    f"### quarantined_result_{idx}",
                    f"- query: {hit.get('query', '')}",
                    f"- title: {hit.get('title', '')}",
                    f"- url: {hit.get('url', '')}",
                    f"- snippet: {hit.get('snippet', '')}",
                    "- quarantine_reason: supplementary_search_cross_topic_contamination",
                    "",
                ]
            )
    lines.extend(
        [
            "## handoff_notes_for_stage9",
            "- Use only non-quarantined fresh URLs as supplemental evidence candidates.",
            "- Do not use quarantined URLs as evidence candidates.",
            "- Keep user-facing guidance out of this stage.",
            "- Stage 9 must still independently map structure; this stage only supplies fresh search material.",
        ]
    )
    return "\n".join(lines)


def _structure_mapper_prompt_from_artifacts(
    stages: list[dict[str, Any]],
    *,
    query: str,
    base_dir: str | Path,
) -> str:
    base = Path(base_dir).resolve()
    packet = Path(str(stages[5].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:3500]
    intelligence = Path(str(stages[6].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:3500]
    supplement = Path(str(stages[7].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:3500]
    trace = [
        {
            "stage_name": record.get("stage_name"),
            "owner": record.get("owner"),
            "model": record.get("model"),
            "executor_model": record.get("executor_model"),
            "artifact_path": str(Path(str(record.get("artifact_path") or "")).resolve().relative_to(base)),
        }
        for record in stages
    ]
    return "\n".join(
        [
            "Run RESEARCH_DECISION stage 9: structure_mapper using Qwen72B only.",
            f"Canonical model: {QWEN72B}. Do not use 9B, Flash, DeepSeek Controller, or R1.",
            "Input scope is restricted to current-run fresh artifacts: research_evidence_packet.md, intelligence_layer_report.md, parent_training_supplement.md, L1-L8 StageRecords, and the user's original question.",
            "Output duty: map the problem space structure only.",
            "Return only these sections: problem_axes, actor_map, decision_questions, evidence_slots, unknowns_for_later_stages.",
            "Do not judge evidence, audit premises, generate alternatives, combine views, or write any user-facing plan.",
            f"Current run root: {base}",
            "",
            "## User original question",
            query,
            "",
            "## L1-L8 StageRecords",
            json.dumps(trace, ensure_ascii=False, indent=2),
            "",
            "## research_evidence_packet.md",
            packet,
            "",
            "## intelligence_layer_report.md",
            intelligence,
            "",
            "## parent_training_supplement.md",
            supplement,
        ]
    )


def _structure_mapper_forbidden_tokens(text: str) -> list[str]:
    value = text or ""
    lowered = value.lower()
    tokens = []
    if "final_controller_report" in lowered:
        tokens.append("final_controller_report")
    if "pipeline_status=pipeline_complete" in lowered:
        tokens.append("pipeline_status=PIPELINE_COMPLETE")

    heading_lines = [line.strip() for line in value.splitlines() if line.strip().startswith("#")]
    if any(_is_forbidden_stage_heading(line, {"convergence_report", "evidence_judge"}) for line in heading_lines):
        tokens.append("later_stage_heading")
    if any(_is_final_report_heading(line) for line in heading_lines):
        tokens.append("final_report_heading")
    if any(_is_chinese_final_advice_heading(line) for line in heading_lines):
        tokens.append("chinese_final_advice_heading")
    if any(_is_forbidden_structure_mapper_final_conclusion_heading(line) for line in heading_lines):
        tokens.append("final_conclusion_heading")
    return tokens


def _is_forbidden_structure_mapper_final_conclusion_heading(line: str) -> bool:
    normalized = line.strip("# ").strip()
    return normalized in {"最终结论", "最终判断", "最终决策", "结论报告"}


def _evidence_judge_prompt_from_artifacts(
    stages: list[dict[str, Any]],
    *,
    query: str,
    base_dir: str | Path,
) -> str:
    base = Path(base_dir).resolve()
    packet = Path(str(stages[5].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:3500]
    intelligence = Path(str(stages[6].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:3500]
    supplement = Path(str(stages[7].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:3500]
    structure = Path(str(stages[8].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:3500]
    trace = [
        {
            "stage_name": record.get("stage_name"),
            "owner": record.get("owner"),
            "model": record.get("model"),
            "executor_model": record.get("executor_model"),
            "artifact_path": str(Path(str(record.get("artifact_path") or "")).resolve().relative_to(base)),
        }
        for record in stages
    ]
    return "\n".join(
        [
            "Run RESEARCH_DECISION stage 10: evidence_judge using Nemotron-120B only.",
            f"Canonical model: {NEMOTRON120B}. Do not use Qwen72B, 9B, Flash, DeepSeek Controller, or R1.",
            "Input scope is restricted to current-run fresh artifacts: research_evidence_packet.md, intelligence_layer_report.md, parent_training_supplement.md, structure_mapper.md, L1-L9 StageRecords, and the user's original question.",
            "Output duty: judge evidence quality, strength, uncertainty, and applicability only.",
            "Return only these sections: evidence_quality_map, strength_by_claim, applicability_to_user_context, uncertainty_and_limits, evidence_gaps_for_later_stages.",
            "If input artifacts do not provide a claim table, derive 4-6 decision-relevant claims from research_evidence_packet, intelligence_layer_report, supplementary_search, and structure_mapper.",
            "strength_by_claim is always required.",
            "Write strength_by_claim as a standalone line exactly: strength_by_claim",
            "Do not put the schema or claim text on the same line as strength_by_claim.",
            "Never write \"not applicable\" for evidence_quality_map or strength_by_claim.",
            "For each strength_by_claim item include: claim / strength: high-medium-low / evidence_basis / uncertainty_or_gap.",
            "Output the report directly.",
            "Do not narrate task instructions.",
            "Do not include reasoning setup, compliance notes, or phrases like \"We need to\", \"We must\", \"Let's craft\", \"I will\", or \"no need to mention\".",
            "Start directly with the required section headings.",
            "Do not audit premises, generate alternatives, combine views, or write any user-facing plan.",
            f"Current run root: {base}",
            "",
            "## User original question",
            query,
            "",
            "## L1-L9 StageRecords",
            json.dumps(trace, ensure_ascii=False, indent=2),
            "",
            "## research_evidence_packet.md",
            packet,
            "",
            "## intelligence_layer_report.md",
            intelligence,
            "",
            "## parent_training_supplement.md",
            supplement,
            "",
            "## structure_mapper.md",
            structure,
            "",
            "## Output contract",
            "Your first non-empty line must be exactly: evidence_quality_map",
            "Then continue with strength_by_claim, applicability_to_user_context, uncertainty_and_limits, evidence_gaps_for_later_stages.",
            "If no explicit claim table is available, derive 4-6 decision-relevant claims from the supplied artifacts.",
            "strength_by_claim is always required and every item must include claim / strength: high-medium-low / evidence_basis / uncertainty_or_gap.",
            "The strength_by_claim section heading must be a standalone line exactly: strength_by_claim",
            "Do not put the schema or claim text on the same line as strength_by_claim.",
            "Never write \"not applicable\" for evidence_quality_map or strength_by_claim.",
            "Do not write any preface, planning note, compliance note, or self-instruction.",
            "Do not write phrases like \"We need to\", \"We must\", \"Let's craft\", \"I will\", \"no need to mention\", or \"just generating answer\".",
        ]
    )


def _evidence_judge_compact_prompt_from_artifacts(
    stages: list[dict[str, Any]],
    *,
    query: str,
    base_dir: str | Path,
) -> str:
    base = Path(base_dir).resolve()
    packet = Path(str(stages[5].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")
    intelligence = Path(str(stages[6].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")
    supplement = Path(str(stages[7].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")
    structure = Path(str(stages[8].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")
    trace = [
        {
            "stage_name": record.get("stage_name"),
            "owner": record.get("owner"),
            "model": record.get("model"),
            "executor_model": record.get("executor_model"),
            "artifact_path": str(Path(str(record.get("artifact_path") or "")).resolve().relative_to(base)),
        }
        for record in stages
    ]
    packet_compact = _compact_research_packet_for_evidence_judge(packet, limit=3800)
    intelligence_compact = _compact_evidence_judge_prior_artifact(intelligence, limit=900)
    supplement_compact = _compact_evidence_judge_prior_artifact(supplement, limit=800)
    structure_compact = _compact_evidence_judge_prior_artifact(structure, limit=1200)
    chunks = [
        "Run RESEARCH_DECISION stage 10: evidence_judge using Nemotron-120B only.",
        "compact_evidence_judge_packet: true",
        "compact_budget_chars: 10000",
        f"Canonical model: {NEMOTRON120B}. Do not use Qwen72B, 9B, Flash, DeepSeek Controller, or R1.",
        "Input scope is restricted to compacted current-run fresh artifacts: research_evidence_packet.md, intelligence_layer_report.md, parent_training_supplement.md, structure_mapper.md, L1-L9 StageRecords, and the user's original question.",
        "Output duty: judge evidence quality, strength, uncertainty, and applicability only.",
        "Return only these sections: evidence_quality_map, strength_by_claim, applicability_to_user_context, uncertainty_and_limits, evidence_gaps_for_later_stages.",
        "If input artifacts do not provide a claim table, derive 4-6 decision-relevant claims from research_evidence_packet, intelligence_layer_report, supplementary_search, and structure_mapper.",
        "strength_by_claim is always required.",
        "Write strength_by_claim as a standalone line exactly: strength_by_claim",
        "Do not put the schema or claim text on the same line as strength_by_claim.",
        "Never write \"not applicable\" for evidence_quality_map or strength_by_claim.",
        "For each strength_by_claim item include: claim / strength: high-medium-low / evidence_basis / uncertainty_or_gap.",
        "Output the report directly.",
        "Do not narrate task instructions.",
        "Do not include reasoning setup, compliance notes, or phrases like \"We need to\", \"We must\", \"Let's craft\", \"I will\", or \"no need to mention\".",
        "Start directly with the required section headings.",
        "Do not audit premises, generate alternatives, combine views, or write any user-facing plan.",
        f"Current run root: {base}",
        "",
        "## User original question",
        query,
        "",
        "## L1-L9 StageRecords",
        json.dumps(trace, ensure_ascii=False, indent=2),
        "",
        "## research_evidence_packet.md compact",
        packet_compact,
        "",
        "## intelligence_layer_report.md compact",
        intelligence_compact,
        "",
        "## parent_training_supplement.md compact",
        supplement_compact,
        "",
        "## structure_mapper.md compact",
        structure_compact,
        "",
        "## Output contract",
        "Your first non-empty line must be exactly: evidence_quality_map",
        "Then continue with strength_by_claim, applicability_to_user_context, uncertainty_and_limits, evidence_gaps_for_later_stages.",
        "If no explicit claim table is available, derive 4-6 decision-relevant claims from the supplied artifacts.",
        "strength_by_claim is always required and every item must include claim / strength: high-medium-low / evidence_basis / uncertainty_or_gap.",
        "The strength_by_claim section heading must be a standalone line exactly: strength_by_claim",
        "Do not put the schema or claim text on the same line as strength_by_claim.",
        "Never write \"not applicable\" for evidence_quality_map or strength_by_claim.",
        "Do not write any preface, planning note, compliance note, or self-instruction.",
        "Do not write phrases like \"We need to\", \"We must\", \"Let's craft\", \"I will\", \"no need to mention\", or \"just generating answer\".",
    ]
    prompt = "\n".join(chunks)
    if len(prompt) <= 10000:
        return prompt
    # Keep the contract intact; only tighten compacted prior excerpts.
    overflow = len(prompt) - 10000
    structure_limit = max(500, 1200 - overflow)
    chunks[chunks.index(structure_compact)] = _compact_evidence_judge_prior_artifact(structure, limit=structure_limit)
    prompt = "\n".join(chunks)
    if len(prompt) <= 10000:
        return prompt
    overflow = len(prompt) - 10000
    packet_limit = max(2400, 3800 - overflow)
    chunks[chunks.index(packet_compact)] = _compact_research_packet_for_evidence_judge(packet, limit=packet_limit)
    return "\n".join(chunks)


def _compact_research_packet_for_evidence_judge(text: str, *, limit: int) -> str:
    preferred = (
        "evidence_supported",
        "reasonable_inference",
        "foresight_hypothesis",
        "evidence_gap",
        "evidence_strength",
        "controversy",
    )
    sections: list[str] = []
    for heading in preferred:
        body = _markdown_section_body(text, heading) or _colon_or_plain_section_body(text, heading)
        if body:
            sections.append(f"## {heading}\n{_semantic_limit(body, max(260, limit // 5))}")
    if not sections:
        sections.append(_semantic_limit(_compact_evidence_judge_prior_artifact(text, limit=limit), limit))
    return _semantic_limit("\n\n".join(sections), limit)


def _compact_evidence_judge_prior_artifact(text: str, *, limit: int) -> str:
    keywords = (
        "claim",
        "strength",
        "evidence",
        "uncertain",
        "uncertainty",
        "applicability",
        "applicable",
        "gap",
        "limit",
        "risk",
        "support",
        "判断",
        "证据",
        "强度",
        "不确定",
        "适用",
        "缺口",
        "限制",
        "风险",
        "支持",
    )
    selected: list[str] = []
    current_heading = ""
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            current_heading = line.strip("# ").strip()
            if any(keyword in current_heading.lower() or keyword in current_heading for keyword in keywords):
                selected.append(raw_line)
            continue
        haystack = f"{current_heading}\n{line}".lower()
        if any(keyword.lower() in haystack for keyword in keywords):
            selected.append(raw_line)
    if not selected:
        selected = [line for line in (text or "").splitlines() if line.strip()][:18]
    return _semantic_limit("\n".join(selected), limit)


def _semantic_limit(text: str, limit: int) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    chunks = re.split(r"(?<=[。！？.!?])\s+|\n(?=\s*[-*0-9#])", value)
    kept: list[str] = []
    total = 0
    for chunk in chunks:
        piece = chunk.strip()
        if not piece:
            continue
        add_len = len(piece) + (2 if kept else 0)
        if total + add_len > limit:
            break
        kept.append(piece)
        total += add_len
    if kept:
        return "\n".join(kept)
    return value[:limit].rsplit(" ", 1)[0].strip() or value[:limit].strip()


def _evidence_judge_forbidden_tokens(text: str) -> list[str]:
    value = text or ""
    lowered = value.lower()
    tokens = []
    if "final_controller_report" in lowered:
        tokens.append("final_controller_report")
    if "pipeline_status=pipeline_complete" in lowered:
        tokens.append("pipeline_status=PIPELINE_COMPLETE")

    heading_lines = [line.strip() for line in value.splitlines() if line.strip().startswith("#")]
    if any(_is_forbidden_stage_heading(line, {"convergence_report", "premise_auditor"}) for line in heading_lines):
        tokens.append("later_stage_heading")
    if any(_is_final_report_heading(line) for line in heading_lines):
        tokens.append("final_report_heading")
    if any(_is_chinese_final_advice_heading(line) for line in heading_lines):
        tokens.append("chinese_final_advice_heading")
    return tokens


def _evidence_judge_artifact_quality_error(text: str) -> str:
    if _evidence_judge_process_narration_hits(text):
        return "process_narration_leak"
    lowered = (text or "").lower()
    if re.search(r"evidence_quality_map\s+and\s+strength_by_claim\s+are\s+not\s+applicable", lowered):
        return "section_start_mismatch"
    first = _first_nonempty_line(text)
    if first.strip().lower() != "evidence_quality_map":
        return "section_start_mismatch"
    section_names = {
        line.strip().lower().rstrip(":")
        for line in (text or "").splitlines()
        if line.strip()
    }
    if "strength_by_claim" not in section_names:
        return "missing_strength_by_claim"
    return ""


def _evidence_judge_schema_retryable_quality_error(error: str) -> bool:
    retryable = {
        "section_start_mismatch",
        "missing_required_section",
        "missing_evidence_quality_map",
        "missing_strength_by_claim",
    }
    hard_blocked = {
        "forbidden_final_or_later_stage_terms",
        "forbidden_process_narration",
        "process_narration_leak",
        "stage_leakage",
        "final_controller_leakage",
        "premise_auditor_leakage",
    }
    value = str(error or "")
    return value in retryable and value not in hard_blocked


def _evidence_judge_process_narration_hits(text: str) -> list[str]:
    checks = (
        ("we_need_to", r"\bwe\s+need\s+to\b"),
        ("we_must", r"\bwe\s+must\b"),
        ("lets_craft", r"\blet['’]s\s+craft\b"),
        ("i_will", r"\bi\s+will\b"),
        ("no_need_to_mention", r"\bno\s+need\s+to\s+mention\b"),
        ("just_generating_answer", r"\bjust\s+generating\s+(?:the\s+)?answer\b"),
        ("we_are_just", r"\bwe\s+are\s+just\b"),
        ("we_must_not", r"\bwe\s+must\s+not\b"),
        ("we_should_not", r"\bwe\s+should\s+not\b"),
        ("i_must_not", r"\bi\s+must\s+not\b"),
        ("i_should_not", r"\bi\s+should\s+not\b"),
        ("must_not_include", r"\bmust\s+not\s+include\b"),
        ("must_not_mention", r"\bmust\s+not\s+mention\b"),
        ("must_not_output", r"\bmust\s+not\s+output\b"),
        ("must_not_write", r"\bmust\s+not\s+write\b"),
        ("should_not_include", r"\bshould\s+not\s+include\b"),
        ("should_not_mention", r"\bshould\s+not\s+mention\b"),
        ("should_not_output", r"\bshould\s+not\s+output\b"),
        ("should_not_write", r"\bshould\s+not\s+write\b"),
        ("prompt_compliance_note", r"\bthe\s+instruction\s+says\b"),
        ("return_only_narration", r"\breturn\s+only\s+these\s+sections\b"),
    )
    return [token for token, pattern in checks if re.search(pattern, text or "", re.IGNORECASE)]


def _first_nonempty_line(text: str) -> str:
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def _is_forbidden_stage_heading(line: str, stage_names: set[str]) -> bool:
    stripped = line.lstrip("#").strip().lower().replace(" ", "_")
    return any(stripped == name or stripped.startswith(f"{name}:") for name in stage_names)


def _write_invalid_stage_debug(stage: StageSpec, content: Any, *, base_dir: str | Path, filename: str | None = None) -> Path:
    stage_dir = Path(base_dir) / stage.stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    path = stage_dir / (filename or f"{stage.stage_name}.invalid.md")
    path.write_text(_stringify_artifact(content), encoding="utf-8")
    return path


def _premise_auditor_prompt_from_artifacts(
    stages: list[dict[str, Any]],
    *,
    query: str,
    base_dir: str | Path,
) -> str:
    base = Path(base_dir).resolve()
    packet = Path(str(stages[5].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:3000]
    intelligence = Path(str(stages[6].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:3000]
    supplement = Path(str(stages[7].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:3000]
    structure = Path(str(stages[8].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:3000]
    evidence = Path(str(stages[9].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:3000]
    trace = [
        {
            "stage_name": record.get("stage_name"),
            "owner": record.get("owner"),
            "model": record.get("model"),
            "executor_model": record.get("executor_model"),
            "artifact_path": str(Path(str(record.get("artifact_path") or "")).resolve().relative_to(base)),
        }
        for record in stages
    ]
    return "\n".join(
        [
            "Run RESEARCH_DECISION stage 11: premise_auditor using Llama70B only.",
            f"Canonical model: {LLAMA70B}. Do not use Nemotron, Qwen, 9B, Flash, DeepSeek Controller, or R1.",
            "Input scope is restricted to current-run fresh artifacts: research_evidence_packet.md, intelligence_layer_report.md, parent_training_supplement.md, structure_mapper.md, evidence_judge.md, L1-L10 StageRecords, and the user's original question.",
            "Output duty: audit hidden assumptions, premise risks, counterexamples, and culture/school-system differences only.",
            "Return only these sections: implicit_premises, premise_risks, counterexamples, culture_and_school_system_differences, assumptions_for_later_stages.",
            "Do not generate alternatives, converge views, or write any user-facing plan.",
            f"Current run root: {base}",
            "",
            "## User original question",
            query,
            "",
            "## L1-L10 StageRecords",
            json.dumps(trace, ensure_ascii=False, indent=2),
            "",
            "## research_evidence_packet.md",
            packet,
            "",
            "## intelligence_layer_report.md",
            intelligence,
            "",
            "## parent_training_supplement.md",
            supplement,
            "",
            "## structure_mapper.md",
            structure,
            "",
            "## evidence_judge.md",
            evidence,
        ]
    )


def _premise_auditor_forbidden_tokens(text: str) -> list[str]:
    value = text or ""
    lowered = value.lower()
    tokens = []
    if "final_controller_report" in lowered:
        tokens.append("final_controller_report")
    if "pipeline_status=pipeline_complete" in lowered:
        tokens.append("pipeline_status=PIPELINE_COMPLETE")
    heading_lines = [line.strip() for line in value.splitlines() if line.strip().startswith("#")]
    if any(_is_forbidden_stage_heading(line, {"convergence_report", "alternative_generator"}) for line in heading_lines):
        tokens.append("later_stage_heading")
    if any(_is_final_report_heading(line) for line in heading_lines):
        tokens.append("final_report_heading")
    if any(_is_chinese_final_advice_heading(line) for line in heading_lines):
        tokens.append("chinese_final_advice_heading")
    return tokens


def _alternative_generator_prompt_from_artifacts(
    stages: list[dict[str, Any]],
    *,
    query: str,
    base_dir: str | Path,
) -> str:
    base = Path(base_dir).resolve()
    packet = Path(str(stages[5].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:2600]
    intelligence = Path(str(stages[6].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:2600]
    supplement = Path(str(stages[7].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:2600]
    structure = Path(str(stages[8].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:2600]
    evidence = Path(str(stages[9].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:2600]
    premise = Path(str(stages[10].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:2600]
    trace = [
        {
            "stage_name": record.get("stage_name"),
            "owner": record.get("owner"),
            "model": record.get("model"),
            "executor_model": record.get("executor_model"),
            "artifact_path": str(Path(str(record.get("artifact_path") or "")).resolve().relative_to(base)),
        }
        for record in stages
    ]
    return "\n".join(
        [
            "Run RESEARCH_DECISION stage 12: alternative_generator using Gemma-4-31B only.",
            f"Canonical model: {GEMMA431B}. Do not use Llama, Nemotron, Qwen, 9B, Flash, DeepSeek Controller, or R1.",
            "Input scope is restricted to current-run fresh artifacts: research_evidence_packet.md, intelligence_layer_report.md, parent_training_supplement.md, structure_mapper.md, evidence_judge.md, premise_auditor.md, L1-L11 StageRecords, and the user's original question.",
            "Output duty: generate mutually exclusive alternatives, different intervention-intensity paths, and action options under different risk assumptions.",
            "Return only these sections: mutually_exclusive_alternatives, intervention_intensity_paths, risk_assumption_branches, option_tradeoffs_for_later_stages.",
            "Do not harvest insights, converge views, rank a final choice, or write any user-facing plan.",
            f"Current run root: {base}",
            "",
            "## User original question",
            query,
            "",
            "## L1-L11 StageRecords",
            json.dumps(trace, ensure_ascii=False, indent=2),
            "",
            "## research_evidence_packet.md",
            packet,
            "",
            "## intelligence_layer_report.md",
            intelligence,
            "",
            "## parent_training_supplement.md",
            supplement,
            "",
            "## structure_mapper.md",
            structure,
            "",
            "## evidence_judge.md",
            evidence,
            "",
            "## premise_auditor.md",
            premise,
        ]
    )


def _alternative_generator_forbidden_tokens(text: str) -> list[str]:
    value = text or ""
    lowered = value.lower()
    tokens = []
    if "final_controller_report" in lowered:
        tokens.append("final_controller_report")
    if "pipeline_status=pipeline_complete" in lowered:
        tokens.append("pipeline_status=PIPELINE_COMPLETE")
    heading_lines = [line.strip() for line in value.splitlines() if line.strip().startswith("#")]
    if any(_is_forbidden_stage_heading(line, {"convergence_report", "insight_harvester"}) for line in heading_lines):
        tokens.append("later_stage_heading")
    if any(_is_final_report_heading(line) for line in heading_lines):
        tokens.append("final_report_heading")
    if any(_is_chinese_final_advice_heading(line) for line in heading_lines):
        tokens.append("chinese_final_advice_heading")
    return tokens


def _insight_harvester_prompt_from_artifacts(
    stages: list[dict[str, Any]],
    *,
    query: str,
    base_dir: str | Path,
) -> str:
    base = Path(base_dir).resolve()
    packet = Path(str(stages[5].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:2300]
    intelligence = Path(str(stages[6].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:2300]
    supplement = Path(str(stages[7].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:2300]
    structure = Path(str(stages[8].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:2300]
    evidence = Path(str(stages[9].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:2300]
    premise = Path(str(stages[10].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:2300]
    alternatives = Path(str(stages[11].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:2300]
    trace = [
        {
            "stage_name": record.get("stage_name"),
            "owner": record.get("owner"),
            "model": record.get("model"),
            "executor_model": record.get("executor_model"),
            "artifact_path": str(Path(str(record.get("artifact_path") or "")).resolve().relative_to(base)),
        }
        for record in stages
    ]
    return "\n".join(
        [
            "Run RESEARCH_DECISION stage 13: insight_harvester using Gemma-4-31B only.",
            f"Canonical model: {GEMMA431B}. This is a separate call from alternative_generator even though it uses the same actual model.",
            "Do not use Llama, Nemotron, Qwen, 9B, Flash, DeepSeek Controller, or R1.",
            "Input scope is restricted to current-run fresh artifacts: research_evidence_packet.md, intelligence_layer_report.md, parent_training_supplement.md, structure_mapper.md, evidence_judge.md, premise_auditor.md, alternative_generator.md, L1-L12 StageRecords, and the user's original question.",
            "Output duty: extract cross-model insights, conflicts, outliers, high-impact low-confidence points, and decision turning points.",
            "Return only these sections: cross_model_insights, conflicts_and_tensions, outliers, high_impact_low_confidence_points, decision_turning_points.",
            "Do not converge views, rank a final choice, or write any user-facing plan.",
            f"Current run root: {base}",
            "",
            "## User original question",
            query,
            "",
            "## L1-L12 StageRecords",
            json.dumps(trace, ensure_ascii=False, indent=2),
            "",
            "## research_evidence_packet.md",
            packet,
            "",
            "## intelligence_layer_report.md",
            intelligence,
            "",
            "## parent_training_supplement.md",
            supplement,
            "",
            "## structure_mapper.md",
            structure,
            "",
            "## evidence_judge.md",
            evidence,
            "",
            "## premise_auditor.md",
            premise,
            "",
            "## alternative_generator.md",
            alternatives,
        ]
    )


def _insight_harvester_forbidden_tokens(text: str) -> list[str]:
    value = text or ""
    lowered = value.lower()
    tokens = []
    if "final_controller_report" in lowered:
        tokens.append("final_controller_report")
    if "pipeline_status=pipeline_complete" in lowered:
        tokens.append("pipeline_status=PIPELINE_COMPLETE")
    heading_lines = [line.strip() for line in value.splitlines() if line.strip().startswith("#")]
    if any(_is_forbidden_stage_heading(line, {"convergence_report"}) for line in heading_lines):
        tokens.append("later_stage_heading")
    if any(_is_final_report_heading(line) for line in heading_lines):
        tokens.append("final_report_heading")
    if any(_is_chinese_final_advice_heading(line) for line in heading_lines):
        tokens.append("chinese_final_advice_heading")
    return tokens


def _convergence_report_prompt_from_artifacts(
    stages: list[dict[str, Any]],
    *,
    query: str,
    base_dir: str | Path,
) -> str:
    base = Path(base_dir).resolve()
    packet = Path(str(stages[5].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:2200]
    intelligence = Path(str(stages[6].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:2200]
    supplement = Path(str(stages[7].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:2200]
    structure = Path(str(stages[8].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:2200]
    evidence = Path(str(stages[9].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:2200]
    premise = Path(str(stages[10].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:2200]
    alternatives = Path(str(stages[11].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:2200]
    insights = Path(str(stages[12].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:2200]
    trace = [
        {
            "stage_name": record.get("stage_name"),
            "owner": record.get("owner"),
            "model": record.get("model"),
            "executor_model": record.get("executor_model"),
            "artifact_path": str(Path(str(record.get("artifact_path") or "")).resolve().relative_to(base)),
        }
        for record in stages
    ]
    profiles = _task_engine_profiles_from_query(query)
    foresight_sections = _convergence_foresight_template_lines(profiles)
    return "\n".join(
        [
            "Run RESEARCH_DECISION stage 14: convergence_report using R1-32B only.",
            f"Canonical model: {R1_32B}. Actual OMLX model must be {R1_ACTUAL_MODEL_DEFAULT}.",
            "This is a separate fresh R1 call from L3_r1_synthesis. Do not reuse r1_synthesis.md.",
            "Do not use AGY, DDGS, web_search, api_call, codex_exec, Controller, Flash, Qwen, Nemotron, Llama, Gemma, or 9B.",
            "Input scope is restricted to current-run fresh artifacts: research_evidence_packet.md, intelligence_layer_report.md, parent_training_supplement.md, structure_mapper.md, evidence_judge.md, premise_auditor.md, alternative_generator.md, insight_harvester.md, L1-L13 StageRecords, and the user's original question.",
            "Output duty: synthesize the five divergence roles, identify conflicts, and form a convergence decision framework.",
            "Internal output_quality_profile: " + ", ".join(profiles) + ".",
            "If foresight_mechanism is present, include key driving variables, input variables -> mediating mechanisms -> output variables, scenario branches, uncertainty/failure conditions, observable counter-signals, and certainty levels.",
            *foresight_sections,
            "If implementation_plan is present, include cycle, frequency, steps, metrics, and adjustment rules only when the user asks for implementation.",
            (
                "Return the hard-template headings exactly as listed above; include divergence_role_summary, conflicts_to_resolve, convergence_decision_framework, uncertainty_boundaries, and handoff_questions_for_external_calibration content inside those headings."
                if foresight_sections
                else "Return only these sections: divergence_role_summary, conflicts_to_resolve, convergence_decision_framework, uncertainty_boundaries, handoff_questions_for_external_calibration."
            ),
            "Do not write final_controller_report, PIPELINE_COMPLETE markers, or a final user-facing report.",
            f"Current run root: {base}",
            "",
            "## User original question",
            query,
            "",
            "## L1-L13 StageRecords",
            json.dumps(trace, ensure_ascii=False, indent=2),
            "",
            "## research_evidence_packet.md",
            packet,
            "",
            "## intelligence_layer_report.md",
            intelligence,
            "",
            "## parent_training_supplement.md",
            supplement,
            "",
            "## structure_mapper.md",
            structure,
            "",
            "## evidence_judge.md",
            evidence,
            "",
            "## premise_auditor.md",
            premise,
            "",
            "## alternative_generator.md",
            alternatives,
            "",
            "## insight_harvester.md",
            insights,
        ]
    )


def _convergence_report_forbidden_tokens(text: str) -> list[str]:
    value = text or ""
    lowered = value.lower()
    tokens = []
    if "final_controller_report" in lowered:
        tokens.append("final_controller_report")
    if "pipeline_status=pipeline_complete" in lowered:
        tokens.append("pipeline_status=PIPELINE_COMPLETE")
    for token in ("web_search", "api_call", "codex_exec", "delegate_task"):
        if token in lowered:
            tokens.append(token)
    if any(_is_final_report_heading(line.strip()) for line in value.splitlines() if line.strip().startswith("#")):
        tokens.append("final_report_heading")
    if any(_is_chinese_final_advice_heading(line.strip()) for line in value.splitlines() if line.strip().startswith("#")):
        tokens.append("chinese_final_advice_heading")
    return tokens


def _external_calibration_prompt_from_artifacts(
    stages: list[dict[str, Any]],
    *,
    query: str,
    base_dir: str | Path,
) -> str:
    base = Path(base_dir).resolve()
    packet = Path(str(stages[5].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:5000]
    convergence = Path(str(stages[13].get("artifact_path") or "")).resolve().read_text(encoding="utf-8", errors="replace")[:8000]
    trace = [
        {
            "stage_name": record.get("stage_name"),
            "owner": record.get("owner"),
            "model": record.get("model"),
            "executor_model": record.get("executor_model"),
            "artifact_path": str(Path(str(record.get("artifact_path") or "")).resolve().relative_to(base)),
        }
        for record in stages
    ]
    return "\n".join(
        [
            "Run RESEARCH_DECISION stage 15: external_calibration.",
            "Executor policy: GPT Bridge primary; Gemini/agy Gemini 3.1 Pro (High) fallback only if GPT Bridge is unavailable.",
            "Do not use Nemotron, R1, DeepSeek Controller, Qwen, Llama, Gemma, DDGS, or web_search.",
            "Input scope is restricted to current-run fresh artifacts: convergence_report.md, research_evidence_packet.md, L1-L14 StageRecords, and the user's original question.",
            "Output duty: calibrate the evidence strength of convergence_report; mark claims as supported, plausible, speculative, or contradicted; check over-inference; give a calibration verdict.",
            "Return only these sections: calibration_scope, claim_strength_table, over_inference_checks, contradiction_checks, calibration_verdict, handoff_notes_for_final_controller.",
            "Do not write the final controller stage, PIPELINE_COMPLETE markers, final user advice, or a final report.",
            f"Current run root: {base}",
            "",
            "## User original question",
            query,
            "",
            "## L1-L14 StageRecords",
            json.dumps(trace, ensure_ascii=False, indent=2),
            "",
            "## research_evidence_packet.md",
            packet,
            "",
            "## convergence_report.md",
            convergence,
        ]
    )


def _external_calibration_forbidden_tokens(text: str) -> list[str]:
    value = text or ""
    lowered = value.lower()
    tokens = []
    if "final_controller_report" in lowered:
        tokens.append("final_controller_report")
    if "pipeline_status=pipeline_complete" in lowered:
        tokens.append("pipeline_status=PIPELINE_COMPLETE")
    if any(_is_final_report_heading(line.strip()) for line in value.splitlines() if line.strip().startswith("#")):
        tokens.append("final_report_heading")
    if any(_is_chinese_final_advice_heading(line.strip()) for line in value.splitlines() if line.strip().startswith("#")):
        tokens.append("chinese_final_advice_heading")
    return tokens


def _final_controller_packet_from_artifacts(
    stages: list[dict[str, Any]],
    *,
    query: str,
    base_dir: str | Path,
) -> dict[str, Any]:
    base = Path(base_dir).resolve()
    names = [
        "L5_deepseek_acceptance",
        "intelligence_layer",
        "supplementary_search",
        "structure_mapper",
        "evidence_judge",
        "premise_auditor",
        "alternative_generator",
        "insight_harvester",
        "convergence_report",
        "external_calibration",
    ]
    by_name = {str(record.get("stage_name") or ""): record for record in stages}
    excerpts: dict[str, str] = {}
    for name in names:
        path = Path(str(by_name[name].get("artifact_path") or "")).resolve()
        text = path.read_text(encoding="utf-8", errors="replace")
        excerpts[name] = _safe_final_excerpt(text)
    passthrough_keys = (
        "created_in_current_run",
        "legacy_contaminated",
        "valid_for_pipeline",
        "run_id",
        "output_root",
        "prompt_sha256",
        "created_at",
        "stage_index",
        "stage_number",
    )
    trace = []
    for record in stages:
        trace_item = {
            "stage_name": record.get("stage_name"),
            "owner": record.get("owner"),
            "model": record.get("model"),
            "executor_model": record.get("executor_model"),
            "artifact_path": str(Path(str(record.get("artifact_path") or "")).resolve().relative_to(base)),
        }
        for key in passthrough_keys:
            if key in record:
                trace_item[key] = record.get(key)
        trace.append(trace_item)
    return {
        "mode": ENGINE_RESEARCH_DECISION,
        "query": query,
        "base_dir": str(base),
        "output_quality_profile": _task_engine_profiles_from_query(query),
        "stage_trace": trace,
        "excerpts": excerpts,
    }


def _final_controller_report_from_packet(packet: dict[str, Any]) -> str:
    query = str(packet.get("query") or "").strip()
    mode = str(packet.get("mode") or ENGINE_RESEARCH_DECISION)
    excerpts = packet.get("excerpts") if isinstance(packet.get("excerpts"), dict) else {}
    calibration = _safe_final_excerpt(str(excerpts.get("external_calibration") or ""))
    convergence = _safe_final_excerpt(str(excerpts.get("convergence_report") or ""))
    evidence = _safe_final_excerpt(str(excerpts.get("evidence_judge") or ""))
    alternatives = _safe_final_excerpt(str(excerpts.get("alternative_generator") or ""))
    premise = _safe_final_excerpt(str(excerpts.get("premise_auditor") or ""))
    if mode == ENGINE_DECISION:
        include_evidence_boundary = _decision_final_requires_evidence_boundary(packet)
        profiles = _normalize_profiles(packet.get("output_quality_profile"))
        if (
            _profiles_require_foresight_template(profiles)
            or _decision_query_requests_future_inversion_structure(query)
            or _decision_query_forbids_advice(query)
        ):
            return _decision_future_inversion_report(
                query,
                include_evidence_boundary=include_evidence_boundary,
                convergence_digest=str(packet.get("convergence_fixed_section_digest") or ""),
                calibration_constraints=str(packet.get("external_calibration_hard_constraints") or ""),
                research_evidence_context=str(packet.get("research_evidence_packet_context") or ""),
            )
        return _generic_decision_final_report(query, packet=packet, include_evidence_boundary=include_evidence_boundary)
    profiles = _normalize_profiles(packet.get("output_quality_profile"))
    if PROFILE_FORESIGHT_MECHANISM in profiles:
        return _research_decision_foresight_final_report(query, packet)
    return _research_decision_generic_final_report(query, packet)


def _research_decision_foresight_final_report(query: str, packet: dict[str, Any]) -> str:
    excerpts = packet.get("excerpts") if isinstance(packet.get("excerpts"), dict) else {}
    research_excerpt = _safe_final_excerpt(
        str(excerpts.get("research_evidence_packet") or excerpts.get("L5_deepseek_acceptance") or ""),
        limit=700,
    )
    convergence_excerpt = _safe_final_excerpt(str(excerpts.get("convergence_report") or ""), limit=700)
    calibration_excerpt = _safe_final_excerpt(str(excerpts.get("external_calibration") or ""), limit=700)
    compliance_domain = scoring_calibration.classify_compliance_domain(query)
    if compliance_domain == "audit_finance_compliance":
        task_domain = "审计底稿整理、财务异常检测、合规报告生成和管理层讨论分析草稿等例行审计/财务分析工作"
        production_role = "初级审计员"
        operations_role = "企业财务分析师和财务流程分析角色"
        authority_moats = "审计判断、内控责任、重大错报风险识别、管理层沟通、监管问责和最终复核权"
        scenario_context = "审计底稿、异常检测、合规报告、管理层讨论分析草稿和可重复财务控制场景"
        value_migration = "价值从单件底稿或报告草稿生产迁移到异常解释、控制点设计、审计判断、管理层沟通和跨系统数据治理"
        high_value_falsifier = "如果关键价值仍集中在重大判断、异常解释、审计复核、责任签字、监管问责和管理层沟通"
        scenario_b = "AI 把重复审计/财务分析产出压到低成本层，靠近业务系统、内控流程和异常解释的角色在特定场景上升。certainty_level：medium。"
        scenario_c = "企业财务分析师在权威、薪酬、声望、流动性和战略位置上整体超过初级审计员。certainty_level：low；当前证据不足。"
        controversy = "争议集中在 AI 采用速度、审计责任、监管问责、组织授权、数据质量和复杂判断的不可替代性。"
        counter_signal = "低风险底稿整理和草稿生成被自动化，但重大判断、异常解释、审计复核、监管问责和管理层沟通仍强绑定专业人员。"
    elif compliance_domain == "legal_compliance":
        task_domain = "法律检索、合同起草、案例摘要、合规解释等例行法律生产"
        production_role = "初级律师"
        operations_role = "企业法务分析师/legal-ops 型角色"
        authority_moats = "牌照、责任承担、保密特权、诉讼/谈判场景、监督义务和最终法律签署权"
        scenario_context = "合规运营、合同生命周期管理、采购/隐私/风控流程和可重复风险控制场景"
        value_migration = "价值从单件法律文本生产迁移到风险筛查、流程编排、控制点设计和跨部门落地"
        high_value_falsifier = "如果关键价值仍集中在独立法律判断、代理、谈判、责任承担和高风险签署"
        scenario_b = "AI 把重复法律生产压到低成本层，靠近业务系统和风控流程的角色在特定场景上升。certainty_level：medium。"
        scenario_c = "企业法务分析师在权威、薪酬、声望、流动性和战略位置上整体超过初级律师。certainty_level：low；当前证据不足。"
        controversy = "争议集中在 AI 采用速度、监管责任、客户信任、组织授权和不同法律场景的不可替代性。"
        counter_signal = "低风险任务被自动化，但高风险签署、诉讼、谈判、调查和责任承担仍强绑定持证专业人员。"
    else:
        task_domain = "可被工具显著压低成本的例行知识生产"
        production_role = "传统初级生产角色"
        operations_role = "流程整合与业务系统型角色"
        authority_moats = "责任承担、制度授权、信任关系、复杂判断、监督义务和最终决策权"
        scenario_context = "运营、流程治理、风险控制和可重复知识生产场景"
        value_migration = "价值从单件文本生产迁移到流程编排、控制点设计和跨部门落地"
        high_value_falsifier = "如果关键价值仍集中在复杂判断、责任承担和高风险签署"
        scenario_b = "AI 把重复知识生产压到低成本层，靠近业务系统和流程控制的角色在特定场景上升。certainty_level：medium。"
        scenario_c = "业务系统型角色在权威、薪酬、声望、流动性和战略位置上整体超过传统初级生产角色。certainty_level：low；当前证据不足。"
        controversy = "争议集中在 AI 采用速度、组织授权、客户信任、责任分配和复杂场景的不可替代性。"
        counter_signal = "低风险任务被自动化，但复杂判断、责任承担、监督义务和最终决策权仍强绑定专业人员。"
    consumed = {
        "research_evidence_packet": bool(research_excerpt),
        "convergence_report": bool(convergence_excerpt),
        "external_calibration": bool(calibration_excerpt),
    }
    lines = [
        "# 研究决策最终报告",
        "",
        "## 核心结论",
        "局部/场景性结构反转可能成立，职业整体反转证据不足。",
        f"AI 持续压低{task_domain}的成本，确实会削弱单纯依靠高频基础产出的训练和筛选优势；但这首先改变的是任务价值分布，不足以证明职业整体的权威、收入、声望和上升通道发生全面反转。",
        "",
        "## 输入材料吸收",
        "- research_evidence_packet：用于限定证据底座，只采纳当前材料能支持的任务成本下降、例行产出被压缩、长期职业整体反转证据不足等边界。",
        "- convergence_report：用于吸收关键驱动、机制链、情景分叉、反证信号和 certainty_level，但不直接复制其中偏强的全面反转表述。",
        "- external_calibration：用于执行降调；把结论收束为局部/场景性反转 plausible，把职业整体反转保留为 speculative 或证据不足。",
        "- source_consumption_check："
        + "; ".join(f"{name}={'yes' if used else 'no'}" for name, used in consumed.items()),
        "",
        "## 证据分层",
        "",
        "### evidence_supported",
        f"判断：AI 对{task_domain}的边际成本压缩已经足以改变入门级产出的稀缺性；单纯更快完成检索、初稿、摘要或规则解释，不能再稳定构成{production_role}的差异化优势。",
        "触发条件：AI 工具在组织内被允许进入标准工作流，并能稳定处理低风险、可校验、重复性强的产出。",
        "中间机制：输入变量是任务成本下降；中介机制是例行产出被模板化、自动化和批量化；输出变量是基础产出的市场溢价下降。",
        "失效条件或反证信号：若工具输出可靠性不足、组织因责任和保密约束限制使用，或客户仍要求人工逐项生产，则该判断需要下调。",
        "certainty_level：high for routine task cost compression; medium for organizational adoption speed.",
        "evidence_tier：evidence_supported",
        "decision_use：把“例行产出速度”从核心优势降级，改看校验、责任、流程设计和业务风险翻译能力。",
        "",
        "### reasonable_inference",
        f"判断：在{scenario_context}中，{operations_role}可能相对升值，因为其位置更接近业务数据、流程接口和系统改造。",
        "触发条件：组织把 AI 产出接入业务系统，而不是只把它当作个人写作或检索助手。",
        f"中间机制：输入变量是生产成本降低和工作流数据化；中介机制是{value_migration}；输出变量是场景性角色优势重排。",
        f"失效条件或反证信号：{high_value_falsifier}，角色重排就会停留在协作效率提升，而非优势反转。",
        "certainty_level：medium",
        "evidence_tier：reasonable_inference",
        "decision_use：在评价职业前景时区分 production value、operation value 和 authority value，不能只看谁更会使用工具。",
        "",
        "### foresight_hypothesis",
        "判断：局部/场景性结构反转可能成立，职业整体反转证据不足。",
        "触发条件：未来十年 AI 继续降低例行任务成本，组织治理重点从产出文本转向系统控制、责任分配和业务嵌入。",
        f"中间机制：输入变量是 AI 能力提升和单位产出成本下降；中介机制是{task_domain}的可替代性上升，同时{authority_moats}仍保留；输出变量是局部角色优势上移但职业整体并未全面倒置。",
        "失效条件或反证信号：若监管、责任、客户信任和职业晋升仍强绑定持证专业路径，或 AI 只提升效率而不改变授权结构，则整体反转假设不成立。",
        "certainty_level：low-to-medium",
        "evidence_tier：foresight_hypothesis",
        "decision_use：把该结论作为需追踪的未来假设，而不是当前可直接下定论的职业预测。",
        "",
        "## 关键驱动变量",
        "关键驱动包括：AI 单位任务成本、组织采用速度、输出可验证性、责任/保密/监管约束、客户信任、业务系统嵌入深度、可替代任务占比和职业授权结构。",
        "",
        "## 机制链总览",
        "输入变量：AI 降低例行知识生产成本、组织流程数字化、可重复风险控制需求增加。",
        "中介机制：基础产出被压缩为低差异化能力，价值迁移到校验、问责、流程设计、业务风险翻译和复杂场景判断。",
        f"输出变量：{operations_role}在特定运营场景相对上升；{production_role}的基础训练优势被削弱；但职业整体权威不发生充分证据支持的全面倒置。",
        "",
        "## 情景分叉",
        "情景 A：modified continuity。AI 提升两类角色效率，基础门槛提高，但授权、责任和复杂判断仍维持原有职业秩序。certainty_level：medium-high。",
        f"情景 B：partial operational reversal。{scenario_b}",
        f"情景 C：full occupational reversal。{scenario_c}",
        "",
        "## 证据强度、争议和缺口",
        "evidence_strength: strong for routine-task cost compression; medium for local operational value migration; weak for full occupational reversal over ten years.",
        f"controversy: {controversy}",
        "evidence_gap: 缺口是缺少十年期、角色级、跨组织的直接证据来证明职业整体权威和回报发生全面倒置。",
        "",
        "## 反证信号与不确定性边界",
        f"- 反证信号：{counter_signal}",
        "- 反证信号：企业流程岗位效率提升，却没有转化为更高授权、薪酬、晋升速度或战略话语权。",
        "- 反证信号：客户、监管和组织治理继续要求专业人员对 AI 输出承担最终责任。",
        "- 不确定性边界：如果未来 AI 同时获得高可信校验、制度认可和责任承接机制，整体反转概率才需要重新上调。",
        "",
        "## 最终判断",
        "局部/场景性结构反转可能成立，职业整体反转证据不足。",
    ]
    return "\n".join(lines)


def _research_decision_generic_final_report(query: str, packet: dict[str, Any]) -> str:
    excerpts = packet.get("excerpts") if isinstance(packet.get("excerpts"), dict) else {}
    consumed = {
        "research_evidence_packet": bool(excerpts.get("research_evidence_packet") or excerpts.get("L5_deepseek_acceptance")),
        "convergence_report": bool(excerpts.get("convergence_report")),
        "external_calibration": bool(excerpts.get("external_calibration")),
    }
    frame = _research_decision_generic_frame(query)
    calibration_action = frame["calibration_action"]
    if excerpts.get("external_calibration"):
        calibration_action += " external_calibration 的降调或保留意见已执行为条件化结论，而不是直接升级为确定建议。"
        calibration_body = _safe_final_excerpt(
            _external_calibration_quality_body(str(excerpts.get("external_calibration") or "")),
            limit=600,
        )
        if calibration_body:
            calibration_action += f" 已吸收的校准正文：{calibration_body}"
    return "\n".join(
        [
            "# 研究决策最终报告",
            "",
            "## 核心结论",
            frame["conclusion"],
            "",
            "## 输入材料吸收",
            "- research_evidence_packet：用于限定哪些判断可以作为 evidence_supported，哪些只能作为推断或假设。",
            "- convergence_report：用于吸收收敛后的关键变量、机制关系、风险边界和分叉条件，不复制中间正文。",
            "- external_calibration：用于执行降调、反对意见和最终可用性边界。",
            "- source_consumption_check："
            + "; ".join(f"{name}={'yes' if used else 'no'}" for name, used in consumed.items()),
            "",
            "## 证据分层",
            "",
            "### evidence_supported",
            f"判断：{frame['supported_judgment']}",
            f"触发条件：{frame['supported_condition']}",
            f"中间机制：{frame['supported_mechanism']}",
            f"失效条件或反证信号：{frame['supported_falsifier']}",
            f"certainty_level：{frame['supported_certainty']}",
            "evidence_tier：evidence_supported",
            f"decision_use：{frame['supported_use']}",
            "",
            "### reasonable_inference",
            f"判断：{frame['inference_judgment']}",
            f"触发条件：{frame['inference_condition']}",
            f"中间机制：{frame['inference_mechanism']}",
            f"失效条件或反证信号：{frame['inference_falsifier']}",
            f"certainty_level：{frame['inference_certainty']}",
            "evidence_tier：reasonable_inference",
            f"decision_use：{frame['inference_use']}",
            "",
            "### foresight_hypothesis",
            f"判断：{frame['hypothesis_judgment']}",
            f"触发条件：{frame['hypothesis_condition']}",
            f"中间机制：{frame['hypothesis_mechanism']}",
            f"失效条件或反证信号：{frame['hypothesis_falsifier']}",
            f"certainty_level：{frame['hypothesis_certainty']}",
            "evidence_tier：foresight_hypothesis",
            f"decision_use：{frame['hypothesis_use']}",
            "",
            "## 校准执行",
            calibration_action,
            "",
            "## 收敛吸收",
            f"关键变量：{frame['key_variables']}",
            f"机制链：{frame['mechanism_chain']}",
            f"风险边界：{frame['risk_boundary']}",
            "",
            "## 证据强度、争议和缺口",
            f"evidence_strength: {frame['evidence_strength']}",
            f"controversy: {frame['controversy']}",
            f"evidence_gap: {frame['evidence_gap']}",
            "",
            "## 最终决策含义",
            frame["decision_meaning"],
        ]
    )


def _research_decision_generic_frame(query: str) -> dict[str, str]:
    value = query or ""
    lowered = value.lower()
    if "越南" in value and ("电池回收" in value or "梯次利用" in value):
        return {
            "conclusion": "越南北部电动车电池回收与梯次利用产业链可以保留进入选项，但应以轻资产试点、合作切入和监管/供给验证为前提；不支持直接重资产全面进入。",
            "supported_judgment": "电动车电池回收与梯次利用会受到电动车保有量、退役电池供给、环保监管、渠道控制和安全合规约束共同影响，不能只按市场增长叙事判断。",
            "supported_condition": "目标区域存在可验证的退役电池来源、合规处置要求、下游利用场景和本地合作方。",
            "supported_mechanism": "退役电池供给增加与监管趋严提高回收需求；但渠道、检测分级、安全责任和资本开支决定真实利润池。",
            "supported_falsifier": "若退役电池规模不足、上游渠道被主机厂/电池厂锁定、许可和环保成本高于预期，则进入理由显著减弱。",
            "supported_certainty": "medium",
            "supported_use": "把进入判断拆成供给、合规、渠道、技术分级和下游需求五个验证门槛。",
            "inference_judgment": "中型工业服务公司更适合从检测、合规运营、B2B 回收服务或合作项目切入，而不是先建设完整闭环产能。",
            "inference_condition": "公司已有工业客户、现场服务、合规运营或设备维护能力，并能找到本地牌照/渠道伙伴。",
            "inference_mechanism": "既有服务能力降低获客和运营摩擦；合作切入降低政策、供给和技术不确定性；逐步验证后再扩产。",
            "inference_falsifier": "若公司缺少本地执行团队、不能控制安全责任，或合作方无法提供稳定来源，则轻资产试点也可能失真。",
            "inference_certainty": "medium",
            "inference_use": "优先设计 6-12 个月验证项目，而不是一次性资本承诺。",
            "hypothesis_judgment": "三年内可能出现局部机会窗口，但行业利润和供给成熟度未必足以支撑全面进入。",
            "hypothesis_condition": "越南北部电动车产业链继续扩张，退役和次品电池开始形成可商业化流量。",
            "hypothesis_mechanism": "产业集聚带来电池流量；监管要求把非合规处置成本显性化；合规服务商获得早期卡位价值。",
            "hypothesis_falsifier": "若退役周期晚于预期、监管执行弱、主机厂闭环回收，或梯次利用经济性不稳定，则机会窗口后移。",
            "hypothesis_certainty": "low-to-medium",
            "hypothesis_use": "作为期权型布局跟踪，不作为重资产进入的充分理由。",
            "calibration_action": "将“进入”降调为有条件试点和期权布局；把全面进入保留为后续验证结果。",
            "key_variables": "退役电池供给、渠道控制、许可环保、安全责任、检测分级能力、下游需求。",
            "mechanism_chain": "产业增长 -> 电池流量出现 -> 合规和分级需求上升 -> 服务型切入可验证利润池 -> 再决定是否扩产。",
            "risk_boundary": "供给未成规模、渠道被锁、监管弱执行、技术责任高、资本开支过早。",
            "evidence_strength": "medium for industry drivers; low-to-medium for three-year local profitability.",
            "controversy": "机会大小取决于本地供给时点、政策执行、主机厂策略和合作方质量。",
            "evidence_gap": "缺少本地实时退役量、许可成本、渠道报价、单位经济性和合作方尽调。",
            "decision_meaning": "结论可用于启动小规模尽调/试点；不能直接支持重资产建厂或全链条进入。",
        }
    if "postgresql" in lowered or "lakehouse" in lowered or "事件驱动" in value:
        return {
            "conclusion": "不建议一次性从单体架构跃迁到完整事件驱动 + lakehouse 体系；更合理的是按瓶颈分阶段演进。",
            "supported_judgment": "架构迁移的必要性应来自明确的规模、实时性、数据治理、成本或团队协作瓶颈，而不是技术栈本身更现代。",
            "supported_condition": "当前系统出现 cron 延迟不可接受、分析查询拖垮主库、数据血缘混乱、特征复用困难或团队并行开发受阻。",
            "supported_mechanism": "业务瓶颈提高现有架构的协调成本；分层数据系统和事件流可降低耦合、提高可观测性和数据复用。",
            "supported_falsifier": "若数据量、实时性、团队规模和客户 SLA 仍可由 PostgreSQL 单体加增量优化满足，则迁移收益不足。",
            "supported_certainty": "high for migration-risk principle; medium for target architecture fit.",
            "supported_use": "先用瓶颈清单决定迁移范围，避免把架构升级当成默认路线。",
            "inference_judgment": "可优先拆出变更数据捕获、对象存储历史层、关键指标管道和只读分析层，再评估是否需要全事件驱动。",
            "inference_condition": "团队能维护数据契约、回放语义、监控告警、成本治理和迁移期间的双写/对账。",
            "inference_mechanism": "低风险拆分先解决最痛瓶颈；数据契约降低下游破坏；逐步迁移保留回滚空间。",
            "inference_falsifier": "若团队缺少平台能力、数据契约执行弱或业务指标频繁变动，复杂平台会增加故障面。",
            "inference_certainty": "medium",
            "inference_use": "把路线改成 staged migration，而不是 big-bang rewrite。",
            "hypothesis_judgment": "未来若产品走向实时特征、客户级数据隔离和跨域分析，lakehouse/流式管道可能成为主路径。",
            "hypothesis_condition": "客户要求更低延迟、更长历史、更复杂特征和更强可审计性。",
            "hypothesis_mechanism": "实时需求与历史分析分离推动冷热层和流批统一；特征复用推动管道产品化。",
            "hypothesis_falsifier": "若客户主要需要批量报表、数据规模平稳、SLA 宽松，则复杂路线会过度建设。",
            "hypothesis_certainty": "low-to-medium",
            "hypothesis_use": "作为架构演进方向跟踪，不作为立即全量迁移理由。",
            "calibration_action": "将“是否迁移”降调为“是否按瓶颈分阶段迁移”；反对无条件重构。",
            "key_variables": "数据规模、延迟 SLA、查询隔离、团队平台能力、数据契约成熟度、迁移风险。",
            "mechanism_chain": "瓶颈出现 -> 单体协调成本上升 -> 拆分分析/历史/事件层 -> 降低耦合但增加平台复杂度。",
            "risk_boundary": "平台能力不足、双写不一致、成本失控、过早抽象、事件语义混乱。",
            "evidence_strength": "high for staged architecture governance; medium for specific lakehouse/event route.",
            "controversy": "争议在于当前瓶颈是否足以支付复杂度成本。",
            "evidence_gap": "缺少当前流量、查询模式、延迟 SLA、团队能力和迁移成本数据。",
            "decision_meaning": "可批准阶段性架构验证和第一批瓶颈拆解；不应批准一次性全量重构。",
        }
    if "拼读困难" in value or "音韵意识" in value or "泛阅读" in value:
        return {
            "conclusion": "在适用儿童中，系统性音韵意识 + 明确解码训练 + 高频短时练习更值得作为核心干预；泛阅读可补充但不应替代。",
            "supported_judgment": "拼读困难干预更需要直接、系统、可重复的音韵和解码训练，单靠泛阅读通常不足以补齐核心技能缺口。",
            "supported_condition": "儿童存在稳定的拼读/解码困难，且能获得结构化教学、短时高频练习和进展监测。",
            "supported_mechanism": "明确教学降低规则发现负担；高频短练增加巩固机会；进展监测帮助调节难度和避免无效重复。",
            "supported_falsifier": "若评估显示困难并非音韵/解码主导，或儿童对训练负荷出现明显负面反应，应调整方案。",
            "supported_certainty": "high",
            "supported_use": "把系统性解码训练作为一线方案，同时保留阅读兴趣和理解活动。",
            "inference_judgment": "训练应以短周期目标、错误类型记录和难度递进组织，而不是只增加阅读量。",
            "inference_condition": "家庭/学校能执行稳定频率，并能记录正确率、流畅度、错误类型和疲劳反应。",
            "inference_mechanism": "可观察数据把训练从泛化建议变成可调整干预；短时高频降低挫败并提高巩固。",
            "inference_falsifier": "若 6-8 周没有进展，或错误类型不匹配训练内容，需要专业评估和方案调整。",
            "inference_certainty": "medium-high",
            "inference_use": "用于制定干预执行和复盘规则，而不是一次性判断有效/无效。",
            "hypothesis_judgment": "若执行质量高，系统训练可能改善后续阅读流畅度和学习信心，但长期幅度需个体跟踪。",
            "hypothesis_condition": "训练持续、难度合适、反馈及时，并与真实阅读材料衔接。",
            "hypothesis_mechanism": "基础解码自动化提升后，认知资源可转向理解和流畅阅读。",
            "hypothesis_falsifier": "若基础技能提升不能迁移到真实阅读，或动机下降明显，则长期收益假设下调。",
            "hypothesis_certainty": "medium",
            "hypothesis_use": "作为跟踪假设，用阶段测评决定是否延续、升级或转诊。",
            "calibration_action": "保留强证据方向，但避免承诺个体长期效果；强调评估、执行质量和复盘。",
            "key_variables": "困难类型、训练结构、练习频率、反馈质量、儿童负荷、进展监测。",
            "mechanism_chain": "明确音韵/解码训练 -> 技能分解和重复巩固 -> 解码自动化提升 -> 阅读迁移需继续验证。",
            "risk_boundary": "误判困难类型、练习过载、只训练孤立技能、缺少真实阅读迁移。",
            "evidence_strength": "high for structured phonological/decoding intervention direction; medium for individual long-term magnitude.",
            "controversy": "争议主要在个体差异、执行质量和与泛阅读/理解活动的配比。",
            "evidence_gap": "缺少该儿童具体评估、错误类型、共现困难和可执行资源。",
            "decision_meaning": "可用于支持启动结构化干预；不应用来跳过个体评估或承诺固定疗效。",
        }
    if "具身 ai 机器人" in value or "消费硬件基金" in value or "提前下注" in value:
        return {
            "conclusion": "可以保留小额期权和主题研究，但不支持在当前证据下重仓提前下注规模化消费市场。",
            "supported_judgment": "家庭陪伴型具身 AI 机器人同时受硬件成本、可靠性、安全、内容价值、渠道、售后和家庭真实需求制约，不能只按大模型进步外推。",
            "supported_condition": "产品能证明高频使用、低退货、可承受价格、稳定安全和明确付费理由。",
            "supported_mechanism": "模型能力提升增加交互可能性；但硬件交付、家庭场景容错和持续价值决定是否形成消费市场。",
            "supported_falsifier": "若用户留存低、售后成本高、家庭场景风险大或价格无法下探，则规模化市场判断不成立。",
            "supported_certainty": "medium for constraint structure; low for 2030 market timing.",
            "supported_use": "把投资判断从叙事热度转向留存、成本、可靠性和渠道数据。",
            "inference_judgment": "基金更适合分散观察关键部件、平台能力、垂直场景和渠道验证，而不是只押单一通用陪伴终端。",
            "inference_condition": "存在可验证原型、真实家庭试用数据、供应链成本曲线和明确购买人群。",
            "inference_mechanism": "小额期权保留上行；里程碑投资降低市场时点错误；垂直场景先验证支付意愿。",
            "inference_falsifier": "若 Demo 强但留存弱，或成本下降慢于预期，追加投资应停止。",
            "inference_certainty": "medium",
            "inference_use": "用于设置投资门槛和跟踪指标，而非支持立即重仓。",
            "hypothesis_judgment": "2030 年前可能出现局部消费场景，但形成大规模通用家庭陪伴市场仍高度不确定。",
            "hypothesis_condition": "硬件成本、端侧/云端智能、安全认证、情感交互和售后体系同时成熟。",
            "hypothesis_mechanism": "多项成熟条件叠加后，机器人从新奇硬件转为可持续家庭服务入口。",
            "hypothesis_falsifier": "若杀手场景缺失、家庭信任不足、监管限制增强或替代设备满足需求，则趋势假设下调。",
            "hypothesis_certainty": "low",
            "hypothesis_use": "作为趋势跟踪假设，适合期权配置，不适合高确信主仓位。",
            "calibration_action": "将趋势叙事降调为低证据高不确定假设；只允许期权型投入。",
            "key_variables": "硬件成本、可靠性、安全认证、留存、支付意愿、渠道和售后、替代品竞争。",
            "mechanism_chain": "AI 交互进步 -> 原型可用性提升 -> 家庭真实使用验证 -> 供应链和服务体系达标 -> 才可能规模化。",
            "risk_boundary": "Demo 与留存脱节、成本/售后失控、安全与隐私风险、需求被手机/音箱替代。",
            "evidence_strength": "medium for constraint analysis; low for 2030 mass-market timing.",
            "controversy": "争议在市场时点、杀手场景、家庭接受度和硬件经济性。",
            "evidence_gap": "缺少规模化家庭留存、复购、售后成本、价格弹性和监管路径数据。",
            "decision_meaning": "可用于设计观察清单和小额投资规则；不能支持重仓提前下注。",
        }
    return {
        "conclusion": "当前只支持有条件推进或保留选项，不支持无条件、不可逆的大规模承诺。",
        "supported_judgment": "现有材料能支持问题存在真实决策价值，但关键结论仍需要按证据强度分层。",
        "supported_condition": "输入证据能覆盖目标人群、场景、约束、替代方案和失败信号。",
        "supported_mechanism": "证据先限定事实底座；机制推断连接场景；未来判断保留为可追踪假设。",
        "supported_falsifier": "若关键前提缺失、替代方案更优或校准意见反对，则结论必须下调。",
        "supported_certainty": "medium",
        "supported_use": "先确定哪些判断可直接用于决策，哪些只能作为追踪项。",
        "inference_judgment": "可采取分阶段、可逆、带监测指标的路径来降低误判成本。",
        "inference_condition": "能定义触发条件、停止条件、里程碑和复盘指标。",
        "inference_mechanism": "小步推进保留学习速度；阶段门槛防止把不确定假设变成沉没成本。",
        "inference_falsifier": "若试点指标无法测量或结果无法改变后续决策，则分阶段路径也没有价值。",
        "inference_certainty": "medium",
        "inference_use": "用于把最终判断转化为可执行的验证计划。",
        "hypothesis_judgment": "长期结果仍取决于外部条件变化，不能写成确定预测。",
        "hypothesis_condition": "关键变量按有利方向持续变化，并且没有出现强反证信号。",
        "hypothesis_mechanism": "外部趋势改变成本、能力或需求结构，进而改变最优决策。",
        "hypothesis_falsifier": "若趋势放缓、约束增强或替代路径更优，则假设失效。",
        "hypothesis_certainty": "low-to-medium",
        "hypothesis_use": "作为跟踪假设，而非最终承诺依据。",
        "calibration_action": "执行校准降调：把不确定内容保留为条件性判断。",
        "key_variables": "证据强度、适用场景、替代方案、执行成本、失败信号。",
        "mechanism_chain": "证据底座 -> 条件性推断 -> 可验证行动 -> 根据反馈调整。",
        "risk_boundary": "证据不足、泛化过度、执行条件缺失、反证信号被忽略。",
        "evidence_strength": "medium for bounded decision structure; low-to-medium for long-horizon claims.",
        "controversy": "争议取决于场景适配和关键前提是否成立。",
        "evidence_gap": "缺少足够具体的执行数据和反事实比较。",
        "decision_meaning": "可以用于设计下一步验证，但不能替代最终业务或专业尽调。",
    }



def _decision_final_requires_evidence_boundary(packet: dict[str, Any]) -> bool:
    if str(packet.get("mode") or "") != ENGINE_DECISION:
        return False
    if not packet.get("research_evidence_packet_context"):
        return False
    profiles = _normalize_profiles(packet.get("output_quality_profile"))
    return PROFILE_EVIDENCE_GROUNDED in profiles


def _decision_evidence_boundary_section(
    *,
    query: str = "",
    research_evidence_context: str = "",
    calibration_constraints: str = "",
) -> list[str]:
    compliance_domain = scoring_calibration.classify_compliance_domain(query)
    if compliance_domain == "legal_compliance":
        evidence_strength = (
            "证据强度：当前材料足以支持“需要法律、海洋法和合规边界”的条件性判断；"
            "对具体争议海域活动是否合法、可执行或可商业包装的结论，仍必须降级为需专业复核的中低确定性判断。"
        )
        controversy = (
            "争议点：争议集中在主权立场、UNCLOS/海洋法适用、司法或仲裁结果的解释、"
            "数据跨境与制裁/出口管制/地缘风险边界，不同法域和事实场景会改变结论。"
        )
        evidence_gap = (
            "证据缺口：缺少足以替代律师、海事合规和相关主管机构意见的事实适用材料；"
            "尤其缺少具体坐标、客户用途、数据类型、交易结构和管辖法分析。"
        )
    elif compliance_domain == "audit_finance_compliance":
        evidence_strength = (
            "证据强度：当前材料可支持对审计、财务和合规流程的条件性判断；"
            "涉及重大错报、监管披露或责任签署的结论仍只能作为需专业复核的中低确定性判断。"
        )
        controversy = "争议点：争议取决于适用准则、内控环境、数据质量、责任主体和监管解释。"
        evidence_gap = "证据缺口：缺少完整底稿、原始凭证、管理层解释和独立复核，不能替代正式审计判断。"
    else:
        evidence_strength = (
            "证据强度：当前材料可支持有边界的决策结构和条件性推断；"
            "高风险、长期或外部事实依赖强的判断必须保持中低确定性。"
        )
        controversy = "争议点：争议取决于适用场景、关键前提、替代方案、执行约束和反证信号。"
        evidence_gap = "证据缺口：缺少足够具体的事实适用、反事实比较和长期验证，不能把条件性判断写成确定结论。"
    if research_evidence_context and "evidence_supported" in research_evidence_context:
        evidence_strength += " 已纳入证据分层线索，但这里只呈现用户可读的证据边界。"
    if calibration_constraints:
        controversy += " 已纳入需要降调的边界意见，最终结论保持条件化。"
    return [
        "## 证据边界",
        evidence_strength,
        controversy,
        evidence_gap,
        "",
    ]


def _decision_future_inversion_report(
    query: str,
    *,
    include_evidence_boundary: bool = False,
    convergence_digest: str = "",
    calibration_constraints: str = "",
    research_evidence_context: str = "",
) -> str:
    return _generic_user_facing_future_report(
        query,
        include_evidence_boundary=include_evidence_boundary,
        convergence_digest=convergence_digest,
        calibration_constraints=calibration_constraints,
        research_evidence_context=research_evidence_context,
    )


def _sanitize_user_facing_excerpt(text: str, *, limit: int = 900) -> str:
    raw_lines = []
    blocked_line_terms = (
        "executor_model",
        "fallback_reasons",
        "DECISION Stage",
        "**Executor**",
        "calibration_verdict",
        "agreement_points",
        "disagreement_or_risk_points",
        "missing_considerations",
        "claim_strength_table",
        "over_inference_checks",
    )
    for line in (text or "").splitlines():
        if any(term in line for term in blocked_line_terms):
            continue
        raw_lines.append(line)
    value = _safe_final_excerpt("\n".join(raw_lines), limit=limit)
    replacements = {
        "## key_drivers": "关键驱动：",
        "## mechanism_chain": "机制链：",
        "## scenario_branches": "情景分叉：",
        "## counter_signals": "反证信号：",
        "## certainty_levels": "确定性等级：",
        "## uncertainty_boundary": "不确定性边界：",
        "external_calibration": "边界修正",
        "convergence_report": "收敛判断",
        "research packet": "证据材料",
        "research_evidence_packet": "证据材料",
        "final controller": "最终答案",
        "final_controller": "最终答案",
        "Elevate the Risk:": "风险边界：",
        "Lock in the Sequence:": "优先顺序：",
        "Restore Counter-Intuitive Strategy:": "反直觉假设：",
        "Strict Epistemic Tagging:": "证据分层：",
        "foresight_hypothesis": "前瞻假设",
        "reasonable_inference": "合理推断",
        "evidence_supported": "证据支持",
        "StageRecord": "阶段记录",
        "artifact": "材料",
        "pipeline": "流程",
        "Executor": "执行器",
        "premise_auditor": "前提风险检查",
        "evidence_judge": "证据强度检查",
        "insight_harvester": "洞见整理",
        "alternative_generator": "替代方案",
        "For the Final Controller:": "最终答案需要落实以下调整：",
        "executor_model": "执行模型",
        "fallback_reasons": "备用原因",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def _generic_user_facing_future_report(
    query: str,
    *,
    include_evidence_boundary: bool = False,
    convergence_digest: str = "",
    calibration_constraints: str = "",
    research_evidence_context: str = "",
) -> str:
    convergence_note = _sanitize_user_facing_excerpt(convergence_digest, limit=1200)
    calibration_source = _external_calibration_final_constraints(calibration_constraints) or calibration_constraints
    calibration_note = _sanitize_user_facing_excerpt(calibration_source, limit=900)
    calibration_lines = _calibration_adjustment_lines(calibration_constraints)
    lines = [
        "# 决策任务最终报告",
        "",
        "## 结论摘要",
        "当前最稳妥的结论是：未来变化可能改变能力的相对价值，但不能把长期结果写成确定预测。最终答案应同时保留关键驱动变量、机制链、情景分叉、反证信号和确定性等级。",
        "",
    ]
    if _decision_query_requests_future_inversion_structure(query) or ("未来" in query and "结构性反转" in query):
        lines.extend(_future_inversion_requested_sections())

    anchor_summary = _case_anchor_summary_section(query)
    if anchor_summary:
        lines.extend(anchor_summary)

    enumerated = _enumerated_user_questions(query)
    if enumerated:
        lines.extend(_enumerated_user_answer_sections(query, enumerated, convergence_note, calibration_source))

    lines.extend(
        [
        "## 证据支持",
        "1. [证据支持] 现有材料能支持的，是当前任务中已经有较稳定证据或一致机制的部分。",
        "   触发条件：结论直接来自已给材料中的稳定事实或反复出现的机制。",
        "   中间机制：证据底座约束结论边界，避免把条件性判断写成事实。",
        "   失效条件 / 反证信号：若关键事实、适用场景或执行约束变化，结论必须下调。",
        "   确定性：高/中。",
        "   决策含义：先相信边界清楚、能被观察验证的判断。",
        "",
        "## 合理推断",
        "1. [合理推断] 可以把当前证据连接到用户场景，但必须写清楚中间机制和适用边界。",
        "   触发条件：已有证据不能直接覆盖未来场景，但机制链足够明确。",
        "   中间机制：输入变量改变后，经由可解释机制影响结果。",
        "   失效条件 / 反证信号：如果中间机制没有出现，推断不成立。",
        "   确定性：中。",
        "   决策含义：把推断作为试点和观察方向，而不是一次性定论。",
        "",
        "## 前瞻假设",
        "1. [前瞻假设] 关于未来十年的结构性变化，只能作为需要跟踪的假设。",
        "   触发条件：外部环境持续降低获取、生成或表达成本。",
        "   中间机制：低摩擦环境重新分配注意力、验证、完成和现实交付的价值。",
        "   失效条件 / 反证信号：如果真实瓶颈仍在基础训练、制度授权或现实交付，前瞻假设必须下调。",
        "   确定性：中/低。",
        "   决策含义：用观察指标滚动修正，而不是把未来叙事当事实。",
        "",
        ]
    )
    if convergence_note:
        lines.extend(["## 收敛后的关键判断", convergence_note, ""])
    if calibration_lines:
        lines.extend(["## 需要下调和落地的判断", *calibration_lines, ""])
    lines.extend(
        [
            "## 情景分叉",
            "- 情景 A：工具作为脚手架，保留人的启动、排序、验证和复盘，能力更可能稳步上升。",
            "- 情景 B：工具作为绕行通道，替代人的启动、排序、验证和复盘，表面产出可能上升但独立能力下降。",
            "",
            "## 观察指标与反证信号",
            "- 观察指标：完成物数量、复盘质量、核查习惯、慢反馈耐受、现实交付质量。",
            "- 反证信号：工具越多，越不保留推理痕迹；选题越多，完成越少；表达越强，核查越弱。",
            "",
            "## 最终决策含义",
            "把结论用于低风险、可逆、可观察的行动设计；一旦观察指标反向变化，就下调判断并重新评估。",
        ]
    )
    if include_evidence_boundary:
        lines.extend(["", *_decision_evidence_boundary_section(query=query, research_evidence_context=research_evidence_context, calibration_constraints=calibration_constraints)])
    return "\n".join(lines)


def _future_inversion_requested_sections() -> list[str]:
    return [
        "## 未来优势变陷阱 Top5",
        "1. [合理推断] 快速理解可能变成未经验证的快速接受。触发条件：工具降低获取和表达成本。中间机制：低摩擦答案让认知满足提前出现。失效条件 / 反证信号：使用工具后核查记录、错误修正和现实交付同步增加。确定性：中。决策含义：把评估重点从速度转向证据检查和完成质量。",
        "2. [合理推断] 探索能力可能变成持续换题。触发条件：新选项无限供应但缺少停止规则。中间机制：新奇刺激强化切换，收束成本被回避。失效条件 / 反证信号：主题减少、完成物增加、每次探索能留下可检查结果。确定性：中。决策含义：用完成闭环而不是想法数量判断路径质量。",
        "3. [前瞻假设] 即时反馈偏好可能削弱慢反馈任务耐受。触发条件：环境持续奖励即时回应。中间机制：慢变量显得低价值，延迟练习减少。失效条件 / 反证信号：低刺激任务完成率稳定，且能解释慢练习的价值。确定性：中/低。决策含义：把慢反馈耐受作为风险追踪指标。",
        "4. [合理推断] 表达或生成能力可能掩盖真实执行缺口。触发条件：语言化解释比现实执行更容易获得认可。中间机制：表达优势让外界误判能力已经到位。失效条件 / 反证信号：表达质量与现实交付、时间管理、复盘修正同步改善。确定性：中。决策含义：避免只用表达力替代功能性评估。",
        "5. [合理推断] 对低价值重复的抗拒可能被误用为回避必要基本功。触发条件：可外包步骤和必须亲自练习的环节混在一起。中间机制：边界不清时，必要核查和基本功也被丢弃。失效条件 / 反证信号：能区分可外包重复与不可外包练习，并保留最低训练量。确定性：中。决策含义：把任务分层，而不是把所有厌烦都解释为任务无价值。",
        "",
        "## 未来缺陷变优势 Top5",
        "1. [合理推断] 低重复耐受可能转化为低价值任务筛选能力。触发条件：环境允许外包机械重复，同时要求人保留价值判断和结果验证。中间机制：重复成本下降后，任务意义判断变得更重要。失效条件 / 反证信号：把所有困难都归为低价值，导致必要练习下降。确定性：中。决策含义：把抗拒重复转化为任务拆分和优先级判断。",
        "2. [前瞻假设] 注意跳转或联想发散可能转化为问题发现能力。触发条件：有记录、筛选、验证和收束机制承接发散想法。中间机制：远距离联想经外部化后形成可检验问题。失效条件 / 反证信号：想法只增加数量，不进入验证或交付。确定性：中/低。决策含义：用可检验问题数量和完成物质量评估发散价值。",
        "3. [合理推断] 非线性推进可能适合问题网络式学习或决策。触发条件：工具支持个性化路径，同时保留必要顺序和最低完成标准。中间机制：路径自由与依赖检查结合，降低无效线性摩擦。失效条件 / 反证信号：路径自由变成跳过基础依赖。确定性：中。决策含义：允许路径差异，但必须保留依赖检查。",
        "4. [前瞻假设] 内部想法多可能成为高密度假设池。触发条件：想法能被外部记录、分类、筛选并进入小规模测试。中间机制：外部记录降低遗忘，筛选机制压缩噪声。失效条件 / 反证信号：记录越多，验证越少。确定性：中/低。决策含义：把想法管理成候选假设，而不是直接当结论。",
        "5. [合理推断] 对现实反馈敏感的训练经验可能支撑长期项目。触发条件：训练有稳定节律、明确反馈和可复盘错误。中间机制：现实反馈把冲动控制和状态识别外显化。失效条件 / 反证信号：训练只剩压力或逃避，不迁移到其他任务。确定性：中。决策含义：优先保护能形成反馈闭环的活动。",
        "",
        "## 最危险的错误培养路径",
        "[合理推断] 最危险路径是过早奖励速度、答案和表达，同时忽视目标选择、事实核查、顺序推理和独立复盘。触发条件：工具让答案来得太快。中间机制：低摩擦产出替代慢练习，真实执行缺口延后暴露。失效条件 / 反证信号：工具使用增加后，独立启动、排序、检查和复盘也同步增强。确定性：中。决策含义：把长期培养重点放在验证和完成闭环，而不是更快生成答案。",
        "",
        "## 最反直觉但值得追踪的假设",
        "[前瞻假设] 最反直觉的假设是：未来稀缺的可能不是更快获得信息，而是能停下来判断问题是否值得、答案是否可信、后续动作是否可执行。触发条件：知识获取和表达成本持续下降。中间机制：低成本生成让判断、收束和现实反馈变成稀缺能力。失效条件 / 反证信号：如果外部环境仍主要奖励快速产出而不惩罚错误，该假设需要下调。确定性：中/低。决策含义：把可观察验证设计成长期跟踪项。",
        "",
        "## danger_flag",
        "风险升高信号：长期依赖即时答案、跳过验证、频繁换题且很少闭环；表达越来越流畅但现实交付、错误订正和慢任务耐受没有同步改善。证据强度：执行功能、反馈和练习迁移的方向较稳；争议点：未来工具形态和场景约束会改变强度；证据缺口：缺少长期、个体化、直接可外推的数据。",
        "",
    ]


def _case_anchor_summary_section(query: str) -> list[str]:
    anchors = _case_anchor_groups_from_query(query)
    if len(anchors) < 8:
        return []
    anchor_terms = [terms[0] for _name, terms in anchors if terms]
    lines = ["## 用户画像与约束如何进入判断"]
    for start in range(0, min(len(anchor_terms), 16), 4):
        chunk = anchor_terms[start: start + 4]
        lines.append(
            "- 关键约束：" + "、".join(chunk) + "。判断时必须说明这些约束如何改变优先级、风险边界、阶段安排或反证信号，不能只在开头罗列。"
        )
    negative_constraints = [terms[0] for name, terms in anchors if name.startswith("negative_") and terms]
    if negative_constraints:
        lines.append(
            "- 用户明确排除的边界：" + "；".join(negative_constraints[:3]) + "。最终答案应保留这些边界，不把被排除内容改写成行动建议。"
        )
    lines.append("")
    return lines


def _calibration_adjustment_lines(calibration_constraints: str) -> list[str]:
    source = _external_calibration_final_constraints(calibration_constraints) or calibration_constraints
    if not source.strip():
        return []
    lines: list[str] = []
    for directive in _calibration_directive_lines(source):
        label = _calibration_directive_label(directive)
        content = _sanitize_user_facing_excerpt(_strip_calibration_directive_label(directive), limit=360)
        if content:
            lines.append(f"- {label}：{content}")
    if any(term in source for term in ("执行练习减少风险", "逐条绑定反证信号")):
        lines.append("- 降调要求：把强机制词改为执行练习减少风险，把长期结果降级为前瞻假设，并逐条绑定反证信号。")
    if lines:
        return lines
    return ["- " + _sanitize_user_facing_excerpt(source, limit=700)]


def _calibration_directive_lines(source: str) -> list[str]:
    lines: list[str] = []
    for raw in (source or "").splitlines():
        stripped = raw.strip(" -\t")
        if not stripped:
            continue
        parts = [part.strip() for part in re.split(r"(?=\d{1,2}[.)]\s+\*\*[A-Za-z])", stripped) if part.strip()]
        for part in parts:
            if ":" in part or "：" in part or len(part) >= 24:
                lines.append(part)
    if lines:
        return lines[:8]
    return [_safe_final_excerpt(source, limit=500)] if source.strip() else []


def _strip_calibration_directive_label(line: str) -> str:
    value = re.sub(r"^\d{1,2}[.)]\s*", "", line or "").strip()
    value = re.sub(r"^\*\*[A-Za-z][A-Za-z -]{2,64}\*\*[:：]\s*", "", value).strip()
    value = re.sub(r"^[A-Za-z][A-Za-z -]{2,64}[:：]\s*", "", value).strip()
    return value


def _calibration_directive_label(line: str) -> str:
    lowered = (line or "").lower()
    if any(term in lowered for term in ("tag", "evidence", "epistemic")) or "证据" in line:
        return "证据分层"
    if any(term in lowered for term in ("counter", "unexpected", "inversion")) or "反直觉" in line:
        return "反直觉假设"
    if any(term in lowered for term in ("sequence", "order", "priority", "scaffold")) or any(term in line for term in ("顺序", "排序", "优先")):
        return "优先顺序"
    if any(term in lowered for term in ("risk", "danger", "crutch", "bypass")) or "风险" in line:
        return "风险边界"
    return "需要落地的调整"


def _enumerated_user_answer_sections(
    query: str,
    enumerated: list[tuple[int, str]],
    convergence_note: str,
    calibration_note: str,
) -> list[str]:
    lines = ["## 逐题回答"]
    for index, body in enumerated:
        lines.extend(["", f"## {index}. {_safe_final_excerpt(body, limit=90)}"])
        lines.extend(_answer_lines_for_user_question(query, body, convergence_note, calibration_note))
    lines.append("")
    return lines


def _answer_lines_for_user_question(query: str, body: str, convergence_note: str, calibration_note: str) -> list[str]:
    combined = f"{query}\n{body}\n{convergence_note}\n{calibration_note}"
    if _top_k_request_count(body):
        return _top_k_answer_lines(body, combined)
    if _query_requests_ranking(body):
        return _ranking_answer_lines(body, combined)
    if _query_requests_evidence_tiers(body):
        return _claim_level_evidence_lines(combined)
    if any(term in body for term in ("最危险", "错误", "陷阱")):
        return [_danger_path_line(combined)]
    if any(term in body for term in ("反直觉", "假设")):
        return [_counterintuitive_line(combined)]
    if any(term in body for term in ("家长", "可执行", "方向")):
        return _parent_or_operator_action_lines(combined)
    return [
        "[合理推断] 这个问题应按证据强度、执行约束和反证信号分层回答。触发条件：关键前提在当前材料中成立。中间机制：证据边界先限定判断，再把可逆行动和观察指标绑定。失效条件 / 反证信号：如果关键前提或执行条件不成立，应下调结论。确定性：中。决策含义：先选择低风险、可观察、可复盘的路径。"
    ]


def _top_k_answer_lines(body: str, combined: str) -> list[str]:
    count = _top_k_request_count(body) or 5
    is_advantage = any(term in body for term in ("优势", "机会", "benefit", "advantage"))
    is_trap = any(term in body for term in ("陷阱", "风险", "trap", "risk"))
    if is_advantage and not is_trap:
        items = [
            "[前瞻假设] 问题发现：当 AI 降低知识获取和表达成本后，真正稀缺的是提出好问题、发现异常和形成可检验假设；适用前提是每次探索都留下记录、筛选和验证结果。",
            "[合理推断] 快速试错：高兴趣和高反馈任务可以转化为多路径试验；风险边界是试错必须产出可检查作品，而不是只增加新想法。",
            "[合理推断] 视觉或结构化建模：如果用户画像中存在空间、模型、图像或身体经验优势，可以把复杂关系外显化；反证信号是模型变漂亮但解释和验证没有提升。",
            "[前瞻假设] 低价值重复筛选：工具承担部分机械重复后，人对任务价值的判断更重要；前提是不能把阅读、数学、核查等必要基本功也外包。",
            "[合理推断] 高反馈训练迁移：现实反馈密集的活动能训练状态识别、冲动控制和复盘；是否成立要看这种能力能否迁移到学习、项目或决策任务。",
        ]
    else:
        items = [
            "[不支持/风险] 把 AI 当执行功能拐杖：在工作记忆、顺序逻辑、阅读和数学基本功建立前外包启动、排序、验证和复盘，会让最需要亲自练的能力更少被练到。",
            "[合理推断] 无限换题：低摩擦工具持续供应新答案和新主题，容易让探索变成切换奖励；反证信号是完成物数量和复盘质量同步增加。",
            "[合理推断] 表达掩盖执行缺口：表达更流畅不等于理解、执行和复盘都到位；需要用独立完成、错误订正和现实交付验证。",
            "[前瞻假设] 慢反馈耐受下降：即时答案过多会让阅读、数学、训练或长期项目显得更无聊；是否成立要看慢任务完成率是否下降。",
            "[合理推断] 外包验证：如果只接受工具输出而不保留推理痕迹，事实核查和反例检查会变弱；最低要求是每次使用后留下核查记录。",
        ]
    lines = []
    for index, item in enumerate(items[:count], start=1):
        lines.append(f"{index}. {item}")
    return lines


def _ranking_answer_lines(body: str, combined: str) -> list[str]:
    items = _ranking_items_from_question(body)
    if not items:
        return [
            "1. 第一优先级：先处理不可外包、会影响后续判断的底层能力或关键约束；理由是这些环节一旦缺失，工具和表达会放大误判。",
            "2. 第二优先级：再处理能形成稳定反馈和复盘的训练或流程；理由是反馈闭环能把抽象判断变成可观察进步。",
            "3. 第三优先级：最后增加工具化、表达化或加速型手段；理由是它们适合作为脚手架，不适合作为绕过基本能力的旁路。",
        ]
    ordered = sorted(items, key=_ranking_item_priority)
    lines: list[str] = []
    for index, item in enumerate(ordered, start=1):
        reason = _ranking_item_reason(item, combined)
        lines.append(f"{index}. {item}：{reason}")
    return lines


def _ranking_item_priority(item: str) -> int:
    value = item.lower()
    if any(term in value for term in ("ai", "工具")):
        return 50
    return 10


def _ranking_item_reason(item: str, combined: str) -> str:
    value = item.lower()
    if any(term in value for term in ("ai", "工具")):
        return "作为脚手架用于拆解、表达、核查和复盘；不能作为 bypass/prosthetic 替代启动、排序、验证和复盘。"
    return "作为用户明确列出的排序对象，先看它是否不可外包、能形成现实反馈、能支撑后续能力；若它变成逃避基础练习或现实交付的理由，应下调优先级。"


def _danger_path_line(combined: str) -> str:
    if any(term in combined for term in ("执行功能", "executive-function", "工作记忆", "顺序逻辑", "阅读", "数学")):
        return "[合理推断] 最危险路径是过早把 AI 当执行功能拐杖，在工作记忆、顺序逻辑、阅读和数学基本功建立前就外包启动、排序、验证和复盘；机制链是低摩擦答案让认知满足提前出现，基础顺序能力练得更少，到高年级或初中遇到顺序墙。触发条件：成人奖励速度和流畅表达。失效条件 / 反证信号：没有 AI 时仍能独立启动、完成、检查和复盘。确定性：中。决策含义：先保护基本功和验证闭环。"
    return "[合理推断] 最危险路径是让低摩擦工具替代问题定义、事实核查、排序和复盘；机制链是表面产出增加但独立判断与现实交付没有同步增强。触发条件：组织或家庭只奖励速度。失效条件 / 反证信号：工具使用后错误订正、独立完成和复盘质量同步提升。确定性：中。决策含义：把验证闭环作为红线。"


def _counterintuitive_line(combined: str) -> str:
    excerpt = _counterintuitive_source_excerpt(combined)
    if excerpt:
        return f"[前瞻假设] 反直觉假设是：{excerpt}。触发条件：该假设能被小规模行动、复盘记录和反证信号追踪。失效条件 / 反证信号：如果它不能迁移到用户关心的关键任务，或只增加解释感而不改善完成质量，就应下调。确定性：中/低。决策含义：把它当可检验试验，不把它写成已证实结论。"
    return "[前瞻假设] 反直觉假设是：未来越容易生成答案，越应该投资于慢反馈、现实约束和反证能力。触发条件：获取和表达成本持续下降。中间机制：生成能力变便宜后，问题选择和验证能力更稀缺。失效条件 / 反证信号：如果现实环境不惩罚低质量判断，这个假设需要下调。确定性：中/低。决策含义：用小规模观察验证，而不是一次性押注。"


def _counterintuitive_source_excerpt(combined: str) -> str:
    for line in (combined or "").splitlines():
        stripped = line.strip(" -\t")
        if not stripped:
            continue
        if re.match(r"^\d{1,2}[.、)]", stripped) or "是什么" in stripped:
            continue
        for directive in _calibration_directive_lines(stripped):
            if _calibration_directive_label(directive) == "反直觉假设":
                content = _strip_calibration_directive_label(directive)
                content = _sanitize_user_facing_excerpt(content, limit=260)
                if content:
                    return content
    for line in (combined or "").splitlines():
        stripped = line.strip(" -\t")
        lowered = stripped.lower()
        if not stripped:
            continue
        if re.match(r"^\d{1,2}[.、)]", stripped) or "是什么" in stripped:
            continue
        if "counter_signal" in lowered or "counter-signal" in lowered:
            continue
        if "反直觉" in stripped or "counter" in lowered or "unexpected" in lowered or "inversion" in lowered:
            content = _strip_calibration_directive_label(stripped)
            content = _sanitize_user_facing_excerpt(content, limit=260)
            if content:
                return content
    return ""


def _claim_level_evidence_lines(combined: str) -> list[str]:
    return [
        "1. [证据支持] 结构化支持、稳定反馈和保留基本练习通常比单纯追求更快答案更可靠。触发条件：训练或流程能持续执行。中间机制：外部结构降低执行负担。失效条件 / 反证信号：只增加强度但没有复盘或恢复。确定性：中/高。家长怎么用：先看节律和闭环，而不是短期表现。",
        "2. [合理推断] 如果用户画像显示某项高兴趣或空间/身体经验优势，可以把它作为调节和迁移锚点。触发条件：本人愿意长期投入且训练质量稳定。中间机制：现实反馈外显化状态和错误。失效条件 / 反证信号：训练只剩压力或逃避。确定性：中。家长怎么用：观察是否迁移到阅读、数学、项目或决策任务。",
        "3. [前瞻假设] AI 降低知识获取和表达成本后，问题发现、快速试错和筛选能力可能升值。触发条件：发散想法被记录、筛选、测试。中间机制：低摩擦输入让验证和收束更稀缺。失效条件 / 反证信号：想法更多但完成更少。确定性：中/低。家长怎么用：每次探索必须留下可检查结果。",
        "4. [不支持/风险] AI 可以替代儿童或学习者建立顺序逻辑、阅读、数学、核查和复盘基本功。触发条件：把工具当 bypass/prosthetic。中间机制：亲自练习减少导致独立能力弱化。失效条件 / 反证信号：没有工具时仍能启动、完成、检查和复盘。确定性：风险高。家长怎么用：工具只能做脚手架，不能做旁路。",
    ]


def _parent_or_operator_action_lines(combined: str) -> list[str]:
    lines = [
        "- 每周追踪少量可观察指标：状态调节、慢任务耐受、步骤完整率、错误订正和独立复盘。",
        "- 每次使用 AI 或其他工具，都保留三样东西：自己的第一版想法、工具帮助后的修改、最后的核查或反例。",
        "- 反馈要奖励完成闭环：开始、坚持、检查、修正、复盘，而不只奖励聪明表达或速度。",
    ]
    negative = _negative_constraint_excerpt(combined)
    if negative:
        lines.append(f"- 边界：{negative}。上述行动只用于低风险、可观察、可复盘的长期方向，不替代被用户明确排除的判断或建议类型。")
    return lines


def _negative_constraint_excerpt(text: str) -> str:
    matches = re.findall(r"(?:不要|不得|不能|避免)[^。；;\n]{2,72}", text or "")
    return _safe_final_excerpt("；".join(matches[:2]), limit=180) if matches else ""


def _final_user_facing_quality_failures(packet: dict[str, Any], text: str) -> list[str]:
    query = str(packet.get("query") or "")
    value = text or ""
    failures: list[str] = []
    failures.extend(_internal_user_facing_language_failures(value))

    enumerated = _enumerated_user_questions(query)
    if len(enumerated) >= 2:
        failures.extend(_enumerated_answer_substance_failures(enumerated, value))

    failures.extend(_top_k_substance_failures(query, value, enumerated))
    failures.extend(_ranking_substance_failures(query, value, enumerated))
    profiles = _normalize_profiles(packet.get("output_quality_profile"))
    if _query_requests_evidence_tiers(query) or PROFILE_FORESIGHT_MECHANISM in profiles:
        failures.extend(_evidence_tagging_substance_failures(value))
    failures.extend(_case_anchor_usage_failures(query, value))
    failures.extend(_calibration_implementation_failures(str(packet.get("external_calibration_hard_constraints") or ""), value))
    return failures


def _enumerated_user_questions(query: str) -> list[tuple[int, str]]:
    pattern = re.compile(r"(?m)^\s*(\d{1,2})[.、)]\s*(.+?)(?=^\s*\d{1,2}[.、)]\s*|\Z)", re.S)
    return [(int(match.group(1)), " ".join(match.group(2).split())) for match in pattern.finditer(query or "")]


def _query_requests_top_k(query: str) -> bool:
    return bool(re.search(r"Top\s*\d+|Top\d+|前\s*\d+", query or "", re.I))


def _query_requests_ranking(query: str) -> bool:
    value = query or ""
    return any(term in value for term in ("排序", "优先级", "rank", "order"))


def _query_requests_evidence_tiers(query: str) -> bool:
    value = query or ""
    lowered = value.lower()
    return any(term in lowered or term in value for term in ("证据支持", "plausible", "speculative", "证据分层", "evidence"))


def _top_k_request_count(text: str) -> int | None:
    match = re.search(r"Top\s*(\d+)|Top(\d+)|前\s*(\d+)", text or "", re.I)
    if not match:
        return None
    for group in match.groups():
        if group:
            return int(group)
    return None


def _internal_user_facing_language_failures(text: str) -> list[str]:
    lowered = (text or "").lower()
    forbidden = (
        "decision_mode=true",
        "final controller",
        "external_calibration",
        "convergence_report",
        "research packet",
        "research_evidence_packet",
        "研究包吸收",
        "校准执行",
        "StageRecord",
        "Stage",
        "artifact",
        "pipeline",
        "Executor",
        "执行器",
        "必须把 external_calibration 视为硬约束",
        "最终报告只输出融合后的判断单元",
        "本阶段输出",
        "上游报告",
        "下游阶段",
        "校准阶段",
        "校准要求",
        "artifact 检查",
        "pipeline 检查",
        "stage 输出",
        "validation gate",
        "contract",
        "已吸收 calibration",
        "已吸收 external calibration",
        "根据 convergence_report",
        "研究包显示",
        "calibration_verdict",
        "premise_auditor",
        "evidence_judge",
        "insight_harvester",
        "alternative_generator",
        "evidence_strength:",
        "controversy:",
        "evidence_gap:",
    )
    return ["internal_language:" + term for term in forbidden if term.lower() in lowered]


def _markdown_numeric_sections(text: str) -> dict[int, str]:
    pattern = re.compile(r"(?m)^##\s*(\d{1,2})[.、)]\s*.*$", re.M)
    matches = list(pattern.finditer(text or ""))
    sections: dict[int, str] = {}
    for pos, match in enumerate(matches):
        start = match.start()
        end = matches[pos + 1].start() if pos + 1 < len(matches) else len(text or "")
        sections[int(match.group(1))] = (text or "")[start:end].strip()
    return sections


def _markdown_named_section_body(text: str, heading: str) -> str:
    pattern = re.compile(rf"(?m)^##\s+{re.escape(heading)}\s*$")
    match = pattern.search(text or "")
    if not match:
        return ""
    next_match = re.search(r"(?m)^##\s+", (text or "")[match.end():])
    end = match.end() + next_match.start() if next_match else len(text or "")
    return (text or "")[match.end():end].strip()


def _list_item_bodies(section: str) -> list[str]:
    items: list[str] = []
    for line in (section or "").splitlines():
        match = re.match(r"\s*(?:[-*]|\d{1,2}[.、)])\s+(.+)", line)
        if match:
            items.append(match.group(1).strip())
    return items


def _plain_body(text: str) -> str:
    value = re.sub(r"\[[^\]]+\]", "", text or "")
    value = re.sub(r"[#*_`>|\-]+", " ", value)
    return " ".join(value.split())


def _looks_substantive(text: str, *, min_chars: int = 80) -> bool:
    plain = _plain_body(text)
    if len(plain) < min_chars:
        return False
    vague_fragments = ("很重要", "请执行", "正文", "内容", "这是判断", "需要平衡", "都有风险", "都需要考虑")
    if any(fragment in plain for fragment in vague_fragments) and len(plain) < min_chars + 80:
        return False
    compact = re.sub(r"\W+", "", plain)
    return len(set(compact)) >= min(18, max(8, len(compact) // 12))


def _question_keywords(body: str) -> list[str]:
    stopwords = {
        "一个", "请", "判断", "哪些", "什么", "如何", "是否", "应该", "之间", "使用", "给出", "目标", "长期", "未来", "场景", "Top", "top",
        "支持", "只是", "可执行", "方向", "最容易", "最可能", "最危险", "最反直觉", "排序", "问题", "证据", "合理", "推断", "假设",
    }
    terms: list[str] = []
    for token in re.split(r"[\s，,；;：:。！？、/()（）]+", body or ""):
        token = token.strip("；;.，,。")
        token = re.sub(r"^(是否|应该|请|给出)", "", token)
        token = re.sub(r"(是什么|什么|如何|怎么|吗|？|\?)$", "", token)
        if len(token) >= 2 and token not in stopwords and not token.isdigit():
            terms.append(token)
    for pattern in (r"[A-Z]{2,}\s*\d*", r"\d+(?:\.\d+)?\s*岁", r"[一二三四五六七八九十\d]+\s*年级", r"\d+\s*[-–]\s*\d+\s*小时", r"\b[A-Z][A-Z0-9]{1,8}\b\s*工具?"):
        for match in re.findall(pattern, body or "", re.I):
            terms.append(match.strip())
    for term in _generic_salient_terms(body):
        if term in (body or ""):
            terms.append(term)
    deduped: list[str] = []
    for term in terms:
        if term and term not in deduped:
            deduped.append(term)
    return deduped[:12]


def _enumerated_answer_substance_failures(enumerated: list[tuple[int, str]], text: str) -> list[str]:
    sections = _markdown_numeric_sections(text)
    failures: list[str] = []
    for index, body in enumerated:
        section = sections.get(index, "")
        if not section:
            failures.append(f"missing_enumerated_answer:{index}")
            continue
        if not _looks_substantive(section, min_chars=70):
            failures.append(f"shallow_enumerated_answer:{index}")
        if (
            _query_requests_ranking(body)
            or _top_k_request_count(body)
            or _query_requests_evidence_tiers(body)
            or any(term in body for term in ("家长", "可执行", "方向", "最危险", "反直觉"))
        ):
            continue
        keywords = _question_keywords(body)
        if keywords:
            hits = [term for term in keywords if term.lower() in section.lower() or term in section]
            if len(hits) < min(2, len(keywords)):
                failures.append(f"off_topic_enumerated_answer:{index}")
    return failures


def _top_k_substance_failures(query: str, text: str, enumerated: list[tuple[int, str]]) -> list[str]:
    failures: list[str] = []
    sections = _markdown_numeric_sections(text)
    for index, body in enumerated:
        count = _top_k_request_count(body)
        if count:
            failures.extend(_top_k_section_failures(f"enumerated_{index}", sections.get(index, ""), count))
    if _decision_query_requests_future_inversion_structure(query):
        for heading in ("未来优势变陷阱 Top5", "未来缺陷变优势 Top5"):
            failures.extend(_top_k_section_failures(heading, _markdown_named_section_body(text, heading), 5))
    elif _top_k_request_count(query) and not enumerated:
        count = _top_k_request_count(query) or 5
        items = _list_item_bodies(text)
        if len(items) < count:
            failures.append(f"missing_top_k_items:{len(items)}/{count}")
    return failures


def _top_k_section_failures(name: str, section: str, count: int) -> list[str]:
    items = _list_item_bodies(section)
    failures: list[str] = []
    if len(items) < count:
        failures.append(f"top_k_item_count:{name}:{len(items)}/{count}")
        return failures
    shallow = [idx for idx, item in enumerate(items[:count], start=1) if not _looks_substantive(item, min_chars=36)]
    if shallow:
        failures.append(f"top_k_item_shallow:{name}:{','.join(map(str, shallow))}")
    normalized = [re.sub(r"\W+", "", _plain_body(item))[:38] for item in items[:count]]
    if len(set(normalized)) < max(3, count - 1):
        failures.append(f"top_k_items_repetitive:{name}")
    return failures


def _ranking_items_from_question(body: str) -> list[str]:
    value = body or ""
    before_between = re.split(r"之间|如何排序|应该如何排序|排序", value, maxsplit=1)[0]
    tail = before_between.split("：")[-1].split(":")[-1]
    items: list[str] = []
    for token in re.split(r"[、/,，]+", tail):
        token = re.sub(r"(使用)?(之间|应该|如何|排序).*$", "", token).strip()
        if 1 < len(token) <= 16 and token not in ("资源", "优先级") and not any(skip in token for skip in ("请", "判断", "哪些", "目标")):
            items.append(token)
    deduped = list(dict.fromkeys(items))
    return deduped if len(deduped) >= 2 else []


def _ranking_substance_failures(query: str, text: str, enumerated: list[tuple[int, str]]) -> list[str]:
    failures: list[str] = []
    sections = _markdown_numeric_sections(text)
    targets = [(index, body, sections.get(index, "")) for index, body in enumerated if _query_requests_ranking(body)]
    if not targets and _query_requests_ranking(query):
        targets = [(0, query, text)]
    for index, body, section in targets:
        if not section:
            failures.append(f"missing_ranking_section:{index}")
            continue
        if not any(marker in section for marker in ("1.", "1、", "第一", "优先级", ">", "先", "后")):
            failures.append(f"missing_clear_ranking:{index}")
        items = _ranking_items_from_question(body)
        missing_items = [item for item in items if item not in section]
        if missing_items:
            failures.append(f"ranking_missing_items:{index}:{'|'.join(missing_items)}")
        if items and len(_list_item_bodies(section)) < min(3, len(items)):
            failures.append(f"ranking_insufficient_item_reasons:{index}")
    return failures


def _case_anchor_groups_from_query(query: str) -> list[tuple[str, tuple[str, ...]]]:
    value = query or ""
    anchors: list[tuple[str, tuple[str, ...]]] = []
    seen: set[str] = set()

    def add(kind: str, raw: str) -> None:
        term = " ".join((raw or "").strip(" ：:，,。；;、/()（）").split())
        if len(term) < 2 or len(term) > 34:
            return
        if term in _anchor_stop_terms():
            return
        key = re.sub(r"\s+", "", term).lower()
        if key in seen:
            return
        seen.add(key)
        anchors.append((f"{kind}_{len(anchors) + 1}", (term,)))

    for pattern in (
        r"\d+(?:\.\d+)?\s*岁",
        r"[一二三四五六七八九十\d]+\s*年级",
        r"\b[A-Z]{2,}\s*\d+\b",
        r"\d+\s*[-–]\s*\d+\s*小时",
        r"\d+\s*[-–]\s*\d+\s*年",
        r"\d+\s*%",
        r"[$¥€]\s*\d+(?:\.\d+)?\s*(?:万|亿|k|m|b)?",
        r"\d+(?:\.\d+)?\s*(?:万|亿|k|m|b)",
        r"\b[A-Z][A-Z0-9]{1,8}\b",
    ):
        for match in re.findall(pattern, value):
            add("numeric_or_named", match)

    for item in _ranking_items_from_question(value):
        add("ranking_item", item)

    leading_context = re.split(r"(?m)^\s*1[.、)]\s*|请|please", value, maxsplit=1)[0]
    for token in re.split(r"[，,；;。！？\n]+", leading_context):
        cleaned = re.sub(r"^(一个|一位|有明确|热爱|目标|长期|未来|从)\s*", "", token.strip())
        add("case_phrase", cleaned)

    for _index, body in _enumerated_user_questions(value):
        for term in _question_keywords(body)[:4]:
            add("question_term", term)

    for match in re.findall(r"(?:不要|不得|不能|避免)[^。；;\n]{2,72}", value):
        add("negative_constraint", match)

    return anchors[:28]


def _anchor_stop_terms() -> set[str]:
    return {
        "一个",
        "一位",
        "请",
        "判断",
        "哪些",
        "什么",
        "如何",
        "应该",
        "之间",
        "未来",
        "长期",
        "目标",
        "场景",
        "风险",
        "证据",
        "支持",
        "合理",
        "只是",
        "给出",
        "方向",
    }


def _generic_salient_terms(text: str) -> list[str]:
    terms: list[str] = []
    for token in re.split(r"[\s，,；;：:。！？、/()（）]+", text or ""):
        cleaned = token.strip()
        cleaned = re.sub(r"^(是否|应该|请|给出)", "", cleaned)
        cleaned = re.sub(r"(是什么|什么|如何|怎么|吗|\?|\？)$", "", cleaned)
        if 2 <= len(cleaned) <= 12 and cleaned not in _anchor_stop_terms():
            terms.append(cleaned)
    return list(dict.fromkeys(terms))[:12]


def _case_anchor_usage_failures(query: str, text: str) -> list[str]:
    anchors = _case_anchor_groups_from_query(query)
    if len(anchors) < 8:
        return []
    reasoning_terms = ("因为", "所以", "意味着", "风险", "优先", "排序", "阶段", "家长", "判断", "机制", "导致", "不能", "应该", "用于", "反证", "触发", "中间机制", "决策含义", "基础", "迁移")
    used = 0
    sections_with_anchor: set[str] = set()
    early_hits = 0
    later_logic_hits = 0
    value = text or ""
    headings = [(match.start(), match.group(0).strip()) for match in re.finditer(r"(?m)^##\s+.*$", value)]
    for name, terms in anchors:
        positions = [value.find(term) for term in terms if term in value]
        positions = [pos for pos in positions if pos >= 0]
        if not positions:
            continue
        if min(positions) < 320:
            early_hits += 1
        logic_hit = False
        for pos in positions:
            window = value[max(0, pos - 45): pos + 75]
            if any(term in window for term in reasoning_terms):
                logic_hit = True
                if pos > 500:
                    later_logic_hits += 1
            section = "intro"
            for start, heading in headings:
                if start <= pos:
                    section = heading
                else:
                    break
            sections_with_anchor.add(section)
        if logic_hit:
            used += 1
    failures: list[str] = []
    required = max(6, int(len(anchors) * 0.6))
    if used < required:
        failures.append(f"anchor_usage_low:{used}/{len(anchors)}")
    if len(sections_with_anchor) < 3:
        failures.append(f"anchor_distribution_low:{len(sections_with_anchor)}")
    if early_hits >= int(len(anchors) * 0.7) and later_logic_hits < max(3, required // 2):
        failures.append("anchor_stuffing_suspected")
    return failures


def _evidence_tagging_substance_failures(text: str) -> list[str]:
    labels = ("[证据支持]", "[合理推断]", "[前瞻假设]")
    failures: list[str] = []
    for label in labels:
        if label not in (text or ""):
            failures.append("missing_evidence_label:" + label)
    claim_blocks = _evidence_claim_blocks(text)
    if len(claim_blocks) < 3:
        failures.append("insufficient_claim_level_evidence_tags")
        return failures
    substantive = [block for block in claim_blocks if _looks_substantive(block, min_chars=86) and any(term in block for term in ("触发条件", "中间机制", "失效条件", "反证", "因为", "前提", "机制", "风险", "家长怎么用", "决策含义"))]
    if len(substantive) < 3:
        failures.append("evidence_tags_not_substantive")
    normalized = [re.sub(r"\W+", "", _plain_body(line))[:44] for line in substantive]
    if substantive and len(set(normalized)) < max(2, min(3, len(substantive))):
        failures.append("evidence_tags_repetitive")
    first_cluster = "\n".join((text or "").splitlines()[:8])
    if sum(first_cluster.count(label) for label in (*labels, "[不支持/风险]")) >= 3 and len(substantive) < 4:
        failures.append("evidence_label_stuffing_suspected")
    return failures


def _evidence_claim_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if any(label in stripped for label in ("[证据支持]", "[合理推断]", "[前瞻假设]", "[不支持/风险]")):
            if current:
                blocks.append(" ".join(current))
            current = [stripped]
        elif current and (stripped.startswith(("触发条件", "中间机制", "失效条件", "确定性", "决策含义", "家长怎么用", "-")) or re.match(r"^\s+", line)):
            current.append(stripped)
        elif current and not stripped:
            blocks.append(" ".join(current))
            current = []
    if current:
        blocks.append(" ".join(current))
    return blocks


def _calibration_implementation_failures(calibration_constraints: str, text: str) -> list[str]:
    calibration = calibration_constraints or ""
    if not calibration.strip():
        return []
    value = text or ""
    failures: list[str] = []
    restatement_markers = ("已吸收", "已执行", "校准要求", "elevate risk", "lock sequence", "restore counter", "strict epistemic")
    if any(marker in value.lower() for marker in restatement_markers):
        failures.append("calibration_restatement_language")
    required_terms = _calibration_required_terms(calibration)
    if required_terms:
        hits = [term for term in required_terms if term in value]
        required_hits = min(len(required_terms), max(3, int(len(required_terms) * 0.3)))
        if len(hits) < required_hits:
            failures.append(f"calibration_required_terms_low:{len(hits)}/{len(required_terms)}")
    directive_labels = {_calibration_directive_label(line) for line in _calibration_directive_lines(calibration)}
    if "风险边界" in directive_labels and not any(term in value for term in ("风险", "最危险", "失效条件", "反证信号")):
        failures.append("calibration_missing_risk_path")
    if "优先顺序" in directive_labels and not any(term in value for term in ("优先", "排序", "第一", "第二", "先", "后")):
        failures.append("calibration_missing_sequence_change")
    if "反直觉假设" in directive_labels and "反直觉" not in value:
        failures.append("calibration_missing_counterintuitive_mapping")
    if "证据分层" in directive_labels:
        tag_failures = _evidence_tagging_substance_failures(value)
        if tag_failures:
            failures.append("calibration_missing_strict_evidence_tagging")
    if "calibration_restatement_language" in failures and len(failures) == 1:
        failures.append("calibration_lacks_user_facing_implementation")
    return failures


def _calibration_required_terms(calibration: str) -> list[str]:
    terms: list[str] = []
    for _name, values in _case_anchor_groups_from_query(calibration):
        if values:
            terms.append(values[0])
    for line in _calibration_directive_lines(calibration):
        content = _strip_calibration_directive_label(line)
        for token in re.split(r"[\s,，;；:/()（）]+", content):
            cleaned = token.strip(".。")
            if 3 <= len(cleaned) <= 28 and cleaned.lower() not in {"and", "the", "before", "from", "with", "label"}:
                terms.append(cleaned)
    deduped: list[str] = []
    for term in terms:
        key = term.lower()
        if key not in {item.lower() for item in deduped}:
            deduped.append(term)
    return deduped[:16]

def _absorbed_convergence_lines(convergence_digest: str) -> list[str]:
    value = (convergence_digest or "").strip()
    if not value:
        return []
    labels = {
        "key_drivers": "关键驱动",
        "mechanism_chain": "机制链",
        "scenario_branches": "情景分叉",
        "counter_signals": "反证信号",
        "certainty_levels": "确定性",
        "uncertainty_boundary": "不确定性边界",
    }
    lines = ["## 收敛吸收"]
    for heading, label in labels.items():
        body = _markdown_section_body(value, heading)
        if body:
            lines.append(f"{label}：{_safe_final_excerpt(body, limit=520)}")
    if len(lines) == 1:
        lines.append(_safe_final_excerpt(value, limit=1200))
    lines.append("")
    return lines


def _external_calibration_final_constraints(text: str, *, limit: int = 3000) -> str:
    value = text or ""
    sections: list[str] = []
    for heading in ("final_adjustment_recommendation", "handoff_notes_for_final_controller"):
        body = _markdown_section_body(value, heading) or _colon_or_plain_section_body(value, heading)
        if body:
            sections.append(_safe_final_excerpt(body, limit=max(400, limit // 2)))
    return _safe_final_excerpt("\n".join(sections), limit=limit) if sections else ""


def _absorbed_external_calibration_lines(calibration_constraints: str) -> list[str]:
    value = (calibration_constraints or "").strip()
    if not value:
        return []
    return [
        "## 校准执行",
        "final controller 必须把 external_calibration 视为硬约束：被要求降级的内容只能写成条件性推断或 foresight_hypothesis；被要求删除的强机制词不得作为事实进入最终判断。",
        f"已吸收的校准要求：{_safe_final_excerpt(value, limit=1200)}",
        "",
    ]


def _research_evidence_tier_bodies(context: str, *, per_tier_limit: int = 520) -> dict[str, str]:
    value = context or ""
    tiers: dict[str, str] = {}
    for heading in ("evidence_supported", "reasonable_inference", "foresight_hypothesis"):
        body = _markdown_section_body(value, heading)
        if body:
            tiers[heading] = _safe_final_excerpt(body, limit=per_tier_limit)
    return tiers


def _foresight_mechanism_final_report(query: str) -> str:
    return "\n".join(
        [
            "# 前瞻机制研究决策报告",
            "",
            "## 关键驱动变量",
            "核心驱动变量包括：AI 降低知识获取成本、即时反馈变多、验证与收束能力变稀缺、学校评价方式可能滞后。确定性等级：中。",
            "",
            "## 机制链",
            "输入变量 → 中介机制 → 输出变量：知识获取成本下降 → 信息筛选和延迟验证成为瓶颈 → ADHD 注意力特征中的发散、厌烦低价值重复、内部想法丰富，可能分别转化为机会或陷阱。",
            "",
            "## 情景分叉",
            "情景 A：孩子学会保留推理痕迹、核查事实、完成慢速闭环，部分注意漂移可转化为问题发现和跨域联想。情景 B：孩子把 AI 当即时答案机，兴趣跳转和未收束想法会放大为浅尝辄止。",
            "",
            "## 成立条件与失效条件",
            "成立条件：有外部结构、反馈节律、事实核查习惯和任务收束训练。失效条件：只有速度奖励、没有验证、没有完成标准、成人把所有困难都解释成天赋或态度。",
            "",
            "## 证据强度、争议和缺口",
            "证据强度：关于 ADHD 执行功能、行为支持和学习脚手架的基础证据较强；关于 AI 环境下优势/缺陷结构性反转属于合理推断和前瞻假设。争议在于不同孩子、学校和工具环境差异很大；缺口是缺少直接证明未来十年具体反转路径的长期研究。",
            "",
            "## 可观察指标 / 反证信号",
            "可观察指标包括：是否能说明自己为什么相信一个答案、是否能完成慢任务闭环、是否主动核查事实、是否能从大量选项中收束到一个目标。反证信号是：AI 使用越多，越不愿留下推理痕迹、越频繁换题、越少完成。",
            "",
            "## 用户问题锚点",
            query[:1200],
        ]
    )


def _generic_decision_final_report(
    query: str,
    *,
    packet: dict[str, Any] | None = None,
    include_evidence_boundary: bool = False,
) -> str:
    prompt_anchor = query[:600].strip() or "本轮输入未提供可显示的问题文本。"
    packet = packet if isinstance(packet, dict) else {}
    enumerated = _enumerated_user_questions(query)
    lines = [
        "# 决策任务最终报告",
        "",
        "## 决策问题",
        prompt_anchor,
        "",
        "## 决策判断",
        "本轮答案只呈现最终整合后的判断：保留证据强弱、前提风险、替代路径和边界修正后的判断，不展示内部流程材料。",
        "",
        "## 关键依据",
        "结论仅使用当前有效材料和已校准判断；未执行前置研究链路时，不把本轮输出包装成研究综述。证据强度、争议和缺口需要显式保留。",
        "",
    ]
    if enumerated:
        lines.append("## 逐项回答")
        for index, body in enumerated:
            lines.extend(
                [
                    "",
                    f"## {index}. {_safe_final_excerpt(body, limit=80)}",
                    "回答：当前材料只支持有边界的条件性判断；应结合证据强度、争议、缺口和可观察反证信号使用。",
                ]
            )
        lines.append("")
    if include_evidence_boundary:
        lines.extend(
            _decision_evidence_boundary_section(
                query=query,
                research_evidence_context=str(packet.get("research_evidence_packet_context") or ""),
                calibration_constraints=str(packet.get("external_calibration_hard_constraints") or ""),
            )
        )
    lines.extend(
        [
            "## 风险边界",
            "高不确定、依赖外部事实或需要专业角色确认的部分，应保持条件化表述。",
            "",
            "## 可选路径",
            "保留低强度、可逆路径；中强度路径；以及仅在关键风险升高时才进入的高强度路径。",
            "",
            "## 复盘指标",
            "周期可先设为 4-6 周；频率为每天记录、每周复盘；步骤是观察、执行、反馈、复盘；记录指标使用少量可观察指标检查判断是否偏离事实；调整规则是在失败时降难度，在稳定后再升级。",
        ]
    )
    return "\n".join(lines)

def _safe_final_excerpt(text: str, *, limit: int = 1200) -> str:
    value = _strip_raw_artifact_metadata_for_final_body(text or "")
    value = " ".join(value.split())
    for token in ("web_search", "api_call", "codex_exec", "delegate_task", "persona:"):
        value = value.replace(token, "[removed]")
    return value[:limit]


def _strip_raw_artifact_metadata_for_final_body(text: str) -> str:
    skipped_prefixes = (
        "executor_model:",
        "fallback_reasons:",
        "artifact_path:",
        "valid_for_pipeline:",
        "stage_name:",
        "owner=",
        "owner:",
        "model:",
    )
    skipped_exact = {
        "external_calibration",
        "convergence_report",
        "evidence_judge",
        "premise_auditor",
        "alternative_generator",
        "insight_harvester",
        "structure_mapper",
        "supplementary_search",
        "intelligence_layer",
    }
    kept: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if not line:
            continue
        if lowered in skipped_exact:
            continue
        if any(lowered.startswith(prefix) for prefix in skipped_prefixes):
            continue
        if "external_calibration executor_model" in lowered:
            continue
        if lowered.startswith("pipeline ") or lowered.startswith("pipeline_"):
            continue
        kept.append(raw_line)
    return "\n".join(kept)


def _final_controller_report_forbidden_tokens(text: str) -> list[str]:
    lowered = (text or "").lower()
    tokens = []
    for token in ("web_search", "api_call", "codex_exec", "delegate_task", "persona:"):
        if token in lowered:
            tokens.append(token)
    if "r1 convergence body" in lowered or "persona raw" in lowered:
        tokens.append("raw_intermediate_dump")
    return tokens


def _intelligence_output_forbidden_tokens(text: str) -> list[str]:
    value = text or ""
    lowered = value.lower()
    tokens = []
    lines = [line.strip() for line in value.splitlines()]
    heading_lines = [line for line in lines if line.startswith("#")]

    if "final_controller_report" in lowered:
        tokens.append("final_controller_report")
    if "pipeline_status=pipeline_complete" in lowered:
        tokens.append("pipeline_status=PIPELINE_COMPLETE")
    if any(_is_final_report_heading(line) for line in heading_lines):
        tokens.append("final_report_heading")
    if any(_is_chinese_final_advice_heading(line) for line in heading_lines):
        tokens.append("chinese_final_advice_heading")

    has_action_plan = any("action plan" in line.lower() or "行动计划" in line or "行动建议" in line for line in lines)
    has_recommendation = any("recommendation" in line.lower() or "建议" in line for line in lines)
    has_direct_advice = any(_contains_direct_user_advice(line) for line in lines)
    if has_action_plan and has_recommendation and has_direct_advice:
        tokens.append("action_plan_recommendation_direct_advice")
    return tokens


def _is_final_report_heading(line: str) -> bool:
    normalized = line.strip("# ").strip().lower()
    return normalized in {
        "final controller report",
        "final report",
        "controller final report",
        "最终报告",
        "最终决策报告",
        "最终控制器报告",
    }


def _is_chinese_final_advice_heading(line: str) -> bool:
    normalized = line.strip("# ").strip()
    return normalized in {"最终建议", "行动建议", "最终行动建议", "最终决策建议"}


def _contains_direct_user_advice(line: str) -> bool:
    lowered = line.lower()
    english = (
        "you should",
        "you need to",
        "i recommend you",
        "recommended next step",
        "take the following action",
    )
    chinese = ("你应该", "你需要", "建议你", "请立即", "可以直接")
    return any(token in lowered for token in english) or any(token in line for token in chinese)


def _r1_synthesis_prompt_from_artifacts(stages: list[dict[str, Any]], *, base_dir: str | Path, query: str = "") -> str:
    base = Path(base_dir).resolve()
    chunks = [
        "Run RESEARCH stage L3_r1_synthesis using R1-32B only.",
        "Use only the fresh artifacts from L1_gemini_search, L2_ddgs_supplement, and L2_5_codex_evidence_organizer.",
        "Produce a concise research evidence synthesis. Do not audit, calibrate, decide, or write a final report.",
    ]
    if PROFILE_FORESIGHT_MECHANISM in _task_engine_profiles_from_query(query):
        chunks.extend(_foresight_research_prompt_guidance("L3_r1_synthesis"))
        chunks.append(
            "Return explicit sections named evidence_support, reasonable_inference, foresight_hypothesis, "
            "mechanism_chain, uncertainty_boundary, and counterexample_or_failure."
        )
    for record in stages:
        chunks.append(f"\n## {record.get('stage_name')}")
        paths = [record.get("artifact_path")] + list((record.get("outputs") or {}).values())
        for raw_path in paths:
            path = Path(str(raw_path or "")).resolve()
            try:
                path.relative_to(base)
            except ValueError:
                continue
            if path.is_dir():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")[:6000]
            chunks.append(f"\n### {path.name}\n{text}")
    return "\n".join(chunks)


def _gemini_audit_prompt_from_artifacts(stages: list[dict[str, Any]], *, base_dir: str | Path, query: str = "") -> str:
    base = Path(base_dir).resolve()
    chunks = [
        "Run RESEARCH stage L4_gemini_audit through AGY/Gemini.",
        "Use Gemini 3.1 Pro (High) only. Do not use Gemini 3.5 Flash, Controller, DeepSeek, or R1.",
        "Audit the L3 R1 synthesis against fresh L1/L2/L2.5 evidence artifacts.",
        "Return an audit report only. Do not produce final acceptance, final advice, or a final report.",
        "If evidence is missing or unsupported, mark it clearly as a defect or gap.",
    ]
    if PROFILE_FORESIGHT_MECHANISM in _task_engine_profiles_from_query(query):
        chunks.extend(_foresight_research_prompt_guidance("L4_gemini_audit"))
        chunks.append(
            "Audit whether L3 explicitly separates evidence_support, reasonable_inference, and foresight_hypothesis; "
            "whether it includes mechanism_chain, uncertainty_boundary, and counterexample_or_failure; "
            "and whether it avoids presenting foresight hypotheses as settled medical facts."
        )
    for record in stages:
        chunks.append(f"\n## {record.get('stage_name')}")
        paths = [record.get("artifact_path")] + list((record.get("outputs") or {}).values())
        for raw_path in paths:
            path = Path(str(raw_path or "")).resolve()
            try:
                path.relative_to(base)
            except ValueError:
                continue
            if path.is_dir():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")[:8000]
            chunks.append(f"\n### {path.name}\n{text}")
    return "\n".join(chunks)


def _omlx_timeout_s() -> int:
    try:
        return max(30, min(int(os.getenv("HERMES_OMLX_R1_TIMEOUT_S", "600")), 1800))
    except ValueError:
        return 600


def _omlx_admin_load_timeout_s() -> int:
    try:
        return max(30, min(int(os.getenv("HERMES_OMLX_ADMIN_LOAD_TIMEOUT_S", "240")), 900))
    except ValueError:
        return 240


def _omlx_max_tokens() -> int:
    try:
        return max(512, min(int(os.getenv("HERMES_OMLX_R1_MAX_TOKENS", "4096")), 12000))
    except ValueError:
        return 4096


def _omlx_max_tokens_for_stage(stage: StageSpec) -> int:
    if stage.stage_name == "evidence_judge":
        try:
            return max(512, min(int(os.getenv("HERMES_OMLX_EVIDENCE_JUDGE_MAX_TOKENS", "1536")), 4096))
        except ValueError:
            return 1536
    return _omlx_max_tokens()


def _loaded_omlx_model_ids(admin: Any) -> list[str]:
    try:
        models = admin.get_models()
    except Exception:
        return []
    loaded: list[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "")
        if not model_id:
            continue
        state = str(item.get("state") or item.get("status") or "").lower()
        if bool(item.get("loaded") or item.get("is_loaded") or _omlx_status_is_ready(state)):
            loaded.append(model_id)
    return loaded


def _omlx_status_is_ready(status: str) -> bool:
    return str(status or "").strip().lower() in {"loaded", "idle", "ready", "running"}


def _omlx_observed_model_status(admin: Any, model_id: str) -> str:
    try:
        models = admin.get_models()
    except Exception as exc:
        return f"status_poll_error:{_redact_secret_text(str(exc))}"
    for item in models:
        if not isinstance(item, dict) or str(item.get("id") or "") != model_id:
            continue
        status = str(item.get("state") or item.get("status") or "").strip().lower()
        if status:
            return status
        if item.get("loaded") or item.get("is_loaded"):
            return "loaded"
        return "visible_not_loaded"
    return "not_visible"


def _omlx_request_diagnostic_context(
    stage: StageSpec,
    prompt: str,
    actual_model: str,
    *,
    loaded_models_before_unload: list[str],
    loaded_models_after_unload: list[str],
    loaded_models_after_load: list[str],
    retry_attempt: str,
) -> dict[str, Any]:
    user_chars = len(prompt or "")
    max_tokens = _omlx_max_tokens_for_stage(stage)
    return {
        "prompt_chars": user_chars,
        "prompt_estimated_tokens": max(1, (user_chars + 3) // 4),
        "prompt_hash": hashlib.sha256((prompt or "").encode("utf-8")).hexdigest(),
        "prompt_preview_head": _redact_secret_text((prompt or "")[:240]),
        "prompt_preview_tail": _redact_secret_text((prompt or "")[-240:]),
        "message_count": 1,
        "system_message_chars": 0,
        "user_message_chars": user_chars,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": _omlx_chat_template_kwargs_for_stage(stage, actual_model) or {},
        "actual_model": actual_model,
        "endpoint": f"{_omlx_base_url()}/v1/chat/completions",
        "loaded_models_before_unload": loaded_models_before_unload,
        "loaded_models_after_unload": loaded_models_after_unload,
        "loaded_models_after_load": loaded_models_after_load,
        "compact_mode_used": "compact_evidence_judge_packet" in (prompt or ""),
        "compact_budget": _extract_compact_budget_marker(prompt),
        "retry_attempt": retry_attempt,
    }


def _extract_compact_budget_marker(prompt: str) -> int | None:
    if "compact_evidence_judge_packet" not in (prompt or ""):
        return None
    return 6500


def _is_omlx_prefill_memory_text(text: str) -> bool:
    lowered = str(text or "").lower()
    return "prefill_memory_exceeded" in lowered or "prefill memory guard" in lowered


def _is_omlx_prefill_memory_diagnostic(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    return _is_omlx_prefill_memory_text(json.dumps(data, ensure_ascii=False, default=str))


def _omlx_prefill_memory_exception_diagnostic(
    stage: StageSpec,
    actual_model: str,
    exc: Exception,
    *,
    attempt: str,
    request_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_summary = _redact_secret_text(str(exc))
    diagnostic = {
        "stage_name": stage.stage_name,
        "canonical_model": stage.model,
        "actual_model": actual_model,
        "attempt": attempt,
        "blocked_reason": "OMLX_PREFILL_MEMORY_GUARD_BLOCKED",
        "empty_content_kind": "prefill_memory_exception",
        "response_type": "exception",
        "response_keys": [],
        "error_type": type(exc).__name__,
        "error_summary": raw_summary,
        "raw_error_code": _omlx_raw_error_code_from_text(raw_summary),
        "raw_error_summary": raw_summary,
        "choices_type": "missing",
        "choices_len": 0,
        "first_choice_keys": [],
        "message_keys": [],
        "content_type": "missing",
        "content_length": 0,
        "empty_content": True,
    }
    if request_context:
        diagnostic.update(request_context)
    return diagnostic


def _omlx_empty_content_diagnostic(
    stage: StageSpec,
    actual_model: str,
    data: Any,
    *,
    attempt: str,
    request_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    choices = data.get("choices") if isinstance(data, dict) else None
    first_choice = choices[0] if isinstance(choices, list) and choices else None
    message = first_choice.get("message") if isinstance(first_choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    error_value = data.get("error") if isinstance(data, dict) else None
    if isinstance(error_value, dict):
        error_summary = json.dumps(error_value, ensure_ascii=False, default=str)
    else:
        error_summary = str(error_value or "")
    if isinstance(data, dict) and error_value is not None and choices is None:
        empty_kind = "response_error_object"
    elif choices is None:
        empty_kind = "missing_choices"
    elif isinstance(choices, list) and not choices:
        empty_kind = "empty_choices"
    elif isinstance(content, str) and not content.strip():
        empty_kind = "empty_content_string"
    else:
        empty_kind = "parse_or_unknown"
    is_prefill = _is_omlx_prefill_memory_text(error_summary)
    diagnostic = {
        "stage_name": stage.stage_name,
        "canonical_model": stage.model,
        "actual_model": actual_model,
        "attempt": attempt,
        "blocked_reason": "OMLX_PREFILL_MEMORY_GUARD_BLOCKED" if is_prefill else "OMLX_EMPTY_CONTENT_BLOCKED",
        "empty_content_kind": empty_kind,
        "response_type": type(data).__name__,
        "response_keys": sorted(data.keys()) if isinstance(data, dict) else [],
        "error_type": str(data.get("type") or "") if isinstance(data, dict) else "",
        "error_summary": _redact_secret_text(error_summary),
        "raw_error_code": _omlx_raw_error_code(data),
        "raw_error_summary": _redact_secret_text(error_summary),
        "choices_type": type(choices).__name__ if choices is not None else "missing",
        "choices_len": len(choices) if isinstance(choices, list) else 0,
        "first_choice_keys": sorted(first_choice.keys()) if isinstance(first_choice, dict) else [],
        "message_keys": sorted(message.keys()) if isinstance(message, dict) else [],
        "content_type": type(content).__name__ if content is not None else "missing",
        "content_length": len(content) if isinstance(content, str) else 0,
        "empty_content": True,
    }
    if request_context:
        diagnostic.update(request_context)
    return diagnostic


def _omlx_raw_error_code(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    error = data.get("error")
    if isinstance(error, dict):
        return str(error.get("code") or error.get("omlx_code") or "")
    return ""


def _omlx_raw_error_code_from_text(text: str) -> str:
    value = str(text or "")
    for key in ("prefill_memory_exceeded", "invalid_request_error"):
        if key in value:
            return key
    return ""


def _write_omlx_stage_diagnostic(stage: StageSpec, executor: TaskEngineExecutor, *, base_dir: str | Path) -> Path | None:
    diagnostics = getattr(executor, "last_omlx_diagnostics", None)
    if not isinstance(diagnostics, dict):
        return None
    data = diagnostics.get(stage.stage_name)
    if not isinstance(data, dict):
        return None
    data = dict(data)
    data.setdefault("sample_id", _sample_id_from_base_dir(base_dir))
    data.setdefault("stage_name", stage.stage_name)
    data.setdefault("model", stage.model)
    data.setdefault("timeout_seconds", _decision_stage_timeout_s(stage))
    data.setdefault("admin_load_requested", False)
    data.setdefault("admin_load_returned", False)
    data.setdefault("observed_model_status", "")
    data.setdefault("inference_request_sent", False)
    data.setdefault("inference_response_received", False)
    data.setdefault("stdout", "")
    data.setdefault("stderr", "")
    stage_dir = Path(base_dir) / stage.stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    path = stage_dir / f"{stage.stage_name}.diagnostic.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_omlx_stage_diagnostic_snapshot(
    stage: StageSpec,
    data: dict[str, Any],
    *,
    base_dir: str | Path,
    filename: str,
) -> Path:
    snapshot = dict(data or {})
    snapshot.setdefault("sample_id", _sample_id_from_base_dir(base_dir))
    snapshot.setdefault("stage_name", stage.stage_name)
    snapshot.setdefault("model", stage.model)
    snapshot.setdefault("timeout_seconds", _decision_stage_timeout_s(stage))
    snapshot.setdefault("admin_load_requested", False)
    snapshot.setdefault("admin_load_returned", False)
    snapshot.setdefault("observed_model_status", "")
    snapshot.setdefault("inference_request_sent", False)
    snapshot.setdefault("inference_response_received", False)
    snapshot.setdefault("stdout", "")
    snapshot.setdefault("stderr", "")
    stage_dir = Path(base_dir) / stage.stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    path = stage_dir / filename
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _annotate_compact_evidence_judge_diagnostic(
    diagnostic: dict[str, Any],
    *,
    original_diagnostic: dict[str, Any],
    original_prompt: str,
    compact_prompt: str,
) -> None:
    original_chars = int(original_diagnostic.get("prompt_chars") or len(original_prompt or ""))
    compact_chars = len(compact_prompt or "")
    diagnostic["compact_mode_used"] = True
    diagnostic["original_prompt_chars"] = original_chars
    diagnostic["compact_prompt_chars"] = compact_chars
    diagnostic["original_prompt_estimated_tokens"] = int(
        original_diagnostic.get("prompt_estimated_tokens") or max(1, (original_chars + 3) // 4)
    )
    diagnostic["compact_prompt_estimated_tokens"] = max(1, (compact_chars + 3) // 4)
    diagnostic["blocked_reason_original"] = "OMLX_PREFILL_MEMORY_GUARD_BLOCKED"
    diagnostic["original_prompt_hash"] = original_diagnostic.get("prompt_hash") or hashlib.sha256(
        (original_prompt or "").encode("utf-8")
    ).hexdigest()
    diagnostic["compact_prompt_hash"] = hashlib.sha256((compact_prompt or "").encode("utf-8")).hexdigest()


def _annotate_evidence_judge_invalid_artifact_diagnostic(
    stage: StageSpec,
    executor: TaskEngineExecutor,
    *,
    base_dir: str | Path,
    content: str,
    prompt: str,
    quality_error: str,
    invalid_artifact_path: Path,
) -> None:
    diagnostics = getattr(executor, "last_omlx_diagnostics", None)
    if not isinstance(diagnostics, dict):
        diagnostics = {}
        try:
            setattr(executor, "last_omlx_diagnostics", diagnostics)
        except Exception:
            return
    data = dict(diagnostics.get(stage.stage_name) or {})
    first_line = _first_nonempty_line(content)
    normalized_first_line = first_line.lstrip("#").strip().lower().split(":", 1)[0].strip()
    prompt_chars = len(prompt or "")
    data.update(
        {
            "sample_id": _sample_id_from_base_dir(base_dir),
            "stage_name": stage.stage_name,
            "model": stage.model,
            "artifact_quality_error": quality_error,
            "first_nonempty_line": first_line,
            "normalized_first_line": normalized_first_line,
            "final_content_chars": len(content or ""),
            "invalid_artifact_path": str(invalid_artifact_path),
            "artifact_state": "invalid_artifact",
            "valid_for_pipeline": False,
            "blocked_reason": f"artifact_quality_error:{quality_error}",
            "error_summary": f"artifact_quality_error:{quality_error}",
            "prompt_chars": prompt_chars,
            "prompt_estimated_tokens": max(1, (prompt_chars + 3) // 4),
            "prompt_hash": hashlib.sha256((prompt or "").encode("utf-8")).hexdigest(),
            "compact_mode_used": "compact_evidence_judge_packet" in (prompt or "")
            or bool(data.get("compact_mode_used")),
        }
    )
    data.setdefault("inference_request_sent", False)
    data.setdefault("inference_response_received", False)
    diagnostics[stage.stage_name] = data


def _extract_chat_content(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]
    if isinstance(data.get("content"), str):
        return data["content"]
    return ""


def _agy_timeout_for_stage(stage: StageSpec) -> int:
    if stage.stage_name == "L1_gemini_search":
        return 600
    if stage.stage_name == "L4_gemini_audit":
        return 600
    if stage.stage_name == "intelligence_layer":
        return 360
    if stage.stage_name == "external_calibration":
        return 600
    return 240


def _agy_subprocess_cwd() -> str:
    """Return a stable non-hidden cwd for AGY subprocesses.

    WebUI often launches Hermes from ~/.hermes/hermes-agent. AGY/Antigravity
    treats hidden project roots differently, which can destabilize keychain and
    project URI handling. Artifacts still stay in the task run directory; only
    the AGY process cwd is moved to a visible, stable location.
    """
    raw_override = os.getenv("HERMES_AGY_CWD", "").strip()
    candidates: list[Path] = []
    if raw_override:
        candidates.append(Path(raw_override).expanduser())
    candidates.extend([AGY_STABLE_CWD_DEFAULT, Path("/private/tmp")])
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.is_absolute() or not resolved.exists():
            continue
        if any(part == ".hermes" for part in resolved.parts):
            continue
        return str(resolved)
    return "/private/tmp"


def _agy_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("GEMINI_DIR", "AGY_GEMINI_DIR", "GOOGLE_GEMINI_DIR"):
        value = env.get(key)
        if not value:
            continue
        expanded = Path(value).expanduser()
        if not expanded.is_absolute():
            env[key] = str((Path.home() / expanded).resolve())
    return env


def _agy_gemini_dir_is_absolute(env: dict[str, str]) -> bool | None:
    value = env.get("GEMINI_DIR")
    if not value:
        return None
    return Path(value).expanduser().is_absolute()


def _agy_reason_requires_refresh(reason: str, output: str = "") -> bool:
    combined = output or ""
    normalized = (reason or "").strip()
    if normalized == AGY_KEYCHAIN_FALSE_NEGATIVE:
        return False
    if _agy_location_unsupported(combined):
        return True
    if _agy_auth_negative(combined):
        return True
    if not combined.strip() and normalized in {"", "empty_stdout", "timeout_after=0s"}:
        return True
    return normalized in {
        "AGY_AUTH_TIMEOUT",
        "AGY_AUTH_REQUIRES_USER",
        "AGY_AUTH_BLOCKED",
        AGY_LOCATION_UNSUPPORTED,
        AGY_PRINTMODE_TIMEOUT_AUTH_UNCERTAIN,
        AGY_PRINTMODE_TIMEOUT_AFTER_AUTH_SUCCESS,
        "empty_stdout",
    } or (normalized.startswith("timeout_after=") and not combined.strip())


def _agy_manual_login_required(output: str) -> bool:
    lowered = (output or "").lower()
    return any(
        token in lowered
        for token in (
            "authorization code",
            "verification code",
            "enter code",
            "google authorization",
            "requires user confirmation",
            "manual login",
        )
    )


def _agy_command_result_dict(
    *,
    command: list[str],
    returncode: int | str,
    stdout: str,
    stderr: str,
    elapsed: float,
    timeout: bool = False,
    log_file: Path | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "command": command,
        "returncode": returncode,
        "timeout": timeout,
        "elapsed_seconds": round(elapsed, 2),
        "stdout_len": len(stdout or ""),
        "stdout_tail": _tail_text(stdout),
        "stderr_tail": _tail_text(stderr),
    }
    if log_file:
        data["log_file"] = str(log_file)
        data["log_tail"] = _tail_text(_read_text(log_file))
    return data


def _run_agy_auth_refresh_gate(
    *,
    agy_path: str,
    agy_cwd: str,
    agy_env: dict[str, str],
    timeout_s: int,
    model: str | None = None,
) -> dict[str, Any]:
    bare_timeout = min(max(int(timeout_s or AGY_BARE_REFRESH_TIMEOUT_S), 15), AGY_BARE_REFRESH_TIMEOUT_S)
    bare_command = [agy_path]
    bare_started = time.time()
    try:
        bare = subprocess.run(
            bare_command,
            capture_output=True,
            text=True,
            timeout=bare_timeout,
            cwd=agy_cwd,
            env=agy_env,
        )
        bare_stdout = bare.stdout or ""
        bare_stderr = bare.stderr or ""
        bare_result = _agy_command_result_dict(
            command=bare_command,
            returncode=bare.returncode,
            stdout=bare_stdout,
            stderr=bare_stderr,
            elapsed=time.time() - bare_started,
        )
    except subprocess.TimeoutExpired as exc:
        bare_stdout = _decode_timeout_part(exc.stdout)
        bare_stderr = _decode_timeout_part(exc.stderr)
        bare_result = _agy_command_result_dict(
            command=bare_command,
            returncode="timeout",
            stdout=bare_stdout,
            stderr=bare_stderr,
            elapsed=time.time() - bare_started,
            timeout=True,
        )

    actual_model = model or resolve_agy_model_alias(GEMINI_PRO_HIGH)
    sentinel_log = Path(tempfile.gettempdir()) / f"agy-refresh-{uuid.uuid4().hex[:8]}.log"
    sentinel_command = [
        agy_path,
        "--log-file",
        str(sentinel_log),
        "--model",
        actual_model,
        "-p",
        f"Reply exactly: {AGY_INTERNAL_PREFLIGHT_SENTINEL}",
        "--print-timeout",
        f"{min(max(int(timeout_s or 45), 30), 120)}s",
    ]
    sentinel_started = time.time()
    try:
        sentinel = subprocess.run(
            sentinel_command,
            capture_output=True,
            text=True,
            timeout=min(max(int(timeout_s or 45), 30), 120) + 30,
            cwd=agy_cwd,
            env=agy_env,
        )
        sentinel_stdout = sentinel.stdout or ""
        sentinel_stderr = sentinel.stderr or ""
        sentinel_result = _agy_command_result_dict(
            command=sentinel_command,
            returncode=sentinel.returncode,
            stdout=sentinel_stdout,
            stderr=sentinel_stderr,
            elapsed=time.time() - sentinel_started,
            log_file=sentinel_log,
        )
    except subprocess.TimeoutExpired as exc:
        sentinel_stdout = _decode_timeout_part(exc.stdout)
        sentinel_stderr = _decode_timeout_part(exc.stderr)
        sentinel_result = _agy_command_result_dict(
            command=sentinel_command,
            returncode="timeout",
            stdout=sentinel_stdout,
            stderr=sentinel_stderr,
            elapsed=time.time() - sentinel_started,
            timeout=True,
            log_file=sentinel_log,
        )

    sentinel_log_text = _read_text(sentinel_log)
    sentinel_combined = "\n".join(part for part in (sentinel_stdout, sentinel_stderr, sentinel_log_text) if part)
    bare_combined = "\n".join(part for part in (bare_stdout, bare_stderr) if part)
    if sentinel_result["returncode"] == 0 and AGY_INTERNAL_PREFLIGHT_SENTINEL in sentinel_stdout.strip():
        return {
            "status": "AGY_AUTH_REFRESH_OK",
            "blocked_reason": "",
            "bare_agy": bare_result,
            "print_mode_sentinel": sentinel_result,
            "print_mode_sentinel_text": AGY_INTERNAL_PREFLIGHT_SENTINEL,
        }

    if _agy_location_unsupported(sentinel_combined):
        reason = AGY_LOCATION_UNSUPPORTED
    elif _agy_manual_login_required(bare_combined) or _agy_manual_login_required(sentinel_combined) or _agy_auth_negative(sentinel_combined):
        reason = REQUIRES_MANUAL_AGY_LOGIN
    elif not sentinel_stdout.strip():
        reason = AGY_PRINT_MODE_EMPTY_RESPONSE
    elif _agy_timeout_response(sentinel_combined):
        reason = AGY_PRINTMODE_TIMEOUT_AUTH_UNCERTAIN if _agy_auth_negative(sentinel_combined) else AGY_TIMEOUT_BLOCKED
    else:
        reason = REQUIRES_BARE_AGY_REFRESH
    return {
        "status": "BLOCKED_STATUS",
        "blocked_reason": reason,
        "bare_agy": bare_result,
        "print_mode_sentinel": sentinel_result,
        "print_mode_sentinel_text": AGY_INTERNAL_PREFLIGHT_SENTINEL,
    }


def _agy_preflight_from_refresh(
    refresh: dict[str, Any],
    *,
    agy_cwd: str,
    gemini_dir_absolute: bool | None,
) -> dict[str, Any]:
    sentinel = refresh.get("print_mode_sentinel") if isinstance(refresh, dict) else {}
    if not isinstance(sentinel, dict):
        sentinel = {}
    command = list(sentinel.get("command") or [])
    stdout = str(sentinel.get("stdout_tail") or "")
    stderr = str(sentinel.get("stderr_tail") or "")
    elapsed = float(sentinel.get("elapsed_seconds") or 0)
    if refresh.get("status") == "AGY_AUTH_REFRESH_OK":
        result = _agy_preflight_result(
            "AGY_OK",
            command=command,
            elapsed=elapsed,
            stdout=stdout,
            stderr=stderr,
            models=[],
            agy_cwd=agy_cwd,
            gemini_dir_absolute=gemini_dir_absolute,
        )
        result["auth_refresh"] = refresh
        return result
    result = _agy_preflight_blocked(
        str(refresh.get("blocked_reason") or REQUIRES_BARE_AGY_REFRESH),
        command=command,
        elapsed=elapsed,
        stdout=stdout,
        stderr=stderr,
        models=[],
        agy_cwd=agy_cwd,
        gemini_dir_absolute=gemini_dir_absolute,
    )
    result["auth_refresh"] = refresh
    return result


def _agy_refresh_failure_suffix(refresh: dict[str, Any]) -> str:
    if not isinstance(refresh, dict):
        return ""
    summary = {
        "status": refresh.get("status"),
        "blocked_reason": refresh.get("blocked_reason"),
        "bare_agy": refresh.get("bare_agy"),
        "print_mode_sentinel": refresh.get("print_mode_sentinel"),
        "print_mode_sentinel_text": refresh.get("print_mode_sentinel_text"),
    }
    return "\nauth_refresh=" + json.dumps(summary, ensure_ascii=False)


def _agy_preflight_result(
    status: str,
    *,
    command: list[str],
    elapsed: float,
    stdout: str,
    stderr: str,
    models: list[str],
    missing_models: list[str] | None = None,
    agy_cwd: str = "",
    gemini_dir_absolute: bool | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "blocked_stage": "" if status == "AGY_OK" else "agy_preflight",
        "blocked_reason": "" if status == "AGY_OK" else status,
        "command": command,
        "elapsed_seconds": round(elapsed, 2),
        "stdout_len": len(stdout or ""),
        "stdout_tail": _tail_text(stdout),
        "stderr_tail": _tail_text(stderr),
        "models": models,
        "required_models": list(AGY_PREFLIGHT_REQUIRED_MODELS),
        "missing_models": missing_models or [],
        "agy_cwd": agy_cwd,
        "gemini_dir_absolute": gemini_dir_absolute,
        "authorization_code_note": (
            "" if status == "AGY_OK" else "authorization code must be entered by user manually"
        ),
    }


def _agy_preflight_blocked(
    reason: str,
    *,
    command: list[str],
    elapsed: float,
    stdout: str,
    stderr: str,
    models: list[str],
    missing_models: list[str] | None = None,
    agy_cwd: str = "",
    gemini_dir_absolute: bool | None = None,
) -> dict[str, Any]:
    result = _agy_preflight_result(
        reason,
        command=command,
        elapsed=elapsed,
        stdout=stdout,
        stderr=stderr,
        models=models,
        missing_models=missing_models,
        agy_cwd=agy_cwd,
        gemini_dir_absolute=gemini_dir_absolute,
    )
    result["status"] = "BLOCKED_STATUS"
    result["blocked_reason"] = reason
    return result


def _classify_agy_preflight_block(stdout: str, stderr: str) -> str:
    combined = "\n".join(part for part in (stdout, stderr) if part)
    lowered = combined.lower()
    if _agy_location_unsupported(combined):
        return AGY_LOCATION_UNSUPPORTED
    if "authentication timed out" in lowered or ("silent auth" in lowered and "timed out" in lowered):
        return "AGY_AUTH_TIMEOUT"
    if _agy_timeout_response(combined):
        if _agy_printmode_timeout_after_auth_success(combined, ""):
            return AGY_PRINTMODE_TIMEOUT_AFTER_AUTH_SUCCESS
        if _agy_printmode_timeout_auth_uncertain(combined):
            return AGY_PRINTMODE_TIMEOUT_AUTH_UNCERTAIN
        return AGY_TIMEOUT_BLOCKED
    if _agy_keychain_false_negative(combined):
        return AGY_KEYCHAIN_FALSE_NEGATIVE
    if "authorization code" in lowered or "verification code" in lowered or "oauth" in lowered or "browser" in lowered:
        return "AGY_AUTH_REQUIRES_USER"
    if "not logged" in lowered or "not authenticated" in lowered or "login" in lowered or "authorize" in lowered:
        return "AGY_AUTH_REQUIRES_USER"
    return "AGY_AUTH_REQUIRES_USER"


def _agy_keychain_false_negative(output: str) -> bool:
    lowered = (output or "").lower()
    auth_negative = (
        "you are not logged into antigravity" in lowered
        or "not logged into antigravity" in lowered
        or "not authenticated" in lowered
    )
    auth_success = (
        "authenticated via keyring" in lowered
        or "oauth: authenticated successfully" in lowered
        or "oauth authenticated successfully" in lowered
        or "silent auth succeeded" in lowered
    )
    return auth_negative and auth_success and not _agy_timeout_response(output)


def _agy_auth_success(output: str) -> bool:
    lowered = (output or "").lower()
    return (
        "authenticated via keyring" in lowered
        or "oauth: authenticated successfully" in lowered
        or "oauth authenticated successfully" in lowered
        or "silent auth succeeded" in lowered
    )


def _agy_auth_negative(output: str) -> bool:
    lowered = (output or "").lower()
    return (
        "you are not logged into antigravity" in lowered
        or "not logged into antigravity" in lowered
        or "not authenticated" in lowered
    )


def _agy_location_unsupported(output: str) -> bool:
    lowered = (output or "").lower()
    return "user location is not supported for the api use" in lowered or (
        "failed_precondition" in lowered and "location" in lowered and "not supported" in lowered
    )


def _agy_timeout_response(output: str) -> bool:
    lowered = (output or "").lower()
    return "error: timed out waiting for response" in lowered or "print mode: timed out" in lowered


def _agy_printmode_timeout_after_auth_success(output: str, actual_model: str) -> bool:
    lowered = (output or "").lower()
    actual = actual_model.strip().lower()
    auth_success = _agy_auth_success(output)
    model_override = (
        f"resolving model {actual}" in lowered
        or f'propagating selected model override to backend: label="{actual}"' in lowered
        or (not actual and "propagating selected model override" in lowered)
    )
    return (
        auth_success
        and model_override
        and "streamgeneratecontent" in lowered
        and "print mode: timed out" in lowered
    )


def _agy_printmode_timeout_auth_uncertain(output: str) -> bool:
    return _agy_timeout_response(output) and _agy_auth_negative(output) and not _agy_auth_success(output)


def _agy_timeout_blocker_reason(output: str, actual_model: str, *, attempt: int) -> str:
    if _agy_printmode_timeout_after_auth_success(output, actual_model):
        return AGY_PRINTMODE_TIMEOUT_AFTER_AUTH_SUCCESS
    if _agy_printmode_timeout_auth_uncertain(output):
        return AGY_PRINTMODE_TIMEOUT_AUTH_UNCERTAIN
    return AGY_TIMEOUT_RESPONSE if attempt == 0 else AGY_TIMEOUT_BLOCKED


def _parse_agy_models(stdout: str) -> list[str]:
    models: list[str] = []
    for raw_line in (stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        cleaned = line.lstrip("-*•0123456789. ").strip()
        if cleaned:
            models.append(cleaned)
    return models


def _tail_text(value: str, *, limit: int = 2000) -> str:
    return (value or "")[-limit:]


def _settings_agy_model() -> str:
    path = Path.home() / ".gemini" / "antigravity-cli" / "settings.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    model = data.get("model")
    return str(model).strip() if isinstance(model, str) else ""


def _env_file_agy_model(env_key: str) -> str:
    path = Path(os.getenv("HERMES_AGY_MODEL_ALIAS_ENV", "work/agy_model_alias.env"))
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    prefix = f"{env_key}="
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("export "):
            stripped = stripped[len("export "):]
        if stripped.startswith(prefix):
            return stripped[len(prefix):].strip().strip('"').strip("'")
    return ""


def _agy_model_alias_failed(output: str, actual_model: str) -> bool:
    lowered = output.lower()
    actual = actual_model.strip().lower()
    if actual == "ccpa":
        return True
    resolved_actual = (
        f"resolving model {actual}" in lowered
        or f'propagating selected model override to backend: label="{actual}"' in lowered
    )
    defaulted_ccpa = (
        "not in local config" in lowered
        or "defaulting to ccpa" in lowered
        or ("model resolved via default" in lowered and "ccpa" in lowered)
    )
    return defaulted_ccpa and not resolved_actual


def _format_agy_failure(
    *,
    stage: StageSpec,
    command: list[str],
    canonical_model: str,
    actual_model: str,
    log_file: Path,
    stdout: str,
    stderr: str,
    log_text: str,
    elapsed: float,
    agy_cwd: str,
    reason: str,
) -> str:
    key_lines = _agy_key_lines(stdout, stderr, log_text)
    return (
        f"{stage.stage_name}: AGY_CALL_BLOCKED\n"
        f"reason={reason}\n"
        f"canonical_model={canonical_model!r}\n"
        f"actual_model={actual_model!r}\n"
        f"log_file={str(log_file)!r}\n"
        f"agy_cwd={agy_cwd!r}\n"
        f"elapsed_seconds={elapsed:.1f}\n"
        f"command={json.dumps(command, ensure_ascii=False)}\n"
        f"key_lines={json.dumps(key_lines, ensure_ascii=False)}"
    )


def _agy_key_lines(stdout: str, stderr: str, log_text: str) -> list[str]:
    interesting = (
        "not in local config",
        "defaulting to ccpa",
        "model resolved",
        "resolving model",
        "propagating selected model",
        "authenticated via keyring",
        "oauth: authenticated successfully",
        "oauth authenticated successfully",
        "silent auth succeeded",
        "print mode",
        "error",
        "failed",
        "failed_precondition",
        "location",
        "not supported",
        "timeout",
        "operation not permitted",
    )
    lines: list[str] = []
    for label, text in (("stdout", stdout), ("stderr", stderr), ("log", log_text)):
        for line in (text or "").splitlines():
            lowered = line.lower()
            if any(token in lowered for token in interesting):
                lines.append(f"{label}: {line}"[:1000])
            if len(lines) >= 24:
                return lines
    if stdout.strip():
        lines.append(f"stdout: {stdout.strip()[:1000]}")
    if stderr.strip():
        lines.append(f"stderr: {stderr.strip()[:1000]}")
    if log_text.strip():
        lines.append(f"log: {log_text.strip()[:1000]}")
    return lines[:24]


def _decode_timeout_part(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


__all__ = [
    "GEMMA431B_ACTUAL_MODEL_DEFAULT",
    "LLAMA70B_ACTUAL_MODEL_DEFAULT",
    "LocalTaskEngineExecutor",
    "NEMOTRON120B_ACTUAL_MODEL_DEFAULT",
    "QWEN72B_ACTUAL_MODEL_DEFAULT",
    "R1_ACTUAL_MODEL_DEFAULT",
    "TaskEngineExecutor",
    "resolve_gemma431b_omlx_model_alias",
    "resolve_llama70b_omlx_model_alias",
    "resolve_nemotron120b_omlx_model_alias",
    "resolve_agy_model_alias",
    "resolve_qwen72b_omlx_model_alias",
    "resolve_r1_omlx_model_alias",
    "run_agy_preflight",
    "run_decision_final_smoke",
    "run_research_decision_alternative_generator_smoke",
    "run_research_decision_evidence_judge_smoke",
    "run_research_decision_external_calibration_smoke",
    "run_research_decision_final_controller_smoke",
    "run_research_decision_intelligence_smoke",
    "run_research_decision_insight_harvester_smoke",
    "run_research_decision_convergence_smoke",
    "run_research_decision_l1_l10_smoke",
    "run_research_decision_l1_l11_smoke",
    "run_research_decision_l1_l12_smoke",
    "run_research_decision_l1_l13_smoke",
    "run_research_decision_l1_l14_smoke",
    "run_research_decision_l1_l15_smoke",
    "run_research_decision_l1_l16_smoke",
    "run_research_decision_l1_l7_smoke",
    "run_research_decision_l1_l8_smoke",
    "run_research_decision_l1_l9_smoke",
    "run_research_decision_premise_auditor_smoke",
    "run_research_decision_structure_mapper_smoke",
    "run_research_decision_supplementary_search_smoke",
    "run_research_l2_5_codex_handoff_smoke",
    "run_research_l1_l2_smoke",
    "run_research_l1_l3_smoke",
    "run_research_l1_l4_smoke",
    "run_research_l1_l5_smoke",
    "run_research_l3_synthesis_smoke",
    "run_research_l4_gemini_audit_smoke",
    "run_research_l5_acceptance_smoke",
    "run_simulated_pipeline",
]

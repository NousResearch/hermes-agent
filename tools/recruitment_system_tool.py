"""Hermes tool wrapper for RecruitmentSystem HTTP APIs."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any

from recruitment_system.formatter import format_answer
from recruitment_system.intent import (
    MY_ATTENDANCE_BY_TIME_RANGE,
    MY_ATTENDANCE_RECENT_WEEK,
    RECRUITING_JOB_DETAIL,
    RECRUITING_JOB_LIST,
    UNKNOWN_QUERY_INTENT,
    recognize_intent,
)
from tools.registry import registry


@dataclass(frozen=True)
class RecruitmentAPIConfig:
    base_url: str
    tenant_id: str = ""
    user_id: str = ""
    token: str = ""
    timeout_seconds: float = 15.0

    @classmethod
    def from_env(cls) -> "RecruitmentAPIConfig":
        return cls(
            base_url=os.getenv("RECRUITMENT_API_BASE_URL", "").rstrip("/"),
            tenant_id=os.getenv("RECRUITMENT_API_TENANT_ID", ""),
            user_id=os.getenv("RECRUITMENT_API_USER_ID", ""),
            token=os.getenv("RECRUITMENT_API_TOKEN", ""),
            timeout_seconds=_env_float("RECRUITMENT_API_TIMEOUT_SECONDS", 15.0),
        )

    def is_complete(self) -> bool:
        return bool(self.base_url)

    def missing_keys(self) -> list[str]:
        return [] if self.base_url else ["RECRUITMENT_API_BASE_URL"]


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return float(raw)


def _is_recruitment_configured() -> bool:
    try:
        return RecruitmentAPIConfig.from_env().is_complete()
    except Exception:
        return False


def _query_handler(args: dict, **_kwargs) -> str:
    config = RecruitmentAPIConfig.from_env()
    question = str(args.get("user_question") or args.get("question") or "")
    tenant_id = _tenant_id(args, config)
    user_id = _user_id(args, config)
    trace_id = uuid.uuid4().hex

    if not config.is_complete():
        return _error(trace_id, "CONFIG_ERROR", "RecruitmentSystem API config is incomplete", missing=config.missing_keys())
    if not tenant_id:
        return _error(trace_id, "INVALID_REQUEST", "tenant_id 或 RECRUITMENT_API_TENANT_ID 不能为空")
    if not question.strip():
        return _error(trace_id, "INVALID_REQUEST", "user_question 不能为空")

    intent = recognize_intent(question)
    try:
        if intent.name == RECRUITING_JOB_LIST:
            return json.dumps(_query_job_list(config, args, tenant_id, user_id, trace_id, intent.name), ensure_ascii=False)
        if intent.name == RECRUITING_JOB_DETAIL:
            job_name = str(args.get("job_name") or intent.slots.get("job_name") or "").strip()
            return json.dumps(
                _query_job_detail(config, args, tenant_id, user_id, trace_id, job_name),
                ensure_ascii=False,
            )
        if intent.name in {MY_ATTENDANCE_RECENT_WEEK, MY_ATTENDANCE_BY_TIME_RANGE}:
            return _error(
                trace_id,
                "UNSUPPORTED_INTENT",
                "当前 API 工具仅支持招聘岗位查询和新增；考勤查询需要接入 recruitmentSystem 考勤 API 后再启用。",
                intent=intent.name,
            )
        return _error(
            trace_id,
            "UNKNOWN_QUERY_INTENT",
            "暂时无法识别该查询，请尝试查询招聘岗位或新增招聘岗位。",
            intent=UNKNOWN_QUERY_INTENT,
        )
    except RecruitmentAPIError as exc:
        return json.dumps(exc.to_payload(trace_id=trace_id, intent=intent.name), ensure_ascii=False, default=str)
    except Exception as exc:
        return _error(trace_id, "QUERY_FAILED", f"{type(exc).__name__}: {exc}", intent=intent.name)


def _create_job_handler(args: dict, **_kwargs) -> str:
    config = RecruitmentAPIConfig.from_env()
    tenant_id = _tenant_id(args, config)
    user_id = _user_id(args, config)
    trace_id = uuid.uuid4().hex

    if not config.is_complete():
        return _error(trace_id, "CONFIG_ERROR", "RecruitmentSystem API config is incomplete", missing=config.missing_keys())
    if not tenant_id:
        return _error(trace_id, "INVALID_REQUEST", "tenant_id 或 RECRUITMENT_API_TENANT_ID 不能为空")

    job_name = str(args.get("job_name") or args.get("jobName") or "").strip()
    if not job_name:
        return _error(trace_id, "INVALID_REQUEST", "job_name 不能为空")

    try:
        existing = _find_exact_job(config, tenant_id, user_id, trace_id, job_name)
        if existing and not bool(args.get("allow_duplicate", False)):
            rows = [_normalize_job(existing)]
            return json.dumps(
                {
                    "success": True,
                    "created": False,
                    "intent": "recruiting_job_create",
                    "answer": f"已存在同名岗位：{job_name}，未重复创建。",
                    "data": rows,
                    "safe": True,
                    "trace_id": trace_id,
                    "api": {"method": "GET", "path": "/api/v1/jobs"},
                },
                ensure_ascii=False,
                default=str,
            )

        create_payload = _job_create_payload(args, job_name, user_id)
        job_id = _api_request(
            config,
            "POST",
            "/api/v1/jobs",
            tenant_id=tenant_id,
            user_id=user_id,
            trace_id=trace_id,
            json_body=create_payload,
        )
        if not isinstance(job_id, int):
            job_id = int(job_id)

        requirement_payload = _job_requirement_payload(args)
        _api_request(
            config,
            "PUT",
            f"/api/v1/jobs/{job_id}/requirements",
            tenant_id=tenant_id,
            user_id=user_id,
            trace_id=trace_id,
            json_body=requirement_payload,
        )

        published = bool(args.get("publish", True))
        if published:
            _api_request(
                config,
                "POST",
                f"/api/v1/jobs/{job_id}/online",
                tenant_id=tenant_id,
                user_id=user_id,
                trace_id=trace_id,
            )

        detail = _api_request(
            config,
            "GET",
            f"/api/v1/jobs/{job_id}",
            tenant_id=tenant_id,
            user_id=user_id,
            trace_id=trace_id,
        )
        row = _normalize_job(detail if isinstance(detail, dict) else {"id": job_id, **create_payload})
        status_text = "已上线" if published else "已创建为草稿"
        return json.dumps(
            {
                "success": True,
                "created": True,
                "intent": "recruiting_job_create",
                "answer": f"已通过 recruitmentSystem API 新增岗位：{job_name}，当前状态：{status_text}。",
                "data": [row],
                "safe": True,
                "trace_id": trace_id,
                "api": {
                    "create": "POST /api/v1/jobs",
                    "requirements": f"PUT /api/v1/jobs/{job_id}/requirements",
                    "online": f"POST /api/v1/jobs/{job_id}/online" if published else None,
                },
            },
            ensure_ascii=False,
            default=str,
        )
    except RecruitmentAPIError as exc:
        return json.dumps(exc.to_payload(trace_id=trace_id, intent="recruiting_job_create"), ensure_ascii=False, default=str)
    except Exception as exc:
        return _error(trace_id, "CREATE_JOB_FAILED", f"{type(exc).__name__}: {exc}", intent="recruiting_job_create")


def _health_handler(_args: dict, **_kwargs) -> str:
    config = RecruitmentAPIConfig.from_env()
    payload: dict[str, Any] = {
        "configured": config.is_complete(),
        "missing": config.missing_keys(),
        "base_url": config.base_url,
        "tenant_id_configured": bool(config.tenant_id),
    }
    if not config.is_complete():
        payload.update({"healthy": False, "message": "RecruitmentSystem API config is incomplete"})
        return json.dumps(payload, ensure_ascii=False)

    trace_id = uuid.uuid4().hex
    try:
        _api_request(
            config,
            "GET",
            "/api/v1/jobs",
            params={"pageNo": 1, "pageSize": 1},
            tenant_id=config.tenant_id,
            user_id=config.user_id,
            trace_id=trace_id,
        )
        payload.update({"healthy": True, "message": "ok", "trace_id": trace_id})
    except RecruitmentAPIError as exc:
        payload.update(exc.to_payload(trace_id=trace_id, intent="health"))
        payload["healthy"] = False
    return json.dumps(payload, ensure_ascii=False, default=str)


def _query_job_list(
    config: RecruitmentAPIConfig,
    args: dict,
    tenant_id: str,
    user_id: str,
    trace_id: str,
    intent: str,
) -> dict[str, Any]:
    params = {
        "pageNo": int(args.get("page_no") or args.get("pageNo") or 1),
        "pageSize": min(int(args.get("page_size") or args.get("pageSize") or 50), 100),
        "status": str(args.get("status") or "ONLINE"),
    }
    for src, dst in (
        ("keyword", "keyword"),
        ("department", "department"),
        ("employment_type", "employmentType"),
        ("employmentType", "employmentType"),
        ("work_location", "workLocation"),
        ("workLocation", "workLocation"),
        ("owner_user_id", "ownerUserId"),
        ("ownerUserId", "ownerUserId"),
    ):
        value = args.get(src)
        if value:
            params[dst] = value

    data = _api_request(
        config,
        "GET",
        "/api/v1/jobs",
        params=params,
        tenant_id=tenant_id,
        user_id=user_id,
        trace_id=trace_id,
    )
    records = data.get("records", []) if isinstance(data, dict) else []
    rows = [_normalize_job(row) for row in records]
    return {
        "success": True,
        "intent": intent,
        "answer": format_answer(RECRUITING_JOB_LIST, rows),
        "data": rows,
        "safe": True,
        "trace_id": trace_id,
        "total": data.get("total", len(rows)) if isinstance(data, dict) else len(rows),
        "api": {"method": "GET", "path": "/api/v1/jobs", "params": params},
    }


def _query_job_detail(
    config: RecruitmentAPIConfig,
    args: dict,
    tenant_id: str,
    user_id: str,
    trace_id: str,
    job_name: str,
) -> dict[str, Any]:
    job_id = args.get("job_id") or args.get("jobId")
    if not job_id:
        if not job_name:
            raise RecruitmentAPIError("INVALID_REQUEST", "岗位详情查询需要 job_name 或 job_id")
        found = _find_exact_job(config, tenant_id, user_id, trace_id, job_name) or _find_first_job(
            config, tenant_id, user_id, trace_id, job_name
        )
        if not found:
            rows: list[dict[str, Any]] = []
            return {
                "success": True,
                "intent": RECRUITING_JOB_DETAIL,
                "answer": format_answer(RECRUITING_JOB_DETAIL, rows),
                "data": rows,
                "safe": True,
                "trace_id": trace_id,
                "api": {"method": "GET", "path": "/api/v1/jobs", "params": {"keyword": job_name}},
            }
        job_id = found.get("id")

    detail = _api_request(
        config,
        "GET",
        f"/api/v1/jobs/{job_id}",
        tenant_id=tenant_id,
        user_id=user_id,
        trace_id=trace_id,
    )
    rows = [_normalize_job(detail)]
    return {
        "success": True,
        "intent": RECRUITING_JOB_DETAIL,
        "answer": format_answer(RECRUITING_JOB_DETAIL, rows),
        "data": rows,
        "safe": True,
        "trace_id": trace_id,
        "api": {"method": "GET", "path": f"/api/v1/jobs/{job_id}"},
    }


def _find_exact_job(
    config: RecruitmentAPIConfig,
    tenant_id: str,
    user_id: str,
    trace_id: str,
    job_name: str,
) -> dict[str, Any] | None:
    data = _api_request(
        config,
        "GET",
        "/api/v1/jobs",
        params={"pageNo": 1, "pageSize": 100, "keyword": job_name},
        tenant_id=tenant_id,
        user_id=user_id,
        trace_id=trace_id,
    )
    records = data.get("records", []) if isinstance(data, dict) else []
    for row in records:
        if str(row.get("jobName") or "").strip().lower() == job_name.lower():
            return row
    return None


def _find_first_job(
    config: RecruitmentAPIConfig,
    tenant_id: str,
    user_id: str,
    trace_id: str,
    keyword: str,
) -> dict[str, Any] | None:
    data = _api_request(
        config,
        "GET",
        "/api/v1/jobs",
        params={"pageNo": 1, "pageSize": 1, "keyword": keyword},
        tenant_id=tenant_id,
        user_id=user_id,
        trace_id=trace_id,
    )
    records = data.get("records", []) if isinstance(data, dict) else []
    return records[0] if records else None


def _api_request(
    config: RecruitmentAPIConfig,
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    tenant_id: str = "",
    user_id: str = "",
    trace_id: str = "",
) -> Any:
    if not tenant_id:
        raise RecruitmentAPIError("INVALID_REQUEST", "tenant_id 或 RECRUITMENT_API_TENANT_ID 不能为空")

    url = f"{config.base_url}{path}"
    if params:
        query = urllib.parse.urlencode({key: value for key, value in params.items() if value not in (None, "")})
        if query:
            url = f"{url}?{query}"

    body = None
    headers = {
        "Accept": "application/json",
        "X-Tenant-Id": str(tenant_id),
        "X-Trace-Id": trace_id or uuid.uuid4().hex,
    }
    if user_id:
        headers["X-User-Id"] = str(user_id)
    if config.token:
        headers["Authorization"] = config.token if config.token.lower().startswith("bearer ") else f"Bearer {config.token}"
    if json_body is not None:
        body = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=body, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            raw = response.read().decode("utf-8")
            return _parse_api_payload(raw, response.status)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {"message": raw}
        raise RecruitmentAPIError(
            str(parsed.get("code") or f"HTTP_{exc.code}"),
            str(parsed.get("message") or exc.reason),
            status=exc.code,
            trace_id=str(parsed.get("traceId") or ""),
            response=parsed,
        ) from exc
    except urllib.error.URLError as exc:
        raise RecruitmentAPIError("API_CONNECTION_FAILED", str(exc.reason)) from exc


def _parse_api_payload(raw: str, status: int) -> Any:
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise RecruitmentAPIError("INVALID_API_RESPONSE", f"API returned non-JSON response with HTTP {status}") from exc

    if isinstance(parsed, dict) and parsed.get("code") not in (None, "0", 0):
        raise RecruitmentAPIError(
            str(parsed.get("code")),
            str(parsed.get("message") or "RecruitmentSystem API error"),
            status=status,
            trace_id=str(parsed.get("traceId") or ""),
            response=parsed,
        )
    if isinstance(parsed, dict) and "data" in parsed:
        return parsed.get("data")
    return parsed


def _job_create_payload(args: dict, job_name: str, user_id: str) -> dict[str, Any]:
    return {
        "jobCode": _blank_to_none(args.get("job_code") or args.get("jobCode")),
        "jobName": job_name,
        "department": str(args.get("department") or os.getenv("RECRUITMENT_DEFAULT_DEPARTMENT") or "研发中心"),
        "employmentType": str(
            args.get("employment_type") or args.get("employmentType") or os.getenv("RECRUITMENT_DEFAULT_EMPLOYMENT_TYPE") or "FULL_TIME"
        ),
        "workLocation": str(
            args.get("work_location") or args.get("workLocation") or os.getenv("RECRUITMENT_DEFAULT_WORK_LOCATION") or "上海"
        ),
        "headcount": int(args.get("headcount") or os.getenv("RECRUITMENT_DEFAULT_HEADCOUNT") or 1),
        "ownerUserId": str(
            args.get("owner_user_id")
            or args.get("ownerUserId")
            or os.getenv("RECRUITMENT_DEFAULT_OWNER_USER_ID")
            or user_id
            or "hr_mgr_1001"
        ),
        "ownerUserName": str(
            args.get("owner_user_name")
            or args.get("ownerUserName")
            or os.getenv("RECRUITMENT_DEFAULT_OWNER_USER_NAME")
            or "HR经理"
        ),
    }


def _job_requirement_payload(args: dict) -> dict[str, Any]:
    return {
        "educationRequirement": str(args.get("education_requirement") or args.get("educationRequirement") or "UNLIMITED"),
        "majorRequirement": _blank_to_none(args.get("major_requirement") or args.get("majorRequirement")),
        "experienceMinYears": int(args.get("experience_min_years") or args.get("experienceMinYears") or 0),
        "experienceMaxYears": int(args.get("experience_max_years") or args.get("experienceMaxYears") or 0),
        "skillRequirements": _string_list(args.get("skill_requirements") or args.get("skillRequirements")),
        "certificateRequirements": _string_list(args.get("certificate_requirements") or args.get("certificateRequirements")),
        "softSkillRequirements": _string_list(args.get("soft_skill_requirements") or args.get("softSkillRequirements")),
        "salaryMin": float(args.get("salary_min") or args.get("salaryMin") or 0),
        "salaryMax": float(args.get("salary_max") or args.get("salaryMax") or 0),
        "salaryCurrency": str(args.get("salary_currency") or args.get("salaryCurrency") or "CNY"),
        "remark": _blank_to_none(args.get("remark")),
    }


def _normalize_job(row: dict[str, Any] | None) -> dict[str, Any]:
    row = row or {}
    requirement = row.get("requirement") or {}
    skills = requirement.get("skillRequirements") or []
    soft_skills = requirement.get("softSkillRequirements") or []
    requirements_text = "；".join(str(item) for item in [*skills, *soft_skills] if item)
    return {
        "id": row.get("id"),
        "job_code": row.get("jobCode"),
        "job_name": row.get("jobName"),
        "department_name": row.get("department"),
        "employment_type": row.get("employmentType"),
        "work_location": row.get("workLocation"),
        "headcount": row.get("headcount"),
        "owner_user_id": row.get("ownerUserId"),
        "owner_user_name": row.get("ownerUserName"),
        "job_status": row.get("status"),
        "recruit_status": _recruit_status(row.get("status")),
        "online_at": row.get("onlineAt"),
        "updated_at": row.get("updatedAt"),
        "job_requirement": requirements_text or requirement.get("remark"),
        "job_description": requirement.get("remark"),
        "salary_min": requirement.get("salaryMin"),
        "salary_max": requirement.get("salaryMax"),
        "salary_currency": requirement.get("salaryCurrency"),
    }


def _recruit_status(status: Any) -> str:
    if status == "ONLINE":
        return "正在招聘"
    if status == "DRAFT":
        return "草稿"
    if status == "OFFLINE":
        return "已下线"
    if status == "CLOSED":
        return "已关闭"
    return str(status or "")


def _tenant_id(args: dict, config: RecruitmentAPIConfig) -> str:
    return str(args.get("tenant_id") or args.get("tenantId") or config.tenant_id or "").strip()


def _user_id(args: dict, config: RecruitmentAPIConfig) -> str:
    return str(args.get("user_id") or args.get("userId") or config.user_id or "").strip()


def _string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _blank_to_none(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def _error(trace_id: str, error_code: str, message: str, *, intent: str = UNKNOWN_QUERY_INTENT, **extra: Any) -> str:
    payload = {
        "success": False,
        "intent": intent,
        "answer": message,
        "data": [],
        "safe": False,
        "trace_id": trace_id,
        "error_code": error_code,
        "message": message,
    }
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False, default=str)


class RecruitmentAPIError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: int | None = None,
        trace_id: str = "",
        response: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.trace_id = trace_id
        self.response = response or {}

    def to_payload(self, *, trace_id: str, intent: str) -> dict[str, Any]:
        return {
            "success": False,
            "intent": intent,
            "answer": self.message,
            "data": [],
            "safe": False,
            "trace_id": self.trace_id or trace_id,
            "error_code": self.code,
            "message": self.message,
            "http_status": self.status,
            "api_response": self.response or None,
        }


def query_recruitment_system_api(payload: dict[str, Any]) -> dict[str, Any]:
    """Public helper for HTTP surfaces that need the same API-only query path."""
    return json.loads(_query_handler(payload))


def create_recruitment_job_api(payload: dict[str, Any]) -> dict[str, Any]:
    """Public helper for HTTP surfaces that need the same API-only create path."""
    return json.loads(_create_job_handler(payload))


registry.register(
    name="recruitment_system_query",
    toolset="recruitment_system",
    schema={
        "name": "recruitment_system_query",
        "description": (
            "Query recruitmentSystem business data through the official HTTP API. "
            "Use for recruiting job list and job detail questions. Do not access MySQL directly."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_question": {"type": "string", "description": "Natural-language user question."},
                "question": {"type": "string", "description": "Alias for user_question."},
                "user_id": {"type": "string", "description": "Current authenticated user ID."},
                "tenant_id": {"type": "string", "description": "Current tenant ID. Sent as X-Tenant-Id."},
                "job_name": {"type": "string", "description": "Optional job name for detail lookup."},
                "job_id": {"type": "integer", "description": "Optional job ID for detail lookup."},
                "status": {"type": "string", "description": "Job status filter, e.g. ONLINE."},
                "keyword": {"type": "string", "description": "Keyword filter for job name or code."},
                "department": {"type": "string", "description": "Department filter."},
                "page_no": {"type": "integer", "description": "Page number, default 1."},
                "page_size": {"type": "integer", "description": "Page size, max 100."},
            },
            "required": ["user_question"],
        },
    },
    handler=_query_handler,
    check_fn=_is_recruitment_configured,
    requires_env=["RECRUITMENT_API_BASE_URL"],
    description="RecruitmentSystem API query tool",
    emoji="🧾",
)


registry.register(
    name="recruitment_system_create_job",
    toolset="recruitment_system",
    schema={
        "name": "recruitment_system_create_job",
        "description": (
            "Create a recruitmentSystem job through the official HTTP API. "
            "Creates the job, writes requirements, and publishes it by default. Do not access MySQL directly."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "job_name": {"type": "string", "description": "Job title/name to create."},
                "tenant_id": {"type": "string", "description": "Current tenant ID. Sent as X-Tenant-Id."},
                "user_id": {"type": "string", "description": "Current authenticated user ID. Sent as X-User-Id."},
                "job_code": {"type": "string", "description": "Optional job code; omit to let API generate it."},
                "department": {"type": "string", "description": "Department, default from env or 研发中心."},
                "employment_type": {"type": "string", "description": "FULL_TIME, PART_TIME, INTERN, CONTRACT, OUTSOURCING."},
                "work_location": {"type": "string", "description": "Work location, default from env or 上海."},
                "headcount": {"type": "integer", "description": "Headcount, default 1."},
                "owner_user_id": {"type": "string", "description": "Recruitment owner user ID."},
                "owner_user_name": {"type": "string", "description": "Recruitment owner display name."},
                "publish": {"type": "boolean", "description": "Whether to publish/online after creation. Default true."},
                "allow_duplicate": {"type": "boolean", "description": "Allow creating when an exact same job name exists."},
                "education_requirement": {"type": "string", "description": "Default UNLIMITED."},
                "experience_min_years": {"type": "integer", "description": "Default 0."},
                "experience_max_years": {"type": "integer", "description": "Default 0."},
                "skill_requirements": {"type": "array", "items": {"type": "string"}},
                "soft_skill_requirements": {"type": "array", "items": {"type": "string"}},
                "salary_min": {"type": "number", "description": "Default 0."},
                "salary_max": {"type": "number", "description": "Default 0."},
                "salary_currency": {"type": "string", "description": "Default CNY."},
                "remark": {"type": "string", "description": "Optional requirement remark."},
            },
            "required": ["job_name"],
        },
    },
    handler=_create_job_handler,
    check_fn=_is_recruitment_configured,
    requires_env=["RECRUITMENT_API_BASE_URL"],
    description="RecruitmentSystem API job creation tool",
    emoji="🧾",
)


registry.register(
    name="recruitment_system_health",
    toolset="recruitment_system",
    schema={
        "name": "recruitment_system_health",
        "description": "Check RecruitmentSystem HTTP API configuration and connectivity.",
        "parameters": {"type": "object", "properties": {}},
    },
    handler=_health_handler,
    description="RecruitmentSystem API health check",
    emoji="🧾",
)

---
sidebar_position: 2
title: "Adding Tools"
description: "Hermes 에이전트에 새로운 도구를 추가하는 방법 - 스키마, 핸들러, 등록 및 도구 세트"
---

# 도구 추가하기

도구를 작성하기 전에 스스로에게 물어보세요: **대신 [스킬](creating-skills.md)이 되어야 하지 않을까요?**

:::warning 내장 핵심 도구 전용
이 페이지는 저장소 자체에 **내장 Hermes 도구**를 추가하기 위한 것입니다.
만약 Hermes 핵심 코드를 수정하지 않고 개인적, 프로젝트 로컬, 혹은 다른 방식의 커스텀 도구를 원한다면, 플러그인 방식을 사용하세요:

- [플러그인](/user-guide/features/plugins)
- [Hermes 플러그인 빌드하기](/guides/build-a-hermes-plugin)

대부분의 커스텀 도구 제작 시에는 플러그인을 기본으로 사용하세요. 명시적으로 `tools/`와 `toolsets.py`에 새로운 내장 도구를 탑재하려는 경우에만 이 페이지를 따르세요.
:::

기능을 지침(instructions) + 셸 명령어 + 기존 도구(arXiv 검색, git 워크플로우, Docker 관리, PDF 처리)로 표현할 수 있다면 **스킬(Skill)**로 만드세요.

API 키와의 엔드투엔드(end-to-end) 통합, 사용자 지정 처리 로직, 바이너리 데이터 처리 또는 스트리밍(브라우저 자동화, TTS, 비전 분석)이 필요하다면 **도구(Tool)**로 만드세요.

## 개요

도구 추가는 **2개의 파일**을 건드립니다:

1. **`tools/your_tool.py`** — 핸들러, 스키마, 검사 함수, `registry.register()` 호출
2. **`toolsets.py`** — 도구 이름을 `_HERMES_CORE_TOOLS` (또는 특정 도구 세트)에 추가

최상위에 `registry.register()` 호출이 있는 모든 `tools/*.py` 파일은 시작 시 자동으로 검색됩니다 — 수동으로 가져오기(import) 목록을 관리할 필요가 없습니다.

## 1단계: 내장 도구 파일 생성

모든 도구 파일은 동일한 구조를 따릅니다:

```python
# tools/weather_tool.py
"""Weather Tool -- 위치에 대한 현재 날씨를 조회합니다."""

import json
import os
import logging

logger = logging.getLogger(__name__)


# --- 가용성 검사 (Availability check) ---

def check_weather_requirements() -> bool:
    """도구의 종속성을 사용할 수 있는 경우 True를 반환합니다."""
    return bool(os.getenv("WEATHER_API_KEY"))


# --- 핸들러 (Handler) ---

def weather_tool(location: str, units: str = "metric") -> str:
    """위치에 대한 날씨를 가져옵니다. JSON 문자열을 반환합니다."""
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return json.dumps({"error": "WEATHER_API_KEY not configured"})
    try:
        # ... 날씨 API 호출 ...
        return json.dumps({"location": location, "temp": 22, "units": units})
    except Exception as e:
        return json.dumps({"error": str(e)})


# --- 스키마 (Schema) ---

WEATHER_SCHEMA = {
    "name": "weather",
    "description": "Get current weather for a location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name or coordinates (e.g. 'London' or '51.5,-0.1')"
            },
            "units": {
                "type": "string",
                "enum": ["metric", "imperial"],
                "description": "Temperature units (default: metric)",
                "default": "metric"
            }
        },
        "required": ["location"]
    }
}


# --- 등록 (Registration) ---

from tools.registry import registry

registry.register(
    name="weather",
    toolset="weather",
    schema=WEATHER_SCHEMA,
    handler=lambda args, **kw: weather_tool(
        location=args.get("location", ""),
        units=args.get("units", "metric")),
    check_fn=check_weather_requirements,
    requires_env=["WEATHER_API_KEY"],
)
```

### 주요 규칙

:::danger 중요
- 핸들러는 원시 딕셔너리가 아닌 **반드시** JSON 문자열(`json.dumps()`를 통해)을 반환해야 합니다.
- 오류는 예외를 발생시키지 말고 **반드시** `{"error": "메시지"}` 형태로 반환해야 합니다.
- `check_fn`은 도구 정의를 구성할 때 호출됩니다 — `False`를 반환하면 도구는 조용히 제외됩니다.
- `handler`는 `(args: dict, **kwargs)`를 받으며, 여기서 `args`는 LLM의 도구 호출 인수입니다.
:::

## 2단계: 도구 세트에 내장 도구 추가

`toolsets.py`에서 도구 이름을 추가합니다:

```python
# 모든 플랫폼(CLI + 메시징)에서 사용할 수 있어야 하는 경우:
_HERMES_CORE_TOOLS = [
    ...
    "weather",  # <-- 여기에 추가
]

# 또는 새로운 독립 실행형 도구 세트를 생성하는 경우:
"weather": {
    "description": "날씨 조회 도구",
    "tools": ["weather"],
    "includes": []
},
```

## ~~3단계: 검색 경로(Import) 추가~~ (더 이상 필요하지 않음)

최상위 `registry.register()` 호출이 있는 도구 모듈은 `tools/registry.py`의 `discover_builtin_tools()`에 의해 자동으로 검색됩니다. 수동으로 유지 관리할 가져오기 목록이 없습니다 — `tools/`에 파일을 생성하기만 하면 시작 시 선택됩니다.

## 비동기(Async) 핸들러

핸들러에 비동기 코드가 필요한 경우, `is_async=True`로 표시합니다:

```python
async def weather_tool_async(location: str) -> str:
    async with aiohttp.ClientSession() as session:
        ...
    return json.dumps(result)

registry.register(
    name="weather",
    toolset="weather",
    schema=WEATHER_SCHEMA,
    handler=lambda args, **kw: weather_tool_async(args.get("location", "")),
    check_fn=check_weather_requirements,
    is_async=True,  # 레지스트리가 _run_async()를 자동으로 호출합니다
)
```

레지스트리는 비동기 브리징을 투명하게 처리합니다 — 직접 `asyncio.run()`을 호출하지 마세요.

## task_id가 필요한 핸들러

세션별 상태를 관리하는 도구는 `**kwargs`를 통해 `task_id`를 받습니다:

```python
def _handle_weather(args, **kw):
    task_id = kw.get("task_id")
    return weather_tool(args.get("location", ""), task_id=task_id)

registry.register(
    name="weather",
    ...
    handler=_handle_weather,
)
```

## 에이전트 루프에서 가로채는 도구

일부 도구(`todo`, `memory`, `session_search`, `delegate_task`)는 세션별 에이전트 상태에 접근해야 합니다. 이들은 레지스트리에 도달하기 전에 `run_agent.py`에 의해 가로채어집니다. 레지스트리는 여전히 스키마를 보유하지만, 가로채기를 우회한 경우 `dispatch()`가 폴백(fallback) 오류를 반환합니다.

## 선택 사항: 설정 마법사 통합

도구에 API 키가 필요한 경우 `hermes_cli/config.py`에 추가하세요:

```python
OPTIONAL_ENV_VARS = {
    ...
    "WEATHER_API_KEY": {
        "description": "날씨 조회를 위한 날씨 API 키",
        "prompt": "Weather API key",
        "url": "https://weatherapi.com/",
        "tools": ["weather"],
        "password": True,
    },
}
```

## 체크리스트

- [ ] 핸들러, 스키마, 검사 함수 및 등록을 포함하여 도구 파일 생성됨
- [ ] `toolsets.py`의 적절한 도구 세트에 추가됨
- [ ] 이것이 플러그인이 아닌 진정한 내장/핵심 도구여야 함을 확인했음
- [ ] 핸들러가 JSON 문자열을 반환하고 오류는 `{"error": "..."}`로 반환함
- [ ] 선택 사항: API 키가 `hermes_cli/config.py`의 `OPTIONAL_ENV_VARS`에 추가됨
- [ ] 선택 사항: 일괄 처리를 위해 `toolset_distributions.py`에 추가됨
- [ ] `hermes chat -q "Use the weather tool for London"` 명령어로 테스트 완료

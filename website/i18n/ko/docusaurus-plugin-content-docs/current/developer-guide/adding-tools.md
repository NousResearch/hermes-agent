---
sidebar_position: 2
title: "도구 추가"
description: "Hermes Agent에 새 도구를 추가하는 방법 — 스키마, 핸들러, 등록, 도구 세트"
---

# 도구 추가

도구를 작성하기 전에 스스로 물어보세요. **대신 [스킬](creating-skills.md)로 만들어야 하지 않을까요?**

:::warning 내장 코어 도구 전용
이 페이지는 저장소 자체에 **내장 Hermes 도구**를 추가하는 방법을 설명합니다.
Hermes 코어를 수정하지 않고 개인용·프로젝트 로컬·기타 사용자 지정 도구를 만들고 싶다면 대신 플러그인 방식을 사용하세요:

- [플러그인](/user-guide/features/plugins)
- [Hermes 플러그인 빌드](/developer-guide/plugins)

대부분의 사용자 지정 도구는 플러그인을 기본으로 선택하세요. `tools/`와 `toolsets.py`에 새 내장 도구를 명시적으로 포함하려는 경우에만 이 페이지를 따르세요.
:::

기능을 지침 + 셸 명령 + 기존 도구로 표현할 수 있다면 **스킬**로 만드세요(arXiv 검색, git 워크플로, Docker 관리, PDF 처리).

API 키와의 종단 간 통합, 사용자 지정 처리 로직, 바이너리 데이터 처리 또는 스트리밍이 필요하다면 **도구**로 만드세요(브라우저 자동화, TTS, 비전 분석).

## 개요

도구를 추가하려면 **2개 파일**을 수정합니다:

1. **`tools/your_tool.py`** — 핸들러, 스키마, 검사 함수, `registry.register()` 호출
2. **`toolsets.py`** — `_HERMES_CORE_TOOLS`(또는 특정 도구 세트)에 도구 이름 추가

최상위 수준에서 `registry.register()`를 호출하는 모든 `tools/*.py` 파일은 시작 시 자동으로 탐색됩니다. 수동 import 목록을 관리할 필요가 없습니다.

## 1단계: 내장 도구 파일 만들기

모든 도구 파일은 동일한 구조를 따릅니다:

```python
# tools/weather_tool.py
"""Weather Tool -- look up current weather for a location."""

import json
import os
import logging

logger = logging.getLogger(__name__)


# --- Availability check ---

def check_weather_requirements() -> bool:
    """Return True if the tool's dependencies are available."""
    return bool(os.getenv("WEATHER_API_KEY"))


# --- Handler ---

def weather_tool(location: str, units: str = "metric") -> str:
    """Fetch weather for a location. Returns JSON string."""
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return json.dumps({"error": "WEATHER_API_KEY not configured"})
    try:
        # ... call weather API ...
        return json.dumps({"location": location, "temp": 22, "units": units})
    except Exception as e:
        return json.dumps({"error": str(e)})


# --- Schema ---

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


# --- Registration ---

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

### 핵심 규칙

:::danger 중요
- 핸들러는 원시 dict가 아니라 반드시 JSON 문자열(`json.dumps()` 사용)을 반환해야 합니다
- 오류는 예외로 발생시키지 말고 반드시 `{"error": "message"}` 형태로 반환해야 합니다
- 도구 정의를 만들 때 `check_fn`이 호출됩니다 — `False`를 반환하면 도구가 조용히 제외됩니다
- `handler`는 `(args: dict, **kwargs)`를 받으며, `args`는 LLM의 도구 호출 인자입니다
:::

## 2단계: 도구 세트에 내장 도구 추가

`toolsets.py`에 도구 이름을 추가합니다:

```python
# If it should be available on all platforms (CLI + messaging):
_HERMES_CORE_TOOLS = [
    ...
    "weather",  # <-- add here
]

# Or create a new standalone toolset:
"weather": {
    "description": "Weather lookup tools",
    "tools": ["weather"],
    "includes": []
},
```

## ~~3단계: 탐색 import 추가~~ (더 이상 필요하지 않음)

최상위 수준에서 `registry.register()`를 호출하는 도구 모듈은 `tools/registry.py`의 `discover_builtin_tools()`가 자동으로 탐색합니다. 수동 import 목록을 유지할 필요가 없습니다. `tools/`에 파일을 만들기만 하면 시작 시 자동으로 선택됩니다.

## 비동기 핸들러

핸들러에 비동기 코드가 필요하다면 `is_async=True`로 표시합니다:

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
    is_async=True,  # registry calls _run_async() automatically
)
```

레지스트리가 비동기 연결을 투명하게 처리하므로 직접 `asyncio.run()`을 호출할 필요가 없습니다.

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

일부 도구(`todo`, `memory`, `session_search`, `delegate_task`)는 세션별 에이전트 상태에 접근해야 합니다. 이러한 도구는 레지스트리에 도달하기 전에 `run_agent.py`에서 가로챕니다. 레지스트리에는 여전히 해당 스키마가 보관되지만, 가로채기가 우회되면 `dispatch()`가 대체 오류를 반환합니다.

## 선택 사항: 설정 마법사 통합

도구에 API 키가 필요하다면 `hermes_cli/config.py`에 추가합니다:

```python
OPTIONAL_ENV_VARS = {
    ...
    "WEATHER_API_KEY": {
        "description": "Weather API key for weather lookup",
        "prompt": "Weather API key",
        "url": "https://weatherapi.com/",
        "tools": ["weather"],
        "password": True,
    },
}
```

## 체크리스트

- [ ] 핸들러, 스키마, 검사 함수, 등록 코드가 포함된 도구 파일 생성
- [ ] `toolsets.py`의 적절한 도구 세트에 추가
- [ ] 이것이 플러그인이 아닌 정말 내장/코어 도구여야 하는지 확인
- [ ] 핸들러가 JSON 문자열을 반환하고 오류가 `{"error": "..."}`로 반환되는지 확인
- [ ] 선택 사항: `hermes_cli/config.py`의 `OPTIONAL_ENV_VARS`에 API 키 추가
- [ ] 선택 사항: 일괄 처리용 `toolset_distributions.py`에 추가
- [ ] `hermes chat -q "Use the weather tool for London"`으로 테스트

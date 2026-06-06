---
sidebar_position: 9
sidebar_label: "플러그인 만들기 (Build a Plugin)"
title: "Hermes 플러그인 만들기 (Build a Hermes Plugin)"
description: "도구, 훅, 데이터 파일 및 스킬을 포함하는 완전한 Hermes 플러그인을 구축하기 위한 단계별 가이드"
---

# Hermes 플러그인 만들기 (Build a Hermes Plugin)

이 가이드는 바닥부터 시작하여 완전한 Hermes 플러그인을 구축하는 과정을 안내합니다. 이 가이드를 마치고 나면, 여러 도구, 수명 주기(lifecycle) 훅, 동봉된 데이터 파일, 그리고 번들로 포함된 스킬을 갖춘 작동하는 플러그인(플러그인 시스템이 지원하는 모든 것)을 갖게 될 것입니다.

:::info 어떤 가이드가 필요한지 확실하지 않으신가요?
Hermes에는 여러 가지의 명확한 플러그 가능한 인터페이스(pluggable interfaces)가 있습니다 — 어떤 것들은 Python `register_*` API를 사용하고, 다른 것들은 설정 기반이거나 지정된 폴더에 떨구기만 하면 되는 방식(drop-in)입니다. 먼저 이 지도를 참고하세요:

| 추가하려는 기능이... | 이 문서를 읽어보세요 |
|---|---|
| 사용자 정의 도구, 훅, 슬래시 명령어, 스킬, 또는 CLI 하위 명령어 | **이 가이드** (일반적인 플러그인 영역) |
| **LLM / 추론 백엔드** (새로운 제공자) | [모델 제공자 플러그인](/developer-guide/model-provider-plugin) |
| **게이트웨이 채널** (Discord/Telegram/IRC/Teams 등) | [플랫폼 어댑터 추가하기](/developer-guide/adding-platform-adapters) |
| **메모리 백엔드** (Honcho/Mem0/Supermemory 등) | [메모리 제공자 플러그인](/developer-guide/memory-provider-plugin) |
| **컨텍스트 압축 엔진** | [컨텍스트 엔진 플러그인](/developer-guide/context-engine-plugin) |
| **이미지 생성 백엔드** | [이미지 생성 제공자 플러그인](/developer-guide/image-gen-provider-plugin) |
| **비디오 생성 백엔드** | [비디오 생성 제공자 플러그인](/developer-guide/video-gen-provider-plugin) |
| **TTS 백엔드** (Piper, VoxCPM, Kokoro, 음성 복제 등 모든 CLI) | [TTS 사용자 정의 명령어 제공자](/user-guide/features/tts#custom-command-providers) — 구성 기반, Python 불필요 |
| **STT 백엔드** (사용자 지정 whisper / ASR CLI) | [음성 메시지 전사(STT)](/user-guide/features/tts#voice-message-transcription-stt) — `HERMES_LOCAL_STT_COMMAND`를 쉘 템플릿으로 설정 |
| **MCP를 통한 외부 도구** (파일 시스템, GitHub, Linear, 모든 MCP 서버) | [MCP](/user-guide/features/mcp) — `config.yaml`에 `mcp_servers.<name>` 선언 |
| **게이트웨이 이벤트 훅** (시작 시, 세션 이벤트, 명령어 실행 시 발동) | [이벤트 훅](/user-guide/features/hooks#gateway-event-hooks) — `~/.hermes/hooks/<name>/`에 `HOOK.yaml` + `handler.py` 놓기 |
| **쉘 훅** (이벤트 발생 시 쉘 명령어 실행) | [쉘 훅](/user-guide/features/hooks#shell-hooks) — `config.yaml`의 `hooks:` 아래에 선언 |
| **추가적인 스킬 소스** (사용자 지정 GitHub 저장소, 프라이빗 스킬 인덱스) | [스킬](/user-guide/features/skills) — `hermes skills tap add <repo>` · [tap 퍼블리싱](/user-guide/features/skills#publishing-a-custom-skill-tap) |
| 퍼스트 클래스 **핵심** 추론 제공자 (플러그인이 아님) | [제공자 추가하기](/developer-guide/adding-providers) |

설정 기반(TTS, STT, MCP, 쉘 훅) 및 드롭인 디렉터리(게이트웨이 훅) 스타일을 포함한 모든 확장 영역(extension surface)에 대한 통합된 보기는 전체 [플러그 가능한 인터페이스 표](/user-guide/features/plugins#pluggable-interfaces--where-to-go-for-each)를 확인하세요.
:::

## 우리가 만들 것

두 가지 도구를 제공하는 **계산기(calculator)** 플러그인:
- `calculate` — 수식 평가 (`2**16`, `sqrt(144)`, `pi * 5**2`)
- `unit_convert` — 단위 변환 (`100 F → 37.78 C`, `5 km → 3.11 mi`)

여기에 모든 도구 호출을 로깅하는 훅 하나, 그리고 번들 스킬 파일 하나를 더해볼 것입니다.

## 1단계: 플러그인 디렉터리 생성

```bash
mkdir -p ~/.hermes/plugins/calculator
cd ~/.hermes/plugins/calculator
```

## 2단계: 매니페스트(manifest) 작성

`plugin.yaml` 파일을 만듭니다:

```yaml
name: calculator
version: 1.0.0
description: Math calculator — evaluate expressions and convert units
provides_tools:
  - calculate
  - unit_convert
provides_hooks:
  - post_tool_call
```

이 파일은 Hermes에게 다음을 알려줍니다: "나는 calculator라는 플러그인이고, 도구와 훅을 제공해." `provides_tools`와 `provides_hooks` 필드는 플러그인이 등록할 항목들의 목록입니다.

선택적으로 추가할 수 있는 필드들:
```yaml
author: Your Name
requires_env:          # 환경 변수가 있어야 로드됨; 설치 중 프롬프트로 입력받음
  - SOME_API_KEY       # 단순 형식 — 없으면 플러그인이 비활성화됨
  - name: OTHER_KEY    # 풍부한(rich) 형식 — 설치 중 설명/url 표시
    description: "Key for the Other service"
    url: "https://other.com/keys"
    secret: true
```

## 3단계: 도구 스키마 작성

`schemas.py` 파일을 만듭니다 — 이것은 LLM이 언제 여러분의 도구를 호출할지 결정하기 위해 읽는 내용입니다:

```python
"""Tool schemas — what the LLM sees."""

CALCULATE = {
    "name": "calculate",
    "description": (
        "수학 표현식을 평가하고 결과를 반환합니다. "
        "산술 연산(+, -, *, /, **), 함수(sqrt, sin, cos, "
        "log, abs, round, floor, ceil), 그리고 상수(pi, e)를 지원합니다. "
        "사용자가 수학과 관련된 질문을 할 때 이 도구를 사용하세요."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "평가할 수학 표현식 (예: '2**10', 'sqrt(144)')",
            },
        },
        "required": ["expression"],
    },
}

UNIT_CONVERT = {
    "name": "unit_convert",
    "description": (
        "단위 간에 값을 변환합니다. 길이(m, km, mi, ft, in), "
        "무게(kg, lb, oz, g), 온도(C, F, K), 데이터(B, KB, MB, GB, TB), "
        "그리고 시간(s, min, hr, day)을 지원합니다."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "value": {
                "type": "number",
                "description": "변환할 숫자 값",
            },
            "from_unit": {
                "type": "string",
                "description": "원본 단위 (예: 'km', 'lb', 'F', 'GB')",
            },
            "to_unit": {
                "type": "string",
                "description": "목표 단위 (예: 'mi', 'kg', 'C', 'MB')",
            },
        },
        "required": ["value", "from_unit", "to_unit"],
    },
}
```

**스키마가 중요한 이유:** LLM은 `description` 필드를 보고 여러분의 도구를 사용할 시기를 결정합니다. 도구가 무엇을 하는지, 언제 사용해야 하는지 구체적으로 명시하세요. `parameters`는 LLM이 전달할 인자(arguments)를 정의합니다.

## 4단계: 도구 핸들러 작성

`tools.py` 파일을 만듭니다 — 이 파일은 LLM이 여러분의 도구를 호출할 때 실제로 실행될 코드입니다:

```python
"""Tool handlers — the code that runs when the LLM calls each tool."""

import json
import math

# 수식 평가를 위한 안전한 전역 객체 — 파일/네트워크 액세스 불가
_SAFE_MATH = {
    "abs": abs, "round": round, "min": min, "max": max,
    "pow": pow, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
    "tan": math.tan, "log": math.log, "log2": math.log2, "log10": math.log10,
    "floor": math.floor, "ceil": math.ceil,
    "pi": math.pi, "e": math.e,
    "factorial": math.factorial,
}


def calculate(args: dict, **kwargs) -> str:
    """수식을 안전하게 평가합니다.

    핸들러 규칙:
    1. args (dict) 수신 — LLM이 전달한 매개변수
    2. 작업 수행
    3. JSON 문자열 반환 — 오류 발생 시에도 **항상** 반환
    4. 향후 호환성을 위해 **kwargs 허용
    """
    expression = args.get("expression", "").strip()
    if not expression:
        return json.dumps({"error": "No expression provided"})

    try:
        result = eval(expression, {"__builtins__": {}}, _SAFE_MATH)
        return json.dumps({"expression": expression, "result": result})
    except ZeroDivisionError:
        return json.dumps({"expression": expression, "error": "Division by zero"})
    except Exception as e:
        return json.dumps({"expression": expression, "error": f"Invalid: {e}"})


# 변환 표 — 값은 기본 단위(base unit) 기준입니다
_LENGTH = {"m": 1, "km": 1000, "mi": 1609.34, "ft": 0.3048, "in": 0.0254, "cm": 0.01}
_WEIGHT = {"kg": 1, "g": 0.001, "lb": 0.453592, "oz": 0.0283495}
_DATA = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
_TIME = {"s": 1, "ms": 0.001, "min": 60, "hr": 3600, "day": 86400}


def _convert_temp(value, from_u, to_u):
    # 섭씨로 정규화
    c = {"F": (value - 32) * 5/9, "K": value - 273.15}.get(from_u, value)
    # 목표 단위로 변환
    return {"F": c * 9/5 + 32, "K": c + 273.15}.get(to_u, c)


def unit_convert(args: dict, **kwargs) -> str:
    """단위 간 변환을 수행합니다."""
    value = args.get("value")
    from_unit = args.get("from_unit", "").strip()
    to_unit = args.get("to_unit", "").strip()

    if value is None or not from_unit or not to_unit:
        return json.dumps({"error": "Need value, from_unit, and to_unit"})

    try:
        # 온도 변환
        if from_unit.upper() in {"C","F","K"} and to_unit.upper() in {"C","F","K"}:
            result = _convert_temp(float(value), from_unit.upper(), to_unit.upper())
            return json.dumps({"input": f"{value} {from_unit}", "result": round(result, 4),
                             "output": f"{round(result, 4)} {to_unit}"})

        # 비율 기반 변환
        for table in (_LENGTH, _WEIGHT, _DATA, _TIME):
            lc = {k.lower(): v for k, v in table.items()}
            if from_unit.lower() in lc and to_unit.lower() in lc:
                result = float(value) * lc[from_unit.lower()] / lc[to_unit.lower()]
                return json.dumps({"input": f"{value} {from_unit}",
                                 "result": round(result, 6),
                                 "output": f"{round(result, 6)} {to_unit}"})

        return json.dumps({"error": f"Cannot convert {from_unit} → {to_unit}"})
    except Exception as e:
        return json.dumps({"error": f"Conversion failed: {e}"})
```

**핸들러 핵심 규칙:**
1. **서명(Signature):** `def my_handler(args: dict, **kwargs) -> str`
2. **반환값(Return):** 성공이든 실패이든, 항상 JSON 문자열을 반환해야 합니다.
3. **에러 발생 금지(Never raise):** 모든 예외를 잡아서(catch) 대신 오류 JSON을 반환하세요.
4. **`**kwargs` 허용:** Hermes가 향후 추가적인 컨텍스트를 전달할 수 있도록 합니다.

## 5단계: 등록 코드 작성

`__init__.py` 파일을 만듭니다 — 이 파일은 스키마와 핸들러를 연결해줍니다:

```python
"""Calculator plugin — registration."""

import logging

from . import schemas, tools

logger = logging.getLogger(__name__)

# 훅을 통해 도구 사용 내역 추적
_call_log = []

def _on_post_tool_call(tool_name, args, result, task_id, **kwargs):
    """Hook: (우리가 만든 도구뿐만 아니라) 모든 도구 호출 후에 실행됩니다."""
    _call_log.append({"tool": tool_name, "session": task_id})
    if len(_call_log) > 100:
        _call_log.pop(0)
    logger.debug("Tool called: %s (session %s)", tool_name, task_id)


def register(ctx):
    """스키마를 핸들러에 연결하고 훅을 등록합니다."""
    ctx.register_tool(name="calculate",    toolset="calculator",
                      schema=schemas.CALCULATE,    handler=tools.calculate)
    ctx.register_tool(name="unit_convert", toolset="calculator",
                      schema=schemas.UNIT_CONVERT, handler=tools.unit_convert)

    # 이 훅은 우리가 만든 도구만이 아니라 모든 도구 호출에 대해 발동합니다.
    ctx.register_hook("post_tool_call", _on_post_tool_call)
```

**`register()`가 하는 일:**
- 시작 시 정확히 한 번 호출됩니다.
- `ctx.register_tool()`은 여러분의 도구를 레지스트리에 넣습니다 — 모델은 이를 즉시 볼 수 있습니다.
- `ctx.register_hook()`은 수명 주기 이벤트(lifecycle events)를 구독합니다.
- `ctx.register_cli_command()`는 CLI 하위 명령어를 등록합니다 (예: `hermes my-plugin <subcommand>`).
- `ctx.register_command()`는 세션 내 슬래시 명령어를 등록합니다 (예: CLI / 게이트웨이 채팅 안에서 `/myplugin <args>`) — 아래의 [슬래시 명령어 등록](#register-slash-commands) 섹션을 참조하세요.
- `ctx.dispatch_tool(name, arguments)` — 다른 모든 도구(기본 내장 도구나 다른 플러그인의 도구 등)를 상위 에이전트의 컨텍스트(승인 상태, 자격 증명, task_id 등)가 자동으로 설정된 상태로 호출할 수 있습니다. 슬래시 명령어 핸들러에서 모델이 직접 호출한 것처럼 `terminal`, `read_file` 등의 도구를 호출해야 할 때 유용합니다.
- 만약 이 함수에서 충돌(crash)이 나면 해당 플러그인만 비활성화되고 Hermes 자체는 정상적으로 계속 작동합니다.

**`dispatch_tool` 예제 — 도구를 실행하는 슬래시 명령어:**

```python
def handle_scan(ctx, argstr):
    """레지스트리를 통해 terminal 도구를 호출하여 /scan을 구현합니다."""
    result = ctx.dispatch_tool("terminal", {"command": f"find . -name '{argstr}'"})
    return result  # 호출자의 채팅 UI로 반환됨

def register(ctx):
    ctx.register_command("scan", handle_scan, help="Find files matching a glob")
```

이렇게 디스패치된(dispatched) 도구는 정상적인 승인(approval), 데이터 마스킹(redaction), 예산(budget) 파이프라인을 거칩니다 — 우회로가 아니라 실제 도구를 호출하는 것과 똑같이 취급됩니다.

## 6단계: 테스트

Hermes를 시작합니다:

```bash
hermes
```

상단 배너의 도구 목록에서 `calculator: calculate, unit_convert`를 확인할 수 있어야 합니다.

다음과 같은 프롬프트로 시도해 보세요:
```
2의 16승은 무엇인가요?
100 화씨를 섭씨로 변환해줘
2 곱하기 파이(pi)의 제곱근은?
1.5 테라바이트는 몇 기가바이트야?
```

플러그인 상태 확인:
```
/plugins
```

출력:
```
Plugins (1):
  ✓ calculator v1.0.0 (2 tools, 1 hooks)
```

### 플러그인 검색 디버깅

플러그인이 표시되지 않거나 — 표시되는데 로드되지 않는 경우 — `HERMES_PLUGINS_DEBUG=1`을 설정하여 stderr에서 상세한 검색 로그를 얻을 수 있습니다:

```bash
HERMES_PLUGINS_DEBUG=1 hermes plugins list
```

모든 플러그인 소스(번들, 사용자, 프로젝트, 진입점(entry-points))에 대해 다음 정보들을 확인할 수 있습니다:

- 어떤 디렉터리들을 스캔했고 각기 매니페스트(manifest)를 몇 개 반환했는지
- 매니페스트 별 정보: 해결된 키(resolved key), 이름, 종류(kind), 출처(source), 디스크 상의 경로
- 스킵된 이유: `disabled via config(설정에서 비활성화됨)`, `not enabled in config(설정에서 켜져있지 않음)`, `exclusive plugin(독점 플러그인)`, `no plugin.yaml, depth cap reached(plugin.yaml 없음, 검색 깊이 한계 도달)`
- 로드 시: 가져오고(import) 있는 플러그인, 그리고 `register(ctx)`가 등록한 내용들(도구, 훅, 슬래시 명령어, CLI 명령어)에 대한 한 줄 요약
- 구문 분석 실패 시: 예외(YAML 스캐너 오류 등)에 대한 전체 추적 내역(traceback)
- `register()` 실패 시: 예외를 발생시킨 `__init__.py` 내부의 줄을 가리키는 전체 추적 내역

환경 변수가 설정되면 동일한 로그가 항상 `~/.hermes/logs/agent.log` 파일에도 기록됩니다. (실패 시에만 WARNING 수준으로 기록되고, 모든 것은 DEBUG 수준으로 기록됨). 따라서 게이트웨이 내부 등에서 환경 변수를 넣고 실행하기 힘든 경우 대신 로그 파일을 통해 확인하세요:

```bash
hermes logs --level WARNING | grep -i plugin
```

플러그인이 표시되지 않는 일반적인 이유:

- **설정에서 활성화되지 않음** — 플러그인은 옵트인(opt-in) 방식입니다. `hermes plugins enable <name>` 명령을 실행하세요 (이름은 `plugins list` 출력에서 가져올 수 있으며, 중첩된 레이아웃의 경우 `<category>/<plugin>` 형식이 될 수 있습니다).
- **잘못된 디렉터리 구조** — `~/.hermes/plugins/<plugin-name>/plugin.yaml` (단일 계층) 이거나 `~/.hermes/plugins/<category>/<plugin-name>/plugin.yaml` (최대 1단계 카테고리 중첩) 형태여야 합니다. 그보다 더 깊은 경로는 무시됩니다.
- **`__init__.py` 누락** — 플러그인 디렉터리에는 `plugin.yaml` 파일과 `register(ctx)` 함수가 있는 `__init__.py` 파일이 모두 필요합니다.
- **잘못된 종류(`kind`)** — 게이트웨이 어댑터의 경우 매니페스트에 `kind: platform`이 명시되어야 합니다. 메모리 제공자는 `kind: exclusive`로 자동 감지되며, `plugins.enabled` 항목이 아니라 `memory.provider` 설정을 통해 라우팅됩니다.

## 완성된 플러그인의 최종 구조

```
~/.hermes/plugins/calculator/
├── plugin.yaml      # "나는 calculator라는 플러그인이고, 도구와 훅을 제공해."
├── __init__.py      # 연결: 스키마 → 핸들러, 훅 등록
├── schemas.py       # LLM이 읽게 될 내용 (설명 + 매개변수 스펙)
└── tools.py         # 실행될 실제 로직 (calculate, unit_convert 함수들)
```

4개의 파일, 명확한 역할 분리:
- **매니페스트 (Manifest)**: 플러그인이 무엇인지 선언합니다
- **스키마 (Schemas)**: LLM을 위해 도구들을 설명합니다
- **핸들러 (Handlers)**: 실제 로직을 구현합니다
- **등록 파일 (Registration)**: 모든 것을 하나로 연결합니다

## 플러그인으로 또 무엇을 할 수 있나요?

### 데이터 파일 동봉하기

플러그인 디렉터리에 파일을 넣고 가져오기(import) 시점에 읽을 수 있습니다:

```python
# tools.py 나 __init__.py 안에서
from pathlib import Path

_PLUGIN_DIR = Path(__file__).parent
_DATA_FILE = _PLUGIN_DIR / "data" / "languages.yaml"

with open(_DATA_FILE) as f:
    _DATA = yaml.safe_load(f)
```

### 스킬을 번들로 포함하기

플러그인은 에이전트가 `skill_view("plugin:skill")` 형태로 로드할 수 있는 스킬 파일을 동봉(ship)할 수 있습니다. `__init__.py`에서 등록하세요:

```
~/.hermes/plugins/my-plugin/
├── __init__.py
├── plugin.yaml
└── skills/
    ├── my-workflow/
    │   └── SKILL.md
    └── my-checklist/
        └── SKILL.md
```

```python
from pathlib import Path

def register(ctx):
    skills_dir = Path(__file__).parent / "skills"
    for child in sorted(skills_dir.iterdir()):
        skill_md = child / "SKILL.md"
        if child.is_dir() and skill_md.exists():
            ctx.register_skill(child.name, skill_md)
```

이제 에이전트는 플러그인의 스킬을 네임스페이스가 포함된 이름으로 로드할 수 있습니다:

```python
skill_view("my-plugin:my-workflow")   # → 플러그인의 버전
skill_view("my-workflow")              # → 내장 버전 (변함없음)
```

**핵심 속성:**
- 플러그인 스킬은 **읽기 전용**입니다 — `~/.hermes/skills/`에 들어가지 않으며 `skill_manage`를 통해 편집할 수 없습니다.
- 플러그인 스킬은 시스템 프롬프트의 `<available_skills>` 인덱스에 나열되지 **않습니다** — 명시적으로 호출해서 로드해야만 하는(opt-in) 옵션입니다.
- 네임스페이스가 없는 단순 스킬 이름은 영향을 받지 않습니다 — 네임스페이스가 있어 내장된 스킬과의 충돌을 막아줍니다.
- 에이전트가 플러그인 스킬을 로드할 때, 동일한 플러그인에서 제공되는 형제 스킬들을 나열하는 번들 컨텍스트 배너가 최상단에 붙습니다.

:::tip 레거시 패턴
기존의 `shutil.copy2` 패턴(스킬을 `~/.hermes/skills/`로 복사하는 방식)도 여전히 작동하지만 내장 스킬과 이름이 충돌할 위험이 있습니다. 새로운 플러그인의 경우 `ctx.register_skill()`을 사용하는 것이 좋습니다.
:::

### 환경 변수에 따른 조건부 실행(Gate)

플러그인에 API 키가 필요한 경우:

```yaml
# plugin.yaml — 단순 형식 (이전 버전과 호환됨)
requires_env:
  - WEATHER_API_KEY
```

`WEATHER_API_KEY`가 설정되어 있지 않으면 플러그인은 비활성화되며 명확한 메시지를 띄웁니다. 에이전트에 오류가 나거나 충돌이 발생하지 않고, 단지 "Plugin weather disabled (missing: WEATHER_API_KEY)"라는 안내만 보여줍니다.

사용자가 `hermes plugins install`을 실행할 때 누락된 `requires_env` 변수가 있다면 프롬프트를 통해 **대화형으로 입력**을 받습니다. 입력한 값은 `.env`에 자동으로 저장됩니다.

더 나은 설치 경험을 제공하려면 설명과 가입 URL이 있는 풍부한 형식을 사용하세요:

```yaml
# plugin.yaml — 풍부한 형식
requires_env:
  - name: WEATHER_API_KEY
    description: "API key for OpenWeather"
    url: "https://openweathermap.org/api"
    secret: true
```

| 필드 | 필수 여부 | 설명 |
|-------|----------|-------------|
| `name` | Yes | 환경 변수 이름 |
| `description` | No | 설치 프롬프트 도중 사용자에게 보여질 설명 |
| `url` | No | 자격 증명(키)을 얻을 수 있는 주소 |
| `secret` | No | `true`일 경우, 입력 시 화면에 보이지 않음 (비밀번호 필드처럼) |

두 형식 모두 같은 목록 안에서 혼합하여 사용할 수 있습니다. 이미 설정된 변수는 조용히 건너뜁니다.

### 선택적인 Python 의존성 지연 설치 (Lazy-install)

모든 사용자가 설치하고 있지는 않을 법한 SDK(벤더 SDK, 무거운 머신러닝 라이브러리, 특정 플랫폼 전용 패키지 등)를 플러그인이 래핑하는 경우, 모듈 최상단에서 `import`를 수행하지 마세요. 대신 도구 핸들러 내부에서 `tools.lazy_deps.ensure(...)` 도우미(helper)를 사용하세요 — Hermes는 사용자의 `security.allow_lazy_installs` 설정에 따라 해당 패키지를 처음 사용할 때 설치합니다.

```python
# tools.py
from tools.lazy_deps import ensure, FeatureUnavailable

def my_tool_handler(args, **kwargs):
    try:
        ensure("my-plugin.my-backend")   # 키(key)는 LAZY_DEPS에 있어야 합니다
    except FeatureUnavailable as exc:
        return {"error": str(exc)}

    import my_backend_sdk   # 이제 안전하게 임포트할 수 있습니다
    ...
```

`tools/lazy_deps.py` 보안 모델의 2가지 규칙:

| 규칙 | 이유 |
|---|---|
| 여러분의 기능(feature) 키는 소스 트리의 `LAZY_DEPS` 허용 목록에 나타나야 합니다. | 악성 설정이 Hermes를 조종하여 임의의 패키지를 설치하게 하는 것을 방지합니다 — Hermes가 자체적으로 배포하는 스펙(specs)만 설치 대상이 됩니다. |
| 스펙은 오직 PyPI에서 이름으로만(PyPI-by-name only) 가능합니다. | `--index-url`, `git+https://`, 또는 `file:` 경로는 불가능합니다. PEP 440 (`"my-sdk>=1.2,<2"`)을 사용해 허용 목록 내부에서 버전을 고정하세요. |

pip를 통해 배포되는 서드파티 플러그인의 경우, 개발자 본인의 `pyproject.toml` 안에서 선택적 의존성을 `[project.optional-dependencies]`의 extras로 선언하고 사용자에게 `pip install your-plugin[backend]`를 실행하라고 안내하세요 — 이 경로는 `lazy_deps`를 거치지 않습니다. 지연 설치(lazy-install) 방식은 무거운 의존성을 하드 코딩하여 배포하면 기본 Hermes의 크기가 너무 커지게 되는 **번들형(bundled)** 플러그인에 가장 유용합니다.

전역적으로 `security.allow_lazy_installs: false`가 설정되어 있다면, `ensure()`는 해결 방법을 제시하는 힌트와 함께 `FeatureUnavailable` 예외를 즉시 발생시킵니다 — 여러분의 플러그인은 도구 루프(tool loop)를 충돌시키지 말고 이 예외를 잡아서(catch) 대신 부드럽게 성능을 낮추어야 합니다 (오류 결과를 반환).

### 조건부 도구 가용성

선택적 라이브러리에 의존하는 도구의 경우:

```python
ctx.register_tool(
    name="my_tool",
    schema={...},
    handler=my_handler,
    check_fn=lambda: _has_optional_lib(),  # False면 모델에서 해당 도구가 숨겨짐
)
```

### 내장 도구 덮어쓰기 (Overriding)

기본 내장 도구를 여러분이 구현한 것으로 바꾸려면 (예: 기본 브라우저 도구를 headless Chrome CDP 백엔드로 바꾸거나, `web_search`를 커스텀 기업용 인덱스로 교체) `override=True`를 넘겨줍니다:

```python
def register(ctx):
    ctx.register_tool(
        name="browser_navigate",             # 내장 도구와 동일한 이름
        toolset="plugin_my_browser",         # 플러그인 고유의 도구 모음 네임스페이스
        schema={...},
        handler=my_custom_navigate,
        override=True,                       # 명시적 활성화 (opt-in)
    )
```

`override=True`가 없으면, 레지스트리는 의도치 않은 덮어쓰기를 방지하기 위해 다른 도구 모음에 있는 기존 도구 이름을 가리는(shadow) 어떤 등록 시도도 거부합니다. 오버라이드(override) 사실은 INFO 레벨에서 로깅되므로 `~/.hermes/logs/agent.log`에서 감사가 가능합니다. 플러그인은 기본 내장 도구보다 늦게 로드되므로 등록 순서가 올바릅니다: 여러분의 핸들러가 기본 내장 핸들러를 대체하게 됩니다.

### 여러 개의 훅 등록하기

```python
def register(ctx):
    ctx.register_hook("pre_tool_call", before_any_tool)
    ctx.register_hook("post_tool_call", after_any_tool)
    ctx.register_hook("pre_llm_call", inject_memory)
    ctx.register_hook("on_session_start", on_new_session)
    ctx.register_hook("on_session_end", on_session_end)
```

### 훅 (Hook) 참조

각각의 훅은 **[이벤트 훅(Event Hooks) 참조](/user-guide/features/hooks#plugin-hooks)** 페이지에 상세히 문서화되어 있습니다 — 콜백 서명(callback signatures), 매개변수 테이블, 각 이벤트가 언제 발동하는지, 그리고 예제 코드들을 확인하세요. 요약하자면 다음과 같습니다:

| 훅 (Hook) | 발동 시점 (Fires when) | 콜백 서명 (Callback signature) | 반환값 |
|------|-----------|-------------------|---------|
| [`pre_tool_call`](/user-guide/features/hooks#pre_tool_call) | 어떤 도구든 실행되기 직전 | `tool_name: str, args: dict, task_id: str` | 무시됨 |
| [`post_tool_call`](/user-guide/features/hooks#post_tool_call) | 어떤 도구든 결과를 반환한 직후 | `tool_name: str, args: dict, result: str, task_id: str, duration_ms: int` | 무시됨 |
| [`pre_llm_call`](/user-guide/features/hooks#pre_llm_call) | 각 턴(turn)마다 한 번, 도구 호출 루프(tool-calling loop) 직전 | `session_id: str, user_message: str, conversation_history: list, is_first_turn: bool, model: str, platform: str` | [컨텍스트 주입](#pre_llm_call-context-injection) |
| [`post_llm_call`](/user-guide/features/hooks#post_llm_call) | 각 턴마다 한 번, 도구 호출 루프 직후 (성공적인 턴에서만) | `session_id: str, user_message: str, assistant_response: str, conversation_history: list, model: str, platform: str` | 무시됨 |
| [`on_session_start`](/user-guide/features/hooks#on_session_start) | 새로운 세션이 생성되었을 때 (첫 턴에만) | `session_id: str, model: str, platform: str` | 무시됨 |
| [`on_session_end`](/user-guide/features/hooks#on_session_end) | 모든 `run_conversation` 호출이 끝날 때 + CLI 종료 시 | `session_id: str, completed: bool, interrupted: bool, model: str, platform: str` | 무시됨 |
| [`on_session_finalize`](/user-guide/features/hooks#on_session_finalize) | CLI/게이트웨이가 활성 세션을 분해(tears down)할 때 | `session_id: str \| None, platform: str` | 무시됨 |
| [`on_session_reset`](/user-guide/features/hooks#on_session_reset) | 게이트웨이가 새 세션 키로 스왑할 때 (`/new`, `/reset`) | `session_id: str, platform: str` | 무시됨 |

대부분의 훅은 관찰만 하고 끝나는(fire-and-forget) 관찰자 역할을 합니다 — 그들의 반환값은 무시됩니다. 예외적으로 `pre_llm_call` 훅은 컨텍스트(context)를 대화에 주입할 수 있습니다.

모든 콜백은 향후 호환성을 위해 `**kwargs`를 받아들여야 합니다. 훅의 콜백에서 예외나 충돌이 발생하면 기록만 남기고 넘어갑니다(skipped). 다른 훅과 에이전트는 정상적으로 계속 실행됩니다.

### `pre_llm_call` 컨텍스트 주입 (context injection)

이 훅은 반환값이 의미를 가지는 유일한 훅입니다. `pre_llm_call` 콜백이 `"context"` 키를 가진 딕셔너리(또는 일반 문자열)를 반환할 때, Hermes는 그 텍스트를 **현재 턴(turn)의 사용자 메시지**에 주입(inject)합니다. 이는 메모리 플러그인, RAG 통합, 가드레일, 그리고 모델에 추가 컨텍스트를 제공해야 하는 모든 플러그인을 위한 메커니즘입니다.

#### 반환 형식 (Return format)

```python
# context 키를 포함하는 Dict 반환
return {"context": "회상된 메모리:\n- 사용자는 다크 모드를 선호함\n- 마지막 프로젝트: hermes-agent"}

# 일반 문자열 반환 (위의 딕셔너리 형태와 동일함)
return "회상된 메모리:\n- 사용자는 다크 모드를 선호함"

# None을 반환하거나 아무것도 반환하지 않음 → 주입 없음 (관찰자 역할만 수행)
return None
```

`"context"` 키가 있는 내용이 비어있지 않은(non-empty) 딕셔너리를 반환하거나(또는 비어있지 않은 일반 문자열을 반환하면), 그 내용들이 수집되어 현재 턴의 사용자 메시지에 추가됩니다.

#### 주입 작동 방식 (How injection works)

주입된 컨텍스트는 시스템 프롬프트가 아니라 **사용자 메시지**의 뒤에 추가됩니다. 이는 의도적인 설계 선택입니다:

- **프롬프트 캐시 보존(Prompt cache preservation)** — 시스템 프롬프트는 턴을 거치며 변경되지 않고 똑같이 유지됩니다. Anthropic과 OpenRouter는 시스템 프롬프트 접두사를 캐시하므로, 시스템 프롬프트를 안정적으로 유지하면 다중 턴(multi-turn) 대화에서 입력 토큰 비용의 75% 이상을 절약할 수 있습니다. 플러그인이 시스템 프롬프트를 수정한다면, 매 턴마다 캐시 미스(cache miss)가 발생할 것입니다.
- **일시적(Ephemeral)** — 주입은 API를 호출하는 시점에만 일어납니다. 대화 내역에 있는 원본 사용자 메시지는 절대 변형되지 않으며, 데이터베이스에도 주입된 내용은 저장되지 않습니다.
- **시스템 프롬프트는 오직 Hermes만의 영역입니다** — 그곳에는 모델별 지침, 도구 강제(enforcement) 규칙, 페르소나 지시 사항, 캐시된 스킬 내용들이 포함되어 있습니다. 플러그인은 에이전트의 핵심 지침을 변경하는 방식이 아니라, 사용자의 입력 곁에 컨텍스트를 덧붙이는(contribute) 방식으로 동작해야 합니다.

#### 예제: 메모리 회상 플러그인

```python
"""Memory plugin — 벡터 스토어에서 관련 컨텍스트를 회상합니다."""

import httpx

MEMORY_API = "https://your-memory-api.example.com"

def recall_context(session_id, user_message, is_first_turn, **kwargs):
    """매 LLM 턴 전에 호출됩니다. 회상된 메모리를 반환합니다."""
    try:
        resp = httpx.post(f"{MEMORY_API}/recall", json={
            "session_id": session_id,
            "query": user_message,
        }, timeout=3)
        memories = resp.json().get("results", [])
        if not memories:
            return None  # 주입할 내용 없음

        text = "이전 세션에서 회상된 컨텍스트:\n"
        text += "\n".join(f"- {m['text']}" for m in memories)
        return {"context": text}
    except Exception:
        return None  # 실패해도 에이전트를 망가뜨리지 않고 조용히 넘깁니다.

def register(ctx):
    ctx.register_hook("pre_llm_call", recall_context)
```

#### 예제: 가드레일 플러그인

```python
"""Guardrails plugin — 콘텐츠 정책을 강제합니다."""

POLICY = """이번 세션에서 다음 콘텐츠 정책을 반드시 따라야 합니다:
- 작업 디렉터리 외부에 접근하는 코드를 절대 생성하지 마십시오
- 파괴적인 작업을 실행하기 전에 항상 경고를 표시하십시오
- 개인 데이터를 추출하는 요청은 거부하십시오"""

def inject_guardrails(**kwargs):
    """모든 턴에 정책 텍스트를 주입합니다."""
    return {"context": POLICY}

def register(ctx):
    ctx.register_hook("pre_llm_call", inject_guardrails)
```

#### 예제: 관찰자 전용 훅 (주입 없음)

```python
"""Analytics plugin — 컨텍스트 주입 없이 턴 메타데이터를 추적합니다."""

import logging
logger = logging.getLogger(__name__)

def log_turn(session_id, user_message, model, is_first_turn, **kwargs):
    """각 LLM 호출 전에 발동합니다. None 반환 — 주입되는 컨텍스트 없음."""
    logger.info("Turn: session=%s model=%s first=%s msg_len=%d",
                session_id, model, is_first_turn, len(user_message or ""))
    # 아무것도 반환하지 않음 → 주입 안 됨

def register(ctx):
    ctx.register_hook("pre_llm_call", log_turn)
```

#### 여러 플러그인이 컨텍스트를 반환할 때

여러 플러그인이 `pre_llm_call`에서 컨텍스트를 반환하면 그 반환값들은 두 개의 줄바꿈 문자(`\n\n`)로 연결되어 한꺼번에 사용자 메시지의 끝에 추가됩니다. 합쳐지는 순서는 플러그인이 발견된 순서(플러그인 디렉터리 이름의 알파벳 순서)를 따릅니다.

### CLI 명령어 등록

플러그인은 `hermes <plugin>` 이라는 고유의 하위 명령어 트리(subcommand tree)를 추가할 수 있습니다:

```python
def _my_command(args):
    """hermes my-plugin <subcommand>에 대한 핸들러."""
    sub = getattr(args, "my_command", None)
    if sub == "status":
        print("All good!")
    elif sub == "config":
        print("Current config: ...")
    else:
        print("Usage: hermes my-plugin <status|config>")

def _setup_argparse(subparser):
    """hermes my-plugin을 위한 argparse 트리를 만듭니다."""
    subs = subparser.add_subparsers(dest="my_command")
    subs.add_parser("status", help="Show plugin status")
    subs.add_parser("config", help="Show plugin config")
    subparser.set_defaults(func=_my_command)

def register(ctx):
    ctx.register_tool(...)
    ctx.register_cli_command(
        name="my-plugin",
        help="Manage my plugin",
        setup_fn=_setup_argparse,
        handler_fn=_my_command,
    )
```

등록한 뒤에 사용자는 `hermes my-plugin status`, `hermes my-plugin config` 등을 실행할 수 있습니다.

**메모리 제공자(Memory provider) 플러그인**은 대신에 관례 기반의 접근 방식을 사용합니다: 플러그인의 `cli.py` 파일에 `register_cli(subparser)` 함수를 추가하면 됩니다. 메모리 플러그인 검색 시스템이 이를 자동으로 찾으므로 `ctx.register_cli_command()`를 명시적으로 호출할 필요가 없습니다. 자세한 사항은 [메모리 제공자 플러그인 가이드](/developer-guide/memory-provider-plugin#adding-cli-commands)를 확인하세요.

**활성화된 제공자만 표시(Active-provider gating):** 메모리 플러그인의 CLI 명령어는 설정에서 해당 제공자가 현재 활성화된 `memory.provider` 일 때만 나타납니다. 사용자가 제공자를 설정하지 않은 경우, 여러분의 플러그인의 CLI 명령어들이 도움말(help) 출력을 복잡하게 만들지 않습니다.

### 슬래시 명령어 등록

플러그인은 세션 내의 슬래시 명령어 — 사용자가 대화 중에 타이핑하는 명령어 (예: `/lcm status` 나 `/ping`)를 등록할 수 있습니다. 이는 CLI와 게이트웨이(Telegram, Discord 등) 모두에서 작동합니다.

```python
def _handle_status(raw_args: str) -> str:
    """/mystatus에 대한 핸들러 — 명령어 이름 뒤에 오는 모든 문자열이 전달됩니다."""
    if raw_args.strip() == "help":
        return "Usage: /mystatus [help|check]"
    return "Plugin status: all systems nominal"

def register(ctx):
    ctx.register_command(
        "mystatus",
        handler=_handle_status,
        description="Show plugin status",
    )
```

등록 후에는 사용자가 어느 세션에서나 `/mystatus`를 타이핑할 수 있습니다. 이 명령어는 자동 완성, `/help` 출력, 그리고 Telegram 봇 메뉴에도 표시됩니다.

**서명 (Signature):** `ctx.register_command(name: str, handler: Callable, description: str = "")`

| 매개변수 | 자료형 (Type) | 설명 |
|-----------|------|-------------|
| `name` | `str` | 선행 슬래시(/)를 제외한 명령어 이름 (예: `"lcm"`, `"mystatus"`) |
| `handler` | `Callable[[str], str \| None]` | 명령어 뒤에 오는 인자(원시 문자열)와 함께 호출됨. `async` 가능. |
| `description` | `str` | `/help`, 자동 완성, Telegram 봇 메뉴에 표시됨 |

**`register_cli_command()`와의 주요 차이점:**

| | `register_command()` | `register_cli_command()` |
|---|---|---|
| 호출 방식 | 세션 내에서 `/name` 입력 | 터미널에서 `hermes name` 입력 |
| 작동하는 곳 | CLI 세션, Telegram, Discord 등 | 터미널 전용 |
| 핸들러가 받는 인자 | 원시 문자열 (Raw args string) | argparse `Namespace` |
| 주 용도 | 진단, 상태 확인, 빠른 액션 수행 | 복잡한 하위 명령어 트리, 설정 마법사 |

**충돌 방지 (Conflict protection):** 플러그인이 시스템 내장 명령어(`help`, `model`, `new` 등)와 충돌하는 이름을 등록하려 하면, 등록이 조용히 거부되고 로그에 경고가 남습니다. 내장 명령어가 언제나 우선 순위를 갖습니다.

**비동기 핸들러 (Async handlers):** 게이트웨이 디스패치는 비동기 핸들러를 자동으로 감지하여 await(대기)하므로, 동기(sync)나 비동기(async) 함수 모두 사용할 수 있습니다:

```python
async def _handle_check(raw_args: str) -> str:
    result = await some_async_operation()
    return f"Check result: {result}"

def register(ctx):
    ctx.register_command("check", handler=_handle_check, description="Run async check")
```

### 슬래시 명령어에서 도구 디스패치(Dispatch)하기

도구를 오케스트레이션(orchestrate)해야 하는 슬래시 명령어 핸들러의 경우 (`delegate_task`를 통해 하위 에이전트를 생성하거나, `file_edit`을 호출하는 등), 프레임워크 내부 요소를 직접 건드리지 말고 `ctx.dispatch_tool()`을 사용하세요. 상위 에이전트의 컨텍스트(작업 공간 정보, 스피너, 모델 상속 등)가 자동으로 연결됩니다.

```python
def register(ctx):
    def _handle_deliver(raw_args: str):
        result = ctx.dispatch_tool(
            "delegate_task",
            {
                "goal": raw_args,
                "toolsets": ["terminal", "file", "web"],
            },
        )
        return result

    ctx.register_command(
        "deliver",
        handler=_handle_deliver,
        description="Delegate a goal to a subagent",
    )
```

**서명 (Signature):** `ctx.dispatch_tool(name: str, args: dict, *, parent_agent=None) -> str`

| 매개변수 | 자료형 (Type) | 설명 |
|-----------|------|-------------|
| `name` | `str` | 도구 레지스트리에 등록된 도구 이름 (예: `"delegate_task"`, `"file_edit"`) |
| `args` | `dict` | 모델이 보낼 때와 같은 형태의 도구 인자(arguments) 딕셔너리 |
| `parent_agent` | `Agent \| None` | 선택적 재정의 속성. 생략 시, 현재 CLI 에이전트에서 리졸브되거나 (게이트웨이 모드에서는 부드럽게 성능이 저하되는 방향으로) 처리됩니다 |

**런타임 동작 (Runtime behavior):**

- **CLI 모드:** 활성화된 CLI 에이전트로부터 `parent_agent`가 해석되므로, 작업 공간 힌트, 스피너, 모델 선택 사항이 예상대로 상속됩니다.
- **게이트웨이 모드:** CLI 에이전트가 없기 때문에 도구는 정상적으로 기능이 조금 저하되는 형태(degrade gracefully)를 취합니다 — 설정된 터미널 작업 디렉터리에서 워크스페이스를 읽으며 스피너는 표시되지 않습니다.
- **명시적 재정의:** 호출자가 명시적으로 `parent_agent=`를 전달하면, 그 값이 존중되며 덮어씌워지지 않습니다.

이것이 플러그인 명령어에서 도구를 호출하기 위한 공용의 안정적인 인터페이스입니다. 플러그인은 `ctx._cli_ref.agent`나 이와 유사한 비공개 상태(private state)에 접근해서는 안 됩니다.

:::tip
이 가이드는 **일반 플러그인** (도구, 훅, 슬래시 명령어, CLI 명령어)을 다룹니다. 아래 섹션들은 전문화된 각각의 플러그인 유형을 작성하는 패턴을 개략적으로 설명하며; 각각의 플러그인 필드에 대한 참조와 예제가 있는 전체 가이드 문서로 링크를 제공합니다.
:::

## 전문화된 플러그인 유형 (Specialized plugin types)

Hermes에는 일반 플러그인 영역(general surface) 외에도 5가지의 특수한 플러그인 유형이 있습니다. 이들은 각각 `plugins/<category>/<name>/` (번들용) 또는 `~/.hermes/plugins/<category>/<name>/` (사용자용) 디렉터리에 배포됩니다. 카테고리별로 계약(contract) 조건이 다르므로 — 필요한 것을 골라 전체 가이드를 읽어보세요.

### 모델 제공자 (Model provider) 플러그인 — LLM 백엔드 추가

`plugins/model-providers/<name>/` 안에 프로필(profile)을 놓으세요:

```python
# plugins/model-providers/acme/__init__.py
from providers import register_provider
from providers.base import ProviderProfile

register_provider(ProviderProfile(
    name="acme",
    aliases=("acme-inference",),
    display_name="Acme Inference",
    env_vars=("ACME_API_KEY", "ACME_BASE_URL"),
    base_url="https://api.acme.example.com/v1",
    auth_type="api_key",
    default_aux_model="acme-small-fast",
    fallback_models=("acme-large-v3", "acme-medium-v3"),
))
```

```yaml
# plugins/model-providers/acme/plugin.yaml
name: acme-provider
kind: model-provider
version: 1.0.0
description: Acme Inference — OpenAI-compatible direct API
```

`get_provider_profile()` 또는 `list_providers()`를 호출하는 함수들 — `auth.py`, `config.py`, `doctor.py`, `models.py`, `runtime_provider.py` 및 chat_completions 트랜스포트 — 에 의해 처음으로 호출될 때 게으르게 감지(Lazy-discovered)되어 자동으로 연결됩니다. 사용자의 플러그인은 같은 이름의 번들 플러그인을 덮어씁니다.

**전체 가이드:** [모델 제공자 플러그인](/developer-guide/model-provider-plugin) — 필드 참조 문서, 오버라이드 가능한 훅들 (`prepare_messages`, `build_extra_body`, `build_api_kwargs_extras`, `fetch_models`), api_mode 선택, 인증 유형(auth types), 테스트 방법 포함.

### 플랫폼 플러그인 — 게이트웨이 채널 추가

어댑터를 `plugins/platforms/<name>/` 안에 넣으세요:

```python
# plugins/platforms/myplatform/adapter.py
from gateway.platforms.base import BasePlatformAdapter

class MyPlatformAdapter(BasePlatformAdapter):
    async def connect(self): ...
    async def send(self, chat_id, text): ...
    async def disconnect(self): ...

def check_requirements():
    import os
    return bool(os.environ.get("MYPLATFORM_TOKEN"))

def _env_enablement():
    import os
    tok = os.getenv("MYPLATFORM_TOKEN", "").strip()
    if not tok:
        return None
    return {"token": tok}

def register(ctx):
    ctx.register_platform(
        name="myplatform",
        label="MyPlatform",
        adapter_factory=lambda cfg: MyPlatformAdapter(cfg),
        check_fn=check_requirements,
        required_env=["MYPLATFORM_TOKEN"],
        # SDK를 인스턴스화하지 않고도, 환경 변수 전용 설정만으로
        # `hermes gateway status`에 표시되도록 PlatformConfig.extra를 환경 변수로부터 자동 채웁니다.
        env_enablement_fn=_env_enablement,
        # 크론 전달 옵트인(Opt in): `deliver=myplatform` 일 때 이 변수로 전달을 라우팅합니다.
        cron_deliver_env_var="MYPLATFORM_HOME_CHANNEL",
        emoji="💬",
        platform_hint="You are chatting via MyPlatform. Keep responses concise.",
    )
```

```yaml
# plugins/platforms/myplatform/plugin.yaml
name: myplatform-platform
label: MyPlatform
kind: platform
version: 1.0.0
description: MyPlatform gateway adapter
requires_env:
  - name: MYPLATFORM_TOKEN
    description: "Bot token from the MyPlatform console"
    password: true
optional_env:
  - name: MYPLATFORM_HOME_CHANNEL
    description: "Default channel for cron delivery"
    password: false
```

**전체 가이드:** [플랫폼 어댑터 추가하기](/developer-guide/adding-platform-adapters) — 완성된 `BasePlatformAdapter` 계약 규칙, 메시지 라우팅, 인증 게이트, 설정 마법사 연동. Python 표준 라이브러리(stdlib-only)만 사용한 실제 동작 예제는 `plugins/platforms/irc/`를 참조하세요.

### 메모리 제공자 (Memory provider) 플러그인 — 교차 세션 (cross-session) 지식 백엔드 추가

`MemoryProvider` 구현체를 `plugins/memory/<name>/`에 넣으세요:

```python
# plugins/memory/my-memory/__init__.py
from agent.memory_provider import MemoryProvider

class MyMemoryProvider(MemoryProvider):
    @property
    def name(self) -> str:
        return "my-memory"

    def is_available(self) -> bool:
        import os
        return bool(os.environ.get("MY_MEMORY_API_KEY"))

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id

    def sync_turn(self, user_message, assistant_response, **kwargs) -> None:
        ...

    def prefetch(self, query: str, **kwargs) -> str | None:
        ...

def register(ctx):
    ctx.register_memory_provider(MyMemoryProvider())
```

메모리 제공자는 단일 선택 방식(single-select)입니다 — `config.yaml`의 `memory.provider`를 통해 한 번에 하나만 활성화됩니다.

**전체 가이드:** [메모리 제공자 플러그인](/developer-guide/memory-provider-plugin) — 전체 `MemoryProvider` ABC (추상 기본 클래스), 쓰레드 규칙(threading contract), 프로필 분리 정책(profile isolation), `cli.py`를 통한 CLI 명령어 등록.

### 컨텍스트 엔진 (Context engine) 플러그인 — 컨텍스트 압축기(compressor) 교체

```python
# plugins/context_engine/my-engine/__init__.py
from agent.context_engine import ContextEngine

class MyContextEngine(ContextEngine):
    @property
    def name(self) -> str:
        return "my-engine"

    def should_compress(self, messages, model) -> bool: ...
    def compress(self, messages, model) -> list[dict]: ...

def register(ctx):
    ctx.register_context_engine(MyContextEngine())
```

컨텍스트 엔진은 단일 선택 방식(single-select)입니다 — `config.yaml`의 `context.engine`을 통해 선택됩니다.

**전체 가이드:** [컨텍스트 엔진 플러그인](/developer-guide/context-engine-plugin).

### 이미지 생성 (Image-generation) 백엔드

제공자를 `plugins/image_gen/<name>/` 안에 넣으세요:

```python
# plugins/image_gen/my-imggen/__init__.py
from agent.image_gen_provider import ImageGenProvider

class MyImageGenProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "my-imggen"

    def is_available(self) -> bool: ...
    def generate(self, prompt: str, **kwargs) -> str: ...   # image 경로를 반환합니다

def register(ctx):
    ctx.register_image_gen_provider(MyImageGenProvider())
```

```yaml
# plugins/image_gen/my-imggen/plugin.yaml
name: my-imggen
kind: backend
version: 1.0.0
description: Custom image generation backend
```

**전체 가이드:** [이미지 생성 제공자 플러그인](/developer-guide/image-gen-provider-plugin) — 전체 `ImageGenProvider` ABC, `list_models()` / `get_setup_schema()` 메타데이터, `success_response()`/`error_response()` 도우미(helpers), base64 방식 대 URL 출력, 사용자 재정의, pip 배포 방식 지원.

**참조 예제:** `plugins/image_gen/openai/` (OpenAI SDK를 통한 DALL-E / GPT-Image), `plugins/image_gen/openai-codex/`, `plugins/image_gen/xai/` (Grok 이미지 생성).

## 비 파이썬(Non-Python) 확장 표면 (extension surfaces)

Hermes는 Python 플러그인이 전혀 아닌 확장도 허용합니다. 이러한 기능은 [플러그 가능한 인터페이스 표(Pluggable interfaces table)](/user-guide/features/plugins#pluggable-interfaces--where-to-go-for-each)에 표시되어 있으며, 아래 각 섹션에서는 각 작성 스타일에 대해 간략하게 설명합니다.

### MCP 서버 — 외부 도구 등록

Model Context Protocol (MCP) 서버는 Python 플러그인 없이 고유한 도구를 Hermes에 등록합니다. `~/.hermes/config.yaml` 파일에 선언하세요:

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
    timeout: 120

  linear:
    url: "https://mcp.linear.app/sse"
    auth:
      type: "oauth"
```

Hermes는 시작 시 각 서버에 연결하여 해당 도구를 나열하고 내장 도구들과 함께 등록합니다. LLM은 이들을 다른 도구들과 정확히 똑같이 인식합니다. **전체 가이드:** [MCP](/user-guide/features/mcp).

### 게이트웨이 이벤트 훅 (Gateway event hooks) — 수명 주기 이벤트(lifecycle events)에서 발동

매니페스트와 핸들러를 `~/.hermes/hooks/<name>/` 에 넣으세요:

```yaml
# ~/.hermes/hooks/long-task-alert/HOOK.yaml
name: long-task-alert
description: Send a push notification when a long task finishes
events:
  - agent:end
```

```python
# ~/.hermes/hooks/long-task-alert/handler.py
async def handle(event_type: str, context: dict) -> None:
    if context.get("duration_seconds", 0) > 120:
        # 푸시 알림 전송 …
        pass
```

이벤트에는 `gateway:startup`, `session:start`, `session:end`, `session:reset`, `agent:start`, `agent:step`, `agent:end`, 그리고 와일드카드 `command:*`가 포함됩니다. 훅에서 발생한 오류는 잡혀서(caught) 로깅만 되며 메인 파이프라인을 절대 멈추거나 방해하지 않습니다.

**전체 가이드:** [게이트웨이 이벤트 훅](/user-guide/features/hooks#gateway-event-hooks).

### 쉘 훅 (Shell hooks) — 도구 호출 시 쉘 명령어 실행

Python 코드를 작성하지 않고도 단순히 도구가 호출될 때 스크립트를 실행하고 싶다면(알림, 감사 로그, 데스크톱 경고 알림, 자동 포맷팅 기능 등), `config.yaml` 파일의 쉘 훅을 사용하세요:

```yaml
hooks:
  - event: post_tool_call
    command: "notify-send 'Tool ran: {tool_name}'"
    when:
      tools: [terminal, patch, write_file]
```

Python 플러그인 훅과 동일한 모든 이벤트를 지원하며(`pre_tool_call`, `post_tool_call`, `pre_llm_call`, `post_llm_call`, `on_session_start`, `on_session_end`, `pre_gateway_dispatch`), 구조화된 JSON 출력을 통해 `pre_tool_call` 단계에서 도구 실행을 막는 결정(blocking decisions)을 지원합니다.

**전체 가이드:** [쉘 훅](/user-guide/features/hooks#shell-hooks).

### 스킬 소스 (Skill sources) — 사용자 정의 스킬 레지스트리 추가

GitHub 저장소를 유지 관리하거나 (내장 소스 외의) 커뮤니티 인덱스에서 스킬을 가져오려면, 이를 **tap**으로 추가하세요:

```bash
hermes skills tap add myorg/skills-repo
hermes skills search my-workflow --source myorg/skills-repo
hermes skills install myorg/skills-repo/my-workflow
```

자신만의 tap을 배포(Publishing)하는 것은 서버 관리나 레지스트리 가입이 필요 없이 `skills/<skill-name>/SKILL.md` 디렉터리가 있는 GitHub 저장소를 만드는 것만으로 가능합니다.

**전체 가이드:** [스킬 허브(Skills Hub)](/user-guide/features/skills#skills-hub) · [사용자 지정 tap 배포하기](/user-guide/features/skills#publishing-a-custom-skill-tap) (저장소 레이아웃 구조, 최소 예제 코드, 기본 설정이 아닌 경로들, 신뢰 수준 안내).

### 명령어 템플릿을 통한 TTS / STT

Python 코드 작성 없이도 오디오나 텍스트를 읽고 쓰는 모든 CLI를 `config.yaml`을 통해 연결(plug-in)할 수 있습니다:

```yaml
tts:
  provider: voxcpm
  providers:
    voxcpm:
      type: command
      command: "voxcpm --ref ~/voice.wav --text-file {input_path} --out {output_path}"
      output_format: mp3
      voice_compatible: true
```

STT의 경우, `HERMES_LOCAL_STT_COMMAND`가 쉘 템플릿을 가리키도록 설정하세요. 지원되는 플레이스홀더(자리 표시자): (TTS용) `{input_path}`, `{output_path}`, `{format}`, `{voice}`, `{model}`, `{speed}` ; (STT용) `{input_path}`, `{output_dir}`, `{language}`, `{model}`. 경로와 상호작용하는 모든 CLI 프로그램은 자동으로 플러그인이 됩니다.

**전체 가이드:** [TTS 사용자 정의 명령어 제공자](/user-guide/features/tts#custom-command-providers) · [STT](/user-guide/features/tts#voice-message-transcription-stt).

## pip를 통한 배포

플러그인을 대중에 공유하려면, Python 패키지에 진입점(entry point)을 추가하세요:

```toml
# pyproject.toml
[project.entry-points."hermes_agent.plugins"]
my-plugin = "my_plugin_package"
```

```bash
pip install hermes-plugin-calculator
# 다음 Hermes 시작 시 플러그인이 자동으로 검색(auto-discovered)됨
```

## NixOS를 위한 배포

진입점(entry points)이 명시된 `pyproject.toml`을 제공하면 NixOS 사용자들은 플러그인을 선언적으로 설치할 수 있습니다:

**진입점(Entry-point) 플러그인** (배포 시 권장 방식):
```nix
# User's configuration.nix
services.hermes-agent.extraPythonPackages = [
  (pkgs.python312Packages.buildPythonPackage {
    pname = "my-plugin";
    version = "1.0.0";
    src = pkgs.fetchFromGitHub {
      owner = "you";
      repo = "hermes-my-plugin";
      rev = "v1.0.0";
      hash = "sha256-...";  # nix-prefetch-url --unpack
    };
    format = "pyproject";
    build-system = [ pkgs.python312Packages.setuptools ];
  })
];
```

**디렉터리 플러그인** (`pyproject.toml` 불필요):
```nix
services.hermes-agent.extraPlugins = [
  (pkgs.fetchFromGitHub {
    owner = "you";
    repo = "hermes-my-plugin";
    rev = "v1.0.0";
    hash = "sha256-...";
  })
];
```

오버레이(overlay) 사용 및 충돌 검사를 포함한 전체 문서 내용은 [Nix 설정 가이드](/getting-started/nix-setup#plugins)를 참조하세요.

## 자주 하는 실수들 (Common mistakes)

**핸들러가 JSON 문자열을 반환하지 않음:**
```python
# 잘못된 방식 — 사전(dict)을 반환함
def handler(args, **kwargs):
    return {"result": 42}

# 올바른 방식 — JSON 문자열을 반환함
def handler(args, **kwargs):
    return json.dumps({"result": 42})
```

**핸들러의 서명(signature)에 `**kwargs`가 누락됨:**
```python
# 잘못된 방식 — Hermes가 추가적인 컨텍스트를 전달할 경우 코드가 고장(break)납니다
def handler(args):
    ...

# 올바른 방식
def handler(args, **kwargs):
    ...
```

**핸들러가 예외를 발생(raise exceptions)시킴:**
```python
# 잘못된 방식 — 예외가 외부로 전파되어 도구 호출이 실패합니다
def handler(args, **kwargs):
    result = 1 / int(args["value"])  # ZeroDivisionError 발생 가능!
    return json.dumps({"result": result})

# 올바른 방식 — 잡아서 에러 JSON을 반환합니다
def handler(args, **kwargs):
    try:
        result = 1 / int(args.get("value", 0))
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})
```

**스키마의 설명(description)이 너무 모호함:**
```python
# 잘못된 방식 — 모델이 이것을 언제 사용해야 할지 알 수 없습니다
"description": "Does stuff"

# 올바른 방식 — 모델이 언제, 어떻게 사용해야 하는지 정확히 압니다
"description": "수학 표현식을 평가합니다. 산술 연산, 삼각 함수, 로그 계산에 사용하세요. 지원: +, -, *, /, **, sqrt, sin, cos, log, pi, e."
```

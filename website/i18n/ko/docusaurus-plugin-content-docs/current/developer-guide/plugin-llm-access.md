---
sidebar_position: 11
title: "플러그인 LLM 액세스"
description: "ctx.llm을 통해 플러그인 내부에서 모든 LLM 호출을 실행하세요 — 채팅 또는 구조화된 출력, 동기 또는 비동기. 호스트가 관리하는 인증, 기본 거부 신뢰 게이트, 선택적 JSON Schema 검증을 제공합니다."
---

# 플러그인 LLM 액세스

`ctx.llm`은 플러그인이 LLM 호출을 실행할 때 사용하는 공식적인 방법입니다.
채팅 완성, 구조화된 추출, 동기, 비동기, 이미지 포함 여부와 관계없이
동일한 인터페이스, 동일한 신뢰 게이트, 동일하게 호스트가 관리하는 자격 증명을 사용합니다.

플러그인이 에이전트의 대화에 포함되지 않는 모델 관련 작업을 해야 할 때
이 기능을 사용합니다. 도구 오류를 비엔지니어가 읽을 수 있는 내용으로
다시 작성하는 훅, 수신 메시지를 대기열에 넣기 전에 번역하는
게이트웨이 어댑터, 긴 붙여넣기 내용을 요약하는 슬래시 명령,
어제 활동을 평가하고 상태 보드에 한 줄을 기록하는 예약 작업,
메시지를 깨울 가치가 있는지 자체를 판단하는 사전 필터 등이 그 예입니다.

이런 작업에는 에이전트가 개입할 필요가 없습니다. 한 번의
LLM 호출, 타입이 지정된 답변을 받고 끝내면 됩니다.

## 가장 간단한 호출

```python
result = ctx.llm.complete(messages=[{"role": "user", "content": "ping"}])
return result.text
```

이 한 줄이 API 전체를 보여줍니다. 키도, 프로바이더 설정도,
SDK 초기화도 필요하지 않습니다. 플러그인은 사용자가 현재 사용하는
프로바이더와 모델을 그대로 사용하며, 사용자가 프로바이더를 전환하면
플러그인도 자동으로 따라갑니다.

## 더 완전한 채팅 예제

```python
result = ctx.llm.complete(
    messages=[
        {"role": "system", "content": "Rewrite errors as one short sentence a non-engineer can act on."},
        {"role": "user",   "content": traceback_text},
    ],
    max_tokens=64,
    purpose="hooks.error-rewrite",
)
return result.text
```

`purpose`는 자유 형식의 감사 문자열입니다. `agent.log`와
`result.audit`에 표시되므로 운영자는 어떤 플러그인이 어떤
호출을 실행했는지 확인할 수 있습니다. 선택 사항이지만 자주 실행되는
호출에는 지정하는 것을 권장합니다.

## 구조화된 출력

플러그인이 타입이 지정된 답변을 필요로 한다면 구조화된 경로로 전환합니다.

```python
result = ctx.llm.complete_structured(
    instructions="Score this support reply for urgency (0–1) and pick a category.",
    input=[{"type": "text", "text": message_body}],
    json_schema=TRIAGE_SCHEMA,
    purpose="support.triage",
    temperature=0.0,
    max_tokens=128,
)

if result.parsed["urgency"] > 0.8:
    await dispatch_to_oncall(result.parsed["category"], message_body)
```

호스트는 프로바이더에 JSON 출력을 요청하고, 대체 경로로 로컬에서
파싱하며, `jsonschema`가 설치되어 있으면 스키마에 맞게 검증한 뒤
`result.parsed`에 Python 객체를 반환합니다. 모델이 유효한 JSON을
생성하지 못하면 `result.parsed`는 `None`이 되고,
`result.text`에 원시 응답이 담깁니다.

## 이 경로가 제공하는 것

* **한 번의 호출, 네 가지 형태.** 채팅에는 `complete()`,
  타입이 지정된 JSON에는 `complete_structured()`, asyncio에는
  `acomplete()` 및 `acomplete_structured()`를 사용합니다. 인자와
  결과 객체는 동일합니다.
* **호스트가 관리하는 자격 증명.** OAuth 토큰, 갱신 흐름,
  자격 증명 풀, 작업별 보조 재정의 등 Hermes가 이미 지원하는 모든
  자격 증명 개념이 적용됩니다. 플러그인은 토큰을 볼 수 없으며,
  호스트가 `result.audit`를 통해 호출을 다시 추적합니다.
* **범위가 제한됩니다.** 단일 동기 또는 비동기 호출입니다. 스트리밍도,
  도구 루프도, 관리해야 할 대화 상태도 없습니다. 입력을 지정하고,
  결과를 받아 반환하면 됩니다.
* **기본 거부 신뢰.** 한 번도 설정하지 않은 플러그인은 자체적으로
  프로바이더, 모델, 에이전트 또는 저장된 자격 증명을 선택할 수 없습니다.
  기본 동작은 "사용자가 사용하는 것을 사용"하는 것입니다. 운영자는
  `config.yaml`에서 플러그인별로 특정 재정의를 허용합니다.

## 빠른 시작

아래에는 완전한 플러그인 두 가지가 있습니다. 하나는 채팅용이고,
다른 하나는 구조화된 출력용입니다. 둘 다 하나의 `register(ctx)`
함수 안에 포함되어 있으며 사용자가 활성화한 모델을 대상으로 실행하는 데
외부 설정이 전혀 필요하지 않습니다.

### 채팅 완성 — `/tldr`

```python
def register(ctx):
    ctx.register_command(
        name="tldr",
        handler=lambda raw: _tldr(ctx, raw),
        description="Summarise the supplied text in one paragraph.",
        args_hint="<text>",
    )


def _tldr(ctx, raw_args: str) -> str:
    text = raw_args.strip()
    if not text:
        return "Usage: /tldr <text to summarise>"
    result = ctx.llm.complete(
        messages=[
            {"role": "system",
             "content": "Summarise the user's text in one tight paragraph. No preamble."},
            {"role": "user", "content": text},
        ],
        max_tokens=256,
        temperature=0.3,
        purpose="tldr",
    )
    return result.text
```

`result.text`에는 모델의 응답이, `result.usage`에는 토큰 수가,
`result.provider`와 `result.model`에는 호출의 출처 정보가 담깁니다.

### 구조화된 추출 — `/paste-to-tasks`

```python
def register(ctx):
    ctx.register_command(
        name="paste-to-tasks",
        handler=lambda raw: _paste_to_tasks(ctx, raw),
        description="Turn freeform meeting notes into structured tasks.",
        args_hint="<text>",
    )


_TASKS_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "owner":  {"type": "string"},
                    "action": {"type": "string"},
                    "due":    {"type": "string", "description": "ISO date or empty"},
                },
                "required": ["action"],
            },
        },
    },
    "required": ["tasks"],
}


def _paste_to_tasks(ctx, raw_args: str) -> str:
    if not raw_args.strip():
        return "Usage: /paste-to-tasks <meeting notes>"
    result = ctx.llm.complete_structured(
        instructions=(
            "Extract concrete action items from these meeting notes. "
            "One task per actionable line. If no owner is named, leave 'owner' blank."
        ),
        input=[{"type": "text", "text": raw_args}],
        json_schema=_TASKS_SCHEMA,
        schema_name="meeting.tasks",
        purpose="paste-to-tasks",
        temperature=0.0,
        max_tokens=512,
    )
    if result.parsed is None:
        return f"Couldn't parse a response. Raw output:\n{result.text}"
    lines = [f"- [{t.get('owner') or '?'}] {t['action']}" for t in result.parsed["tasks"]]
    return "\n".join(lines) or "(no tasks found)"
```

이번에는 이미지 입력을 사용하는 세 번째 실습 예제가
[`hermes-example-plugins`](https://github.com/NousResearch/hermes-example-plugins/tree/main/plugin-llm-example)
저장소에 있습니다(참조용 플러그인용 동반 저장소이며
hermes-agent에 포함되어 있지 않습니다). 비동기 인터페이스
(`acomplete()` / `acomplete_structured()` 및 `asyncio.gather()`)는
같은 저장소의
[`plugin-llm-async-example`](https://github.com/NousResearch/hermes-example-plugins/tree/main/plugin-llm-async-example)
를 참조하세요.

## 어떤 것을 사용해야 할까요?

| 원하는 것… | 사용 방법 |
|---|---|
| 자유 형식의 텍스트 응답(번역, 요약, 다시 쓰기, 생성) | `complete()` |
| 다중 턴 프롬프트(시스템 + few-shot 예제 + 사용자) | `complete()` |
| 스키마에 맞게 검증된 타입 지정 dict 반환 | `complete_structured()` |
| 타입 지정 dict를 반환하는 이미지 또는 텍스트 입력 | `complete_structured()` |
| 비동기 코드(게이트웨이 어댑터, 비동기 훅)에서 동일한 호출 | `acomplete()` / `acomplete_structured()` |

그 외의 모든 것 — 프로바이더 선택, 모델 확인, 인증, 대체 경로,
타임아웃, 비전 라우팅 — 은 네 가지 모두에서 동일합니다.

## API 표면

`ctx.llm`은 `agent.plugin_llm.PluginLlm`의 인스턴스입니다.

### `complete()`

```python
result = ctx.llm.complete(
    messages=[{"role": "user", "content": "Hi"}],
    provider=None,         # optional, gated — Hermes provider id (e.g. "openrouter")
    model=None,            # optional, gated — whatever string that provider expects
    temperature=None,
    max_tokens=None,
    timeout=None,          # seconds
    agent_id=None,         # optional, gated
    profile=None,          # optional, gated — explicit auth-profile name
    purpose="optional-audit-string",
    task=None,             # optional — a plugin-registered auxiliary slot
)
# → PluginLlmCompleteResult(text, provider, model, agent_id, usage, audit)
```

일반적인 채팅 완성입니다. `messages`는 표준 OpenAI 형식인
`{"role": "...", "content": "..."}` dict의 목록입니다. 다중 턴
프롬프트(시스템 + few-shot 사용자/어시스턴트 쌍 + 최종 사용자)는
OpenAI SDK에서와 정확히 동일하게 작동합니다.

`provider=`와 `model=`은 독립적이며 호스트의 주 설정
(`model.provider` + `model.model`)과 동일한 형태를 따릅니다.
`model=`만 지정하면 사용자의 활성 프로바이더에서 다른 모델을 사용합니다.
둘 다 지정하면 프로바이더를 완전히 전환합니다. 운영자의 허용 없이
둘 중 하나를 사용하면 `PluginLlmTrustError`가 발생합니다.

### `complete_structured()`

```python
result = ctx.llm.complete_structured(
    instructions="What you want extracted.",
    input=[
        {"type": "text",  "text": "..."},
        {"type": "image", "data": b"...", "mime_type": "image/png"},
        {"type": "image", "url":  "https://..."},
    ],
    json_schema={...},     # optional — triggers parsed result + validation
    json_mode=False,       # set True without a schema to ask for JSON anyway
    schema_name=None,      # optional human-readable schema name
    system_prompt=None,
    provider=None,         # optional, gated
    model=None,            # optional, gated
    temperature=None,
    max_tokens=None,
    timeout=None,
    agent_id=None,
    profile=None,
    purpose=None,
    task=None,             # optional — a plugin-registered auxiliary slot
)
# → PluginLlmStructuredResult(text, provider, model, agent_id,
#                             usage, parsed, content_type, audit)
```

입력은 타입이 지정된 텍스트 또는 이미지 블록입니다(원시 바이트는
자동으로 base64 인코딩되어 `data:` URL이 됩니다). `json_schema` 또는
`json_mode=True`가 제공되면 호스트는 `response_format`을 통해
JSON 출력을 요청하고, 대체 경로로 로컬에서 파싱하며,
`jsonschema`가 설치되어 있으면 스키마에 맞게 검증합니다.

* `result.content_type == "json"` — `result.parsed`는 스키마와
  일치하는 Python 객체입니다.
* `result.content_type == "text"` — 파싱 또는 검증에 실패했습니다.
  원시 모델 응답은 `result.text`에서 확인하세요.

### 비동기

```python
result = await ctx.llm.acomplete(messages=..., task="classifier")
result = await ctx.llm.acomplete_structured(
    instructions=..., input=..., task="classifier"
)
```

동기 버전과 인자 및 결과 타입이 동일합니다. 게이트웨이 어댑터,
비동기 훅 또는 이미 asyncio 루프에서 실행 중인 모든 플러그인 코드에서
사용하세요.

### 작업으로 라우팅하는 보조 호출

플러그인에 자체 설정된 보조 라우트가 필요할 때 네 가지 호출 형태
어디에서든 `task=`를 전달할 수 있습니다. 플러그인 설정 중에 해당
작업을 등록하세요. 운영자가 `auxiliary.<task>`에서 프로바이더와
모델을 재정의하기 전까지는 플러그인 기본값이 적용됩니다.

```python
def register(ctx):
    ctx.register_auxiliary_task(
        "classifier", display_name="Classifier", description="Classify input."
    )


result = ctx.llm.complete(messages=[...], task="classifier")
result = ctx.llm.complete_structured(instructions=..., input=..., task="classifier")
```

```yaml
auxiliary:
  classifier:
    provider: openrouter
    model: vendor/model-id
```

플러그인은 자체 작업에 대해 프로바이더/모델 등록 기본값을 제공할 수
있습니다. `auxiliary.<task>`의 운영자 설정이 해당 기본값을 덮어쓰고
배포 선택을 제어합니다. 플러그인은 자신이 직접 등록한 작업만 사용할 수
있으며, 알 수 없거나 다른 플러그인의 작업 이름은 프로바이더 호출 전에
실패합니다. `allow_task_override: true`는 Hermes 내장 보조 작업을
사용하기 위한 명시적인 운영자 허용이며, 다른 플러그인의 작업을 허용하지는
않습니다. 활성 메인 프로바이더/모델을 유지하려면 `task=`를 생략하거나
(`"auto"`를 사용하여) 지정하세요.

### 결과 속성

```python
@dataclass
class PluginLlmCompleteResult:
    text: str                    # the assistant's response
    provider: str                # e.g. "openrouter", "anthropic"
    model: str                   # whatever the provider returned for this call
    agent_id: str                # whose model/auth was used
    usage: PluginLlmUsage        # tokens + cache + cost estimate
    audit: Dict[str, Any]        # plugin_id, purpose, profile

@dataclass
class PluginLlmStructuredResult:
    # same fields as PluginLlmCompleteResult, plus:
    parsed: Optional[Any]        # JSON object when content_type == "json"
    content_type: str            # "json" or "text"
    # audit also carries schema_name when supplied
```

`usage`에는 프로바이더가 해당 필드를 반환할 때
`input_tokens`, `output_tokens`, `total_tokens`,
`cache_read_tokens`, `cache_write_tokens`, `cost_usd`가 포함됩니다.

## 신뢰 게이트

기본 동작은 기본 거부입니다. `plugins.entries` 설정 블록이 없으면
플러그인은 다음을 수행할 수 있습니다.

* 사용자의 활성 프로바이더와 모델을 대상으로 네 가지 메서드 중
  어느 것이나 실행할 수 있습니다.
* 요청 형태 인자(`temperature`, `max_tokens`, `timeout`,
  `system_prompt`, `purpose`, `messages`, `instructions`,
  `input`, `json_schema`)를 설정할 수 있습니다.

…그 외에는 할 수 없습니다. 운영자가 허용할 때까지 `provider=`,
`model=`, `agent_id=`, `profile=` 인자를 사용하면
`PluginLlmTrustError`가 발생합니다. 마찬가지로 `task=`는 플러그인이
등록한 자체 보조 작업만 사용할 수 있으며, 운영자가 내장 작업에 대해
`allow_task_override`를 허용한 경우에만 예외가 적용됩니다.

**대부분의 플러그인은 이 섹션이 필요하지 않습니다.**
재정의 없이 `ctx.llm.complete(messages=...)`만 호출하는 플러그인은
사용자가 활성화한 항목을 대상으로 실행되며 설정 없이 작동합니다.
아래 블록은 플러그인이 사용자와 다른 모델이나 프로바이더를 명시적으로
고정하려는 경우에만 해당합니다.

```yaml
plugins:
  entries:
    my-plugin:
      llm:
        # Allow this plugin to choose a different Hermes provider
        # (must be one Hermes already knows about — same names as
        # `hermes model` and config.yaml model.provider).
        allow_provider_override: true

        # Optionally restrict which providers. Use ["*"] for any.
        allowed_providers:
          - openrouter
          - anthropic

        # Allow this plugin to ask for a specific model.
        allow_model_override: true

        # Optionally restrict which models. Use ["*"] for any.
        # Models are matched literally against whatever string the
        # plugin sends — Hermes does not look anything up.
        allowed_models:
          - openai/gpt-4o-mini
          - anthropic/claude-3-5-haiku

        # Allow cross-agent calls (rare).
        allow_agent_id_override: false

        # Allow the plugin to request a specific stored auth profile
        # (e.g. a different OAuth account on the same provider).
        allow_profile_override: false
```

플러그인 ID는 일반 플러그인의 매니페스트 `name:` 필드 또는
중첩 플러그인의 경로에서 파생된 키입니다(`image_gen/openai`,
`memory/honcho` 등).

### 게이트가 적용하는 항목

| 재정의 | 기본값 | 설정 키 |
| --------------- | ------- | -------------------------------- |
| `provider=` | 거부 | `allow_provider_override: true` |
| ↳ 허용 목록 | — | `allowed_providers: [...]` |
| `model=` | 거부 | `allow_model_override: true` |
| ↳ 허용 목록 | — | `allowed_models: [...]` |
| `agent_id=` | 거부 | `allow_agent_id_override: true` |
| `profile=` | 거부 | `allow_profile_override: true` |
| 내장 `task=` | 거부 | `allow_task_override: true` |

각 재정의는 독립적으로 게이트됩니다. `allow_model_override`를
허용해도 `allow_provider_override`까지 허용되는 것은 아닙니다.
모델 선택을 신뢰받은 플러그인도 프로바이더 게이트를 별도로 받지 않는 한
사용자의 활성 프로바이더에 계속 고정됩니다.

### 게이트가 적용할 필요가 없는 항목

* 요청 형태 인자 — `temperature`, `max_tokens`, `timeout`,
  `system_prompt`, `purpose`, `messages`, `instructions`,
  `input`, `json_schema`, `schema_name`, `json_mode` — 는 항상
  허용됩니다. 이들은 자격 증명이나 라우트를 선택하지 않습니다.
* 기본 거부 방식에서도 설정되지 않은 플러그인은 유용한 작업을 수행할 수
  있습니다. 단지 활성 프로바이더와 모델을 대상으로 실행될 뿐입니다.
  더 세밀한 라우팅을 원하는 플러그인일 때만 운영자가
  `plugins.entries`를 고려하면 됩니다.

## 호스트가 소유하는 것

플러그인이 직접 처리하지 않아도 되도록 `ctx.llm`이 처리하는 항목의
전체 목록입니다.

* **프로바이더 확인.** 사용자의 설정에서 `model.provider` +
  `model.model`을 읽습니다(신뢰된 경우 명시적 재정의 사용).
* **인증.** `~/.hermes/auth.json` / 환경 변수에서 API 키, OAuth 토큰
  또는 갱신 토큰을 가져오며, 설정되어 있으면 자격 증명 풀도 사용합니다.
  플러그인은 이를 볼 수 없습니다.
* **비전 라우팅.** 이미지 입력이 제공되고 사용자의 활성 텍스트 모델이
  텍스트 전용이면 호스트가 설정된 비전 모델로 자동 대체합니다.
* **대체 경로.** 사용자의 기본 프로바이더가 5xx 또는 429를 반환하면
  플러그인에 오류를 반환하기 전에 Hermes의 일반적인 애그리게이터 인식
  대체 경로를 거칩니다.
* **타임아웃.** `timeout=` 인자를 따르며, 없으면
  `auxiliary.<task>.timeout` 설정 또는 전역 보조 기본값으로 대체합니다.
* **JSON 구성.** JSON을 요청하면 프로바이더에 `response_format`을
  보내고, 프로바이더가 코드 펜스로 감싼 응답을 반환한 경우 로컬에서
  다시 파싱합니다.
* **스키마 검증.** `json_schema`에 맞게 검증하고
  `jsonschema`가 설치되어 있으면, 그렇지 않으면 디버그 한 줄을 기록한
  뒤 엄격한 검증을 건너뜁니다.
* **감사 로그.** 각 호출은 플러그인 ID, 프로바이더/모델, 목적,
  토큰 합계가 담긴 INFO 한 줄을 `agent.log`에 기록합니다.

## 플러그인이 소유하는 것

* **요청 형태.** 채팅에는 `messages`, 구조화된 출력에는
  `instructions` + `input`을 사용합니다. 플러그인이 프롬프트를
  구성하고 호스트가 실행합니다.
* **스키마.** 반환받고 싶은 형태는 무엇이든 사용할 수 있습니다.
  호스트가 대신 추론해 주지는 않습니다.
* **오류 처리.** `complete_structured()`는 빈 입력 및 스키마 검증
  실패 시 `ValueError`를 발생시킵니다. 신뢰 게이트가 재정의를
  거부하면 `PluginLlmTrustError`가 발생합니다. 그 외의 오류
  (프로바이더 5xx, 설정된 자격 증명 없음, 타임아웃)는
  `auxiliary_client.call_llm()`이 발생시키는 오류를 그대로 발생시킵니다.
* **비용.** 모든 호출은 사용자의 유료 프로바이더를 대상으로 실행됩니다.
  토큰 비용을 고려하지 않고 모든 게이트웨이 메시지마다
  `complete()`를 반복 호출하지 마세요.

## 플러그인 표면에서의 위치

기존 `ctx.*` 메서드는 Hermes의 기존 하위 시스템을 확장합니다.

| `ctx.register_tool` | 에이전트가 호출할 수 있는 도구를 추가 |
| `ctx.register_platform` | 새 게이트웨이 어댑터 연결 |
| `ctx.register_image_gen_provider` | 이미지 생성 백엔드 교체 |
| `ctx.register_memory_provider` | 메모리 백엔드 교체 |
| `ctx.register_context_engine` | 컨텍스트 압축기 교체 |
| `ctx.register_hook` | 라이프사이클 이벤트 관찰 |

`ctx.llm`은 위 기능 중 어느 것도 사용하지 않고 플러그인이 사용자가
대화 중인 모델과 동일한 모델을 *대화 외부에서* 실행할 수 있게 하는
첫 번째 표면입니다. 이것이 이 기능의 유일한 목적입니다. 에이전트가
호출할 도구를 등록하려면 `register_tool`을 사용하세요. 라이프사이클
이벤트에 반응하려면 `register_hook`을 사용하세요. 구조화 여부와
관계없이 자체 모델 호출을 실행하려면 `ctx.llm`을 사용하세요.

## 참고 자료

* 구현: [`agent/plugin_llm.py`](https://github.com/NousResearch/hermes-agent/blob/main/agent/plugin_llm.py)
* 테스트: [`tests/agent/test_plugin_llm.py`](https://github.com/NousResearch/hermes-agent/blob/main/tests/agent/test_plugin_llm.py)
* 참조 플러그인(동반 저장소):
  * [`plugin-llm-example`](https://github.com/NousResearch/hermes-example-plugins/tree/main/plugin-llm-example) — 이미지 입력을 사용하는 동기 구조화 추출
  * [`plugin-llm-async-example`](https://github.com/NousResearch/hermes-example-plugins/tree/main/plugin-llm-async-example) — `asyncio.gather()`를 사용하는 비동기 호출
* 보조 클라이언트(내부 엔진): [프로바이더 런타임](/developer-guide/provider-runtime)을 참조하세요.

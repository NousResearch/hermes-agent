---
sidebar_position: 11
title: "플러그인 LLM 액세스 (Plugin LLM Access)"
description: "ctx.llm을 통해 플러그인 내부에서 모든 LLM 호출 실행 — 채팅 또는 구조화된 출력, 동기식 또는 비동기식 처리. 호스트가 소유한 인증, 페일-클로즈 신뢰 게이트, 선택적 JSON 스키마 유효성 검사 지원."
---

# 플러그인 LLM 액세스 (Plugin LLM Access)

플러그인이 LLM 호출을 만들 때 지원되는 공식 방법은 `ctx.llm`입니다.
채팅 완성(chat completion), 구조화된 데이터 추출, 동기/비동기 호출, 이미지 포함 여부에 상관없이 동일한 인터페이스, 동일한 신뢰 게이트(trust gate), 동일한 호스트 소유 자격 증명(credentials)을 사용합니다.

플러그인 개발자들은 모델과 관련된 작업이 필요하지만 에이전트의 메인 대화 루프에는 속하지 않아야 할 때 이 기능을 사용합니다. 예를 들어 도구의 오류 메시지를 일반 사용자가 읽을 수 있도록 재작성하는 훅, 메시지를 큐에 넣기 전에 번역하는 게이트웨이 어댑터, 긴 텍스트의 요약본을 제공하는 슬래시 명령어, 어제 활동을 평가하고 상태 보드에 한 줄을 적는 스케줄링된 작업, 에이전트를 깨울 만한 메시지인지 미리 결정하는 프리필터(pre-filter) 등이 있습니다.

이러한 작업에는 에이전트가 개입하지 않아야 합니다. 단 하나의 LLM 호출과 타입이 지정된 답변(typed answer)을 받은 후 마무리되어야 합니다.

## 가장 작은 호출 형태

```python
result = ctx.llm.complete(messages=[{"role": "user", "content": "ping"}])
return result.text
```

위의 단 한 줄이 전체 API의 모습입니다. API 키도, 제공자(provider) 설정도, SDK 초기화도 필요 없습니다. 플러그인은 사용자가 현재 사용 중인 제공자와 모델에 맞추어 실행됩니다 — 사용자가 제공자를 바꾸면 플러그인도 자동으로 따라갑니다.

## 더 완성도 있는 채팅 예시

```python
result = ctx.llm.complete(
    messages=[
        {"role": "system", "content": "비개발자도 이해하고 조치를 취할 수 있도록 짧은 한 문장으로 오류를 다시 작성하세요."},
        {"role": "user",   "content": traceback_text},
    ],
    max_tokens=64,
    purpose="hooks.error-rewrite",
)
return result.text
```

`purpose`는 자유 형식의 감사(audit) 문자열입니다 — 이는 `agent.log`와 `result.audit`에 표시되므로, 운영자가 어떤 플러그인이 어떤 호출을 했는지 파악할 수 있게 해줍니다. 선택 사항이지만 자주 실행되는 작업에는 지정하는 것을 권장합니다.

## 구조화된 출력 (Structured output)

플러그인에서 특정 타입의 답변이 필요할 때는, 구조화된 출력 기능을 사용하세요:

```python
result = ctx.llm.complete_structured(
    instructions="이 고객 지원 답변을 분석하여 긴급도(0–1) 점수를 매기고 카테고리를 선택하세요.",
    input=[{"type": "text", "text": message_body}],
    json_schema=TRIAGE_SCHEMA,
    purpose="support.triage",
    temperature=0.0,
    max_tokens=128,
)

if result.parsed["urgency"] > 0.8:
    await dispatch_to_oncall(result.parsed["category"], message_body)
```

호스트는 제공자(provider)에게 JSON 출력을 요청하고, 대체 수단(fallback)으로 로컬에서 파싱하며, `jsonschema` 패키지가 설치되어 있는 경우 제공된 스키마에 대해 유효성 검사를 수행한 후 파이썬 객체를 `result.parsed`에 반환합니다. 모델이 유효한 JSON을 생성하지 못한 경우 `result.parsed`는 `None`이 되고 `result.text`에는 원본 응답 문자가 담깁니다.

## 이 기능이 제공하는 이점

* **단 하나의 호출, 네 가지 형태.** 채팅을 위한 `complete()`, 형식화된 JSON을 위한 `complete_structured()`, 그리고 비동기처리를 위한 `acomplete()` 및 `acomplete_structured()`. 모두 같은 인자를 받고 동일한 형태의 결과 객체를 반환합니다.
* **호스트 소유 자격 증명.** OAuth 토큰, 토큰 갱신(refresh) 과정, 자격 증명 풀, 작업별 보조(aux) 재정의 — Hermes가 갖춘 모든 자격 증명 개념이 적용됩니다. 플러그인은 토큰을 볼 수 없습니다. 호스트는 `result.audit`을 통해 호출 이력을 기록합니다.
* **제한된 범위(Bounded).** 단일 동기 또는 비동기 호출입니다. 스트리밍도 없고, 도구 반복 호출(tool loops)도 없으며, 관리할 대화 상태(conversation state)도 없습니다. 입력을 선언하고, 결과를 받아 반환하기만 하면 됩니다.
* **페일-클로즈 신뢰 (Fail-closed trust).** 명시적으로 설정하지 않은 플러그인은 자체적으로 제공자, 모델, 에이전트, 저장된 자격 증명을 선택할 수 없습니다. 기본 기조는 "사용자가 쓰고 있는 것을 쓴다"는 것입니다. 필요한 경우 운영자가 `config.yaml`의 플러그인별 설정을 통해 명시적으로 선택적 재정의(override)를 활성화해야 합니다.

## 빠른 시작 (Quick start)

아래는 단일 `register(ctx)` 함수 내에서 구현되고 별도의 설정 없이 사용자의 활성 모델에서 실행되는 채팅 플러그인과 구조화된 플러그인 두 가지의 완전한 예시입니다.

### 채팅 완성 (Chat completion) — `/tldr`

```python
def register(ctx):
    ctx.register_command(
        name="tldr",
        handler=lambda raw: _tldr(ctx, raw),
        description="제공된 텍스트를 한 문단으로 요약합니다.",
        args_hint="<text>",
    )


def _tldr(ctx, raw_args: str) -> str:
    text = raw_args.strip()
    if not text:
        return "사용법: /tldr <요약할 텍스트>"
    result = ctx.llm.complete(
        messages=[
            {"role": "system",
             "content": "사용자의 텍스트를 간결한 한 문단으로 요약하세요. 서두는 생략하세요."},
            {"role": "user", "content": text},
        ],
        max_tokens=256,
        temperature=0.3,
        purpose="tldr",
    )
    return result.text
```

`result.text`는 모델의 응답이고; `result.usage`에는 토큰 사용량이 들어 있으며; `result.provider`와 `result.model`에는 어떤 모델을 사용했는지에 대한 정보가 담겨있습니다.

### 구조화된 데이터 추출 (Structured extraction) — `/paste-to-tasks`

```python
def register(ctx):
    ctx.register_command(
        name="paste-to-tasks",
        handler=lambda raw: _paste_to_tasks(ctx, raw),
        description="자유 형식의 회의 노트를 구조화된 작업(tasks)으로 변환합니다.",
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
                    "due":    {"type": "string", "description": "ISO 날짜 형식 또는 공백"},
                },
                "required": ["action"],
            },
        },
    },
    "required": ["tasks"],
}


def _paste_to_tasks(ctx, raw_args: str) -> str:
    if not raw_args.strip():
        return "사용법: /paste-to-tasks <회의 노트>"
    result = ctx.llm.complete_structured(
        instructions=(
            "이 회의 노트에서 구체적인 조치 항목(action items)을 추출하세요. "
            "실행 가능한 한 줄마다 하나의 태스크를 만드세요. 만약 담당자(owner)가 명시되지 않았다면 'owner'를 비워두세요."
        ),
        input=[{"type": "text", "text": raw_args}],
        json_schema=_TASKS_SCHEMA,
        schema_name="meeting.tasks",
        purpose="paste-to-tasks",
        temperature=0.0,
        max_tokens=512,
    )
    if result.parsed is None:
        return f"응답을 파싱할 수 없습니다. 원본 출력:\n{result.text}"
    lines = [f"- [{t.get('owner') or '?'}] {t['action']}" for t in result.parsed["tasks"]]
    return "\n".join(lines) or "(찾은 태스크가 없습니다)"
```

세 번째 작동 예시로, 이미지 입력이 포함된 버전은 [`hermes-example-plugins`](https://github.com/NousResearch/hermes-example-plugins/tree/main/plugin-llm-example) 저장소(레퍼런스 플러그인용 추가 저장소 — hermes-agent 자체와 함께 번들로 제공되지 않음)에 있습니다. 비동기 환경(`asyncio.gather()`와 함께 사용하는 `acomplete()` / `acomplete_structured()`)에 대해서는 같은 저장소의 [`plugin-llm-async-example`](https://github.com/NousResearch/hermes-example-plugins/tree/main/plugin-llm-async-example)을 참조하세요.

## 언제 무엇을 사용해야 할까?

| 필요 사항 | 적합한 도구 |
|---|---|
| 자유 형식의 텍스트 응답 (번역, 요약, 재작성, 텍스트 생성) | `complete()` |
| 다중 턴 프롬프트 (시스템 + 몇 가지 예시 + 사용자) | `complete()` |
| 스키마로 검증된 타입 딕셔너리(typed dict) 형태의 반환 값 | `complete_structured()` |
| 이미지 또는 텍스트 입력과 그에 대한 타입 딕셔너리 반환 | `complete_structured()` |
| 비동기 코드에서 위의 작업 호출 (게이트웨이 어댑터, 비동기 훅) | `acomplete()` / `acomplete_structured()` |

제공자 선택, 모델 확인, 인증, 대체수단, 타임아웃, 비전 라우팅 등 나머지 모든 것은 위 4가지에서 동일하게 작동합니다.

## API 목록 (API surface)

`ctx.llm`은 `agent.plugin_llm.PluginLlm`의 인스턴스입니다.

### `complete()`

```python
result = ctx.llm.complete(
    messages=[{"role": "user", "content": "Hi"}],
    provider=None,         # 선택 사항, 신뢰 게이트 통과 필요 — Hermes 제공자 ID (예: "openrouter")
    model=None,            # 선택 사항, 신뢰 게이트 통과 필요 — 해당 제공자가 예상하는 문자열
    temperature=None,
    max_tokens=None,
    timeout=None,          # 초 단위 (seconds)
    agent_id=None,         # 선택 사항, 신뢰 게이트 통과 필요
    profile=None,          # 선택 사항, 신뢰 게이트 통과 필요 — 명시적 인증 프로필 이름
    purpose="optional-audit-string",
)
# → PluginLlmCompleteResult(text, provider, model, agent_id, usage, audit)
```

일반 채팅 완성입니다. `messages`는 `{"role": "...", "content": "..."}` 형태의 딕셔너리 목록인 표준 OpenAI 형식입니다. 다중 턴 프롬프트(시스템 프롬프트 + 몇 개의 사용자/어시스턴트 질문/답변 쌍 + 최종 사용자 프롬프트)는 OpenAI SDK와 정확히 동일하게 작동합니다.

`provider=` 와 `model=` 인자는 서로 독립적이며, 호스트의 메인 설정(`model.provider` + `model.model`)과 동일한 형태를 따릅니다. 사용자의 활성화된 제공자에서 다른 모델을 사용하려면 `model=` 값만 설정하세요. 제공자 자체를 바꾸려면 둘 다 설정하세요. 운영자의 수락(opt-in) 없이 이 두 인자 중 하나라도 사용하면 `PluginLlmTrustError`가 발생합니다.

### `complete_structured()`

```python
result = ctx.llm.complete_structured(
    instructions="What you want extracted.",
    input=[
        {"type": "text",  "text": "..."},
        {"type": "image", "data": b"...", "mime_type": "image/png"},
        {"type": "image", "url":  "https://..."},
    ],
    json_schema={...},     # 선택 사항 — 결과 파싱 및 유효성 검사를 트리거함
    json_mode=False,       # 스키마 없이 무조건 JSON 결과물을 요구할 경우 True로 설정
    schema_name=None,      # 선택적 사람이 읽을 수 있는 스키마 이름
    system_prompt=None,
    provider=None,         # 선택 사항, 신뢰 게이트 통과 필요
    model=None,            # 선택 사항, 신뢰 게이트 통과 필요
    temperature=None,
    max_tokens=None,
    timeout=None,
    agent_id=None,
    profile=None,
    purpose=None,
)
# → PluginLlmStructuredResult(text, provider, model, agent_id,
#                             usage, parsed, content_type, audit)
```

입력은 텍스트 또는 이미지 블록 타입입니다 (원시 바이트(raw bytes) 데이터는 자동으로 `data:` URL로 base64 인코딩됩니다). `json_schema`나 `json_mode=True`가 주어지면, 호스트는 제공자에게 `response_format`을 통해 JSON 출력을 요구하고, 폴백(fallback)으로 자체 로컬 파싱을 수행하며, `jsonschema` 패키지가 있다면 스키마와 대조하여 유효성을 검사합니다.

* `result.content_type == "json"` — `result.parsed`는 사용자의 스키마와 일치하는 파이썬 객체입니다.
* `result.content_type == "text"` — 파싱이나 유효성 검사가 실패했습니다; 모델이 만든 원본 응답 내용을 보려면 `result.text`를 살펴보세요.

### 비동기 (Async)

```python
result = await ctx.llm.acomplete(messages=...)
result = await ctx.llm.acomplete_structured(instructions=..., input=...)
```

동기 버전과 인자 및 반환 타입이 동일합니다. 게이트웨이 어댑터, 비동기 훅 또는 이미 asyncio 루프에서 실행 중인 플러그인 코드에서 이를 사용하세요.

### 결과 속성 (Result attributes)

```python
@dataclass
class PluginLlmCompleteResult:
    text: str                    # 어시스턴트의 응답
    provider: str                # 예: "openrouter", "anthropic"
    model: str                   # 이 호출에서 제공자가 반환한 모델명
    agent_id: str                # 누구의 모델/인증이 사용되었는지 식별
    usage: PluginLlmUsage        # 사용된 토큰 + 캐시 + 비용 추정치
    audit: Dict[str, Any]        # plugin_id, purpose, profile

@dataclass
class PluginLlmStructuredResult(PluginLlmCompleteResult):
    parsed: Optional[Any]        # content_type == "json" 인 경우 파싱된 JSON 객체
    content_type: str            # "json" 또는 "text"
    # 스키마 이름이 제공된 경우 audit 속성에도 schema_name이 포함됩니다
```

`usage`에는 제공자가 지원하는 경우 `input_tokens`, `output_tokens`, `total_tokens`, `cache_read_tokens`, `cache_write_tokens`, 및 `cost_usd`가 포함됩니다.

## 신뢰 게이트 (Trust gate)

기본 동작은 페일-클로즈(fail-closed) 상태입니다. `plugins.entries` 설정 블록이 없다면 플러그인이 할 수 있는 작업은 다음과 같습니다:

* 사용자의 활성 제공자 및 모델에 대해 4가지 메서드 중 하나를 실행,
* 호출 형태(request-shaping) 인자 설정 (`temperature`, `max_tokens`, `timeout`, `system_prompt`, `purpose`, `messages`, `instructions`, `input`, `json_schema`),

…이게 전부입니다. `provider=`, `model=`, `agent_id=`, 및 `profile=` 인자는 운영자가 명시적으로 허용하기 전까지 `PluginLlmTrustError`를 발생시킵니다.

**대부분의 플러그인은 이 섹션이 필요하지 않습니다.** 재정의(overrides) 없이 `ctx.llm.complete(messages=...)` 만 호출하는 플러그인은 제로 컨피그(zero-config) 상태로 사용자의 활성 모델에서 정상 작동합니다. 아래 설정 블록은 오직 사용자가 지정한 모델/제공자가 아닌 특정 모델이나 제공자에 종속되어야만 하는 플러그인에게만 관련이 있습니다.

```yaml
plugins:
  entries:
    my-plugin:
      llm:
        # 이 플러그인이 다른 Hermes 제공자를 선택할 수 있도록 허용합니다
        # (Hermes가 이미 알고 있는 제공자 중 하나여야 합니다 — 
        # `hermes model` 및 config.yaml의 model.provider와 동일한 이름).
        allow_provider_override: true

        # 선택적으로 특정 제공자만 허용하도록 제한합니다. 모두 허용하려면 ["*"]를 사용하세요.
        allowed_providers:
          - openrouter
          - anthropic

        # 이 플러그인이 특정 모델을 요청할 수 있도록 허용합니다.
        allow_model_override: true

        # 선택적으로 특정 모델만 허용하도록 제한합니다. 모두 허용하려면 ["*"]를 사용하세요.
        # 모델은 플러그인이 전송하는 문자열과 정확히 일치(literally matched)해야 합니다 — Hermes가 이를 자동으로 조회하거나 매핑해주지 않습니다.
        allowed_models:
          - openai/gpt-4o-mini
          - anthropic/claude-3-5-haiku

        # 교차 에이전트 호출을 허용합니다 (드묾).
        allow_agent_id_override: false

        # 플러그인이 저장된 특정 인증 프로필을 요청할 수 있도록 허용합니다
        # (예: 동일 제공자에 대해 서로 다른 OAuth 계정 사용).
        allow_profile_override: false
```

플러그인 ID는 기본 구조의 플러그인의 경우 manifest의 `name:` 필드이거나 중첩된 플러그인(`image_gen/openai`, `memory/honcho` 등)의 경우 경로에서 파생된 키입니다.

### 신뢰 게이트가 강제하는 것들

| 재정의 (Override) | 기본값 | 설정 키 (Config key)                 |
| --------------- | ------- | -------------------------------- |
| `provider=`     | 거부됨  | `allow_provider_override: true`  |
| ↳ 허용 목록     | —       | `allowed_providers: [...]`       |
| `model=`        | 거부됨  | `allow_model_override: true`     |
| ↳ 허용 목록     | —       | `allowed_models: [...]`          |
| `agent_id=`     | 거부됨  | `allow_agent_id_override: true`  |
| `profile=`      | 거부됨  | `allow_profile_override: true`   |

각 재정의는 독립적으로 제어됩니다. `allow_model_override`를 부여한다고 해서 `allow_provider_override`가 자동으로 함께 부여되는 것은 **아닙니다** — 모델을 지정하도록 신뢰받은 플러그인이라도, 제공자에 대한 권한을 따로 얻지 않는 한 여전히 사용자의 활성화된 제공자에 고정됩니다.

### 신뢰 게이트가 강제하지 않는 (허용하는) 것들

* 요청을 형성하는 인자들 — `temperature`, `max_tokens`, `timeout`, `system_prompt`, `purpose`, `messages`, `instructions`, `input`, `json_schema`, `schema_name`, `json_mode` — 이들은 언제나 허용됩니다; 이들은 자격 증명이나 라우팅을 선택하지 않습니다.
* 기본 거부 방침은 구성되지 않은(unconfigured) 플러그인이라도 사용자의 활성 제공자와 모델에서 동작하므로 유용한 작업을 할 수 있음을 의미합니다. 운영자는 세밀한 라우팅이 필요한 플러그인에 대해서만 `plugins.entries`를 신경 쓰면 됩니다.

## 호스트(Host)가 관리하는 것들

플러그인 작성자가 직접 하지 않아도 되게끔 `ctx.llm`이 대신 처리해주는 모든 목록입니다:

* **제공자 확인 (Provider resolution).** 사용자 설정(신뢰할 경우 명시적 재정의 포함)에서 `model.provider` + `model.model`을 읽습니다.
* **인증 (Auth).** 자격 증명 풀(구성된 경우)을 포함하여 `~/.hermes/auth.json` 또는 환경 변수에서 API 키, OAuth 토큰, 갱신 토큰(refresh tokens)을 가져옵니다. 플러그인은 이들을 볼 수 없습니다.
* **비전 라우팅 (Vision routing).** 이미지 입력이 제공되었는데 사용자의 텍스트 모델이 텍스트 전용일 경우 호스트는 자동으로 설정된 비전 모델로 대체(fallback)합니다.
* **대체수단 체인 (Fallback chain).** 사용자의 기본 제공자가 5xx나 429 오류를 뱉어내면, 플러그인에 에러를 반환하기 전에 Hermes의 애그리게이터를 인지한 폴백 경로를 통해 재시도합니다.
* **타임아웃 (Timeout).** 사용자가 설정한 `timeout=` 값을 따르거나 지정되지 않은 경우 `auxiliary.<task>.timeout` 설정이나 전역 aux 기본값으로 대체합니다.
* **JSON 형태 지정 (JSON shaping).** JSON을 요청할 때 제공자에게 `response_format`을 전송하며, 제공자가 마크다운 코드 블록 형태로 넘겨줄 경우 내부적으로 다시 파싱합니다.
* **스키마 유효성 검사 (Schema validation).** `jsonschema`가 설치되어 있으면 `json_schema`에 대해 유효성 검사를 수행합니다. 그렇지 않은 경우 엄격한 검사를 건너뛰고 디버그 라인을 기록합니다.
* **감사 로그 (Audit log).** 매 호출마다 플러그인 ID, 제공자/모델, 목적(purpose), 토큰 합계가 기록된 정보 수준(INFO) 한 줄을 `agent.log`에 남깁니다.

## 플러그인(Plugin)이 관리하는 것들

* **요청 형태 (Request shape).** 채팅용 `messages`, 구조화된 추출용 `instructions` + `input`. 플러그인이 프롬프트를 만들고, 호스트가 그것을 실행합니다.
* **스키마 (Schema).** 여러분이 돌려받길 원하는 어떠한 형태든 지원합니다. 호스트가 그것을 대신 추론해주지는 않습니다.
* **에러 핸들링 (Error handling).** `complete_structured()`는 입력이 비어 있거나 스키마 유효성 검사에 실패하면 `ValueError`를 발생시킵니다. 신뢰 게이트에서 재정의가 거부되면 `PluginLlmTrustError`가 발생합니다. 그 밖의 모든 것(제공자 측 5xx 에러, 인증 자격 증명 미설정, 타임아웃 등)은 `auxiliary_client.call_llm()`이 뱉는 예외를 그대로 발생시킵니다.
* **비용 (Cost).** 모든 호출은 유료인 사용자의 제공자 계정으로 작동합니다. 토큰 비용을 고려하지 않고 게이트웨이로 들어오는 모든 메시지에 대해 `complete()`를 무작정 반복(loop) 호출하지 않도록 주의하세요.

## 플러그인 생태계에서 이 기능의 위치

기존의 `ctx.*` 메서드들은 이미 존재하는 Hermes 하위 시스템을 확장합니다:

| `ctx.register_tool` | 에이전트가 호출할 수 있는 도구를 추가합니다 |
| `ctx.register_platform` | 새로운 게이트웨이 어댑터를 연결합니다 |
| `ctx.register_image_gen_provider` | 이미지 생성 백엔드를 교체합니다 |
| `ctx.register_memory_provider` | 메모리 백엔드를 교체합니다 |
| `ctx.register_context_engine` | 컨텍스트 압축기(context compressor)를 교체합니다 |
| `ctx.register_hook` | 생명주기 이벤트(lifecycle event)를 관찰합니다 |

`ctx.llm`은 플러그인이 위의 기존 요소들을 거치지 않고도 *대역 외(out of band)에서* 사용자가 현재 대화 중인 모델과 똑같은 모델을 실행할 수 있게 해주는 첫 번째 API입니다. 오직 이것만이 `ctx.llm`의 임무입니다.
에이전트가 사용할 도구를 만들어야 한다면 `register_tool`을 쓰세요. 특정 생명주기 이벤트에 반응해야 한다면 `register_hook`을 쓰세요. 그 밖의 어떤 이유로든(구조화 여부에 관계 없이) 플러그인 자체가 모델 호출을 실행해야 한다면, 그때 `ctx.llm`을 사용하면 됩니다.

## 참조 (Reference)

* 구현: [`agent/plugin_llm.py`](https://github.com/NousResearch/hermes-agent/blob/main/agent/plugin_llm.py)
* 테스트: [`tests/agent/test_plugin_llm.py`](https://github.com/NousResearch/hermes-agent/blob/main/tests/agent/test_plugin_llm.py)
* 레퍼런스 플러그인 (관련 저장소):
  * [`plugin-llm-example`](https://github.com/NousResearch/hermes-example-plugins/tree/main/plugin-llm-example) — 이미지 입력을 포함한 동기식(sync) 구조화 추출
  * [`plugin-llm-async-example`](https://github.com/NousResearch/hermes-example-plugins/tree/main/plugin-llm-async-example) — `asyncio.gather()`와 함께 사용하는 비동기식(async) 추출
* 보조 클라이언트 (내부 작동 엔진):
  [제공자 런타임 (Provider Runtime)](/developer-guide/provider-runtime) 문서를 참조하세요.

---
sidebar_position: 5
title: "Hermes를 Python 라이브러리로 사용하기"
description: "고유한 Python 스크립트, 웹 앱 또는 자동화 파이프라인에 AIAgent를 임베드하세요 — CLI 불필요"
---

# Hermes를 Python 라이브러리로 사용하기

Hermes는 단순한 CLI 도구가 아닙니다. `AIAgent`를 직접 임포트하여 고유한 Python 스크립트, 웹 애플리케이션 또는 자동화 파이프라인에서 프로그래밍 방식으로 사용할 수 있습니다. 이 가이드에서는 그 방법을 보여줍니다.

---

## 설치

저장소에서 Hermes를 직접 설치하세요:

```bash
pip install git+https://github.com/NousResearch/hermes-agent.git
```

또는 [uv](https://docs.astral.sh/uv/)를 사용하여 설치:

```bash
uv pip install git+https://github.com/NousResearch/hermes-agent.git
```

`requirements.txt`에 핀(pin)으로 고정할 수도 있습니다:

```text
hermes-agent @ git+https://github.com/NousResearch/hermes-agent.git
```

:::tip
라이브러리로서 Hermes를 사용할 때에도 CLI에서 사용하는 것과 동일한 환경 변수가 필요합니다. 최소한 `OPENROUTER_API_KEY` (또는 제공자에게 직접 접근하는 경우 `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`)를 설정하세요.
:::

---

## 기본 사용법

Hermes를 사용하는 가장 간단한 방법은 `chat()` 메서드입니다 — 메시지를 전달하고 문자열을 반환받습니다:

```python
from run_agent import AIAgent

agent = AIAgent(
    model="anthropic/claude-sonnet-4.6",
    quiet_mode=True,
)
response = agent.chat("What is the capital of France?")
print(response)
```

`chat()`은 도구 호출, 재시도 등 전체 대화 루프를 내부적으로 처리하고 최종 텍스트 응답만을 반환합니다.

:::warning
사용자의 코드에 Hermes를 임베드할 때는 항상 `quiet_mode=True`를 설정하세요. 이 설정이 없으면 에이전트가 CLI 스피너, 진행률 표시기 및 기타 터미널 출력을 인쇄하여 애플리케이션의 출력을 어지럽히게 됩니다.
:::

---

## 전체 대화 제어 (Full Conversation Control)

대화에 대한 더 많은 제어가 필요하다면 `run_conversation()`을 직접 사용하세요. 전체 응답, 메시지 내역 및 메타데이터가 포함된 딕셔너리(dictionary)를 반환합니다:

```python
agent = AIAgent(
    model="anthropic/claude-sonnet-4.6",
    quiet_mode=True,
)

result = agent.run_conversation(
    user_message="최신 Python 3.13 기능을 검색해줘",
    task_id="my-task-1",
)

print(result["final_response"])
print(f"교환된 메시지 수: {len(result['messages'])}")
```

반환된 딕셔너리에는 다음이 포함됩니다:
- **`final_response`** — 에이전트의 최종 텍스트 답변
- **`messages`** — 전체 메시지 내역 (시스템, 사용자, 어시스턴트, 도구 호출)

(전달한 `task_id`는 VM 격리를 위해 에이전트 인스턴스에 저장되지만 반환 딕셔너리에는 포함되지 않습니다.)

해당 호출에 대한 일회성 시스템 프롬프트를 덮어쓰는 사용자 지정 시스템 메시지를 전달할 수도 있습니다:

```python
result = agent.run_conversation(
    user_message="퀵 정렬(quicksort)을 설명해줘",
    system_message="당신은 컴퓨터 과학 튜터입니다. 간단한 비유를 사용하세요.",
)
```

---

## 도구 구성 (Configuring Tools)

`enabled_toolsets` 또는 `disabled_toolsets`를 사용하여 에이전트가 액세스할 수 있는 도구 모음을 제어하세요:

```python
# 웹 도구(브라우징, 검색)만 활성화
agent = AIAgent(
    model="anthropic/claude-sonnet-4.6",
    enabled_toolsets=["web"],
    quiet_mode=True,
)

# 터미널 액세스를 제외한 모든 기능 활성화
agent = AIAgent(
    model="anthropic/claude-sonnet-4.6",
    disabled_toolsets=["terminal"],
    quiet_mode=True,
)
```

:::tip
최소한으로 제한된 에이전트가 필요할 때(예: 리서치 봇을 위한 웹 검색 전용) `enabled_toolsets`를 사용하세요. 대부분의 기능이 필요하지만 특정 기능을 제한해야 할 때(예: 공유 환경에서 터미널 액세스 금지) `disabled_toolsets`를 사용하세요.
:::

---

## 다중 턴 대화 (Multi-turn Conversations)

메시지 내역을 다시 전달하여 여러 턴(turn)에 걸쳐 대화 상태를 유지하세요:

```python
agent = AIAgent(
    model="anthropic/claude-sonnet-4.6",
    quiet_mode=True,
)

# 첫 번째 턴
result1 = agent.run_conversation("제 이름은 Alice입니다")
history = result1["messages"]

# 두 번째 턴 — 에이전트가 컨텍스트를 기억합니다
result2 = agent.run_conversation(
    "내 이름이 뭐지?",
    conversation_history=history,
)
print(result2["final_response"])  # "당신의 이름은 Alice입니다."
```

`conversation_history` 매개변수는 이전 결과의 `messages` 목록을 허용합니다. 에이전트는 이를 내부적으로 복사하므로 원래 목록은 절대 변형되지 않습니다.

---

## 궤적 저장 (Saving Trajectories)

훈련 데이터 생성이나 디버깅에 유용한 ShareGPT 형식으로 대화를 캡처하려면 궤적 저장을 활성화하세요:

```python
agent = AIAgent(
    model="anthropic/claude-sonnet-4.6",
    save_trajectories=True,
    quiet_mode=True,
)

agent.chat("리스트를 정렬하는 Python 함수를 작성해")
# ShareGPT 형식의 trajectory_samples.jsonl에 저장됩니다
```

각 대화는 단일 JSONL 줄로 추가되므로 자동화된 실행에서 데이터 세트를 수집하기 쉽습니다.

---

## 사용자 지정 시스템 프롬프트 (Custom System Prompts)

에이전트의 동작을 안내하지만 궤적 파일에는 **저장되지 않는** (훈련 데이터를 깔끔하게 유지) 사용자 지정 시스템 프롬프트를 설정하려면 `ephemeral_system_prompt`를 사용하세요:

```python
agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    ephemeral_system_prompt="당신은 SQL 전문가입니다. 데이터베이스 질문에만 대답하세요.",
    quiet_mode=True,
)

response = agent.chat("JOIN 쿼리를 어떻게 작성하나요?")
print(response)
```

이 기능은 모두 동일한 기본 도구를 사용하면서 코드 리뷰어, 문서 작성자, SQL 어시스턴트 등 전문화된 에이전트를 구축하는 데 이상적입니다.

---

## 일괄 처리 (Batch Processing)

많은 프롬프트를 병렬로 실행하기 위해 Hermes에는 `batch_runner.py`가 포함되어 있습니다. 적절한 리소스 격리와 함께 동시 `AIAgent` 인스턴스를 관리합니다:

```bash
python batch_runner.py --input prompts.jsonl --output results.jsonl
```

각 프롬프트는 고유한 `task_id`와 격리된 환경을 갖습니다. 사용자 지정 일괄 로직이 필요한 경우 `AIAgent`를 직접 사용하여 고유한 로직을 구축할 수 있습니다:

```python
import concurrent.futures
from run_agent import AIAgent

prompts = [
    "재귀(recursion)에 대해 설명해",
    "해시 테이블(hash table)이 뭐야?",
    "가비지 컬렉션(garbage collection)은 어떻게 작동해?",
]

def process_prompt(prompt):
    # 스레드 안전성을 위해 작업당 새로운 에이전트를 생성
    agent = AIAgent(
        model="anthropic/claude-sonnet-4",
        quiet_mode=True,
        skip_memory=True,
    )
    return agent.chat(prompt)

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(process_prompt, prompts))

for prompt, result in zip(prompts, results):
    print(f"질문: {prompt}\n답변: {result}\n")
```

:::warning
항상 **스레드나 작업당 새로운 `AIAgent` 인스턴스**를 생성하세요. 에이전트는 공유하기에 스레드에 안전하지 않은(not thread-safe) 내부 상태(대화 기록, 도구 세션, 반복 카운터)를 유지합니다.
:::

---

## 통합 예시 (Integration Examples)

### FastAPI 엔드포인트

```python
from fastapi import FastAPI
from pydantic import BaseModel
from run_agent import AIAgent

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    model: str = "anthropic/claude-sonnet-4"

@app.post("/chat")
async def chat(request: ChatRequest):
    agent = AIAgent(
        model=request.model,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    response = agent.chat(request.message)
    return {"response": response}
```

### Discord 봇

```python
import discord
from run_agent import AIAgent

client = discord.Client(intents=discord.Intents.default())

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.content.startswith("!hermes "):
        query = message.content[8:]
        agent = AIAgent(
            model="anthropic/claude-sonnet-4",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            platform="discord",
        )
        response = agent.chat(query)
        await message.channel.send(response[:2000])

client.run("YOUR_DISCORD_TOKEN")
```

### CI/CD 파이프라인 단계

```python
#!/usr/bin/env python3
"""CI step: auto-review a PR diff."""
import subprocess
from run_agent import AIAgent

diff = subprocess.check_output(["git", "diff", "main...HEAD"]).decode()

agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    quiet_mode=True,
    skip_context_files=True,
    skip_memory=True,
    disabled_toolsets=["terminal", "browser"],
)

review = agent.chat(
    f"이 PR diff에서 버그, 보안 문제 및 스타일 문제를 검토해:\n\n{diff}"
)
print(review)
```

---

## 주요 생성자 매개변수 (Key Constructor Parameters)

| 매개변수 | 유형 | 기본값 | 설명 |
|-----------|------|---------|-------------|
| `model` | `str` | `""` | OpenRouter 형식의 모델 (기본값은 빈 문자열; 런타임에 hermes 설정에서 해결됨) |
| `quiet_mode` | `bool` | `False` | CLI 출력 억제 |
| `enabled_toolsets` | `List[str]` | `None` | 특정 도구 모음 허용 (Whitelist) |
| `disabled_toolsets` | `List[str]` | `None` | 특정 도구 모음 제한 (Blacklist) |
| `save_trajectories` | `bool` | `False` | 대화를 JSONL 형식으로 저장 |
| `ephemeral_system_prompt` | `str` | `None` | 사용자 지정 시스템 프롬프트 (궤적 파일에 저장되지 않음) |
| `max_iterations` | `int` | `90` | 대화당 최대 도구 호출 반복 횟수 |
| `skip_context_files` | `bool` | `False` | AGENTS.md 파일 로드 건너뛰기 |
| `skip_memory` | `bool` | `False` | 영구 메모리 읽기/쓰기 비활성화 |
| `api_key` | `str` | `None` | API 키 (환경 변수로 대체됨) |
| `base_url` | `str` | `None` | 사용자 지정 API 엔드포인트 URL |
| `platform` | `str` | `None` | 플랫폼 힌트 (`"discord"`, `"telegram"` 등) |

---

## 중요 참고 사항 (Important Notes)

:::tip
- 작업 디렉토리의 `AGENTS.md` 파일이 시스템 프롬프트에 로드되는 것을 원하지 않으면 **`skip_context_files=True`**로 설정하세요.
- 에이전트가 영구 메모리를 읽거나 쓰는 것을 방지하려면 **`skip_memory=True`**로 설정하세요 — 상태를 저장하지 않는(stateless) API 엔드포인트에 권장됩니다.
- `platform` 매개변수(예: `"discord"`, `"telegram"`)는 플랫폼별 포맷팅 힌트를 주입하여 에이전트가 출력 스타일을 조정할 수 있도록 합니다.
:::

:::warning
- **스레드 안전성 (Thread safety)**: 스레드나 작업당 하나의 `AIAgent`를 생성하세요. 절대로 여러 동시 호출 간에 인스턴스를 공유하지 마세요.
- **리소스 정리 (Resource cleanup)**: 에이전트는 대화가 끝날 때 리소스(터미널 세션, 브라우저 인스턴스)를 자동으로 정리합니다. 장기 실행 프로세스에서 실행 중인 경우 각 대화가 정상적으로 완료되는지 확인하세요.
- **반복 횟수 제한 (Iteration limits)**: 기본값인 `max_iterations=90`은 넉넉한 편입니다. 간단한 Q&A 사용 사례의 경우 도구 호출 루프가 폭주하는 것을 방지하고 비용을 제어하기 위해 횟수를 낮추는 것을 고려하세요 (예: `max_iterations=10`).
:::

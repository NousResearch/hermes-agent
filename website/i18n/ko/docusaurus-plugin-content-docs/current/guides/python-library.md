---
sidebar_position: 5
title: "Hermes를 Python 라이브러리로 사용하기"
description: "CLI 없이 자체 Python 스크립트, 웹 앱 또는 자동화 파이프라인에 AIAgent 임베드하기"
---

# Hermes를 Python 라이브러리로 사용하기

Hermes는 단순한 CLI 도구가 아닙니다. `AIAgent`를 직접 가져와 자체 Python 스크립트, 웹 애플리케이션 또는 자동화 파이프라인에서 프로그래밍 방식으로 사용할 수 있습니다. 이 가이드에서는 그 방법을 설명합니다.

---

## 설치

Hermes를 복제하고 지원되는 editable 개발 환경을 만드세요.

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
uv sync
```

해당 체크아웃 디렉터리에서 `uv run python your_app.py`로 애플리케이션을 실행하세요. Hermes는 `requirements.txt` 설치를 위한 지원 wheel 또는 소스 배포판을 게시하지 않습니다.

:::tip
Hermes를 라이브러리로 사용할 때도 CLI에서 사용하는 것과 동일한 환경 변수가 필요합니다. 최소한 `OPENROUTER_API_KEY`를 설정하세요(직접 provider에 액세스하는 경우에는 `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`).
:::

---

## 기본 사용법

Hermes를 사용하는 가장 간단한 방법은 `chat()` 메서드입니다. 메시지를 전달하면 문자열을 반환합니다.

```python
from run_agent import AIAgent

agent = AIAgent(
    model="anthropic/claude-sonnet-4.6",
    quiet_mode=True,
)
response = agent.chat("What is the capital of France?")
print(response)
```

`chat()`은 도구 호출, 재시도 등 전체 대화 루프를 내부적으로 처리하고 최종 텍스트 응답만 반환합니다.

:::warning
Hermes를 자체 코드에 임베드할 때는 항상 `quiet_mode=True`로 설정하세요. 그렇지 않으면 에이전트가 CLI 스피너, 진행 표시기 및 기타 터미널 출력을 표시해 애플리케이션의 출력이 지저분해집니다.
:::

---

## 전체 대화 제어

대화를 더 세밀하게 제어하려면 `run_conversation()`을 직접 사용하세요. 이 메서드는 전체 응답, 메시지 기록 및 메타데이터가 담긴 사전을 반환합니다.

```python
agent = AIAgent(
    model="anthropic/claude-sonnet-4.6",
    quiet_mode=True,
)

result = agent.run_conversation(
    user_message="Search for recent Python 3.13 features",
    task_id="my-task-1",
)

print(result["final_response"])
print(f"Messages exchanged: {len(result['messages'])}")
```

반환되는 사전에는 다음이 포함됩니다.
- **`final_response`** — 에이전트의 최종 텍스트 답변
- **`messages`** — 전체 메시지 기록(system, user, assistant, tool 호출)

(`task_id`로 전달한 값은 VM 격리를 위해 에이전트 인스턴스에 저장되지만 반환되는 사전에는 포함되지 않습니다.)

호출에 사용할 임시 시스템 프롬프트를 재정의하는 사용자 지정 시스템 메시지도 전달할 수 있습니다.

```python
result = agent.run_conversation(
    user_message="Explain quicksort",
    system_message="You are a computer science tutor. Use simple analogies.",
)
```

---

## 도구 구성

`enabled_toolsets` 또는 `disabled_toolsets`를 사용해 에이전트가 액세스할 수 있는 도구 세트를 제어하세요.

```python
# Only enable web tools (browsing, search)
agent = AIAgent(
    model="anthropic/claude-sonnet-4.6",
    enabled_toolsets=["web"],
    quiet_mode=True,
)

# Enable everything except terminal access
agent = AIAgent(
    model="anthropic/claude-sonnet-4.6",
    disabled_toolsets=["terminal"],
    quiet_mode=True,
)
```

:::tip
최소한으로 제한된 에이전트(예: 리서치 봇에서 웹 검색만 허용)를 원한다면 `enabled_toolsets`를 사용하세요. 대부분의 기능은 허용하되 특정 기능(예: 공유 환경에서 터미널 액세스)을 제한해야 한다면 `disabled_toolsets`를 사용하세요.
:::

---

## 다중 턴 대화

메시지 기록을 다시 전달하면 여러 턴에 걸쳐 대화 상태를 유지할 수 있습니다.

```python
agent = AIAgent(
    model="anthropic/claude-sonnet-4.6",
    quiet_mode=True,
)

# First turn
result1 = agent.run_conversation("My name is Alice")
history = result1["messages"]

# Second turn — agent remembers the context
result2 = agent.run_conversation(
    "What's my name?",
    conversation_history=history,
)
print(result2["final_response"])  # "Your name is Alice."
```

`conversation_history` 매개변수에는 이전 결과의 `messages` 목록을 전달합니다. 에이전트는 이를 내부적으로 복사하므로 원래 목록은 절대 변경되지 않습니다.

---

## 궤적 저장

대화를 ShareGPT 형식으로 저장하도록 설정하면 학습 데이터 생성이나 디버깅에 유용합니다.

```python
agent = AIAgent(
    model="anthropic/claude-sonnet-4.6",
    save_trajectories=True,
    quiet_mode=True,
)

agent.chat("Write a Python function to sort a list")
# Saves to trajectory_samples.jsonl in ShareGPT format
```

각 대화는 하나의 JSONL 줄로 추가되므로 자동화된 실행에서 데이터셋을 쉽게 수집할 수 있습니다.

---

## 사용자 지정 시스템 프롬프트

`ephemeral_system_prompt`를 사용하면 에이전트의 동작을 안내하는 사용자 지정 시스템 프롬프트를 설정할 수 있습니다. 이 프롬프트는 궤적 파일에 저장되지 않으므로 학습 데이터를 깔끔하게 유지할 수 있습니다.

```python
agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    ephemeral_system_prompt="You are a SQL expert. Only answer database questions.",
    quiet_mode=True,
)

response = agent.chat("How do I write a JOIN query?")
print(response)
```

동일한 기반 도구를 사용해 코드 리뷰어, 문서 작성 도우미, SQL 어시스턴트 등 특화된 에이전트를 구축하는 데 적합합니다.

---

## 일괄 처리

많은 프롬프트를 병렬로 실행하려면 Hermes에 포함된 `batch_runner.py`를 사용하세요. 이 도구는 적절한 리소스 격리와 함께 여러 `AIAgent` 인스턴스를 동시에 관리합니다.

```bash
python batch_runner.py --input prompts.jsonl --output results.jsonl
```

각 프롬프트에는 자체 `task_id`와 격리된 환경이 할당됩니다. 사용자 지정 일괄 처리 로직이 필요하다면 `AIAgent`를 직접 사용해 자체 로직을 구축할 수 있습니다.

```python
import concurrent.futures
from run_agent import AIAgent

prompts = [
    "Explain recursion",
    "What is a hash table?",
    "How does garbage collection work?",
]

def process_prompt(prompt):
    # Create a fresh agent per task for thread safety
    agent = AIAgent(
        model="anthropic/claude-sonnet-4",
        quiet_mode=True,
        skip_memory=True,
    )
    return agent.chat(prompt)

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(process_prompt, prompts))

for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}\nA: {result}\n")
```

:::warning
항상 **스레드 또는 작업마다 새로운 `AIAgent` 인스턴스**를 생성하세요. 에이전트가 내부 상태(대화 기록, 도구 세션, 반복 횟수)를 유지하므로 여러 호출에서 하나의 인스턴스를 공유하는 것은 스레드 안전하지 않습니다.
:::

---

## 통합 예시

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
    f"Review this PR diff for bugs, security issues, and style problems:\n\n{diff}"
)
print(review)
```

---

## 주요 생성자 매개변수

| 매개변수 | 타입 | 기본값 | 설명 |
|-----------|------|---------|-------------|
| `model` | `str` | `""` | OpenRouter 형식의 모델(기본값은 빈 값이며, 런타임에 hermes 설정에서 확인) |
| `quiet_mode` | `bool` | `False` | CLI 출력 억제 |
| `enabled_toolsets` | `List[str]` | `None` | 특정 도구 세트를 허용 목록으로 지정 |
| `disabled_toolsets` | `List[str]` | `None` | 특정 도구 세트를 차단 목록으로 지정 |
| `save_trajectories` | `bool` | `False` | 대화를 JSONL로 저장 |
| `ephemeral_system_prompt` | `str` | `None` | 사용자 지정 시스템 프롬프트(궤적에 저장되지 않음) |
| `max_iterations` | `int` | `500` | 대화당 최대 도구 호출 반복 횟수 |
| `skip_context_files` | `bool` | `False` | AGENTS.md 파일 로드 건너뛰기 |
| `skip_memory` | `bool` | `False` | 영구 메모리 읽기/쓰기 비활성화 |
| `api_key` | `str` | `None` | API 키(환경 변수로 대체) |
| `base_url` | `str` | `None` | 사용자 지정 API 엔드포인트 URL |
| `platform` | `str` | `None` | 플랫폼 힌트(`"discord"`, `"telegram"` 등) |

---

## 중요 참고 사항

:::tip
- 작업 디렉터리의 `AGENTS.md` 파일을 시스템 프롬프트에 로드하지 않으려면 **`skip_context_files=True`**로 설정하세요.
- 에이전트가 영구 메모리를 읽거나 쓰지 않도록 하려면 **`skip_memory=True`**로 설정하세요. 상태 비저장 API 엔드포인트에 권장됩니다.
- `platform` 매개변수(예: `"discord"`, `"telegram"`)는 플랫폼별 형식 힌트를 주입해 에이전트가 출력 스타일을 조정하도록 합니다.
:::

:::warning
- **스레드 안전성**: 스레드 또는 작업마다 하나의 `AIAgent`를 생성하세요. 동시 호출에서 인스턴스를 공유하지 마세요.
- **리소스 정리**: 대화가 끝나면 에이전트가 리소스(터미널 세션, 브라우저 인스턴스)를 자동으로 정리합니다. 장기 실행 프로세스에서 사용하는 경우 각 대화가 정상적으로 완료되도록 하세요.
- **반복 제한**: 기본 `max_iterations=500`은 넉넉한 값입니다. 단순한 Q&A 사용 사례에서는 도구 호출 루프가 폭주하는 것을 방지하고 비용을 제어하기 위해 더 낮은 값(예: `max_iterations=10`)을 고려하세요.
:::

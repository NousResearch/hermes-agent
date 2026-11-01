---
title: "Guidance — 문법으로 LLM 출력을 제한하고 유효한 JSON 보장"
sidebar_label: "Guidance"
description: "문법으로 LLM 출력을 제한하고 유효한 JSON 보장"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Guidance

문법으로 LLM 출력을 제한하고 유효한 JSON을 보장합니다.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/mlops/guidance`로 설치 |
| 경로 | `optional-skills/mlops/guidance` |
| 버전 | `1.0.1` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 의존성 | `guidance`, `transformers` |
| 플랫폼 | linux, macos, windows |
| 태그 | `Prompt Engineering`, `Guidance`, `Constrained Generation`, `Structured Output`, `JSON Validation`, `Grammar`, `Microsoft Research`, `Format Enforcement`, `Multi-Step Workflows` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Guidance: 제약된 LLM 생성

## 이 스킬을 사용할 때

다음이 필요한 경우 Guidance를 사용하세요.
- 정규식 또는 문법으로 **LLM 출력 구문 제어**
- **유효한 JSON/XML/코드** 생성 보장
- 기존 프롬프팅 방식 대비 **지연 시간 단축**
- **구조화된 형식 강제**(날짜, 이메일, ID 등)
- Python 방식 제어 흐름으로 **다단계 워크플로 구축**
- 문법적 제약을 통한 **잘못된 출력 방지**

**GitHub 스타**: 18,000+ | **출처**: Microsoft Research

## 설치

```bash
# Base installation
pip install guidance

# With specific backends
pip install guidance[transformers]  # Hugging Face models
pip install guidance[llama_cpp]     # llama.cpp models
```

## 빠른 시작

### 기본 예시: 구조화된 생성

```python
from guidance import models, gen

# Load model (supports OpenAI, Transformers, llama.cpp)
lm = models.OpenAI("gpt-4")

# Generate with constraints
result = lm + "The capital of France is " + gen("capital", max_tokens=5)

print(result["capital"])  # "Paris"
```

### 로컬 모델을 사용하는 채팅 형식

> **제약 지원에는 로컬 로짓 접근이 필요합니다.** 정규식, `select()`, 그리고
> 문법 기반 제약 생성은 로컬 백엔드(`Transformers`, `LlamaCpp`)에서만 작동합니다.
> 원격 API 백엔드(`OpenAI` 및 Azure 변형)는 제약 없는 `gen()` / 채팅만 지원하며,
> 토큰 수준의 제약을 강제할 수 없습니다. guidance 0.3.x에는 `models.Anthropic` 클래스가 없습니다.

```python
from guidance import models, gen, system, user, assistant

# Local model (supports constrained generation)
lm = models.Transformers("microsoft/Phi-4-mini-instruct")

# Use context managers for chat format
with system():
    lm += "You are a helpful assistant."

with user():
    lm += "What is the capital of France?"

with assistant():
    lm += gen(max_tokens=20)
```

## 핵심 개념

### 1. 컨텍스트 관리자

Guidance는 채팅 스타일 상호작용에 Python 방식의 컨텍스트 관리자를 사용합니다.

```python
from guidance import system, user, assistant, gen

lm = models.Transformers("microsoft/Phi-4-mini-instruct")

# System message
with system():
    lm += "You are a JSON generation expert."

# User message
with user():
    lm += "Generate a person object with name and age."

# Assistant response
with assistant():
    lm += gen("response", max_tokens=100)

print(lm["response"])
```

**장점:**
- 자연스러운 채팅 흐름
- 명확한 역할 분리
- 읽고 유지보수하기 쉬움

### 2. 제약된 생성

Guidance는 정규식 또는 문법을 사용하여 출력이 지정한 패턴과 일치하도록 보장합니다.

#### 정규식 제약

```python
from guidance import models, gen

lm = models.Transformers("microsoft/Phi-4-mini-instruct")

# Constrain to valid email format
lm += "Email: " + gen("email", regex=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# Constrain to date format (YYYY-MM-DD)
lm += "Date: " + gen("date", regex=r"\d{4}-\d{2}-\d{2}")

# Constrain to phone number
lm += "Phone: " + gen("phone", regex=r"\d{3}-\d{3}-\d{4}")

print(lm["email"])  # Guaranteed valid email
print(lm["date"])   # Guaranteed YYYY-MM-DD format
```

**작동 방식:**
- 정규식을 토큰 수준의 문법으로 변환
- 생성 중 잘못된 토큰을 필터링
- 모델은 일치하는 출력만 생성 가능

#### 선택 제약

```python
from guidance import models, gen, select

lm = models.Transformers("microsoft/Phi-4-mini-instruct")

# Constrain to specific choices
lm += "Sentiment: " + select(["positive", "negative", "neutral"], name="sentiment")

# Multiple-choice selection
lm += "Best answer: " + select(
    ["A) Paris", "B) London", "C) Berlin", "D) Madrid"],
    name="answer"
)

print(lm["sentiment"])  # One of: positive, negative, neutral
print(lm["answer"])     # One of: A, B, C, or D
```

### 3. 토큰 힐링

Guidance는 프롬프트와 생성 결과 사이의 토큰 경계를 자동으로 "힐링"합니다.

**문제:** 토큰화는 부자연스러운 경계를 만듭니다.

```python
# Without token healing
prompt = "The capital of France is "
# Last token: " is "
# First generated token might be " Par" (with leading space)
# Result: "The capital of France is  Paris" (double space!)
```

**해결책:** Guidance는 토큰 하나를 되돌린 뒤 다시 생성합니다.

```python
from guidance import models, gen

lm = models.Transformers("microsoft/Phi-4-mini-instruct")

# Token healing enabled by default
lm += "The capital of France is " + gen("capital", max_tokens=5)
# Result: "The capital of France is Paris" (correct spacing)
```

**장점:**
- 자연스러운 텍스트 경계
- 어색한 공백 문제 없음
- 더 나은 모델 성능(자연스러운 토큰 시퀀스를 확인)

### 4. 문법 기반 생성

문법 함수를 조합하여 복잡한 구조를 정의합니다. 템플릿 문자열
`grammar=` 형식은 현재 guidance에 포함되어 있지 않습니다. 조합 가능한 함수로 문법을 구축하거나
JSON에는 `guidance.json()`을 사용하세요.

```python
from guidance import models, gen
from guidance import json as gen_json
from pydantic import BaseModel, Field

lm = models.Transformers("microsoft/Phi-4-mini-instruct")

# JSON via a Pydantic schema (guidance.json compiles the schema to a grammar)
class Person(BaseModel):
    name: str = Field(pattern=r"[A-Za-z ]+")
    age: int
    email: str = Field(pattern=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

lm += gen_json(name="person", schema=Person)

print(lm["person"])  # Guaranteed valid JSON matching the schema

# Or compose grammar functions directly:
grammar = "name=" + gen("name", regex=r"[A-Za-z ]+") + " age=" + gen("age", regex=r"[0-9]+")
lm += grammar
```

**사용 사례:**
- 복잡한 구조화된 출력
- 중첩 데이터 구조
- 프로그래밍 언어 구문
- 도메인 특화 언어

### 5. Guidance 함수

`@guidance` 데코레이터로 재사용 가능한 생성 패턴을 만드세요.

```python
from guidance import guidance, gen, models

@guidance
def generate_person(lm):
    """Generate a person with name and age."""
    lm += "Name: " + gen("name", max_tokens=20, stop="\n")
    lm += "\nAge: " + gen("age", regex=r"[0-9]+", max_tokens=3)
    return lm

# Use the function
lm = models.Transformers("microsoft/Phi-4-mini-instruct")
lm = generate_person(lm)

print(lm["name"])
print(lm["age"])
```

**상태 유지 함수:**

```python
@guidance(stateless=False)
def react_agent(lm, question, tools, max_rounds=5):
    """ReAct agent with tool use."""
    lm += f"Question: {question}\n\n"

    for i in range(max_rounds):
        # Thought
        lm += f"Thought {i+1}: " + gen("thought", stop="\n")

        # Action
        lm += "\nAction: " + select(list(tools.keys()), name="action")

        # Execute tool
        tool_result = tools[lm["action"]]()
        lm += f"\nObservation: {tool_result}\n\n"

        # Check if done
        lm += "Done? " + select(["Yes", "No"], name="done")
        if lm["done"] == "Yes":
            break

    # Final answer
    lm += "\nFinal Answer: " + gen("answer", max_tokens=100)
    return lm
```

## 백엔드 구성

### OpenAI(원격 — 제약 없음)

> 원격 API 백엔드는 제약된 생성(정규식/select/문법)을 수행할 수 없습니다.
> 일반 채팅/`gen()`에만 사용하세요. 제약이 필요한 경우 로컬 백엔드를 사용하세요.

```python
from guidance import models

lm = models.OpenAI(
    model="gpt-4o-mini",
    api_key="your-api-key"  # Or set OPENAI_API_KEY env var
)
```

### 로컬 모델(Transformers)

```python
from guidance.models import Transformers

lm = Transformers(
    "microsoft/Phi-4-mini-instruct",
    device="cuda"  # Or "cpu"
)
```

### 로컬 모델(llama.cpp)

```python
from guidance.models import LlamaCpp

lm = LlamaCpp(
    model_path="/path/to/model.gguf",
    n_ctx=4096,
    n_gpu_layers=35
)
```

## 일반적인 패턴

### 패턴 1: JSON 생성

```python
from guidance import models, gen, system, user, assistant

lm = models.Transformers("microsoft/Phi-4-mini-instruct")

with system():
    lm += "You generate valid JSON."

with user():
    lm += "Generate a user profile with name, age, and email."

with assistant():
    lm += """{
    "name": """ + gen("name", regex=r'"[A-Za-z ]+"', max_tokens=30) + """,
    "age": """ + gen("age", regex=r"[0-9]+", max_tokens=3) + """,
    "email": """ + gen("email", regex=r'"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"', max_tokens=50) + """
}"""

print(lm)  # Valid JSON guaranteed
```

### 패턴 2: 분류

```python
from guidance import models, gen, select

lm = models.Transformers("microsoft/Phi-4-mini-instruct")

text = "This product is amazing! I love it."

lm += f"Text: {text}\n"
lm += "Sentiment: " + select(["positive", "negative", "neutral"], name="sentiment")
lm += "\nConfidence: " + gen("confidence", regex=r"[0-9]+", max_tokens=3) + "%"

print(f"Sentiment: {lm['sentiment']}")
print(f"Confidence: {lm['confidence']}%")
```

### 패턴 3: 다단계 추론

```python
from guidance import models, gen, guidance

@guidance
def chain_of_thought(lm, question):
    """Generate answer with step-by-step reasoning."""
    lm += f"Question: {question}\n\n"

    # Generate multiple reasoning steps
    for i in range(3):
        lm += f"Step {i+1}: " + gen(f"step_{i+1}", stop="\n", max_tokens=100) + "\n"

    # Final answer
    lm += "\nTherefore, the answer is: " + gen("answer", max_tokens=50)

    return lm

lm = models.Transformers("microsoft/Phi-4-mini-instruct")
lm = chain_of_thought(lm, "What is 15% of 200?")

print(lm["answer"])
```

### 패턴 4: ReAct 에이전트

```python
from guidance import models, gen, select, guidance

@guidance(stateless=False)
def react_agent(lm, question):
    """ReAct agent with tool use."""
    tools = {
        "calculator": lambda expr: eval(expr),
        "search": lambda query: f"Search results for: {query}",
    }

    lm += f"Question: {question}\n\n"

    for round in range(5):
        # Thought
        lm += f"Thought: " + gen("thought", stop="\n") + "\n"

        # Action selection
        lm += "Action: " + select(["calculator", "search", "answer"], name="action")

        if lm["action"] == "answer":
            lm += "\nFinal Answer: " + gen("answer", max_tokens=100)
            break

        # Action input
        lm += "\nAction Input: " + gen("action_input", stop="\n") + "\n"

        # Execute tool
        if lm["action"] in tools:
            result = tools[lm["action"]](lm["action_input"])
            lm += f"Observation: {result}\n\n"

    return lm

lm = models.Transformers("microsoft/Phi-4-mini-instruct")
lm = react_agent(lm, "What is 25 * 4 + 10?")
print(lm["answer"])
```

### 패턴 5: 데이터 추출

```python
from guidance import models, gen, guidance

@guidance
def extract_entities(lm, text):
    """Extract structured entities from text."""
    lm += f"Text: {text}\n\n"

    # Extract person
    lm += "Person: " + gen("person", stop="\n", max_tokens=30) + "\n"

    # Extract organization
    lm += "Organization: " + gen("organization", stop="\n", max_tokens=30) + "\n"

    # Extract date
    lm += "Date: " + gen("date", regex=r"\d{4}-\d{2}-\d{2}", max_tokens=10) + "\n"

    # Extract location
    lm += "Location: " + gen("location", stop="\n", max_tokens=30) + "\n"

    return lm

text = "Tim Cook announced at Apple Park on 2024-09-15 in Cupertino."

lm = models.Transformers("microsoft/Phi-4-mini-instruct")
lm = extract_entities(lm, text)

print(f"Person: {lm['person']}")
print(f"Organization: {lm['organization']}")
print(f"Date: {lm['date']}")
print(f"Location: {lm['location']}")
```

## 모범 사례

### 1. 형식 검증에 정규식 사용

```python
# ✅ Good: Regex ensures valid format
lm += "Email: " + gen("email", regex=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# ❌ Bad: Free generation may produce invalid emails
lm += "Email: " + gen("email", max_tokens=50)
```

### 2. 고정 카테고리에는 select() 사용

```python
# ✅ Good: Guaranteed valid category
lm += "Status: " + select(["pending", "approved", "rejected"], name="status")

# ❌ Bad: May generate typos or invalid values
lm += "Status: " + gen("status", max_tokens=20)
```

### 3. 토큰 힐링 활용

```python
# Token healing is enabled by default
# No special action needed - just concatenate naturally
lm += "The capital is " + gen("capital")  # Automatic healing
```

### 4. 중지 시퀀스 사용

```python
# ✅ Good: Stop at newline for single-line outputs
lm += "Name: " + gen("name", stop="\n")

# ❌ Bad: May generate multiple lines
lm += "Name: " + gen("name", max_tokens=50)
```

### 5. 재사용 가능한 함수 만들기

```python
# ✅ Good: Reusable pattern
@guidance
def generate_person(lm):
    lm += "Name: " + gen("name", stop="\n")
    lm += "\nAge: " + gen("age", regex=r"[0-9]+")
    return lm

# Use multiple times
lm = generate_person(lm)
lm += "\n\n"
lm = generate_person(lm)
```

### 6. 제약의 균형 맞추기

```python
# ✅ Good: Reasonable constraints
lm += gen("name", regex=r"[A-Za-z ]+", max_tokens=30)

# ❌ Too strict: May fail or be very slow
lm += gen("name", regex=r"^(John|Jane)$", max_tokens=10)
```

## 대안과의 비교

| 기능 | Guidance | Instructor | Outlines | LMQL |
|---------|----------|------------|----------|------|
| 정규식 제약 | ✅ 예 | ❌ 아니요 | ✅ 예 | ✅ 예 |
| 문법 지원 | ✅ CFG | ❌ 아니요 | ✅ CFG | ✅ CFG |
| Pydantic 검증 | ❌ 아니요 | ✅ 예 | ✅ 예 | ❌ 아니요 |
| 토큰 힐링 | ✅ 예 | ❌ 아니요 | ✅ 예 | ❌ 아니요 |
| 로컬 모델 | ✅ 예 | ⚠️ 제한적 | ✅ 예 | ✅ 예 |
| API 모델 | ✅ 예 | ✅ 예 | ⚠️ 제한적 | ✅ 예 |
| Python 방식 구문 | ✅ 예 | ✅ 예 | ✅ 예 | ❌ SQL과 유사 |
| 학습 난이도 | 낮음 | 낮음 | 보통 | 높음 |

**Guidance를 선택할 때:**
- 정규식/문법 제약이 필요함
- 토큰 힐링이 필요함
- 제어 흐름이 있는 복잡한 워크플로를 구축함
- 로컬 모델(Transformers, llama.cpp)을 사용함
- Python 방식 구문을 선호함

**대안을 선택할 때:**
- Instructor: 자동 재시도와 함께 Pydantic 검증이 필요함
- Outlines: JSON 스키마 검증이 필요함
- LMQL: 선언형 쿼리 구문을 선호함

## 성능 특성

**지연 시간 감소:**
- 제약된 출력에서 기존 프롬프팅보다 30~50% 더 빠름
- 토큰 힐링으로 불필요한 재생성 감소
- 문법 제약으로 잘못된 토큰 생성 방지

**메모리 사용량:**
- 제약 없는 생성 대비 오버헤드 최소화
- 첫 사용 후 문법 컴파일 결과 캐시
- 추론 시 효율적인 토큰 필터링

**토큰 효율성:**
- 잘못된 출력에 낭비되는 토큰 방지
- 재시도 루프 불필요
- 유효한 출력으로 직접 연결

## 리소스

- **문서**: https://guidance.readthedocs.io
- **GitHub**: https://github.com/guidance-ai/guidance (18k+ stars)
- **노트북**: https://github.com/guidance-ai/guidance/tree/main/notebooks
- **Discord**: 커뮤니티 지원 제공

## 함께 보기

- `references/constraints.md` - 포괄적인 정규식 및 문법 패턴
- `references/backends.md` - 백엔드별 구성
- `references/examples.md` - 프로덕션 준비 예시

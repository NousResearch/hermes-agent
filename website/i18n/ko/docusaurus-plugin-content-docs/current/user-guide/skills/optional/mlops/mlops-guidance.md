---
title: "Guidance"
sidebar_label: "Guidance"
description: "정규식과 문법을 사용하여 LLM 출력을 제어하고, 유효한 JSON/XML/코드 생성을 보장하며, 구조화된 형식을 강제하고, Guidance를 통해 다단계 워크플로우를 구축합니다."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Guidance

정규식(regex)과 문법(grammars)을 사용하여 LLM 출력을 제어하고, 유효한 JSON/XML/코드 생성을 보장하며, 구조화된 형식을 강제하고, Microsoft Research의 제약 조건 생성 프레임워크인 Guidance를 통해 다단계 워크플로우를 구축합니다.

## Skill metadata

| | |
|---|---|
| Source | Optional — `hermes skills install official/mlops/guidance`로 설치 |
| Path | `optional-skills/mlops/guidance` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `guidance`, `transformers` |
| Platforms | linux, macos, windows |
| Tags | `Prompt Engineering`, `Guidance`, `Constrained Generation`, `Structured Output`, `JSON Validation`, `Grammar`, `Microsoft Research`, `Format Enforcement`, `Multi-Step Workflows` |

## Reference: full SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Guidance: 제약 기반 LLM 생성 (Constrained LLM Generation)

## 이 스킬을 사용하는 경우

다음과 같은 요구 사항이 있을 때 Guidance를 사용하세요:
- 정규식 또는 문법으로 **LLM 출력 구문을 제어**해야 할 때
- **유효한 JSON/XML/코드** 생성을 보장해야 할 때
- 기존 프롬프팅 방식에 비해 **지연 시간(latency)을 줄여야 할 때**
- (날짜, 이메일, ID 등의) **구조화된 형식을 강제**해야 할 때
- Pythonic한 제어 흐름으로 **다단계 워크플로우를 구축**할 때
- 문법적 제약 조건을 통해 **잘못된 출력을 방지**할 때

**GitHub 별(Stars)**: 18,000+ | **개발**: Microsoft Research

## 설치

```bash
# 기본 설치
pip install guidance

# 특정 백엔드 포함 설치
pip install guidance[transformers]  # Hugging Face 모델
pip install guidance[llama_cpp]     # llama.cpp 모델
```

## 빠른 시작

### 기본 예제: 구조화된 생성

```python
from guidance import models, gen

# 모델 불러오기 (OpenAI, Transformers, llama.cpp 지원)
lm = models.OpenAI("gpt-4")

# 제약 조건을 사용한 생성
result = lm + "프랑스의 수도는 " + gen("capital", max_tokens=5)

print(result["capital"])  # "파리"
```

### Anthropic Claude 사용

```python
from guidance import models, gen, system, user, assistant

# Claude 설정
lm = models.Anthropic("claude-sonnet-4-5-20250929")

# 채팅 형식을 위한 컨텍스트 매니저 사용
with system():
    lm += "당신은 유용한 어시스턴트입니다."

with user():
    lm += "프랑스의 수도는 어디입니까?"

with assistant():
    lm += gen(max_tokens=20)
```

## 핵심 개념

### 1. 컨텍스트 매니저 (Context Managers)

Guidance는 채팅 스타일의 상호 작용을 위해 Pythonic한 컨텍스트 매니저를 사용합니다.

```python
from guidance import system, user, assistant, gen

lm = models.Anthropic("claude-sonnet-4-5-20250929")

# 시스템 메시지
with system():
    lm += "당신은 JSON 생성 전문가입니다."

# 사용자 메시지
with user():
    lm += "이름과 나이가 포함된 사람 객체를 생성하세요."

# 어시스턴트 응답
with assistant():
    lm += gen("response", max_tokens=100)

print(lm["response"])
```

**이점:**
- 자연스러운 채팅 흐름
- 명확한 역할 분리
- 읽기 쉽고 유지 관리하기 쉬움

### 2. 제약 기반 생성 (Constrained Generation)

Guidance는 정규식이나 문법을 사용하여 출력이 지정된 패턴과 일치하도록 보장합니다.

#### 정규식 제약 조건 (Regex Constraints)

```python
from guidance import models, gen

lm = models.Anthropic("claude-sonnet-4-5-20250929")

# 유효한 이메일 형식으로 제한
lm += "이메일: " + gen("email", regex=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# 날짜 형식(YYYY-MM-DD)으로 제한
lm += "날짜: " + gen("date", regex=r"\d{4}-\d{2}-\d{2}")

# 전화번호 형식으로 제한
lm += "전화번호: " + gen("phone", regex=r"\d{3}-\d{3}-\d{4}")

print(lm["email"])  # 유효한 이메일 보장
print(lm["date"])   # YYYY-MM-DD 형식 보장
```

**작동 방식:**
- 정규식이 토큰 수준의 문법으로 변환됨
- 생성 중에 잘못된 토큰이 필터링됨
- 모델은 일치하는 출력만 생성할 수 있음

#### 선택 제약 조건 (Selection Constraints)

```python
from guidance import models, gen, select

lm = models.Anthropic("claude-sonnet-4-5-20250929")

# 특정 선택지로 제한
lm += "감정: " + select(["긍정적", "부정적", "중립적"], name="sentiment")

# 다중 선택 (Multiple-choice)
lm += "최고의 답변: " + select(
    ["A) 파리", "B) 런던", "C) 베를린", "D) 마드리드"],
    name="answer"
)

print(lm["sentiment"])  # 긍정적, 부정적, 중립적 중 하나
print(lm["answer"])     # A, B, C, D 중 하나
```

### 3. 토큰 힐링 (Token Healing)

Guidance는 프롬프트와 생성 텍스트 사이의 토큰 경계를 자동으로 "치료(heal)"합니다.

**문제:** 토큰화 과정에서 부자연스러운 경계가 생성될 수 있습니다.

```python
# 토큰 힐링을 사용하지 않는 경우
prompt = "The capital of France is "
# 마지막 토큰: " is "
# 처음 생성되는 토큰이 " Par"일 수 있음 (앞에 공백 포함)
# 결과: "The capital of France is  Paris" (공백이 두 개!)
```

**해결:** Guidance가 한 토큰 뒤로 물러나서 다시 생성합니다.

```python
from guidance import models, gen

lm = models.Anthropic("claude-sonnet-4-5-20250929")

# 토큰 힐링이 기본적으로 활성화되어 있습니다
lm += "The capital of France is " + gen("capital", max_tokens=5)
# 결과: "The capital of France is Paris" (올바른 공백)
```

**이점:**
- 자연스러운 텍스트 경계
- 어색한 공백 문제 해결
- 모델 성능 향상 (자연스러운 토큰 시퀀스 제공)

### 4. 문법 기반 생성 (Grammar-Based Generation)

컨텍스트 없는 문법(context-free grammars)을 사용하여 복잡한 구조를 정의합니다.

```python
from guidance import models, gen

lm = models.Anthropic("claude-sonnet-4-5-20250929")

# JSON 문법 (단순화됨)
json_grammar = """
{
    "name": <gen name regex="[A-Za-z ]+" max_tokens=20>,
    "age": <gen age regex="[0-9]+" max_tokens=3>,
    "email": <gen email regex="[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}" max_tokens=50>
}
"""

# 유효한 JSON 생성
lm += gen("person", grammar=json_grammar)

print(lm["person"])  # 유효한 JSON 구조 보장
```

**사용 사례:**
- 복잡하게 구조화된 출력
- 중첩된 데이터 구조
- 프로그래밍 언어 구문
- 도메인 특화 언어(DSL)

### 5. Guidance 함수

`@guidance` 데코레이터를 사용하여 재사용 가능한 생성 패턴을 만듭니다.

```python
from guidance import guidance, gen, models

@guidance
def generate_person(lm):
    """이름과 나이가 포함된 사람 생성"""
    lm += "이름: " + gen("name", max_tokens=20, stop="\n")
    lm += "\n나이: " + gen("age", regex=r"[0-9]+", max_tokens=3)
    return lm

# 함수 사용
lm = models.Anthropic("claude-sonnet-4-5-20250929")
lm = generate_person(lm)

print(lm["name"])
print(lm["age"])
```

**상태 유지 함수 (Stateful Functions):**

```python
@guidance(stateless=False)
def react_agent(lm, question, tools, max_rounds=5):
    """도구를 사용하는 ReAct 에이전트"""
    lm += f"질문: {question}\n\n"

    for i in range(max_rounds):
        # 생각 (Thought)
        lm += f"생각 {i+1}: " + gen("thought", stop="\n")

        # 행동 (Action)
        lm += "\n행동: " + select(list(tools.keys()), name="action")

        # 도구 실행
        tool_result = tools[lm["action"]]()
        lm += f"\n관찰: {tool_result}\n\n"

        # 완료 확인
        lm += "완료되었습니까? " + select(["예", "아니오"], name="done")
        if lm["done"] == "예":
            break

    # 최종 답변
    lm += "\n최종 답변: " + gen("answer", max_tokens=100)
    return lm
```

## 백엔드 구성

### Anthropic Claude

```python
from guidance import models

lm = models.Anthropic(
    model="claude-sonnet-4-5-20250929",
    api_key="your-api-key"  # 또는 ANTHROPIC_API_KEY 환경 변수 설정
)
```

### OpenAI

```python
lm = models.OpenAI(
    model="gpt-4o-mini",
    api_key="your-api-key"  # 또는 OPENAI_API_KEY 환경 변수 설정
)
```

### 로컬 모델 (Transformers)

```python
from guidance.models import Transformers

lm = Transformers(
    "microsoft/Phi-4-mini-instruct",
    device="cuda"  # 또는 "cpu"
)
```

### 로컬 모델 (llama.cpp)

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

lm = models.Anthropic("claude-sonnet-4-5-20250929")

with system():
    lm += "당신은 유효한 JSON을 생성합니다."

with user():
    lm += "이름, 나이, 이메일이 포함된 사용자 프로필을 생성하세요."

with assistant():
    lm += """{
    "name": """ + gen("name", regex=r'"[A-Za-z ]+"', max_tokens=30) + """,
    "age": """ + gen("age", regex=r"[0-9]+", max_tokens=3) + """,
    "email": """ + gen("email", regex=r'"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"', max_tokens=50) + """
}"""

print(lm)  # 유효한 JSON 보장
```

### 패턴 2: 분류 (Classification)

```python
from guidance import models, gen, select

lm = models.Anthropic("claude-sonnet-4-5-20250929")

text = "이 제품은 정말 놀랍습니다! 아주 마음에 듭니다."

lm += f"텍스트: {text}\n"
lm += "감정: " + select(["긍정적", "부정적", "중립적"], name="sentiment")
lm += "\n신뢰도: " + gen("confidence", regex=r"[0-9]+", max_tokens=3) + "%"

print(f"감정: {lm['sentiment']}")
print(f"신뢰도: {lm['confidence']}%")
```

### 패턴 3: 다단계 추론 (Multi-Step Reasoning)

```python
from guidance import models, gen, guidance

@guidance
def chain_of_thought(lm, question):
    """단계별 추론을 통한 답변 생성"""
    lm += f"질문: {question}\n\n"

    # 여러 추론 단계 생성
    for i in range(3):
        lm += f"단계 {i+1}: " + gen(f"step_{i+1}", stop="\n", max_tokens=100) + "\n"

    # 최종 답변
    lm += "\n따라서 정답은: " + gen("answer", max_tokens=50)

    return lm

lm = models.Anthropic("claude-sonnet-4-5-20250929")
lm = chain_of_thought(lm, "200의 15%는 얼마입니까?")

print(lm["answer"])
```

### 패턴 4: ReAct 에이전트

```python
from guidance import models, gen, select, guidance

@guidance(stateless=False)
def react_agent(lm, question):
    """도구를 사용하는 ReAct 에이전트"""
    tools = {
        "calculator": lambda expr: eval(expr),
        "search": lambda query: f"다음 검색 결과: {query}",
    }

    lm += f"질문: {question}\n\n"

    for round in range(5):
        # 생각
        lm += f"생각: " + gen("thought", stop="\n") + "\n"

        # 행동 선택
        lm += "행동: " + select(["calculator", "search", "answer"], name="action")

        if lm["action"] == "answer":
            lm += "\n최종 답변: " + gen("answer", max_tokens=100)
            break

        # 행동 입력
        lm += "\n행동 입력: " + gen("action_input", stop="\n") + "\n"

        # 도구 실행
        if lm["action"] in tools:
            result = tools[lm["action"]](lm["action_input"])
            lm += f"관찰: {result}\n\n"

    return lm

lm = models.Anthropic("claude-sonnet-4-5-20250929")
lm = react_agent(lm, "25 * 4 + 10은 무엇입니까?")
print(lm["answer"])
```

### 패턴 5: 데이터 추출

```python
from guidance import models, gen, guidance

@guidance
def extract_entities(lm, text):
    """텍스트에서 구조화된 엔티티 추출"""
    lm += f"텍스트: {text}\n\n"

    # 인물 추출
    lm += "인물: " + gen("person", stop="\n", max_tokens=30) + "\n"

    # 조직 추출
    lm += "조직: " + gen("organization", stop="\n", max_tokens=30) + "\n"

    # 날짜 추출
    lm += "날짜: " + gen("date", regex=r"\d{4}-\d{2}-\d{2}", max_tokens=10) + "\n"

    # 위치 추출
    lm += "위치: " + gen("location", stop="\n", max_tokens=30) + "\n"

    return lm

text = "Tim Cook은 2024-09-15에 Cupertino의 Apple Park에서 발표했습니다."

lm = models.Anthropic("claude-sonnet-4-5-20250929")
lm = extract_entities(lm, text)

print(f"인물: {lm['person']}")
print(f"조직: {lm['organization']}")
print(f"날짜: {lm['date']}")
print(f"위치: {lm['location']}")
```

## 모범 사례

### 1. 포맷 유효성 검사에 정규식 사용

```python
# ✅ 올바른 예: 정규식으로 유효한 형식을 보장합니다
lm += "이메일: " + gen("email", regex=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# ❌ 잘못된 예: 자유 생성은 유효하지 않은 이메일을 생성할 수 있습니다
lm += "이메일: " + gen("email", max_tokens=50)
```

### 2. 고정된 범주에는 select() 사용

```python
# ✅ 올바른 예: 유효한 범주가 보장됩니다
lm += "상태: " + select(["대기 중", "승인됨", "거절됨"], name="status")

# ❌ 잘못된 예: 오타가 있거나 유효하지 않은 값이 생성될 수 있습니다
lm += "상태: " + gen("status", max_tokens=20)
```

### 3. 토큰 힐링 적극 활용

```python
# 토큰 힐링은 기본적으로 활성화되어 있습니다
# 특별한 조치 없이 자연스럽게 연결하기만 하면 됩니다
lm += "The capital is " + gen("capital")  # 자동 힐링
```

### 4. stop 시퀀스 사용

```python
# ✅ 올바른 예: 단일 줄 출력을 위해 개행 문자에서 중지합니다
lm += "이름: " + gen("name", stop="\n")

# ❌ 잘못된 예: 여러 줄을 생성할 수 있습니다
lm += "이름: " + gen("name", max_tokens=50)
```

### 5. 재사용 가능한 함수 생성

```python
# ✅ 올바른 예: 재사용 가능한 패턴
@guidance
def generate_person(lm):
    lm += "이름: " + gen("name", stop="\n")
    lm += "\n나이: " + gen("age", regex=r"[0-9]+")
    return lm

# 여러 번 사용
lm = generate_person(lm)
lm += "\n\n"
lm = generate_person(lm)
```

### 6. 제약 조건의 균형 유지

```python
# ✅ 올바른 예: 합리적인 제약 조건
lm += gen("name", regex=r"[A-Za-z ]+", max_tokens=30)

# ❌ 너무 엄격한 예: 실패하거나 매우 느려질 수 있습니다
lm += gen("name", regex=r"^(John|Jane)$", max_tokens=10)
```

## 대안 비교

| 기능 | Guidance | Instructor | Outlines | LMQL |
|---------|----------|------------|----------|------|
| 정규식 제약 조건 | ✅ 예 | ❌ 아니오 | ✅ 예 | ✅ 예 |
| 문법(Grammar) 지원 | ✅ CFG | ❌ 아니오 | ✅ CFG | ✅ CFG |
| Pydantic 유효성 검사 | ❌ 아니오 | ✅ 예 | ✅ 예 | ❌ 아니오 |
| 토큰 힐링(Token Healing) | ✅ 예 | ❌ 아니오 | ✅ 예 | ❌ 아니오 |
| 로컬 모델 지원 | ✅ 예 | ⚠️ 제한적 | ✅ 예 | ✅ 예 |
| API 모델 지원 | ✅ 예 | ✅ 예 | ⚠️ 제한적 | ✅ 예 |
| Pythonic 구문 | ✅ 예 | ✅ 예 | ✅ 예 | ❌ SQL 스타일 |
| 학습 곡선 | 낮음 | 낮음 | 중간 | 높음 |

**Guidance를 사용하는 경우:**
- 정규식/문법 제약 조건이 필요할 때
- 토큰 힐링을 원할 때
- 제어 흐름이 포함된 복잡한 워크플로우를 구축할 때
- 로컬 모델(Transformers, llama.cpp)을 사용할 때
- Pythonic한 구문을 선호할 때

**대안을 선택해야 하는 경우:**
- Instructor: 자동 재시도 기능을 포함한 Pydantic 유효성 검사가 필요할 때
- Outlines: JSON Schema 유효성 검사가 필요할 때
- LMQL: 선언적인 쿼리 구문을 선호할 때

## 성능 특성

**지연 시간 (Latency) 감소:**
- 제약된 출력에 대해 기존 프롬프팅보다 30-50% 빠릅니다.
- 토큰 힐링이 불필요한 재생성을 줄입니다.
- 문법 제약 조건이 유효하지 않은 토큰 생성을 차단합니다.

**메모리 사용량:**
- 제약 조건이 없는 생성과 비교할 때 오버헤드가 최소화됩니다.
- 문법 컴파일은 첫 번째 사용 후 캐시됩니다.
- 추론 시점에서 토큰 필터링이 효율적으로 이루어집니다.

**토큰 효율성:**
- 유효하지 않은 출력에 소모되는 낭비 토큰을 방지합니다.
- 재시도 루프가 필요하지 않습니다.
- 유효한 출력으로 곧바로 이어집니다.

## 리소스

- **문서**: https://guidance.readthedocs.io
- **GitHub**: https://github.com/guidance-ai/guidance (18k+ stars)
- **Notebooks**: https://github.com/guidance-ai/guidance/tree/main/notebooks
- **Discord**: 커뮤니티 지원 가능

## 참고 자료 (See Also)

- `references/constraints.md` - 포괄적인 정규식 및 문법 패턴
- `references/backends.md` - 백엔드별 환경 설정
- `references/examples.md` - 프로덕션 지원 가능 예제

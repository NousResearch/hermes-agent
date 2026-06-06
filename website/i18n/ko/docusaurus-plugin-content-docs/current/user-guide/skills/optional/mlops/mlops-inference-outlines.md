---
title: "Outlines — Outlines: 구조화된 JSON/정규표현식/Pydantic LLM 생성"
sidebar_label: "Outlines"
description: "Outlines: 구조화된 JSON/정규표현식/Pydantic LLM 생성"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py를 통해 스킬의 SKILL.md에서 자동으로 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 수정하세요. */}

# Outlines

Outlines: 구조화된 JSON/정규표현식/Pydantic LLM 생성.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택사항 — `hermes skills install official/mlops/outlines`로 설치 |
| 경로 | `optional-skills/mlops/inference/outlines` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 의존성 | `outlines`, `transformers`, `vllm`, `pydantic` |
| 플랫폼 | linux, macos, windows |
| 태그 | `Prompt Engineering`, `Outlines`, `Structured Generation`, `JSON Schema`, `Pydantic`, `Local Models`, `Grammar-Based Generation`, `vLLM`, `Transformers`, `Type Safety` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이는 스킬이 활성화되었을 때 에이전트가 지시사항으로 보는 내용입니다.
:::

# Outlines: 구조화된 텍스트 생성

## 이 스킬을 사용하는 경우

다음과 같은 작업이 필요할 때 Outlines를 사용하세요:
- 생성 중 **유효한 JSON/XML/코드 구조 보장**
- 타입 안전성 출력을 위한 **Pydantic 모델 사용**
- **로컬 모델 지원** (Transformers, llama.cpp, vLLM)
- 제로 오버헤드의 구조화된 생성으로 **추론 속도 극대화**
- **JSON 스키마에 맞춰** 자동으로 생성
- 문법 수준에서 **토큰 샘플링 제어**

**GitHub Stars**: 8,000+ | **출처**: dottxt.ai (이전의 .txt)

## 설치

```bash
# 기본 설치
pip install outlines

# 특정 백엔드와 함께 설치
pip install outlines transformers  # Hugging Face 모델
pip install outlines llama-cpp-python  # llama.cpp
pip install outlines vllm  # 높은 처리량을 위한 vLLM
```

## 빠른 시작

### 기본 예제: 분류

```python
import outlines
from typing import Literal

# 모델 로드
model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")

# 타입 제약과 함께 생성
prompt = "'This product is amazing!'의 감정: "
generator = outlines.generate.choice(model, ["positive", "negative", "neutral"])
sentiment = generator(prompt)

print(sentiment)  # "positive" (선택지 중 하나가 될 것을 보장함)
```

### Pydantic 모델과 함께 사용

```python
from pydantic import BaseModel
import outlines

class User(BaseModel):
    name: str
    age: int
    email: str

model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")

# 구조화된 출력 생성
prompt = "사용자 추출: John Doe, 30 years old, john@example.com"
generator = outlines.generate.json(model, User)
user = generator(prompt)

print(user.name)   # "John Doe"
print(user.age)    # 30
print(user.email)  # "john@example.com"
```

## 핵심 개념

### 1. 제한된 토큰 샘플링

Outlines는 유한 상태 기계(Finite State Machines, FSM)를 사용하여 로짓 수준에서 토큰 생성을 제한합니다.

**작동 방식:**
1. 스키마(JSON/Pydantic/정규표현식)를 문맥 자유 문법(Context-Free Grammar, CFG)으로 변환
2. CFG를 유한 상태 기계(FSM)로 변환
3. 생성 중 각 단계에서 유효하지 않은 토큰을 필터링
4. 유효한 토큰이 하나만 존재할 때는 빨리 감기(Fast-forward)

**이점:**
- **제로 오버헤드**: 필터링이 토큰 수준에서 일어남
- **속도 향상**: 결정론적 경로를 통한 빨리 감기
- **유효성 보장**: 유효하지 않은 출력 불가

```python
import outlines

# Pydantic 모델 -> JSON 스키마 -> CFG -> FSM
class Person(BaseModel):
    name: str
    age: int

model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")

# 내부 작동:
# 1. Person -> JSON 스키마
# 2. JSON 스키마 -> CFG
# 3. CFG -> FSM
# 4. FSM이 생성 중에 토큰을 필터링

generator = outlines.generate.json(model, Person)
result = generator("Generate person: Alice, 25")
```

### 2. 구조화된 제너레이터

Outlines는 여러 출력 타입에 대한 특화된 제너레이터를 제공합니다.

#### 선택 제너레이터 (Choice Generator)

```python
# 객관식 선택
generator = outlines.generate.choice(
    model,
    ["positive", "negative", "neutral"]
)

sentiment = generator("Review: This is great!")
# 결과: 세 가지 선택지 중 하나
```

#### JSON 제너레이터 (JSON Generator)

```python
from pydantic import BaseModel

class Product(BaseModel):
    name: str
    price: float
    in_stock: bool

# 스키마와 일치하는 유효한 JSON 생성
generator = outlines.generate.json(model, Product)
product = generator("Extract: iPhone 15, $999, available")

# Product 인스턴스임이 보장됨
print(type(product))  # <class '__main__.Product'>
```

#### 정규표현식 제너레이터 (Regex Generator)

```python
# 정규표현식과 일치하는 텍스트 생성
generator = outlines.generate.regex(
    model,
    r"[0-9]{3}-[0-9]{3}-[0-9]{4}"  # 전화번호 패턴
)

phone = generator("Generate phone number:")
# 결과: "555-123-4567" (패턴과 일치함이 보장됨)
```

#### 정수/실수 제너레이터 (Integer/Float Generators)

```python
# 특정 숫자 타입 생성
int_generator = outlines.generate.integer(model)
age = int_generator("Person's age:")  # 정수 보장

float_generator = outlines.generate.float(model)
price = float_generator("Product price:")  # 실수 보장
```

### 3. 모델 백엔드

Outlines는 여러 로컬 및 API 기반 백엔드를 지원합니다.

#### Transformers (Hugging Face)

```python
import outlines

# Hugging Face에서 로드
model = outlines.models.transformers(
    "microsoft/Phi-3-mini-4k-instruct",
    device="cuda"  # 또는 "cpu"
)

# 아무 제너레이터나 사용 가능
generator = outlines.generate.json(model, YourModel)
```

#### llama.cpp

```python
# GGUF 모델 로드
model = outlines.models.llamacpp(
    "./models/llama-3.1-8b-instruct.Q4_K_M.gguf",
    n_gpu_layers=35
)

generator = outlines.generate.json(model, YourModel)
```

#### vLLM (높은 처리량)

```python
# 프로덕션 배포용
model = outlines.models.vllm(
    "meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=2  # 멀티 GPU
)

generator = outlines.generate.json(model, YourModel)
```

#### OpenAI (제한적 지원)

```python
# 기본 OpenAI 지원
model = outlines.models.openai(
    "gpt-4o-mini",
    api_key="your-api-key"
)

# 주의: API 모델의 경우 일부 기능이 제한됨
generator = outlines.generate.json(model, YourModel)
```

### 4. Pydantic 연동

Outlines는 자동 스키마 변환을 지원하는 Pydantic과의 일급 연동을 갖추고 있습니다.

#### 기본 모델

```python
from pydantic import BaseModel, Field

class Article(BaseModel):
    title: str = Field(description="Article title")
    author: str = Field(description="Author name")
    word_count: int = Field(description="Number of words", gt=0)
    tags: list[str] = Field(description="List of tags")

model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")
generator = outlines.generate.json(model, Article)

article = generator("Generate article about AI")
print(article.title)
print(article.word_count)  # 0보다 크다는 것이 보장됨
```

#### 중첩 모델

```python
class Address(BaseModel):
    street: str
    city: str
    country: str

class Person(BaseModel):
    name: str
    age: int
    address: Address  # 중첩 모델

generator = outlines.generate.json(model, Person)
person = generator("Generate person in New York")

print(person.address.city)  # "New York"
```

#### Enums와 Literals

```python
from enum import Enum
from typing import Literal

class Status(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

class Application(BaseModel):
    applicant: str
    status: Status  # enum 값 중 하나여야 함
    priority: Literal["low", "medium", "high"]  # literal 중 하나여야 함

generator = outlines.generate.json(model, Application)
app = generator("Generate application")

print(app.status)  # Status.PENDING (또는 APPROVED/REJECTED)
```

## 일반적인 패턴

### 패턴 1: 데이터 추출

```python
from pydantic import BaseModel
import outlines

class CompanyInfo(BaseModel):
    name: str
    founded_year: int
    industry: str
    employees: int

model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")
generator = outlines.generate.json(model, CompanyInfo)

text = """
Apple Inc. was founded in 1976 in the technology industry.
The company employs approximately 164,000 people worldwide.
"""

prompt = f"Extract company information:\n{text}\n\nCompany:"
company = generator(prompt)

print(f"Name: {company.name}")
print(f"Founded: {company.founded_year}")
print(f"Industry: {company.industry}")
print(f"Employees: {company.employees}")
```

### 패턴 2: 분류

```python
from typing import Literal
import outlines

model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")

# 이진 분류
generator = outlines.generate.choice(model, ["spam", "not_spam"])
result = generator("Email: Buy now! 50% off!")

# 다중 클래스 분류
categories = ["technology", "business", "sports", "entertainment"]
category_gen = outlines.generate.choice(model, categories)
category = category_gen("Article: Apple announces new iPhone...")

# 신뢰도와 함께
class Classification(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float

classifier = outlines.generate.json(model, Classification)
result = classifier("Review: This product is okay, nothing special")
```

### 패턴 3: 구조화된 폼

```python
class UserProfile(BaseModel):
    full_name: str
    age: int
    email: str
    phone: str
    country: str
    interests: list[str]

model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")
generator = outlines.generate.json(model, UserProfile)

prompt = """
Extract user profile from:
Name: Alice Johnson
Age: 28
Email: alice@example.com
Phone: 555-0123
Country: USA
Interests: hiking, photography, cooking
"""

profile = generator(prompt)
print(profile.full_name)
print(profile.interests)  # ["hiking", "photography", "cooking"]
```

### 패턴 4: 다중 엔티티 추출

```python
class Entity(BaseModel):
    name: str
    type: Literal["PERSON", "ORGANIZATION", "LOCATION"]

class DocumentEntities(BaseModel):
    entities: list[Entity]

model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")
generator = outlines.generate.json(model, DocumentEntities)

text = "Tim Cook met with Satya Nadella at Microsoft headquarters in Redmond."
prompt = f"Extract entities from: {text}"

result = generator(prompt)
for entity in result.entities:
    print(f"{entity.name} ({entity.type})")
```

### 패턴 5: 코드 생성

```python
class PythonFunction(BaseModel):
    function_name: str
    parameters: list[str]
    docstring: str
    body: str

model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")
generator = outlines.generate.json(model, PythonFunction)

prompt = "Generate a Python function to calculate factorial"
func = generator(prompt)

print(f"def {func.function_name}({', '.join(func.parameters)}):")
print(f'    """{func.docstring}"""')
print(f"    {func.body}")
```

### 패턴 6: 일괄 처리

```python
def batch_extract(texts: list[str], schema: type[BaseModel]):
    """여러 텍스트에서 구조화된 데이터를 추출합니다."""
    model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")
    generator = outlines.generate.json(model, schema)

    results = []
    for text in texts:
        result = generator(f"Extract from: {text}")
        results.append(result)

    return results

class Person(BaseModel):
    name: str
    age: int

texts = [
    "John is 30 years old",
    "Alice is 25 years old",
    "Bob is 40 years old"
]

people = batch_extract(texts, Person)
for person in people:
    print(f"{person.name}: {person.age}")
```

## 백엔드 구성

### Transformers

```python
import outlines

# 기본 사용법
model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")

# GPU 구성
model = outlines.models.transformers(
    "microsoft/Phi-3-mini-4k-instruct",
    device="cuda",
    model_kwargs={"torch_dtype": "float16"}
)

# 인기 있는 모델들
model = outlines.models.transformers("meta-llama/Llama-3.1-8B-Instruct")
model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.3")
model = outlines.models.transformers("Qwen/Qwen2.5-7B-Instruct")
```

### llama.cpp

```python
# GGUF 모델 로드
model = outlines.models.llamacpp(
    "./models/llama-3.1-8b.Q4_K_M.gguf",
    n_ctx=4096,         # 컨텍스트 윈도우
    n_gpu_layers=35,    # GPU 레이어
    n_threads=8         # CPU 스레드
)

# 전체 GPU 오프로드
model = outlines.models.llamacpp(
    "./models/model.gguf",
    n_gpu_layers=-1  # 모든 레이어를 GPU에
)
```

### vLLM (프로덕션)

```python
# 단일 GPU
model = outlines.models.vllm("meta-llama/Llama-3.1-8B-Instruct")

# 멀티 GPU
model = outlines.models.vllm(
    "meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4  # 4 GPUs
)

# 양자화와 함께 사용
model = outlines.models.vllm(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization="awq"  # 또는 "gptq"
)
```

## 모범 사례

### 1. 구체적인 타입 사용

```python
# ✅ 좋음: 구체적인 타입들
class Product(BaseModel):
    name: str
    price: float  # str이 아님
    quantity: int  # str이 아님
    in_stock: bool  # str이 아님

# ❌ 나쁨: 모든 것이 문자열
class Product(BaseModel):
    name: str
    price: str  # float여야 함
    quantity: str  # int여야 함
```

### 2. 제약 조건 추가

```python
from pydantic import Field

# ✅ 좋음: 제약 조건 있음
class User(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    age: int = Field(ge=0, le=120)
    email: str = Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")

# ❌ 나쁨: 제약 조건 없음
class User(BaseModel):
    name: str
    age: int
    email: str
```

### 3. 카테고리에 Enums 사용

```python
# ✅ 좋음: 고정된 집합에 대한 Enum
class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Task(BaseModel):
    title: str
    priority: Priority

# ❌ 나쁨: 자유 형식 문자열
class Task(BaseModel):
    title: str
    priority: str  # 무엇이든 될 수 있음
```

### 4. 프롬프트에 문맥 제공

```python
# ✅ 좋음: 명확한 문맥
prompt = """
Extract product information from the following text.
Text: iPhone 15 Pro costs $999 and is currently in stock.
Product:
"""

# ❌ 나쁨: 최소한의 문맥
prompt = "iPhone 15 Pro costs $999 and is currently in stock."
```

### 5. 선택적(Optional) 필드 처리

```python
from typing import Optional

# ✅ 좋음: 불완전한 데이터를 위한 선택적 필드
class Article(BaseModel):
    title: str  # 필수
    author: Optional[str] = None  # 선택적
    date: Optional[str] = None  # 선택적
    tags: list[str] = []  # 기본값 빈 리스트

# 저자/날짜가 없어도 성공할 수 있음
```

## 대안과의 비교

| 기능 | Outlines | Instructor | Guidance | LMQL |
|---------|----------|------------|----------|------|
| Pydantic 지원 | ✅ 네이티브 | ✅ 네이티브 | ❌ 아니오 | ❌ 아니오 |
| JSON 스키마 | ✅ 예 | ✅ 예 | ⚠️ 제한적 | ✅ 예 |
| 정규표현식 제약 | ✅ 예 | ❌ 아니오 | ✅ 예 | ✅ 예 |
| 로컬 모델 | ✅ 전체 지원 | ⚠️ 제한적 | ✅ 전체 지원 | ✅ 전체 지원 |
| API 모델 | ⚠️ 제한적 | ✅ 전체 지원 | ✅ 전체 지원 | ✅ 전체 지원 |
| 제로 오버헤드 | ✅ 예 | ❌ 아니오 | ⚠️ 부분적 | ✅ 예 |
| 자동 재시도 | ❌ 아니오 | ✅ 예 | ❌ 아니오 | ❌ 아니오 |
| 학습 곡선 | 낮음 | 낮음 | 낮음 | 높음 |

**Outlines를 선택해야 하는 경우:**
- 로컬 모델(Transformers, llama.cpp, vLLM)을 사용 중인 경우
- 최대 추론 속도가 필요한 경우
- Pydantic 모델 지원을 원하는 경우
- 제로 오버헤드의 구조화된 생성이 필요한 경우
- 토큰 샘플링 프로세스를 제어해야 하는 경우

**대안을 선택해야 하는 경우:**
- Instructor: 자동 재시도 기능이 있는 API 모델이 필요한 경우
- Guidance: 토큰 힐링 및 복잡한 워크플로우가 필요한 경우
- LMQL: 선언형 쿼리 문법을 선호하는 경우

## 성능 특성

**속도:**
- **제로 오버헤드**: 구조화된 생성이 제약 없는 생성만큼 빠릅니다.
- **빨리 감기 최적화**: 결정론적 토큰을 건너뜁니다.
- 사후 생성 검증 접근 방식보다 **1.2-2배 더 빠릅니다**.

**메모리:**
- 스키마당 한 번 FSM 컴파일(캐시됨)
- 최소 런타임 오버헤드
- 높은 처리량을 위한 vLLM과 효율적

**정확도:**
- **100% 유효한 출력** (FSM이 보장)
- 재시도 루프가 필요 없음
- 결정론적 토큰 필터링

## 리소스

- **문서**: https://outlines-dev.github.io/outlines
- **GitHub**: https://github.com/outlines-dev/outlines (8k+ stars)
- **Discord**: https://discord.gg/R9DSu34mGd
- **블로그**: https://blog.dottxt.co

## 참고 항목

- `references/json_generation.md` - 포괄적인 JSON 및 Pydantic 패턴
- `references/backends.md` - 백엔드 관련 구성
- `references/examples.md` - 프로덕션 지원 예제

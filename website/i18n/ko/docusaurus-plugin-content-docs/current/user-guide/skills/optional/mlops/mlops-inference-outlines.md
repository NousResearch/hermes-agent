---
title: "Outlines — Outlines: 구조화된 JSON/정규식/Pydantic LLM 생성"
sidebar_label: "Outlines"
description: "Outlines: 구조화된 JSON/정규식/Pydantic LLM 생성"
---

{/* 이 페이지는 skill의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Outlines

Outlines: 구조화된 JSON/정규식/Pydantic LLM 생성.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/mlops/outlines`로 설치 |
| 경로 | `optional-skills/mlops/inference/outlines` |
| 버전 | `1.0.1` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `outlines`, `transformers`, `vllm`, `pydantic` |
| 플랫폼 | linux, macos, windows |
| 태그 | `프롬프트 엔지니어링`, `Outlines`, `구조화된 생성`, `JSON 스키마`, `Pydantic`, `로컬 모델`, `문법 기반 생성`, `vLLM`, `Transformers`, `타입 안전성` |

## 참고: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬이 활성화될 때 로드하는 전체 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 지침으로 보는 내용입니다.
:::

# Outlines: 구조화된 텍스트 생성

## 이 스킬을 사용하는 경우

다음이 필요할 때 Outlines를 사용하세요.
- **생성 중 유효한 JSON/XML/코드 구조를 보장해야 할 때**
- **타입 안전한 출력을 위해 Pydantic 모델을 사용할 때**
- **로컬 모델(Transformers, llama.cpp, vLLM)을 지원할 때**
- **오버헤드 없는 구조화된 생성을 통해 추론 속도를 극대화할 때**
- **자동으로 JSON 스키마를 대상으로 생성할 때**
- **문법 수준에서 토큰 샘플링을 제어할 때**

**GitHub 스타**: 12,000+ | **출처**: dottxt.ai (이전 명칭 .txt)

> **API 참고 사항(Outlines 1.x):** 이 스킬은 현재 v1 API를 대상으로 합니다.
> v1.0 이전의 헬퍼(`outlines.models.transformers(...)`,
> `outlines.generate.json/choice/regex/...`)는 **삭제되었습니다**. v1에서는
> `outlines.from_transformers(...)`(또는 `from_vllm`,
> `from_llamacpp`, `from_openai`)로 모델을 생성한 다음, 출력 타입과 함께
> 모델을 직접 호출합니다: `model(prompt, output_type)`. JSON/Pydantic 출력은
> **JSON 문자열**로 반환되므로 `YourModel.model_validate_json(result)`로
> 유효성을 검사하세요.

## 설치

```bash
# Base installation
pip install outlines

# With specific backends
pip install outlines transformers  # Hugging Face models
pip install outlines llama-cpp-python  # llama.cpp
pip install outlines vllm  # vLLM for high-throughput
```

## 빠른 시작

### 기본 예제: 분류

```python
import outlines
from typing import Literal
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# v1: wrap a Transformers model + tokenizer
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto"),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)

# Call the model directly with an output type
prompt = "Sentiment of 'This product is amazing!': "
sentiment = model(prompt, Literal["positive", "negative", "neutral"])

print(sentiment)  # "positive" (guaranteed one of these)
```

### Pydantic 모델 사용

```python
from pydantic import BaseModel
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

class User(BaseModel):
    name: str
    age: int
    email: str

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto"),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)

# Generate structured output (returns a JSON string)
prompt = "Extract user: John Doe, 30 years old, john@example.com"
result = model(prompt, User, max_new_tokens=200)

user = User.model_validate_json(result)  # parse into the Pydantic model
print(user.name)   # "John Doe"
print(user.age)    # 30
print(user.email)  # "john@example.com"
```

## 핵심 개념

### 1. 제약된 토큰 샘플링

Outlines는 출력 타입에서 파생되어 컴파일된 오토마톤을 사용해 로짓 수준에서 토큰 생성을 제약합니다.

**작동 방식:**
1. 출력 타입(JSON/Pydantic/정규식/`Literal`)을 스키마/문법으로 변환합니다.
2. 문법을 토큰 수준 오토마톤으로 컴파일합니다.
3. 생성 중 각 단계에서 유효하지 않은 토큰을 필터링합니다.
4. 유효한 토큰이 하나뿐이면 빠르게 진행합니다.

**장점:**
- **오버헤드 없음**: 토큰 수준에서 필터링이 이루어집니다.
- **속도 향상**: 결정적인 경로를 빠르게 통과합니다.
- **유효성 보장**: 유효하지 않은 출력은 불가능합니다.

```python
import outlines
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

class Person(BaseModel):
    name: str
    age: int

model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", device_map="auto"),
    AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
)

result = model("Generate person: Alice, 25", Person)
person = Person.model_validate_json(result)
```

### 2. 출력 타입

v1에서는 원하는 **출력 타입**을 두 번째 인수로 직접 전달합니다.

#### 다중 선택(`Literal`)

```python
from typing import Literal

sentiment = model("Review: This is great!", Literal["positive", "negative", "neutral"])
# Result: one of the three choices
```

#### Pydantic을 통한 JSON

```python
from pydantic import BaseModel

class Product(BaseModel):
    name: str
    price: float
    in_stock: bool

result = model("Extract: iPhone 15, $999, available", Product)
product = Product.model_validate_json(result)  # valid Product instance
```

#### 정규식(정규식 문자열 전달)

```python
# Generate text matching a regex pattern
phone = model("Generate phone number:", r"[0-9]{3}-[0-9]{3}-[0-9]{4}")
# Result: "555-123-4567" (guaranteed to match the pattern)
```

#### 숫자 타입

```python
# Pass the Python type directly
age = model("Person's age:", int)      # guaranteed integer
price = model("Product price:", float)  # guaranteed float
```

### 3. 모델 백엔드

Outlines는 `from_*` 팩토리를 통해 여러 로컬 및 API 기반 백엔드를 지원합니다.

#### Transformers(Hugging Face)

```python
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", device_map="auto"),
    AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
)

result = model(prompt, YourModel)
```

#### llama.cpp

```python
import outlines
from llama_cpp import Llama

llm = Llama("./models/llama-3.1-8b-instruct.Q4_K_M.gguf", n_gpu_layers=35, n_ctx=4096)
model = outlines.from_llamacpp(llm)

result = model(prompt, YourModel)
```

#### vLLM(높은 처리량)

```python
import outlines
from vllm import LLM

llm = LLM("meta-llama/Llama-3.1-8B-Instruct", tensor_parallel_size=2)
model = outlines.from_vllm(llm)

result = model(prompt, YourModel)
```

#### OpenAI(서버 측 제약된 JSON)

```python
import outlines
from openai import OpenAI

client = OpenAI()
model = outlines.from_openai(client, "gpt-4o-mini")

# API backends support JSON-schema style structured output
result = model(prompt, YourModel)
```

### 4. Pydantic 통합

Outlines는 스키마를 자동으로 변환하는 일급 Pydantic 지원을 제공합니다.
생성 결과는 JSON 문자열이므로 `model_validate_json`을 호출해 인스턴스를 얻으세요.

#### 기본 모델

```python
from pydantic import BaseModel, Field

class Article(BaseModel):
    title: str = Field(description="Article title")
    author: str = Field(description="Author name")
    word_count: int = Field(description="Number of words", gt=0)
    tags: list[str] = Field(description="List of tags")

result = model("Generate article about AI", Article, max_new_tokens=300)
article = Article.model_validate_json(result)
print(article.title)
print(article.word_count)  # Guaranteed > 0
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
    address: Address  # Nested model

result = model("Generate person in New York", Person)
person = Person.model_validate_json(result)
print(person.address.city)  # "New York"
```

#### 열거형 및 리터럴

```python
from enum import Enum
from typing import Literal

class Status(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

class Application(BaseModel):
    applicant: str
    status: Status  # Must be one of enum values
    priority: Literal["low", "medium", "high"]  # Must be one of literals

result = model("Generate application", Application)
app = Application.model_validate_json(result)
print(app.status)  # Status.PENDING (or APPROVED/REJECTED)
```

## 일반적인 패턴

### 패턴 1: 데이터 추출

```python
from pydantic import BaseModel
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

class CompanyInfo(BaseModel):
    name: str
    founded_year: int
    industry: str
    employees: int

model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", device_map="auto"),
    AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
)

text = """
Apple Inc. was founded in 1976 in the technology industry.
The company employs approximately 164,000 people worldwide.
"""

prompt = f"Extract company information:\n{text}\n\nCompany:"
company = CompanyInfo.model_validate_json(model(prompt, CompanyInfo, max_new_tokens=200))

print(f"Name: {company.name}")
print(f"Founded: {company.founded_year}")
print(f"Industry: {company.industry}")
print(f"Employees: {company.employees}")
```

### 패턴 2: 분류

```python
from typing import Literal
from pydantic import BaseModel

# Binary classification
result = model("Email: Buy now! 50% off!", Literal["spam", "not_spam"])

# Multi-class classification
category = model(
    "Article: Apple announces new iPhone...",
    Literal["technology", "business", "sports", "entertainment"],
)

# With confidence
class Classification(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float

out = model("Review: This product is okay, nothing special", Classification)
result = Classification.model_validate_json(out)
```

### 패턴 3: 구조화된 양식

```python
class UserProfile(BaseModel):
    full_name: str
    age: int
    email: str
    phone: str
    country: str
    interests: list[str]

prompt = """
Extract user profile from:
Name: Alice Johnson
Age: 28
Email: alice@example.com
Phone: 555-0123
Country: USA
Interests: hiking, photography, cooking
"""

profile = UserProfile.model_validate_json(model(prompt, UserProfile, max_new_tokens=250))
print(profile.full_name)
print(profile.interests)  # ["hiking", "photography", "cooking"]
```

### 패턴 4: 다중 엔터티 추출

```python
from typing import Literal

class Entity(BaseModel):
    name: str
    type: Literal["PERSON", "ORGANIZATION", "LOCATION"]

class DocumentEntities(BaseModel):
    entities: list[Entity]

text = "Tim Cook met with Satya Nadella at Microsoft headquarters in Redmond."
prompt = f"Extract entities from: {text}"

result = DocumentEntities.model_validate_json(model(prompt, DocumentEntities, max_new_tokens=300))
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

prompt = "Generate a Python function to calculate factorial"
func = PythonFunction.model_validate_json(model(prompt, PythonFunction, max_new_tokens=300))

print(f"def {func.function_name}({', '.join(func.parameters)}):")
print(f'    """{func.docstring}"""')
print(f"    {func.body}")
```

### 패턴 6: 일괄 처리

```python
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", device_map="auto"),
    AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
)

texts = [
    "John is 30 years old",
    "Alice is 25 years old",
    "Bob is 40 years old",
]

# v1 accepts a list of prompts for batched generation
prompts = [f"Extract from: {t}" for t in texts]
outputs = model(prompts, Person, max_new_tokens=100)
people = [Person.model_validate_json(o) for o in outputs]
for person in people:
    print(f"{person.name}: {person.age}")
```

## 백엔드 구성

### Transformers

```python
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# Basic usage
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto"),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)

# GPU + dtype configuration is set on the HF model itself
import torch
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="cuda", torch_dtype=torch.float16),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)

# Popular models
for name in [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
]:
    model = outlines.from_transformers(
        AutoModelForCausalLM.from_pretrained(name, device_map="auto"),
        AutoTokenizer.from_pretrained(name),
    )
```

### llama.cpp

```python
import outlines
from llama_cpp import Llama

# Load GGUF model
llm = Llama(
    "./models/llama-3.1-8b.Q4_K_M.gguf",
    n_ctx=4096,       # Context window
    n_gpu_layers=35,  # GPU layers
    n_threads=8,      # CPU threads
)
model = outlines.from_llamacpp(llm)

# Full GPU offload: set n_gpu_layers=-1 on the Llama object
```

### vLLM(프로덕션)

```python
import outlines
from vllm import LLM

# Single GPU
model = outlines.from_vllm(LLM("meta-llama/Llama-3.1-8B-Instruct"))

# Multi-GPU
model = outlines.from_vllm(LLM("meta-llama/Llama-3.1-70B-Instruct", tensor_parallel_size=4))

# With quantization
model = outlines.from_vllm(LLM("meta-llama/Llama-3.1-8B-Instruct", quantization="awq"))
```

## 모범 사례

### 1. 구체적인 타입 사용

```python
# ✅ Good: Specific types
class Product(BaseModel):
    name: str
    price: float  # Not str
    quantity: int  # Not str
    in_stock: bool  # Not str

# ❌ Bad: Everything as string
class Product(BaseModel):
    name: str
    price: str  # Should be float
    quantity: str  # Should be int
```

### 2. 제약 조건 추가

```python
from pydantic import Field

# ✅ Good: With constraints
class User(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    age: int = Field(ge=0, le=120)
    email: str = Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")

# ❌ Bad: No constraints
class User(BaseModel):
    name: str
    age: int
    email: str
```

### 3. 범주에 열거형 사용

```python
# ✅ Good: Enum for fixed set
class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Task(BaseModel):
    title: str
    priority: Priority

# ❌ Bad: Free-form string
class Task(BaseModel):
    title: str
    priority: str  # Can be anything
```

### 4. 프롬프트에 맥락 제공

```python
# ✅ Good: Clear context
prompt = """
Extract product information from the following text.
Text: iPhone 15 Pro costs $999 and is currently in stock.
Product:
"""

# ❌ Bad: Minimal context
prompt = "iPhone 15 Pro costs $999 and is currently in stock."
```

### 5. 선택적 필드 처리

```python
from typing import Optional

# ✅ Good: Optional fields for incomplete data
class Article(BaseModel):
    title: str  # Required
    author: Optional[str] = None  # Optional
    date: Optional[str] = None  # Optional
    tags: list[str] = []  # Default empty list

# Can succeed even if author/date missing
```

### 6. JSON 출력은 항상 유효성 검사

```python
# v1 returns a JSON string for Pydantic/JSON output types.
result = model(prompt, Article)          # str
article = Article.model_validate_json(result)  # Article instance
```

## 대안과의 비교

| 기능 | Outlines | Instructor | Guidance | LMQL |
|---------|----------|------------|----------|------|
| Pydantic 지원 | ✅ 기본 제공 | ✅ 기본 제공 | ✅ 지원 | ❌ 미지원 |
| JSON 스키마 | ✅ 지원 | ✅ 지원 | ✅ 지원 | ✅ 지원 |
| 정규식 제약 | ✅ 지원 | ❌ 미지원 | ✅ 지원 | ✅ 지원 |
| 로컬 모델 | ✅ 전체 지원 | ⚠️ 제한적 | ✅ 전체 지원 | ✅ 전체 지원 |
| API 모델 | ✅ 지원 | ✅ 전체 지원 | ✅ 지원 | ✅ 전체 지원 |
| 오버헤드 없음 | ✅ 지원 | ❌ 미지원 | ⚠️ 부분 지원 | ✅ 지원 |
| 자동 재시도 | ❌ 미지원 | ✅ 지원 | ❌ 미지원 | ❌ 미지원 |
| 학습 난이도 | 낮음 | 낮음 | 낮음 | 높음 |

**Outlines를 선택할 때:**
- 로컬 모델(Transformers, llama.cpp, vLLM)을 사용할 때
- 최대 추론 속도가 필요할 때
- Pydantic 모델 지원을 원할 때
- 오버헤드 없는 구조화된 생성이 필요할 때
- 토큰 샘플링 프로세스를 제어할 때

**대안을 선택할 때:**
- Instructor: 자동 재시도가 포함된 API 모델이 필요할 때
- Guidance: 토큰 복구와 복잡한 워크플로가 필요할 때
- LMQL: 선언적 쿼리 구문을 선호할 때

## 성능 특성

**속도:**
- **오버헤드 없음**: 구조화된 생성이 제약 없는 생성만큼 빠릅니다.
- **빠른 진행 최적화**: 결정적인 토큰을 건너뜁니다.
- **생성 후 유효성 검사 방식보다 1.2~2배 빠름**

**메모리:**
- 출력 타입별로 오토마톤을 한 번 컴파일합니다(캐시됨).
- 런타임 오버헤드가 최소입니다.
- 높은 처리량을 위한 vLLM과 효율적으로 작동합니다.

**정확도:**
- **유효한 출력 100%**(제약된 오토마톤이 보장)
- 재시도 루프가 필요하지 않습니다.
- 결정적인 토큰 필터링

## 리소스

- **문서**: https://dottxt-ai.github.io/outlines/
- **GitHub**: https://github.com/dottxt-ai/outlines (12k+ 스타)
- **Discord**: https://discord.gg/R9DSu34mGd
- **블로그**: https://blog.dottxt.co

## 관련 문서

- `references/json_generation.md` - JSON 및 Pydantic 패턴 종합 안내
- `references/backends.md` - 백엔드별 구성
- `references/examples.md` - 프로덕션용 예제

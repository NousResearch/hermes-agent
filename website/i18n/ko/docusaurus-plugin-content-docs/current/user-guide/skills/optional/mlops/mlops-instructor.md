---
title: "Instructor"
sidebar_label: "Instructor"
description: "LLM 응답에서 Pydantic 검증을 사용하여 구조화된 데이터를 추출하고, 실패한 추출을 자동으로 재시도하며, 타입 안전성을 갖춘 복잡한 JSON을 구문 분석하고, 스트리밍합니다..."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Instructor

전투 테스트를 거친 구조화된 출력 라이브러리인 Instructor를 사용하여 Pydantic 검증과 함께 LLM 응답에서 구조화된 데이터를 추출하고, 실패한 추출을 자동으로 재시도하며, 타입 안전성을 갖춘 복잡한 JSON을 구문 분석하고, 부분 결과를 스트리밍합니다.

## 스킬 메타데이터

| | |
|---|---|
| Source | Optional — `hermes skills install official/mlops/instructor`로 설치 |
| Path | `optional-skills/mlops/instructor` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `instructor`, `pydantic`, `openai`, `anthropic` |
| Platforms | linux, macos, windows |
| Tags | `Prompt Engineering`, `Instructor`, `Structured Output`, `Pydantic`, `Data Extraction`, `JSON Parsing`, `Type Safety`, `Validation`, `Streaming`, `OpenAI`, `Anthropic` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되어 있을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Instructor: 구조화된 LLM 출력

## 이 스킬을 사용해야 할 때

다음과 같은 경우에 Instructor를 사용하세요:
- LLM 응답에서 **구조화된 데이터를 안정적으로 추출**해야 할 때
- Pydantic 스키마를 대상으로 **출력을 자동으로 검증**해야 할 때
- 자동 오류 처리를 통해 **실패한 추출을 재시도**해야 할 때
- 타입 안전성과 유효성 검사로 **복잡한 JSON을 구문 분석**해야 할 때
- 실시간 처리를 위해 **부분 결과를 스트리밍**해야 할 때
- 일관된 API로 **여러 LLM 제공 업체를 지원**해야 할 때

**GitHub Stars**: 15,000+ | **전투 테스트**: 100,000+ 개발자

## 설치

```bash
# 기본 설치
pip install instructor

# 특정 제공 업체 포함
pip install "instructor[anthropic]"  # Anthropic Claude
pip install "instructor[openai]"     # OpenAI
pip install "instructor[all]"        # 모든 제공 업체
```

## 빠른 시작

### 기본 예제: 사용자 데이터 추출

```python
import instructor
from pydantic import BaseModel
from anthropic import Anthropic

# 출력 구조 정의
class User(BaseModel):
    name: str
    age: int
    email: str

# instructor 클라이언트 생성
client = instructor.from_anthropic(Anthropic())

# 구조화된 데이터 추출
user = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": "John Doe is 30 years old. His email is john@example.com"
    }],
    response_model=User
)

print(user.name)   # "John Doe"
print(user.age)    # 30
print(user.email)  # "john@example.com"
```

### OpenAI와 함께 사용

```python
from openai import OpenAI

client = instructor.from_openai(OpenAI())

user = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=User,
    messages=[{"role": "user", "content": "Extract: Alice, 25, alice@email.com"}]
)
```

## 핵심 개념

### 1. 응답 모델 (Pydantic)

응답 모델은 LLM 출력에 대한 구조와 유효성 검사 규칙을 정의합니다.

#### 기본 모델

```python
from pydantic import BaseModel, Field

class Article(BaseModel):
    title: str = Field(description="Article title")
    author: str = Field(description="Author name")
    word_count: int = Field(description="Number of words", gt=0)
    tags: list[str] = Field(description="List of relevant tags")

article = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": "Analyze this article: [article text]"
    }],
    response_model=Article
)
```

**이점:**
- Python 타입 힌트를 사용한 타입 안전성
- 자동 유효성 검사 (word_count > 0)
- Field 설명을 통한 자체 문서화
- IDE 자동 완성 지원

#### 중첩된 모델

```python
class Address(BaseModel):
    street: str
    city: str
    country: str

class Person(BaseModel):
    name: str
    age: int
    address: Address  # 중첩된 모델

person = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": "John lives at 123 Main St, Boston, USA"
    }],
    response_model=Person
)

print(person.address.city)  # "Boston"
```

#### 선택적 필드

```python
from typing import Optional

class Product(BaseModel):
    name: str
    price: float
    discount: Optional[float] = None  # 선택적
    description: str = Field(default="No description")  # 기본값

# LLM은 discount 또는 description을 제공할 필요가 없습니다.
```

#### 제약 조건을 위한 Enum

```python
from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class Review(BaseModel):
    text: str
    sentiment: Sentiment  # 이 3가지 값만 허용됨

review = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": "This product is amazing!"
    }],
    response_model=Review
)

print(review.sentiment)  # Sentiment.POSITIVE
```

### 2. 유효성 검사 (Validation)

Pydantic은 LLM 출력을 자동으로 검증합니다. 유효성 검사에 실패하면 Instructor가 다시 시도합니다.

#### 내장 유효성 검사기

```python
from pydantic import Field, EmailStr, HttpUrl

class Contact(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    age: int = Field(ge=0, le=120)  # 0 <= age <= 120
    email: EmailStr  # 이메일 형식 검증
    website: HttpUrl  # URL 형식 검증

# LLM이 유효하지 않은 데이터를 제공하면 Instructor가 자동으로 다시 시도합니다.
```

#### 사용자 정의 유효성 검사기

```python
from pydantic import field_validator

class Event(BaseModel):
    name: str
    date: str
    attendees: int

    @field_validator('date')
    def validate_date(cls, v):
        """날짜가 YYYY-MM-DD 형식인지 확인합니다."""
        import re
        if not re.match(r'\d{4}-\d{2}-\d{2}', v):
            raise ValueError('Date must be YYYY-MM-DD format')
        return v

    @field_validator('attendees')
    def validate_attendees(cls, v):
        """참석자 수가 양수인지 확인합니다."""
        if v < 1:
            raise ValueError('Must have at least 1 attendee')
        return v
```

#### 모델 수준 유효성 검사기

```python
from pydantic import model_validator

class DateRange(BaseModel):
    start_date: str
    end_date: str

    @model_validator(mode='after')
    def check_dates(self):
        """end_date가 start_date 이후인지 확인합니다."""
        from datetime import datetime
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')

        if end < start:
            raise ValueError('end_date must be after start_date')
        return self
```

### 3. 자동 재시도

Instructor는 유효성 검사에 실패하면 LLM에 오류 피드백을 제공하여 자동으로 재시도합니다.

```python
# 유효성 검사 실패 시 최대 3회 재시도
user = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": "Extract user from: John, age unknown"
    }],
    response_model=User,
    max_retries=3  # 기본값은 3
)

# age를 추출할 수 없는 경우 Instructor가 LLM에 알려줍니다:
# "Validation error: age - field required"
# LLM은 더 나은 추출을 위해 다시 시도합니다.
```

**작동 방식:**
1. LLM이 출력을 생성합니다.
2. Pydantic이 유효성을 검사합니다.
3. 유효하지 않은 경우: 오류 메시지가 LLM으로 다시 전송됩니다.
4. LLM이 오류 피드백을 바탕으로 다시 시도합니다.
5. max_retries에 도달할 때까지 반복됩니다.

### 4. 스트리밍

실시간 처리를 위해 부분 결과를 스트리밍합니다.

#### 부분 객체 스트리밍

```python
from instructor import Partial

class Story(BaseModel):
    title: str
    content: str
    tags: list[str]

# LLM이 생성할 때 부분 업데이트 스트리밍
for partial_story in client.messages.create_partial(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": "Write a short sci-fi story"
    }],
    response_model=Story
):
    print(f"Title: {partial_story.title}")
    print(f"Content so far: {partial_story.content[:100]}...")
    # 실시간으로 UI 업데이트
```

#### Iterable 스트리밍

```python
class Task(BaseModel):
    title: str
    priority: str

# 생성되는 대로 목록 항목 스트리밍
tasks = client.messages.create_iterable(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": "Generate 10 project tasks"
    }],
    response_model=Task
)

for task in tasks:
    print(f"- {task.title} ({task.priority})")
    # 도착하는 각 작업을 처리합니다.
```

## 제공 업체 구성

### Anthropic Claude

```python
import instructor
from anthropic import Anthropic

client = instructor.from_anthropic(
    Anthropic(api_key="your-api-key")
)

# Claude 모델과 함께 사용
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[...],
    response_model=YourModel
)
```

### OpenAI

```python
from openai import OpenAI

client = instructor.from_openai(
    OpenAI(api_key="your-api-key")
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=YourModel,
    messages=[...]
)
```

### 로컬 모델 (Ollama)

```python
from openai import OpenAI

# 로컬 Ollama 서버 가리키기
client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"  # 필수 항목이지만 무시됨
    ),
    mode=instructor.Mode.JSON
)

response = client.chat.completions.create(
    model="llama3.1",
    response_model=YourModel,
    messages=[...]
)
```

## 일반적인 패턴

### 패턴 1: 텍스트에서 데이터 추출

```python
class CompanyInfo(BaseModel):
    name: str
    founded_year: int
    industry: str
    employees: int
    headquarters: str

text = """
Tesla, Inc. was founded in 2003. It operates in the automotive and energy
industry with approximately 140,000 employees. The company is headquartered
in Austin, Texas.
"""

company = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": f"Extract company information from: {text}"
    }],
    response_model=CompanyInfo
)
```

### 패턴 2: 분류 (Classification)

```python
class Category(str, Enum):
    TECHNOLOGY = "technology"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    OTHER = "other"

class ArticleClassification(BaseModel):
    category: Category
    confidence: float = Field(ge=0.0, le=1.0)
    keywords: list[str]

classification = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": "Classify this article: [article text]"
    }],
    response_model=ArticleClassification
)
```

### 패턴 3: 다중 개체 추출 (Multi-Entity Extraction)

```python
class Person(BaseModel):
    name: str
    role: str

class Organization(BaseModel):
    name: str
    industry: str

class Entities(BaseModel):
    people: list[Person]
    organizations: list[Organization]
    locations: list[str]

text = "Tim Cook, CEO of Apple, announced at the event in Cupertino..."

entities = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": f"Extract all entities from: {text}"
    }],
    response_model=Entities
)

for person in entities.people:
    print(f"{person.name} - {person.role}")
```

### 패턴 4: 구조적 분석 (Structured Analysis)

```python
class SentimentAnalysis(BaseModel):
    overall_sentiment: Sentiment
    positive_aspects: list[str]
    negative_aspects: list[str]
    suggestions: list[str]
    score: float = Field(ge=-1.0, le=1.0)

review = "The product works well but setup was confusing..."

analysis = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": f"Analyze this review: {review}"
    }],
    response_model=SentimentAnalysis
)
```

### 패턴 5: 일괄 처리 (Batch Processing)

```python
def extract_person(text: str) -> Person:
    return client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"Extract person from: {text}"
        }],
        response_model=Person
    )

texts = [
    "John Doe is a 30-year-old engineer",
    "Jane Smith, 25, works in marketing",
    "Bob Johnson, age 40, software developer"
]

people = [extract_person(text) for text in texts]
```

## 고급 기능

### 공용체 타입 (Union Types)

```python
from typing import Union

class TextContent(BaseModel):
    type: str = "text"
    content: str

class ImageContent(BaseModel):
    type: str = "image"
    url: HttpUrl
    caption: str

class Post(BaseModel):
    title: str
    content: Union[TextContent, ImageContent]  # 두 타입 중 하나

# LLM은 내용을 기반으로 적절한 타입을 선택합니다.
```

### 동적 모델 (Dynamic Models)

```python
from pydantic import create_model

# 런타임에 모델 생성
DynamicUser = create_model(
    'User',
    name=(str, ...),
    age=(int, Field(ge=0)),
    email=(EmailStr, ...)
)

user = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[...],
    response_model=DynamicUser
)
```

### 사용자 정의 모드 (Custom Modes)

```python
# 기본 구조화된 출력이 없는 제공 업체의 경우
client = instructor.from_anthropic(
    Anthropic(),
    mode=instructor.Mode.JSON  # JSON 모드
)

# 사용 가능한 모드:
# - Mode.ANTHROPIC_TOOLS (Claude 권장)
# - Mode.JSON (대체)
# - Mode.TOOLS (OpenAI 도구)
```

### 컨텍스트 관리 (Context Management)

```python
# 일회용 클라이언트
with instructor.from_anthropic(Anthropic()) as client:
    result = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=[...],
        response_model=YourModel
    )
    # 클라이언트가 자동으로 닫힙니다.
```

## 오류 처리

### 유효성 검사 오류 처리

```python
from pydantic import ValidationError

try:
    user = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=[...],
        response_model=User,
        max_retries=3
    )
except ValidationError as e:
    print(f"Failed after retries: {e}")
    # 우아하게 처리

except Exception as e:
    print(f"API error: {e}")
```

### 사용자 정의 오류 메시지

```python
class ValidatedUser(BaseModel):
    name: str = Field(description="Full name, 2-100 characters")
    age: int = Field(description="Age between 0 and 120", ge=0, le=120)
    email: EmailStr = Field(description="Valid email address")

    class Config:
        # 사용자 정의 오류 메시지
        json_schema_extra = {
            "examples": [
                {
                    "name": "John Doe",
                    "age": 30,
                    "email": "john@example.com"
                }
            ]
        }
```

## 모범 사례

### 1. 명확한 필드 설명

```python
# ❌ 나쁨: 모호함
class Product(BaseModel):
    name: str
    price: float

# ✅ 좋음: 설명적임
class Product(BaseModel):
    name: str = Field(description="Product name from the text")
    price: float = Field(description="Price in USD, without currency symbol")
```

### 2. 적절한 유효성 검사 사용

```python
# ✅ 좋음: 값 제한
class Rating(BaseModel):
    score: int = Field(ge=1, le=5, description="Rating from 1 to 5 stars")
    review: str = Field(min_length=10, description="Review text, at least 10 chars")
```

### 3. 프롬프트에 예제 제공

```python
messages = [{
    "role": "user",
    "content": """Extract person info from: "John, 30, engineer"

Example format:
{
  "name": "John Doe",
  "age": 30,
  "occupation": "engineer"
}"""
}]
```

### 4. 고정된 범주에 Enum 사용

```python
# ✅ 좋음: Enum은 유효한 값을 보장합니다.
class Status(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

class Application(BaseModel):
    status: Status  # LLM은 열거형에서 선택해야 합니다.
```

### 5. 누락된 데이터를 우아하게 처리

```python
class PartialData(BaseModel):
    required_field: str
    optional_field: Optional[str] = None
    default_field: str = "default_value"

# LLM은 required_field만 제공하면 됩니다.
```

## 대안과의 비교

| 기능 | Instructor | 수동 JSON | LangChain | DSPy |
|---------|------------|-------------|-----------|------|
| 타입 안전성 | ✅ 예 | ❌ 아니오 | ⚠️ 부분적 | ✅ 예 |
| 자동 유효성 검사 | ✅ 예 | ❌ 아니오 | ❌ 아니오 | ⚠️ 제한적 |
| 자동 재시도 | ✅ 예 | ❌ 아니오 | ❌ 아니오 | ✅ 예 |
| 스트리밍 | ✅ 예 | ❌ 아니오 | ✅ 예 | ❌ 아니오 |
| 다중 제공 업체 | ✅ 예 | ⚠️ 수동 | ✅ 예 | ✅ 예 |
| 학습 곡선 | 낮음 | 낮음 | 중간 | 높음 |

**Instructor를 선택해야 할 때:**
- 구조화되고 검증된 출력이 필요할 때
- 타입 안전성과 IDE 지원을 원할 때
- 자동 재시도가 필요할 때
- 데이터 추출 시스템을 구축할 때

**대안을 선택해야 할 때:**
- DSPy: 프롬프트 최적화가 필요할 때
- LangChain: 복잡한 체인을 구축할 때
- 수동: 간단하고 일회성 추출일 때

## 리소스

- **문서**: https://python.useinstructor.com
- **GitHub**: https://github.com/jxnl/instructor (15k+ stars)
- **Cookbook**: https://python.useinstructor.com/examples
- **Discord**: 커뮤니티 지원 가능

## 함께 보기

- `references/validation.md` - 고급 유효성 검사 패턴
- `references/providers.md` - 제공 업체별 구성
- `references/examples.md` - 실제 사용 사례

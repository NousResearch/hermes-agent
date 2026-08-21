---
title: "Instructor — Pydantic으로 검증하는 구조화된 LLM 출력"
sidebar_label: "Instructor"
description: "Pydantic으로 검증하는 구조화된 LLM 출력"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Instructor

Pydantic으로 검증하는 구조화된 LLM 출력입니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/mlops/instructor`로 설치 |
| 경로 | `optional-skills/mlops/instructor` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `instructor`, `pydantic`, `openai`, `anthropic` |
| 플랫폼 | linux, macos, windows |
| 태그 | `Prompt Engineering`, `Instructor`, `Structured Output`, `Pydantic`, `Data Extraction`, `JSON Parsing`, `Type Safety`, `Validation`, `Streaming`, `OpenAI`, `Anthropic` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 지침입니다.
:::

# Instructor: 구조화된 LLM 출력

## 이 스킬을 사용하는 경우

다음이 필요할 때 Instructor를 사용하세요.
- **LLM 응답에서 구조화된 데이터 추출**
- **Pydantic 스키마에 따라 출력을 자동으로 검증**
- **자동 오류 처리를 통한 실패한 추출 재시도**
- **타입 안전성과 검증을 적용한 복잡한 JSON 파싱**
- **실시간 처리를 위한 부분 결과 스트리밍**
- **일관된 API로 여러 LLM 공급자 지원**

**GitHub 스타**: 15,000+ | **실전 검증**: 100,000명 이상의 개발자

## 설치

```bash
# Base installation
pip install instructor

# With specific providers
pip install "instructor[anthropic]"  # Anthropic Claude
pip install "instructor[openai]"     # OpenAI
pip install "instructor[all]"        # All providers
```

## 빠른 시작

### 기본 예제: 사용자 데이터 추출

```python
import instructor
from pydantic import BaseModel
from anthropic import Anthropic

# Define output structure
class User(BaseModel):
    name: str
    age: int
    email: str

# Create instructor client
client = instructor.from_anthropic(Anthropic())

# Extract structured data
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

### OpenAI 사용

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

### 1. 응답 모델(Pydantic)

응답 모델은 LLM 출력의 구조와 검증 규칙을 정의합니다.

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

**장점:**
- Python 타입 힌트를 통한 타입 안전성
- 자동 검증(`word_count > 0`)
- Field 설명을 통한 자체 문서화
- IDE 자동 완성 지원

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
    discount: Optional[float] = None  # Optional
    description: str = Field(default="No description")  # Default value

# LLM doesn't need to provide discount or description
```

#### 제약 조건을 위한 열거형

```python
from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class Review(BaseModel):
    text: str
    sentiment: Sentiment  # Only these 3 values allowed

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

### 2. 검증

Pydantic은 LLM 출력을 자동으로 검증합니다. 검증에 실패하면 Instructor가 재시도합니다.

#### 내장 검증기

```python
from pydantic import Field, EmailStr, HttpUrl

class Contact(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    age: int = Field(ge=0, le=120)  # 0 <= age <= 120
    email: EmailStr  # Validates email format
    website: HttpUrl  # Validates URL format

# If LLM provides invalid data, Instructor retries automatically
```

#### 사용자 지정 검증기

```python
from pydantic import field_validator

class Event(BaseModel):
    name: str
    date: str
    attendees: int

    @field_validator('date')
    def validate_date(cls, v):
        """Ensure date is in YYYY-MM-DD format."""
        import re
        if not re.match(r'\d{4}-\d{2}-\d{2}', v):
            raise ValueError('Date must be YYYY-MM-DD format')
        return v

    @field_validator('attendees')
    def validate_attendees(cls, v):
        """Ensure positive attendees."""
        if v < 1:
            raise ValueError('Must have at least 1 attendee')
        return v
```

#### 모델 수준 검증

```python
from pydantic import model_validator

class DateRange(BaseModel):
    start_date: str
    end_date: str

    @model_validator(mode='after')
    def check_dates(self):
        """Ensure end_date is after start_date."""
        from datetime import datetime
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')

        if end < start:
            raise ValueError('end_date must be after start_date')
        return self
```

### 3. 자동 재시도

Instructor는 검증에 실패하면 자동으로 재시도하며, LLM에 오류 피드백을 제공합니다.

```python
# Retries up to 3 times if validation fails
user = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": "Extract user from: John, age unknown"
    }],
    response_model=User,
    max_retries=3  # Default is 3
)

# If age can't be extracted, Instructor tells the LLM:
# "Validation error: age - field required"
# LLM tries again with better extraction
```

**작동 방식:**
1. LLM이 출력을 생성합니다.
2. Pydantic이 검증합니다.
3. 유효하지 않으면 오류 메시지를 LLM에 다시 보냅니다.
4. LLM이 오류 피드백을 반영해 다시 시도합니다.
5. `max_retries`까지 반복합니다.

### 4. 스트리밍

실시간 처리를 위해 부분 결과를 스트리밍합니다.

#### 부분 객체 스트리밍

```python
from instructor import Partial

class Story(BaseModel):
    title: str
    content: str
    tags: list[str]

# Stream partial updates as LLM generates
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
    # Update UI in real-time
```

#### 이터러블 스트리밍

```python
class Task(BaseModel):
    title: str
    priority: str

# Stream list items as they're generated
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
    # Process each task as it arrives
```

## 공급자 설정

### Anthropic Claude

```python
import instructor
from anthropic import Anthropic

client = instructor.from_anthropic(
    Anthropic(api_key="your-api-key")
)

# Use with Claude models
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

### 로컬 모델(Ollama)

```python
from openai import OpenAI

# Point to local Ollama server
client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"  # Required but ignored
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

### 패턴 2: 분류

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

### 패턴 3: 여러 엔터티 추출

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

### 패턴 4: 구조화된 분석

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

### 패턴 5: 일괄 처리

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

### 유니온 타입

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
    content: Union[TextContent, ImageContent]  # Either type

# LLM chooses appropriate type based on content
```

### 동적 모델

```python
from pydantic import create_model

# Create model at runtime
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

### 사용자 지정 모드

```python
# For providers without native structured outputs
client = instructor.from_anthropic(
    Anthropic(),
    mode=instructor.Mode.JSON  # JSON mode
)

# Available modes:
# - Mode.ANTHROPIC_TOOLS (recommended for Claude)
# - Mode.JSON (fallback)
# - Mode.TOOLS (OpenAI tools)
```

### 컨텍스트 관리

```python
# Single-use client
with instructor.from_anthropic(Anthropic()) as client:
    result = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=[...],
        response_model=YourModel
    )
    # Client closed automatically
```

## 오류 처리

### 검증 오류 처리

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
    # Handle gracefully

except Exception as e:
    print(f"API error: {e}")
```

### 사용자 지정 오류 메시지

```python
class ValidatedUser(BaseModel):
    name: str = Field(description="Full name, 2-100 characters")
    age: int = Field(description="Age between 0 and 120", ge=0, le=120)
    email: EmailStr = Field(description="Valid email address")

    class Config:
        # Custom error messages
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
# ❌ Bad: Vague
class Product(BaseModel):
    name: str
    price: float

# ✅ Good: Descriptive
class Product(BaseModel):
    name: str = Field(description="Product name from the text")
    price: float = Field(description="Price in USD, without currency symbol")
```

### 2. 적절한 검증 사용

```python
# ✅ Good: Constrain values
class Rating(BaseModel):
    score: int = Field(ge=1, le=5, description="Rating from 1 to 5 stars")
    review: str = Field(min_length=10, description="Review text, at least 10 chars")
```

### 3. 프롬프트에 예시 제공

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

### 4. 고정된 범주에 열거형 사용

```python
# ✅ Good: Enum ensures valid values
class Status(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

class Application(BaseModel):
    status: Status  # LLM must choose from enum
```

### 5. 누락된 데이터를 유연하게 처리

```python
class PartialData(BaseModel):
    required_field: str
    optional_field: Optional[str] = None
    default_field: str = "default_value"

# LLM only needs to provide required_field
```

## 대안과 비교

| 기능 | Instructor | 수동 JSON | LangChain | DSPy |
|---------|------------|-------------|-----------|------|
| 타입 안전성 | ✅ 예 | ❌ 아니요 | ⚠️ 부분 지원 | ✅ 예 |
| 자동 검증 | ✅ 예 | ❌ 아니요 | ❌ 아니요 | ⚠️ 제한적 |
| 자동 재시도 | ✅ 예 | ❌ 아니요 | ❌ 아니요 | ✅ 예 |
| 스트리밍 | ✅ 예 | ❌ 아니요 | ✅ 예 | ❌ 아니요 |
| 여러 공급자 | ✅ 예 | ⚠️ 수동 | ✅ 예 | ✅ 예 |
| 학습 곡선 | 낮음 | 낮음 | 중간 | 높음 |

**Instructor를 선택할 경우:**
- 구조화되고 검증된 출력이 필요할 때
- 타입 안전성과 IDE 지원을 원할 때
- 자동 재시도가 필요할 때
- 데이터 추출 시스템을 구축할 때

**대안을 선택할 경우:**
- DSPy: 프롬프트 최적화가 필요할 때
- LangChain: 복잡한 체인을 구축할 때
- 수동 방식: 간단한 일회성 추출일 때

## 리소스

- **문서**: https://python.useinstructor.com
- **GitHub**: https://github.com/jxnl/instructor (스타 15k+)
- **쿡북**: https://python.useinstructor.com/examples
- **Discord**: 커뮤니티 지원 이용 가능

## 함께 보기

- `references/validation.md` - 고급 검증 패턴
- `references/providers.md` - 공급자별 설정
- `references/examples.md` - 실제 사용 사례

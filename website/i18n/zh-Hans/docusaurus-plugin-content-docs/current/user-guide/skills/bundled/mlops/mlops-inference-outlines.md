---
title: "Outlines — Outlines：结构化 JSON/正则/Pydantic LLM 生成"
sidebar_label: "Outlines"
description: "Outlines：结构化 JSON/正则/Pydantic LLM 生成"
---
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
Outlines: structured JSON/regex/Pydantic LLM generation.
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
## 技能元数据
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
| | |
|---|---|
| | |
| 路径 | `skills/mlops/inference/outlines` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `outlines`, `transformers`, `vllm`, `pydantic` |
| 标签 | `提示工程、Outlines、结构化生成、JSON Schema、Pydantic、本地模型、基于语法的生成、vLLM、Transformers、类型安全` |
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
## 参考：完整 SKILL.md
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. 这是代理在技能激活时看到的指令。
:::
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
## When to Use This Skill
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
Use Outlines when you need to:
- **Guarantee valid JSON/XML/code** structure during generation
- **Use Pydantic models** for type-safe outputs
- **Support local models** (Transformers, llama.cpp, vLLM)
- **Maximize inference speed** with zero-overhead structured generation
- **Generate against JSON schemas** automatically
- **Control token sampling** at the grammar level
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
**GitHub Stars**: 8,000+ | **From**: dottxt.ai (formerly .txt)
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
## Installation
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```bash
# Outlines
pip install outlines
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
pip install outlines transformers  # Hugging Face models
pip install outlines llama-cpp-python  # llama.cpp
pip install outlines vllm  # vLLM for high-throughput
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
## Quick Start
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
### Basic Example: Classification
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```python
import outlines
from typing import Literal
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
prompt = "Sentiment of 'This product is amazing!': "
generator = outlines.generate.choice(model, ["positive", "negative", "neutral"])
sentiment = generator(prompt)
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
print(sentiment)  # "positive" (guaranteed one of these)
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
### With Pydantic Models
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```python
from pydantic import BaseModel
import outlines
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
class User(BaseModel):
    name: str
    age: int
    email: str
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
prompt = "Extract user: John Doe, 30 years old, john@example.com"
generator = outlines.generate.json(model, User)
user = generator(prompt)
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
print(user.name)   # "John Doe"
print(user.age)    # 30
print(user.email)  # "john@example.com"
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
## Core Concepts
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
### 1. Constrained Token Sampling
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
Outlines uses Finite State Machines (FSM) to constrain token generation at the logit level.
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
**How it works:**
1. Convert schema (JSON/Pydantic/regex) to context-free grammar (CFG)
2. Transform CFG into Finite State Machine (FSM)
3. Filter invalid tokens at each step during generation
4. Fast-forward when only one valid token exists
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
**Benefits:**
- **Zero overhead**: Filtering happens at token level
- **Speed improvement**: Fast-forward through deterministic paths
- **Guaranteed validity**: Invalid outputs impossible
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```python
import outlines
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
class Person(BaseModel):
    name: str
    age: int
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
# Outlines
# Outlines
# Outlines
# Outlines
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
generator = outlines.generate.json(model, Person)
result = generator("Generate person: Alice, 25")
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
### 2. Structured Generators
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
Outlines provides specialized generators for different output types.
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
#### Choice Generator
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```python
# Outlines
generator = outlines.generate.choice(
    model,
    ["positive", "negative", "neutral"]
)
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
sentiment = generator("Review: This is great!")
# Outlines
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
#### JSON Generator
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```python
from pydantic import BaseModel
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
class Product(BaseModel):
    name: str
    price: float
    in_stock: bool
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
generator = outlines.generate.json(model, Product)
product = generator("Extract: iPhone 15, $999, available")
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
print(type(product))  # <class '__main__.Product'>
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
#### Regex Generator
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```python
# Outlines
generator = outlines.generate.regex(
    model,
    r"[0-9]{3}-[0-9]{3}-[0-9]{4}"  # Phone number pattern
)
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
phone = generator("Generate phone number:")
# Outlines
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
#### Integer/Float Generators
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```python
# Outlines
int_generator = outlines.generate.integer(model)
age = int_generator("Person's age:")  # Guaranteed integer
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
float_generator = outlines.generate.float(model)
price = float_generator("Product price:")  # Guaranteed float
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
### 3. Model Backends
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
Outlines supports multiple local and API-based backends.
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
#### Transformers (Hugging Face)
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```python
import outlines
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
model = outlines.models.transformers(
    "microsoft/Phi-3-mini-4k-instruct",
    device="cuda"  # Or "cpu"
)
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
generator = outlines.generate.json(model, YourModel)
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
#### llama.cpp
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```python
# Outlines
model = outlines.models.llamacpp(
    "./models/llama-3.1-8b-instruct.Q4_K_M.gguf",
    n_gpu_layers=35
)
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
generator = outlines.generate.json(model, YourModel)
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
#### vLLM (High Throughput)
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```python
# Outlines
model = outlines.models.vllm(
    "meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=2  # Multi-GPU
)
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
generator = outlines.generate.json(model, YourModel)
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
#### OpenAI (Limited Support)
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```python
# Outlines
model = outlines.models.openai(
    "gpt-4o-mini",
    api_key="your-api-key"
)
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
generator = outlines.generate.json(model, YourModel)
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
### 4. Pydantic Integration
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
Outlines has first-class Pydantic support with automatic schema translation.
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
#### Basic Models
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```python
from pydantic import BaseModel, Field
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
class Article(BaseModel):
    title: str = Field(description="Article title")
    author: str = Field(description="Author name")
    word_count: int = Field(description="Number of words", gt=0)
    tags: list[str] = Field(description="List of tags")
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")
generator = outlines.generate.json(model, Article)
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
article = generator("Generate article about AI")
print(article.title)
print(article.word_count)  # Guaranteed > 0
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
#### Nested Models
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```python
class Address(BaseModel):
    street: str
    city: str
    country: str
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
class Person(BaseModel):
    name: str
    age: int
    address: Address  # Nested model
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
generator = outlines.generate.json(model, Person)
person = generator("Generate person in New York")
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
print(person.address.city)  # "New York"
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
#### Enums and Literals
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```python
from enum import Enum
from typing import Literal
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
class Status(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
class Application(BaseModel):
    applicant: str
    status: Status  # Must be one of enum values
    priority: Literal["low", "medium", "high"]  # Must be one of literals
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
generator = outlines.generate.json(model, Application)
app = generator("Generate application")
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
print(app.status)  # Status.PENDING (or APPROVED/REJECTED)
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
## Common Patterns
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
### Pattern 1: Data Extraction
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```python
from pydantic import BaseModel
import outlines
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
class CompanyInfo(BaseModel):
    name: str
    founded_year: int
    industry: str
    employees: int
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")
generator = outlines.generate.json(model, CompanyInfo)
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
text = """
Apple Inc. was founded in 1976 in the technology industry.
The company employs approximately 164,000 people worldwide.
"""
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
prompt = f"Extract company information:\n{text}\n\nCompany:"
company = generator(prompt)
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
print(f"Name: {company.name}")
print(f"Founded: {company.founded_year}")
print(f"Industry: {company.industry}")
print(f"Employees: {company.employees}")
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
### Pattern 2: Classification
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```python
from typing import Literal
import outlines
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
generator = outlines.generate.choice(model, ["spam", "not_spam"])
result = generator("Email: Buy now! 50% off!")
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
categories = ["technology", "business", "sports", "entertainment"]
category_gen = outlines.generate.choice(model, categories)
category = category_gen("Article: Apple announces new iPhone...")
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
class Classification(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
classifier = outlines.generate.json(model, Classification)
result = classifier("Review: This product is okay, nothing special")
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
### Pattern 3: Structured Forms
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```python
class UserProfile(BaseModel):
    full_name: str
    age: int
    email: str
    phone: str
    country: str
    interests: list[str]
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")
generator = outlines.generate.json(model, UserProfile)
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
prompt = """
Extract user profile from:
Name: Alice Johnson
Age: 28
Email: alice@example.com
Phone: 555-0123
Country: USA
Interests: hiking, photography, cooking
"""
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
profile = generator(prompt)
print(profile.full_name)
print(profile.interests)  # ["hiking", "photography", "cooking"]
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
### Pattern 4: Multi-Entity Extraction
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```python
class Entity(BaseModel):
    name: str
    type: Literal["PERSON", "ORGANIZATION", "LOCATION"]
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
class DocumentEntities(BaseModel):
    entities: list[Entity]
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")
generator = outlines.generate.json(model, DocumentEntities)
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
text = "Tim Cook met with Satya Nadella at Microsoft headquarters in Redmond."
prompt = f"Extract entities from: {text}"
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
result = generator(prompt)
for entity in result.entities:
    print(f"{entity.name} ({entity.type})")
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
### Pattern 5: Code Generation
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```python
class PythonFunction(BaseModel):
    function_name: str
    parameters: list[str]
    docstring: str
    body: str
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")
generator = outlines.generate.json(model, PythonFunction)
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
prompt = "Generate a Python function to calculate factorial"
func = generator(prompt)
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
print(f"def {func.function_name}({', '.join(func.parameters)}):")
print(f'    """{func.docstring}"""')
print(f"    {func.body}")
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
### Pattern 6: Batch Processing
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```python
def batch_extract(texts: list[str], schema: type[BaseModel]):
    """Extract structured data from multiple texts."""
    model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")
    generator = outlines.generate.json(model, schema)
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
    results = []
    for text in texts:
        result = generator(f"Extract from: {text}")
        results.append(result)
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
    return results
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
class Person(BaseModel):
    name: str
    age: int
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
texts = [
    "John is 30 years old",
    "Alice is 25 years old",
    "Bob is 40 years old"
]
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
people = batch_extract(texts, Person)
for person in people:
    print(f"{person.name}: {person.age}")
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
## Backend Configuration
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
### Transformers
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```python
import outlines
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
model = outlines.models.transformers(
    "microsoft/Phi-3-mini-4k-instruct",
    device="cuda",
    model_kwargs={"torch_dtype": "float16"}
)
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
model = outlines.models.transformers("meta-llama/Llama-3.1-8B-Instruct")
model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.3")
model = outlines.models.transformers("Qwen/Qwen2.5-7B-Instruct")
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
### llama.cpp
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```python
# Outlines
model = outlines.models.llamacpp(
    "./models/llama-3.1-8b.Q4_K_M.gguf",
    n_ctx=4096,         # Context window
    n_gpu_layers=35,    # GPU layers
    n_threads=8         # CPU threads
)
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
model = outlines.models.llamacpp(
    "./models/model.gguf",
    n_gpu_layers=-1  # All layers on GPU
)
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
### vLLM (Production)
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```python
# Outlines
model = outlines.models.vllm("meta-llama/Llama-3.1-8B-Instruct")
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
model = outlines.models.vllm(
    "meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4  # 4 GPUs
)
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
model = outlines.models.vllm(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization="awq"  # Or "gptq"
)
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
## Best Practices
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
### 1. Use Specific Types
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```python
# Outlines
class Product(BaseModel):
    name: str
    price: float  # Not str
    quantity: int  # Not str
    in_stock: bool  # Not str
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
class Product(BaseModel):
    name: str
    price: str  # Should be float
    quantity: str  # Should be int
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
### 2. Add Constraints
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```python
from pydantic import Field
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
class User(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    age: int = Field(ge=0, le=120)
    email: str = Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
class User(BaseModel):
    name: str
    age: int
    email: str
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
### 3. Use Enums for Categories
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```python
# Outlines
class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
class Task(BaseModel):
    title: str
    priority: Priority
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
class Task(BaseModel):
    title: str
    priority: str  # Can be anything
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
### 4. Provide Context in Prompts
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```python
# Outlines
prompt = """
Extract product information from the following text.
Text: iPhone 15 Pro costs $999 and is currently in stock.
Product:
"""
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
prompt = "iPhone 15 Pro costs $999 and is currently in stock."
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
### 5. Handle Optional Fields
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
```python
from typing import Optional
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
class Article(BaseModel):
    title: str  # Required
    author: Optional[str] = None  # Optional
    date: Optional[str] = None  # Optional
    tags: list[str] = []  # Default empty list
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
# Outlines
```
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
## Comparison to Alternatives
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
| Feature | Outlines | Instructor | Guidance | LMQL |
|---------|----------|------------|----------|------|
| Pydantic Support | ✅ Native | ✅ Native | ❌ No | ❌ No |
| JSON Schema | ✅ Yes | ✅ Yes | ⚠️ Limited | ✅ Yes |
| Regex Constraints | ✅ Yes | ❌ No | ✅ Yes | ✅ Yes |
| Local Models | ✅ Full | ⚠️ Limited | ✅ Full | ✅ Full |
| API Models | ⚠️ Limited | ✅ Full | ✅ Full | ✅ Full |
| Zero Overhead | ✅ Yes | ❌ No | ⚠️ Partial | ✅ Yes |
| Automatic Retrying | ❌ No | ✅ Yes | ❌ No | ❌ No |
| Learning Curve | Low | Low | Low | High |
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
**When to choose Outlines:**
- Using local models (Transformers, llama.cpp, vLLM)
- Need maximum inference speed
- Want Pydantic model support
- Require zero-overhead structured generation
- Control token sampling process
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
**When to choose alternatives:**
- Instructor: Need API models with automatic retrying
- Guidance: Need token healing and complex workflows
- LMQL: Prefer declarative query syntax
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
## Performance Characteristics
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
**Speed:**
- **Zero overhead**: Structured generation as fast as unconstrained
- **Fast-forward optimization**: Skips deterministic tokens
- **1.2-2x faster** than post-generation validation approaches
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
**Memory:**
- FSM compiled once per schema (cached)
- Minimal runtime overhead
- Efficient with vLLM for high throughput
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
**Accuracy:**
- **100% valid outputs** (guaranteed by FSM)
- No retry loops needed
- Deterministic token filtering
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
## Resources
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
- **Documentation**: https://outlines-dev.github.io/outlines
- **GitHub**: https://github.com/outlines-dev/outlines (8k+ stars)
- **Discord**: https://discord.gg/R9DSu34mGd
- **Blog**: https://blog.dottxt.co
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
## See Also
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
- `references/json_generation.md` - Comprehensive JSON and Pydantic patterns
- `references/backends.md` - Backend-specific configuration
- `references/examples.md` - Production-ready examples
Outlines：结构化 JSON/正则/Pydantic LLM 生成。
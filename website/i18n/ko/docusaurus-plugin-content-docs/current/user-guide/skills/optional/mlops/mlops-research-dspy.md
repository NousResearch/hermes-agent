---
title: "Dspy — DSPy: 선언적 LM 프로그래밍, 프롬프트 자동 최적화, RAG"
sidebar_label: "Dspy"
description: "DSPy: 선언적 LM 프로그래밍, 프롬프트 자동 최적화, RAG"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Dspy

DSPy: 선언적 언어 모델(LM) 프로그램, 프롬프트 자동 최적화, RAG(검색 증강 생성).

## Skill metadata

| | |
|---|---|
| Source | Optional — `hermes skills install official/mlops/dspy`로 설치 |
| Path | `optional-skills/mlops/research/dspy` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `dspy`, `openai`, `anthropic` |
| Platforms | linux, macos, windows |
| Tags | `Prompt Engineering`, `DSPy`, `Declarative Programming`, `RAG`, `Agents`, `Prompt Optimization`, `LM Programming`, `Stanford NLP`, `Automatic Optimization`, `Modular AI` |

## Reference: full SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# DSPy: 선언적 언어 모델 프로그래밍 (Declarative Language Model Programming)

## 이 스킬을 사용하는 경우

다음과 같은 상황에서 DSPy를 사용하세요:
- 여러 구성 요소와 워크플로우를 갖춘 **복잡한 AI 시스템을 구축**할 때
- 수동적인 프롬프트 엔지니어링 대신 **선언적으로 LM을 프로그래밍**할 때
- 데이터 기반 방법을 사용하여 **프롬프트를 자동으로 최적화**할 때
- 유지 관리와 이식이 쉬운 **모듈식 AI 파이프라인을 생성**할 때
- 최적화기(Optimizer)를 사용하여 **모델 출력을 체계적으로 개선**할 때
- 더 나은 안정성을 갖춘 **RAG 시스템, 에이전트 또는 분류기(classifier)를 구축**할 때

**GitHub 별(Stars)**: 22,000+ | **제작**: Stanford NLP

## 설치

```bash
# 안정적인 릴리스(Stable release)
pip install dspy

# 최신 개발 버전
pip install git+https://github.com/stanfordnlp/dspy.git

# 특정 LM 제공자와 함께 설치
pip install dspy[openai]        # OpenAI
pip install dspy[anthropic]     # Anthropic Claude
pip install dspy[all]           # 모든 제공자
```

## 빠른 시작

### 기본 예제: 질문 답변 (QA)

```python
import dspy

# 언어 모델 설정
lm = dspy.Claude(model="claude-sonnet-4-5-20250929")
dspy.settings.configure(lm=lm)

# 시그니처 정의 (입력 → 출력)
class QA(dspy.Signature):
    """짧은 사실적 답변으로 질문에 대답합니다."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="보통 1단어에서 5단어 사이")

# 모듈 생성
qa = dspy.Predict(QA)

# 사용
response = qa(question="프랑스의 수도는 어디입니까?")
print(response.answer)  # "파리"
```

### 사고의 사슬 추론 (Chain of Thought Reasoning)

```python
import dspy

lm = dspy.Claude(model="claude-sonnet-4-5-20250929")
dspy.settings.configure(lm=lm)

# 더 나은 추론을 위해 ChainOfThought 사용
class MathProblem(dspy.Signature):
    """수학 문장제 문제를 풉니다."""
    problem = dspy.InputField()
    answer = dspy.OutputField(desc="숫자로 된 답변")

# ChainOfThought는 추론 단계를 자동으로 생성합니다
cot = dspy.ChainOfThought(MathProblem)

response = cot(problem="John이 사과를 5개 가지고 있고 Mary에게 2개를 준다면, 그는 몇 개의 사과를 가지게 됩니까?")
print(response.rationale)  # 추론 단계를 보여줍니다
print(response.answer)     # "3"
```

## 핵심 개념

### 1. 시그니처 (Signatures)

시그니처는 AI 작업의 구조(입력 → 출력)를 정의합니다:

```python
# 인라인 시그니처 (간단함)
qa = dspy.Predict("question -> answer")

# 클래스 시그니처 (상세함)
class Summarize(dspy.Signature):
    """텍스트를 핵심 요점으로 요약합니다."""
    text = dspy.InputField()
    summary = dspy.OutputField(desc="글머리 기호, 3-5개 항목")

summarizer = dspy.ChainOfThought(Summarize)
```

**사용 기준:**
- **인라인**: 빠른 프로토타이핑, 간단한 작업
- **클래스**: 복잡한 작업, 타입 힌트, 더 나은 문서화

### 2. 모듈 (Modules)

모듈은 입력을 출력으로 변환하는 재사용 가능한 컴포넌트입니다:

#### dspy.Predict
기본적인 예측 모듈입니다:

```python
predictor = dspy.Predict("context, question -> answer")
result = predictor(context="파리는 프랑스의 수도입니다.",
                   question="수도는 어디입니까?")
```

#### dspy.ChainOfThought
답변하기 전에 추론 단계를 생성합니다:

```python
cot = dspy.ChainOfThought("question -> answer")
result = cot(question="하늘은 왜 파란색인가요?")
print(result.rationale)  # 추론 단계
print(result.answer)     # 최종 답변
```

#### dspy.ReAct
도구(tools)를 사용하는 에이전트와 유사한 추론입니다:

```python
from dspy.predict import ReAct

class SearchQA(dspy.Signature):
    """검색을 사용하여 질문에 대답합니다."""
    question = dspy.InputField()
    answer = dspy.OutputField()

def search_tool(query: str) -> str:
    """위키백과를 검색합니다."""
    # 검색 구현
    return results

react = ReAct(SearchQA, tools=[search_tool])
result = react(question="파이썬은 언제 만들어졌나요?")
```

#### dspy.ProgramOfThought
추론을 위해 코드를 생성하고 실행합니다:

```python
pot = dspy.ProgramOfThought("question -> answer")
result = pot(question="240의 15%는 얼마입니까?")
# 다음 코드를 생성합니다: answer = 240 * 0.15
```

### 3. 최적화기 (Optimizers)

최적화기는 학습 데이터를 사용하여 모듈을 자동으로 개선합니다:

#### BootstrapFewShot
예시로부터 학습합니다:

```python
from dspy.teleprompt import BootstrapFewShot

# 학습 데이터
trainset = [
    dspy.Example(question="2+2는 무엇입니까?", answer="4").with_inputs("question"),
    dspy.Example(question="3+5는 무엇입니까?", answer="8").with_inputs("question"),
]

# 지표(Metric) 정의
def validate_answer(example, pred, trace=None):
    return example.answer == pred.answer

# 최적화
optimizer = BootstrapFewShot(metric=validate_answer, max_bootstrapped_demos=3)
optimized_qa = optimizer.compile(qa, trainset=trainset)

# 이제 optimized_qa의 성능이 더 좋아졌습니다!
```

#### MIPRO (Most Important Prompt Optimization)
프롬프트를 반복적으로 개선합니다:

```python
from dspy.teleprompt import MIPRO

optimizer = MIPRO(
    metric=validate_answer,
    num_candidates=10,
    init_temperature=1.0
)

optimized_cot = optimizer.compile(
    cot,
    trainset=trainset,
    num_trials=100
)
```

#### BootstrapFinetune
모델 파인튜닝을 위한 데이터셋을 생성합니다:

```python
from dspy.teleprompt import BootstrapFinetune

optimizer = BootstrapFinetune(metric=validate_answer)
optimized_module = optimizer.compile(qa, trainset=trainset)

# 파인튜닝을 위해 학습 데이터를 내보냅니다.
```

### 4. 복잡한 시스템 구축 (Building Complex Systems)

#### 다단계 파이프라인 (Multi-Stage Pipeline)

```python
import dspy

class MultiHopQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=3)
        self.generate_query = dspy.ChainOfThought("question -> search_query")
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        # 1단계: 검색 쿼리 생성
        search_query = self.generate_query(question=question).search_query

        # 2단계: 컨텍스트 검색
        passages = self.retrieve(search_query).passages
        context = "\n".join(passages)

        # 3단계: 답변 생성
        answer = self.generate_answer(context=context, question=question).answer
        return dspy.Prediction(answer=answer, context=context)

# 파이프라인 사용
qa_system = MultiHopQA()
result = qa_system(question="영화 블레이드 러너에 영감을 준 책을 쓴 사람은 누구인가요?")
```

#### 최적화를 포함한 RAG 시스템

```python
import dspy
from dspy.retrieve.chromadb_rm import ChromadbRM

# 리트리버(Retriever) 설정
retriever = ChromadbRM(
    collection_name="documents",
    persist_directory="./chroma_db"
)

class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

# 생성 및 최적화
rag = RAG()

# 학습 데이터로 최적화
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(metric=validate_answer)
optimized_rag = optimizer.compile(rag, trainset=trainset)
```

## LM 제공자(Provider) 설정

### Anthropic Claude

```python
import dspy

lm = dspy.Claude(
    model="claude-sonnet-4-5-20250929",
    api_key="your-api-key",  # 또는 ANTHROPIC_API_KEY 환경 변수 설정
    max_tokens=1000,
    temperature=0.7
)
dspy.settings.configure(lm=lm)
```

### OpenAI

```python
lm = dspy.OpenAI(
    model="gpt-4",
    api_key="your-api-key",
    max_tokens=1000
)
dspy.settings.configure(lm=lm)
```

### 로컬 모델 (Ollama)

```python
lm = dspy.OllamaLocal(
    model="llama3.1",
    base_url="http://localhost:11434"
)
dspy.settings.configure(lm=lm)
```

### 다중 모델 (Multiple Models)

```python
# 작업에 따라 다른 모델 사용
cheap_lm = dspy.OpenAI(model="gpt-3.5-turbo")
strong_lm = dspy.Claude(model="claude-sonnet-4-5-20250929")

# 검색에는 저렴한 모델, 추론에는 강력한 모델 사용
with dspy.settings.context(lm=cheap_lm):
    context = retriever(question)

with dspy.settings.context(lm=strong_lm):
    answer = generator(context=context, question=question)
```

## 일반적인 패턴 (Common Patterns)

### 패턴 1: 구조화된 출력 (Structured Output)

```python
from pydantic import BaseModel, Field

class PersonInfo(BaseModel):
    name: str = Field(description="성명")
    age: int = Field(description="나이")
    occupation: str = Field(description="현재 직업")

class ExtractPerson(dspy.Signature):
    """텍스트에서 개인 정보를 추출합니다."""
    text = dspy.InputField()
    person: PersonInfo = dspy.OutputField()

extractor = dspy.TypedPredictor(ExtractPerson)
result = extractor(text="John Doe는 35세의 소프트웨어 엔지니어입니다.")
print(result.person.name)  # "John Doe"
print(result.person.age)   # 35
```

### 패턴 2: 단언 기반 최적화 (Assertion-Driven Optimization)

```python
import dspy
from dspy.primitives.assertions import assert_transform_module, backtrack_handler

class MathQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.solve = dspy.ChainOfThought("problem -> solution: float")

    def forward(self, problem):
        solution = self.solve(problem=problem).solution

        # 정답이 숫자인지 단언(Assert)
        dspy.Assert(
            isinstance(float(solution), float),
            "정답은 숫자여야 합니다",
            backtrack=backtrack_handler
        )

        return dspy.Prediction(solution=solution)
```

### 패턴 3: 자가 일관성 (Self-Consistency)

```python
import dspy
from collections import Counter

class ConsistentQA(dspy.Module):
    def __init__(self, num_samples=5):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")
        self.num_samples = num_samples

    def forward(self, question):
        # 여러 답변 생성
        answers = []
        for _ in range(self.num_samples):
            result = self.qa(question=question)
            answers.append(result.answer)

        # 가장 흔한 답변 반환
        most_common = Counter(answers).most_common(1)[0][0]
        return dspy.Prediction(answer=most_common)
```

### 패턴 4: 재순위화를 포함한 검색 (Retrieval with Reranking)

```python
class RerankedRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=10)
        self.rerank = dspy.Predict("question, passage -> relevance_score: float")
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        # 후보 검색
        passages = self.retrieve(question).passages

        # 구절(Passages) 재순위화
        scored = []
        for passage in passages:
            score = float(self.rerank(question=question, passage=passage).relevance_score)
            scored.append((score, passage))

        # 상위 3개 취합
        top_passages = [p for _, p in sorted(scored, reverse=True)[:3]]
        context = "\n\n".join(top_passages)

        # 답변 생성
        return self.answer(context=context, question=question)
```

## 평가 및 지표 (Evaluation and Metrics)

### 사용자 정의 지표 (Custom Metrics)

```python
def exact_match(example, pred, trace=None):
    """정확한 일치 여부 확인 지표."""
    return example.answer.lower() == pred.answer.lower()

def f1_score(example, pred, trace=None):
    """텍스트 겹침에 대한 F1 점수."""
    pred_tokens = set(pred.answer.lower().split())
    gold_tokens = set(example.answer.lower().split())

    if not pred_tokens:
        return 0.0

    precision = len(pred_tokens & gold_tokens) / len(pred_tokens)
    recall = len(pred_tokens & gold_tokens) / len(gold_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)
```

### 평가 (Evaluation)

```python
from dspy.evaluate import Evaluate

# 평가기(Evaluator) 생성
evaluator = Evaluate(
    devset=testset,
    metric=exact_match,
    num_threads=4,
    display_progress=True
)

# 모델 평가
score = evaluator(qa_system)
print(f"정확도: {score}")

# 최적화 전후 비교
score_before = evaluator(qa)
score_after = evaluator(optimized_qa)
print(f"향상도: {score_after - score_before:.2%}")
```

## 모범 사례 (Best Practices)

### 1. 간단하게 시작하고, 반복하기

```python
# Predict로 시작
qa = dspy.Predict("question -> answer")

# 필요시 추론(reasoning) 추가
qa = dspy.ChainOfThought("question -> answer")

# 데이터가 확보되면 최적화 추가
optimized_qa = optimizer.compile(qa, trainset=data)
```

### 2. 설명이 포함된 시그니처 사용

```python
# ❌ 나쁜 예: 모호함
class Task(dspy.Signature):
    input = dspy.InputField()
    output = dspy.OutputField()

# ✅ 좋은 예: 구체적임
class SummarizeArticle(dspy.Signature):
    """뉴스 기사를 3-5개의 주요 요점으로 요약합니다."""
    article = dspy.InputField(desc="기사 전체 텍스트")
    summary = dspy.OutputField(desc="글머리 기호, 3-5개 항목")
```

### 3. 대표성을 지닌 데이터로 최적화

```python
# 다양한 학습 예제 생성
trainset = [
    dspy.Example(question="사실적 질문", answer="...").with_inputs("question"),
    dspy.Example(question="추론 질문", answer="...").with_inputs("question"),
    dspy.Example(question="계산 질문", answer="...").with_inputs("question"),
]

# 지표 측정 시 검증(validation) 셋을 사용
def metric(example, pred, trace=None):
    return example.answer in pred.answer
```

### 4. 최적화된 모델 저장 및 불러오기

```python
# 저장
optimized_qa.save("models/qa_v1.json")

# 불러오기
loaded_qa = dspy.ChainOfThought("question -> answer")
loaded_qa.load("models/qa_v1.json")
```

### 5. 모니터링 및 디버깅

```python
# 추적(Tracing) 활성화
dspy.settings.configure(lm=lm, trace=[])

# 예측 실행
result = qa(question="...")

# 추적 내용 검사
for call in dspy.settings.trace:
    print(f"프롬프트: {call['prompt']}")
    print(f"응답: {call['response']}")
```

## 다른 접근 방식과의 비교

| 기능 | 수동 프롬프팅 (Manual) | LangChain | DSPy |
|---------|-----------------|-----------|------|
| 프롬프트 엔지니어링 | 수동 | 수동 | 자동 |
| 최적화 | 시행착오 | 없음 | 데이터 기반 |
| 모듈성 | 낮음 | 중간 | 높음 |
| 타입 안전성 | 없음 | 제한적 | 있음 (시그니처) |
| 이식성 | 낮음 | 중간 | 높음 |
| 학습 곡선 | 낮음 | 중간 | 중간-높음 |

**DSPy를 선택해야 하는 경우:**
- 학습 데이터가 있거나 생성할 수 있을 때
- 체계적인 프롬프트 개선이 필요할 때
- 다단계 복합 시스템을 구축할 때
- 서로 다른 LM 전반에 걸쳐 최적화하고 싶을 때

**대안을 선택해야 하는 경우:**
- 빠른 프로토타이핑이 필요할 때 (수동 프롬프팅)
- 기존 도구를 사용하여 간단한 체인을 구성할 때 (LangChain)
- 커스텀 최적화 로직이 필요할 때

## 리소스

- **문서**: https://dspy.ai
- **GitHub**: https://github.com/stanfordnlp/dspy (22k+ stars)
- **Discord**: https://discord.gg/XCGy2WDCQB
- **Twitter**: @DSPyOSS
- **논문**: "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines"

## 참고 자료 (See Also)

- `references/modules.md` - 모듈 상세 가이드 (Predict, ChainOfThought, ReAct, ProgramOfThought)
- `references/optimizers.md` - 최적화 알고리즘 (BootstrapFewShot, MIPRO, BootstrapFinetune)
- `references/examples.md` - 실제 적용 사례 (RAG, agents, classifiers)

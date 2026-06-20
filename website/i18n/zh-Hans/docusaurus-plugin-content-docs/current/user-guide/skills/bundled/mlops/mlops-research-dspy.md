---
title: "Dspy — DSPy：声明式语言模型程序、自动优化 prompt、RAG"
sidebar_label: "Dspy"
description: "DSPy：声明式语言模型程序、自动优化 prompt、RAG"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Dspy

DSPy：声明式语言模型程序、自动优化 prompt（提示词）、RAG（检索增强生成）。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/mlops/research/dspy` |
| 版本 | `1.0.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖 | `dspy`, `openai`, `anthropic` |
| 平台 | linux, macos, windows |
| 标签 | `Prompt Engineering`, `DSPy`, `Declarative Programming`, `RAG`, `Agents`, `Prompt Optimization`, `LM Programming`, `Stanford NLP`, `Automatic Optimization`, `Modular AI` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# DSPy：声明式语言模型编程

## 何时使用此 Skill

在以下场景中使用 DSPy：
- **构建复杂 AI 系统**，包含多个组件和工作流
- **以声明式方式编程语言模型**，而非手动进行 prompt 工程
- **使用数据驱动方法自动优化 prompt**
- **创建可维护、可移植的模块化 AI 流水线**
- **通过优化器系统性地改善模型输出**
- **构建可靠性更高的 RAG 系统、agent 或分类器**

**GitHub Stars**：22,000+ | **创建者**：Stanford NLP

## 安装

```bash
# 稳定版本
pip install dspy

# 最新开发版本
pip install git+https://github.com/stanfordnlp/dspy.git

# 指定语言模型提供商
pip install dspy[openai]        # OpenAI
pip install dspy[anthropic]     # Anthropic Claude
pip install dspy[all]           # 所有提供商
```

## 快速开始

### 基础示例：问答

```python
import dspy

# 配置语言模型
lm = dspy.Claude(model="claude-sonnet-4-5-20250929")
dspy.settings.configure(lm=lm)

# 定义 signature（输入 → 输出）
class QA(dspy.Signature):
    """Answer questions with short factual answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

# 创建模块
qa = dspy.Predict(QA)

# 使用
response = qa(question="What is the capital of France?")
print(response.answer)  # "Paris"
```

### 思维链推理

```python
import dspy

lm = dspy.Claude(model="claude-sonnet-4-5-20250929")
dspy.settings.configure(lm=lm)

# 使用 ChainOfThought 获得更好的推理效果
class MathProblem(dspy.Signature):
    """Solve math word problems."""
    problem = dspy.InputField()
    answer = dspy.OutputField(desc="numerical answer")

# ChainOfThought 自动生成推理步骤
cot = dspy.ChainOfThought(MathProblem)

response = cot(problem="If John has 5 apples and gives 2 to Mary, how many does he have?")
print(response.rationale)  # 显示推理步骤
print(response.answer)     # "3"
```

## 核心概念

### 1. Signature

Signature 定义 AI 任务的结构（输入 → 输出）：

```python
# 内联 signature（简单形式）
qa = dspy.Predict("question -> answer")

# 类 signature（详细形式）
class Summarize(dspy.Signature):
    """Summarize text into key points."""
    text = dspy.InputField()
    summary = dspy.OutputField(desc="bullet points, 3-5 items")

summarizer = dspy.ChainOfThought(Summarize)
```

**各形式适用场景：**
- **内联**：快速原型开发、简单任务
- **类**：复杂任务、类型提示、更好的文档说明

### 2. 模块

模块是将输入转换为输出的可复用组件：

#### dspy.Predict
基础预测模块：

```python
predictor = dspy.Predict("context, question -> answer")
result = predictor(context="Paris is the capital of France",
                   question="What is the capital?")
```

#### dspy.ChainOfThought
在回答前生成推理步骤：

```python
cot = dspy.ChainOfThought("question -> answer")
result = cot(question="Why is the sky blue?")
print(result.rationale)  # 推理步骤
print(result.answer)     # 最终答案
```

#### dspy.ReAct
带工具的类 agent 推理：

```python
from dspy.predict import ReAct

class SearchQA(dspy.Signature):
    """Answer questions using search."""
    question = dspy.InputField()
    answer = dspy.OutputField()

def search_tool(query: str) -> str:
    """Search Wikipedia."""
    # 你的搜索实现
    return results

react = ReAct(SearchQA, tools=[search_tool])
result = react(question="When was Python created?")
```

#### dspy.ProgramOfThought
生成并执行代码进行推理：

```python
pot = dspy.ProgramOfThought("question -> answer")
result = pot(question="What is 15% of 240?")
# 生成：answer = 240 * 0.15
```

### 3. 优化器

优化器使用训练数据自动改善你的模块：

#### BootstrapFewShot
从示例中学习：

```python
from dspy.teleprompt import BootstrapFewShot

# 训练数据
trainset = [
    dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
    dspy.Example(question="What is 3+5?", answer="8").with_inputs("question"),
]

# 定义指标
def validate_answer(example, pred, trace=None):
    return example.answer == pred.answer

# 优化
optimizer = BootstrapFewShot(metric=validate_answer, max_bootstrapped_demos=3)
optimized_qa = optimizer.compile(qa, trainset=trainset)

# 现在 optimized_qa 性能更好！
```

#### MIPRO（最重要的 Prompt 优化）
迭代式改善 prompt：

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
为模型微调创建数据集：

```python
from dspy.teleprompt import BootstrapFinetune

optimizer = BootstrapFinetune(metric=validate_answer)
optimized_module = optimizer.compile(qa, trainset=trainset)

# 导出用于微调的训练数据
```

### 4. 构建复杂系统

#### 多阶段流水线

```python
import dspy

class MultiHopQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=3)
        self.generate_query = dspy.ChainOfThought("question -> search_query")
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        # 阶段 1：生成搜索查询
        search_query = self.generate_query(question=question).search_query

        # 阶段 2：检索上下文
        passages = self.retrieve(search_query).passages
        context = "\n".join(passages)

        # 阶段 3：生成答案
        answer = self.generate_answer(context=context, question=question).answer
        return dspy.Prediction(answer=answer, context=context)

# 使用流水线
qa_system = MultiHopQA()
result = qa_system(question="Who wrote the book that inspired the movie Blade Runner?")
```

#### 带优化的 RAG 系统

```python
import dspy
from dspy.retrieve.chromadb_rm import ChromadbRM

# 配置检索器
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

# 创建并优化
rag = RAG()

# 使用训练数据优化
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(metric=validate_answer)
optimized_rag = optimizer.compile(rag, trainset=trainset)
```

## 语言模型提供商配置

### Anthropic Claude

```python
import dspy

lm = dspy.Claude(
    model="claude-sonnet-4-5-20250929",
    api_key="your-api-key",  # 或设置 ANTHROPIC_API_KEY 环境变量
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

### 本地模型（Ollama）

```python
lm = dspy.OllamaLocal(
    model="llama3.1",
    base_url="http://localhost:11434"
)
dspy.settings.configure(lm=lm)
```

### 多模型

```python
# 不同任务使用不同模型
cheap_lm = dspy.OpenAI(model="gpt-3.5-turbo")
strong_lm = dspy.Claude(model="claude-sonnet-4-5-20250929")

# 检索使用廉价模型，推理使用强力模型
with dspy.settings.context(lm=cheap_lm):
    context = retriever(question)

with dspy.settings.context(lm=strong_lm):
    answer = generator(context=context, question=question)
```

## 常见模式

### 模式 1：结构化输出

```python
from pydantic import BaseModel, Field

class PersonInfo(BaseModel):
    name: str = Field(description="Full name")
    age: int = Field(description="Age in years")
    occupation: str = Field(description="Current job")

class ExtractPerson(dspy.Signature):
    """Extract person information from text."""
    text = dspy.InputField()
    person: PersonInfo = dspy.OutputField()

extractor = dspy.TypedPredictor(ExtractPerson)
result = extractor(text="John Doe is a 35-year-old software engineer.")
print(result.person.name)  # "John Doe"
print(result.person.age)   # 35
```

### 模式 2：断言驱动优化

```python
import dspy
from dspy.primitives.assertions import assert_transform_module, backtrack_handler

class MathQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.solve = dspy.ChainOfThought("problem -> solution: float")

    def forward(self, problem):
        solution = self.solve(problem=problem).solution

        # 断言解答为数值
        dspy.Assert(
            isinstance(float(solution), float),
            "Solution must be a number",
            backtrack=backtrack_handler
        )

        return dspy.Prediction(solution=solution)
```

### 模式 3：自洽性

```python
import dspy
from collections import Counter

class ConsistentQA(dspy.Module):
    def __init__(self, num_samples=5):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")
        self.num_samples = num_samples

    def forward(self, question):
        # 生成多个答案
        answers = []
        for _ in range(self.num_samples):
            result = self.qa(question=question)
            answers.append(result.answer)

        # 返回最常见的答案
        most_common = Counter(answers).most_common(1)[0][0]
        return dspy.Prediction(answer=most_common)
```

### 模式 4：带重排序的检索

```python
class RerankedRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=10)
        self.rerank = dspy.Predict("question, passage -> relevance_score: float")
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        # 检索候选段落
        passages = self.retrieve(question).passages

        # 对段落重排序
        scored = []
        for passage in passages:
            score = float(self.rerank(question=question, passage=passage).relevance_score)
            scored.append((score, passage))

        # 取前 3 名
        top_passages = [p for _, p in sorted(scored, reverse=True)[:3]]
        context = "\n\n".join(top_passages)

        # 生成答案
        return self.answer(context=context, question=question)
```

## 评估与指标

### 自定义指标

```python
def exact_match(example, pred, trace=None):
    """精确匹配指标。"""
    return example.answer.lower() == pred.answer.lower()

def f1_score(example, pred, trace=None):
    """文本重叠的 F1 分数。"""
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

### 评估

```python
from dspy.evaluate import Evaluate

# 创建评估器
evaluator = Evaluate(
    devset=testset,
    metric=exact_match,
    num_threads=4,
    display_progress=True
)

# 评估模型
score = evaluator(qa_system)
print(f"Accuracy: {score}")

# 比较优化前后
score_before = evaluator(qa)
score_after = evaluator(optimized_qa)
print(f"Improvement: {score_after - score_before:.2%}")
```

## 最佳实践

### 1. 从简单开始，逐步迭代

```python
# 从 Predict 开始
qa = dspy.Predict("question -> answer")

# 如有需要，添加推理
qa = dspy.ChainOfThought("question -> answer")

# 有数据后进行优化
optimized_qa = optimizer.compile(qa, trainset=data)
```

### 2. 使用描述性 Signature

```python
# ❌ 差：模糊
class Task(dspy.Signature):
    input = dspy.InputField()
    output = dspy.OutputField()

# ✅ 好：描述性强
class SummarizeArticle(dspy.Signature):
    """Summarize news articles into 3-5 key points."""
    article = dspy.InputField(desc="full article text")
    summary = dspy.OutputField(desc="bullet points, 3-5 items")
```

### 3. 使用有代表性的数据进行优化

```python
# 创建多样化的训练示例
trainset = [
    dspy.Example(question="factual", answer="...).with_inputs("question"),
    dspy.Example(question="reasoning", answer="...").with_inputs("question"),
    dspy.Example(question="calculation", answer="...").with_inputs("question"),
]

# 使用验证集计算指标
def metric(example, pred, trace=None):
    return example.answer in pred.answer
```

### 4. 保存和加载优化后的模型

```python
# 保存
optimized_qa.save("models/qa_v1.json")

# 加载
loaded_qa = dspy.ChainOfThought("question -> answer")
loaded_qa.load("models/qa_v1.json")
```

### 5. 监控与调试

```python
# 启用追踪
dspy.settings.configure(lm=lm, trace=[])

# 运行预测
result = qa(question="...")

# 检查追踪记录
for call in dspy.settings.trace:
    print(f"Prompt: {call['prompt']}")
    print(f"Response: {call['response']}")
```

## 与其他方案的对比

| 特性 | 手动 Prompt | LangChain | DSPy |
|---------|-----------------|-----------|------|
| Prompt 工程 | 手动 | 手动 | 自动 |
| 优化方式 | 试错 | 无 | 数据驱动 |
| 模块化程度 | 低 | 中 | 高 |
| 类型安全 | 否 | 有限 | 是（Signature） |
| 可移植性 | 低 | 中 | 高 |
| 学习曲线 | 低 | 中 | 中高 |

**选择 DSPy 的场景：**
- 你有训练数据或可以生成训练数据
- 你需要系统性地改善 prompt
- 你在构建复杂的多阶段系统
- 你希望跨不同语言模型进行优化

**选择其他方案的场景：**
- 快速原型开发（手动 prompt）
- 使用现有工具的简单链式调用（LangChain）
- 需要自定义优化逻辑

## 资源

- **文档**：https://dspy.ai
- **GitHub**：https://github.com/stanfordnlp/dspy（22k+ stars）
- **Discord**：https://discord.gg/XCGy2WDCQB
- **Twitter**：@DSPyOSS
- **论文**："DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines"

## 另请参阅

- `references/modules.md` — 详细模块指南（Predict、ChainOfThought、ReAct、ProgramOfThought）
- `references/optimizers.md` — 优化算法（BootstrapFewShot、MIPRO、BootstrapFinetune）
- `references/examples.md` — 真实世界示例（RAG、agent、分类器）
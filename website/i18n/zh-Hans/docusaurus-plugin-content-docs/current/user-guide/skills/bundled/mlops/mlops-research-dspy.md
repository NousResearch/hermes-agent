---
title: "Dspy — DSPy：声明式 LM 程序，自动优化提示，RAG"
sidebar_label: "Dspy"
description: "DSPy：声明式 LM 程序，自动优化提示，RAG"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Dspy

DSPy：声明式 LM 程序，自动优化提示，RAG。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/mlops/research/dspy` |
| 版本 | `1.0.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖项 | `dspy`, `openai`, `anthropic` |
| 标签 | `提示工程`, `DSPy`, `声明式编程`, `RAG`, `代理`, `提示优化`, `LM 编程`, `Stanford NLP`, `自动优化`, `模块化 AI` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此技能时加载的完整技能定义。这是技能激活时代理看到的指令内容。
:::

# DSPy：声明式语言模型编程

## 何时使用此技能

在以下情况下使用 DSPy：
- **构建复杂 AI 系统**，涉及多个组件和工作流
- **以声明式方式编程 LM**，而非手动提示工程
- **使用数据驱动方法自动优化提示**
- **创建模块化 AI 流水线**，可维护且可移植
- **使用优化器系统化地改进模型输出**
- **构建 RAG 系统、代理或分类器**，提高可靠性

**GitHub Stars**：22,000+ | **创建者**：Stanford NLP

## 安装

```bash
# 稳定版本
pip install dspy

# 最新开发版本
pip install git+https://github.com/stanfordnlp/dspy.git

# 指定 LM 提供商
pip install dspy[openai]        # OpenAI
pip install dspy[anthropic]     # Anthropic Claude
pip install dspy[all]           # 所有提供商
```

## 快速开始

### 基本示例：问答

```python
import dspy

# 配置语言模型
lm = dspy.Claude(model="claude-sonnet-4-5-20250929")
dspy.settings.configure(lm=lm)

# 定义签名（输入 → 输出）
class QA(dspy.Signature):
    """用简短的事实性答案回答问题。"""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="通常 1 到 5 个词")

# 创建模块
qa = dspy.Predict(QA)

# 使用
response = qa(question="法国的首都是什么？")
print(response.answer)  # "巴黎"
```

### 思维链推理

```python
import dspy

lm = dspy.Claude(model="claude-sonnet-4-5-20250929")
dspy.settings.configure(lm=lm)

# 使用 ChainOfThought 进行更好的推理
class MathProblem(dspy.Signature):
    """解数学应用题。"""
    problem = dspy.InputField()
    answer = dspy.OutputField(desc="数值答案")

# ChainOfThought 自动生成推理步骤
cot = dspy.ChainOfThought(MathProblem)

response = cot(problem="如果约翰有 5 个苹果，给了玛丽 2 个，他还剩几个？")
print(response.rationale)  # 显示推理步骤
print(response.answer)     # "3"
```

## 核心概念

### 1. 签名

签名定义 AI 任务的结构（输入 → 输出）：

```python
# 内联签名（简单）
qa = dspy.Predict("question -> answer")

# 类签名（详细）
class Summarize(dspy.Signature):
    """将文本总结为要点。"""
    text = dspy.InputField()
    summary = dspy.OutputField(desc="要点，3-5 条")

summarizer = dspy.ChainOfThought(Summarize)
```

**何时使用哪种：**
- **内联**：快速原型开发，简单任务
- **类**：复杂任务，类型提示，更好的文档

### 2. 模块

模块是可复用组件，将输入转换为输出：

#### dspy.Predict
基本预测模块：

```python
predictor = dspy.Predict("context, question -> answer")
result = predictor(context="巴黎是法国的首都",
                   question="首都是什么？")
```

#### dspy.ChainOfThought
在回答之前生成推理步骤：

```python
cot = dspy.ChainOfThought("question -> answer")
result = cot(question="为什么天空是蓝色的？")
print(result.rationale)  # 推理步骤
print(result.answer)     # 最终答案
```

#### dspy.ReAct
带工具的类代理推理：

```python
from dspy.predict import ReAct

class SearchQA(dspy.Signature):
    """使用搜索回答问题。"""
    question = dspy.InputField()
    answer = dspy.OutputField()

def search_tool(query: str) -> str:
    """搜索维基百科。"""
    # 你的搜索实现
    return results

react = ReAct(SearchQA, tools=[search_tool])
result = react(question="Python 是什么时候创建的？")
```

#### dspy.ProgramOfThought
生成并执行代码进行推理：

```python
pot = dspy.ProgramOfThought("question -> answer")
result = pot(question="240 的 15% 是多少？")
# 生成：answer = 240 * 0.15
```

### 3. 优化器

优化器使用训练数据自动改进模块：

#### BootstrapFewShot
从示例中学习：

```python
from dspy.teleprompt import BootstrapFewShot

# 训练数据
trainset = [
    dspy.Example(question="2+2 是多少？", answer="4").with_inputs("question"),
    dspy.Example(question="3+5 是多少？", answer="8").with_inputs("question"),
]

# 定义度量
def validate_answer(example, pred, trace=None):
    return example.answer == pred.answer

# 优化
optimizer = BootstrapFewShot(metric=validate_answer, max_bootstrapped_demos=3)
optimized_qa = optimizer.compile(qa, trainset=trainset)

# 现在 optimized_qa 表现更好！
```

#### MIPRO（最重要的提示优化）
迭代改进提示：

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

# 导出训练数据用于微调
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
result = qa_system(question="谁写了那本启发了电影《银翼杀手》的书？")
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

## LM 提供商配置

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

# 检索使用便宜模型，推理使用强模型
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
    name: str = Field(description="全名")
    age: int = Field(description="年龄")
    occupation: str = Field(description="当前职业")

class ExtractPerson(dspy.Signature):
    """从文本中提取人物信息。"""
    text = dspy.InputField()
    person: PersonInfo = dspy.OutputField()

extractor = dspy.TypedPredictor(ExtractPerson)
result = extractor(text="John Doe 是一名 35 岁的软件工程师。")
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

        # 断言解是数值
        dspy.Assert(
            isinstance(float(solution), float),
            "解必须是数字",
            backtrack=backtrack_handler
        )

        return dspy.Prediction(solution=solution)
```

### 模式 3：自一致性

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
        # 检索候选
        passages = self.retrieve(question).passages

        # 重排序段落
        scored = []
        for passage in passages:
            score = float(self.rerank(question=question, passage=passage).relevance_score)
            scored.append((score, passage))

        # 取前 3 个
        top_passages = [p for _, p in sorted(scored, reverse=True)[:3]]
        context = "\n\n".join(top_passages)

        # 生成答案
        return self.answer(context=context, question=question)
```

## 评估与度量

### 自定义度量

```python
def exact_match(example, pred, trace=None):
    """精确匹配度量。"""
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
print(f"准确率：{score}")

# 比较优化前后
score_before = evaluator(qa)
score_after = evaluator(optimized_qa)
print(f"提升：{score_after - score_before:.2%}")
```

## 最佳实践

### 1. 从简单开始，迭代

```python
# 从 Predict 开始
qa = dspy.Predict("question -> answer")

# 如需推理则添加
qa = dspy.ChainOfThought("question -> answer")

# 有数据时添加优化
optimized_qa = optimizer.compile(qa, trainset=data)
```

### 2. 使用描述性签名

```python
# ❌ 不好的：模糊
class Task(dspy.Signature):
    input = dspy.InputField()
    output = dspy.OutputField()

# ✅ 好的：描述性
class SummarizeArticle(dspy.Signature):
    """将新闻文章总结为 3-5 个要点。"""
    article = dspy.InputField(desc="完整文章文本")
    summary = dspy.OutputField(desc="要点，3-5 条")
```

### 3. 使用代表性数据进行优化

```python
# 创建多样化的训练示例
trainset = [
    dspy.Example(question="事实型", answer="...).with_inputs("question"),
    dspy.Example(question="推理型", answer="...").with_inputs("question"),
    dspy.Example(question="计算型", answer="...").with_inputs("question"),
]

# 使用验证集进行度量
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

### 5. 监控和调试

```python
# 启用追踪
dspy.settings.configure(lm=lm, trace=[])

# 运行预测
result = qa(question="...")

# 检查追踪
for call in dspy.settings.trace:
    print(f"提示：{call['prompt']}")
    print(f"响应：{call['response']}")
```

## 与其他方法的比较

| 特性 | 手动提示 | LangChain | DSPy |
|------|-----------------|-----------|------|
| 提示工程 | 手动 | 手动 | 自动 |
| 优化 | 试错 | 无 | 数据驱动 |
| 模块化 | 低 | 中 | 高 |
| 类型安全 | 无 | 有限 | 有（签名） |
| 可移植性 | 低 | 中 | 高 |
| 学习曲线 | 低 | 中 | 中高 |

**何时选择 DSPy：**
- 你有训练数据或可以生成训练数据
- 需要系统化的提示改进
- 正在构建复杂的多阶段系统
- 想在不同 LM 之间优化

**何时选择替代方案：**
- 快速原型（手动提示）
- 使用现有工具的简单链（LangChain）
- 需要自定义优化逻辑

## 资源

- **文档**：https://dspy.ai
- **GitHub**：https://github.com/stanfordnlp/dspy (22k+ stars)
- **Discord**：https://discord.gg/XCGy2WDCQB
- **Twitter**：@DSPyOSS
- **论文**："DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines"

## 另见

- `references/modules.md` - 详细模块指南（Predict、ChainOfThought、ReAct、ProgramOfThought）
- `references/optimizers.md` - 优化算法（BootstrapFewShot、MIPRO、BootstrapFinetune）
- `references/examples.md` - 实际示例（RAG、代理、分类器）

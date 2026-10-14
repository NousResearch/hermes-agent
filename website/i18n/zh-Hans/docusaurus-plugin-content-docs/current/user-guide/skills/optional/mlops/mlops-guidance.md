---
title: "Guidance"
sidebar_label: "Guidance"
description: "使用正则表达式和语法控制 LLM 输出，保证生成有效的 JSON/XML/代码，强制结构化格式，并使用 Guidance（微软研究院的约束生成框架）构建多步骤工作流..."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Guidance

使用正则表达式和语法控制 LLM 输出，保证生成有效的 JSON/XML/代码，强制结构化格式，并使用 Guidance（微软研究院的约束生成框架）构建多步骤工作流

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/mlops/guidance` 安装 |
| 路径 | `optional-skills/mlops/guidance` |
| 版本 | `1.0.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖项 | `guidance`, `transformers` |
| 平台 | linux, macos, windows |
| 标签 | `Prompt Engineering`, `Guidance`, `Constrained Generation`, `Structured Output`, `JSON Validation`, `Grammar`, `Microsoft Research`, `Format Enforcement`, `Multi-Step Workflows` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时看到的指令内容。
:::

# Guidance：约束 LLM 生成

## 何时使用此 Skill

在以下情况下使用 Guidance：
- **使用正则表达式或语法控制 LLM 输出语法**
- **保证生成有效的 JSON/XML/代码**
- **相比传统 prompting（提示词）方式降低延迟**
- **强制结构化格式**（日期、邮箱、ID 等）
- **使用 Python 风格的控制流构建多步骤工作流**
- **通过语法约束防止无效输出**

**GitHub Stars**：18,000+ | **来自**：微软研究院

## 安装

```bash
# 基础安装
pip install guidance

# 指定后端
pip install guidance[transformers]  # Hugging Face 模型
pip install guidance[llama_cpp]     # llama.cpp 模型
```

## 快速开始

### 基础示例：结构化生成

```python
from guidance import models, gen

# 加载模型（支持 OpenAI、Transformers、llama.cpp）
lm = models.OpenAI("gpt-4")

# 带约束生成
result = lm + "The capital of France is " + gen("capital", max_tokens=5)

print(result["capital"])  # "Paris"
```

### 使用 Anthropic Claude

```python
from guidance import models, gen, system, user, assistant

# 配置 Claude
lm = models.Anthropic("claude-sonnet-4-5-20250929")

# 使用上下文管理器实现对话格式
with system():
    lm += "You are a helpful assistant."

with user():
    lm += "What is the capital of France?"

with assistant():
    lm += gen(max_tokens=20)
```

## 核心概念

### 1. 上下文管理器

Guidance 使用 Python 风格的上下文管理器实现对话式交互。

```python
from guidance import system, user, assistant, gen

lm = models.Anthropic("claude-sonnet-4-5-20250929")

# 系统消息
with system():
    lm += "You are a JSON generation expert."

# 用户消息
with user():
    lm += "Generate a person object with name and age."

# 助手回复
with assistant():
    lm += gen("response", max_tokens=100)

print(lm["response"])
```

**优势：**
- 自然的对话流程
- 清晰的角色分离
- 易于阅读和维护

### 2. 约束生成

Guidance 使用正则表达式或语法确保输出符合指定模式。

#### 正则表达式约束

```python
from guidance import models, gen

lm = models.Anthropic("claude-sonnet-4-5-20250929")

# 约束为有效邮箱格式
lm += "Email: " + gen("email", regex=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# 约束为日期格式（YYYY-MM-DD）
lm += "Date: " + gen("date", regex=r"\d{4}-\d{2}-\d{2}")

# 约束为电话号码
lm += "Phone: " + gen("phone", regex=r"\d{3}-\d{3}-\d{4}")

print(lm["email"])  # 保证为有效邮箱
print(lm["date"])   # 保证为 YYYY-MM-DD 格式
```

**工作原理：**
- 正则表达式在 token（词元）级别转换为语法
- 生成过程中过滤无效 token
- 模型只能生成符合匹配条件的输出

#### 选择约束

```python
from guidance import models, gen, select

lm = models.Anthropic("claude-sonnet-4-5-20250929")

# 约束为特定选项
lm += "Sentiment: " + select(["positive", "negative", "neutral"], name="sentiment")

# 多选题选择
lm += "Best answer: " + select(
    ["A) Paris", "B) London", "C) Berlin", "D) Madrid"],
    name="answer"
)

print(lm["sentiment"])  # 其中之一：positive、negative、neutral
print(lm["answer"])     # 其中之一：A、B、C 或 D
```

### 3. Token 修复（Token Healing）

Guidance 自动"修复" prompt 与生成内容之间的 token 边界。

**问题：** 分词会产生不自然的边界。

```python
# 不使用 token 修复
prompt = "The capital of France is "
# 最后一个 token：" is "
# 第一个生成的 token 可能是 " Par"（带前导空格）
# 结果："The capital of France is  Paris"（双空格！）
```

**解决方案：** Guidance 回退一个 token 并重新生成。

```python
from guidance import models, gen

lm = models.Anthropic("claude-sonnet-4-5-20250929")

# 默认启用 token 修复
lm += "The capital of France is " + gen("capital", max_tokens=5)
# 结果："The capital of France is Paris"（间距正确）
```

**优势：**
- 自然的文本边界
- 无尴尬的间距问题
- 更好的模型性能（模型看到自然的 token 序列）

### 4. 基于语法的生成

使用上下文无关语法定义复杂结构。

```python
from guidance import models, gen

lm = models.Anthropic("claude-sonnet-4-5-20250929")

# JSON 语法（简化版）
json_grammar = """
{
    "name": <gen name regex="[A-Za-z ]+" max_tokens=20>,
    "age": <gen age regex="[0-9]+" max_tokens=3>,
    "email": <gen email regex="[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}" max_tokens=50>
}
"""

# 生成有效 JSON
lm += gen("person", grammar=json_grammar)

print(lm["person"])  # 保证为有效 JSON 结构
```

**使用场景：**
- 复杂结构化输出
- 嵌套数据结构
- 编程语言语法
- 领域特定语言

### 5. Guidance 函数

使用 `@guidance` 装饰器创建可复用的生成模式。

```python
from guidance import guidance, gen, models

@guidance
def generate_person(lm):
    """生成包含姓名和年龄的人物信息。"""
    lm += "Name: " + gen("name", max_tokens=20, stop="\n")
    lm += "\nAge: " + gen("age", regex=r"[0-9]+", max_tokens=3)
    return lm

# 使用该函数
lm = models.Anthropic("claude-sonnet-4-5-20250929")
lm = generate_person(lm)

print(lm["name"])
print(lm["age"])
```

**有状态函数：**

```python
@guidance(stateless=False)
def react_agent(lm, question, tools, max_rounds=5):
    """带工具调用的 ReAct agent。"""
    lm += f"Question: {question}\n\n"

    for i in range(max_rounds):
        # 思考
        lm += f"Thought {i+1}: " + gen("thought", stop="\n")

        # 动作
        lm += "\nAction: " + select(list(tools.keys()), name="action")

        # 执行工具
        tool_result = tools[lm["action"]]()
        lm += f"\nObservation: {tool_result}\n\n"

        # 检查是否完成
        lm += "Done? " + select(["Yes", "No"], name="done")
        if lm["done"] == "Yes":
            break

    # 最终答案
    lm += "\nFinal Answer: " + gen("answer", max_tokens=100)
    return lm
```

## 后端配置

### Anthropic Claude

```python
from guidance import models

lm = models.Anthropic(
    model="claude-sonnet-4-5-20250929",
    api_key="your-api-key"  # 或设置 ANTHROPIC_API_KEY 环境变量
)
```

### OpenAI

```python
lm = models.OpenAI(
    model="gpt-4o-mini",
    api_key="your-api-key"  # 或设置 OPENAI_API_KEY 环境变量
)
```

### 本地模型（Transformers）

```python
from guidance.models import Transformers

lm = Transformers(
    "microsoft/Phi-4-mini-instruct",
    device="cuda"  # 或 "cpu"
)
```

### 本地模型（llama.cpp）

```python
from guidance.models import LlamaCpp

lm = LlamaCpp(
    model_path="/path/to/model.gguf",
    n_ctx=4096,
    n_gpu_layers=35
)
```

## 常用模式

### 模式 1：JSON 生成

```python
from guidance import models, gen, system, user, assistant

lm = models.Anthropic("claude-sonnet-4-5-20250929")

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

print(lm)  # 保证为有效 JSON
```

### 模式 2：分类

```python
from guidance import models, gen, select

lm = models.Anthropic("claude-sonnet-4-5-20250929")

text = "This product is amazing! I love it."

lm += f"Text: {text}\n"
lm += "Sentiment: " + select(["positive", "negative", "neutral"], name="sentiment")
lm += "\nConfidence: " + gen("confidence", regex=r"[0-9]+", max_tokens=3) + "%"

print(f"Sentiment: {lm['sentiment']}")
print(f"Confidence: {lm['confidence']}%")
```

### 模式 3：多步骤推理

```python
from guidance import models, gen, guidance

@guidance
def chain_of_thought(lm, question):
    """逐步推理生成答案。"""
    lm += f"Question: {question}\n\n"

    # 生成多个推理步骤
    for i in range(3):
        lm += f"Step {i+1}: " + gen(f"step_{i+1}", stop="\n", max_tokens=100) + "\n"

    # 最终答案
    lm += "\nTherefore, the answer is: " + gen("answer", max_tokens=50)

    return lm

lm = models.Anthropic("claude-sonnet-4-5-20250929")
lm = chain_of_thought(lm, "What is 15% of 200?")

print(lm["answer"])
```

### 模式 4：ReAct Agent

```python
from guidance import models, gen, select, guidance

@guidance(stateless=False)
def react_agent(lm, question):
    """带工具调用的 ReAct agent。"""
    tools = {
        "calculator": lambda expr: eval(expr),
        "search": lambda query: f"Search results for: {query}",
    }

    lm += f"Question: {question}\n\n"

    for round in range(5):
        # 思考
        lm += f"Thought: " + gen("thought", stop="\n") + "\n"

        # 动作选择
        lm += "Action: " + select(["calculator", "search", "answer"], name="action")

        if lm["action"] == "answer":
            lm += "\nFinal Answer: " + gen("answer", max_tokens=100)
            break

        # 动作输入
        lm += "\nAction Input: " + gen("action_input", stop="\n") + "\n"

        # 执行工具
        if lm["action"] in tools:
            result = tools[lm["action"]](lm["action_input"])
            lm += f"Observation: {result}\n\n"

    return lm

lm = models.Anthropic("claude-sonnet-4-5-20250929")
lm = react_agent(lm, "What is 25 * 4 + 10?")
print(lm["answer"])
```

### 模式 5：数据提取

```python
from guidance import models, gen, guidance

@guidance
def extract_entities(lm, text):
    """从文本中提取结构化实体。"""
    lm += f"Text: {text}\n\n"

    # 提取人物
    lm += "Person: " + gen("person", stop="\n", max_tokens=30) + "\n"

    # 提取组织
    lm += "Organization: " + gen("organization", stop="\n", max_tokens=30) + "\n"

    # 提取日期
    lm += "Date: " + gen("date", regex=r"\d{4}-\d{2}-\d{2}", max_tokens=10) + "\n"

    # 提取地点
    lm += "Location: " + gen("location", stop="\n", max_tokens=30) + "\n"

    return lm

text = "Tim Cook announced at Apple Park on 2024-09-15 in Cupertino."

lm = models.Anthropic("claude-sonnet-4-5-20250929")
lm = extract_entities(lm, text)

print(f"Person: {lm['person']}")
print(f"Organization: {lm['organization']}")
print(f"Date: {lm['date']}")
print(f"Location: {lm['location']}")
```

## 最佳实践

### 1. 使用正则表达式进行格式验证

```python
# ✅ 好：正则表达式确保格式有效
lm += "Email: " + gen("email", regex=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# ❌ 差：自由生成可能产生无效邮箱
lm += "Email: " + gen("email", max_tokens=50)
```

### 2. 对固定类别使用 select()

```python
# ✅ 好：保证为有效类别
lm += "Status: " + select(["pending", "approved", "rejected"], name="status")

# ❌ 差：可能生成拼写错误或无效值
lm += "Status: " + gen("status", max_tokens=20)
```

### 3. 利用 Token 修复

```python
# 默认启用 token 修复
# 无需特殊操作——自然拼接即可
lm += "The capital is " + gen("capital")  # 自动修复
```

### 4. 使用停止序列

```python
# ✅ 好：在换行处停止，适用于单行输出
lm += "Name: " + gen("name", stop="\n")

# ❌ 差：可能生成多行内容
lm += "Name: " + gen("name", max_tokens=50)
```

### 5. 创建可复用函数

```python
# ✅ 好：可复用模式
@guidance
def generate_person(lm):
    lm += "Name: " + gen("name", stop="\n")
    lm += "\nAge: " + gen("age", regex=r"[0-9]+")
    return lm

# 多次使用
lm = generate_person(lm)
lm += "\n\n"
lm = generate_person(lm)
```

### 6. 平衡约束力度

```python
# ✅ 好：合理的约束
lm += gen("name", regex=r"[A-Za-z ]+", max_tokens=30)

# ❌ 过于严格：可能失败或非常缓慢
lm += gen("name", regex=r"^(John|Jane)$", max_tokens=10)
```

## 与替代方案的对比

| 特性 | Guidance | Instructor | Outlines | LMQL |
|---------|----------|------------|----------|------|
| 正则表达式约束 | ✅ 支持 | ❌ 不支持 | ✅ 支持 | ✅ 支持 |
| 语法支持 | ✅ CFG | ❌ 不支持 | ✅ CFG | ✅ CFG |
| Pydantic 验证 | ❌ 不支持 | ✅ 支持 | ✅ 支持 | ❌ 不支持 |
| Token 修复 | ✅ 支持 | ❌ 不支持 | ✅ 支持 | ❌ 不支持 |
| 本地模型 | ✅ 支持 | ⚠️ 有限 | ✅ 支持 | ✅ 支持 |
| API 模型 | ✅ 支持 | ✅ 支持 | ⚠️ 有限 | ✅ 支持 |
| Python 风格语法 | ✅ 支持 | ✅ 支持 | ✅ 支持 | ❌ 类 SQL |
| 学习曲线 | 低 | 低 | 中 | 高 |

**何时选择 Guidance：**
- 需要正则表达式/语法约束
- 需要 token 修复
- 构建带控制流的复杂工作流
- 使用本地模型（Transformers、llama.cpp）
- 偏好 Python 风格语法

**何时选择替代方案：**
- Instructor：需要带自动重试的 Pydantic 验证
- Outlines：需要 JSON schema 验证
- LMQL：偏好声明式查询语法

## 性能特性

**延迟降低：**
- 对于约束输出，比传统 prompting 快 30–50%
- Token 修复减少不必要的重新生成
- 语法约束防止无效 token 的生成

**内存占用：**
- 相比无约束生成，额外开销极小
- 语法编译结果在首次使用后缓存
- 推理时高效过滤 token

**Token 效率：**
- 防止在无效输出上浪费 token
- 无需重试循环
- 直接生成有效输出

## 资源

- **文档**：https://guidance.readthedocs.io
- **GitHub**：https://github.com/guidance-ai/guidance（18k+ stars）
- **Notebooks**：https://github.com/guidance-ai/guidance/tree/main/notebooks
- **Discord**：提供社区支持

## 另请参阅

- `references/constraints.md` — 全面的正则表达式和语法模式
- `references/backends.md` — 后端专项配置
- `references/examples.md` — 生产就绪示例
---
title: "DSPy — 声明式 LM 程序、自动优化提示、RAG"
sidebar_label: "DSPy"
description: "DSPy：声明式 LM 程序、自动优化提示、RAG"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# DSPy

DSPy：声明式 LM 程序、自动优化提示、RAG。

## 技能元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/mlops/dspy` 安装 |
| 路径 | `optional-skills/mlops/research/dspy` |
| 版本 | `1.0.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖 | `dspy`、`openai`、`anthropic` |
| 平台 | linux、macos、windows |
| 标签 | `Prompt Engineering`、`DSPy`、`Declarative Programming`、`RAG`、`Agents`、`Prompt Optimization`、`LM Programming`、`Stanford NLP`、`Automatic Optimization`、`Modular AI` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在此技能触发时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# DSPy：声明式语言模型编程

## 何时使用

- **构建复杂 AI 系统**，包含多个组件和工作流
- **声明式编程 LM**，而非手动 prompt engineering
- **使用数据驱动方法自动优化提示**
- **创建可维护和可移植的模块化 AI 管道**
- **使用优化器系统性地改善模型输出**
- **构建 RAG 系统、代理或分类器**以获得更好的可靠性

**GitHub Stars**：22,000+ | **创建者**：Stanford NLP

## 安装

```bash
pip install dspy
pip install dspy[openai]
pip install dspy[anthropic]
pip install dspy[all]
```

## 核心概念

### 1. Signature

Signature 定义 AI 任务的结构（输入 → 输出）：

```python
class QA(dspy.Signature):
    """Answer questions with short factual answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
```

### 2. Module

Module 是可复用组件：`dspy.Predict`、`dspy.ChainOfThought`、`dspy.ReAct`、`dspy.ProgramOfThought`。

### 3. 优化器

- `BootstrapFewShot` — 从示例学习
- `MIPRO` — 迭代改善提示
- `BootstrapFinetune` — 为微调创建数据集

### 4. 构建复杂系统

多阶段管道和带优化的 RAG 系统。

## LM 提供商配置

支持 Anthropic Claude、OpenAI、Ollama 本地模型和多模型组合。

## 常见模式

结构化输出、断言驱动优化、自一致性、带重排序的检索。

## 评估和指标

自定义指标和评估器。

## 与其它方法比较

| 特性 | 手动提示 | LangChain | DSPy |
|------|---------|-----------|------|
| 提示工程 | 手动 | 手动 | 自动 |
| 优化 | 试错 | 无 | 数据驱动 |
| 模块化 | 低 | 中 | 高 |
| 可移植性 | 低 | 中 | 高 |

## 资源

- **文档**：https://dspy.ai
- **GitHub**：https://github.com/stanfordnlp/dspy

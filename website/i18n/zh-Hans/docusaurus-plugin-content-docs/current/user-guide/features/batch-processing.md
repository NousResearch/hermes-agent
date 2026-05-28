---
sidebar_position: 12
title: "批量处理"
description: "大规模生成 agent 轨迹——并行处理、断点续跑与工具集分布"
---

# 批量处理

批量处理让你能够并行地在数百乃至数千个 prompt（提示词）上运行 Hermes agent，生成结构化的轨迹数据。其主要用途是**训练数据生成**——产出包含工具使用统计信息的 ShareGPT 格式轨迹，可用于微调或评估。

## 概述

批量运行器（`batch_runner.py`）处理一个由 prompt 组成的 JSONL 数据集，将每条 prompt 通过完整的 agent 会话（含工具访问权限）运行一遍。每条 prompt 都拥有独立隔离的环境。输出为结构化轨迹数据，包含完整对话历史、工具调用统计信息以及推理覆盖率指标。

## 快速开始

```bash
# 基本批量运行
python batch_runner.py \
    --dataset_file=data/prompts.jsonl \
    --batch_size=10 \
    --run_name=my_first_run \
    --model=anthropic/claude-sonnet-4.6 \
    --num_workers=4

# 恢复中断的运行
python batch_runner.py \
    --dataset_file=data/prompts.jsonl \
    --batch_size=10 \
    --run_name=my_first_run \
    --resume

# 列出可用的工具集分布
python batch_runner.py --list_distributions
```

:::tip 大规模运行下的可预测成本
批量运行会启动大量并发 agent 会话，每个会话都会调用模型和工具。[Nous Portal](/user-guide/features/tool-gateway) 订阅将模型访问、网页搜索、图像生成、TTS 以及云端浏览器统一计费——当你希望在不同供应商账户间稳定控制每条轨迹成本、避免触碰速率限制时非常实用。使用 `hermes setup --portal` 完成配置，然后将 `--model` 指向 Nous 模型。
:::

## 数据集格式

输入数据集为 JSONL 文件（每行一个 JSON 对象）。每条记录必须包含 `prompt` 字段：

```jsonl
{"prompt": "Write a Python function that finds the longest palindromic substring"}
{"prompt": "Create a REST API endpoint for user authentication using Flask"}
{"prompt": "Debug this error: TypeError: cannot unpack non-iterable NoneType object"}
```

记录还可以选填以下字段：
- `image` 或 `docker_image`：用于该 prompt 沙箱的容器镜像（适用于 Docker、Modal 和 Singularity 后端）
- `cwd`：任务终端会话的工作目录覆盖值

## 配置选项

| 参数 | 默认值 | 说明 |
|-----------|---------|-------------|
| `--dataset_file` | （必填） | JSONL 数据集路径 |
| `--batch_size` | （必填） | 每批处理的 prompt 数量 |
| `--run_name` | （必填） | 本次运行的名称（用于输出目录和断点续跑） |
| `--distribution` | `"default"` | 采样所用的工具集分布 |
| `--model` | `claude-sonnet-4.6` | 使用的模型 |
| `--base_url` | `https://openrouter.ai/api/v1` | API 基础 URL |
| `--api_key` | （环境变量） | 模型的 API 密钥 |
| `--max_turns` | `10` | 每条 prompt 的最大工具调用轮次 |
| `--num_workers` | `4` | 并行工作进程数 |
| `--resume` | `false` | 从断点恢复 |
| `--verbose` | `false` | 启用详细日志 |
| `--max_samples` | 全部 | 仅处理数据集中前 N 条样本 |
| `--max_tokens` | 模型默认值 | 每次模型响应的最大 token 数 |

### 供应商路由（OpenRouter）

| 参数 | 说明 |
|-----------|-------------|
| `--providers_allowed` | 允许的供应商，逗号分隔（例如 `"anthropic,openai"`） |
| `--providers_ignored` | 忽略的供应商，逗号分隔（例如 `"together,deepinfra"`） |
| `--providers_order` | 首选供应商顺序，逗号分隔 |
| `--provider_sort` | 按 `"price"`、`"throughput"` 或 `"latency"` 排序 |

### 推理控制

| 参数 | 说明 |
|-----------|-------------|
| `--reasoning_effort` | 推理力度：`none`、`minimal`、`low`、`medium`、`high`、`xhigh` |
| `--reasoning_disabled` | 完全禁用推理/思考 token |

### 高级选项

| 参数 | 说明 |
|-----------|-------------|
| `--ephemeral_system_prompt` | 执行时使用但**不**保存到轨迹中的系统 prompt |
| `--log_prefix_chars` | 日志预览中显示的字符数（默认：100） |
| `--prefill_messages_file` | 包含 few-shot 预填充消息的 JSON 文件路径 |

## 工具集分布

每条 prompt 会从一个**分布**中随机采样一组工具集。这确保训练数据覆盖多样化的工具组合。使用 `--list_distributions` 查看所有可用分布。

在当前实现中，分布为**每个独立工具集**分配一个概率。采样器对每个工具集独立进行伯努利抽样，并保证至少有一个工具集被启用。这与手工编写的预设组合表不同。

## 输出格式

所有输出写入 `data/<run_name>/`：

```text
data/my_run/
├── trajectories.jsonl    # 合并后的最终输出（所有批次合并）
├── batch_0.jsonl         # 各批次结果
├── batch_1.jsonl
├── ...
├── checkpoint.json       # 断点续跑检查点
└── statistics.json       # 汇总工具使用统计
```

### 轨迹格式

`trajectories.jsonl` 中每行是一个 JSON 对象：

```json
{
  "prompt_index": 42,
  "conversations": [
    {"from": "human", "value": "Write a function..."},
    {"from": "gpt", "value": "I'll create that function...",
     "tool_calls": [...]},
    {"from": "tool", "value": "..."},
    {"from": "gpt", "value": "Here's the completed function..."}
  ],
  "metadata": {
    "batch_num": 2,
    "timestamp": "2026-01-15T10:30:00",
    "model": "anthropic/claude-sonnet-4.6"
  },
  "completed": true,
  "partial": false,
  "api_calls": 3,
  "toolsets_used": ["terminal", "file"],
  "tool_stats": {
    "terminal": {"count": 2, "success": 2, "failure": 0},
    "read_file": {"count": 1, "success": 1, "failure": 0}
  },
  "tool_error_counts": {
    "terminal": 0,
    "read_file": 0
  }
}
```

`conversations` 字段使用类 ShareGPT 格式，包含 `from` 和 `value` 字段。工具统计信息经过规范化处理，所有可能的工具均以零值默认填充，确保各条记录的 schema 一致，兼容 HuggingFace 数据集格式。

## 断点续跑

批量运行器具备健壮的断点续跑机制以应对故障：

- **检查点文件：** 每批完成后保存，记录已完成的 prompt 索引
- **基于内容的恢复：** 使用 `--resume` 时，运行器扫描现有批次文件，通过实际文本内容（而非索引）匹配已完成的 prompt，即使数据集顺序发生变化也能正常恢复
- **失败的 prompt：** 只有成功完成的 prompt 才会被标记为已完成——失败的 prompt 在恢复时会重新尝试
- **批次合并：** 完成后，所有批次文件（包括之前运行的）会合并为单个 `trajectories.jsonl`

### 恢复流程

1. 扫描所有 `batch_*.jsonl` 文件，通过内容匹配找出已完成的 prompt
2. 过滤数据集，排除已完成的 prompt
3. 对剩余 prompt 重新分批
4. 仅处理剩余 prompt
5. 将所有批次文件（旧的 + 新的）合并为最终输出

## 质量过滤

批量运行器会自动进行质量过滤：

- **无推理过滤：** 所有 assistant 轮次均不包含推理内容（无 `<REASONING_SCRATCHPAD>` 或原生思考 token）的样本将被丢弃
- **损坏条目过滤：** 包含幻觉工具名称（不在有效工具列表中）的条目在最终合并时会被过滤掉
- **推理统计：** 跟踪整个运行过程中包含/不包含推理内容的轮次百分比

## 统计信息

完成后，运行器会打印全面的统计信息：

- **工具使用情况：** 每个工具的调用次数、成功/失败率
- **推理覆盖率：** 包含推理内容的 assistant 轮次百分比
- **丢弃样本数：** 因缺少推理内容而被过滤的样本数量
- **耗时：** 总处理时间

统计信息同时保存至 `statistics.json`，便于程序化分析。

## 使用场景

### 训练数据生成

生成多样化的工具使用轨迹用于微调：

```bash
python batch_runner.py \
    --dataset_file=data/coding_prompts.jsonl \
    --batch_size=20 \
    --run_name=coding_v1 \
    --model=anthropic/claude-sonnet-4.6 \
    --num_workers=8 \
    --distribution=default \
    --max_turns=15
```

### 模型评估

在标准化 prompt 集上评估模型的工具使用能力：

```bash
python batch_runner.py \
    --dataset_file=data/eval_suite.jsonl \
    --batch_size=10 \
    --run_name=eval_gpt4 \
    --model=openai/gpt-4o \
    --num_workers=4 \
    --max_turns=10
```

### 按 Prompt 指定容器镜像

对于需要特定环境的基准测试，每条 prompt 可以指定自己的容器镜像：

```jsonl
{"prompt": "Install numpy and compute eigenvalues of a 3x3 matrix", "image": "python:3.11-slim"}
{"prompt": "Compile this Rust program and run it", "image": "rust:1.75"}
{"prompt": "Set up a Node.js Express server", "image": "node:20-alpine", "cwd": "/app"}
```

批量运行器会在运行每条 prompt 前验证 Docker 镜像是否可访问。
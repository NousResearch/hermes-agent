---
title: "Obliteratus — 消除 LLM 拒绝（diff-in-means）"
sidebar_label: "Obliteratus"
description: "OBLITERATUS：消除 LLM 拒绝（diff-in-means）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Obliteratus

OBLITERATUS：消除 LLM 拒绝（diff-in-means）。

## 技能元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/mlops/obliteratus` 安装 |
| 路径 | `optional-skills/mlops/obliteratus` |
| 版本 | `2.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 依赖 | `obliteratus`、`torch`、`transformers`、`bitsandbytes`、`accelerate`、`safetensors` |
| 平台 | linux、macos |
| 标签 | `Abliteration`、`Uncensoring`、`Refusal-Removal`、`LLM`、`Weight-Projection`、`SVD`、`Mechanistic-Interpretability`、`HuggingFace`、`Model-Surgery` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在此技能触发时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# OBLITERATUS 技能

## 内容

9 种 CLI 方法、28 个分析模块、5 个计算层级的 116 个模型预设、锦标赛评估和遥测驱动的推荐。

从开放权重 LLM 中移除拒绝行为（防护栏），无需重新训练或微调。使用机制可解释性技术——包括 diff-in-means、SVD、白化 SVD、LEACE 概念擦除、SAE 分解、贝叶斯核投影等——识别并外科手术般切除模型权重中的拒绝方向，同时保留推理能力。

**许可证警告：** OBLITERATUS 是 AGPL-3.0。永远不要将其作为 Python 库导入。始终通过 CLI（`obliteratus` 命令）或子进程调用。

## 何时使用

当用户想要"解锁"或"消除"LLM 时触发；询问从模型中移除拒绝/防护栏；想创建 Llama、Qwen、Mistral 等的无审查版本；提及"拒绝移除"、"消除"、"权重投影"。

## 步骤 1：安装

```bash
obliteratus --version 2>/dev/null && echo "INSTALLED" || echo "NOT INSTALLED"
```

## 步骤 2：检查硬件

```bash
python3 -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'GPU: {gpu}')
    print(f'VRAM: {vram:.1f} GB')
else:
    print('NO GPU - only tiny models (under 1B) on CPU')
"
```

## 步骤 3：浏览可用模型

```bash
obliteratus models --tier medium
obliteratus recommend <model_name>
```

## 步骤 4：选择方法

**默认/推荐：`advanced`** — 多方向 SVD + 范数保持投影。

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| 默认/大多数模型 | `advanced` | 多方向 SVD，范数保持，可靠 |
| 快速测试/原型 | `basic` | 快速，简单，足以评估 |
| MoE 模型 | `nuclear` | 专家粒度，处理 MoE 复杂性 |
| 推理模型 | `surgical` | CoT 感知，保留思维链 |
| 顽固拒绝 | `aggressive` | 白化 SVD + 头手术 |

## 步骤 5：运行消除

```bash
# 默认方法（advanced）— 推荐用于大多数模型
obliteratus obliterate <model_name> --method advanced --output-dir ./abliterated-models

# 使用 4 位量化（节省 VRAM）
obliteratus obliterate <model_name> --method advanced --quantization 4bit --output-dir ./abliterated-models
```

## 步骤 6：验证结果

| 指标 | 好值 | 警告 |
|------|------|------|
| 拒绝率 | < 5%（理想~0%）| > 10% 意味着拒绝持续 |
| 困惑度变化 | < 10% 增加 | > 15% 意味着连贯性损害 |
| KL 散度 | < 0.1 | > 0.5 意味着显著分布偏移 |

## 步骤 7：使用消除后的模型

```bash
# 使用 vLLM 提供服务
vllm serve ./abliterated-models/<model>
```

## 常见陷阱

1. **不要将 `informed` 作为默认** — 它是实验性的且更慢。
2. **低于约 1B 的模型对消除响应不佳** — 拒绝行为浅薄且碎片化。
3. **始终检查困惑度** — 如果飙升 > 15%，模型已损坏。
4. **量化模型不能重新量化** — 先消除全精度模型，然后量化输出。
5. **推理模型敏感** — 对 R1 蒸馏使用 `surgical`。
6. **AGPL 许可证** — 永远不要在 MIT/Apache 项目中 `import obliteratus`。

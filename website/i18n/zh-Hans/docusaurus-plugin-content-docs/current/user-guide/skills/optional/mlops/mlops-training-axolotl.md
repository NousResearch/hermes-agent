---
title: "Axolotl — Axolotl：基于 YAML 的 LLM 微调（LoRA、DPO、GRPO）"
sidebar_label: "Axolotl"
description: "Axolotl：基于 YAML 的 LLM 微调（LoRA、DPO、GRPO）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Axolotl

Axolotl：基于 YAML 的 LLM 微调（LoRA、DPO、GRPO）。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/mlops/axolotl` 安装 |
| 路径 | `optional-skills/mlops/training/axolotl` |
| 版本 | `1.0.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖项 | `axolotl`, `torch`, `transformers`, `datasets`, `peft`, `accelerate`, `deepspeed` |
| 平台 | linux, macos |
| 标签 | `Fine-Tuning`, `Axolotl`, `LLM`, `LoRA`, `QLoRA`, `DPO`, `KTO`, `ORPO`, `GRPO`, `YAML`, `HuggingFace`, `DeepSpeed`, `Multimodal` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Axolotl Skill

## 内容概览

使用 Axolotl 微调 LLM 的专家指导 — YAML 配置、100+ 模型、LoRA/QLoRA、DPO/KTO/ORPO/GRPO、多模态支持。

基于官方文档生成的 axolotl 开发全面辅助。

## 何时使用此 Skill

以下情况应触发此 skill：
- 使用 axolotl 进行开发
- 询问 axolotl 功能或 API
- 实现 axolotl 解决方案
- 调试 axolotl 代码
- 学习 axolotl 最佳实践

## 快速参考

### 常用模式

**模式 1：** 若要验证训练任务是否具备可接受的数据传输速度，运行 NCCL Tests 有助于定位瓶颈，例如：

```
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 3
```

**模式 2：** 在 Axolotl yaml 中配置模型以使用 FSDP，例如：

```
fsdp_version: 2
fsdp_config:
  offload_params: true
  state_dict_type: FULL_STATE_DICT
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: LlamaDecoderLayer
  reshard_after_forward: true
```

**模式 3：** `context_parallel_size` 应为 GPU 总数的因数，例如：

```
context_parallel_size
```

**模式 4：** 例如：- 使用 8 块 GPU 且不启用序列并行时：每步处理 8 个不同批次 - 使用 8 块 GPU 且 `context_parallel_size=4` 时：每步仅处理 2 个不同批次（每个批次跨 4 块 GPU 拆分）- 若每块 GPU 的 `micro_batch_size` 为 2，全局批次大小将从 16 降至 4

```
context_parallel_size=4
```

**模式 5：** 在配置中设置 `save_compressed: true` 可启用压缩格式保存模型，效果如下：- 磁盘空间占用减少约 40% - 保持与 vLLM 的兼容性以加速推理 - 保持与 llmcompressor 的兼容性以进行进一步优化（例如：量化）

```
save_compressed: true
```

**模式 6：** 注意：无需将集成放置在 `integrations` 文件夹中。只要安装在 Python 环境的某个包中，可位于任意位置。参见此示例仓库：https://github.com/axolotl-ai-cloud/diff-transformer

```
integrations
```

**模式 7：** 同时处理单样本和批量数据。- 单样本：`sample['input_ids']` 为 `list[int]` - 批量数据：`sample['input_ids']` 为 `list[list[int]]`

```
utils.trainer.drop_long_seq(sample, sequence_len=2048, min_sequence_len=2)
```

### 代码示例模式

**示例 1**（python）：
```python
cli.cloud.modal_.ModalCloud(config, app=None)
```

**示例 2**（python）：
```python
cli.cloud.modal_.run_cmd(cmd, run_folder, volumes=None)
```

**示例 3**（python）：
```python
core.trainers.base.AxolotlTrainer(
    *_args,
    bench_data_collator=None,
    eval_data_collator=None,
    dataset_tags=None,
    **kwargs,
)
```

**示例 4**（python）：
```python
core.trainers.base.AxolotlTrainer.log(logs, start_time=None)
```

**示例 5**（python）：
```python
prompt_strategies.input_output.RawInputOutputPrompter()
```

## 参考文件

此 skill 在 `references/` 中包含完整文档：

- **api.md** - API 文档
- **dataset-formats.md** - Dataset-Formats 文档
- **other.md** - 其他文档

需要详细信息时，使用 `view` 读取特定参考文件。

## 使用此 Skill

### 初学者
从 `getting_started` 或 `tutorials` 参考文件入手，了解基础概念。

### 特定功能
使用对应分类的参考文件（api、guides 等）获取详细信息。

### 代码示例
上方快速参考部分包含从官方文档中提取的常用模式。

## 资源

### references/
从官方来源提取的有组织文档，包含：
- 详细说明
- 带语言标注的代码示例
- 原始文档链接
- 便于快速导航的目录

### scripts/
在此添加常见自动化任务的辅助脚本。

### assets/
在此添加模板、样板代码或示例项目。

## 说明

- 此 skill 由官方文档自动生成
- 参考文件保留了源文档的结构与示例
- 代码示例包含语言检测以提供更好的语法高亮
- 快速参考模式从文档中的常见用法示例中提取

## 更新

若要使用最新文档刷新此 skill：
1. 使用相同配置重新运行爬取程序
2. Skill 将以最新信息重新构建
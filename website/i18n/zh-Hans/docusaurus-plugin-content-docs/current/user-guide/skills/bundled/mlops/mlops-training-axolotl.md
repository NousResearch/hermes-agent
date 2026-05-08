---
title: "Axolotl — Axolotl：YAML LLM 微调（LoRA、DPO、GRPO）"
sidebar_label: "Axolotl"
description: "Axolotl：YAML LLM 微调（LoRA、DPO、GRPO）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Axolotl

Axolotl：YAML LLM 微调（LoRA、DPO、GRPO）。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/mlops/training/axolotl` |
| 版本 | `1.0.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖项 | `axolotl`, `torch`, `transformers`, `datasets`, `peft`, `accelerate`, `deepspeed` |
| 标签 | `微调`, `Axolotl`, `LLM`, `LoRA`, `QLoRA`, `DPO`, `KTO`, `ORPO`, `GRPO`, `YAML`, `HuggingFace`, `DeepSpeed`, `多模态` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此技能时加载的完整技能定义。这是技能激活时代理看到的指令内容。
:::

# Axolotl 技能

## 内容概览

使用 Axolotl 微调 LLM 的专家指南 — YAML 配置、100+ 模型、LoRA/QLoRA、DPO/KTO/ORPO/GRPO、多模态支持。

基于官方文档生成的 Axolotl 开发综合辅助资料。

## 何时使用此技能

此技能应在以下情况下触发：
- 使用 Axolotl 工作
- 询问 Axolotl 功能或 API
- 实现 Axolotl 解决方案
- 调试 Axolotl 代码
- 学习 Axolotl 最佳实践

## 快速参考

### 常见模式

**模式 1：** 要验证训练任务是否存在可接受的数据传输速度，运行 NCCL 测试可以帮助定位瓶颈，例如：

```
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 3
```

**模式 2：** 在 Axolotl yaml 中配置模型使用 FSDP。例如：

```
fsdp_version: 2
fsdp_config:
  offload_params: true
  state_dict_type: FULL_STATE_DICT
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: LlamaDecoderLayer
  reshard_after_forward: true
```

**模式 3：** context_parallel_size 应该是 GPU 总数的约数。例如：

```
context_parallel_size
```

**模式 4：** 例如：- 8 个 GPU 且无序列并行：每步处理 8 个不同的批次 - 8 个 GPU 且 context_parallel_size=4：每步仅处理 2 个不同的批次（每个拆分到 4 个 GPU） - 如果每个 GPU 的 micro_batch_size 为 2，全局批次大小从 16 减少到 4

```
context_parallel_size=4
```

**模式 5：** 在配置中设置 save_compressed: true 可以启用以压缩格式保存模型，其效果：- 减少约 40% 的磁盘空间使用 - 保持与 vLLM 的兼容性用于加速推理 - 保持与 llmcompressor 的兼容性用于进一步优化（例如：量化）

```
save_compressed: true
```

**模式 6：** 注意，不需要将集成放在 integrations 文件夹中。它可以位于任何位置，只要安装在 Python 环境的包中即可。参见此仓库示例：https://github.com/axolotl-ai-cloud/diff-transformer

```
integrations
```

**模式 7：** 同时处理单样本和批量数据。- 单样本：sample['input_ids'] 是 list[int] - 批量数据：sample['input_ids'] 是 list[list[int]]

```
utils.trainer.drop_long_seq(sample, sequence_len=2048, min_sequence_len=2)
```

### 示例代码模式

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

此技能包含 `references/` 中的完整文档：

- **api.md** - API 文档
- **dataset-formats.md** - 数据集格式文档
- **other.md** - 其他文档

需要详细信息时，使用 `view` 读取特定参考文件。

## 使用此技能

### 初学者
从 getting_started 或 tutorials 参考文件开始了解基础概念。

### 特定功能
使用相应的分类参考文件（api、guides 等）获取详细信息。

### 代码示例
上面的快速参考部分包含从官方文档中提取的常见模式。

## 资源

### references/
从官方来源提取的组织化文档。这些文件包含：
- 详细说明
- 带语言标注的代码示例
- 原始文档链接
- 目录用于快速导航

### scripts/
在此处添加用于常见自动化任务的辅助脚本。

### assets/
在此处添加模板、样板代码或示例项目。

## 注意事项

- 此技能是根据官方文档自动生成的
- 参考文件保留了源文档的结构和示例
- 代码示例包含语言检测以获得更好的语法高亮
- 快速参考模式从文档中的常见用例中提取

## 更新

要使用更新的文档刷新此技能：
1. 使用相同配置重新运行抓取器
2. 技能将使用最新信息重新构建

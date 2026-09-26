---
title: "Huggingface Accelerate — 最简分布式训练 API"
sidebar_label: "Huggingface Accelerate"
description: "最简分布式训练 API"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Huggingface Accelerate

最简分布式训练 API。仅需 4 行代码即可为任意 PyTorch 脚本添加分布式支持。统一的 DeepSpeed/FSDP/Megatron/DDP API。自动设备放置、混合精度（FP16/BF16/FP8）。交互式配置，单条启动命令。HuggingFace 生态系统标准。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/mlops/accelerate` 安装 |
| 路径 | `optional-skills/mlops/accelerate` |
| 版本 | `1.0.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖项 | `accelerate`, `torch`, `transformers` |
| 平台 | linux, macos, windows |
| 标签 | `Distributed Training`, `HuggingFace`, `Accelerate`, `DeepSpeed`, `FSDP`, `Mixed Precision`, `PyTorch`, `DDP`, `Unified API`, `Simple` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发该 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# HuggingFace Accelerate - 统一分布式训练

## 快速开始

Accelerate 将分布式训练简化为 4 行代码。

**安装**：
```bash
pip install accelerate
```

**转换 PyTorch 脚本**（4 行）：
```python
import torch
+ from accelerate import Accelerator

+ accelerator = Accelerator()

  model = torch.nn.Transformer()
  optimizer = torch.optim.Adam(model.parameters())
  dataloader = torch.utils.data.DataLoader(dataset)

+ model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

  for batch in dataloader:
      optimizer.zero_grad()
      loss = model(batch)
-     loss.backward()
+     accelerator.backward(loss)
      optimizer.step()
```

**运行**（单条命令）：
```bash
accelerate launch train.py
```

## 常见工作流

### 工作流 1：从单 GPU 到多 GPU

**原始脚本**：
```python
# train.py
import torch

model = torch.nn.Linear(10, 2).to('cuda')
optimizer = torch.optim.Adam(model.parameters())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

for epoch in range(10):
    for batch in dataloader:
        batch = batch.to('cuda')
        optimizer.zero_grad()
        loss = model(batch).mean()
        loss.backward()
        optimizer.step()
```

**使用 Accelerate**（新增 4 行）：
```python
# train.py
import torch
from accelerate import Accelerator  # +1

accelerator = Accelerator()  # +2

model = torch.nn.Linear(10, 2)
optimizer = torch.optim.Adam(model.parameters())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)  # +3

for epoch in range(10):
    for batch in dataloader:
        # 无需 .to('cuda') — 自动处理！
        optimizer.zero_grad()
        loss = model(batch).mean()
        accelerator.backward(loss)  # +4
        optimizer.step()
```

**配置**（交互式）：
```bash
accelerate config
```

**问题**：
- 使用哪种机器？（单/多 GPU/TPU/CPU）
- 机器数量？（1）
- 混合精度？（no/fp16/bf16/fp8）
- DeepSpeed？（no/yes）

**启动**（适用于任意配置）：
```bash
# 单 GPU
accelerate launch train.py

# 多 GPU（8 个 GPU）
accelerate launch --multi_gpu --num_processes 8 train.py

# 多节点
accelerate launch --multi_gpu --num_processes 16 \
  --num_machines 2 --machine_rank 0 \
  --main_process_ip $MASTER_ADDR \
  train.py
```

### 工作流 2：混合精度训练

**启用 FP16/BF16**：
```python
from accelerate import Accelerator

# FP16（带梯度缩放）
accelerator = Accelerator(mixed_precision='fp16')

# BF16（无缩放，更稳定）
accelerator = Accelerator(mixed_precision='bf16')

# FP8（H100+）
accelerator = Accelerator(mixed_precision='fp8')

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# 其余均自动处理！
for batch in dataloader:
    with accelerator.autocast():  # 可选，已自动完成
        loss = model(batch)
    accelerator.backward(loss)
```

### 工作流 3：DeepSpeed ZeRO 集成

**启用 DeepSpeed ZeRO-2**：
```python
from accelerate import Accelerator

accelerator = Accelerator(
    mixed_precision='bf16',
    deepspeed_plugin={
        "zero_stage": 2,  # ZeRO-2
        "offload_optimizer": False,
        "gradient_accumulation_steps": 4
    }
)

# 代码与之前完全相同！
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```

**或通过配置**：
```bash
accelerate config
# 选择：DeepSpeed → ZeRO-2
```

**deepspeed_config.json**：
```json
{
    "fp16": {"enabled": false},
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"},
        "allgather_bucket_size": 5e8,
        "reduce_bucket_size": 5e8
    }
}
```

**启动**：
```bash
accelerate launch --config_file deepspeed_config.json train.py
```

### 工作流 4：FSDP（全分片数据并行）

**启用 FSDP**：
```python
from accelerate import Accelerator, FullyShardedDataParallelPlugin

fsdp_plugin = FullyShardedDataParallelPlugin(
    sharding_strategy="FULL_SHARD",  # 等价于 ZeRO-3
    auto_wrap_policy="TRANSFORMER_AUTO_WRAP",
    cpu_offload=False
)

accelerator = Accelerator(
    mixed_precision='bf16',
    fsdp_plugin=fsdp_plugin
)

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```

**或通过配置**：
```bash
accelerate config
# 选择：FSDP → Full Shard → No CPU Offload
```

### 工作流 5：梯度累积

**累积梯度**：
```python
from accelerate import Accelerator

accelerator = Accelerator(gradient_accumulation_steps=4)

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for batch in dataloader:
    with accelerator.accumulate(model):  # 自动处理累积
        optimizer.zero_grad()
        loss = model(batch)
        accelerator.backward(loss)
        optimizer.step()
```

**有效批大小**：`batch_size * num_gpus * gradient_accumulation_steps`

## 与替代方案的对比

**适合使用 Accelerate 的场景**：
- 需要最简单的分布式训练方式
- 需要单脚本适配任意硬件
- 使用 HuggingFace 生态系统
- 需要灵活性（DDP/DeepSpeed/FSDP/Megatron）
- 需要快速原型开发

**核心优势**：
- **4 行代码**：代码改动极少
- **统一 API**：同一套代码适用于 DDP、DeepSpeed、FSDP、Megatron
- **自动化**：设备放置、混合精度、分片均自动处理
- **交互式配置**：无需手动配置启动器
- **单条启动命令**：适用于所有环境

**适合使用替代方案的场景**：
- **PyTorch Lightning**：需要回调机制、高层抽象
- **Ray Train**：多节点编排、超参数调优
- **DeepSpeed**：直接 API 控制、高级特性
- **原生 DDP**：最大控制权、最少抽象层

## 常见问题

**问题：设备放置错误**

不要手动移动到设备：
```python
# 错误
batch = batch.to('cuda')

# 正确
# Accelerate 在 prepare() 之后自动处理
```

**问题：梯度累积不生效**

使用上下文管理器：
```python
# 正确
with accelerator.accumulate(model):
    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()
```

**问题：分布式环境下的检查点保存**

使用 accelerator 方法：
```python
# 仅在主进程保存
if accelerator.is_main_process:
    accelerator.save_state('checkpoint/')

# 在所有进程上加载
accelerator.load_state('checkpoint/')
```

**问题：FSDP 结果不一致**

确保使用相同的随机种子：
```python
from accelerate.utils import set_seed
set_seed(42)
```

## 高级主题

**Megatron 集成**：张量并行、流水线并行和序列并行的配置，请参阅 [references/megatron-integration.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/accelerate/references/megatron-integration.md)。

**自定义插件**：创建自定义分布式插件及高级配置，请参阅 [references/custom-plugins.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/accelerate/references/custom-plugins.md)。

**性能调优**：性能分析、内存优化及最佳实践，请参阅 [references/performance.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/accelerate/references/performance.md)。

## 硬件要求

- **CPU**：支持（速度较慢）
- **单 GPU**：支持
- **多 GPU**：DDP（默认）、DeepSpeed 或 FSDP
- **多节点**：DDP、DeepSpeed、FSDP、Megatron
- **TPU**：支持
- **Apple MPS**：支持

**启动器要求**：
- **DDP**：`torch.distributed.run`（内置）
- **DeepSpeed**：`deepspeed`（pip install deepspeed）
- **FSDP**：PyTorch 1.12+（内置）
- **Megatron**：需自定义配置

## 资源

- 文档：https://huggingface.co/docs/accelerate
- GitHub：https://github.com/huggingface/accelerate
- 版本：1.11.0+
- 教程："Accelerate your scripts"
- 示例：https://github.com/huggingface/accelerate/tree/main/examples
- 使用方：HuggingFace Transformers、TRL、PEFT 及所有 HF 库
---
title: "Pytorch Lightning"
sidebar_label: "Pytorch Lightning"
description: "基于 PyTorch 的高层框架，提供 Trainer 类、自动分布式训练（DDP/FSDP/DeepSpeed）、回调系统及极简样板代码"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Pytorch Lightning

基于 PyTorch 的高层框架，提供 Trainer 类、自动分布式训练（DDP/FSDP/DeepSpeed）、回调（callbacks）系统及极简样板代码。同一套代码可从笔记本扩展至超级计算机。适用于希望以内置最佳实践编写整洁训练循环的场景。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/mlops/pytorch-lightning` 安装 |
| 路径 | `optional-skills/mlops/pytorch-lightning` |
| 版本 | `1.0.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖项 | `lightning`, `torch`, `transformers` |
| 平台 | linux, macos, windows |
| 标签 | `PyTorch Lightning`, `Training Framework`, `Distributed Training`, `DDP`, `FSDP`, `DeepSpeed`, `High-Level API`, `Callbacks`, `Best Practices`, `Scalable` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发该 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# PyTorch Lightning - 高层训练框架

## 快速开始

PyTorch Lightning 对 PyTorch 代码进行组织，在保持灵活性的同时消除样板代码。

**安装**：
```bash
pip install lightning
```

**将 PyTorch 转换为 Lightning**（3 步）：

```python
import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Step 1: Define LightningModule (organize your PyTorch code)
class LitModel(L.LightningModule):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10)
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)  # Auto-logged to TensorBoard
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Step 2: Create data
train_loader = DataLoader(train_dataset, batch_size=32)

# Step 3: Train with Trainer (handles everything else!)
trainer = L.Trainer(max_epochs=10, accelerator='gpu', devices=2)
model = LitModel()
trainer.fit(model, train_loader)
```

**就这些！** Trainer 负责处理：
- GPU/TPU/CPU 切换
- 分布式训练（DDP、FSDP、DeepSpeed）
- 混合精度（FP16、BF16）
- 梯度累积
- 检查点保存
- 日志记录
- 进度条

## 常见工作流

### 工作流 1：从 PyTorch 迁移到 Lightning

**原始 PyTorch 代码**：
```python
model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
model.to('cuda')

for epoch in range(max_epochs):
    for batch in train_loader:
        batch = batch.to('cuda')
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

**Lightning 版本**：
```python
class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MyModel()

    def training_step(self, batch, batch_idx):
        loss = self.model(batch)  # No .to('cuda') needed!
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

# Train
trainer = L.Trainer(max_epochs=10, accelerator='gpu')
trainer.fit(LitModel(), train_loader)
```

**优势**：40+ 行 → 15 行，无需设备管理，自动分布式

### 工作流 2：验证与测试

```python
class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MyModel()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        val_loss = nn.functional.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', val_loss)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        test_loss = nn.functional.cross_entropy(y_hat, y)
        self.log('test_loss', test_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Train with validation
trainer = L.Trainer(max_epochs=10)
trainer.fit(model, train_loader, val_loader)

# Test
trainer.test(model, test_loader)
```

**自动功能**：
- 默认每个 epoch 运行验证
- 指标自动记录到 TensorBoard
- 基于 val_loss 保存最优模型检查点

### 工作流 3：分布式训练（DDP）

```python
# Same code as single GPU!
model = LitModel()

# 8 GPUs with DDP (automatic!)
trainer = L.Trainer(
    accelerator='gpu',
    devices=8,
    strategy='ddp'  # Or 'fsdp', 'deepspeed'
)

trainer.fit(model, train_loader)
```

**启动**：
```bash
# Single command, Lightning handles the rest
python train.py
```

**无需任何改动**：
- 自动数据分发
- 梯度同步
- 多节点支持（只需设置 `num_nodes=2`）

### 工作流 4：用于监控的回调（Callbacks）

```python
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

# Create callbacks
checkpoint = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=3,
    filename='model-{epoch:02d}-{val_loss:.2f}'
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min'
)

lr_monitor = LearningRateMonitor(logging_interval='epoch')

# Add to Trainer
trainer = L.Trainer(
    max_epochs=100,
    callbacks=[checkpoint, early_stop, lr_monitor]
)

trainer.fit(model, train_loader, val_loader)
```

**效果**：
- 自动保存最优的 3 个模型
- 若 5 个 epoch 内无改善则提前停止
- 将学习率记录到 TensorBoard

### 工作流 5：学习率调度

```python
class LitModel(L.LightningModule):
    # ... (training_step, etc.)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        # Cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-5
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # Update per epoch
                'frequency': 1
            }
        }

# Learning rate auto-logged!
trainer = L.Trainer(max_epochs=100)
trainer.fit(model, train_loader)
```

## 何时使用与替代方案对比

**适合使用 PyTorch Lightning 的场景**：
- 希望代码整洁、结构清晰
- 需要生产级训练循环
- 在单 GPU、多 GPU、TPU 之间切换
- 希望使用内置回调和日志记录
- 团队协作（标准化结构）

**核心优势**：
- **有组织**：将研究代码与工程代码分离
- **自动化**：一行代码启用 DDP、FSDP、DeepSpeed
- **回调**：模块化训练扩展
- **可复现**：样板代码更少 = 更少 bug
- **经过验证**：每月下载量 100 万+，久经考验

**改用其他方案的场景**：
- **Accelerate**：对现有代码改动最小，灵活性更高
- **Ray Train**：多节点编排、超参数调优
- **原生 PyTorch**：最大控制权，适合学习目的
- **Keras**：TensorFlow 生态系统

## 常见问题

**问题：损失不下降**

检查数据和模型设置：
```python
# Add to training_step
def training_step(self, batch, batch_idx):
    if batch_idx == 0:
        print(f"Batch shape: {batch[0].shape}")
        print(f"Labels: {batch[1]}")
    loss = ...
    return loss
```

**问题：内存不足**

减小 batch size 或使用梯度累积：
```python
trainer = L.Trainer(
    accumulate_grad_batches=4,  # Effective batch = batch_size × 4
    precision='bf16'  # Or 'fp16', reduces memory 50%
)
```

**问题：验证未运行**

确保传入了 val_loader：
```python
# WRONG
trainer.fit(model, train_loader)

# CORRECT
trainer.fit(model, train_loader, val_loader)
```

**问题：DDP 意外启动多个进程**

Lightning 会自动检测 GPU。请显式设置 devices：
```python
# Test on CPU first
trainer = L.Trainer(accelerator='cpu', devices=1)

# Then GPU
trainer = L.Trainer(accelerator='gpu', devices=1)
```

## 进阶主题

**回调**：参见 [references/callbacks.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/pytorch-lightning/references/callbacks.md)，了解 EarlyStopping、ModelCheckpoint、自定义回调及回调钩子（hook）。

**分布式策略**：参见 [references/distributed.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/pytorch-lightning/references/distributed.md)，了解 DDP、FSDP、DeepSpeed ZeRO 集成及多节点配置。

**超参数调优**：参见 [references/hyperparameter-tuning.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/pytorch-lightning/references/hyperparameter-tuning.md)，了解与 Optuna、Ray Tune 及 WandB sweeps 的集成。

## 硬件要求

- **CPU**：支持（适合调试）
- **单 GPU**：支持
- **多 GPU**：DDP（默认）、FSDP 或 DeepSpeed
- **多节点**：DDP、FSDP、DeepSpeed
- **TPU**：支持（8 核）
- **Apple MPS**：支持

**精度选项**：
- FP32（默认）
- FP16（V100 及较旧 GPU）
- BF16（A100/H100，推荐）
- FP8（H100）

## 资源

- 文档：https://lightning.ai/docs/pytorch/stable/
- GitHub：https://github.com/Lightning-AI/pytorch-lightning ⭐ 29,000+
- 版本：2.5.5+
- 示例：https://github.com/Lightning-AI/pytorch-lightning/tree/master/examples
- Discord：https://discord.gg/lightning-ai
- 使用者：Kaggle 获奖者、科研实验室、生产团队
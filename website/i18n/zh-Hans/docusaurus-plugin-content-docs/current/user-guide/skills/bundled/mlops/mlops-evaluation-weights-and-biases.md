---
title: "Weights And Biases — W&B：记录 ML 实验、sweeps、模型注册表、仪表盘"
sidebar_label: "Weights And Biases"
description: "W&B：记录 ML 实验、sweeps、模型注册表、仪表盘"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Weights And Biases

W&B：记录 ML 实验、sweeps、模型注册表、仪表盘。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/mlops/evaluation/weights-and-biases` |
| 版本 | `1.0.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖 | `wandb` |
| 平台 | linux, macos, windows |
| 标签 | `MLOps`, `Weights And Biases`, `WandB`, `Experiment Tracking`, `Hyperparameter Tuning`, `Model Registry`, `Collaboration`, `Real-Time Visualization`, `PyTorch`, `TensorFlow`, `HuggingFace` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Weights & Biases：ML 实验追踪与 MLOps

## 适用场景

在以下情况下使用 Weights & Biases（W&B）：
- **追踪 ML 实验**，自动记录指标
- **实时仪表盘可视化**训练过程
- **跨超参数和配置对比运行结果**
- **自动化 sweeps 优化超参数**
- **管理模型注册表**，支持版本控制与血缘追踪
- **团队协作开展 ML 项目**，共享工作区
- **追踪 artifacts**（数据集、模型、代码）及其血缘关系

**用户数**：20 万+ ML 从业者 | **GitHub Stars**：10.5k+ | **集成数**：100+

## 安装

```bash
# 安装 W&B
pip install wandb

# 登录（创建 API key）
wandb login

# 或以编程方式设置 API key
export WANDB_API_KEY=your_api_key_here
```

## 快速开始

### 基础实验追踪

```python
import wandb

# 初始化一次运行
run = wandb.init(
    project="my-project",
    config={
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 32,
        "architecture": "ResNet50"
    }
)

# 训练循环
for epoch in range(run.config.epochs):
    # 你的训练代码
    train_loss = train_epoch()
    val_loss = validate()

    # 记录指标
    wandb.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "val/loss": val_loss,
        "train/accuracy": train_acc,
        "val/accuracy": val_acc
    })

# 结束运行
wandb.finish()
```

### 与 PyTorch 配合使用

```python
import torch
import wandb

# 初始化
wandb.init(project="pytorch-demo", config={
    "lr": 0.001,
    "epochs": 10
})

# 访问配置
config = wandb.config

# 训练循环
for epoch in range(config.epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # 前向传播
        output = model(data)
        loss = criterion(output, target)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每 100 个 batch 记录一次
        if batch_idx % 100 == 0:
            wandb.log({
                "loss": loss.item(),
                "epoch": epoch,
                "batch": batch_idx
            })

# 保存模型
torch.save(model.state_dict(), "model.pth")
wandb.save("model.pth")  # 上传至 W&B

wandb.finish()
```

## 核心概念

### 1. Projects 与 Runs

**Project**：相关实验的集合
**Run**：训练脚本的单次执行

```python
# 创建/使用 project
run = wandb.init(
    project="image-classification",
    name="resnet50-experiment-1",  # 可选的运行名称
    tags=["baseline", "resnet"],    # 使用标签组织
    notes="First baseline run"      # 添加备注
)

# 每次运行都有唯一 ID
print(f"Run ID: {run.id}")
print(f"Run URL: {run.url}")
```

### 2. 配置追踪

自动追踪超参数：

```python
config = {
    # 模型架构
    "model": "ResNet50",
    "pretrained": True,

    # 训练参数
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 50,
    "optimizer": "Adam",

    # 数据参数
    "dataset": "ImageNet",
    "augmentation": "standard"
}

wandb.init(project="my-project", config=config)

# 训练过程中访问配置
lr = wandb.config.learning_rate
batch_size = wandb.config.batch_size
```

### 3. 指标记录

```python
# 记录标量
wandb.log({"loss": 0.5, "accuracy": 0.92})

# 记录多个指标
wandb.log({
    "train/loss": train_loss,
    "train/accuracy": train_acc,
    "val/loss": val_loss,
    "val/accuracy": val_acc,
    "learning_rate": current_lr,
    "epoch": epoch
})

# 使用自定义 x 轴记录
wandb.log({"loss": loss}, step=global_step)

# 记录媒体（图像、音频、视频）
wandb.log({"examples": [wandb.Image(img) for img in images]})

# 记录直方图
wandb.log({"gradients": wandb.Histogram(gradients)})

# 记录表格
table = wandb.Table(columns=["id", "prediction", "ground_truth"])
wandb.log({"predictions": table})
```

### 4. 模型检查点

```python
import torch
import wandb

# 保存模型检查点
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}

torch.save(checkpoint, 'checkpoint.pth')

# 上传至 W&B
wandb.save('checkpoint.pth')

# 或使用 Artifacts（推荐）
artifact = wandb.Artifact('model', type='model')
artifact.add_file('checkpoint.pth')
wandb.log_artifact(artifact)
```

## 超参数 Sweeps

自动搜索最优超参数。

### 定义 Sweep 配置

```python
sweep_config = {
    'method': 'bayes',  # 或 'grid'、'random'
    'metric': {
        'name': 'val/accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform',
            'min': 1e-5,
            'max': 1e-1
        },
        'batch_size': {
            'values': [16, 32, 64, 128]
        },
        'optimizer': {
            'values': ['adam', 'sgd', 'rmsprop']
        },
        'dropout': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5
        }
    }
}

# 初始化 sweep
sweep_id = wandb.sweep(sweep_config, project="my-project")
```

### 定义训练函数

```python
def train():
    # 初始化运行
    run = wandb.init()

    # 访问 sweep 参数
    lr = wandb.config.learning_rate
    batch_size = wandb.config.batch_size
    optimizer_name = wandb.config.optimizer

    # 使用 sweep 配置构建模型
    model = build_model(wandb.config)
    optimizer = get_optimizer(optimizer_name, lr)

    # 训练循环
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, optimizer, batch_size)
        val_acc = validate(model)

        # 记录指标
        wandb.log({
            "train/loss": train_loss,
            "val/accuracy": val_acc
        })

# 运行 sweep
wandb.agent(sweep_id, function=train, count=50)  # 运行 50 次试验
```

### Sweep 策略

```python
# 网格搜索 - 穷举
sweep_config = {
    'method': 'grid',
    'parameters': {
        'lr': {'values': [0.001, 0.01, 0.1]},
        'batch_size': {'values': [16, 32, 64]}
    }
}

# 随机搜索
sweep_config = {
    'method': 'random',
    'parameters': {
        'lr': {'distribution': 'uniform', 'min': 0.0001, 'max': 0.1},
        'dropout': {'distribution': 'uniform', 'min': 0.1, 'max': 0.5}
    }
}

# 贝叶斯优化（推荐）
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val/loss', 'goal': 'minimize'},
    'parameters': {
        'lr': {'distribution': 'log_uniform', 'min': 1e-5, 'max': 1e-1}
    }
}
```

## Artifacts

追踪数据集、模型及其他文件的血缘关系。

### 记录 Artifacts

```python
# 创建 artifact
artifact = wandb.Artifact(
    name='training-dataset',
    type='dataset',
    description='ImageNet training split',
    metadata={'size': '1.2M images', 'split': 'train'}
)

# 添加文件
artifact.add_file('data/train.csv')
artifact.add_dir('data/images/')

# 记录 artifact
wandb.log_artifact(artifact)
```

### 使用 Artifacts

```python
# 下载并使用 artifact
run = wandb.init(project="my-project")

# 下载 artifact
artifact = run.use_artifact('training-dataset:latest')
artifact_dir = artifact.download()

# 使用数据
data = load_data(f"{artifact_dir}/train.csv")
```

### 模型注册表

```python
# 将模型记录为 artifact
model_artifact = wandb.Artifact(
    name='resnet50-model',
    type='model',
    metadata={'architecture': 'ResNet50', 'accuracy': 0.95}
)

model_artifact.add_file('model.pth')
wandb.log_artifact(model_artifact, aliases=['best', 'production'])

# 链接到模型注册表
run.link_artifact(model_artifact, 'model-registry/production-models')
```

## 集成示例

### HuggingFace Transformers

```python
from transformers import Trainer, TrainingArguments
import wandb

# 初始化 W&B
wandb.init(project="hf-transformers")

# 带 W&B 的训练参数
training_args = TrainingArguments(
    output_dir="./results",
    report_to="wandb",  # 启用 W&B 日志
    run_name="bert-finetuning",
    logging_steps=100,
    save_steps=500
)

# Trainer 自动记录至 W&B
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
```

### PyTorch Lightning

```python
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb

# 创建 W&B logger
wandb_logger = WandbLogger(
    project="lightning-demo",
    log_model=True  # 记录模型检查点
)

# 与 Trainer 配合使用
trainer = Trainer(
    logger=wandb_logger,
    max_epochs=10
)

trainer.fit(model, datamodule=dm)
```

### Keras/TensorFlow

```python
import wandb
from wandb.keras import WandbCallback

# 初始化
wandb.init(project="keras-demo")

# 添加回调
model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    callbacks=[WandbCallback()]  # 自动记录指标
)
```

## 可视化与分析

### 自定义图表

```python
# 记录自定义可视化
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(x, y)
wandb.log({"custom_plot": wandb.Image(fig)})

# 记录混淆矩阵
wandb.log({"conf_mat": wandb.plot.confusion_matrix(
    probs=None,
    y_true=ground_truth,
    preds=predictions,
    class_names=class_names
)})
```

### Reports

在 W&B UI 中创建可分享的报告：
- 组合运行结果、图表与文本
- 支持 Markdown
- 可嵌入的可视化内容
- 团队协作

## 最佳实践

### 1. 使用标签和分组进行组织

```python
wandb.init(
    project="my-project",
    tags=["baseline", "resnet50", "imagenet"],
    group="resnet-experiments",  # 对相关运行分组
    job_type="train"             # 任务类型
)
```

### 2. 记录所有相关信息

```python
# 记录系统指标
wandb.log({
    "gpu/util": gpu_utilization,
    "gpu/memory": gpu_memory_used,
    "cpu/util": cpu_utilization
})

# 记录代码版本
wandb.log({"git_commit": git_commit_hash})

# 记录数据划分
wandb.log({
    "data/train_size": len(train_dataset),
    "data/val_size": len(val_dataset)
})
```

### 3. 使用描述性名称

```python
# ✅ 好：描述性运行名称
wandb.init(
    project="nlp-classification",
    name="bert-base-lr0.001-bs32-epoch10"
)

# ❌ 差：通用名称
wandb.init(project="nlp", name="run1")
```

### 4. 保存重要 Artifacts

```python
# 保存最终模型
artifact = wandb.Artifact('final-model', type='model')
artifact.add_file('model.pth')
wandb.log_artifact(artifact)

# 保存预测结果以供分析
predictions_table = wandb.Table(
    columns=["id", "input", "prediction", "ground_truth"],
    data=predictions_data
)
wandb.log({"predictions": predictions_table})
```

### 5. 在网络不稳定时使用离线模式

```python
import os

# 启用离线模式
os.environ["WANDB_MODE"] = "offline"

wandb.init(project="my-project")
# ... 你的代码 ...

# 稍后同步
# wandb sync <run_directory>
```

## 团队协作

### 分享运行结果

```python
# 运行结果可通过 URL 自动分享
run = wandb.init(project="team-project")
print(f"Share this URL: {run.url}")
```

### 团队项目

- 在 wandb.ai 创建团队账号
- 添加团队成员
- 设置项目可见性（私有/公开）
- 使用团队级 artifacts 和模型注册表

## 定价

- **免费版**：无限公开项目，100GB 存储
- **学术版**：学生/研究人员免费使用
- **团队版**：$50/席位/月，私有项目，无限存储
- **企业版**：定制定价，支持本地部署

## 资源

- **文档**：https://docs.wandb.ai
- **GitHub**：https://github.com/wandb/wandb（10.5k+ stars）
- **示例**：https://github.com/wandb/examples
- **社区**：https://wandb.ai/community
- **Discord**：https://wandb.me/discord

## 另请参阅

- `references/sweeps.md` — 超参数优化综合指南
- `references/artifacts.md` — 数据与模型版本控制模式
- `references/integrations.md` — 框架专项示例
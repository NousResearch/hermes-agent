---
title: "Weights & Biases — W&B：记录 ML 实验、超参搜索、模型注册、仪表盘"
sidebar_label: "Weights & Biases"
description: "W&B：记录 ML 实验、超参搜索、模型注册、仪表盘"
---
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
W&B: log ML experiments, sweeps, model registry, dashboards.
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
## 技能元数据
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
| | |
|---|---|
| | |
| 路径 | `skills/mlops/evaluation/weights-and-biases` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `wandb` |
| 标签 | `MLOps、Weights & Biases、WandB、实验跟踪、超参数调优、模型注册、协作、实时可视化、PyTorch、TensorFlow、HuggingFace` |
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
## 参考：完整 SKILL.md
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. 这是代理在技能激活时看到的指令。
:::
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
## When to Use This Skill
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
Use Weights & Biases (W&B) when you need to:
- **Track ML experiments** with automatic metric logging
- **Visualize training** in real-time dashboards
- **Compare runs** across hyperparameters and configurations
- **Optimize hyperparameters** with automated sweeps
- **Manage model registry** with versioning and lineage
- **Collaborate on ML projects** with team workspaces
- **Track artifacts** (datasets, models, code) with lineage
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
**Users**: 200,000+ ML practitioners | **GitHub Stars**: 10.5k+ | **Integrations**: 100+
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
## Installation
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
```bash
# Weights & Biases
pip install wandb
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
wandb login
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
export WANDB_API_KEY=your_api_key_here
```
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
## Quick Start
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
### Basic Experiment Tracking
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
```python
import wandb
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
run = wandb.init(
    project="my-project",
    config={
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 32,
        "architecture": "ResNet50"
    }
)
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
for epoch in range(run.config.epochs):
    # Your training code
    train_loss = train_epoch()
    val_loss = validate()
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
    # Log metrics
    wandb.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "val/loss": val_loss,
        "train/accuracy": train_acc,
        "val/accuracy": val_acc
    })
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
wandb.finish()
```
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
### With PyTorch
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
```python
import torch
import wandb
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
wandb.init(project="pytorch-demo", config={
    "lr": 0.001,
    "epochs": 10
})
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
config = wandb.config
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
for epoch in range(config.epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
        # Log every 100 batches
        if batch_idx % 100 == 0:
            wandb.log({
                "loss": loss.item(),
                "epoch": epoch,
                "batch": batch_idx
            })
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
torch.save(model.state_dict(), "model.pth")
wandb.save("model.pth")  # Upload to W&B
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
wandb.finish()
```
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
## Core Concepts
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
### 1. Projects and Runs
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
**Project**: Collection of related experiments
**Run**: Single execution of your training script
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
```python
# Weights & Biases
run = wandb.init(
    project="image-classification",
    name="resnet50-experiment-1",  # Optional run name
    tags=["baseline", "resnet"],    # Organize with tags
    notes="First baseline run"      # Add notes
)
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
print(f"Run ID: {run.id}")
print(f"Run URL: {run.url}")
```
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
### 2. Configuration Tracking
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
Track hyperparameters automatically:
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
```python
config = {
    # Model architecture
    "model": "ResNet50",
    "pretrained": True,
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
    # Training params
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 50,
    "optimizer": "Adam",
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
    # Data params
    "dataset": "ImageNet",
    "augmentation": "standard"
}
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
wandb.init(project="my-project", config=config)
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
lr = wandb.config.learning_rate
batch_size = wandb.config.batch_size
```
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
### 3. Metric Logging
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
```python
# Weights & Biases
wandb.log({"loss": 0.5, "accuracy": 0.92})
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
wandb.log({
    "train/loss": train_loss,
    "train/accuracy": train_acc,
    "val/loss": val_loss,
    "val/accuracy": val_acc,
    "learning_rate": current_lr,
    "epoch": epoch
})
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
wandb.log({"loss": loss}, step=global_step)
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
wandb.log({"examples": [wandb.Image(img) for img in images]})
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
wandb.log({"gradients": wandb.Histogram(gradients)})
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
table = wandb.Table(columns=["id", "prediction", "ground_truth"])
wandb.log({"predictions": table})
```
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
### 4. Model Checkpointing
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
```python
import torch
import wandb
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
torch.save(checkpoint, 'checkpoint.pth')
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
wandb.save('checkpoint.pth')
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
artifact = wandb.Artifact('model', type='model')
artifact.add_file('checkpoint.pth')
wandb.log_artifact(artifact)
```
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
## Hyperparameter Sweeps
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
Automatically search for optimal hyperparameters.
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
### Define Sweep Configuration
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
```python
sweep_config = {
    'method': 'bayes',  # or 'grid', 'random'
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
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
sweep_id = wandb.sweep(sweep_config, project="my-project")
```
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
### Define Training Function
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
```python
def train():
    # Initialize run
    run = wandb.init()
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
    # Access sweep parameters
    lr = wandb.config.learning_rate
    batch_size = wandb.config.batch_size
    optimizer_name = wandb.config.optimizer
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
    # Build model with sweep config
    model = build_model(wandb.config)
    optimizer = get_optimizer(optimizer_name, lr)
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
    # Training loop
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, optimizer, batch_size)
        val_acc = validate(model)
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
        # Log metrics
        wandb.log({
            "train/loss": train_loss,
            "val/accuracy": val_acc
        })
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
wandb.agent(sweep_id, function=train, count=50)  # Run 50 trials
```
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
### Sweep Strategies
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
```python
# Weights & Biases
sweep_config = {
    'method': 'grid',
    'parameters': {
        'lr': {'values': [0.001, 0.01, 0.1]},
        'batch_size': {'values': [16, 32, 64]}
    }
}
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
sweep_config = {
    'method': 'random',
    'parameters': {
        'lr': {'distribution': 'uniform', 'min': 0.0001, 'max': 0.1},
        'dropout': {'distribution': 'uniform', 'min': 0.1, 'max': 0.5}
    }
}
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val/loss', 'goal': 'minimize'},
    'parameters': {
        'lr': {'distribution': 'log_uniform', 'min': 1e-5, 'max': 1e-1}
    }
}
```
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
## Artifacts
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
Track datasets, models, and other files with lineage.
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
### Log Artifacts
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
```python
# Weights & Biases
artifact = wandb.Artifact(
    name='training-dataset',
    type='dataset',
    description='ImageNet training split',
    metadata={'size': '1.2M images', 'split': 'train'}
)
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
artifact.add_file('data/train.csv')
artifact.add_dir('data/images/')
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
wandb.log_artifact(artifact)
```
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
### Use Artifacts
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
```python
# Weights & Biases
run = wandb.init(project="my-project")
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
artifact = run.use_artifact('training-dataset:latest')
artifact_dir = artifact.download()
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
data = load_data(f"{artifact_dir}/train.csv")
```
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
### Model Registry
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
```python
# Weights & Biases
model_artifact = wandb.Artifact(
    name='resnet50-model',
    type='model',
    metadata={'architecture': 'ResNet50', 'accuracy': 0.95}
)
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
model_artifact.add_file('model.pth')
wandb.log_artifact(model_artifact, aliases=['best', 'production'])
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
run.link_artifact(model_artifact, 'model-registry/production-models')
```
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
## Integration Examples
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
### HuggingFace Transformers
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
```python
from transformers import Trainer, TrainingArguments
import wandb
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
wandb.init(project="hf-transformers")
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
training_args = TrainingArguments(
    output_dir="./results",
    report_to="wandb",  # Enable W&B logging
    run_name="bert-finetuning",
    logging_steps=100,
    save_steps=500
)
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
trainer.train()
```
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
### PyTorch Lightning
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
```python
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
wandb_logger = WandbLogger(
    project="lightning-demo",
    log_model=True  # Log model checkpoints
)
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
trainer = Trainer(
    logger=wandb_logger,
    max_epochs=10
)
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
trainer.fit(model, datamodule=dm)
```
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
### Keras/TensorFlow
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
```python
import wandb
from wandb.keras import WandbCallback
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
wandb.init(project="keras-demo")
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    callbacks=[WandbCallback()]  # Auto-logs metrics
)
```
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
## Visualization & Analysis
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
### Custom Charts
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
```python
# Weights & Biases
import matplotlib.pyplot as plt
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
fig, ax = plt.subplots()
ax.plot(x, y)
wandb.log({"custom_plot": wandb.Image(fig)})
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
wandb.log({"conf_mat": wandb.plot.confusion_matrix(
    probs=None,
    y_true=ground_truth,
    preds=predictions,
    class_names=class_names
)})
```
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
### Reports
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
Create shareable reports in W&B UI:
- Combine runs, charts, and text
- Markdown support
- Embeddable visualizations
- Team collaboration
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
## Best Practices
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
### 1. Organize with Tags and Groups
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
```python
wandb.init(
    project="my-project",
    tags=["baseline", "resnet50", "imagenet"],
    group="resnet-experiments",  # Group related runs
    job_type="train"             # Type of job
)
```
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
### 2. Log Everything Relevant
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
```python
# Weights & Biases
wandb.log({
    "gpu/util": gpu_utilization,
    "gpu/memory": gpu_memory_used,
    "cpu/util": cpu_utilization
})
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
wandb.log({"git_commit": git_commit_hash})
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
wandb.log({
    "data/train_size": len(train_dataset),
    "data/val_size": len(val_dataset)
})
```
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
### 3. Use Descriptive Names
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
```python
# Weights & Biases
wandb.init(
    project="nlp-classification",
    name="bert-base-lr0.001-bs32-epoch10"
)
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
wandb.init(project="nlp", name="run1")
```
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
### 4. Save Important Artifacts
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
```python
# Weights & Biases
artifact = wandb.Artifact('final-model', type='model')
artifact.add_file('model.pth')
wandb.log_artifact(artifact)
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
predictions_table = wandb.Table(
    columns=["id", "input", "prediction", "ground_truth"],
    data=predictions_data
)
wandb.log({"predictions": predictions_table})
```
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
### 5. Use Offline Mode for Unstable Connections
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
```python
import os
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
os.environ["WANDB_MODE"] = "offline"
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
wandb.init(project="my-project")
# Weights & Biases
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
# Weights & Biases
# Weights & Biases
```
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
## Team Collaboration
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
### Share Runs
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
```python
# Weights & Biases
run = wandb.init(project="team-project")
print(f"Share this URL: {run.url}")
```
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
### Team Projects
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
- Create team account at wandb.ai
- Add team members
- Set project visibility (private/public)
- Use team-level artifacts and model registry
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
## Pricing
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
- **Free**: Unlimited public projects, 100GB storage
- **Academic**: Free for students/researchers
- **Teams**: $50/seat/month, private projects, unlimited storage
- **Enterprise**: Custom pricing, on-prem options
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
## Resources
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
- **Documentation**: https://docs.wandb.ai
- **GitHub**: https://github.com/wandb/wandb (10.5k+ stars)
- **Examples**: https://github.com/wandb/examples
- **Community**: https://wandb.ai/community
- **Discord**: https://wandb.me/discord
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
## See Also
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
- `references/sweeps.md` - Comprehensive hyperparameter optimization guide
- `references/artifacts.md` - Data and model versioning patterns
- `references/integrations.md` - Framework-specific examples
W&B：记录 ML 实验、超参搜索、模型注册、仪表盘。
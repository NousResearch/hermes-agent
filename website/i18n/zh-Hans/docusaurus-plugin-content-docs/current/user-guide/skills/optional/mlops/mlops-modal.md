---
title: "Modal Serverless Gpu — 用于运行 ML 工作负载的无服务器 GPU 云平台"
sidebar_label: "Modal Serverless Gpu"
description: "用于运行 ML 工作负载的无服务器 GPU 云平台"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Modal Serverless Gpu

用于运行 ML 工作负载的无服务器 GPU 云平台。适用于需要按需 GPU 访问而无需管理基础设施、将 ML 模型部署为 API，或运行具有自动扩缩容的批处理作业的场景。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/mlops/modal` 安装 |
| 路径 | `optional-skills/mlops/modal` |
| 版本 | `1.0.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖 | `modal>=0.64.0` |
| 平台 | linux, macos, windows |
| 标签 | `Infrastructure`, `Serverless`, `GPU`, `Cloud`, `Deployment`, `Modal` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Modal Serverless GPU

在 Modal 无服务器 GPU 云平台上运行 ML 工作负载的完整指南。

## 何时使用 Modal

**在以下情况下使用 Modal：**
- 运行 GPU 密集型 ML 工作负载而无需管理基础设施
- 将 ML 模型部署为自动扩缩容 API
- 运行批处理作业（训练、推理、数据处理）
- 需要按秒计费的 GPU 定价，无空闲成本
- 快速原型化 ML 应用
- 运行定时作业（类 cron 工作负载）

**主要特性：**
- **无服务器 GPU**：按需提供 T4、L4、A10G、L40S、A100、H100、H200、B200
- **Python 原生**：用 Python 代码定义基础设施，无需 YAML
- **自动扩缩容**：缩容至零，或瞬间扩容至 100+ 个 GPU
- **亚秒级冷启动**：基于 Rust 的基础设施，实现快速容器启动
- **容器缓存**：镜像层缓存，支持快速迭代
- **Web 端点**：将函数部署为 REST API，支持零停机更新

**以下情况请使用替代方案：**
- **RunPod**：适用于需要持久状态的长时间运行 pod
- **Lambda Labs**：适用于预留 GPU 实例
- **SkyPilot**：适用于多云编排和成本优化
- **Kubernetes**：适用于复杂的多服务架构

## 快速开始

### 安装

```bash
pip install modal
modal setup  # Opens browser for authentication
```

### GPU Hello World

```python
import modal

app = modal.App("hello-gpu")

@app.function(gpu="T4")
def gpu_info():
    import subprocess
    return subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout

@app.local_entrypoint()
def main():
    print(gpu_info.remote())
```

运行：`modal run hello_gpu.py`

### 基础推理端点

```python
import modal

app = modal.App("text-generation")
image = modal.Image.debian_slim().pip_install("transformers", "torch", "accelerate")

@app.cls(gpu="A10G", image=image)
class TextGenerator:
    @modal.enter()
    def load_model(self):
        from transformers import pipeline
        self.pipe = pipeline("text-generation", model="gpt2", device=0)

    @modal.method()
    def generate(self, prompt: str) -> str:
        return self.pipe(prompt, max_length=100)[0]["generated_text"]

@app.local_entrypoint()
def main():
    print(TextGenerator().generate.remote("Hello, world"))
```

## 核心概念

### 关键组件

| 组件 | 用途 |
|-----------|---------|
| `App` | 函数和资源的容器 |
| `Function` | 带计算规格的无服务器函数 |
| `Cls` | 带生命周期 hook 的基于类的函数 |
| `Image` | 容器镜像定义 |
| `Volume` | 用于模型/数据的持久存储 |
| `Secret` | 安全凭证存储 |

### 执行模式

| 命令 | 描述 |
|---------|-------------|
| `modal run script.py` | 执行后退出 |
| `modal serve script.py` | 开发模式，支持热重载 |
| `modal deploy script.py` | 持久化云端部署 |

## GPU 配置

### 可用 GPU

| GPU | 显存 | 最适用于 |
|-----|------|----------|
| `T4` | 16GB | 经济型推理、小型模型 |
| `L4` | 24GB | 推理，Ada Lovelace 架构 |
| `A10G` | 24GB | 训练/推理，比 T4 快 3.3 倍 |
| `L40S` | 48GB | 推荐用于推理（最佳性价比） |
| `A100-40GB` | 40GB | 大型模型训练 |
| `A100-80GB` | 80GB | 超大型模型 |
| `H100` | 80GB | 最快，支持 FP8 + Transformer Engine |
| `H200` | 141GB | 从 H100 自动升级，4.8TB/s 带宽 |
| `B200` | 最新 | Blackwell 架构 |

### GPU 规格配置模式

```python
# Single GPU
@app.function(gpu="A100")

# Specific memory variant
@app.function(gpu="A100-80GB")

# Multiple GPUs (up to 8)
@app.function(gpu="H100:4")

# GPU with fallbacks
@app.function(gpu=["H100", "A100", "L40S"])

# Any available GPU
@app.function(gpu="any")
```

## 容器镜像

```python
# Basic image with pip
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.1.0", "transformers==4.36.0", "accelerate"
)

# From CUDA base
image = modal.Image.from_registry(
    "nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04",
    add_python="3.11"
).pip_install("torch", "transformers")

# With system packages
image = modal.Image.debian_slim().apt_install("git", "ffmpeg").pip_install("whisper")
```

## 持久存储

```python
volume = modal.Volume.from_name("model-cache", create_if_missing=True)

@app.function(gpu="A10G", volumes={"/models": volume})
def load_model():
    import os
    model_path = "/models/llama-7b"
    if not os.path.exists(model_path):
        model = download_model()
        model.save_pretrained(model_path)
        volume.commit()  # Persist changes
    return load_from_path(model_path)
```

## Web 端点

### FastAPI 端点装饰器

```python
@app.function()
@modal.fastapi_endpoint(method="POST")
def predict(text: str) -> dict:
    return {"result": model.predict(text)}
```

### 完整 ASGI 应用

```python
from fastapi import FastAPI
web_app = FastAPI()

@web_app.post("/predict")
async def predict(text: str):
    return {"result": await model.predict.remote.aio(text)}

@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app
```

### Web 端点类型

| 装饰器 | 使用场景 |
|-----------|----------|
| `@modal.fastapi_endpoint()` | 简单函数 → API |
| `@modal.asgi_app()` | 完整 FastAPI/Starlette 应用 |
| `@modal.wsgi_app()` | Django/Flask 应用 |
| `@modal.web_server(port)` | 任意 HTTP 服务器 |

## 动态批处理

```python
@app.function()
@modal.batched(max_batch_size=32, wait_ms=100)
async def batch_predict(inputs: list[str]) -> list[dict]:
    # Inputs automatically batched
    return model.batch_predict(inputs)
```

## 密钥管理

```bash
# Create secret
modal secret create huggingface HF_TOKEN=hf_xxx
```

```python
@app.function(secrets=[modal.Secret.from_name("huggingface")])
def download_model():
    import os
    token = os.environ["HF_TOKEN"]
```

## 定时任务

```python
@app.function(schedule=modal.Cron("0 0 * * *"))  # Daily midnight
def daily_job():
    pass

@app.function(schedule=modal.Period(hours=1))
def hourly_job():
    pass
```

## 性能优化

### 冷启动缓解

```python
@app.function(
    container_idle_timeout=300,  # Keep warm 5 min
    allow_concurrent_inputs=10,  # Handle concurrent requests
)
def inference():
    pass
```

### 模型加载最佳实践

```python
@app.cls(gpu="A100")
class Model:
    @modal.enter()  # Run once at container start
    def load(self):
        self.model = load_model()  # Load during warm-up

    @modal.method()
    def predict(self, x):
        return self.model(x)
```

## 并行处理

```python
@app.function()
def process_item(item):
    return expensive_computation(item)

@app.function()
def run_parallel():
    items = list(range(1000))
    # Fan out to parallel containers
    results = list(process_item.map(items))
    return results
```

## 常用配置

```python
@app.function(
    gpu="A100",
    memory=32768,              # 32GB RAM
    cpu=4,                     # 4 CPU cores
    timeout=3600,              # 1 hour max
    container_idle_timeout=120,# Keep warm 2 min
    retries=3,                 # Retry on failure
    concurrency_limit=10,      # Max concurrent containers
)
def my_function():
    pass
```

## 调试

```python
# Test locally
if __name__ == "__main__":
    result = my_function.local()

# View logs
# modal app logs my-app
```

## 常见问题

| 问题 | 解决方案 |
|-------|----------|
| 冷启动延迟 | 增大 `container_idle_timeout`，使用 `@modal.enter()` |
| GPU 内存溢出 | 使用更大 GPU（`A100-80GB`），启用梯度检查点 |
| 镜像构建失败 | 固定依赖版本，检查 CUDA 兼容性 |
| 超时错误 | 增大 `timeout`，添加检查点 |

## 参考资料

- **[高级用法](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/modal/references/advanced-usage.md)** - 多 GPU、分布式训练、成本优化
- **[故障排查](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/modal/references/troubleshooting.md)** - 常见问题与解决方案

## 资源

- **文档**：https://modal.com/docs
- **示例**：https://github.com/modal-labs/modal-examples
- **定价**：https://modal.com/pricing
- **Discord**：https://discord.gg/modal
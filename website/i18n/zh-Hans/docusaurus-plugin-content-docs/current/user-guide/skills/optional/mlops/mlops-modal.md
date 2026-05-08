---
title: "Modal 无服务器 Gpu — 用于运行 ML 工作负载的无服务器 GPU 云平台"
sidebar_label: "Modal 无服务器 Gpu"
description: "用于运行 ML 工作负载的无服务器 GPU 云平台"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Modal 无服务器 Gpu

用于运行 ML 工作负载的无服务器 GPU 云平台. 适用于需要按需 GPU 访问而无需管理基础设施、将 ML 模型部署为 API 或运行自动扩缩的批处理作业。

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/mlops/modal` |
| Path | `optional-skills/mlops/modal` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `modal>=0.64.0` |
| Tags | `Infrastructure`, `无服务器`, `GPU`, `Cloud`, `Deployment`, `Modal` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Modal 无服务器 GPU

Comprehensive guide to running ML workloads on Modal's serverless GPU cloud platform.

## 何时使用 Modal

**Use Modal when:**
- Running GPU-intensive ML workloads without managing infrastructure
- Deploying ML models as auto-scaling APIs
- Running batch processing jobs (training, inference, data processing)
- Need pay-per-second GPU pricing without idle costs
- Prototyping ML applications quickly
- Running scheduled jobs (cron-like workloads)

**Key features:**
- **无服务器 GPUs**: T4, L4, A10G, L40S, A100, H100, H200, B200 on-demand
- **Python-native**: Define infrastructure in Python code, no YAML
- **Auto-scaling**: Scale to zero, scale to 100+ GPUs instantly
- **Sub-second cold starts**: Rust-based infrastructure for fast container launches
- **Container caching**: Image layers cached for rapid iteration
- **Web 端点**: Deploy functions as REST APIs with zero-downtime updates

**替代方案:**
- **RunPod**: For longer-running pods with persistent state
- **Lambda Labs**: For reserved GPU 实例
- **SkyPilot**: For multi-cloud orchestration and cost optimization
- **Kubernetes**: For complex multi-service architectures

## 快速入门

### 安装

```bash
pip install modal
modal setup  # Opens browser for authentication
```

### 使用 GPU 的 Hello World

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

Run: `modal run hello_gpu.py`

### 基本推理端点

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

## Core concepts

### 关键组件

| 组件 | 用途 |
|-----------|---------|
| `App` | Container for functions and resources |
| `Function` | 无服务器 function with compute specs |
| `Cls` | Class-based functions with lifecycle hooks |
| `Image` | Container image definition |
| `Volume` | 持久存储 for models/data |
| `Secret` | Secure credential storage |

### 执行模式

| 命令 | 描述 |
|---------|-------------|
| `modal run script.py` | Execute and exit |
| `modal serve script.py` | Development with live reload |
| `modal deploy script.py` | Persistent cloud deployment |

## GPU configuration

### 可用 GPU

| GPU | 显存 | 最适合 |
|-----|------|----------|
| `T4` | 16GB | Budget inference, small models |
| `L4` | 24GB | Inference, Ada Lovelace arch |
| `A10G` | 24GB | 训练ing/inference, 3.3x faster than T4 |
| `L40S` | 48GB | 推荐 for inference (best cost/perf) |
| `A100-40GB` | 40GB | Large model training |
| `A100-80GB` | 80GB | Very large models |
| `H100` | 80GB | Fastest, FP8 + Transformer Engine |
| `H200` | 141GB | Auto-upgrade from H100, 4.8TB/s bandwidth |
| `B200` | Latest | Blackwell architecture |

### GPU 规格模式

```python
# 单 GPU
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

# 来自 CUDA base
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

| 装饰器 | 用例 |
|-----------|----------|
| `@modal.fastapi_endpoint()` | Simple function → API |
| `@modal.asgi_app()` | Full FastAPI/Starlette apps |
| `@modal.wsgi_app()` | Django/Flask apps |
| `@modal.web_server(port)` | Arbitrary HTTP servers |

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

## 调度

```python
@app.function(schedule=modal.Cron("0 0 * * *"))  # Daily midnight
def daily_job():
    pass

@app.function(schedule=modal.Period(hours=1))
def hourly_job():
    pass
```

## Performance optimization

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
# 测试 locally
if __name__ == "__main__":
    result = my_function.local()

# View logs
# modal app logs my-app
```

## Common issues

| 问题 | 解决方案 |
|-------|----------|
| Cold start latency | Increase `container_idle_timeout`, use `@modal.enter()` |
| GPU OOM | Use larger GPU (`A100-80GB`), enable gradient checkpointing |
| Image build fails | Pin dependency versions, check CUDA compatibility |
| Timeout errors | Increase `timeout`, add checkpointing |

## References

- **[Advanced Usage](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/modal/references/advanced-usage.md)** - 多 GPU, distributed training, cost optimization
- **[Troubleshooting](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/modal/references/troubleshooting.md)** - Common issues and solutions

## Resources

- **Documentation**: https://modal.com/docs
- **Examples**: https://github.com/modal-labs/modal-examples
- **Pricing**: https://modal.com/pricing
- **Discord**: https://discord.gg/modal

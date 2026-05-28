---
title: "Llama Cpp — llama"
sidebar_label: "Llama Cpp"
description: "llama"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Llama Cpp

llama.cpp 本地 GGUF 推理 + HF Hub 模型发现。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/mlops/inference/llama-cpp` |
| 版本 | `2.1.2` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖 | `llama-cpp-python>=0.2.0` |
| 平台 | linux, macos, windows |
| 标签 | `llama.cpp`, `GGUF`, `Quantization`, `Hugging Face Hub`, `CPU Inference`, `Apple Silicon`, `Edge Deployment`, `AMD GPUs`, `Intel GPUs`, `NVIDIA`, `URL-first` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# llama.cpp + GGUF

本 skill 用于本地 GGUF 推理、量化（Quantization）选择，以及 Hugging Face 仓库发现（用于 llama.cpp）。

## 使用场景

- 在 CPU、Apple Silicon、CUDA、ROCm 或 Intel GPU 上运行本地模型
- 为特定 Hugging Face 仓库找到合适的 GGUF 文件
- 从 Hub 构建 `llama-server` 或 `llama-cli` 命令
- 在 Hub 上搜索已支持 llama.cpp 的模型
- 枚举某个仓库中可用的 `.gguf` 文件及其大小
- 根据用户的 RAM 或 VRAM 在 Q4/Q5/Q6/IQ 变体之间做出选择

## 模型发现工作流

优先使用 URL 工作流，再考虑 `hf`、Python 或自定义脚本。

1. 在 Hub 上搜索候选仓库：
   - 基础地址：`https://huggingface.co/models?apps=llama.cpp&sort=trending`
   - 添加 `search=<term>` 以搜索特定模型系列
   - 当用户有参数量限制时，添加 `num_parameters=min:0,max:24B` 或类似参数
2. 使用 llama.cpp 本地应用视图打开仓库：
   - `https://huggingface.co/<repo>?local-app=llama.cpp`
3. 当 local-app 代码片段可见时，将其作为权威来源：
   - 复制完整的 `llama-server` 或 `llama-cli` 命令
   - 严格按照 HF 显示的推荐量化标签进行报告
4. 将同一 `?local-app=llama.cpp` URL 作为页面文本或 HTML 读取，并提取 `Hardware compatibility` 部分：
   - 优先使用其中的精确量化标签和大小，而非通用表格
   - 保留仓库特有的标签，如 `UD-Q4_K_M` 或 `IQ4_NL_XL`
   - 如果该部分在获取的页面源码中不可见，请说明并回退到 tree API 加通用量化指导
5. 查询 tree API 以确认实际存在的文件：
   - `https://huggingface.co/api/models/<repo>/tree/main?recursive=true`
   - 保留 `type` 为 `file` 且 `path` 以 `.gguf` 结尾的条目
   - 以 `path` 和 `size` 作为文件名和字节大小的权威来源
   - 将量化检查点与 `mmproj-*.gguf` 投影文件及 `BF16/` 分片文件分开处理
   - 仅将 `https://huggingface.co/<repo>/tree/main` 作为人工备用方案
6. 如果 local-app 代码片段不可见，则从仓库和所选量化重建命令：
   - 简写量化选择：`llama-server -hf <repo>:<QUANT>`
   - 精确文件备用：`llama-server --hf-repo <repo> --hf-file <filename.gguf>`
7. 仅当仓库未暴露 GGUF 文件时，才建议从 Transformers 权重进行转换。

## 快速开始

### 安装 llama.cpp

```bash
# macOS / Linux（最简方式）
brew install llama.cpp
```

```bash
winget install llama.cpp
```

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release
```

### 直接从 Hugging Face Hub 运行

```bash
llama-cli -hf bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0
```

```bash
llama-server -hf bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0
```

### 从 Hub 运行精确的 GGUF 文件

当 tree API 显示自定义文件命名或缺少精确 HF 代码片段时使用此方式。

```bash
llama-server \
    --hf-repo microsoft/Phi-3-mini-4k-instruct-gguf \
    --hf-file Phi-3-mini-4k-instruct-q4.gguf \
    -c 4096
```

### OpenAI 兼容服务器检查

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Write a limerick about Python exceptions"}
    ]
  }'
```

## Python 绑定（llama-cpp-python）

`pip install llama-cpp-python`（CUDA：`CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir`；Metal：`CMAKE_ARGS="-DGGML_METAL=on" ...`）。

### 基础生成

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./model-q4_k_m.gguf",
    n_ctx=4096,
    n_gpu_layers=35,     # 0 为 CPU，99 为全部卸载到 GPU
    n_threads=8,
)

out = llm("What is machine learning?", max_tokens=256, temperature=0.7)
print(out["choices"][0]["text"])
```

### 对话 + 流式输出

```python
llm = Llama(
    model_path="./model-q4_k_m.gguf",
    n_ctx=4096,
    n_gpu_layers=35,
    chat_format="llama-3",   # 或 "chatml"、"mistral" 等
)

resp = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
    ],
    max_tokens=256,
)
print(resp["choices"][0]["message"]["content"])

# 流式输出
for chunk in llm("Explain quantum computing:", max_tokens=256, stream=True):
    print(chunk["choices"][0]["text"], end="", flush=True)
```

### Embedding（嵌入向量）

```python
llm = Llama(model_path="./model-q4_k_m.gguf", embedding=True, n_gpu_layers=35)
vec = llm.embed("This is a test sentence.")
print(f"Embedding dimension: {len(vec)}")
```

也可以直接从 Hub 加载 GGUF：

```python
llm = Llama.from_pretrained(
    repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
    filename="*Q4_K_M.gguf",
    n_gpu_layers=35,
)
```

## 选择量化方案

优先参考 Hub 页面，其次使用通用启发式规则。

- 优先使用 HF 标记为与用户硬件配置兼容的精确量化方案。
- 一般对话场景，从 `Q4_K_M` 开始。
- 代码或技术工作，若内存允许，优先选择 `Q5_K_M` 或 `Q6_K`。
- RAM 非常紧张时，仅在用户明确将适配性置于质量之上时，才考虑 `Q3_K_M`、`IQ` 变体或 `Q2` 变体。
- 对于多模态仓库，单独说明 `mmproj-*.gguf`。投影文件不是主模型文件。
- 不要规范化仓库原生标签。如果页面显示 `UD-Q4_K_M`，就报告 `UD-Q4_K_M`。

## 从仓库提取可用的 GGUF 文件

当用户询问存在哪些 GGUF 时，返回：

- 文件名
- 文件大小
- 量化标签
- 是否为主模型或辅助投影文件

除非被要求，否则忽略：

- README
- BF16 分片文件
- imatrix blob 或校准产物

此步骤使用 tree API：

- `https://huggingface.co/api/models/<repo>/tree/main?recursive=true`

对于 `unsloth/Qwen3.6-35B-A3B-GGUF` 这样的仓库，local-app 页面可显示 `UD-Q4_K_M`、`UD-Q5_K_M`、`UD-Q6_K` 和 `Q8_0` 等量化标签，而 tree API 则暴露精确文件路径（如 `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` 和 `Qwen3.6-35B-A3B-Q8_0.gguf`）及字节大小。使用 tree API 将量化标签转换为精确文件名。

## 搜索模式

直接使用以下 URL 格式：

```text
https://huggingface.co/models?apps=llama.cpp&sort=trending
https://huggingface.co/models?search=<term>&apps=llama.cpp&sort=trending
https://huggingface.co/models?search=<term>&apps=llama.cpp&num_parameters=min:0,max:24B&sort=trending
https://huggingface.co/<repo>?local-app=llama.cpp
https://huggingface.co/api/models/<repo>/tree/main?recursive=true
https://huggingface.co/<repo>/tree/main
```

## 输出格式

回答发现请求时，优先使用如下紧凑结构化结果：

```text
Repo: <repo>
Recommended quant from HF: <label> (<size>)
llama-server: <command>
Other GGUFs:
- <filename> - <size>
- <filename> - <size>
Source URLs:
- <local-app URL>
- <tree API URL>
```

## 参考资料

- **[hub-discovery.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/inference/llama-cpp/references/hub-discovery.md)** — 纯 URL Hugging Face 工作流、搜索模式、GGUF 提取及命令重建
- **[advanced-usage.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/inference/llama-cpp/references/advanced-usage.md)** — 推测解码、批量推理、语法约束生成、LoRA、多 GPU、自定义构建、基准脚本
- **[quantization.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/inference/llama-cpp/references/quantization.md)** — 量化质量权衡、何时使用 Q4/Q5/Q6/IQ、模型大小缩放、imatrix
- **[server.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/inference/llama-cpp/references/server.md)** — 直接从 Hub 启动服务器、OpenAI API 端点、Docker 部署、NGINX 负载均衡、监控
- **[optimization.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/inference/llama-cpp/references/optimization.md)** — CPU 线程、BLAS、GPU 卸载启发式、批处理调优、基准测试
- **[troubleshooting.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/inference/llama-cpp/references/troubleshooting.md)** — 安装/转换/量化/推理/服务器问题、Apple Silicon、调试

## 资源

- **GitHub**：https://github.com/ggml-org/llama.cpp
- **Hugging Face GGUF + llama.cpp 文档**：https://huggingface.co/docs/hub/gguf-llamacpp
- **Hugging Face 本地应用文档**：https://huggingface.co/docs/hub/main/local-apps
- **Hugging Face 本地 Agent 文档**：https://huggingface.co/docs/hub/agents-local
- **local-app 页面示例**：https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF?local-app=llama.cpp
- **tree API 示例**：https://huggingface.co/api/models/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main?recursive=true
- **llama.cpp 搜索示例**：https://huggingface.co/models?num_parameters=min:0,max:24B&apps=llama.cpp&sort=trending
- **许可证**：MIT
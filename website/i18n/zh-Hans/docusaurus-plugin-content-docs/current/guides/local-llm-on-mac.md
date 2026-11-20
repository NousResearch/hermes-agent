---
sidebar_position: 2
title: "在 Mac 上运行本地 LLM"
description: "使用 llama.cpp 或 MLX 在 macOS 上搭建兼容 OpenAI 的本地 LLM 服务器，涵盖模型选择、内存优化以及 Apple Silicon 上的实测基准数据"
---

# 在 Mac 上运行本地 LLM

本指南介绍如何在 macOS 上运行一个兼容 OpenAI API 的本地 LLM 服务器。你将获得完整的隐私保护、零 API 费用，以及 Apple Silicon 上出乎意料的出色性能。

我们涵盖两个后端：

| 后端 | 安装方式 | 优势 | 格式 |
|---------|---------|---------|--------|
| **llama.cpp** | `brew install llama.cpp` | 首 token 延迟最低，量化 KV 缓存节省内存 | GGUF |
| **omlx** | [omlx.ai](https://omlx.ai) | token 生成速度最快，原生 Metal 优化 | MLX (safetensors) |

两者均暴露兼容 OpenAI 的 `/v1/chat/completions` 端点。Hermes 支持任意一个——只需将其指向 `http://localhost:8080` 或 `http://localhost:8000`。

:::info 仅限 Apple Silicon
本指南面向搭载 Apple Silicon（M1 及更新）的 Mac。Intel Mac 可使用 llama.cpp，但无 GPU 加速——性能会明显更慢。
:::

---

## 选择模型

入门推荐 **Qwen3.5-9B**——这是一个强推理模型，量化后可在 8GB+ 统一内存上轻松运行。

| 变体 | 磁盘占用 | 所需内存（128K 上下文） | 后端 |
|---------|-------------|---------------------------|---------|
| Qwen3.5-9B-Q4_K_M (GGUF) | 5.3 GB | ~10 GB（含量化 KV 缓存） | llama.cpp |
| Qwen3.5-9B-mlx-lm-mxfp4 (MLX) | ~5 GB | ~12 GB | omlx |

**内存估算规则：** 模型大小 + KV 缓存。9B Q4 模型约 5 GB。128K 上下文下 Q4 量化的 KV 缓存额外占用约 4–5 GB。若使用默认（f16）KV 缓存，则会膨胀至约 16 GB。llama.cpp 中的量化 KV 缓存参数是内存受限系统的关键技巧。

对于更大的模型（27B、35B），你需要 32 GB+ 的统一内存。9B 是 8–16 GB 机器的最佳选择。

---

## 方案 A：llama.cpp

llama.cpp 是移植性最强的本地 LLM 运行时。在 macOS 上，它开箱即用地通过 Metal 进行 GPU 加速。

### 安装

```bash
brew install llama.cpp
```

安装后即可全局使用 `llama-server` 命令。

### 下载模型

你需要 GGUF 格式的模型。最简便的来源是通过 `huggingface-cli` 从 Hugging Face 下载：

```bash
brew install huggingface-cli
```

然后下载：

```bash
huggingface-cli download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir ~/models
```

:::tip 受限模型
Hugging Face 上的部分模型需要身份验证。如果遇到 401 或 404 错误，请先运行 `huggingface-cli login`。
:::

### 启动服务器

```bash
llama-server -m ~/models/Qwen3.5-9B-Q4_K_M.gguf \
  -ngl 99 \
  -c 131072 \
  -np 1 \
  -fa on \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --host 0.0.0.0
```

各参数说明：

| 参数 | 用途 |
|------|---------|
| `-ngl 99` | 将所有层卸载到 GPU（Metal）。设置较大的数值以确保没有层留在 CPU 上。 |
| `-c 131072` | 上下文窗口大小（128K token）。内存不足时可减小此值。 |
| `-np 1` | 并行槽数量。单用户使用时保持为 1——更多槽会分摊内存预算。 |
| `-fa on` | Flash attention。减少内存占用并加速长上下文推理。 |
| `--cache-type-k q4_0` | 将 key 缓存量化为 4-bit。**这是最大的内存节省手段。** |
| `--cache-type-v q4_0` | 将 value 缓存量化为 4-bit。与上一项合用，相比 f16 可将 KV 缓存内存减少约 75%。 |
| `--host 0.0.0.0` | 监听所有网络接口。若不需要网络访问，可改为 `127.0.0.1`。 |

当你看到以下输出时，服务器已就绪：

```
main: server is listening on http://0.0.0.0:8080
srv  update_slots: all slots are idle
```

### 内存受限系统的优化

`--cache-type-k q4_0 --cache-type-v q4_0` 参数是内存有限系统最重要的优化手段。以下是 128K 上下文下的影响对比：

| KV 缓存类型 | KV 缓存内存（128K 上下文，9B 模型） |
|---------------|--------------------------------------|
| f16（默认） | ~16 GB |
| q8_0 | ~8 GB |
| **q4_0** | **~4 GB** |

在 8 GB Mac 上，使用 `q4_0` KV 缓存并将上下文缩减为 `-c 32768`（32K）。在 16 GB 上，可以轻松使用 128K 上下文。在 32 GB+ 上，可以运行更大的模型或多个并行槽。

如果仍然内存不足，优先减小上下文大小（`-c`），然后尝试更小的量化级别（Q3_K_M 代替 Q4_K_M）。

### 测试

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }' | jq .choices[0].message.content
```

### 获取模型名称

如果忘记了模型名称，可查询 models 端点：

```bash
curl -s http://localhost:8080/v1/models | jq '.data[].id'
```

---

## 方案 B：通过 omlx 使用 MLX

[omlx](https://omlx.ai) 是一款 macOS 原生应用，用于管理和提供 MLX 模型服务。MLX 是 Apple 自研的机器学习框架，专为 Apple Silicon 统一内存架构优化。

### 安装

从 [omlx.ai](https://omlx.ai) 下载并安装。它提供图形界面用于模型管理，并内置服务器。

### 下载模型

使用 omlx 应用浏览并下载模型。搜索 `Qwen3.5-9B-mlx-lm-mxfp4` 并下载。模型存储在本地（通常位于 `~/.omlx/models/`）。

### 启动服务器

omlx 默认在 `http://127.0.0.1:8000` 上提供服务。通过应用 UI 启动服务，或在可用时使用 CLI。

### 测试

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-mlx-lm-mxfp4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }' | jq .choices[0].message.content
```

### 列出可用模型

omlx 可同时提供多个模型的服务：

```bash
curl -s http://127.0.0.1:8000/v1/models | jq '.data[].id'
```

---

## 基准测试：llama.cpp vs MLX

两个后端在同一台机器（Apple M5 Max，128 GB 统一内存）上测试，使用相同模型（Qwen3.5-9B），量化级别相当（GGUF 使用 Q4_K_M，MLX 使用 mxfp4）。五个不同 prompt，每个运行三次，后端顺序测试以避免资源竞争。

### 结果

| 指标 | llama.cpp (Q4_K_M) | MLX (mxfp4) | 胜者 |
|--------|-------------------|-------------|--------|
| **TTFT（首 token 延迟，均值）** | **67 ms** | 289 ms | llama.cpp（快 4.3 倍） |
| **TTFT（p50）** | **66 ms** | 286 ms | llama.cpp（快 4.3 倍） |
| **生成速度（均值）** | 70 tok/s | **96 tok/s** | MLX（快 37%） |
| **生成速度（p50）** | 70 tok/s | **96 tok/s** | MLX（快 37%） |
| **总耗时（512 token）** | 7.3s | **5.5s** | MLX（快 25%） |

### 含义解读

- **llama.cpp** 在 prompt 处理上表现突出——其 flash attention + 量化 KV 缓存流水线可在约 66ms 内返回第一个 token。如果你在构建对响应速度敏感的交互式应用（聊天机器人、自动补全），这是显著优势。

- **MLX** 一旦开始生成，token 速度快约 37%。对于批量任务、长文本生成，或任何更关注总完成时间而非初始延迟的场景，MLX 完成得更快。

- 两个后端都**极为稳定**——多次运行间的方差可忽略不计。这些数据可作为可靠参考。

### 如何选择？

| 使用场景 | 推荐 |
|----------|---------------|
| 交互式聊天、低延迟工具 | llama.cpp |
| 长文本生成、批量处理 | MLX (omlx) |
| 内存受限（8–16 GB） | llama.cpp（量化 KV 缓存无可匹敌） |
| 同时提供多个模型服务 | omlx（内置多模型支持） |
| 最大兼容性（含 Linux） | llama.cpp |

---

## 连接 Hermes

本地服务器启动后：

```bash
hermes model
```

选择 **Custom endpoint**，按提示操作。系统会询问 base URL 和模型名称——使用你所配置的后端对应的值即可。

---

## 超时设置

Hermes 会自动检测本地端点（localhost、局域网 IP）并放宽其流式传输超时限制。大多数情况下无需额外配置。

如果仍然遇到超时错误（例如在慢速硬件上使用超大上下文），可以覆盖流式读取超时：

```bash
# 在 .env 中——将默认的 120s 提高到 30 分钟
HERMES_STREAM_READ_TIMEOUT=1800
```

| 超时类型 | 默认值 | 本地自动调整 | 环境变量覆盖 |
|---------|---------|----------------------|------------------|
| 流式读取（socket 级别） | 120s | 提升至 1800s | `HERMES_STREAM_READ_TIMEOUT` |
| 停滞流检测 | 180s | 完全禁用 | `HERMES_STREAM_STALE_TIMEOUT` |
| API 调用（非流式） | 1800s | 无需调整 | `HERMES_API_TIMEOUT` |

流式读取超时最容易引发问题——它是接收下一个数据块的 socket 级别截止时间。在大上下文的预填充（prefill）阶段，本地模型可能在处理 prompt 时数分钟内没有任何输出。自动检测机制会透明地处理这一情况。
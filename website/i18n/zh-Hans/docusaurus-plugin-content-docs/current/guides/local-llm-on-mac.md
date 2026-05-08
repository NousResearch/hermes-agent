---
sidebar_position: 2
title: "在 Mac 上运行本地 LLM"
description: "在 macOS 上使用 llama.cpp 或 MLX 设置本地 OpenAI 兼容的 LLM 服务器，包括模型选择、内存优化和 Apple Silicon 上的真实基准测试"
---

# 在 Mac 上运行本地 LLM

本指南将引导你在 macOS 上使用 OpenAI 兼容的 API 运行本地 LLM 服务器。你可以获得完全隐私、零 API 成本，以及在 Apple Silicon 上令人惊讶的良好性能。

我们涵盖两个后端：

| 后端 | 安装 | 最擅长 | 格式 |
|---------|---------|---------|--------|
| **llama.cpp** | `brew install llama.cpp` | 最快的首令牌时间，量化 KV 缓存以低内存使用 | GGUF |
| **omlx** | [omlx.ai](https://omlx.ai) | 最快的令牌生成，原生 Metal 优化 | MLX (safetensors) |

两者都公开 OpenAI 兼容的 `/v1/chat/completions` 端点。Hermes 可以与任一后端工作——只需将其指向 `http://localhost:8080` 或 `http://localhost:8000`。

:::info 仅限 Apple Silicon
本指南针对配备 Apple Silicon 的 Mac（M1 及更高版本）。Intel Mac 可以使用 llama.cpp 但没有 GPU 加速——预计性能明显较慢。
:::

---

## 选择模型

对于入门，我们推荐 **Qwen3.5-9B**——这是一个强大的推理模型，在量化后可以舒适地放入 8GB+ 的统一内存中。

| 变体 | 磁盘大小 | 需要的内存（128K 上下文） | 后端 |
|---------|-------------|---------------------------|---------|
| Qwen3.5-9B-Q4_K_M (GGUF) | 5.3 GB | 量化 KV 缓存约 10 GB | llama.cpp |
| Qwen3.5-9B-mlx-lm-mxfp4 (MLX) | ~5 GB | ~12 GB | omlx |

**内存经验法则：** 模型大小 + KV 缓存。9B Q4 模型约 5 GB。128K 上下文的 KV 缓存在 Q4 量化下增加约 4-5 GB。使用默认（f16）KV 缓存，会膨胀到约 16 GB。llama.cpp 中的量化 KV 缓存标志是内存受限系统的关键技巧。

对于更大的模型（27B、35B），你需要 32 GB+ 的统一内存。9B 是 8-16 GB 机器的甜点。

---

## 选项 A：llama.cpp

llama.cpp 是最可移植的本地 LLM 运行时。在 macOS 上，它开箱即用使用 Metal 进行 GPU 加速。

### 安装

```bash
brew install llama.cpp
```

这让你可以在全局使用 `llama-server` 命令。

### 下载模型

你需要 GGUF 格式的模型。最简单的来源是通过 `huggingface-cli` 的 Hugging Face：

```bash
brew install huggingface-cli
```

然后下载：

```bash
huggingface-cli download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir ~/models
```

:::tip 受限制的模型
Hugging Face 上的一些模型需要身份验证。如果你收到 401 或 404 错误，请先运行 `huggingface-cli login`。
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

每个标志的作用：

| 标志 | 目的 |
|------|---------|
| `-ngl 99` | 将所有层卸载到 GPU（Metal）。使用高数字以确保没有东西留在 CPU 上。 |
| `-c 131072` | 上下文窗口大小（128K 令牌）。如果你的内存不足，请减少此值。 |
| `-np 1` | 并行槽位数。单用户使用保持为 1——更多槽位会分割你的内存预算。 |
| `-fa on` | Flash attention。减少内存使用并加速长上下文推理。 |
| `--cache-type-k q4_0` | 将键缓存量化为 4 位。**这是巨大的内存节省器。** |
| `--cache-type-v q4_0` | 将值缓存量化为 4 位。与上述一起，这比 f16 减少约 75% 的 KV 缓存内存。 |
| `--host 0.0.0.0` | 监听所有接口。如果你不需要网络访问，请使用 `127.0.0.1`。 |

当你看到以下内容时，服务器已准备好：

```
main: server is listening on http://0.0.0.0:8080
srv  update_slots: all slots are idle
```

### 受限系统的内存优化

`--cache-type-k q4_0 --cache-type-v q4_0` 标志对于内存有限的系统是最重要的优化。以下是 128K 上下文的影响：

| KV 缓存类型 | KV 缓存内存（128K ctx，9B 模型） |
|---------------|--------------------------------------|
| f16（默认） | ~16 GB |
| q8_0 | ~8 GB |
| **q4_0** | **~4 GB** |

在 8 GB Mac 上，使用 `q4_0` KV 缓存并将上下文减少到 `-c 32768`（32K）。在 16 GB 上，你可以舒适地使用 128K 上下文。在 32 GB+ 上，你可以运行更大的模型或多个并行槽位。

如果你仍然内存不足，首先减少上下文大小（`-c`），然后尝试更小的量化（Q3_K_M 而不是 Q4_K_M）。

### 测试它

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

如果你忘记模型名称，查询模型端点：

```bash
curl -s http://localhost:8080/v1/models | jq '.data[].id'
```

---

## 选项 B：通过 omlx 使用 MLX

[omlx](https://omlx.ai) 是一个管理并提供 MLX 模型的 macOS 原生应用。MLX 是 Apple 自己的机器学习框架，专门针对 Apple Silicon 的统一内存架构优化。

### 安装

从 [omlx.ai](https://omlx.ai) 下载并安装。它提供模型管理的 GUI 和内置服务器。

### 下载模型

使用 omlx 应用浏览并下载模型。搜索 `Qwen3.5-9B-mlx-lm-mxfp4` 并下载它。模型存储在本地（通常在 `~/.omlx/models/`）。

### 启动服务器

omlx 默认在 `http://127.0.0.1:8000` 提供模型服务。从应用 UI 启动服务，或者如果可用，使用 CLI。

### 测试它

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

omlx 可以同时提供多个模型：

```bash
curl -s http://127.0.0.1:8000/v1/models | jq '.data[].id'
```

---

## 基准测试：llama.cpp vs MLX

两个后端在同一台机器（Apple M5 Max，128 GB 统一内存）上测试，运行相同的模型（Qwen3.5-9B），在可比较的量化级别（GGUF 为 Q4_K_M，MLX 为 mxfp4）。五个不同的提示，每个运行三次，后端顺序测试以避免资源争用。

### 结果

| 指标 | llama.cpp (Q4_K_M) | MLX (mxfp4) | 获胜者 |
|--------|-------------------|-------------|--------|
| **TTFT（平均）** | **67 ms** | 289 ms | llama.cpp（4.3 倍更快） |
| **TTFT（p50）** | **66 ms** | 286 ms | llama.cpp（4.3 倍更快） |
| **生成（平均）** | 70 tok/s | **96 tok/s** | MLX（37% 更快） |
| **生成（p50）** | 70 tok/s | **96 tok/s** | MLX（37% 更快） |
| **总时间（512 令牌）** | 7.3s | **5.5s** | MLX（25% 更快） |

### 这意味着什么

- **llama.cpp** 在提示处理方面表现出色——其 flash attention + 量化 KV 缓存管道让你在约 66ms 内获得第一个令牌。如果你正在构建感知响应性很重要的交互式应用（聊天机器人、自动完成），这是一个有意义的优势。
- **MLX** 一旦启动，生成令牌的速度约快 37%。对于批处理工作负载、长格式生成或总完成时间比初始延迟更重要的任何任务，MLX 完成得更早。
- 两个后端都**极其一致**——跨运行的方差可以忽略不计。你可以依赖这些数字。

### 你应该选择哪个？

| 用例 | 推荐 |
|----------|---------------|
| 交互式聊天、低延迟工具 | llama.cpp |
| 长格式生成、批量处理 | MLX (omlx) |
| 内存受限（8-16 GB） | llama.cpp（量化 KV 缓存无与伦比） |
| 同时提供多个模型 | omlx（内置多模型支持） |
| 最大兼容性（也适用于 Linux） | llama.cpp |

---

## 连接到 Hermes

一旦你的本地服务器正在运行：

```bash
hermes model
```

选择**自定义端点**并按照提示操作。它会询问基础 URL 和模型名称——使用你上面设置的任一后端的值。

---

## 超时

Hermes 自动检测本地端点（localhost、LAN IP）并放宽其流式传输超时。大多数设置不需要配置。

如果你仍然遇到超时错误（例如，在慢速硬件上非常大的上下文），你可以覆盖流式读取超时：

```bash
# 在你的 .env 中——从 120 秒默认值提高到 30 分钟
HERMES_STREAM_READ_TIMEOUT=1800
```

| 超时 | 默认值 | 本地自动调整 | 环境变量覆盖 |
|---------|---------|----------------------|------------------|
| 流读取（套接字级别） | 120s | 提高到 1800s | `HERMES_STREAM_READ_TIMEOUT` |
| 陈旧流检测 | 180s | 完全禁用 | `HERMES_STREAM_STALE_TIMEOUT` |
| API 调用（非流式） | 1800s | 不需要更改 | `HERMES_API_TIMEOUT` |

流读取超时是最可能引起问题的——它是接收下一个数据块的套接字级别截止时间。在大型上下文的预填充期间，本地模型在处理提示时可能在几分钟内不产生输出。自动检测透明地处理这一点。

---
title: "优化注意力 Flash"
sidebar_label: "优化注意力 Flash"
description: "通过 Flash Attention 优化 Transformer 注意力机制，实现 2-4 倍加速和 10-20 倍内存减少"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 优化注意力 Flash

通过 Flash Attention 优化 Transformer 注意力机制，实现 2-4 倍加速和 10-20 倍内存减少。适用于以下场景：使用长序列（>512 token）训练/运行 Transformer、遇到注意力相关的 GPU 内存问题，或需要更快的推理速度。支持 PyTorch 原生 SDPA、flash-attn 库、H100 FP8 以及滑动窗口注意力。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/mlops/flash-attention` 安装 |
| 路径 | `optional-skills/mlops/flash-attention` |
| 版本 | `1.0.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖项 | `flash-attn`, `torch`, `transformers` |
| 平台 | linux, macos |
| 标签 | `Optimization`, `Flash Attention`, `Attention Optimization`, `Memory Efficiency`, `Speed Optimization`, `Long Context`, `PyTorch`, `SDPA`, `H100`, `FP8`, `Transformers` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发该 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Flash Attention - 快速内存高效注意力

## 快速开始

Flash Attention 通过 IO 感知分块（IO-aware tiling）和重计算（recomputation）技术，为 Transformer 注意力提供 2-4 倍加速和 10-20 倍内存减少。

**PyTorch 原生方式（最简单，PyTorch 2.2+）**：
```python
import torch
import torch.nn.functional as F

q = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float16)  # [batch, heads, seq, dim]
k = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float16)

# 如果可用，自动使用 Flash Attention
out = F.scaled_dot_product_attention(q, k, v)
```

**flash-attn 库（功能更多）**：
```bash
pip install flash-attn --no-build-isolation
```

```python
from flash_attn import flash_attn_func

# q, k, v: [batch, seqlen, nheads, headdim]
out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
```

## 常见工作流

### 工作流 1：在现有 PyTorch 模型中启用

复制此检查清单：

```
Flash Attention 集成：
- [ ] 步骤 1：检查 PyTorch 版本（≥2.2）
- [ ] 步骤 2：启用 Flash Attention 后端
- [ ] 步骤 3：通过性能分析验证加速效果
- [ ] 步骤 4：测试精度与基线一致
```

**步骤 1：检查 PyTorch 版本**

```bash
python -c "import torch; print(torch.__version__)"
# 应为 ≥2.2.0
```

如果 &lt;2.2，请升级：
```bash
pip install --upgrade torch
```

**步骤 2：启用 Flash Attention 后端**

替换标准注意力：
```python
# 之前（标准注意力）
attn_weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(d_k), dim=-1)
out = attn_weights @ v

# 之后（Flash Attention）
import torch.nn.functional as F
out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
```

强制使用 Flash Attention 后端：
```python
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=False
):
    out = F.scaled_dot_product_attention(q, k, v)
```

**步骤 3：通过性能分析验证加速效果**

```python
import torch.utils.benchmark as benchmark

def test_attention(use_flash):
    q, k, v = [torch.randn(2, 8, 2048, 64, device='cuda', dtype=torch.float16) for _ in range(3)]

    if use_flash:
        with torch.backends.cuda.sdp_kernel(enable_flash=True):
            return F.scaled_dot_product_attention(q, k, v)
    else:
        attn = (q @ k.transpose(-2, -1) / 8.0).softmax(dim=-1)
        return attn @ v

# 基准测试
t_flash = benchmark.Timer(stmt='test_attention(True)', globals=globals())
t_standard = benchmark.Timer(stmt='test_attention(False)', globals=globals())

print(f"Flash: {t_flash.timeit(100).mean:.3f}s")
print(f"Standard: {t_standard.timeit(100).mean:.3f}s")
```

预期效果：序列长度 >512 token 时有 2-4 倍加速。

**步骤 4：测试精度与基线一致**

```python
# 比较输出
q, k, v = [torch.randn(1, 8, 512, 64, device='cuda', dtype=torch.float16) for _ in range(3)]

# Flash Attention
out_flash = F.scaled_dot_product_attention(q, k, v)

# 标准注意力
attn_weights = torch.softmax(q @ k.transpose(-2, -1) / 8.0, dim=-1)
out_standard = attn_weights @ v

# 检查差异
diff = (out_flash - out_standard).abs().max()
print(f"Max difference: {diff:.6f}")
# float16 下应 <1e-3
```

### 工作流 2：使用 flash-attn 库实现高级功能

适用于多查询注意力（multi-query attention）、滑动窗口或 H100 FP8。

复制此检查清单：

```
flash-attn 库安装：
- [ ] 步骤 1：安装 flash-attn 库
- [ ] 步骤 2：修改注意力代码
- [ ] 步骤 3：启用高级功能
- [ ] 步骤 4：基准测试性能
```

**步骤 1：安装 flash-attn 库**

```bash
# NVIDIA GPU（CUDA 12.0+）
pip install flash-attn --no-build-isolation

# 验证安装
python -c "from flash_attn import flash_attn_func; print('Success')"
```

**步骤 2：修改注意力代码**

```python
from flash_attn import flash_attn_func

# 输入：[batch_size, seq_len, num_heads, head_dim]
# 如需要，从 [batch, heads, seq, dim] 转置
q = q.transpose(1, 2)  # [batch, seq, heads, dim]
k = k.transpose(1, 2)
v = v.transpose(1, 2)

out = flash_attn_func(
    q, k, v,
    dropout_p=0.1,
    causal=True,  # 用于自回归模型
    window_size=(-1, -1),  # 无滑动窗口
    softmax_scale=None  # 自动缩放
)

out = out.transpose(1, 2)  # 转回 [batch, heads, seq, dim]
```

**步骤 3：启用高级功能**

多查询注意力（跨 head 共享 K/V）：
```python
from flash_attn import flash_attn_func

# q: [batch, seq, num_q_heads, dim]
# k, v: [batch, seq, num_kv_heads, dim]  # 更少的 KV head
out = flash_attn_func(q, k, v)  # 自动处理 MQA
```

滑动窗口注意力（局部注意力）：
```python
# 仅关注前后 256 个 token 的窗口
out = flash_attn_func(
    q, k, v,
    window_size=(256, 256),  # (左, 右) 窗口
    causal=True
)
```

**步骤 4：基准测试性能**

```python
import torch
from flash_attn import flash_attn_func
import time

q, k, v = [torch.randn(4, 4096, 32, 64, device='cuda', dtype=torch.float16) for _ in range(3)]

# 预热
for _ in range(10):
    _ = flash_attn_func(q, k, v)

# 基准测试
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    out = flash_attn_func(q, k, v)
    torch.cuda.synchronize()
end = time.time()

print(f"Time per iteration: {(end-start)/100*1000:.2f}ms")
print(f"Memory allocated: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
```

### 工作流 3：H100 FP8 优化（FlashAttention-3）

在 H100 GPU 上获得最大性能。

```
FP8 设置：
- [ ] 步骤 1：确认 H100 GPU 可用
- [ ] 步骤 2：安装支持 FP8 的 flash-attn
- [ ] 步骤 3：将输入转换为 FP8
- [ ] 步骤 4：使用 FP8 注意力运行
```

**步骤 1：确认 H100 GPU**

```bash
nvidia-smi --query-gpu=name --format=csv
# 应显示 "H100" 或 "H800"
```

**步骤 2：安装支持 FP8 的 flash-attn**

```bash
pip install flash-attn --no-build-isolation
# H100 的 FP8 支持已包含在内
```

**步骤 3：将输入转换为 FP8**

```python
import torch

q = torch.randn(2, 4096, 32, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 4096, 32, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 4096, 32, 64, device='cuda', dtype=torch.float16)

# 转换为 float8_e4m3（FP8）
q_fp8 = q.to(torch.float8_e4m3fn)
k_fp8 = k.to(torch.float8_e4m3fn)
v_fp8 = v.to(torch.float8_e4m3fn)
```

**步骤 4：使用 FP8 注意力运行**

```python
from flash_attn import flash_attn_func

# FlashAttention-3 在 H100 上自动使用 FP8 内核
out = flash_attn_func(q_fp8, k_fp8, v_fp8)
# 结果：约 1.2 PFLOPS，比 FP16 快 1.5-2 倍
```

## 何时使用与替代方案

**使用 Flash Attention 的场景：**
- 使用 >512 token 的序列训练 Transformer
- 使用长上下文（>2K token）进行推理
- GPU 内存受限（标准注意力 OOM）
- 需要 2-4 倍加速且不损失精度
- 使用 PyTorch 2.2+ 或可安装 flash-attn

**改用替代方案的场景：**
- **标准注意力**：序列 &lt;256 token（开销不值得）
- **xFormers**：需要更多注意力变体（不仅仅是速度）
- **内存高效注意力**：CPU 推理（Flash Attention 需要 GPU）

## 常见问题

**问题：ImportError: cannot import flash_attn**

使用 no-build-isolation 标志安装：
```bash
pip install flash-attn --no-build-isolation
```

或先安装 CUDA toolkit：
```bash
conda install cuda -c nvidia
pip install flash-attn --no-build-isolation
```

**问题：速度低于预期（无加速效果）**

Flash Attention 的收益随序列长度增加而提升：
- &lt;512 token：加速极小（10-20%）
- 512-2K token：2-3 倍加速
- >2K token：3-4 倍加速

请确认序列长度是否足够。

**问题：RuntimeError: CUDA error**

验证 GPU 是否支持 Flash Attention：
```python
import torch
print(torch.cuda.get_device_capability())
# 应为 ≥(7, 5)，即 Turing 及以上
```

Flash Attention 要求：
- Ampere（A100、A10）：✅ 完全支持
- Turing（T4）：✅ 支持
- Volta（V100）：❌ 不支持

**问题：精度下降**

检查 dtype 是否为 float16 或 bfloat16（而非 float32）：
```python
q = q.to(torch.float16)  # 或 torch.bfloat16
```

Flash Attention 使用 float16/bfloat16 以提升速度，不支持 float32。

## 高级主题

**与 HuggingFace Transformers 集成**：参见 [references/transformers-integration.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/flash-attention/references/transformers-integration.md)，了解如何在 BERT、GPT、Llama 模型中启用 Flash Attention。

**性能基准测试**：参见 [references/benchmarks.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/flash-attention/references/benchmarks.md)，查看跨 GPU 和序列长度的详细速度与内存对比。

## 硬件要求

- **GPU**：NVIDIA Ampere 及以上（A100、A10、A30）或 AMD MI200 及以上
- **显存**：与标准注意力相同（Flash Attention 不增加内存占用）
- **CUDA**：12.0+（最低 11.8）
- **PyTorch**：2.2+ 以获得原生支持

**不支持**：V100（Volta）、CPU 推理

## 资源

- 论文："FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"（NeurIPS 2022）
- 论文："FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"（ICLR 2024）
- 博客：https://tridao.me/blog/2024/flash3/
- GitHub：https://github.com/Dao-AILab/flash-attention
- PyTorch 文档：https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
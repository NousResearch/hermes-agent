---
title: "Heartmula — HeartMuLa：类似 Suno 的歌词+标签歌曲生成"
sidebar_label: "Heartmula"
description: "HeartMuLa：类似 Suno 的歌词+标签歌曲生成"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Heartmula

HeartMuLa：类似 Suno 的歌词+标签歌曲生成。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/media/heartmula` |
| 版本 | `1.0.0` |
| 标签 | `音乐`、`音频`、`生成`、`ai`、`heartmula`、`heartcodec`、`歌词`、`歌曲` |
| 相关技能 | `audiocraft` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此技能时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# HeartMuLa - 开源音乐生成

## 概述
HeartMuLa 是一系列开源音乐基础模型（Apache-2.0），基于歌词和标签生成音乐，支持多语言。从歌词+标签生成完整歌曲。相当于开源版的 Suno。包括：
- **HeartMuLa** — 音乐语言模型（3B/7B），用于从歌词+标签生成
- **HeartCodec** — 12.5Hz 音乐编解码器，用于高保真音频重建
- **HeartTranscriptor** — 基于 Whisper 的歌词转录
- **HeartCLAP** — 音频-文本对齐模型

## 使用场景
- 用户想要从文字描述生成音乐/歌曲
- 用户想要 Suno 的开源替代方案
- 用户想要本地/离线音乐生成
- 用户询问 HeartMuLa、heartlib 或 AI 音乐生成

## 硬件要求
- **最低配置**：8GB 显存，使用 `--lazy_load true`（按顺序加载/卸载模型）
- **推荐配置**：16GB+ 显存，适合单 GPU 舒适使用
- **多 GPU**：使用 `--mula_device cuda:0 --codec_device cuda:1` 分配到不同 GPU
- 3B 模型使用 lazy_load 时峰值约 6.2GB 显存

## 安装步骤

### 1. 克隆仓库
```bash
cd ~/  # 或目标目录
git clone https://github.com/HeartMuLa/heartlib.git
cd heartlib
```

### 2. 创建虚拟环境（需要 Python 3.10）
```bash
uv venv --python 3.10 .venv
. .venv/bin/activate
uv pip install -e .
```

### 3. 修复依赖兼容性问题

**重要**：截至 2026 年 2 月，固定的依赖与新版本包有冲突。应用以下修复：

```bash
# 升级 datasets（旧版本与当前 pyarrow 不兼容）
uv pip install --upgrade datasets

# 升级 transformers（huggingface-hub 1.x 兼容性需要）
uv pip install --upgrade transformers
```

### 4. 修补源代码（transformers 5.x 必需）

**补丁 1 - RoPE 缓存修复**，在 `src/heartlib/heartmula/modeling_heartmula.py` 中：

在 `HeartMuLa` 类的 `setup_caches` 方法中，在 `reset_caches` try/except 块之后、`with device:` 块之前添加 RoPE 重新初始化：

```python
# 重新初始化在 meta-device 加载期间跳过的 RoPE 缓存
from torchtune.models.llama3_1._position_embeddings import Llama3ScaledRoPE
for module in self.modules():
    if isinstance(module, Llama3ScaledRoPE) and not module.is_cache_built:
        module.rope_init()
        module.to(device)
```

**原因**：`from_pretrained` 首先在 meta device 上创建模型；`Llama3ScaledRoPE.rope_init()` 在 meta tensor 上跳过缓存构建，权重加载到真实设备后也不再重建。

**补丁 2 - HeartCodec 加载修复**，在 `src/heartlib/pipelines/music_generation.py` 中：

在所有 `HeartCodec.from_pretrained()` 调用中添加 `ignore_mismatched_sizes=True`（有 2 处：`__init__` 中的即时加载和 `codec` 属性中的延迟加载）。

**原因**：VQ 码本 `initted` 缓冲区在检查点中形状为 `[1]`，在模型中为 `[]`。数据相同，只是标量 vs 0 维张量。可以安全忽略。

### 5. 下载模型检查点
```bash
cd heartlib  # 项目根目录
hf download --local-dir './ckpt' 'HeartMuLa/HeartMuLaGen'
hf download --local-dir './ckpt/HeartMuLa-oss-3B' 'HeartMuLa/HeartMuLa-oss-3B-happy-new-year'
hf download --local-dir './ckpt/HeartCodec-oss' 'HeartMuLa/HeartCodec-oss-20260123'
```

所有 3 个可以并行下载。总大小约几 GB。

## GPU / CUDA

HeartMuLa 默认使用 CUDA（`--mula_device cuda --codec_device cuda`）。如果用户的 NVIDIA GPU 已安装 PyTorch CUDA 支持，无需额外设置。

- 安装的 `torch==2.4.1` 自带 CUDA 12.1 支持
- `torchtune` 可能报告版本 `0.4.0+cpu` — 这只是包元数据，它仍通过 PyTorch 使用 CUDA
- 验证 GPU 是否被使用，查看输出中的"CUDA memory"行（如"CUDA memory before unloading: 6.20 GB"）
- **没有 GPU？** 可以使用 `--mula_device cpu --codec_device cpu` 在 CPU 上运行，但生成速度会**极慢**（单首歌曲可能 30-60+ 分钟 vs GPU 上的约 4 分钟）。CPU 模式还需要大量内存（约 12GB+ 空闲）。如果用户没有 NVIDIA GPU，推荐使用云 GPU 服务（Google Colab 免费 T4、Lambda Labs 等）或在线演示 https://heartmula.github.io/。

## 使用

### 基本生成
```bash
cd heartlib
. .venv/bin/activate
python ./examples/run_music_generation.py \
  --model_path=./ckpt \
  --version="3B" \
  --lyrics="./assets/lyrics.txt" \
  --tags="./assets/tags.txt" \
  --save_path="./assets/output.mp3" \
  --lazy_load true
```

### 输入格式

**标签**（逗号分隔，无空格）：
```
piano,happy,wedding,synthesizer,romantic
```
或
```
rock,energetic,guitar,drums,male-vocal
```

**歌词**（使用带括号的结构标签）：
```
[Intro]

[Verse]
Your lyrics here...

[Chorus]
Chorus lyrics...

[Bridge]
Bridge lyrics...

[Outro]
```

### 关键参数
| 参数 | 默认值 | 描述 |
|---|---|---|
| `--max_audio_length_ms` | 240000 | 最大时长（毫秒）（240秒 = 4分钟） |
| `--topk` | 50 | Top-k 采样 |
| `--temperature` | 1.0 | 采样温度 |
| `--cfg_scale` | 1.5 | 无分类器引导比例 |
| `--lazy_load` | false | 按需加载/卸载模型（节省显存） |
| `--mula_dtype` | bfloat16 | HeartMuLa 的数据类型（推荐 bf16） |
| `--codec_dtype` | float32 | HeartCodec 的数据类型（推荐 fp32 以保证质量） |

### 性能
- RTF（实时因子）≈ 1.0 — 4 分钟的歌曲需要约 4 分钟生成
- 输出：MP3，48kHz 立体声，128kbps

## 常见问题
1. **不要对 HeartCodec 使用 bf16** — 会降低音频质量。使用 fp32（默认值）。
2. **标签可能被忽略** — 已知问题（#90）。歌词倾向于占主导；尝试调整标签顺序。
3. **Triton 在 macOS 上不可用** — GPU 加速仅支持 Linux/CUDA。
4. **RTX 5080 不兼容** — 在上游 issue 中有报告。
5. 依赖固定冲突需要上述手动升级和补丁。

## 链接
- 仓库：https://github.com/HeartMuLa/heartlib
- 模型：https://huggingface.co/HeartMuLa
- 论文：https://arxiv.org/abs/2601.10547
- 许可证：Apache-2.0

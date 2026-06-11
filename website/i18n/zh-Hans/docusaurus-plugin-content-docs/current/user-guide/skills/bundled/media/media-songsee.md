---
title: "Songsee — 通过 CLI 生成音频频谱图/特征（mel、chroma、MFCC）"
sidebar_label: "Songsee"
description: "通过 CLI 生成音频频谱图/特征（mel、chroma、MFCC）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Songsee

通过 CLI 生成音频频谱图/特征（mel、chroma、MFCC）。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/media/songsee` |
| 版本 | `1.0.0` |
| 作者 | community |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `Audio`, `Visualization`, `Spectrogram`, `Music`, `Analysis` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# songsee

从音频文件生成频谱图（spectrogram）及多面板音频特征可视化图。

## 前置条件

需要安装 [Go](https://go.dev/doc/install)：
```bash
go install github.com/steipete/songsee/cmd/songsee@latest
```

可选：安装 `ffmpeg` 以支持 WAV/MP3 以外的格式。

## 快速开始

```bash
# 基本频谱图
songsee track.mp3

# 保存到指定文件
songsee track.mp3 -o spectrogram.png

# 多面板可视化网格
songsee track.mp3 --viz spectrogram,mel,chroma,hpss,selfsim,loudness,tempogram,mfcc,flux

# 时间切片（从 12.5s 开始，持续 8s）
songsee track.mp3 --start 12.5 --duration 8 -o slice.jpg

# 从 stdin 读取
cat track.mp3 | songsee - --format png -o out.png
```

## 可视化类型

使用 `--viz` 并以逗号分隔多个值：

| 类型 | 描述 |
|------|-------------|
| `spectrogram` | 标准频率频谱图 |
| `mel` | Mel 尺度频谱图 |
| `chroma` | 音高类别分布 |
| `hpss` | 谐波/打击乐分离 |
| `selfsim` | 自相似矩阵 |
| `loudness` | 随时间变化的响度 |
| `tempogram` | 节拍估计 |
| `mfcc` | Mel 频率倒谱系数 |
| `flux` | 频谱通量（起始点检测） |

多个 `--viz` 类型将以网格形式渲染为单张图像。

## 常用标志

| 标志 | 描述 |
|------|-------------|
| `--viz` | 可视化类型（逗号分隔） |
| `--style` | 色彩调色板：`classic`、`magma`、`inferno`、`viridis`、`gray` |
| `--width` / `--height` | 输出图像尺寸 |
| `--window` / `--hop` | FFT 窗口和跳跃大小 |
| `--min-freq` / `--max-freq` | 频率范围过滤 |
| `--start` / `--duration` | 音频时间切片 |
| `--format` | 输出格式：`jpg` 或 `png` |
| `-o` | 输出文件路径 |

## 注意事项

- WAV 和 MP3 原生解码；其他格式需要 `ffmpeg`
- 输出图像可使用 `vision_analyze` 进行检查，以实现自动化音频分析
- 适用于比较音频输出、调试合成过程或记录音频处理流水线
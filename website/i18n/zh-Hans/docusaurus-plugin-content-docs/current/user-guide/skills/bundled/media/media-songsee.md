---
title: "Songsee — 音频频谱图/特征（mel、chroma、MFCC）命令行工具"
sidebar_label: "Songsee"
description: "音频频谱图/特征（mel、chroma、MFCC）命令行工具"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Songsee

音频频谱图/特征（mel、chroma、MFCC）命令行工具。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/media/songsee` |
| 版本 | `1.0.0` |
| 作者 | 社区 |
| 许可证 | MIT |
| 标签 | `音频`、`可视化`、`频谱图`、`音乐`、`分析` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此技能时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# songsee

从音频文件生成频谱图和多面板音频特征可视化。

## 前置条件

需要 [Go](https://go.dev/doc/install)：
```bash
go install github.com/steipete/songsee/cmd/songsee@latest
```

可选：`ffmpeg` 用于 WAV/MP3 以外的格式。

## 快速开始

```bash
# 基本频谱图
songsee track.mp3

# 保存到指定文件
songsee track.mp3 -o spectrogram.png

# 多面板可视化网格
songsee track.mp3 --viz spectrogram,mel,chroma,hpss,selfsim,loudness,tempogram,mfcc,flux

# 时间切片（从 12.5 秒开始，持续 8 秒）
songsee track.mp3 --start 12.5 --duration 8 -o slice.jpg

# 从 stdin 读取
cat track.mp3 | songsee - --format png -o out.png
```

## 可视化类型

使用 `--viz` 并以逗号分隔值：

| 类型 | 描述 |
|---|---|
| `spectrogram` | 标准频率频谱图 |
| `mel` | Mel 频谱图 |
| `chroma` | 音高类分布 |
| `hpss` | 谐波/打击分离 |
| `selfsim` | 自相似矩阵 |
| `loudness` | 响度随时间变化 |
| `tempogram` | 速度估计 |
| `mfcc` | Mel 频率倒谱系数 |
| `flux` | 频谱通量（起始检测） |

多个 `--viz` 类型以网格形式渲染在一张图中。

## 常用标志

| 标志 | 描述 |
|---|---|
| `--viz` | 可视化类型（逗号分隔） |
| `--style` | 调色板：`classic`、`magma`、`inferno`、`viridis`、`gray` |
| `--width` / `--height` | 输出图片尺寸 |
| `--window` / `--hop` | FFT 窗口和步长大小 |
| `--min-freq` / `--max-freq` | 频率范围过滤 |
| `--start` / `--duration` | 音频的时间切片 |
| `--format` | 输出格式：`jpg` 或 `png` |
| `-o` | 输出文件路径 |

## 注意

- WAV 和 MP3 原生解码；其他格式需要 `ffmpeg`
- 输出图片可以使用 `vision_analyze` 检查以进行自动化音频分析
- 适用于比较音频输出、调试合成或记录音频处理管道

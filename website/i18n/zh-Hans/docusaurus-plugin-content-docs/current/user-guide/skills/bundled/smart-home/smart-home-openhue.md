---
title: "Openhue — 通过 OpenHue CLI 控制 Philips Hue 灯光、场景、房间"
sidebar_label: "Openhue"
description: "通过 OpenHue CLI 控制 Philips Hue 灯光、场景、房间"
---

{/* 本页面由 website/scripts/generate-skill-docs.py 从技能的 SKILL.md 自动生成。请编辑源 SKILL.md，而非此页面。 */}

# Openhue

通过 OpenHue CLI 控制 Philips Hue 灯光、场景、房间。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/smart-home/openhue` |
| 版本 | `1.0.0` |
| 作者 | 社区 |
| 许可证 | MIT |
| 标签 | `Smart-Home`, `Hue`, `Lights`, `IoT`, `Automation` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在此技能被触发时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# OpenHue CLI

通过 Hue Bridge 从终端控制 Philips Hue 灯光和场景。

## 前提条件

```bash
# Linux（预编译二进制文件）
curl -sL https://github.com/openhue/openhue-cli/releases/latest/download/openhue-linux-amd64 -o ~/.local/bin/openhue && chmod +x ~/.local/bin/openhue

# macOS
brew install openhue/cli/openhue-cli
```

首次运行需要按下 Hue Bridge 上的按钮进行配对。Bridge 必须在同一本地网络上。

## 使用时机

- "打开/关闭灯光"
- "调暗客厅灯光"
- "设置场景"或"电影模式"
- 控制特定的 Hue 房间、区域或单个灯泡
- 调整亮度、颜色或色温

## 常用命令

### 列出资源

```bash
openhue get light       # 列出所有灯光
openhue get room        # 列出所有房间
openhue get scene       # 列出所有场景
```

### 控制灯光

```bash
# 打开/关闭
openhue set light "Bedroom Lamp" --on
openhue set light "Bedroom Lamp" --off

# 亮度 (0-100)
openhue set light "Bedroom Lamp" --on --brightness 50

# 色温（暖到冷：153-500 mirek）
openhue set light "Bedroom Lamp" --on --temperature 300

# 颜色（按名称或十六进制值）
openhue set light "Bedroom Lamp" --on --color red
openhue set light "Bedroom Lamp" --on --rgb "#FF5500"
```

### 控制房间

```bash
# 关闭整个房间
openhue set room "Bedroom" --off

# 设置房间亮度
openhue set room "Bedroom" --on --brightness 30
```

### 场景

```bash
openhue set scene "Relax" --room "Bedroom"
openhue set scene "Concentrate" --room "Office"
```

## 快速预设

```bash
# 睡前（暗暖光）
openhue set room "Bedroom" --on --brightness 20 --temperature 450

# 工作模式（明亮冷光）
openhue set room "Office" --on --brightness 100 --temperature 250

# 电影模式（暗光）
openhue set room "Living Room" --on --brightness 10

# 全部关闭
openhue set room "Bedroom" --off
openhue set room "Office" --off
openhue set room "Living Room" --off
```

## 注意事项

- Bridge 必须与运行 Hermes 的机器在同一本地网络上
- 首次运行需要物理按下 Hue Bridge 上的按钮进行授权
- 颜色仅在支持彩色的灯泡上有效（不适用于纯白光型号）
- 灯光和房间名称区分大小写——使用 `openhue get light` 检查确切名称
- 配合 cron 定时任务效果很好（例如睡前调暗，起床时调亮）

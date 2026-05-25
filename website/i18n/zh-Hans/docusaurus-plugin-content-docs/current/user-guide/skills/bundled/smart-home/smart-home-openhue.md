---
title: "Openhue — 通过 OpenHue CLI 控制 Philips Hue 灯光、场景和房间"
sidebar_label: "Openhue"
description: "通过 OpenHue CLI 控制 Philips Hue 灯光、场景和房间"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Openhue

通过 OpenHue CLI 控制 Philips Hue 灯光、场景和房间。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/smart-home/openhue` |
| 版本 | `1.0.0` |
| 作者 | community |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `Smart-Home`, `Hue`, `Lights`, `IoT`, `Automation` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# OpenHue CLI

通过 Hue Bridge 从终端控制 Philips Hue 灯光和场景。

## 前提条件

```bash
# Linux (pre-built binary)
curl -sL https://github.com/openhue/openhue-cli/releases/latest/download/openhue-linux-amd64 -o ~/.local/bin/openhue && chmod +x ~/.local/bin/openhue

# macOS
brew install openhue/cli/openhue-cli
```

首次运行需要按下 Hue Bridge 上的按钮进行配对。Bridge 必须与运行设备处于同一本地网络。

## 使用场景

- "打开/关闭灯光"
- "调暗客厅灯光"
- "设置场景"或"影院模式"
- 控制特定 Hue 房间、区域或单个灯泡
- 调整亮度、颜色或色温

## 常用命令

### 列出资源

```bash
openhue get light       # List all lights
openhue get room        # List all rooms
openhue get scene       # List all scenes
```

### 控制灯光

```bash
# Turn on/off
openhue set light "Bedroom Lamp" --on
openhue set light "Bedroom Lamp" --off

# Brightness (0-100)
openhue set light "Bedroom Lamp" --on --brightness 50

# Color temperature (warm to cool: 153-500 mirek)
openhue set light "Bedroom Lamp" --on --temperature 300

# Color (by name or hex)
openhue set light "Bedroom Lamp" --on --color red
openhue set light "Bedroom Lamp" --on --rgb "#FF5500"
```

### 控制房间

```bash
# Turn off entire room
openhue set room "Bedroom" --off

# Set room brightness
openhue set room "Bedroom" --on --brightness 30
```

### 场景

```bash
openhue set scene "Relax" --room "Bedroom"
openhue set scene "Concentrate" --room "Office"
```

## 快速预设

```bash
# Bedtime (dim warm)
openhue set room "Bedroom" --on --brightness 20 --temperature 450

# Work mode (bright cool)
openhue set room "Office" --on --brightness 100 --temperature 250

# Movie mode (dim)
openhue set room "Living Room" --on --brightness 10

# Everything off
openhue set room "Bedroom" --off
openhue set room "Office" --off
openhue set room "Living Room" --off
```

## 注意事项

- Bridge 必须与运行 Hermes 的机器处于同一本地网络
- 首次运行需要物理按下 Hue Bridge 上的按钮进行授权
- 颜色功能仅适用于支持彩色的灯泡（不适用于纯白光型号）
- 灯光和房间名称区分大小写——使用 `openhue get light` 查看确切名称
- 可与 cron 作业配合实现定时照明控制（例如：睡前调暗、起床时调亮）
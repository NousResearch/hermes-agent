---
title: "Findmy — 通过 FindMy 追踪 Apple 设备/AirTag"
sidebar_label: "Findmy"
description: "通过 FindMy 追踪 Apple 设备/AirTag"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Findmy

在 macOS 上通过 FindMy.app 追踪 Apple 设备/AirTag。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/apple/findmy` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | macos |
| 标签 | `FindMy`, `AirTag`, `location`, `tracking`, `macOS`, `Apple` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Find My（Apple）

在 macOS 上通过 FindMy.app 追踪 Apple 设备和 AirTag。由于 Apple 未提供 FindMy 的 CLI，此 skill 使用 AppleScript 打开应用并通过截图读取设备位置。

## 前提条件

- **macOS**，已安装 Find My 应用并登录 iCloud
- 设备/AirTag 已在 Find My 中注册
- 终端已获得屏幕录制权限（系统设置 → 隐私与安全 → 屏幕录制）
- **可选但推荐**：安装 `peekaboo` 以获得更好的 UI 自动化体验：
  `brew install steipete/tap/peekaboo`

## 使用场景

- 用户询问"我的[设备/猫/钥匙/包]在哪里？"
- 追踪 AirTag 位置
- 查看设备位置（iPhone、iPad、Mac、AirPods）
- 随时间监控宠物或物品的移动轨迹（AirTag 巡逻路线）

## 方法一：AppleScript + 截图（基础方式）

### 打开 FindMy 并导航

```bash
# 打开 Find My 应用
osascript -e 'tell application "FindMy" to activate'

# 等待加载
sleep 3

# 对 Find My 窗口截图
screencapture -w -o /tmp/findmy.png
```

然后使用 `vision_analyze` 读取截图：
```
vision_analyze(image_url="/tmp/findmy.png", question="What devices/items are shown and what are their locations?")
```

### 切换标签页

```bash
# 切换到"设备"标签页
osascript -e '
tell application "System Events"
    tell process "FindMy"
        click button "Devices" of toolbar 1 of window 1
    end tell
end tell'

# 切换到"物品"标签页（AirTag）
osascript -e '
tell application "System Events"
    tell process "FindMy"
        click button "Items" of toolbar 1 of window 1
    end tell
end tell'
```

## 方法二：Peekaboo UI 自动化（推荐）

如果已安装 `peekaboo`，可使用它进行更可靠的 UI 交互：

```bash
# 打开 Find My
osascript -e 'tell application "FindMy" to activate'
sleep 3

# 捕获并标注 UI
peekaboo see --app "FindMy" --annotate --path /tmp/findmy-ui.png

# 通过元素 ID 点击特定设备/物品
peekaboo click --on B3 --app "FindMy"

# 捕获详情视图
peekaboo image --app "FindMy" --path /tmp/findmy-detail.png
```

然后使用 vision 进行分析：
```
vision_analyze(image_url="/tmp/findmy-detail.png", question="What is the location shown for this device/item? Include address and coordinates if visible.")
```

## 工作流：随时间追踪 AirTag 位置

用于监控 AirTag（例如追踪猫的巡逻路线）：

```bash
# 1. 打开 FindMy 并切换到"物品"标签页
osascript -e 'tell application "FindMy" to activate'
sleep 3

# 2. 点击 AirTag 物品（保持页面停留——AirTag 仅在页面处于活跃显示状态时才更新）

# 3. 定期捕获位置
while true; do
    screencapture -w -o /tmp/findmy-$(date +%H%M%S).png
    sleep 300  # 每 5 分钟一次
done
```

使用 vision 分析每张截图以提取坐标，然后汇总成路线。

## 限制

- FindMy **没有 CLI 或 API**——必须使用 UI 自动化
- AirTag 仅在 FindMy 页面处于活跃显示状态时才更新位置
- 位置精度取决于 FindMy 网络中附近的 Apple 设备
- 截图需要屏幕录制权限
- AppleScript UI 自动化可能在不同 macOS 版本间失效

## 规则

1. 追踪 AirTag 时保持 FindMy 应用在前台（最小化后更新将停止）
2. 使用 `vision_analyze` 读取截图内容——不要尝试直接解析像素
3. 如需持续追踪，使用 cronjob 定期捕获并记录位置
4. 尊重隐私——仅追踪用户本人拥有的设备/物品
---
title: "Findmy — 通过 FindMy 追踪 Apple 设备/AirTags"
sidebar_label: "Findmy"
description: "通过 FindMy 追踪 Apple 设备/AirTags"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Findmy

在 macOS 上通过 FindMy.app 追踪 Apple 设备和 AirTags。

## 技能元数据

| | |
|---|---|
| 来源 | 捆绑（默认安装） |
| 路径 | `skills/apple/findmy` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | macos |
| 标签 | `FindMy`、`AirTag`、`location`、`tracking`、`macOS`、`Apple` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 加载此技能时触发的完整技能定义。这是代理激活技能时看到的指令。
:::

# Find My (Apple)

通过 macOS 上的 FindMy.app 追踪 Apple 设备和 AirTags。由于 Apple 不提供 FindMy 的 CLI，此技能使用 AppleScript 打开应用并通过屏幕截图读取设备位置。

## 前置条件

- **macOS** 带有 Find My 应用且已登录 iCloud
- 设备/AirTags 已注册到 Find My
- 终端的屏幕录制权限（系统设置 → 隐私与安全性 → 屏幕录制）
- **可选但推荐**：安装 `peekaboo` 以获得更好的 UI 自动化：
  `brew install steipete/tap/peekaboo`

## 使用场景

- 用户问"我的 [设备/猫/钥匙/包] 在哪里？"
- 追踪 AirTag 位置
- 检查设备位置（iPhone、iPad、Mac、AirPods）
- 随时间监控宠物或物品移动（AirTag 巡逻路线）

## 方法 1：AppleScript + 截图（基础）

### 打开 FindMy 并导航

```bash
# 打开 Find My 应用
osascript -e 'tell application "FindMy" to activate'

# 等待加载
sleep 3

# 拍摄 Find My 窗口的截图
screencapture -w -o /tmp/findmy.png
```

然后使用 `vision_analyze` 读取截图：
```
vision_analyze(image_url="/tmp/findmy.png", question="What devices/items are shown and what are their locations?")
```

### 在标签页之间切换

```bash
# 切换到设备标签页
osascript -e '
tell application "System Events"
    tell process "FindMy"
        click button "Devices" of toolbar 1 of window 1
    end tell
end tell'

# 切换到物品标签页（AirTags）
osascript -e '
tell application "System Events"
    tell process "FindMy"
        click button "Items" of toolbar 1 of window 1
    end tell
end tell'
```

## 方法 2：Peekaboo UI 自动化（推荐）

如果安装了 `peekaboo`，使用它以获得更可靠的 UI 交互：

```bash
# 打开 Find My
osascript -e 'tell application "FindMy" to activate'
sleep 3

# 捕获并标注 UI
peekaboo see --app "FindMy" --annotate --path /tmp/findmy-ui.png

# 按元素 ID 点击特定设备/物品
peekaboo click --on B3 --app "FindMy"

# 捕获详情视图
peekaboo image --app "FindMy" --path /tmp/findmy-detail.png
```

然后用 vision 分析：
```
vision_analyze(image_url="/tmp/findmy-detail.png", question="What is the location shown for this device/item? Include address and coordinates if visible.")
```

## 工作流程：随时间追踪 AirTag 位置

对于监控 AirTag（例如追踪猫的巡逻路线）：

```bash
# 1. 打开 FindMy 到物品标签页
osascript -e 'tell application "FindMy" to activate'
sleep 3

# 2. 点击 AirTag 项目（保持在页面 — AirTag 仅在页面打开时更新位置）

# 3. 定期捕获位置
while true; do
    screencapture -w -o /tmp/findmy-$(date +%H%M%S).png
    sleep 300  # 每 5 分钟
done
```

用 vision 分析每张截图提取坐标，然后编制路线。

## 限制

- FindMy **没有 CLI 或 API** — 必须使用 UI 自动化
- AirTags 仅在 FindMy 页面主动显示时更新位置
- 位置精度取决于 FindMy 网络中附近的 Apple 设备
- 需要屏幕录制权限才能截图
- AppleScript UI 自动化可能在不同 macOS 版本间失效

## 规则

1. 追踪 AirTags 时保持 FindMy 应用在前台（最小化时更新停止）
2. 使用 `vision_analyze` 读取截图内容 — 不要尝试解析像素
3. 对于持续追踪，使用 cronjob 定期捕获并记录位置
4. 尊重隐私 — 仅追踪用户拥有的设备/物品

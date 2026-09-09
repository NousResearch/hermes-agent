---
title: "Pokemon Player — 通过无头模拟器 + RAM 读取玩 Pokemon"
sidebar_label: "Pokemon Player"
description: "通过无头模拟器 + RAM 读取玩 Pokemon"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Pokemon Player

通过无头模拟器 + RAM 读取玩 Pokemon。

## 技能元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/gaming/pokemon-player` 安装 |
| 路径 | `optional-skills/gaming/pokemon-player` |
| 平台 | linux、macos、windows |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在此技能触发时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# Pokemon Player

通过 `pokemon-agent` 包使用无头模拟玩 Pokemon 游戏。

## 何时使用

- 用户说"play pokemon"、"start pokemon"、"pokemon game"
- 用户询问 Pokemon Red、Blue、Yellow、FireRed 等
- 用户想观看 AI 玩 Pokemon
- 用户引用 ROM 文件（.gb、.gbc、.gba）

## 启动流程

### 1. 首次设置
克隆仓库，设置 Python 3.10+ 虚拟环境。使用 uv（推荐）创建 venv 并安装包。

**你还需要 ROM 文件。询问用户的 ROM。永远不要下载或提供 ROM 文件。**

### 2. 启动游戏服务器

```bash
pokemon-agent serve --rom <path-to-rom> --port 9876
```

### 3. 设置实时仪表板

使用 SSH 反向隧道通过 localhost.run 让用户在浏览器中查看仪表板。

## 保存和加载

- 每 15-20 回合游戏后保存
- 始终在道馆战、对手遭遇或危险战斗前保存

## 游戏循环

### 步骤 1：OBSERVE — 检查状态并截图

GET /state 获取位置、HP、战斗、对话。
GET /screenshot 并保存到 /tmp/pokemon.png，然后使用 vision_analyze。
**始终同时做两者** — RAM 状态给出数字，视觉给出空间感知。

### 步骤 2：ORIENT
- 屏幕上有对话/文字 → 推进
- 战斗中 → 战斗或逃跑
- 队伍受伤 → 前往 Pokemon Center

### 步骤 3：DECIDE
优先级：对话 > 战斗 > 治疗 > 故事目标 > 训练 > 探索

### 步骤 4：ACT — 最多移动 2-4 步，然后重新检查

### 步骤 5：VERIFY — 每次移动序列后截图
这是**最重要的步骤**。没有视觉你**会**迷路。

### 步骤 6：RECORD — 使用 PKM: 前缀记录进度到内存

### 步骤 7：SAVE — 定期保存

## 来自经验的关键技巧

### 持续使用视觉
- 每 2-4 个移动步截图
- RAM 状态告诉你位置和 HP 但**不**告诉你周围是什么
- 悬崖、围栏、标志、建筑门、NPC — 仅通过截图可见

### 传送过渡需要额外等待时间
穿过门或楼梯时，在地图过渡期间屏幕渐黑。你必须等待完成。在任何门/楼梯传送后添加 2-3 个 wait_60 动作。

### 建筑出口陷阱
离开建筑时，你直接出现在门**前方**。向北走会回到里面。始终先侧步。

## 战斗策略

### 决策树
1. 想捕捉？→ 削弱然后投 Poke Ball
2. 不需要的野生？→ 逃跑
3. 属性优势？→ 使用效果拔群的招式

### Gen 1 属性表（关键对决）
- 水克火、地面、岩石
- 火克草、虫、冰
- 草克水、地面、岩石
- 电克水、飞行
- 地面克火、电、岩石、毒
- 超能力克格斗、毒（Gen 1 中统治级！）

## 内存约定

| 前缀 | 用途 | 示例 |
|------|------|------|
| PKM:OBJECTIVE | 当前目标 | 从 Viridian 商店获取包裹 |
| PKM:MAP | 导航知识 | Viridian：商店在东北方 |
| PKM:STRATEGY | 战斗/队伍计划 | 在 Misty 之前需要草属性 |
| PKM:PROGRESS | 里程碑跟踪 | 击败对手，前往 Viridian |
| PKM:STUCK | 卡住情况 | y=28 处悬崖向右绕行 |

## 陷阱

- 永远不要下载或提供 ROM 文件
- 不要在检查视觉前发送超过 4-5 个动作
- 离开建筑后始终先侧步再向北
- 始终在门/楼梯传送后添加 wait_60 x2-3
- 通过 RAM 的对话检测不可靠——用截图验证

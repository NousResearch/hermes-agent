---
title: "Petdex — 为 Hermes 安装和选择动画宠物精灵"
sidebar_label: "Petdex"
description: "为 Hermes 安装和选择动画宠物精灵"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Petdex

为 Hermes 安装和选择动画宠物精灵。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/productivity/petdex` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | linux、macos、windows |
| 标签 | `petdex`、`mascot`、`display`、`cli`、`tui`、`desktop` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在此技能触发时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# Petdex 技能

从公共 [petdex](https://github.com/crafter-station/petdex) 画廊浏览、安装和选择动画"宠物"精灵。已安装的宠物会响应代理活动（空闲、运行工具、审核、错误、完成），跨 Hermes CLI、TUI 和桌面应用。此技能驱动 `hermes pets` CLI 和 `display.pet` 配置——它不生成精灵图。

## 何时使用

- 用户想要桌面/终端吉祥物或询问"宠物"/petdex。
- 用户想要更改、预览或禁用活动宠物。
- 诊断宠物为何不显示（终端图形支持、配置）。

## 前提条件

- 网络访问 `petdex.dev` 获取画廊/清单（只读，无需认证）。
- Pillow（Hermes 核心依赖）用于精灵解码——已安装。
- 为获得完整的终端渲染保真度：支持图形的终端（kitty、
  Ghostty、WezTerm、iTerm2 或 sixel）。否则将自动使用 truecolor Unicode
  半块回退。

## 如何运行

使用 `terminal` 工具运行 `hermes pets <subcommand>`。

## 快速参考

| 目标 | 命令 |
| --- | --- |
| 浏览画廊 | `hermes pets list`（添加子串进行过滤：`hermes pets list cat`） |
| 列出已安装宠物 | `hermes pets list --installed` |
| 安装宠物 | `hermes pets install <slug>`（添加 `--select` 使其激活） |
| 设置活动宠物 | `hermes pets select <slug>`（省略 slug 使用选择器） |
| 调整所有位置的宠物大小 | `hermes pets scale <factor>`（例如 `0.5`，范围 0.1–3.0） |
| 在终端中预览/动画 | `hermes pets show [slug] [--cycle] [--state run]` |
| 禁用宠物 | `hermes pets off` |
| 移除宠物 | `hermes pets remove <slug>` |
| 诊断设置 | `hermes pets doctor` |

## 操作流程

1. 查找宠物：`hermes pets list <query>` 并记下其 `slug`。
2. 安装 + 激活：`hermes pets install <slug> --select`。
3. 预览：`hermes pets show`（Ctrl+C 停止）。
4. 确认设置：`hermes pets doctor`——显示已解析的宠物、配置的
   渲染模式、检测到的终端图形协议和有效模式。

宠物安装到 `<HERMES_HOME>/pets/<slug>/`（感知配置文件）。选择宠物会将
`display.pet.slug` + `display.pet.enabled` 写入 `config.yaml`。

## 配置

在 `config.yaml` 的 `display.pet` 下：

- `enabled`（bool）——主开关。
- `slug`（str）——活动宠物；空 = 第一个已安装的。
- `render_mode`——`auto`（检测）| `kitty` | `iterm` | `sixel` | `unicode` | `off`。
- `scale`（float）——原生 192×208 帧的屏幕大小（默认 0.33，
  范围 0.1–3.0）。一个旋钮调整所有表面大小；通过
  `hermes pets scale <factor>`、`/pet scale` 斜杠命令或桌面
  外观滑块设置。
- `unicode_cols`（int）——Unicode 回退的列宽。

## 陷阱

- 宠物只有在安装且选择后才会显示（`enabled: true`）。
- 在管道/重定向中（无 TTY）终端渲染默认禁用。
- petdex npm CLI 安装到 `~/.codex/pets`；Hermes 使用自己的
  配置文件范围的 `<HERMES_HOME>/pets/`——通过 `hermes pets` 安装。

## 验证

- `hermes pets doctor` 在宠物安装、选择、
  启用且 Pillow 可导入时报告 `✓ ready`。

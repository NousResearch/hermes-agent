---
title: "Apple Notes — 通过 memo CLI 管理 Apple Notes：创建、搜索、编辑"
sidebar_label: "Apple Notes"
description: "通过 memo CLI 管理 Apple Notes：创建、搜索、编辑"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Apple Notes

通过 memo CLI 管理 Apple Notes：创建、搜索、编辑。

## 技能元数据

| | |
|---|---|
| 来源 | 捆绑（默认安装） |
| 路径 | `skills/apple/apple-notes` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | macos |
| 标签 | `Notes`、`Apple`、`macOS`、`note-taking` |
| 相关技能 | [`obsidian`](/docs/user-guide/skills/bundled/note-taking/note-taking-obsidian) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 加载此技能时触发的完整技能定义。这是代理激活技能时看到的指令。
:::

# Apple Notes

使用 `memo` 直接从终端管理 Apple Notes。笔记通过 iCloud 在所有 Apple 设备间同步。

## 前置条件

- **macOS** 带有 Notes.app
- 安装：`brew tap antoniorodr/memo && brew install antoniorodr/memo/memo`
- 出现提示时授予 Notes.app 的自动化访问权限（系统设置 → 隐私与安全性 → 自动化）

## 使用场景

- 用户要求创建、查看或搜索 Apple Notes
- 保存信息到 Notes.app 以便跨设备访问
- 将笔记组织到文件夹
- 导出笔记为 Markdown/HTML

## 不使用场景

- Obsidian 保险库管理 → 使用 `obsidian` 技能
- Bear Notes → 独立应用（此处不支持）
- 快速代理内部笔记 → 改用 `memory` 工具

## 快速参考

### 查看笔记

```bash
memo notes                        # 列出所有笔记
memo notes -f "Folder Name"       # 按文件夹筛选
memo notes -s "query"             # 搜索笔记（模糊匹配）
```

### 创建笔记

```bash
memo notes -a                     # 交互式编辑器
memo notes -a "Note Title"        # 快速添加带标题
```

### 编辑笔记

```bash
memo notes -e                     # 交互式选择编辑
```

### 删除笔记

```bash
memo notes -d                     # 交互式选择删除
```

### 移动笔记

```bash
memo notes -m                     # 将笔记移动到文件夹（交互式）
```

### 导出笔记

```bash
memo notes -ex                    # 导出为 HTML/Markdown
```

## 限制

- 无法编辑包含图片或附件的笔记
- 交互式提示需要终端访问（需要时使用 pty=true）
- 仅 macOS — 需要 Apple Notes.app

## 规则

1. 当用户想要跨设备同步（iPhone/iPad/Mac）时，优先使用 Apple Notes
2. 对于不需要同步的代理内部笔记，使用 `memory` 工具
3. 对于 Markdown 原生的知识管理，使用 `obsidian` 技能

---
title: "Obsidian — 在 Obsidian 知识库中读取、搜索、创建和编辑笔记"
sidebar_label: "Obsidian"
description: "在 Obsidian 知识库中读取、搜索、创建和编辑笔记"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Obsidian

在 Obsidian 知识库中读取、搜索、创建和编辑笔记。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/note-taking/obsidian` |
| 平台 | linux, macos, windows |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# Obsidian 知识库

将此 skill 用于以文件系统为核心的 Obsidian 知识库操作：读取笔记、列出笔记、搜索笔记文件、创建笔记、追加内容以及添加 wikilink。

## 知识库路径

在调用文件工具之前，先确定已知或已解析的知识库路径。

知识库路径的约定文档为 `OBSIDIAN_VAULT_PATH` 环境变量，例如来自 `~/.hermes/.env`。若未设置，则使用 `~/Documents/Obsidian Vault`。

文件工具不会展开 shell 变量。不要将包含 `$OBSIDIAN_VAULT_PATH` 的路径传递给 `read_file`、`write_file`、`patch` 或 `search_files`；应先解析知识库路径，再传入具体的绝对路径。知识库路径可能包含空格，这也是优先使用文件工具而非 shell 命令的另一个原因。

若知识库路径未知，可使用 `terminal` 解析 `OBSIDIAN_VAULT_PATH` 或检查备用路径是否存在。一旦路径确定，切换回文件工具。

## 读取笔记

使用 `read_file` 并传入笔记的已解析绝对路径。优先使用此方式而非 `cat`，因为它提供行号和分页功能。

## 列出笔记

使用 `search_files`，将 `target` 设为 `"files"` 并传入已解析的知识库路径。优先使用此方式而非 `find` 或 `ls`。

- 若要列出所有 markdown 笔记，在知识库路径下使用 `pattern: "*.md"`。
- 若要列出子文件夹，在该子文件夹的绝对路径下进行搜索。

## 搜索

使用 `search_files` 进行文件名和内容搜索。优先使用此方式而非 `grep`、`find` 或 `ls`。

- 搜索文件名时，使用 `search_files`，将 `target` 设为 `"files"` 并指定文件名 `pattern`。
- 搜索笔记内容时，使用 `search_files`，将 `target` 设为 `"content"`，将内容正则表达式作为 `pattern`，并在需要将匹配限制为 markdown 笔记时设置 `file_glob: "*.md"`。

## 创建笔记

使用 `write_file` 并传入已解析的绝对路径和完整 markdown 内容。优先使用此方式而非 shell heredoc 或 `echo`，因为它可避免 shell 引号问题并返回结构化结果。

## 追加内容到笔记

在操作不复杂的情况下，优先使用原生文件工具工作流：

- 使用 `read_file` 读取目标笔记。
- 当存在稳定的上下文时（例如在现有标题后添加章节或在已知尾部块之前追加），使用 `patch` 进行锚定追加。
- 当重写整个笔记比构造脆弱的 patch 更清晰时，使用 `write_file`。

使用 `patch` 进行锚定追加时，将锚点替换为锚点加新内容。

若无稳定上下文的简单追加，且 `terminal` 是最清晰安全的选项，则可接受使用 `terminal`。

## 定向编辑

当现有内容提供稳定上下文时，使用 `patch` 进行笔记的局部修改。优先使用此方式而非 shell 文本重写。

## Wikilink

Obsidian 使用 `[[Note Name]]` 语法链接笔记。创建笔记时，使用这种语法链接相关内容。
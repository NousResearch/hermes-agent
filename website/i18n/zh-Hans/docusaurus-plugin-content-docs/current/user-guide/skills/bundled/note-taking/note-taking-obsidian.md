---
title: "Obsidian — 在 Obsidian 知识库中读取、搜索、创建和编辑笔记"
sidebar_label: "Obsidian"
description: "在 Obsidian 知识库中读取、搜索、创建和编辑笔记"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Obsidian

在 Obsidian 知识库中读取、搜索、创建和编辑笔记。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/note-taking/obsidian` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此技能时加载的完整技能定义。这是技能激活时代理看到的指令内容。
:::

# Obsidian 知识库

使用此技能进行文件系统优先的 Obsidian 知识库操作：读取笔记、列出笔记、搜索笔记文件、创建笔记、追加内容和添加维基链接。

## 知识库路径

在调用文件工具之前，使用已知或已解析的知识库路径。

文档中记录的知识库路径约定是 `OBSIDIAN_VAULT_PATH` 环境变量，例如来自 `~/.hermes/.env`。如果未设置，使用 `~/Documents/Obsidian Vault`。

文件工具不会展开 shell 变量。不要将包含 `$OBSIDIAN_VAULT_PATH` 的路径传递给 `read_file`、`write_file`、`patch` 或 `search_files`；先解析知识库路径，然后传递具体的绝对路径。知识库路径可能包含空格，这也是优先使用文件工具而非 shell 命令的另一个原因。

如果知识库路径未知，使用 `terminal` 来解析 `OBSIDIAN_VAULT_PATH` 或检查备用路径是否存在是可接受的。一旦路径已知，切换回文件工具。

## 读取笔记

使用 `read_file` 并传入笔记的已解析绝对路径。优先使用此方法而非 `cat`，因为它提供行号和分页。

## 列出笔记

使用 `search_files` 并设置 `target: "files"` 和已解析的知识库路径。优先使用此方法而非 `find` 或 `ls`。

- 要列出所有 Markdown 笔记，在知识库路径下使用 `pattern: "*.md"`。
- 要列出子文件夹，在该子文件夹的绝对路径下搜索。

## 搜索

使用 `search_files` 进行文件名和内容搜索。优先使用此方法而非 `grep`、`find` 或 `ls`。

- 对于文件名，使用 `search_files` 并设置 `target: "files"` 和文件名 `pattern`。
- 对于笔记内容，使用 `search_files` 并设置 `target: "content"`，内容正则表达式作为 `pattern`，当需要限制匹配 Markdown 笔记时使用 `file_glob: "*.md"`。

## 创建笔记

使用 `write_file` 并传入已解析的绝对路径和完整的 Markdown 内容。优先使用此方法而非 shell heredoc 或 `echo`，因为它避免 shell 引用问题并返回结构化结果。

## 追加笔记内容

当不显得笨拙时，优先使用原生文件工具工作流：

- 使用 `read_file` 读取目标笔记。
- 当有稳定的上下文时使用 `patch` 进行锚定追加，例如在现有标题后添加节或在已知的尾部块之前追加。
- 当重写整个笔记比构建脆弱的补丁更清晰时使用 `write_file`。

对于使用 `patch` 的锚定追加，将锚点替换为锚点加上新内容。

对于没有稳定上下文的简单追加，如果它是最清晰的安全选项，`terminal` 是可接受的。

## 定向编辑

当当前内容提供稳定上下文时，使用 `patch` 进行聚焦的笔记更改。优先使用此方法而非 shell 文本重写。

## 维基链接

Obsidian 使用 `[[笔记名称]]` 语法链接笔记。创建笔记时，使用这些来链接相关内容。

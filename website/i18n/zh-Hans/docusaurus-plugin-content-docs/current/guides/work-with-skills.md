---
sidebar_position: 12
title: "使用技能"
description: "查找、安装、使用和创建技能 - 这些按需知识会教 Hermes 新工作流"
---

# 使用技能

技能是按需加载的知识文档，用来教 Hermes 如何处理特定任务——从生成 ASCII art 到管理 GitHub PR。这个指南会带你了解日常怎么使用它们。

完整技术参考请见 [技能系统](/user-guide/features/skills)。

## 查找技能

Hermes 安装时会自带 bundled skills。查看可用项：

```bash
# 在任何聊天会话里：
/skills

# 或在 CLI 里：
hermes skills list
```

你会看到一个紧凑列表，包含名称和简介。

### 搜索技能

```bash
/skills search docker
/skills search music
```

### 技能中心

官方 optional skills 可以通过 Hub 浏览：

```bash
/skills browse
/skills search blockchain
```

## 使用技能

每个已安装技能都会自动变成一个斜杠命令。直接输入它的名字：

```bash
/ascii-art Make a banner that says "HELLO WORLD"
/plan Design a REST API for a todo app
/github-pr-workflow Create a PR for the auth refactor

# 只输入 skill 名也可以加载它，然后继续描述任务
/excalidraw
```

你也可以在自然对话中请求 Hermes 使用某个技能，它会通过 `skill_view` 加载。

### 渐进式披露

Skills 采用节省 token 的加载方式：

1. `skills_list()` - 轻量级技能列表
2. `skill_view(name)` - 单个 SKILL.md 的完整内容
3. `skill_view(name, file_path)` - 只在需要时加载附属参考文件

## 从中心安装

官方 optional skills 随 Hermes 一起提供，但默认不会激活。你可以显式安装：

```bash
hermes skills install official/research/arxiv
/skills install official/creative/songwriting-and-ai-music
hermes skills install https://sharethis.chat/SKILL.md
```

安装后：
1. skill 目录会复制到 `~/.hermes/skills/`
2. 它会出现在 `skills_list` 输出中
3. 它会成为可用的 slash command

### 验证安装

```bash
hermes skills list | grep arxiv
```

## 插件提供的技能

插件也可以带自己的 skills。为了避免命名冲突，使用命名空间名字，如 `plugin:skill`。

```bash
skill_view("superpowers:writing-plans")
skill_view("writing-plans")
```

插件技能不会进入系统提示词中的 `<available_skills>` 索引，它们是显式按需加载的。

如果你想在自己的插件里发布技能，见 [构建 Hermes 插件 → 打包技能](/guides/build-a-hermes-plugin#bundle-skills)。

## 配置技能设置

有些技能会在 frontmatter 中声明它们需要的配置：

```yaml
metadata:
  hermes:
    config:
      - key: tenor.api_key
        description: "Tenor API key for GIF search"
        prompt: "Enter your Tenor API key"
        url: "https://developers.google.com/tenor/guides/quickstart"
```

首次加载时，Hermes 会提示你输入这些值，并把它们存进 `config.yaml`。

## 创建你自己的技能

技能本质上就是带 YAML frontmatter 的 Markdown 文件。创建一个通常不到五分钟。

### 1. 创建目录

```bash
mkdir -p ~/.hermes/skills/my-category/my-skill
```

### 2. 编写 SKILL.md

### 3. 添加参考文件（可选）

### 4. 测试它

```bash
hermes chat -q "/my-skill help me with the thing"
```

## 按平台管理技能

```bash
hermes skills
```

这会打开一个交互式 TUI，用来按平台启用/禁用 skills。

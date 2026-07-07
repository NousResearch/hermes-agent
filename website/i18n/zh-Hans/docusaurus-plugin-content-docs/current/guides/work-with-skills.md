---
sidebar_position: 12
title: "使用 Skills"
description: "查找、安装、使用和创建 skills——按需加载的知识文档，用于教会 Hermes 新的工作流程"
---

# 使用 Skills

Skills（技能）是按需加载的知识文档，用于教会 Hermes 如何处理特定任务——从生成 ASCII 艺术到管理 GitHub PR。本指南介绍日常使用方法。

完整技术参考请见 [Skills 系统](/user-guide/features/skills)。

---

## 查找 Skills

每个 Hermes 安装都内置了捆绑的 skills。查看可用列表：

```bash
# 在任意聊天会话中：
/skills

# 或通过 CLI：
hermes skills list
```

输出包含名称和描述的紧凑列表：

```
ascii-art         Generate ASCII art using pyfiglet, cowsay, boxes...
arxiv             Search and retrieve academic papers from arXiv...
github-pr-workflow Full PR lifecycle — create branches, commit...
plan              Plan mode — inspect context, write a markdown...
excalidraw        Create hand-drawn style diagrams using Excalidraw...
```

### 搜索 Skill

```bash
# 按关键词搜索
/skills search docker
/skills search music
```

### Skills Hub

官方可选 skills（较重或小众、默认未激活的 skills）可通过 Hub 获取：

```bash
# 浏览官方可选 skills
/skills browse

# 搜索 Hub
/skills search blockchain
```

---

## 使用 Skill

每个已安装的 skill 自动成为一个斜杠命令。直接输入其名称即可：

```bash
# 加载 skill 并指定任务
/ascii-art Make a banner that says "HELLO WORLD"
/plan Design a REST API for a todo app
/github-pr-workflow Create a PR for the auth refactor

# 只输入 skill 名称（不带任务）会加载它并让你描述需求
/excalidraw
```

你也可以通过自然对话触发 skills——告诉 Hermes 使用某个特定 skill，它会通过 `skill_view` 工具加载。

### 渐进式加载

Skills 采用 token 高效的加载模式，agent 不会一次性加载所有内容：

1. **`skills_list()`** — 所有 skills 的紧凑列表（约 3k tokens），在会话开始时加载。
2. **`skill_view(name)`** — 单个 skill 的完整 SKILL.md 内容，在 agent 判断需要该 skill 时加载。
3. **`skill_view(name, file_path)`** — skill 内的特定参考文件，仅在需要时加载。

这意味着 skills 在真正被使用之前不消耗任何 tokens。

---

## 从 Hub 安装

官方可选 skills 随 Hermes 一起发布，但默认未激活，需显式安装：

```bash
# 安装官方可选 skill
hermes skills install official/research/arxiv

# 在聊天会话中从 Hub 安装
/skills install official/creative/songwriting-and-ai-music

# 直接从任意 HTTP(S) URL 安装单文件 SKILL.md
hermes skills install https://sharethis.chat/SKILL.md
/skills install https://example.com/SKILL.md --name my-skill
```

安装过程：
1. skill 目录被复制到 `~/.hermes/skills/`
2. 出现在 `skills_list` 输出中
3. 成为可用的斜杠命令

:::tip
已安装的 skills 在新会话中生效。如需在当前会话中立即使用，可用 `/reset` 开启新会话，或添加 `--now` 参数立即使 prompt 缓存失效（下一轮会消耗更多 tokens）。
:::

### 验证安装

```bash
# 确认已安装
hermes skills list | grep arxiv

# 或在聊天中
/skills search arxiv
```

---

## 插件提供的 Skills

插件可以使用命名空间名称（`plugin:skill`）捆绑自己的 skills，以避免与内置 skills 发生名称冲突。

```bash
# 通过限定名称加载插件 skill
skill_view("superpowers:writing-plans")

# 同名的内置 skill 不受影响
skill_view("writing-plans")
```

插件 skills **不会**列在系统 prompt 中，也不出现在 `skills_list` 中。它们是按需加载的——当你知道某个插件提供了某个 skill 时，显式加载它。加载后，agent 会看到一个横幅，列出同一插件的其他 skills。

关于如何在自己的插件中捆绑 skills，请参见 [构建 Hermes 插件 → 捆绑 skills](/developer-guide/plugins#bundle-skills)。

---

## 配置 Skill 设置

部分 skills 在 frontmatter 中声明了所需的配置：

```yaml
metadata:
  hermes:
    config:
      - key: tenor.api_key
        description: "Tenor API key for GIF search"
        prompt: "Enter your Tenor API key"
        url: "https://developers.google.com/tenor/guides/quickstart"
```

当带有配置的 skill 首次加载时，Hermes 会提示你输入相应值，并将其存储在 `config.yaml` 的 `skills.config.*` 下。

通过 CLI 管理 skill 配置：

```bash
# 对特定 skill 进行交互式配置
hermes skills config gif-search

# 查看所有 skill 配置
hermes config get skills.config
```

---

## 创建自己的 Skill

Skills 只是带有 YAML frontmatter 的 Markdown 文件，创建一个不超过五分钟。

### 1. 创建目录

```bash
mkdir -p ~/.hermes/skills/my-category/my-skill
```

### 2. 编写 SKILL.md

```markdown title="~/.hermes/skills/my-category/my-skill/SKILL.md"
---
name: my-skill
description: Brief description of what this skill does
version: 1.0.0
metadata:
  hermes:
    tags: [my-tag, automation]
    category: my-category
---

# My Skill

## When to Use
Use this skill when the user asks about [specific topic] or needs to [specific task].

## Procedure
1. First, check if [prerequisite] is available
2. Run `command --with-flags`
3. Parse the output and present results

## Pitfalls
- Common failure: [description]. Fix: [solution]
- Watch out for [edge case]

## Verification
Run `check-command` to confirm the result is correct.
```

### 3. 添加参考文件（可选）

Skills 可以包含 agent 按需加载的辅助文件：

```
my-skill/
├── SKILL.md                    # 主 skill 文档
├── references/
│   ├── api-docs.md             # agent 可查阅的 API 参考
│   └── examples.md             # 示例输入/输出
├── templates/
│   └── config.yaml             # agent 可使用的模板文件
└── scripts/
    └── setup.sh                # agent 可执行的脚本
```

在 SKILL.md 中引用这些文件：

```markdown
For API details, load the reference: `skill_view("my-skill", "references/api-docs.md")`
```

### 4. 测试

开启新会话并测试你的 skill：

```bash
hermes chat -q "/my-skill help me with the thing"
```

Skill 会自动出现——无需注册。放入 `~/.hermes/skills/` 即可立即生效。

:::info
Agent 也可以使用 `skill_manage` 自行创建和更新 skills。解决复杂问题后，Hermes 可能会主动提议将该方法保存为 skill，以便下次使用。
:::

---

## 按平台管理 Skills

控制哪些 skills 在哪些平台上可用：

```bash
hermes skills
```

这会打开一个交互式 TUI，你可以按平台（CLI、Telegram、Discord 等）启用或禁用 skills。当你希望某些 skills 仅在特定场景下可用时非常有用——例如，在 Telegram 上禁用开发类 skills。

---

## Skills 与 Memory 的区别

两者都跨会话持久化，但用途不同：

| | Skills | Memory |
|---|---|---|
| **内容** | 程序性知识——如何做事 | 事实性知识——事物是什么 |
| **时机** | 按需加载，仅在相关时加载 | 自动注入每个会话 |
| **大小** | 可以较大（数百行） | 应保持紧凑（仅关键事实） |
| **开销** | 加载前零 tokens | 少量但持续的 token 开销 |
| **示例** | "如何部署到 Kubernetes" | "用户偏好深色模式，位于 PST 时区" |
| **创建者** | 你、agent 或从 Hub 安装 | Agent，基于对话内容 |

**经验法则：** 如果你会把它写进参考文档，它就是 skill；如果你会把它写在便利贴上，它就是 memory。

---

## 使用技巧

**保持 skills 聚焦。** 试图涵盖"所有 DevOps"的 skill 会过于冗长且模糊。专注于"将 Python 应用部署到 Fly.io"的 skill 才足够具体，真正有用。

**让 agent 创建 skills。** 完成复杂的多步骤任务后，Hermes 通常会主动提议将该方法保存为 skill。接受它——这些由 agent 编写的 skills 会捕捉到完整的工作流程，包括过程中发现的各种坑。

**使用分类目录。** 将 skills 整理到子目录中（`~/.hermes/skills/devops/`、`~/.hermes/skills/research/` 等），保持列表整洁，并帮助 agent 更快找到相关 skills。

**及时更新过时的 skills。** 如果使用某个 skill 时遇到它未覆盖的问题，告诉 Hermes 用你学到的内容更新该 skill。不维护的 skills 会成为负担。

---

*完整的 skills 参考——frontmatter 字段、条件激活、外部目录等——请见 [Skills 系统](/user-guide/features/skills)。*
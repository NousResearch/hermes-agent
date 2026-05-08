---
sidebar_position: 3
title: "创建技能"
description: "如何为 Hermes Agent 创建技能 — SKILL.md 格式、指南和发布"
---

# 创建技能

技能是为 Hermes Agent 添加新能力的首选方式。它们比工具更容易创建，不需要对代理进行代码更改，可以与社区共享。

## 应该是技能还是工具？

符合以下条件时，制作**技能**：
- 能力可以表达为指令 + shell 命令 + 现有工具
- 包装代理可以通过 `terminal` 或 `web_extract` 调用的外部 CLI 或 API
- 不需要烘焙到代理中的自定义 Python 集成或 API 密钥管理
- 示例：arXiv 搜索、git 工作流、Docker 管理、PDF 处理、通过 CLI 工具发送邮件

符合以下条件时，制作**工具**：
- 需要与 API 密钥、auth 流程或多组件配置进行端到端集成
- 需要必须每次精确执行的自定义处理逻辑
- 处理二进制数据、流式传输或实时事件
- 示例：浏览器自动化、TTS、视觉分析

## 技能目录结构

捆绑技能组织在 `skills/` 中按类别分类。官方可选技能在 `optional-skills/` 中使用相同的结构：

```text
skills/
├── research/
│   └── arxiv/
│       ├── SKILL.md              # 必需：主要指令
│       └── scripts/              # 可选：辅助脚本
│           └── search_arxiv.py
├── productivity/
│   └── ocr-and-documents/
│       ├── SKILL.md
│       ├── scripts/
│       └── references/
└── ...
```

## SKILL.md 格式

```markdown
---
name: my-skill
description: Brief description (shown in skill search results)
version: 1.0.0
author: Your Name
license: MIT
platforms: [macos, linux]          # Optional — restrict to specific OS platforms
                                   #   Valid: macos, linux, windows
                                   #   Omit to load on all platforms (default)
metadata:
  hermes:
    tags: [Category, Subcategory, Keywords]
    related_skills: [other-skill-name]
    requires_toolsets: [web]            # Optional — only show when these toolsets are active
    requires_tools: [web_search]        # Optional — only show when these tools are available
    fallback_for_toolsets: [browser]    # Optional — hide when these toolsets are active
    fallback_for_tools: [browser_navigate]  # Optional — hide when these tools exist
    config:                              # Optional — config.yaml settings the skill needs
      - key: my.setting
        description: "What this setting controls"
        default: "sensible-default"
        prompt: "Display prompt for setup"
required_environment_variables:          # Optional — env vars the skill needs
  - name: MY_API_KEY
    prompt: "Enter your API key"
    help: "Get one at https://example.com"
    required_for: "API access"
---

# Skill Title

Brief intro.

## When to Use
Trigger conditions — when should the agent load this skill?

## Quick Reference
Table of common commands or API calls.

## Procedure
Step-by-step instructions the agent follows.

## Pitfalls
Known failure modes and how to handle them.

## Verification
How the agent confirms it worked.
```

### 平台特定技能

技能可以使用 `platforms` 字段限制自己到特定操作系统：

```yaml
platforms: [macos]            # macOS only (e.g., iMessage, Apple Reminders)
platforms: [macos, linux]     # macOS and Linux
platforms: [windows]          # Windows only
```

设置后，技能在 incompatible 平台上自动从系统提示词、`skills_list()` 和斜杠命令中隐藏。如果省略或为空，技能在所有平台上加载（向后兼容）。

### 条件技能激活

技能可以声明对特定工具或工具集的依赖。这控制技能是否出现在给定会话的系统提示词中。

```yaml
metadata:
  hermes:
    requires_toolsets: [web]           # Hide if the web toolset is NOT available
    requires_tools: [web_search]       # Hide if web_search tool is NOT available
    fallback_for_toolsets: [browser]   # Hide if the browser toolset IS active
    fallback_for_tools: [browser_navigate]  # Hide if browser_navigate IS available
```

| 字段 | 行为 |
|-------|----------|
| `requires_toolsets` | 当列表中任何工具集**不可用**时，技能**隐藏** |
| `requires_tools` | 当列表中任何工具**不可用**时，技能**隐藏** |
| `fallback_for_toolsets` | 当列表中任何工具集**可用**时，技能**隐藏** |
| `fallback_for_tools` | 当列表中任何工具**可用**时，技能**隐藏** |

**`fallback_for_*` 的用例：** 创建一个在主要工具不可用时作为备选服务的技能。例如，带有 `fallback_for_tools: [web_search]` 的 `duckduckgo-search` 技能仅在 web 搜索工具（需要 API 密钥）未配置时显示。

**`requires_*` 的用例：** 创建仅在某些工具存在时有意义的技能。例如，带有 `requires_toolsets: [web]` 的网页抓取工作流技能在 web 工具被禁用时不会弄乱提示词。

### 环境变量要求

技能可以声明它们需要的环境变量。当通过 `skill_view` 加载技能时，其必需的变量自动注册以透传到沙盒执行环境（terminal、execute_code）。

```yaml
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: "Tenor API key"               # Shown when prompting user
    help: "Get your key at https://tenor.com"  # Help text or URL
    required_for: "GIF search functionality"   # What needs this var
```

每个条目支持：
- `name`（必需）— 环境变量名
- `prompt`（可选）— 询问用户值时的提示文本
- `help`（可选）— 获取值的帮助文本或 URL
- `required_for`（可选）— 描述哪个功能需要此变量

用户也可以在 `config.yaml` 中手动配置透传变量：

```yaml
terminal:
  env_passthrough:
    - MY_CUSTOM_VAR
    - ANOTHER_VAR
```

参见 `skills/apple/` 获取 macOS 专用技能的示例。

## 加载时的安全设置

当技能需要 API 密钥或 token 时使用 `required_environment_variables`。缺失值**不会**隐藏技能发现。相反，Hermes 在本地 CLI 中加载技能时安全地提示它们。

```yaml
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: Tenor API key
    help: Get a key from https://developers.google.com/tenor
    required_for: full functionality
```

用户可以跳过设置并继续加载技能。Hermes 永远不向模型暴露原始 secret 值。网关和消息会话显示本地设置指导而不是带内收集 secrets。

:::tip 沙盒透传
当技能加载时，任何已设置的声明 `required_environment_variables` 自动透传到 `execute_code` 和 `terminal` 沙盒 — 包括 Docker 和 Modal 等远程后端。您的技能的脚本可以访问 `$TENOR_API_KEY`（或在 Python 中 `os.environ["TENOR_API_KEY"]`），而无需用户配置任何额外内容。详情参见[环境变量透传](/docs/user-guide/security#environment-variable-passthrough)。
:::

旧版 `prerequisites.env_vars` 仍作为向后兼容别名受支持。

### 配置设置（config.yaml）

技能可以声明存储在 `config.yaml` 中 `skills.config` 命名空间下的非 secret 设置。与环境变量（存储在 `.env` 中的 secrets）不同，config 设置用于路径、偏好设置和其他非敏感值。

```yaml
metadata:
  hermes:
    config:
      - key: myplugin.path
        description: Path to the plugin data directory
        default: "~/myplugin-data"
        prompt: Plugin data directory path
      - key: myplugin.domain
        description: Domain the plugin operates on
        default: ""
        prompt: Plugin domain (e.g., AI/ML research)
```

每个条目支持：
- `key`（必需）— 设置的点路径（例如 `myplugin.path`）
- `description`（必需）— 解释设置控制什么
- `default`（可选）— 如果用户未配置则默认值
- `prompt`（可选）— 在 `hermes config migrate` 期间显示的提示文本；回退到 `description`

**工作原理：**

1. **存储：** 值写入 `config.yaml` 下的 `skills.config.<key>`：
   ```yaml
   skills:
     config:
       myplugin:
         path: ~/my-data
   ```

2. **发现：** `hermes config migrate` 扫描所有启用的技能，找到未配置设置并提示用户。设置也出现在 `hermes config show` 下的"技能设置"中。

3. **运行时注入：** 当技能加载时，其配置值被解析并追加到技能消息：
   ```
   [Skill config (from ~/.hermes/config.yaml):
     myplugin.path = /home/user/my-data
   ]
   ```
   代理看到配置值而无需自己读取 `config.yaml`。

4. **手动设置：** 用户也可以直接设置值：
   ```bash
   hermes config set skills.config.myplugin.path ~/my-data
   ```

:::tip 何时使用什么
对 API 密钥、token 和其他 **secrets**（存储在 `~/.hermes/.env`，永不向模型显示）使用 `required_environment_variables`。对**路径、偏好设置和非敏感设置**（存储在 `config.yaml`，在配置显示中可见）使用 `config`。
:::

### 凭据文件要求（OAuth token 等）

使用 OAuth 或基于文件的凭据的技能可以声明需要挂载到远程沙盒的文件。这用于存储为**文件**的凭据（不是 env 变量）— 通常是设置脚本生成的 OAuth token 文件。

```yaml
required_credential_files:
  - path: google_token.json
    description: Google OAuth2 token (created by setup script)
  - path: google_client_secret.json
    description: Google OAuth2 client credentials
```

每个条目支持：
- `path`（必需）— 相对于 `~/.hermes/` 的文件路径
- `description`（可选）— 解释文件是什么以及如何创建

加载时，Hermes 检查这些文件是否存在。缺失文件触发 `setup_needed`。现有文件自动：
- 作为只读绑定挂载到 **Docker** 容器
- 同步到 **Modal** 沙盒（在创建时 + 每个命令之前，这样 mid-session OAuth 有效）
- 在**本地**后端上无需任何特殊处理即可使用

:::tip 何时使用什么
对简单的 API 密钥和 token（存储在 `~/.hermes/.env` 中的字符串）使用 `required_environment_variables`。对 OAuth token 文件、client secrets、服务账户 JSON、证书或磁盘上作为文件的任何凭据使用 `required_credential_files`。
:::

参见 `skills/productivity/google-workspace/SKILL.md` 获取同时使用两者的完整示例。

## 技能指南

### 无外部依赖

优先使用 stdlib Python、curl 和现有 Hermes 工具（`web_extract`、`terminal`、`read_file`）。如果需要依赖，在技能中记录安装步骤。

### 渐进式披露

首先放置最常见的工作流。边缘情况和高级用法放在底部。这在常见任务中保持 token 使用量低。

### 包含辅助脚本

对于 XML/JSON 解析或复杂逻辑，在 `scripts/` 中包含辅助脚本 — 不要期望 LLM 每次内联编写解析器。

#### 从 SKILL.md 引用捆绑脚本

当技能加载时，激活消息将绝对技能目录暴露为 `[Skill directory: /abs/path]`，还可以在 SKILL.md 正文中任意位置替换两个模板 token：

| Token | 替换为 |
|---|---|
| `${HERMES_SKILL_DIR}` | 技能目录的绝对路径 |
| `${HERMES_SESSION_ID}` | 活动会话 id（如果没有会话则保留原位）|

因此 SKILL.md 可以告诉代理直接运行捆绑脚本：

```markdown
To analyse the input, run:

    node ${HERMES_SKILL_DIR}/scripts/analyse.js <input>
```

代理看到替换后的绝对路径并使用准备运行的命令调用 `terminal` 工具 — 无需路径计算，无需额外的 `skill_view` 往返。使用 `skills.template_vars: false` 在 `config.yaml` 中全局禁用替换。

#### 内联 shell 片段（选择加入）

技能也可以嵌入内联 shell 片段，写为 SKILL.md 正文中的 `` !`cmd` ``。启用后，每个片段的 stdout 在代理读取之前内联到消息中，因此技能可以注入动态上下文：

```markdown
Current date: !`date -u +%Y-%m-%d`
Git branch: !`git -C ${HERMES_SKILL_DIR} rev-parse --abbrev-ref HEAD`
```

这**默认关闭** — SKILL.md 中的任何片段在主机上运行无需审批，因此仅对您信任的技能源启用：

```yaml
# config.yaml
skills:
  inline_shell: true
  inline_shell_timeout: 10   # seconds per snippet
```

片段使用技能目录作为工作目录运行，输出上限为 4000 字符。失败（超时、非零退出）显示为短 `[inline-shell error: ...]` 标记，而不是破坏整个技能。

### 测试

运行技能并验证代理正确遵循指令：

```bash
hermes chat --toolsets skills -q "Use the X skill to do Y"
```

## 技能应该放在哪里？

捆绑技能（在 `skills/` 中）随每个 Hermes 安装发货。它们应该**对大多数用户广泛有用**：

- 文档处理、网络研究、常见开发工作流、系统管理
- 被广泛范围的人定期使用

如果您的技能是官方的且有用但不是普遍需要的（例如，付费服务集成、重量级依赖），放在 **`optional-skills/`** 中 — 它随仓库发货，通过 `hermes skills browse` 发现（标记为"official"），并以 builtin trust 安装。

如果您的技能是专业的、社区贡献的或小众的，它更适合**技能中心** — 上传到注册表并通过 `hermes skills install` 分享。

## 发布技能

### 到技能中心

```bash
hermes skills publish skills/my-skill --to github --repo owner/repo
```

### 到自定义仓库

将您的仓库添加为 tap：

```bash
hermes skills tap add owner/repo
```

然后用户可以搜索并从您的仓库安装。

## 安全扫描

所有 hub 安装的技能都通过安全扫描器检查：

- 数据泄露模式
- 提示词注入尝试
- 破坏性命令
- Shell 注入

信任级别：
- `builtin` — 随 Hermes 发货（始终信任）
- `official` — 来自仓库中的 `optional-skills/`（builtin trust，无第三方警告）
- `trusted` — 来自 openai/skills、anthropics/skills
- `community` — 非危险发现可以用 `--force` 覆盖；`dangerous` 裁定保持阻止

Hermes 现在可以从多个外部发现模型消费第三方技能：
- 直接 GitHub 标识符（例如 `openai/skills/k8s`）
- `skills.sh` 标识符（例如 `skills-sh/vercel-labs/json-render/json-render-react`）
- 从 `/.well-known/skills/index.json` 服务的已知端点

如果您希望技能无需特定于 GitHub 的安装程序即可被发现，请考虑除了在仓库或市场中发布外，还从已知端点提供服务。

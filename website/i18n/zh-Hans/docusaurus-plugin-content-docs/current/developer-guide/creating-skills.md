---
sidebar_position: 3
title: "创建 Skill"
description: "如何为 Hermes Agent 创建 skill——SKILL.md 格式、规范与发布"
---

# 创建 Skill

Skill 是为 Hermes Agent 添加新能力的首选方式。与 tool 相比，skill 更易于创建，无需修改 agent 代码，且可与社区共享。

## 应该创建 Skill 还是 Tool？

以下情况创建 **Skill**：
- 该能力可通过指令 + shell 命令 + 现有 tool 来实现
- 封装了 agent 可通过 `terminal` 或 `web_extract` 调用的外部 CLI 或 API
- 不需要将自定义 Python 集成或 API key 管理内置到 agent 中
- 示例：arXiv 搜索、git 工作流、Docker 管理、PDF 处理、通过 CLI 工具发送邮件

以下情况创建 **Tool**：
- 需要与 API key、认证流程或多组件配置进行端到端集成
- 需要每次精确执行的自定义处理逻辑
- 处理二进制数据、流式传输或实时事件
- 示例：浏览器自动化、TTS、视觉分析

## Skill 目录结构

内置 skill 位于 `skills/` 目录下，按类别组织。官方可选 skill 在 `optional-skills/` 中使用相同结构：

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

### 平台专属 Skill

Skill 可通过 `platforms` 字段将自身限制在特定操作系统上：

```yaml
platforms: [macos]            # 仅 macOS（例如 iMessage、Apple Reminders）
platforms: [macos, linux]     # macOS 和 Linux
platforms: [windows]          # 仅 Windows
```

设置后，该 skill 会在不兼容的平台上自动从系统 prompt（提示词）、`skills_list()` 和斜杠命令中隐藏。若省略或留空，则在所有平台上加载（向后兼容）。

### 条件式 Skill 激活

Skill 可声明对特定 tool 或 toolset 的依赖，以控制该 skill 是否出现在当前会话的系统 prompt 中。

```yaml
metadata:
  hermes:
    requires_toolsets: [web]           # 若 web toolset 未激活则隐藏
    requires_tools: [web_search]       # 若 web_search tool 不可用则隐藏
    fallback_for_toolsets: [browser]   # 若 browser toolset 已激活则隐藏
    fallback_for_tools: [browser_navigate]  # 若 browser_navigate 可用则隐藏
```

| 字段 | 行为 |
|-------|----------|
| `requires_toolsets` | 当列出的**任意** toolset **不**可用时，skill **隐藏** |
| `requires_tools` | 当列出的**任意** tool **不**可用时，skill **隐藏** |
| `fallback_for_toolsets` | 当列出的**任意** toolset **已**可用时，skill **隐藏** |
| `fallback_for_tools` | 当列出的**任意** tool **已**可用时，skill **隐藏** |

**`fallback_for_*` 使用场景：** 创建一个在主要 tool 不可用时作为替代方案的 skill。例如，带有 `fallback_for_tools: [web_search]` 的 `duckduckgo-search` skill 仅在未配置需要 API key 的 web search tool 时显示。

**`requires_*` 使用场景：** 创建仅在特定 tool 存在时才有意义的 skill。例如，带有 `requires_toolsets: [web]` 的网页抓取工作流 skill 在 web tool 被禁用时不会出现在 prompt 中。

### 环境变量要求

Skill 可声明所需的环境变量。当通过 `skill_view` 加载 skill 时，其所需变量会自动注册，以便透传（passthrough）到沙箱执行环境（terminal、execute_code）中。

```yaml
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: "Tenor API key"               # 提示用户时显示
    help: "Get your key at https://tenor.com"  # 帮助文本或 URL
    required_for: "GIF search functionality"   # 哪个功能需要此变量
```

每个条目支持：
- `name`（必需）——环境变量名称
- `prompt`（可选）——向用户询问值时的提示文本
- `help`（可选）——获取该值的帮助文本或 URL
- `required_for`（可选）——描述哪个功能需要此变量

用户也可在 `config.yaml` 中手动配置透传变量：

```yaml
terminal:
  env_passthrough:
    - MY_CUSTOM_VAR
    - ANOTHER_VAR
```

macOS 专属 skill 示例请参见 `skills/apple/`。

## 加载时的安全配置

当 skill 需要 API key 或 token 时，使用 `required_environment_variables`。缺少值**不会**将 skill 从发现列表中隐藏。Hermes 会在本地 CLI 加载 skill 时安全地提示用户输入。

```yaml
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: Tenor API key
    help: Get a key from https://developers.google.com/tenor
    required_for: full functionality
```

用户可以跳过配置并继续加载 skill。Hermes 不会将原始密钥值暴露给模型。Gateway 和消息会话会显示本地配置指引，而不是在带内收集密钥。

:::tip 沙箱透传
加载 skill 时，已设置的 `required_environment_variables` 会**自动透传**到 `execute_code` 和 `terminal` 沙箱——包括 Docker 和 Modal 等远程后端。Skill 的脚本无需用户额外配置即可访问 `$TENOR_API_KEY`（或 Python 中的 `os.environ["TENOR_API_KEY"]`）。详见 [环境变量透传](/user-guide/security#environment-variable-passthrough)。
:::

旧版 `prerequisites.env_vars` 作为向后兼容的别名仍受支持。

### Config 配置项（config.yaml）

Skill 可声明非密钥配置项，这些配置项存储在 `config.yaml` 的 `skills.config` 命名空间下。与环境变量（存储密钥）不同，config 配置项用于路径、偏好设置及其他非敏感值。

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
- `key`（必需）——配置项的点路径（例如 `myplugin.path`）
- `description`（必需）——说明该配置项的作用
- `default`（可选）——用户未配置时的默认值
- `prompt`（可选）——`hermes config migrate` 时显示的提示文本；若未设置则回退到 `description`

**工作原理：**

1. **存储：** 值写入 `config.yaml` 的 `skills.config.<key>` 下：
   ```yaml
   skills:
     config:
       myplugin:
         path: ~/my-data
   ```

2. **发现：** `hermes config migrate` 扫描所有已启用的 skill，找出未配置的项并提示用户。配置项也会在 `hermes config show` 的"Skill Settings"部分显示。

3. **运行时注入：** Skill 加载时，其 config 值会被解析并追加到 skill 消息中：
   ```
   [Skill config (from ~/.hermes/config.yaml):
     myplugin.path = /home/user/my-data
   ]
   ```
   Agent 无需自行读取 `config.yaml` 即可看到已配置的值。

4. **手动配置：** 用户也可直接设置值：
   ```bash
   hermes config set skills.config.myplugin.path ~/my-data
   ```

:::tip 如何选择
对 API key、token 及其他**密钥**使用 `required_environment_variables`（存储在 `~/.hermes/.env`，不向模型展示）。对**路径、偏好设置及非敏感配置**使用 `config`（存储在 `config.yaml`，在 config show 中可见）。
:::

### 凭证文件要求（OAuth token 等）

使用 OAuth 或基于文件的凭证的 skill 可声明需要挂载到远程沙箱的文件。这适用于以**文件**形式存储的凭证（而非环境变量）——通常是由配置脚本生成的 OAuth token 文件。

```yaml
required_credential_files:
  - path: google_token.json
    description: Google OAuth2 token (created by setup script)
  - path: google_client_secret.json
    description: Google OAuth2 client credentials
```

每个条目支持：
- `path`（必需）——相对于 `~/.hermes/` 的文件路径
- `description`（可选）——说明该文件的用途及创建方式

加载时，Hermes 会检查这些文件是否存在。缺少文件会触发 `setup_needed`。已存在的文件会自动：
- **挂载到 Docker** 容器中作为只读绑定挂载
- **同步到 Modal** 沙箱（在创建时及每次命令前同步，因此会话中途的 OAuth 也能正常工作）
- 在**本地**后端无需任何特殊处理即可使用

:::tip 如何选择
对简单的 API key 和 token（存储在 `~/.hermes/.env` 中的字符串）使用 `required_environment_variables`。对 OAuth token 文件、客户端密钥、服务账号 JSON、证书或任何以磁盘文件形式存在的凭证使用 `required_credential_files`。
:::

完整示例请参见 `skills/productivity/google-workspace/SKILL.md`，其中同时使用了两者。

## Skill 规范

### 无外部依赖

优先使用标准库 Python、curl 以及现有 Hermes tool（`web_extract`、`terminal`、`read_file`）。若确实需要依赖项，请在 skill 中记录安装步骤。

### 渐进式披露

将最常见的工作流放在最前面。边缘情况和高级用法放在底部。这样可以降低常见任务的 token 消耗。

### 包含辅助脚本

对于 XML/JSON 解析或复杂逻辑，请在 `scripts/` 中包含辅助脚本——不要每次都期望 LLM 内联编写解析器。

### 以文档形式传递媒体（`[[as_document]]`）

如果 skill 生成高分辨率截图、图表或任何有损预览压缩会造成损失的图片，请在响应中某处（通常是最后一行）输出字面指令 `[[as_document]]`。Gateway 会去除该指令，并将该响应中所有提取的媒体路径以可下载文件附件的形式传递，而非内联图片气泡。完整语义请参见 [Skill 输出与媒体传递](../user-guide/features/skills.md#skill-output-and-media-delivery)。

#### 在 SKILL.md 中引用内置脚本

Skill 加载时，激活消息会将 skill 目录的绝对路径以 `[Skill directory: /abs/path]` 的形式暴露，同时在 SKILL.md 正文中替换两个模板 token：

| Token | 替换为 |
|---|---|
| `${HERMES_SKILL_DIR}` | skill 目录的绝对路径 |
| `${HERMES_SESSION_ID}` | 当前会话 ID（若无会话则保留原样） |

因此，SKILL.md 可以直接告知 agent 运行内置脚本：

```markdown
To analyse the input, run:

    node ${HERMES_SKILL_DIR}/scripts/analyse.js <input>
```

Agent 看到替换后的绝对路径，并使用 `terminal` tool 执行已就绪的命令——无需路径计算，无需额外的 `skill_view` 往返。可在 `config.yaml` 中设置 `skills.template_vars: false` 全局禁用替换。

#### 内联 shell 片段（需手动开启）

Skill 也可在 SKILL.md 正文中嵌入以 `` !`cmd` `` 形式编写的内联 shell 片段。启用后，每个片段的 stdout 会在 agent 读取前内联到消息中，从而让 skill 注入动态上下文：

```markdown
Current date: !`date -u +%Y-%m-%d`
Git branch: !`git -C ${HERMES_SKILL_DIR} rev-parse --abbrev-ref HEAD`
```

此功能**默认关闭**——SKILL.md 中的任何片段都会在未经审批的情况下在宿主机上运行，因此仅对你信任的 skill 来源启用：

```yaml
# config.yaml
skills:
  inline_shell: true
  inline_shell_timeout: 10   # 每个片段的超时秒数
```

片段以 skill 目录为工作目录运行，输出上限为 4000 个字符。失败（超时、非零退出）会显示为简短的 `[inline-shell error: ...]` 标记，而不会导致整个 skill 中断。

### 测试

运行 skill 并验证 agent 是否正确遵循指令：

```bash
hermes chat --toolsets skills -q "Use the X skill to do Y"
```

## Skill 应放在哪里？

内置 skill（位于 `skills/`）随每次 Hermes 安装一起发布，应对**大多数用户广泛有用**：

- 文档处理、网页研究、常见开发工作流、系统管理
- 被广泛人群定期使用

如果你的 skill 是官方的且有用，但并非所有人都需要（例如付费服务集成、重量级依赖），请放入 **`optional-skills/`**——它随仓库一起发布，可通过 `hermes skills browse` 发现（标记为"official"），并以内置信任级别安装。

如果你的 skill 是专业化的、社区贡献的或小众的，更适合放在 **Skills Hub**——将其上传到注册表并通过 `hermes skills install` 分享。

## 发布 Skill

### 发布到 Skills Hub

```bash
hermes skills publish skills/my-skill --to github --repo owner/repo
```

### 发布到自定义仓库

将你的仓库添加为 tap：

```bash
hermes skills tap add owner/repo
```

用户随后可从你的仓库搜索并安装。

## 安全扫描

所有从 hub 安装的 skill 都会经过安全扫描器检查：

- 数据泄露模式
- Prompt 注入尝试
- 破坏性命令
- Shell 注入

信任级别：
- `builtin`——随 Hermes 一起发布（始终受信任）
- `official`——来自仓库中的 `optional-skills/`（内置信任，无第三方警告）
- `trusted`——来自 openai/skills、anthropics/skills、huggingface/skills
- `community`——非危险发现可通过 `--force` 覆盖；`dangerous` 判定仍会被阻止

Hermes 现在可以通过多种外部发现模型使用第三方 skill：
- 直接 GitHub 标识符（例如 `openai/skills/k8s`）
- `skills.sh` 标识符（例如 `skills-sh/vercel-labs/json-render/json-render-react`）
- 从 `/.well-known/skills/index.json` 提供的知名端点

如果你希望 skill 无需 GitHub 专属安装器即可被发现，除了在仓库或市场中发布外，还可以考虑通过知名端点提供服务。
---
sidebar_position: 2
title: "Skills 系统"
description: "按需加载的知识文档——渐进式披露、agent 管理的 skills 以及 Skills Hub"
---

# Skills 系统

Skills 是 agent 在需要时可以加载的按需知识文档。它们遵循**渐进式披露**（progressive disclosure）模式以最小化 token 用量，并兼容 [agentskills.io](https://agentskills.io/specification) 开放标准。

所有 skills 存放在 **`~/.hermes/skills/`** 中——这是主目录和唯一可信来源。全新安装时，捆绑的 skills 会从仓库复制过来。通过 Hub 安装和 agent 创建的 skills 也存放在此处。agent 可以修改或删除任何 skill。

你也可以让 Hermes 指向**外部 skill 目录**——与本地目录一起扫描的额外文件夹。参见下方的[外部 Skill 目录](#external-skill-directories)。

另请参阅：

- [捆绑 Skills 目录](/reference/skills-catalog)
- [官方可选 Skills 目录](/reference/optional-skills-catalog)

## 使用 Skills

每个已安装的 skill 都会自动作为斜杠命令可用：

```bash
# 在 CLI 或任何消息平台中：
/gif-search funny cats
/axolotl help me fine-tune Llama 3 on my dataset
/github-pr-workflow create a PR for the auth refactor
/plan design a rollout for migrating our auth provider

# 只输入 skill 名称即可加载它，并让 agent 询问你的需求：
/excalidraw
```

捆绑的 `plan` skill 是一个很好的示例。运行 `/plan [request]` 会加载该 skill 的指令，告知 Hermes 在需要时检查上下文、编写 markdown 实现计划而非直接执行任务，并将结果保存在相对于当前工作区/后端工作目录的 `.hermes/plans/` 下。

你也可以通过自然对话与 skills 交互：

```bash
hermes chat --toolsets skills -q "What skills do you have?"
hermes chat --toolsets skills -q "Show me the axolotl skill"
```

## 渐进式披露

Skills 使用一种节省 token 的加载模式：

```
Level 0: skills_list()           → [{name, description, category}, ...]   (~3k tokens)
Level 1: skill_view(name)        → Full content + metadata       (varies)
Level 2: skill_view(name, path)  → Specific reference file       (varies)
```

agent 只在真正需要时才加载完整的 skill 内容。

## SKILL.md 格式

```markdown
---
name: my-skill
description: Brief description of what this skill does
version: 1.0.0
platforms: [macos, linux]     # Optional — restrict to specific OS platforms
metadata:
  hermes:
    tags: [python, automation]
    category: devops
    fallback_for_toolsets: [web]    # Optional — conditional activation (see below)
    requires_toolsets: [terminal]   # Optional — conditional activation (see below)
    config:                          # Optional — config.yaml settings
      - key: my.setting
        description: "What this controls"
        default: "value"
        prompt: "Prompt for setup"
---

# Skill Title

## When to Use
Trigger conditions for this skill.

## Procedure
1. Step one
2. Step two

## Pitfalls
- Known failure modes and fixes

## Verification
How to confirm it worked.
```

### 平台特定 Skills

Skills 可以使用 `platforms` 字段将自身限制在特定操作系统上：

| 值 | 匹配 |
|-------|---------|
| `macos` | macOS（Darwin） |
| `linux` | Linux |
| `windows` | Windows |

```yaml
platforms: [macos]            # macOS only (e.g., iMessage, Apple Reminders, FindMy)
platforms: [macos, linux]     # macOS and Linux
```

设置后，该 skill 会在不兼容的平台上自动从系统提示词、`skills_list()` 和斜杠命令中隐藏。若省略，则在所有平台上加载。

## Skill 输出与媒体传递

当 skill 响应（或任何 agent 响应）包含指向媒体文件的裸绝对路径时——例如 `/home/user/screenshots/diagram.png`——gateway 会自动检测到它，将其从可见文本中剥离，并以原生方式将文件传递给用户的聊天界面（Telegram 图片、Discord 附件等），而不是在消息中留下原始路径。

对于音频，`[[audio_as_voice]]` 指令会将音频文件提升为在支持该功能的平台（Telegram、WhatsApp）上的原生语音消息气泡。

### 强制文档式传递：`[[as_document]]`

有时你需要与内联预览**相反**的效果：你希望文件作为可下载附件传递，而不是经过重新压缩的图片气泡。典型示例是高分辨率截图或图表——Telegram 的 `sendPhoto` 会将其重新压缩至约 200 KB、1280 px，严重影响可读性。通过 `sendDocument` 发送的 1-2 MB PNG 则保留原始字节完整无损。

如果响应（或其中任何文本——通常是最后一行）包含字面指令 `[[as_document]]`，则从该响应中提取的每个媒体路径都会作为文档/文件附件传递，而不是图片气泡：

```
Here is your rendered chart:

/home/user/.hermes/cache/chart-q4-2025.png

[[as_document]]
```

该指令在传递前会被剥离，用户不会看到它。粒度有意设计为每个响应全有或全无：发出一次 `[[as_document]]`，同一响应中的每个图片路径都会作为文档传递。这与 `[[audio_as_voice]]` 的作用范围一致。

在以下情况下从 skill 中使用它：

- 你生成了用户需要作为文件的截图或图表（用于在其他工具中编辑、存档、完整分享）。
- 默认的有损预览会遮蔽细节（小字体、像素精确的图表、对颜色敏感的渲染）。

没有单独文档路径的平台（如 SMS）会回退到其支持的任何附件机制。

### 条件激活（Fallback Skills）

Skills 可以根据当前会话中可用的工具自动显示或隐藏自身。这对于**fallback skills**（回退 skills）最为有用——仅在高级工具不可用时才应出现的免费或本地替代方案。

```yaml
metadata:
  hermes:
    fallback_for_toolsets: [web]      # Show ONLY when these toolsets are unavailable
    requires_toolsets: [terminal]     # Show ONLY when these toolsets are available
    fallback_for_tools: [web_search]  # Show ONLY when these specific tools are unavailable
    requires_tools: [terminal]        # Show ONLY when these specific tools are available
```

| 字段 | 行为 |
|-------|----------|
| `fallback_for_toolsets` | 当列出的 toolsets 可用时，skill **隐藏**。不可用时显示。 |
| `fallback_for_tools` | 同上，但检查单个工具而非 toolsets。 |
| `requires_toolsets` | 当列出的 toolsets 不可用时，skill **隐藏**。可用时显示。 |
| `requires_tools` | 同上，但检查单个工具。 |

**示例：** 内置的 `duckduckgo-search` skill 使用 `fallback_for_toolsets: [web]`。当你设置了 `FIRECRAWL_API_KEY` 时，web toolset 可用，agent 使用 `web_search`——DuckDuckGo skill 保持隐藏。如果 API key 缺失，web toolset 不可用，DuckDuckGo skill 会自动作为 fallback 出现。

没有任何条件字段的 skills 行为与之前完全相同——始终显示。

## 加载时的安全设置

Skills 可以声明所需的环境变量，而不会从发现列表中消失：

```yaml
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: Tenor API key
    help: Get a key from https://developers.google.com/tenor
    required_for: full functionality
```

当遇到缺失的值时，Hermes 仅在本地 CLI 中实际加载 skill 时才会安全地请求输入。你可以跳过设置并继续使用该 skill。消息平台不会在聊天中请求密钥——它们会告诉你改用本地的 `hermes setup` 或 `~/.hermes/.env`。

一旦设置，声明的环境变量会**自动传递**到 `execute_code` 和 `terminal` 沙箱——skill 的脚本可以直接使用 `$TENOR_API_KEY`。对于非 skill 的环境变量，使用 `terminal.env_passthrough` 配置选项。详情参见[环境变量传递](/user-guide/security#environment-variable-passthrough)。

### Skill 配置设置

Skills 还可以声明存储在 `config.yaml` 中的非密钥配置设置（路径、偏好项）：

```yaml
metadata:
  hermes:
    config:
      - key: myplugin.path
        description: Path to the plugin data directory
        default: "~/myplugin-data"
        prompt: Plugin data directory path
```

设置存储在 config.yaml 的 `skills.config` 下。`hermes config migrate` 会提示配置未设置的项，`hermes config show` 会显示它们。当 skill 加载时，其解析后的配置值会注入到上下文中，agent 会自动知晓已配置的值。

详情参见 [Skill 设置](/user-guide/configuration#skill-settings) 和[创建 Skills——配置设置](/developer-guide/creating-skills#config-settings-configyaml)。

## Skill 目录结构

```text
~/.hermes/skills/                  # Single source of truth
├── mlops/                         # Category directory
│   ├── axolotl/
│   │   ├── SKILL.md               # Main instructions (required)
│   │   ├── references/            # Additional docs
│   │   ├── templates/             # Output formats
│   │   ├── scripts/               # Helper scripts callable from the skill
│   │   ├── examples/              # Referenced example outputs
│   │   └── assets/                # Supplementary files
│   └── vllm/
│       └── SKILL.md
├── devops/
│   └── deploy-k8s/                # Agent-created skill
│       ├── SKILL.md
│       └── references/
├── .hub/                          # Skills Hub state
│   ├── lock.json
│   ├── quarantine/
│   └── audit.log
└── .bundled_manifest              # Tracks seeded bundled skills
```

通过第三方 URL 或 GitHub 安装时，Hermes 会安装 `SKILL.md`，以及其中明确引用且位于 `references/`、`templates/`、`scripts/`、`assets/` 和 `examples/` 下的文件。未引用的仓库文件不会被复制。Hermes 会扫描完整的隔离捆绑包，并在 `skills/.hub/lock.json` 中记录来源 URL、精确内容哈希、扫描器版本、发现项、时间戳，以及本次结果是新扫描还是缓存复用。

## 外部 Skill 目录

如果你在 Hermes 之外维护 skills——例如，供多个 AI 工具使用的共享 `~/.agents/skills/` 目录——你可以告诉 Hermes 也扫描这些目录。

在 `~/.hermes/config.yaml` 的 `skills` 部分下添加 `external_dirs`：

```yaml
skills:
  external_dirs:
    - ~/.agents/skills
    - /home/shared/team-skills
    - ${SKILLS_REPO}/skills
```

路径支持 `~` 展开和 `${VAR}` 环境变量替换。

### 工作原理

- **本地创建，就地更新**：新的 agent 创建的 skills 写入 `~/.hermes/skills/`。现有 skills 在找到的位置被修改，包括 `external_dirs` 下的 skills，当 agent 使用 `skill_manage` 操作（如 `patch`、`edit`、`write_file`、`remove_file` 或 `delete`）时。
- **外部目录不是写保护边界**：如果外部 skill 目录对 Hermes 进程可写，agent 管理的 skill 更新可以修改该目录中的文件。如果共享的外部 skills 必须保持只读，请使用文件系统权限或单独的 profile/toolset 设置。
- **本地优先**：如果同一 skill 名称同时存在于本地目录和外部目录中，本地版本优先。
- **完整集成**：外部 skills 出现在系统提示词索引、`skills_list`、`skill_view` 以及 `/skill-name` 斜杠命令中——与本地 skills 无异。
- **不存在的路径会被静默跳过**：如果配置的目录不存在，Hermes 会忽略它而不报错。适用于可能不在每台机器上都存在的可选共享目录。

### 示例

```text
~/.hermes/skills/               # Local (primary, read-write)
├── devops/deploy-k8s/
│   └── SKILL.md
└── mlops/axolotl/
    └── SKILL.md

~/.agents/skills/               # External (shared, mutable if writable)
├── my-custom-workflow/
│   └── SKILL.md
└── team-conventions/
    └── SKILL.md
```

所有四个 skills 都出现在你的 skill 索引中。如果你在本地创建一个名为 `my-custom-workflow` 的新 skill，它会遮蔽外部版本。

## Skill 捆绑包

Skill 捆绑包是将多个 skills 归组在单个斜杠命令下的小型 YAML 文件。当你运行 `/<bundle-name>` 时，捆绑包中列出的每个 skill 都会同时加载——当某个特定任务总是受益于同一组 skills 时非常有用。

### 快速示例

```bash
# 为后端功能开发创建一个捆绑包
hermes bundles create backend-dev \
  --skill github-code-review \
  --skill test-driven-development \
  --skill github-pr-workflow \
  -d "Backend feature work — review, test, PR workflow"
```

然后在 CLI 或任何 gateway 平台中：

```
/backend-dev refactor the auth middleware
```

agent 接收到所有三个 skills 加载到一条用户消息中，斜杠命令后的任何文本都作为用户指令附加。

### YAML 模式

捆绑包存放在 **`~/.hermes/skill-bundles/<slug>.yaml`** 中，格式如下：

```yaml
name: backend-dev
description: Backend feature work — review, test, PR workflow.
skills:
  - github-code-review
  - test-driven-development
  - github-pr-workflow
instruction: |
  Always start by writing failing tests, then implement.
  Open the PR through the standard workflow with co-author tags.
```

字段说明：
- `name`（可选——默认为文件名主干）——捆绑包的显示名称。规范化为连字符 slug 用于斜杠命令（`Backend Dev` → `/backend-dev`）。
- `description`（可选）——在 `/bundles` 和 `hermes bundles list` 中显示的简短文本。
- `skills`（必填，非空列表）——skill 名称或相对于你的 skills 目录的路径。使用与 `/<skill-name>` 相同的标识符。
- `instruction`（可选）——附加在加载的 skill 内容前的额外指导。适用于固化"我们总是这样一起使用这些 skills"的方式。

### 管理捆绑包

```bash
# 列出所有已安装的捆绑包
hermes bundles list

# 查看某个捆绑包
hermes bundles show backend-dev

# 交互式创建捆绑包（省略 --skill 标志以逐行输入）
hermes bundles create research

# 覆盖现有捆绑包
hermes bundles create backend-dev --skill ... --force

# 删除捆绑包
hermes bundles delete backend-dev

# 重新扫描 ~/.hermes/skill-bundles/ 并报告变更
hermes bundles reload
```

在聊天会话中，`/bundles` 会列出每个已安装的捆绑包及其 skills。

### 行为

- **当 slug 冲突时，捆绑包优先于单个 skills。** 如果你将捆绑包命名为 `research`，同时也有一个名为 `research` 的 skill，`/research` 会调用捆绑包。这是有意为之——你通过命名选择了捆绑包。
- **缺失的 skills 会被跳过，而不是致命错误。** 如果捆绑包列出了 `skill-foo` 但你未安装它，捆绑包仍会加载能解析的 skills，agent 会收到一条列出跳过内容的说明。
- **捆绑包在每个界面都有效**——交互式 CLI、TUI、仪表板聊天以及每个 gateway 平台（Telegram、Discord、Slack……）——因为调度与单个 skill 命令集中在同一位置。
- **捆绑包不会使 prompt 缓存失效。** 它们在调用时生成一条新的用户消息，与 `/<skill-name>` 的方式相同——不修改系统提示词。

### 捆绑包优于逐个手动安装 skill 的场景

在以下情况下使用捆绑包：
- 你总是为某个重复任务配对相同的 skills（`/backend-dev`、`/release-prep`、`/incident-response`）。
- 你想要比依次输入多个 `/skill` 调用更简洁的心智模型。
- 你想通过将捆绑包 YAML 提交到共享 dotfiles 仓库并符号链接到 `~/.hermes/skill-bundles/` 来发布团队范围的"任务配置文件"。

捆绑包只是一个 YAML 别名——它不会为你安装 skills。Skills 本身必须已经存在（在 `~/.hermes/skills/` 或外部 skill 目录中）。否则捆绑包调用只会跳过缺失的 skills。

## Agent 管理的 Skills（skill_manage 工具）

agent 可以通过 `skill_manage` 工具创建、更新和删除自己的 skills。这是 agent 的**程序性记忆**——当它找到一个非平凡的工作流时，它会将该方法保存为 skill 以供将来复用。

### Agent 创建 Skills 的时机

- 成功完成复杂任务后（5+ 次工具调用）
- 遇到错误或死路并找到可行路径时
- 用户纠正了其方法时
- 发现了非平凡的工作流时

### 操作

| 操作 | 用途 | 关键参数 |
|--------|---------|------------|
| `create` | 从头创建新 skill | `name`、`content`（完整 SKILL.md）、可选 `category` |
| `patch` | 针对性修复（首选） | `name`、`old_string`、`new_string` |
| `edit` | 重大结构性重写 | `name`、`content`（完整 SKILL.md 替换） |
| `delete` | 完全删除一个 skill | `name` |
| `write_file` | 添加/更新支持文件 | `name`、`file_path`、`file_content` |
| `remove_file` | 删除支持文件 | `name`、`file_path` |

:::tip
`patch` 操作是更新的首选方式——它比 `edit` 更节省 token，因为工具调用中只出现变更的文本。
:::

## Skills Hub

从在线注册表、`skills.sh`、直接的知名 skill 端点以及官方可选 skills 中浏览、搜索、安装和管理 skills。

### 常用命令

```bash
hermes skills browse                              # Browse all hub skills (official first)
hermes skills browse --source official            # Browse only official optional skills
hermes skills search kubernetes                   # Search all sources
hermes skills search react --source skills-sh     # Search the skills.sh directory
hermes skills search https://mintlify.com/docs --source well-known
hermes skills inspect openai/skills/k8s           # Preview before installing
hermes skills install openai/skills/k8s           # Install with security scan
hermes skills install official/security/1password
hermes skills install skills-sh/vercel-labs/json-render/json-render-react --force
hermes skills install well-known:https://mintlify.com/docs/.well-known/skills/mintlify
hermes skills install https://sharethis.chat/SKILL.md              # 直接 URL（含引用的支持文件）
hermes skills install https://example.com/SKILL.md --name my-skill # Override name when frontmatter has none
hermes skills list --source hub                   # List hub-installed skills
hermes skills check                               # Check installed hub skills for upstream updates
hermes skills update                              # Reinstall hub skills with upstream changes when needed
hermes skills audit                               # Re-scan all hub skills for security
hermes skills uninstall k8s                       # Remove a hub skill
hermes skills reset google-workspace              # Un-stick a bundled skill from "user-modified" (see below)
hermes skills reset google-workspace --restore    # Also restore the bundled version, deleting your local edits
hermes skills publish skills/my-skill --to github --repo owner/repo
hermes skills snapshot export setup.json          # Export skill config
hermes skills tap add myorg/skills-repo           # Add a custom GitHub source
```

### 支持的 hub 来源

| 来源 | 示例 | 说明 |
|--------|---------|-------|
| `official` | `official/security/1password` | Hermes 随附的可选 skills。 |
| `skills-sh` | `skills-sh/vercel-labs/agent-skills/vercel-react-best-practices` | 可通过 `hermes skills search <query> --source skills-sh` 搜索。当 skills.sh slug 与仓库文件夹不同时，Hermes 会解析别名式 skills。 |
| `well-known` | `well-known:https://mintlify.com/docs/.well-known/skills/mintlify` | 直接从网站的 `/.well-known/skills/index.json` 提供的 skills。使用站点或文档 URL 搜索。 |
| `url` | `https://sharethis.chat/SKILL.md` | 指向 `SKILL.md` 及其明确引用的支持文件的直接 HTTP(S) URL。名称解析顺序：frontmatter → URL slug → 交互式提示 → `--name` 标志。 |
| `github` | `openai/skills/k8s` | 直接从 GitHub 仓库/路径安装以及基于 GitHub 的自定义 tap。 |
| `clawhub`、`lobehub`、`browse-sh`、`claude-marketplace` | 来源特定标识符 | 社区或市场集成。 |

### 集成的 hub 和注册表

Hermes 目前与以下 skills 生态系统和发现来源集成：

#### 1. 官方可选 skills（`official`）

这些 skills 在 Hermes 仓库中维护，以内置信任级别安装。

- 目录：[官方可选 Skills 目录](../../reference/optional-skills-catalog)
- 仓库中的来源：`optional-skills/`
- 示例：

```bash
hermes skills browse --source official
hermes skills install official/security/1password
```

#### 2. skills.sh（`skills-sh`）

这是 Vercel 的公共 skills 目录。Hermes 可以直接搜索它、查看 skill 详情页、解析别名式 slug，并从底层源仓库安装。

- 目录：[skills.sh](https://skills.sh/)
- CLI/工具仓库：[vercel-labs/skills](https://github.com/vercel-labs/skills)
- Vercel 官方 skills 仓库：[vercel-labs/agent-skills](https://github.com/vercel-labs/agent-skills)
- 示例：

```bash
hermes skills search react --source skills-sh
hermes skills inspect skills-sh/vercel-labs/json-render/json-render-react
hermes skills install skills-sh/vercel-labs/json-render/json-render-react --force
```

#### 3. Well-known skill 端点（`well-known`）

这是基于 URL 的发现机制，来自发布 `/.well-known/skills/index.json` 的站点。它不是单一的集中式 hub——它是一种 Web 发现约定。

- 示例实时端点：[Mintlify docs skills index](https://mintlify.com/docs/.well-known/skills/index.json)
- 参考服务器实现：[vercel-labs/skills-handler](https://github.com/vercel-labs/skills-handler)
- 示例：

```bash
hermes skills search https://mintlify.com/docs --source well-known
hermes skills inspect well-known:https://mintlify.com/docs/.well-known/skills/mintlify
hermes skills install well-known:https://mintlify.com/docs/.well-known/skills/mintlify
```

#### 4. 直接 GitHub skills（`github`）

Hermes 可以直接从 GitHub 仓库和基于 GitHub 的 tap 安装。当你已知仓库/路径或想添加自己的自定义源仓库时非常有用。

默认 tap（无需任何设置即可浏览）：
- [openai/skills](https://github.com/openai/skills)
- [anthropics/skills](https://github.com/anthropics/skills)
- [huggingface/skills](https://github.com/huggingface/skills)
- [NVIDIA/skills](https://github.com/NVIDIA/skills) — NVIDIA 官方验证的技能（带签名 `skill.oms.sig` 与治理用 `skill-card.md`）
- [VoltAgent/awesome-agent-skills](https://github.com/VoltAgent/awesome-agent-skills)
- [garrytan/gstack](https://github.com/garrytan/gstack)

- 示例：

```bash
hermes skills install openai/skills/k8s
hermes skills tap add myorg/skills-repo
```

#### 5. ClawHub（`clawhub`）

作为社区来源集成的第三方 skills 市场。

- 站点：[clawhub.ai](https://clawhub.ai/)
- Hermes 来源 id：`clawhub`

#### 6. Claude 市场式仓库（`claude-marketplace`）

Hermes 支持发布 Claude 兼容插件/市场清单的市场仓库。

已知集成来源包括：
- [anthropics/skills](https://github.com/anthropics/skills)
- [aiskillstore/marketplace](https://github.com/aiskillstore/marketplace)

Hermes 来源 id：`claude-marketplace`

#### 7. LobeHub（`lobehub`）

Hermes 可以从 LobeHub 的公共目录中搜索并将 agent 条目转换为可安装的 Hermes skills。

- 站点：[LobeHub](https://lobehub.com/)
- 公共 agents 索引：[chat-agents.lobehub.com](https://chat-agents.lobehub.com/)
- 后端仓库：[lobehub/lobe-chat-agents](https://github.com/lobehub/lobe-chat-agents)
- Hermes 来源 id：`lobehub`

#### 8. browse.sh（`browse-sh`）

Hermes 与 [browse.sh](https://browse.sh) 集成，这是 Browserbase 的目录，包含 200+ 个针对特定站点的浏览器自动化 SKILL.md 文件（Airbnb、Amazon、arXiv、12306.cn、Etsy、Xero 等）。每个 skill 描述如何端到端驱动一个网站，适合与 Hermes 的浏览器工具以及你已安装的任何浏览器自动化 skills 配合使用。

- 站点：[browse.sh](https://browse.sh/)
- 目录 API：`https://browse.sh/api/skills`
- Hermes 来源 id：`browse-sh`
- 信任级别：`community`

```bash
hermes skills search airbnb --source browse-sh
hermes skills inspect browse-sh/airbnb.com/search-listings-ddgioa
hermes skills install browse-sh/airbnb.com/search-listings-ddgioa
```

标识符使用 `browse-sh/<hostname>/<task-id>` 的形式，与 browse.sh 目录公开的 slug 匹配。内容通过每个 skill 的详情端点（`/api/skills/<slug>` → `skillMdUrl`）解析，而不是通过目录的 GitHub `sourceUrl`。

#### 9. 直接 URL（`url`）

直接从任何 HTTP(S) URL 安装 `SKILL.md`——当作者在自己的站点上托管 skill 时非常有用（无 hub 列表，无需输入 GitHub 路径）。Hermes 还会获取其中明确引用且位于 `references/`、`templates/`、`scripts/`、`assets/` 和 `examples/` 下的文件，然后扫描并安装完整捆绑包。

- Hermes 来源 id：`url`
- 标识符：URL 本身（无需前缀）
- 范围：`SKILL.md` 加上允许目录中明确引用的支持文件。Hermes 不会枚举或复制托管站点上的其他文件。

```bash
hermes skills install https://sharethis.chat/SKILL.md
hermes skills install https://example.com/my-skill/SKILL.md --category productivity
```

名称解析顺序：
1. SKILL.md YAML frontmatter 中的 `name:` 字段（推荐——每个格式良好的 skill 都有）。
2. URL 路径中的父目录名称（例如 `.../my-skill/SKILL.md` → `my-skill`，或 `.../my-skill.md` → `my-skill`），当它是有效标识符（`^[a-z][a-z0-9_-]*$`）时。
3. 在有 TTY 的终端上的交互式提示。
4. 在非交互式界面（TUI 内的 `/skills install` 斜杠命令、gateway 平台、脚本）上，给出指向 `--name` 覆盖的清晰错误。

```bash
# Frontmatter 没有名称且 URL slug 无意义——手动提供：
hermes skills install https://example.com/SKILL.md --name sharethis-chat

# 或在聊天会话中：
/skills install https://example.com/SKILL.md --name sharethis-chat
```

信任级别始终为 `community`——与所有其他来源一样运行相同的安全扫描。URL 作为安装标识符存储，因此当你想刷新时，`hermes skills update` 会自动从同一 URL 重新获取。

### 安全扫描与 `--force`

所有通过 hub 安装的 skills 都经过**安全扫描器**检查，检测数据泄露、prompt 注入、破坏性命令、供应链信号及其他威胁。

`hermes skills inspect ...` 现在还会在可用时显示上游元数据：
- 仓库 URL
- skills.sh 详情页 URL
- 安装命令
- 每周安装量
- 上游安全审计状态
- well-known 索引/端点 URL

当你已审查第三方 skill 并希望覆盖非危险性策略阻止时，使用 `--force`：

```bash
hermes skills install skills-sh/anthropics/skills/pdf --force
```

重要行为：
- `--force` 可以覆盖谨慎/警告类发现的策略阻止。
- `--force` **不能**覆盖 `dangerous` 扫描结论。
- 官方可选 skills（`official/...`）被视为内置信任，不显示第三方警告面板。

### 信任级别

| 级别 | 来源 | 策略 |
|-------|--------|--------|
| `builtin` | 随 Hermes 附带 | 始终受信任 |
| `official` | 仓库中的 `optional-skills/` | 内置信任，无第三方警告 |
| `trusted` | 受信任的注册表/仓库，如 `openai/skills`、`anthropics/skills`、`huggingface/skills`、`NVIDIA/skills` | 比社区来源更宽松的策略 |
| `community` | 其他所有来源（`skills.sh`、well-known 端点、自定义 GitHub 仓库、大多数市场） | 非危险性发现可用 `--force` 覆盖；`dangerous` 结论保持阻止 |

### 更新生命周期

hub 现在跟踪足够的来源信息以重新检查已安装 skills 的上游副本：

```bash
hermes skills check          # Report which installed hub skills changed upstream
hermes skills update         # Reinstall only the skills with updates available
hermes skills update react   # Update one specific installed hub skill
```

这使用存储的来源标识符加上当前上游捆绑包内容哈希来检测漂移。

:::tip GitHub 速率限制
Skills hub 操作使用 GitHub API，未认证用户的速率限制为每小时 60 次请求。如果在安装或搜索时看到速率限制错误，请在 `.env` 文件中设置 `GITHUB_TOKEN` 以将限制提高到每小时 5,000 次请求。发生此情况时，错误消息会包含可操作的提示。
:::

### 发布自定义 skill tap

如果你想分享一组精选的 skills——为你的团队、组织或公开分享——你可以将它们发布为 **tap**：其他 Hermes 用户通过 `hermes skills tap add <owner/repo>` 添加的 GitHub 仓库。无需服务器，无需注册表注册，无需发布流水线。只需一个包含 `SKILL.md` 文件的目录。

#### 仓库布局

tap 是任何 GitHub 仓库（公开或私有——私有仓库需要 `GITHUB_TOKEN`），布局如下：

```
owner/repo
├── skills/                       # default path; configurable per-tap
│   ├── my-workflow/
│   │   ├── SKILL.md              # required
│   │   ├── references/           # optional supporting files
│   │   ├── templates/
│   │   └── scripts/
│   ├── another-skill/
│   │   └── SKILL.md
│   └── third-skill/
│       └── SKILL.md
└── README.md                     # optional but helpful
```

规则：
- 每个 skill 存放在 tap 根路径（默认 `skills/`）下的独立目录中。
- 目录名成为 skill 的安装 slug。
- 每个 skill 目录必须包含一个带有标准 [SKILL.md frontmatter](#skillmd-format) 的 `SKILL.md`（`name`、`description`，以及可选的 `metadata.hermes.tags`、`version`、`author`、`platforms`、`metadata.hermes.config`）。
- `references/`、`templates/`、`scripts/`、`assets/` 等子目录在安装时与 `SKILL.md` 一起下载。
- 目录名以 `.` 或 `_` 开头的 skills 会被忽略。

Hermes 通过列出 tap 路径的每个子目录并探测每个目录中的 `SKILL.md` 来发现 skills。

#### 最小 tap 示例

```
my-org/hermes-skills
└── skills/
    └── deploy-runbook/
        └── SKILL.md
```

`skills/deploy-runbook/SKILL.md`：

```markdown
---
name: deploy-runbook
description: Our deployment runbook — services, rollback, Slack channels
version: 1.0.0
author: My Org Platform Team
metadata:
  hermes:
    tags: [deployment, runbook, internal]
---

# Deploy Runbook

Step 1: ...
```

将其推送到 GitHub 后，任何 Hermes 用户都可以订阅并安装：

```bash
hermes skills tap add my-org/hermes-skills
hermes skills search deploy
hermes skills install my-org/hermes-skills/deploy-runbook
```

#### 非默认路径

如果你的 skills 不在 `skills/` 下（当你向现有项目添加 `skills/` 子树时很常见），请编辑 `~/.hermes/.hub/taps.json` 中的 tap 条目：

```json
{
  "taps": [
    {"repo": "my-org/platform-docs", "path": "internal/skills/"}
  ]
}
```

`hermes skills tap add` CLI 默认将新 tap 的 `path` 设为 `"skills/"`；如果需要不同路径，请直接编辑该文件。`hermes skills tap list` 显示每个 tap 的有效路径。

#### 直接安装单个 skills（无需添加 tap）

用户也可以从任何公开 GitHub 仓库安装单个 skill，而无需将整个仓库添加为 tap：

```bash
hermes skills install owner/repo/skills/my-workflow
```

当你想分享一个 skill 而不要求用户订阅你的整个注册表时非常有用。

#### tap 的信任级别

新 tap 默认分配 `community` 信任级别。从中安装的 skills 经过标准安全扫描，首次安装时显示第三方警告面板。如果你的组织或广泛受信任的来源应获得更高信任，请将其仓库添加到 `tools/skills_hub.py` 中的 `TRUSTED_REPOS`（需要 Hermes 核心 PR）。

#### Tap 管理

```bash
hermes skills tap list                                # show all configured taps
hermes skills tap add myorg/skills-repo               # add (default path: skills/)
hermes skills tap remove myorg/skills-repo            # remove
```

在运行中的会话内：

```
/skills tap list
/skills tap add myorg/skills-repo
/skills tap remove myorg/skills-repo
```

Tap 存储在 `~/.hermes/.hub/taps.json` 中（按需创建）。

## 捆绑 skill 更新（`hermes skills reset`）

Hermes 在仓库的 `skills/` 中附带一组捆绑 skills。在安装时以及每次 `hermes update` 时，同步过程会将这些 skills 复制到 `~/.hermes/skills/` 中，并在 `~/.hermes/skills/.bundled_manifest` 记录一个清单，将每个 skill 名称映射到同步时的内容哈希（**origin hash**）。

每次同步时，Hermes 重新计算本地副本的哈希并与 origin hash 比较：

- **未更改** → 可以安全拉取上游变更，复制新的捆绑版本，记录新的 origin hash。
- **已更改** → 视为**用户修改**并永久跳过，因此你的编辑不会被覆盖。

这种保护机制很好，但有一个棘手的边缘情况。如果你编辑了一个捆绑 skill，后来想通过从 `~/.hermes/hermes-agent/skills/` 复制粘贴来放弃更改并回到捆绑版本，清单仍然保存着上次成功同步时的*旧* origin hash。你新复制粘贴的内容（当前捆绑哈希）与那个过时的 origin hash 不匹配，因此同步继续将其标记为用户修改。

`hermes skills reset` 是解决此问题的方法：

```bash
# 安全：清除此 skill 的清单条目。你当前的副本被保留，
# 但下次同步会重新以其为基准，使未来的更新正常工作。
hermes skills reset google-workspace

# 完全恢复：同时删除你的本地副本并重新复制当前捆绑版本。
# 当你想要恢复原始上游 skill 时使用此选项。
hermes skills reset google-workspace --restore

# 非交互式（例如在脚本或 TUI 模式中）——跳过 --restore 确认。
hermes skills reset google-workspace --restore --yes
```

同样的命令也可以作为斜杠命令在聊天中使用：

```text
/skills reset google-workspace
/skills reset google-workspace --restore
```

:::note Profiles
每个 profile 在其自己的 `HERMES_HOME` 下有自己的 `.bundled_manifest`，因此 `hermes -p coder skills reset <name>` 只影响该 profile。
:::

### 斜杠命令（在聊天中）

所有相同的命令都可以使用 `/skills` 执行：

```text
/skills browse
/skills search react --source skills-sh
/skills search https://mintlify.com/docs --source well-known
/skills inspect skills-sh/vercel-labs/json-render/json-render-react
/skills install openai/skills/skill-creator --force
/skills check
/skills update
/skills reset google-workspace
/skills list
```

官方可选 skills 仍使用 `official/security/1password` 和 `official/migration/openclaw-migration` 等标识符。
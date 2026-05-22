---
sidebar_position: 2
title: "技能系统"
description: "按需加载的知识文档 —— 渐进式披露、智能体管理的技能和技能中心"
---

# 技能系统

技能是智能体按需加载的知识文档。它们遵循**渐进式披露**模式以最小化 token 使用量，并与 [agentskills.io](https://agentskills.io/specification) 开放标准兼容。

所有技能都位于 **`~/.hermes/skills/`** —— 主目录和事实来源。全新安装时，捆绑技能从仓库复制。中心安装和智能体创建的技能也放在这里。智能体可以修改或删除任何技能。

您还可以将 Hermes 指向**外部技能目录** —— 与本地目录一起扫描的额外文件夹。请参阅下面的 [外部技能目录](#skills-hub)。

另请参阅：

- [捆绑技能目录](/reference/skills-catalog)
- [官方可选技能目录](/reference/optional-skills-catalog)

## 使用技能

每个已安装的技能都自动作为斜杠命令可用：

```bash
# 在 CLI 或任何消息平台中：
/gif-search funny cats
/axolotl help me fine-tune Llama 3 on my dataset
/github-pr-workflow create a PR for the auth refactor
/plan design a rollout for migrating our auth provider

# 仅技能名称会加载它并让智能体询问您需要什么：
/excalidraw
```

捆绑的 `plan` 技能就是一个很好的例子。运行 `/plan [request]` 会加载技能的指令，告诉 Hermes 在需要时检查上下文，编写 markdown 实现计划而非执行任务，并将结果保存在活动工作区/后端工作目录下的 `.hermes/plans/` 中。

您还可以通过自然对话与技能交互：

```bash
hermes chat --toolsets skills -q "What skills do you have?"
hermes chat --toolsets skills -q "Show me the axolotl skill"
```

## 渐进式披露

技能使用 token 高效的加载模式：

```
Level 0: skills_list()           → [{name, description, category}, ...]   (~3k tokens)
Level 1: skill_view(name)        → Full content + metadata       (varies)
Level 2: skill_view(name, path)  → Specific reference file       (varies)
```

智能体仅在真正需要时才加载完整技能内容。

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

### 平台特定技能

技能可以使用 `platforms` 字段将自己限制为特定操作系统：

| 值 | 匹配 |
|-------|---------|
| `macos` | macOS (Darwin) |
| `linux` | Linux |
| `windows` | Windows |

```yaml
platforms: [macos]            # 仅限 macOS（例如，iMessage、Apple Reminders、FindMy）
platforms: [macos, linux]     # macOS 和 Linux
```

设置后，技能会自动从系统提示、`skills_list()` 和不兼容平台上的斜杠命令中隐藏。如果省略，技能在所有平台上加载。

### 条件激活（回退技能）

技能可以根据当前会话中可用的工具自动显示或隐藏自己。这对**回退技能**最有用 —— 当高级工具不可用时才出现的免费或本地替代方案。

```yaml
metadata:
  hermes:
    fallback_for_toolsets: [web]      # 仅当这些工具集不可用时显示
    requires_toolsets: [terminal]     # 仅当这些工具集可用时显示
    fallback_for_tools: [web_search]  # 仅当这些特定工具不可用时显示
    requires_tools: [terminal]        # 仅当这些特定工具可用时显示
```

| 字段 | 行为 |
|-------|----------|
| `fallback_for_toolsets` | 当列出的工具集可用时技能**隐藏**。当它们缺失时显示。 |
| `fallback_for_tools` | 相同，但检查单个工具而非工具集。 |
| `requires_toolsets` | 当列出的工具集不可用时技能**隐藏**。当它们存在时显示。 |
| `requires_tools` | 相同，但检查单个工具。 |

**示例：** 内置的 `duckduckgo-search` 技能使用 `fallback_for_toolsets: [web]`。当您设置了 `FIRECRAWL_API_KEY` 时，web 工具集可用，智能体使用 `web_search` —— DuckDuckGo 技能保持隐藏。如果 API 密钥缺失，web 工具集不可用，DuckDuckGo 技能自动作为回退出现。

没有任何条件字段的技能行为与之前完全相同 —— 它们始终显示。

## 加载时安全设置

技能可以声明所需的环境变量而不会从发现中消失：

```yaml
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: Tenor API key
    help: Get a key from https://developers.google.com/tenor
    required_for: full functionality
```

当遇到缺失值时，Hermes 仅在技能实际在本地 CLI 中加载时安全地询问它。您可以跳过设置并继续使用技能。消息界面从不在聊天中询问机密 —— 它们告诉您改为在本地使用 `hermes setup` 或 `~/.hermes/.env`。

设置后，声明的环境变量**自动传递**到 `execute_code` 和 `terminal` 沙盒 —— 技能的脚本可以直接使用 `$TENOR_API_KEY`。对于非技能环境变量，使用 `terminal.env_passthrough` 配置选项。有关详细信息，请参阅 [环境变量传递](/user-guide/security#environment-variable-passthrough)。

<a id="skill-settings"></a>

### 技能配置设置

技能还可以声明存储在 `config.yaml` 中的非机密配置设置（路径、偏好）：

```yaml
metadata:
  hermes:
    config:
      - key: myplugin.path
        description: Path to the plugin data directory
        default: "~/myplugin-data"
        prompt: Plugin data directory path
```

设置存储在您的 config.yaml 中的 `skills.config` 下。`hermes config migrate` 提示未配置的设置，`hermes config show` 显示它们。当技能加载时，其解析的配置值被注入上下文，以便智能体自动知道配置的值。

有关详细信息，请参阅 [技能设置](/user-guide/configuration#skill-settings) 和 [创建技能 —— 配置设置](/developer-guide/creating-skills#config-settings-configyaml)。

## 技能目录结构

```text
~/.hermes/skills/                  # 单一事实来源
├── mlops/                         # 类别目录
│   ├── axolotl/
│   │   ├── SKILL.md               # 主指令（必需）
│   │   ├── references/            # 附加文档
│   │   ├── templates/             # 输出格式
│   │   ├── scripts/               # 可从技能调用的辅助脚本
│   │   └── assets/                # 补充文件
│   └── vllm/
│       └── SKILL.md
├── devops/
│   └── deploy-k8s/                # 智能体创建的技能
│       ├── SKILL.md
│       └── references/
├── .hub/                          # 技能中心状态
│   ├── lock.json
│   ├── quarantine/
│   └── audit.log
└── .bundled_manifest              # 跟踪播种的捆绑技能
```

## 技能中心 / 外部技能目录 {#skills-hub}

如果您在 Hermes 之外维护技能 —— 例如，多个 AI 工具使用的共享 `~/.agents/skills/` 目录 —— 您可以告诉 Hermes 也扫描这些目录。

在 `~/.hermes/config.yaml` 中的 `skills` 部分下添加 `external_dirs`：

```yaml
skills:
  external_dirs:
    - ~/.agents/skills
    - /home/shared/team-skills
    - ${SKILLS_REPO}/skills
```

路径支持 `~` 扩展和 `${VAR}` 环境变量替换。

### 工作原理 {#publishing-a-custom-skill-tap}

- **只读**：外部目录仅用于技能发现。当智能体创建或编辑技能时，它始终写入 `~/.hermes/skills/`。
- **本地优先**：如果本地目录和外部目录中存在相同的技能名称，本地版本胜出。
- **完全集成**：外部技能出现在系统提示索引、`skills_list`、`skill_view` 中，并作为 `/skill-name` 斜杠命令 —— 与本地技能没有区别。
- **不存在的路径被静默跳过**：如果配置的目录不存在，Hermes 会忽略它而不会报错。对于可能并非存在于每台机器上的可选共享目录很有用。

### 示例

```text
~/.hermes/skills/               # 本地（主，读写）
├── devops/deploy-k8s/
│   └── SKILL.md
└── mlops/axolotl/
    └── SKILL.md

~/.agents/skills/               # 外部（只读，共享）
├── my-custom-workflow/
│   └── SKILL.md
└── team-conventions/
    └── SKILL.md
```

所有四个技能都出现在您的技能索引中。如果您在本地创建一个名为 `my-custom-workflow` 的新技能，它会覆盖外部版本。

## 智能体管理的技能（skill_manage 工具） {#agent-managed-skills-skill_manage-tool}

智能体可以通过 `skill_manage` 工具创建、更新和删除自己的技能。这是智能体的**过程记忆** —— 当它弄清楚一个非平凡的工作流时，它会将方法保存为技能以供将来重用。

### 智能体何时创建技能

- 成功完成复杂任务后（5+ 工具调用）
- 当它遇到错误或死胡同并找到工作路径时
- 当用户纠正其方法时
- 当它发现一个非平凡的工作流时

### 操作

| 操作 | 用于 | 关键参数 |
|--------|---------|------------|
| `create` | 从头开始创建新技能 | `name`, `content`（完整 SKILL.md），可选 `category` |
| `patch` | 定向修复（首选） | `name`, `old_string`, `new_string` |
| `edit` | 重大结构重写 | `name`, `content`（完整 SKILL.md 替换） |
| `delete` | 完全删除技能 | `name` |
| `write_file` | 添加/更新支持文件 | `name`, `file_path`, `file_content` |
| `remove_file` | 删除支持文件 | `name`, `file_path` |

:::tip
`patch` 操作是更新的首选 —— 它比 `edit` 更 token 高效，因为工具调用中只出现更改的文本。
:::

## 技能中心

浏览、搜索、安装和管理来自在线注册表、`skills.sh`、直接知名技能端点和官方可选技能的技能。

### 常用命令

```bash
hermes skills browse                              # 浏览所有中心技能（官方优先）
hermes skills browse --source official            # 仅浏览官方可选技能
hermes skills search kubernetes                   # 搜索所有来源
hermes skills search react --source skills-sh     # 搜索 skills.sh 目录
hermes skills search https://mintlify.com/docs --source well-known
hermes skills inspect openai/skills/k8s           # 安装前预览
hermes skills install openai/skills/k8s           # 带安全扫描安装
hermes skills install official/security/1password
hermes skills install skills-sh/vercel-labs/json-render/json-render-react --force
hermes skills install well-known:https://mintlify.com/docs/.well-known/skills/mintlify
hermes skills install https://sharethis.chat/SKILL.md              # 直接 URL（单文件 SKILL.md）
hermes skills install https://example.com/SKILL.md --name my-skill # 当 frontmatter 没有名称时覆盖名称
hermes skills list --source hub                   # 列出中心安装的技能
hermes skills check                               # 检查已安装的中心技能是否有上游更新
hermes skills update                              # 仅在需要时重新安装有更改的中心技能
hermes skills audit                               # 重新扫描所有中心技能的安全性
hermes skills uninstall k8s                       # 移除中心技能
hermes skills reset google-workspace              # 将捆绑技能从"用户修改"中解除（见下文）
hermes skills reset google-workspace --restore    # 同时恢复捆绑版本，删除您的本地编辑
hermes skills publish skills/my-skill --to github --repo owner/repo
hermes skills snapshot export setup.json          # 导出技能配置
hermes skills tap add myorg/skills-repo           # 添加自定义 GitHub 来源
```

### 支持的中心来源

| 来源 | 示例 | 说明 |
|--------|---------|-------|
| `official` | `official/security/1password` | 随 Hermes 一起提供的可选技能。 |
| `skills-sh` | `skills-sh/vercel-labs/agent-skills/vercel-react-best-practices` | 通过 `hermes skills search <query> --source skills-sh` 搜索。当 skills.sh slug 与仓库文件夹不同时，Hermes 解析别名样式技能。 |
| `well-known` | `well-known:https://mintlify.com/docs/.well-known/skills/mintlify` | 直接从网站上发布 `/.well-known/skills/index.json` 的技能。不是单一集中中心 —— 而是一种 Web 发现约定。 |
| `url` | `https://sharethis.chat/SKILL.md` | 直接 HTTP(S) URL 指向单文件 `SKILL.md`。名称解析：frontmatter → URL slug → 交互式提示 → `--name` 标志。 |
| `github` | `openai/skills/k8s` | 直接 GitHub 仓库/路径安装和自定义 taps。 |
| `clawhub`, `lobehub`, `claude-marketplace` | 来源特定标识符 | 社区或市场集成。 |

### 集成中心和注册表

Hermes 目前与以下技能生态系统和发现来源集成：

#### 1. 官方可选技能 (`official`)

这些在 Hermes 仓库本身中维护，并以内置信任安装。

- 目录：[官方可选技能目录](../../reference/optional-skills-catalog)
- 仓库中的来源：`optional-skills/`
- 示例：

```bash
hermes skills browse --source official
hermes skills install official/security/1password
```

#### 2. skills.sh (`skills-sh`)

这是 Vercel 的公开技能目录。Hermes 可以直接搜索它、检查技能详情页面、解析别名样式 slugs，并从底层来源仓库安装。

- 目录：[skills.sh](https://skills.sh/)
- CLI/工具仓库：[vercel-labs/skills](https://github.com/vercel-labs/skills)
- 官方 Vercel 技能仓库：[vercel-labs/agent-skills](https://github.com/vercel-labs/agent-skills)
- 示例：

```bash
hermes skills search react --source skills-sh
hermes skills inspect skills-sh/vercel-labs/json-render/json-render-react
hermes skills install skills-sh/vercel-labs/json-render/json-render-react --force
```

#### 3. 知名技能端点 (`well-known`)

这是从发布 `/.well-known/skills/index.json` 的网站的基于 URL 的发现。它不是单一集中中心 —— 而是一种 Web 发现约定。

- 示例实时端点：[Mintlify docs skills index](https://mintlify.com/docs/.well-known/skills/index.json)
- 参考服务器实现：[vercel-labs/skills-handler](https://github.com/vercel-labs/skills-handler)
- 示例：

```bash
hermes skills search https://mintlify.com/docs --source well-known
hermes skills inspect well-known:https://mintlify.com/docs/.well-known/skills/mintlify
hermes skills install well-known:https://mintlify.com/docs/.well-known/skills/mintlify
```

#### 4. 直接 GitHub 技能 (`github`)

Hermes 可以直接从 GitHub 仓库和基于 GitHub 的 taps 安装。当您已经知道仓库/路径或想要添加自己的自定义来源仓库时，这很有用。

默认 taps（无需任何设置即可浏览）：
- [openai/skills](https://github.com/openai/skills)
- [anthropics/skills](https://github.com/anthropics/skills)
- [VoltAgent/awesome-agent-skills](https://github.com/VoltAgent/awesome-agent-skills)
- [garrytan/gstack](https://github.com/garrytan/gstack)

- 示例：

```bash
hermes skills install openai/skills/k8s
hermes skills tap add myorg/skills-repo
```

#### 5. ClawHub (`clawhub`)

作为社区来源集成的第三方技能市场。

- 网站：[clawhub.ai](https://clawhub.ai/)
- Hermes 来源 id：`clawhub`

#### 6. Claude 市场风格仓库 (`claude-marketplace`)

Hermes 支持发布 Claude-compatible 插件/市场清单的市场仓库。

已知集成来源包括：
- [anthropics/skills](https://github.com/anthropics/skills)
- [aiskillstore/marketplace](https://github.com/aiskillstore/marketplace)

Hermes 来源 id：`claude-marketplace`

#### 7. LobeHub (`lobehub`)

Hermes 可以从 LobeHub 的公开目录中搜索和转换智能体条目为可安装的 Hermes 技能。

- 网站：[LobeHub](https://lobehub.com/)
- 公开智能体索引：[chat-agents.lobehub.com](https://chat-agents.lobehub.com/)
- 底层仓库：[lobehub/lobe-chat-agents](https://github.com/lobehub/lobe-chat-agents)
- Hermes 来源 id：`lobehub`

#### 8. 直接 URL (`url`)

直接从任何 HTTP(S) URL 安装单文件 `SKILL.md` —— 当作者将技能托管在自己的网站上时很有用（没有中心列表，没有 GitHub 路径可输入）。Hermes 获取 URL、解析 YAML frontmatter、进行安全扫描并安装。

- Hermes 来源 id：`url`
- 标识符：URL 本身（不需要前缀）
- 范围：**仅限单文件 `SKILL.md`**。带有 `references/` 或 `scripts/` 的多文件技能需要清单，应通过上述其他来源之一发布。

```bash
hermes skills install https://sharethis.chat/SKILL.md
hermes skills install https://example.com/my-skill/SKILL.md --category productivity
```

名称解析顺序：
1. SKILL.md YAML frontmatter 中的 `name:` 字段（推荐 —— 每个格式良好的技能都有一个）。
2. URL 路径中的父目录名称（例如 `.../my-skill/SKILL.md` → `my-skill`，或 `.../my-skill.md` → `my-skill`），当它是有效标识符时（`^[a-z][a-z0-9_-]*$`）。
3. 在具有 TTY 的终端上的交互式提示。
4. 在非交互式界面（TUI 内的 `/skills install` 斜杠命令、网关平台、脚本）上，一个指向 `--name` 覆盖的干净错误。

```bash
# Frontmatter 没有名称且 URL slug 没有帮助 —— 提供一个：
hermes skills install https://example.com/SKILL.md --name sharethis-chat

# 或在聊天会话中：
/skills install https://example.com/SKILL.md --name sharethis-chat
```

信任级别始终是 `community` —— 与每个其他来源运行相同的安全扫描。URL 存储为安装标识符，因此当您想要刷新时，`hermes skills update` 自动从同一 URL 重新获取。

### 安全扫描和 `--force`

所有中心安装的技能都经过**安全扫描**，检查数据外泄、提示注入、破坏性命令、供应链信号和其他威胁。

`hermes skills inspect ...` 现在还显示上游元数据（当可用时）：
- 仓库 URL
- skills.sh 详情页面 URL
- 安装命令
- 每周安装量
- 上游安全审计状态
- 知名索引/端点 URL

当您已审查第三方技能并想要覆盖非危险策略阻止时，请使用 `--force`：

```bash
hermes skills install skills-sh/anthropics/skills/pdf --force
```

重要行为：
- `--force` 可以覆盖谨慎/警告样式发现的策略阻止。
- `--force` **不会**覆盖 `dangerous` 扫描裁决。
- 官方可选技能（`official/...`）被视为内置信任，不显示第三方警告面板。

### 信任级别

| 级别 | 来源 | 策略 |
|-------|--------|--------|
| `builtin` | 随 Hermes 一起提供 | 始终信任 |
| `official` | 仓库中的 `optional-skills/` | 内置信任，无第三方警告 |
| `trusted` | 受信任的注册表/仓库，如 `openai/skills`、`anthropics/skills` | 比社区来源更宽松的策略 |
| `community` | 其他一切（`skills.sh`、知名端点、自定义 GitHub 仓库、大多数市场） | 非危险发现可以用 `--force` 覆盖；`dangerous` 裁决保持阻止 |

### 更新生命周期

中心现在跟踪足够的来源以重新检查已安装技能的上游副本：

```bash
hermes skills check          # 报告哪些已安装的中心技能在上游发生更改
hermes skills update         # 仅重新安装有更新的技能
hermes skills update react   # 更新一个特定的已安装中心技能
```

这使用存储的来源标识符加上当前上游包内容哈希来检测漂移。

:::tip GitHub 速率限制
技能中心操作使用 GitHub API，未认证用户的速率限制为 60 请求/小时。如果您在安装或搜索期间看到速率限制错误，请在您的 `.env` 文件中设置 `GITHUB_TOKEN` 以将限制增加到 5,000 请求/小时。发生这种情况时，错误消息包含可操作的提示。
:::

### 发布自定义技能 tap

如果您想要分享一组策划的技能 —— 为您的团队、您的组织或公开 —— 您可以将它们发布为 **tap**：其他 Hermes 用户用 `hermes skills tap add <owner/repo>` 添加的 GitHub 仓库。不需要服务器、不需要注册、不需要发布流水线。只需一个 `SKILL.md` 文件目录。

#### 仓库布局

tap 是任何 GitHub 仓库（公开或私有 —— 私有需要 `GITHUB_TOKEN`），布局如下：

```
owner/repo
├── skills/                       # 默认路径；可按 tap 配置
│   ├── my-workflow/
│   │   ├── SKILL.md              # 必需
│   │   ├── references/           # 可选支持文件
│   │   ├── templates/
│   │   └── scripts/
│   ├── another-skill/
│   │   └── SKILL.md
│   └── third-skill/
│       └── SKILL.md
└── README.md                     # 可选但有帮助
```

规则：
- 每个技能都住在 tap 根路径下的自己的目录中（默认 `skills/`）。
- 目录名称成为技能的安装 slug。
- 每个技能目录必须包含具有标准 [SKILL.md frontmatter](#skillmd-格式) 的 `SKILL.md`（`name`、`description`，加上可选的 `metadata.hermes.tags`、`version`、`author`、`platforms`、`metadata.hermes.config`）。
- 子目录如 `references/`、`templates/`、`scripts/`、`assets/` 在安装时与 `SKILL.md` 一起下载。
- 目录名称以 `.` 或 `_` 开头的技能被忽略。

Hermes 通过列出 tap 路径的每个子目录并探测每个目录中的 `SKILL.md` 来发现技能。

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

推送到 GitHub 后，任何 Hermes 用户都可以订阅和安装：

```bash
hermes skills tap add my-org/hermes-skills
hermes skills search deploy
hermes skills install my-org/hermes-skills/deploy-runbook
```

#### 非默认路径

如果您的技能不在 `skills/` 下（当您向现有项目添加 `skills/` 子树时很常见），请编辑 `~/.hermes/.hub/taps.json` 中的 tap 条目：

```json
{
  "taps": [
    {"repo": "my-org/platform-docs", "path": "internal/skills/"}
  ]
}
```

`hermes skills tap add` CLI 默认新 taps 为 `path: "skills/"`；如果您需要不同的路径，请直接编辑文件。`hermes skills tap list` 显示每个 tap 的有效路径。

#### 直接安装单个技能（无需添加 tap）

用户还可以从任何公开 GitHub 仓库安装单个技能，而无需将整个仓库添加为 tap：

```bash
hermes skills install owner/repo/skills/my-workflow
```

当您想要分享一个技能而无需用户订阅您的整个注册表时很有用。

#### Taps 的信任级别

新 taps 默认分配 `community` 信任。从它们安装的技能运行标准安全扫描，并在首次安装时显示第三方警告面板。如果您的组织或广泛信任的来源应该获得更高信任，请将其仓库添加到 `tools/skills_hub.py` 中的 `TRUSTED_REPOS`（需要 Hermes 核心 PR）。

#### Tap 管理

```bash
hermes skills tap list                                # 显示所有配置的 taps
hermes skills tap add myorg/skills-repo               # 添加（默认路径：skills/）
hermes skills tap remove myorg/skills-repo            # 移除
```

在运行中的会话中：

```
/skills tap list
/skills tap add myorg/skills-repo
/skills tap remove myorg/skills-repo
```

Taps 存储在 `~/.hermes/.hub/taps.json` 中（按需创建）。

## 捆绑技能更新 (`hermes skills reset`)

Hermes 附带一组捆绑技能在仓库内的 `skills/` 中。在安装时和每次 `hermes update` 时，同步过程将它们复制到 `~/.hermes/skills/` 并在 `~/.hermes/skills/.bundled_manifest` 中记录清单，将每个技能名称映射到同步时的内容哈希（**来源哈希**）。

在每次同步时，Hermes 重新计算您本地副本的哈希并将其与来源哈希进行比较：

- **未更改** → 安全拉取上游更改，复制新的捆绑版本，记录新的来源哈希。
- **已更改** → 被视为**用户修改**并永远跳过，因此您的编辑永远不会被覆盖。

保护很好，但它有一个尖锐的边缘。如果您编辑了一个捆绑技能，然后后来想要放弃您的更改并通过从 `~/.hermes/hermes-agent/skills/` 复制粘贴回到捆绑版本，清单仍然持有上次成功同步运行时的*旧*来源哈希。您的新复制粘贴内容（当前捆绑哈希）不会与该过时的来源哈希匹配，因此同步一直将其标记为用户修改。

`hermes skills reset` 是逃生舱口：

```bash
# 安全：清除此技能的清单条目。您当前的副本被保留，
# 但下一次同步会针对它重新基线，以便将来的更新正常工作。
hermes skills reset google-workspace

# 完全恢复：同时删除您的本地副本并重新复制当前捆绑
# 版本。当您想要原始的 upstream 技能时使用此选项。
hermes skills reset google-workspace --restore

# 非交互式（例如在脚本或 TUI 模式中）——跳过 --restore 确认。
hermes skills reset google-workspace --restore --yes
```

相同的命令在聊天中作为斜杠命令工作：

```text
/skills reset google-workspace
/skills reset google-workspace --restore
```

:::note Profiles
每个配置文件都有自己的 `.bundled_manifest`，位于自己的 `HERMES_HOME` 下，因此 `hermes -p coder skills reset <name>` 仅影响该配置文件。
:::

### 斜杠命令（聊天中）

所有相同的命令都适用于 `/skills`：

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

官方可选技能仍然使用 `official/security/1password` 和 `official/migration/openclaw-migration` 等标识符。

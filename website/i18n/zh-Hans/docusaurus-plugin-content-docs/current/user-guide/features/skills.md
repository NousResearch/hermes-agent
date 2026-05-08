---
sidebar_position: 2
title: "技能系统"
description: "按需知识文档 — 渐进式披露、代理管理的技能和技能中心"
---

# 技能系统

技能是代理可以在需要时加载的按需知识文档。它们遵循**渐进式披露**模式以最小化 token 使用，并与 [agentskills.io](https://agentskills.io/specification) 开放标准兼容。

所有技能存储在 **`~/.hermes/skills/`** 中 — 这是主目录和信息来源。首次安装时，捆绑技能从仓库复制到此目录。中心安装和代理创建的技能也放在这里。代理可以修改或删除任何技能。

你还可以让 Hermes 指向**外部技能目录** — 与本地目录一起扫描的额外文件夹。见下方的 [外部技能目录](#外部技能目录)。

另见：

- [捆绑技能目录](/docs/reference/skills-catalog)
- [官方可选技能目录](/docs/reference/optional-skills-catalog)

## 使用技能

每个已安装的技能都自动作为斜杠命令可用：

```bash
# 在 CLI 或任何消息平台上：
/gif-search funny cats
/axolotl help me fine-tune Llama 3 on my dataset
/github-pr-workflow create a PR for the auth refactor
/plan design a rollout for migrating our auth provider

# 只需技能名称即可加载，让代理询问你的需求：
/excalidraw
```

捆绑的 `plan` 技能是一个很好的例子。运行 `/plan [request]` 会加载技能的指令，告诉 Hermes 在需要时检查上下文、编写 markdown 实现计划而不是执行任务，并将结果保存在 `.hermes/plans/` 下（相对于活动工作区/后端工作目录）。

你还可以通过自然对话与技能交互：

```bash
hermes chat --toolsets skills -q "What skills do you have?"
hermes chat --toolsets skills -q "Show me the axolotl skill"
```

## 渐进式披露

技能使用节省 token 的加载模式：

```
Level 0: skills_list()           → [{name, description, category}, ...]   (~3k tokens)
Level 1: skill_view(name)        → Full content + metadata       (varies)
Level 2: skill_view(name, path)  → Specific reference file       (varies)
```

代理仅在确实需要时才加载完整的技能内容。

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

### 特定平台的技能

技能可以使用 `platforms` 字段限制自己到特定操作系统：

| 值 | 匹配 |
|-------|---------|
| `macos` | macOS (Darwin) |
| `linux` | Linux |
| `windows` | Windows |

```yaml
platforms: [macos]            # 仅 macOS（例如 iMessage、Apple 提醒事项、查找）
platforms: [macos, linux]     # macOS 和 Linux
```

设置后，技能在兼容平台上会自动从系统提示、`skills_list()` 和斜杠命令中隐藏。如果省略，技能在所有平台上加载。

### 条件激活（回退技能）

技能可以根据当前会话中可用的工具自动显示或隐藏。这对于**回退技能**最有用 — 仅当高级工具不可用时才应出现的免费或本地替代方案。

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
| `fallback_for_toolsets` | 当列出的工具集可用时技能**隐藏**。当它们缺失时显示。|
| `fallback_for_tools` | 相同，但检查单个工具而不是工具集。|
| `requires_toolsets` | 当列出的工具集不可用时技能**隐藏**。当它们存在时显示。|
| `requires_tools` | 相同，但检查单个工具。|

**示例：** 内置的 `duckduckgo-search` 技能使用 `fallback_for_toolsets: [web]`。当你设置了 `FIRECRAWL_API_KEY` 时，web 工具集可用，代理使用 `web_search` — DuckDuckGo 技能保持隐藏。如果 API 密钥缺失，web 工具集不可用，DuckDuckGo 技能自动作为回退出现。

没有任何条件字段的技能行为与之前完全相同 — 它们始终显示。

## 加载时的安全设置

技能可以声明所需的环境变量而不会从发现中消失：

```yaml
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: Tenor API key
    help: Get a key from https://developers.google.com/tenor
    required_for: full functionality
```

当遇到缺失的值时，Hermes 仅在技能在本地 CLI 中实际加载时才会安全地询问。你可以跳过设置并继续使用技能。消息界面永远不会在聊天中询问密钥 — 它们会告诉你在本地使用 `hermes setup` 或 `~/.hermes/.env`。

设置后，声明的环境变量会**自动传递**到 `execute_code` 和 `terminal` 沙箱 — 技能的脚本可以直接使用 `$TENOR_API_KEY`。对于非技能环境变量，使用 `terminal.env_passthrough` 配置选项。详情见 [环境变量传递](/docs/user-guide/security#environment-variable-passthrough)。

### 技能配置设置

技能还可以声明存储在 `config.yaml` 中的非密钥配置设置（路径、偏好）：

```yaml
metadata:
  hermes:
    config:
      - key: myplugin.path
        description: Path to the plugin data directory
        default: "~/myplugin-data"
        prompt: Plugin data directory path
```

设置存储在 `config.yaml` 的 `skills.config` 下。`hermes config migrate` 会提示输入未配置的设置，`hermes config show` 会显示它们。当技能加载时，其解析的配置值会被注入到上下文中，使代理自动知道配置的值。

详情见 [技能设置](/docs/user-guide/configuration#skill-settings) 和 [创建技能 — 配置设置](/docs/developer-guide/creating-skills#config-settings-configyaml)。

## 技能目录结构

```text
~/.hermes/skills/                  # 唯一的信息来源
├── mlops/                         # 类别目录
│   ├── axolotl/
│   │   ├── SKILL.md               # 主指令（必需）
│   │   ├── references/            # 额外文档
│   │   ├── templates/             # 输出格式
│   │   ├── scripts/               # 可从技能调用的辅助脚本
│   │   └── assets/                # 补充文件
│   └── vllm/
│       └── SKILL.md
├── devops/
│   └── deploy-k8s/                # 代理创建的技能
│       ├── SKILL.md
│       └── references/
├── .hub/                          # 技能中心状态
│   ├── lock.json
│   ├── quarantine/
│   └── audit.log
└── .bundled_manifest              # 跟踪已播种的捆绑技能
```

## 外部技能目录

如果你在 Hermes 之外维护技能 — 例如，多个 AI 工具使用的共享 `~/.agents/skills/` 目录 — 你可以让 Hermes 也扫描这些目录。

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

- **只读**：外部目录仅用于技能发现扫描。当代理创建或编辑技能时，它总是写入 `~/.hermes/skills/`。
- **本地优先**：如果同名技能同时存在于本地目录和外部目录中，本地版本优先。
- **完全集成**：外部技能出现在系统提示索引、`skills_list`、`skill_view` 和 `/skill-name` 斜杠命令中 — 与本地技能没有区别。
- **不存在的路径被静默跳过**：如果配置的目录不存在，Hermes 会忽略它而不会报错。适用于可能并非每台机器上都存在的可选共享目录。

### 示例

```text
~/.hermes/skills/               # 本地（主要，读写）
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

所有四个技能都会出现在你的技能索引中。如果你在本地创建了名为 `my-custom-workflow` 的新技能，它将覆盖外部版本。

## 代理管理的技能（skill_manage 工具）

代理可以通过 `skill_manage` 工具创建、更新和删除自己的技能。这是代理的**程序性记忆** — 当它找到了一个非平凡的工作流程时，会将方法保存为技能以供将来重用。

### 代理何时创建技能

- 成功完成复杂任务（5 次以上工具调用）后
- 当遇到错误或死胡同并找到了可行路径时
- 当用户纠正了它的方法时
- 当发现非平凡的工作流程时

### 操作

| 操作 | 用途 | 关键参数 |
|--------|---------|------------|
| `create` | 从零创建新技能 | `name`、`content`（完整 SKILL.md）、可选 `category` |
| `patch` | 定向修复（首选）| `name`、`old_string`、`new_string` |
| `edit` | 重大结构重写 | `name`、`content`（完整 SKILL.md 替换）|
| `delete` | 完全删除技能 | `name` |
| `write_file` | 添加/更新支持文件 | `name`、`file_path`、`file_content` |
| `remove_file` | 删除支持文件 | `name`、`file_path` |

:::tip
`patch` 操作是更新的首选 — 它比 `edit` 更节省 token，因为只有变更的文本出现在工具调用中。
:::

## 技能中心

浏览、搜索、安装和管理来自在线注册表、`skills.sh`、直接 well-known 技能端点和官方可选技能的技能。

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
hermes skills update                              # 在需要时重新安装有上游更改的中心技能
hermes skills audit                               # 重新扫描所有中心技能的安全性
hermes skills uninstall k8s                       # 移除中心技能
hermes skills reset google-workspace              # 将捆绑技能从"用户修改"状态解除（见下文）
hermes skills reset google-workspace --restore    # 同时恢复捆绑版本，删除本地编辑
hermes skills publish skills/my-skill --to github --repo owner/repo
hermes skills snapshot export setup.json          # 导出技能配置
hermes skills tap add myorg/skills-repo           # 添加自定义 GitHub 来源
```

### 支持的中心来源

| 来源 | 示例 | 备注 |
|--------|---------|-------|
| `official` | `official/security/1password` | 随 Hermes 发布的可选技能。|
| `skills-sh` | `skills-sh/vercel-labs/agent-skills/vercel-react-best-practices` | 可通过 `hermes skills search <query> --source skills-sh` 搜索。当 skills.sh slug 与仓库文件夹不同时，Hermes 会解析别名式技能。|
| `well-known` | `well-known:https://mintlify.com/docs/.well-known/skills/mintlify` | 直接从网站上的 `/.well-known/skills/index.json` 提供的技能。使用网站或文档 URL 搜索。|
| `url` | `https://sharethis.chat/SKILL.md` | 指向单文件 `SKILL.md` 的直接 HTTP(S) URL。名称解析顺序：frontmatter → URL slug → 交互式提示 → `--name` 标志。|
| `github` | `openai/skills/k8s` | 直接 GitHub 仓库/路径安装和自定义 tap。|
| `clawhub`、`lobehub`、`claude-marketplace` | 来源特定标识符 | 社区或市场集成。|

### 集成的中心和注册表

Hermes 目前集成了这些技能生态系统和发现来源：

#### 1. 官方可选技能（`official`）

这些在 Hermes 仓库本身中维护，安装时具有内置信任。

- 目录：[官方可选技能目录](../../reference/optional-skills-catalog)
- 仓库中来源：`optional-skills/`
- 示例：

```bash
hermes skills browse --source official
hermes skills install official/security/1password
```

#### 2. skills.sh（`skills-sh`）

这是 Vercel 的公共技能目录。Hermes 可以直接搜索它、检查技能详情页、解析别名式 slug 并从底层源仓库安装。

- 目录：[skills.sh](https://skills.sh/)
- CLI/工具仓库：[vercel-labs/skills](https://github.com/vercel-labs/skills)
- Vercel 官方技能仓库：[vercel-labs/agent-skills](https://github.com/vercel-labs/agent-skills)
- 示例：

```bash
hermes skills search react --source skills-sh
hermes skills inspect skills-sh/vercel-labs/json-render/json-render-react
hermes skills install skills-sh/vercel-labs/json-render/json-render-react --force
```

#### 3. Well-known 技能端点（`well-known`）

这是基于 URL 的发现机制，适用于发布 `/.well-known/skills/index.json` 的网站。它不是单个集中式中心 — 而是一种 Web 发现约定。

- 实时端点示例：[Mintlify 文档技能索引](https://mintlify.com/docs/.well-known/skills/index.json)
- 参考服务器实现：[vercel-labs/skills-handler](https://github.com/vercel-labs/skills-handler)
- 示例：

```bash
hermes skills search https://mintlify.com/docs --source well-known
hermes skills inspect well-known:https://mintlify.com/docs/.well-known/skills/mintlify
hermes skills install well-known:https://mintlify.com/docs/.well-known/skills/mintlify
```

#### 4. 直接 GitHub 技能（`github`）

Hermes 可以直接从 GitHub 仓库和基于 GitHub 的 tap 安装。当你已经知道仓库/路径或想要添加自己的自定义源仓库时，这很有用。

默认 tap（无需任何设置即可浏览）：
- [openai/skills](https://github.com/openai/skills)
- [anthropics/skills](https://github.com/anthropics/skills)
- [VoltAgent/awesome-agent-skills](https://github.com/VoltAgent/awesome-agent-skills)
- [garrytan/gstack](https://github.com/garrytan/gstack)

- 示例：

```bash
hermes skills install openai/skills/k8s
hermes skills tap add myorg/skills-repo
```

#### 5. ClawHub（`clawhub`）

作为社区来源集成的第三方技能市场。

- 网站：[clawhub.ai](https://clawhub.ai/)
- Hermes 来源 ID：`clawhub`

#### 6. Claude 市场风格仓库（`claude-marketplace`）

Hermes 支持发布 Claude 兼容插件/市场清单的市场仓库。

已知集成的来源包括：
- [anthropics/skills](https://github.com/anthropics/skills)
- [aiskillstore/marketplace](https://github.com/aiskillstore/marketplace)

Hermes 来源 ID：`claude-marketplace`

#### 7. LobeHub（`lobehub`）

Hermes 可以搜索 LobeHub 公共目录中的代理条目并将其转换为可安装的 Hermes 技能。

- 网站：[LobeHub](https://lobehub.com/)
- 公共代理索引：[chat-agents.lobehub.com](https://chat-agents.lobehub.com/)
- 支持仓库：[lobehub/lobe-chat-agents](https://github.com/lobehub/lobe-chat-agents)
- Hermes 来源 ID：`lobehub`

#### 8. 直接 URL（`url`）

从任何 HTTP(S) URL 直接安装单文件 `SKILL.md` — 适用于作者在自己的网站上托管技能（无中心列表，无需输入 GitHub 路径）的情况。Hermes 获取 URL，解析 YAML frontmatter，进行安全扫描并安装。

- Hermes 来源 ID：`url`
- 标识符：URL 本身（无需前缀）
- 范围：**仅限单文件 `SKILL.md`**。带有 `references/` 或 `scripts/` 的多文件技能需要清单，应通过上述其他来源之一发布。

```bash
hermes skills install https://sharethis.chat/SKILL.md
hermes skills install https://example.com/my-skill/SKILL.md --category productivity
```

名称解析顺序：
1. SKILL.md YAML frontmatter 中的 `name:` 字段（推荐 — 每个格式良好的技能都有）。
2. URL 路径中的父目录名称（如 `.../my-skill/SKILL.md` → `my-skill`，或 `.../my-skill.md` → `my-skill`），当它是有效标识符时（`^[a-z][a-z0-9_-]*$`）。
3. 在有 TTY 的终端上的交互式提示。
4. 在非交互式界面（TUI 中的 `/skills install` 斜杠命令、网关平台、脚本）上，一个指向 `--name` 覆盖的清晰错误。

```bash
# frontmatter 没有名称且 URL slug 不够有用 — 提供一个：
hermes skills install https://example.com/SKILL.md --name sharethis-chat

# 或在聊天会话中：
/skills install https://example.com/SKILL.md --name sharethis-chat
```

信任级别始终为 `community` — 与每个其他来源一样运行相同的安全扫描。URL 存储为安装标识符，因此当你想要刷新时，`hermes skills update` 会自动从同一 URL 重新获取。

### 安全扫描和 `--force`

所有中心安装的技能都经过**安全扫描器**检查，检查数据泄露、提示注入、破坏性命令、供应链信号和其他威胁。

`hermes skills inspect ...` 现在还会在可用时显示上游元数据：
- 仓库 URL
- skills.sh 详情页 URL
- 安装命令
- 每周安装量
- 上游安全审计状态
- well-known 索引/端点 URL

当你审查了第三方技能并想要覆盖非危险的策略阻止时，使用 `--force`：

```bash
hermes skills install skills-sh/anthropics/skills/pdf --force
```

重要行为：
- `--force` 可以覆盖警告/谨慎类发现的策略阻止。
- `--force` **不会**覆盖 `dangerous` 扫描判定。
- 官方可选技能（`official/...`）被视为内置信任，不会显示第三方警告面板。

### 信任级别

| 级别 | 来源 | 策略 |
|-------|--------|--------|
| `builtin` | 随 Hermes 发布 | 始终受信任 |
| `official` | 仓库中的 `optional-skills/` | 内置信任，无第三方警告 |
| `trusted` | 受信任的注册表/仓库，如 `openai/skills`、`anthropics/skills` | 比社区来源更宽松的策略 |
| `community` | 其他所有（`skills.sh`、well-known 端点、自定义 GitHub 仓库、大多数市场）| 非危险发现可用 `--force` 覆盖；`dangerous` 判定保持阻止 |

### 更新生命周期

中心现在跟踪足够的来源信息来重新检查已安装技能的上游副本：

```bash
hermes skills check          # 报告哪些已安装的中心技能在上游发生了变化
hermes skills update         # 仅重新安装有更新的技能
hermes skills update react   # 更新一个特定的已安装中心技能
```

这使用存储的来源标识符和当前上游包内容哈希来检测漂移。

:::tip GitHub 速率限制
技能中心操作使用 GitHub API，对未认证用户有每小时 60 次请求的限制。如果在安装或搜索时看到速率限制错误，在 `.env` 文件中设置 `GITHUB_TOKEN` 将限制提高到每小时 5,000 次请求。发生此情况时，错误消息中包含可操作的提示。
:::

### 发布自定义技能 tap

如果你想分享一组策划好的技能 — 为你的团队、你的组织或公开 — 你可以将它们作为 **tap** 发布：一个其他 Hermes 用户通过 `hermes skills tap add <owner/repo>` 添加的 GitHub 仓库。无需服务器，无需注册表注册，无需发布流水线。只需要一个 `SKILL.md` 文件目录。

#### 仓库布局

tap 是任何 GitHub 仓库（公开或私有 — 私有需要 `GITHUB_TOKEN`），布局如下：

```
owner/repo
├── skills/                       # 默认路径；每个 tap 可配置
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
- 每个技能位于 tap 根路径（默认 `skills/`）下的自己的目录中。
- 目录名称成为技能的安装 slug。
- 每个技能目录必须包含带有标准 [SKILL.md frontmatter](#skillmd-格式) 的 `SKILL.md`（`name`、`description`，以及可选的 `metadata.hermes.tags`、`version`、`author`、`platforms`、`metadata.hermes.config`）。
- `references/`、`templates/`、`scripts/`、`assets/` 等子目录在安装时与 `SKILL.md` 一起下载。
- 目录名以 `.` 或 `_` 开头的技能被忽略。

Hermes 通过列出 tap 路径的每个子目录并探测每个目录的 `SKILL.md` 来发现技能。

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

推送到 GitHub 后，任何 Hermes 用户都可以订阅并安装：

```bash
hermes skills tap add my-org/hermes-skills
hermes skills search deploy
hermes skills install my-org/hermes-skills/deploy-runbook
```

#### 非默认路径

如果你的技能不在 `skills/` 下（当你向现有项目添加 `skills/` 子目录时很常见），编辑 `~/.hermes/.hub/taps.json` 中的 tap 条目：

```json
{
  "taps": [
    {"repo": "my-org/platform-docs", "path": "internal/skills/"}
  ]
}
```

`hermes skills tap add` CLI 默认新 tap 的路径为 `path: "skills/"`；如果需要不同的路径，直接编辑文件。`hermes skills tap list` 显示每个 tap 的有效路径。

#### 直接安装单个技能（无需添加 tap）

用户还可以从任何公共 GitHub 仓库安装单个技能，无需将整个仓库添加为 tap：

```bash
hermes skills install owner/repo/skills/my-workflow
```

当你想分享一个技能而不要求用户订阅你的整个注册表时，这很有用。

#### Tap 的信任级别

新 tap 默认分配 `community` 信任级别。从它们安装的技能运行标准安全扫描，并在首次安装时显示第三方警告面板。如果你的组织或广泛信任的来源应该获得更高的信任，将其仓库添加到 `tools/skills_hub.py` 中的 `TRUSTED_REPOS`（需要 Hermes 核心 PR）。

#### Tap 管理

```bash
hermes skills tap list                                # 显示所有配置的 tap
hermes skills tap add myorg/skills-repo               # 添加（默认路径：skills/）
hermes skills tap remove myorg/skills-repo            # 移除
```

在运行中的会话中：

```
/skills tap list
/skills tap add myorg/skills-repo
/skills tap remove myorg/skills-repo
```

Tap 存储在 `~/.hermes/.hub/taps.json` 中（按需创建）。

## 捆绑技能更新（`hermes skills reset`）

Hermes 随仓库中的 `skills/` 附带一组捆绑技能。安装时和每次 `hermes update` 时，同步过程将这些复制到 `~/.hermes/skills/` 并在 `~/.hermes/skills/.bundled_manifest` 中记录清单，将每个技能名称映射到同步时的内容哈希（**来源哈希**）。

每次同步时，Hermes 重新计算本地副本的哈希并与来源哈希比较：

- **未更改** → 可以安全拉取上游更改，复制新的捆绑版本，记录新的来源哈希。
- **已更改** → 被视为**用户修改**并永久跳过，因此你的编辑永远不会被覆盖。

这种保护很好，但有一个尖锐的边缘。如果你编辑了一个捆绑技能，然后后来想放弃更改并通过从 `~/.hermes/hermes-agent/skills/` 复制粘贴回到捆绑版本，清单仍然持有上次成功同步时的*旧*来源哈希。你的新复制粘贴内容（当前捆绑哈希）不会匹配那个陈旧的来源哈希，因此同步会继续将其标记为用户修改。

`hermes skills reset` 是逃生口：

```bash
# 安全：清除此技能的清单条目。保留你当前的副本，
# 但下一次同步会重新以它为基准，使未来更新正常工作。
hermes skills reset google-workspace

# 完全恢复：同时删除本地副本并重新复制当前捆绑版本。
# 当你想要恢复纯净的上游技能时使用。
hermes skills reset google-workspace --restore

# 非交互式（例如在脚本或 TUI 模式中）— 跳过 --restore 确认。
hermes skills reset google-workspace --restore --yes
```

同一命令在聊天中作为斜杠命令使用：

```text
/skills reset google-workspace
/skills reset google-workspace --restore
```

:::note 配置文件
每个配置文件在自己的 `HERMES_HOME` 下有自己的 `.bundled_manifest`，因此 `hermes -p coder skills reset <name>` 仅影响该配置文件。
:::

### 斜杠命令（在聊天中）

所有相同的命令可以通过 `/skills` 使用：

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

官方可选技能仍使用 `official/security/1password` 和 `official/migration/openclaw-migration` 等标识符。

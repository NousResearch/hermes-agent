---
sidebar_position: 3
---

# Profile Distributions：分享完整 agent

**profile distribution** 会把一个完整的 Hermes agent 打包成一个 git 仓库 - 包括人格、技能、cron 任务、MCP 连接和配置。任何人只要能访问这个仓库，就可以用一条命令安装整个 agent，原地更新，而且不会碰到自己的记忆、会话和 API Key。

如果 [profile](./profiles.md) 是本地 agent，那么 distribution 就是这个 agent 的可分享版本。

## 这意味着什么

在 distribution 出现之前，分享一个 Hermes agent 往往意味着要把下面这些东西发给别人：

1. 你的 SOUL.md
2. 需要安装的技能列表
3. 去掉密钥后的 config.yaml
4. 你接好的 MCP server 说明
5. 你安排好的 cron 任务
6. 需要设置哪些环境变量的说明

……然后祈祷对方能正确拼起来。每次版本升级或 bug 修复都要重新交付一次。

有了 distributions，这些内容都放进一个 git 仓库里：

```
my-research-agent/
├── distribution.yaml    # manifest：名称、版本、环境变量要求
├── SOUL.md              # agent 的人格 / system prompt
├── config.yaml          # 模型、温度、推理、工具默认值
├── skills/              # 随 agent 一起提供的 bundled skills
├── cron/                # agent 运行的定时任务
└── mcp.json             # agent 连接的 MCP servers
```

接收者运行：

```bash
hermes profile install github.com/you/my-research-agent --alias
```

……然后他们就拥有了整个 agent。接收者填自己的 API Key（`.env.EXAMPLE` → `.env`）后，就可以运行 `my-research-agent chat`，或者通过 Telegram / Discord / Slack / 其他网关平台来访问它。等你推送新版本时，他们执行 `hermes profile update my-research-agent` 就能拉取你的修改 - 他们自己的记忆和会话不会受影响。

## 为什么用 git？

我们考虑过 tarball、HTTP archive、自定义格式，但都不如 git：

- **作者几乎不需要构建步骤。** Push 到 GitHub，使用者就能安装。不需要“打包、上传、更新索引”这一套流程。
- **tag、branch 和 commit 天然就是版本系统。** tag push 在这里起到的作用和其他工具里“pack + upload release”一样。
- **更新就是一次 fetch。** 不是把整个 archive 重新下载一遍。
- **透明。** 用户可以浏览仓库、查看版本差异、提 issue、fork 后自行修改。
- **私有仓库也免费可用。** SSH key、git credential helper、GitHub CLI 已保存的认证信息——你终端里已经配置好的 git 认证方式都会透明复用。
- **可复现性就是 commit SHA。** 跟 pip 和 npm 记录版本的方式类似。

代价是：接收者需要安装 git。到了 2026 年，能跑 Hermes 的机器基本都已经满足这一点。

## 什么时候该用 distribution？

适合的场景：

- **你要分享一个专门化 agent** - 合规监控、代码审查、研究助手、客服机器人 - 给团队或社区使用。
- **你要把同一个 agent 部署到多台机器上**，不想每次手动复制文件。
- **你在迭代一个 agent**，希望接收者用一条命令就能拿到新版本。
- **你把 agent 当成产品在做** - 有观点的默认值、筛选过的技能、调优过的 prompt - 让别人拿来当起点。

不适合的场景：

- **你只是想在自己机器上备份一个 profile。** 用 [`hermes profile export` / `import`](/reference/profile-commands#hermes-profile-export) - 这就是它们的用途。
- **你想连同 agent 一起分享 API Key。** `auth.json` 和 `.env` 会被刻意排除在 distribution 之外。每个安装者都应该使用自己的凭据。
- **你想分享记忆 / 会话 / 聊天历史。** 那些是用户数据，不是 distribution 内容，绝不会被打包发布。

## 生命周期：作者 -> 安装者 -> 更新

下面是完整端到端流程。你可以只看自己关心的那一边。

---

## 给作者：发布 distribution

### 第 1 步 - 从一个正常工作的 profile 开始

像平常一样构建和打磨这个 agent：

```bash
hermes profile create research-bot
research-bot setup                    # 配置模型、API Key
# 编辑 ~/.hermes/profiles/research-bot/SOUL.md
# 安装技能、接 MCP server、安排 cron 任务等
research-bot chat                     # 自己先用起来，直到觉得顺手
```

### 第 2 步 - 添加 `distribution.yaml`

在 `~/.hermes/profiles/research-bot/distribution.yaml` 中创建清单：

```yaml
name: research-bot
version: 1.0.0
description: "Autonomous research assistant with arXiv and web tools"
hermes_requires: ">=0.12.0"
author: "Your Name"
license: "MIT"

# 告诉安装器这个 agent 需要哪些环境变量。它会拿这些和安装者的 shell
# 以及已有 .env 文件比对，这样已有配置的 key 就不会被重复提示。
env_requires:
  - name: OPENAI_API_KEY
    description: "OpenAI API key (for model access)"
    required: true
  - name: SERPAPI_KEY
    description: "SerpAPI key for web search"
    required: false
    default: ""
```

这就是全部 manifest。除了 `name` 之外，其他字段都有合理默认值。

### 第 3 步 - 推送到 git 仓库

```bash
cd ~/.hermes/profiles/research-bot
git init
git add .
git commit -m "v1.0.0"
git remote add origin git@github.com:you/research-bot.git
git tag v1.0.0
git push -u origin main --tags
```

仓库现在就是一个 distribution。任何有权限的人都可以安装它。

:::note
git 仓库里会包含**profile 目录里的所有内容，除了 distribution 已经明确排除的东西**：`auth.json`、`.env`、`memories/`、`sessions/`、`state.db*`、`logs/`、`workspace/`、`*_cache/`、`local/`。这些都会留在你自己的机器上。你也可以额外添加 `.gitignore` 来排除更多路径。
:::

### 第 4 步 - 标记版本发布

每当 agent 达到一个稳定点，就升级版本并打 tag：

```bash
# 编辑 distribution.yaml：version: 1.1.0
git add distribution.yaml SOUL.md skills/
git commit -m "v1.1.0: tighter research SOUL, add arxiv skill"
git tag v1.1.0
git push --tags
```

接收者运行 `hermes profile update research-bot` 时就会拉取最新版。

### 仓库长什么样

一个完整的 authoring distribution 示例：

```
research-bot/
├── distribution.yaml            # 必需
├── SOUL.md                      # 强烈建议
├── config.yaml                  # 模型、提供商、工具默认值
├── mcp.json                     # MCP server 连接
├── skills/
│   ├── arxiv-search/SKILL.md
│   ├── paper-summarization/SKILL.md
│   └── citation-lookup/SKILL.md
├── cron/
│   └── weekly-digest.json       # 定时任务
└── README.md                    # 面向人的说明文档（可选）
```

### distribution-owned 与 user-owned

当安装者更新到新版本时，有些内容会被替换（作者负责），有些内容会保留（用户负责）。默认规则如下：

| 类别 | 路径 | 更新时 |
|---|---|---|
| **Distribution-owned** | `SOUL.md`、`config.yaml`、`mcp.json`、`skills/`、`cron/`、`distribution.yaml` | 会从新的 clone 覆盖 |
| **Config override** | `config.yaml` | 默认其实会保留 - 安装器可能已经调整过模型或提供商。更新时传 `--force-config` 可以重置。 |
| **User-owned** | `memories/`、`sessions/`、`state.db*`、`auth.json`、`.env`、`logs/`、`workspace/`、`plans/`、`home/`、`*_cache/`、`local/` | 永远不会触碰 |

你可以在 manifest 里覆盖 distribution-owned 列表：

```yaml
distribution_owned:
  - SOUL.md
  - skills/research/            # 只包含我的研究技能；其他已安装技能保留
  - cron/digest.json
```

如果不写，就按上面的默认规则来 - 这也是大多数 distribution 想要的行为。

---

## 给安装者：使用 distribution

### 安装

```bash
hermes profile install github.com/you/research-bot --alias
```

安装过程如下：

1. 把仓库克隆到临时目录。
2. 读取 `distribution.yaml`，展示 manifest（名称、版本、描述、作者、需要的环境变量）。
3. 检查每个必需环境变量是否已经在你的 shell 或目标 profile 的现有 `.env` 中设置好。会标记成 `✓ set` 或 `needs setting`，这样你就知道该配置什么。
4. 请求你确认。加 `-y` / `--yes` 可以跳过。
5. 把 distribution-owned 文件复制到 `~/.hermes/profiles/research-bot/`（或者 manifest 的 `name` 解析出来的目录）。
6. 写出 `.env.EXAMPLE`，把必需的 key 注释进去 - 你只要复制成 `.env` 然后填值。
7. 使用 `--alias` 时，会创建一个 wrapper，让你可以直接运行 `research-bot chat`。

### 来源类型

任何 git URL 都可以：

```bash
# GitHub 简写
hermes profile install github.com/you/research-bot

# 完整 HTTPS
hermes profile install https://github.com/you/research-bot.git

# SSH
hermes profile install git@github.com:you/research-bot.git

# 自建、GitLab、Gitea、Forgejo - 任何 Git 托管都行
hermes profile install https://git.example.com/team/research-bot.git

# 使用你已经配置好的 git 认证访问私有仓库
hermes profile install git@github.com:your-org/internal-bot.git

# 开发期间的本地目录（不用先 push）
hermes profile install ~/my-profile-in-progress/
```

### 覆盖 profile 名称

两个人想把同一个 distribution 装成不同名字的 profile：

```bash
# Alice
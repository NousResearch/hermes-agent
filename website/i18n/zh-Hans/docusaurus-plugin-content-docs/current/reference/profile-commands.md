---
sidebar_position: 7
---

# Profile 命令参考

本页涵盖所有与 [Hermes profiles](../user-guide/profiles.md) 相关的命令。通用 CLI 命令请参阅 [CLI 命令参考](./cli-commands.md)。

## `hermes profile`

```bash
hermes profile <subcommand>
```

管理 profile 的顶级命令。不带子命令运行 `hermes profile` 将显示帮助信息。

| 子命令 | 描述 |
|------------|-------------|
| `list` | 列出所有 profile。 |
| `use` | 设置当前活跃（默认）profile。 |
| `create` | 创建新 profile。 |
| `delete` | 删除 profile。 |
| `show` | 显示 profile 详情。 |
| `alias` | 重新生成 profile 的 shell alias。 |
| `rename` | 重命名 profile。 |
| `export` | 将 profile 导出为 tar.gz 归档文件。 |
| `import` | 从 tar.gz 归档文件导入 profile。 |
| `install` | 从 git URL 或本地目录安装 profile 发行版。参见 [Profile 发行版](../user-guide/profile-distributions.md)。 |
| `update` | 重新拉取发行版管理的 profile 并重新应用其 bundle。 |
| `info` | 显示 profile 的发行版元数据（来源 URL、commit、最后更新时间）。 |

## `hermes profile list`

```bash
hermes profile list
```

列出所有 profile。当前活跃的 profile 以 `*` 标记。

**示例：**

```bash
$ hermes profile list
  default
* work
  dev
  personal
```

无选项。

## `hermes profile use`

```bash
hermes profile use <name>
```

将 `<name>` 设为活跃 profile。此后所有 `hermes` 命令（不带 `-p`）都将使用该 profile。

| 参数 | 描述 |
|----------|-------------|
| `<name>` | 要激活的 profile 名称。使用 `default` 可返回基础 profile。 |

**示例：**

```bash
hermes profile use work
hermes profile use default
```

## `hermes profile create`

```bash
hermes profile create <name> [options]
```

创建新 profile。

| 参数 / 选项 | 描述 |
|-------------------|-------------|
| `<name>` | 新 profile 的名称。必须是合法的目录名（字母数字、连字符、下划线）。 |
| `--clone` | 从当前 profile 复制 `config.yaml`、`.env`、`SOUL.md` 和 skills。 |
| `--clone-all` | 从当前 profile 复制所有内容（config、memories、skills、cron、plugins）。会排除每个 profile 自己的历史数据：sessions、`state.db`、backups、state-snapshots、checkpoints。 |
| `--clone-from <profile>` | 从指定 profile 克隆 config/skills/SOUL，而非当前 profile。除非与 `--clone-all` 配合使用，否则会隐含 `--clone`。 |
| `--no-alias` | 跳过 wrapper 脚本创建。 |
| `--description "<text>"` | 一到两句话描述该 profile 的用途。供 kanban 编排器根据角色而非仅凭 profile 名称来路由任务。可跳过，稍后通过 `hermes profile describe` 添加。持久化保存在 `<profile_dir>/profile.yaml` 中。 |
| `--no-skills` | 创建一个**空** profile，不启用任何内置 skill。会在 profile 目录中写入 `.no-bundled-skills` 标记，使后续 `hermes update` 不再重新植入内置 skill 集，且拒绝与 `--clone`、`--clone-from` 或 `--clone-all` 组合使用（因为这些选项会复制 skill）。适用于不应继承完整 skill 目录的窄化编排器 profile 或沙箱 profile。 |

创建 profile **不会**将该 profile 目录设为终端命令的默认项目/工作目录。如需让某个 profile 从特定项目目录启动，请在该 profile 的 `config.yaml` 中设置 `terminal.cwd`。

**示例：**

```bash
# 空白 profile — 需要完整配置
hermes profile create mybot

# 仅从当前 profile 克隆 config
hermes profile create work --clone

# 从当前 profile 克隆所有内容
hermes profile create backup --clone-all

# 从指定 profile 克隆 config
hermes profile create work2 --clone-from work

# 从指定 profile 克隆所有内容
hermes profile create work2-backup --clone-from work --clone-all
```

## `hermes profile describe`

```bash
hermes profile describe [<name>] [options]
```

读取或设置 profile 的描述。描述由 kanban 编排器使用，用于根据每个 profile 的能力路由任务，而非仅凭 profile 名称猜测。持久化保存在 `<profile_dir>/profile.yaml` 中，重启后仍有效，并与 gateway 共享。

不带任何标志时，打印当前描述（若为空则显示 `(no description set for '<name>')`）。

| 参数 / 选项 | 描述 |
|-------------------|-------------|
| `<name>` | 要描述的 profile。除非使用 `--all --auto`，否则必填。 |
| `--text "<text>"` | 将描述设置为此精确文本（用户编写）。覆盖已有描述。 |
| `--auto` | 通过辅助 LLM 自动生成 1-2 句描述，依据为该 profile 已安装的 skill、配置的模型和名称。在 `config.yaml` 的 `auxiliary.profile_describer` 下配置模型。自动生成的描述会标记 `description_auto: true`，以便 dashboard 标记供审查。 |
| `--overwrite` | 与 `--auto` 配合使用时，也替换用户编写的描述（默认：跳过已明确设置描述的 profile）。 |
| `--all` | 与 `--auto` 配合使用时，扫描所有缺少描述的 profile。 |

**示例：**

```bash
# 读取当前描述
hermes profile describe researcher

# 显式设置描述
hermes profile describe researcher --text "Reads source code and writes findings."

# 让 LLM 生成描述
hermes profile describe researcher --auto

# 为所有没有描述的 profile 填充描述
hermes profile describe --all --auto
```

## `hermes profile delete`

```bash
hermes profile delete <name> [options]
```

删除 profile 并移除其 shell alias。

| 参数 / 选项 | 描述 |
|-------------------|-------------|
| `<name>` | 要删除的 profile。 |
| `--yes`, `-y` | 跳过确认提示。 |

**示例：**

```bash
hermes profile delete mybot
hermes profile delete mybot --yes
```

:::warning
此操作将永久删除 profile 的整个目录，包括所有 config、memories、sessions 和 skills。无法删除当前活跃的 profile。
:::

## `hermes profile show`

```bash
hermes profile show <name>
```

显示 profile 的详细信息，包括其主目录、配置的模型、gateway 状态、skill 数量和配置文件状态。

此处显示的是 profile 的 Hermes 主目录，而非终端工作目录。终端命令从 `terminal.cwd` 启动（或在本地后端 `cwd: "."` 时从启动目录启动）。

| 参数 | 描述 |
|----------|-------------|
| `<name>` | 要查看的 profile。 |

**示例：**

```bash
$ hermes profile show work
Profile: work
Path:    ~/.hermes/profiles/work
Model:   anthropic/claude-sonnet-4 (anthropic)
Gateway: stopped
Skills:  12
.env:    exists
SOUL.md: exists
Alias:   ~/.local/bin/work
```

## `hermes profile alias`

```bash
hermes profile alias <name> [options]
```

重新生成位于 `~/.local/bin/<name>` 的 shell alias 脚本。适用于 alias 被意外删除，或移动 Hermes 安装目录后需要更新的情况。

| 参数 / 选项 | 描述 |
|-------------------|-------------|
| `<name>` | 要创建/更新 alias 的 profile。 |
| `--remove` | 移除 wrapper 脚本而非创建。 |
| `--name <alias>` | 自定义 alias 名称（默认：profile 名称）。 |

**示例：**

```bash
hermes profile alias work
# 创建/更新 ~/.local/bin/work

hermes profile alias work --name mywork
# 创建 ~/.local/bin/mywork

hermes profile alias work --remove
# 移除 wrapper 脚本
```

## `hermes profile rename`

```bash
hermes profile rename <old-name> <new-name>
```

重命名 profile，同时更新目录和 shell alias。

| 参数 | 描述 |
|----------|-------------|
| `<old-name>` | 当前 profile 名称。 |
| `<new-name>` | 新 profile 名称。 |

**示例：**

```bash
hermes profile rename mybot assistant
# ~/.hermes/profiles/mybot → ~/.hermes/profiles/assistant
# ~/.local/bin/mybot → ~/.local/bin/assistant
```

## `hermes profile export`

```bash
hermes profile export <name> [options]
```

将 profile 导出为压缩的 tar.gz 归档文件。

| 参数 / 选项 | 描述 |
|-------------------|-------------|
| `<name>` | 要导出的 profile。 |
| `-o`, `--output <path>` | 输出文件路径（默认：`<name>.tar.gz`）。 |

**示例：**

```bash
hermes profile export work
# 在当前目录创建 work.tar.gz

hermes profile export work -o ./work-2026-03-29.tar.gz
```

## `hermes profile import`

```bash
hermes profile import <archive> [options]
```

从 tar.gz 归档文件导入 profile。

| 参数 / 选项 | 描述 |
|-------------------|-------------|
| `<archive>` | 要导入的 tar.gz 归档文件路径。 |
| `--name <name>` | 导入后的 profile 名称（默认：从归档文件推断）。 |

**示例：**

```bash
hermes profile import ./work-2026-03-29.tar.gz
# 从归档文件推断 profile 名称

hermes profile import ./work-2026-03-29.tar.gz --name work-restored
```

## 发行版命令

:::tip
**初次接触发行版？** 请先阅读 [Profile 发行版用户指南](../user-guide/profile-distributions.md) — 其中通过完整示例介绍了原因、时机和方法。以下章节是在你已知需求时使用的简明 CLI 参考。
:::

发行版将 profile 转变为可共享、有版本的制品，以 **git 仓库**形式发布。接收方只需一条命令即可安装发行版，并可在不影响本地 memories、sessions 或凭据的情况下就地更新。

`auth.json` 和 `.env` 永远不属于发行版的一部分 — 它们保留在安装用户的机器上。

接收方的用户数据（memories、sessions、auth、对 `.env` 的自有编辑）在初次安装和后续更新中始终得到保留。

:::info
`hermes profile export` / `import` 仍是在**本机进行 profile 本地备份和恢复**的正确命令。发行版（`install` / `update` / `info`）是独立概念：通过 git 分发 profile，供他人安装。
:::

### `hermes profile install`

```bash
hermes profile install <source> [--name <name>] [--alias] [--force] [--yes]
```

从 git URL 或本地目录安装 profile 发行版。

| 选项 | 描述 |
|--------|-------------|
| `<source>` | Git URL（`github.com/user/repo`、`https://...`、`git@...`、`ssh://`、`git://`）或包含 `distribution.yaml` 的本地目录根路径。 |
| `--name NAME` | 覆盖 manifest 中的 profile 名称。 |
| `--alias` | 同时创建 shell wrapper（例如 `telemetry` → `hermes -p telemetry`）。 |
| `--force` | 覆盖同名的已有 profile。用户数据仍会保留。 |
| `-y`, `--yes` | 跳过 manifest 预览确认提示。 |

安装程序会显示 manifest、列出所需的环境变量，并在询问确认前提示 cron 任务信息。所需环境变量会写入 `.env.EXAMPLE` 文件，复制为 `.env` 后填写即可。

**示例：**

```bash
# 从 GitHub 仓库安装（简写）
hermes profile install github.com/kyle/telemetry-distribution --alias

# 从完整 HTTPS git URL 安装
hermes profile install https://github.com/kyle/telemetry-distribution.git

# 从 SSH 安装
hermes profile install git@github.com:kyle/telemetry-distribution.git

# 开发时从本地目录安装
hermes profile install ./telemetry/
```

### `hermes profile update`

```bash
hermes profile update <name> [--force-config] [--yes]
```

从记录的来源重新克隆发行版并应用更新。发行版所有的文件（SOUL.md、skills/、cron/、mcp.json）会被覆盖；用户数据（memories、sessions、auth、.env）不会被修改。

默认保留 `config.yaml` 以保持本地覆盖设置。传入 `--force-config` 可将其重置为发行版附带的 config。

### `hermes profile info`

```bash
hermes profile info <name>
```

打印 profile 的发行版 manifest — 名称、版本、所需 Hermes 版本、作者、环境变量要求、来源 URL/路径，以及发行版最后一次 `install` 或 `update` 时记录的 `Installed:` 时间戳。适用于安装前检查共享 profile 的需求，以及发现"该 profile 已安装 6 个月未更新"等情况。

`hermes profile list` 也会在 `Distribution` 列中显示发行版名称和版本，`hermes profile show <name>` / `delete <name>` 会显示来源 URL，让你一眼看出哪些 profile 来自 git 仓库，哪些是本地创建的。

### 私有发行版

私有 git 仓库无需额外配置即可作为发行版来源 — 安装时会调用系统的 `git` 二进制文件，因此 shell 已配置的任何认证方式（SSH 密钥、`git credential` helper、GitHub CLI 存储的 HTTPS 凭据）均可透明生效。

```bash
# 使用 SSH 密钥，与普通 `git clone` 相同
hermes profile install git@github.com:your-org/internal-assistant.git

# 使用 git credential helper
hermes profile install https://github.com/your-org/internal-assistant.git
```

如果克隆时在终端交互式提示输入凭据，该提示会正常显示。请先按照对同一仓库执行 `git clone` 的方式配置好认证，再执行安装。

### 发行版 manifest（`distribution.yaml`）

每个发行版在其仓库根目录都有一个 `distribution.yaml`：

```yaml
name: telemetry
version: 0.1.0
description: "Compliance monitoring harness"
hermes_requires: ">=0.12.0"
author: "Your Name"
license: "MIT"
env_requires:
  - name: OPENAI_API_KEY
    description: "OpenAI API key"
    required: true
  - name: GRAPHITI_MCP_URL
    description: "Memory graph URL"
    required: false
    default: "http://127.0.0.1:8000/sse"
distribution_owned:   # optional; defaults to SOUL.md, config.yaml,
                      #   mcp.json, skills/, cron/, distribution.yaml
  - SOUL.md
  - skills/compliance/
  - cron/
```

`hermes_requires` 支持 `>=`、`<=`、`==`、`!=`、`>`、`<`，或裸版本号（视为 `>=`）。若当前 Hermes 版本不满足规格，安装将失败并给出明确错误。

`distribution_owned` 为可选项。若设置，更新时仅替换这些路径；profile 中的其他内容保持用户所有。若省略，则应用上述默认值。

### 发布发行版

编写发行版就是一次 git push：

1. 在你的 profile 目录中创建 `distribution.yaml`，至少包含 `name` 和 `version`。
2. 初始化 git 仓库（或使用已有仓库），推送到 GitHub / GitLab / 任何 Hermes 可克隆的托管平台。
3. 告知接收方运行 `hermes profile install <your-repo-url>`。

使用 git tag 进行版本化发布 — 克隆 `HEAD` 的接收方将获得最新状态，你也可以随时在 manifest 中更新 `version:`。

## `hermes -p` / `hermes --profile`

```bash
hermes -p <name> <command> [options]
hermes --profile <name> <command> [options]
```

全局标志，用于在不更改默认 profile 的情况下，在指定 profile 下运行任意 Hermes 命令。仅在该命令执行期间覆盖活跃 profile。

| 选项 | 描述 |
|--------|-------------|
| `-p <name>`, `--profile <name>` | 本次命令使用的 profile。 |

**示例：**

```bash
hermes -p work chat -q "Check the server status"
hermes --profile dev gateway start
hermes -p personal skills list
hermes -p work config edit
```

## `hermes completion`

```bash
hermes completion <shell>
```

生成 shell 补全脚本。包含对 profile 名称和 profile 子命令的补全。

| 参数 | 描述 |
|----------|-------------|
| `<shell>` | 要生成补全脚本的 shell：`bash`、`zsh` 或 `fish`。 |

**示例：**

```bash
# 安装补全脚本
hermes completion bash >> ~/.bashrc
hermes completion zsh >> ~/.zshrc
hermes completion fish > ~/.config/fish/completions/hermes.fish

# 重新加载 shell
source ~/.bashrc
```

安装后，Tab 补全适用于：
- `hermes profile <TAB>` — 子命令（list、use、create 等）
- `hermes profile use <TAB>` — profile 名称
- `hermes -p <TAB>` — profile 名称

## 另请参阅

- [Profiles 用户指南](../user-guide/profiles.md)
- [CLI 命令参考](./cli-commands.md)
- [FAQ — Profiles 章节](./faq.md#profiles)
---
sidebar_position: 7
---

# Profile 命令参考

本页面涵盖与 [Hermes profile](../user-guide/profiles.md) 相关的所有命令。一般 CLI 命令请参见 [CLI 命令参考](./cli-commands.md)。

## `hermes profile`

```bash
hermes profile <子命令>
```

管理 profile 的顶级命令。运行 `hermes profile` 不带子命令显示帮助。

| 子命令 | 描述 |
|------------|-------------|
| `list` | 列出所有 profile。 |
| `use` | 设置活动（默认）profile。 |
| `create` | 创建新 profile。 |
| `delete` | 删除 profile。 |
| `show` | 显示 profile 详情。 |
| `alias` | 重新生成 profile 的 shell 别名。 |
| `rename` | 重命名 profile。 |
| `export` | 将 profile 导出为 tar.gz 存档。 |
| `import` | 从 tar.gz 存档导入 profile。 |

## `hermes profile list`

```bash
hermes profile list
```

列出所有 profile。当前活动的 profile 用 `*` 标记。

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
hermes profile use <名称>
```

将 `<名称>` 设置为活动 profile。所有后续 `hermes` 命令（不带 `-p`）将使用此 profile。

| 参数 | 描述 |
|----------|-------------|
| `<名称>` | 要激活的 profile 名称。使用 `default` 返回基础 profile。 |

**示例：**

```bash
hermes profile use work
hermes profile use default
```

## `hermes profile create`

```bash
hermes profile create <名称> [选项]
```

创建新 profile。

| 参数 / 选项 | 描述 |
|-------------------|-------------|
| `<名称>` | 新 profile 的名称。必须是有效目录名（字母数字、连字符、下划线）。 |
| `--clone` | 从当前 profile 复制 `config.yaml`、`.env` 和 `SOUL.md`。 |
| `--clone-all` | 从当前 profile 复制所有内容（配置、记忆、技能、会话、状态）。 |
| `--clone-from <profile>` | 从特定 profile 克隆而不是当前 profile。与 `--clone` 或 `--clone-all` 一起使用。 |
| `--no-alias` | 跳过包装脚本创建。 |

创建 profile **不会**使该 profile 目录成为默认项目/工作区目录用于终端命令。如果你希望 profile 在特定项目中启动，在该 profile 的 `config.yaml` 中设置 `terminal.cwd`。

**示例：**

```bash
# 空白 profile — 需要完整设置
hermes profile create mybot

# 从当前 profile 仅克隆配置
hermes profile create work --clone

# 从当前 profile 克隆所有内容
hermes profile create backup --clone-all

# 从特定 profile 克隆配置
hermes profile create work2 --clone --clone-from work
```

## `hermes profile delete`

```bash
hermes profile delete <名称> [选项]
```

删除 profile 并移除其 shell 别名。

| 参数 / 选项 | 描述 |
|-------------------|-------------|
| `<名称>` | 要删除的 profile。 |
| `--yes`, `-y` | 跳过确认提示。 |

**示例：**

```bash
hermes profile delete mybot
hermes profile delete mybot --yes
```

:::warning
这会永久删除 profile 的整个目录，包括所有配置、记忆、会话和技能。无法删除当前活动的 profile。
:::

## `hermes profile show`

```bash
hermes profile show <名称>
```

显示 profile 详情，包括其主目录、配置的模型、网关状态、技能计数和配置文件状态。

这显示 profile 的 Hermes 主目录，不是终端工作目录。终端命令从 `terminal.cwd` 开始（本地后端的启动目录为 `cwd: "."`）。

| 参数 | 描述 |
|----------|-------------|
| `<名称>` | 要检查的 profile。 |

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
hermes profile alias <名称> [选项]
```

在 `~/.local/bin/<名称>` 重新生成 shell 别名脚本。如果别名被意外删除或需要更新 Hermes 安装后更新，这很有用。

| 参数 / 选项 | 描述 |
|-------------------|-------------|
| `<名称>` | 为其创建/更新别名的 profile。 |
| `--remove` | 移除包装脚本而不是创建。 |
| `--name <别名>` | 自定义别名名称（默认：profile 名称）。 |

**示例：**

```bash
hermes profile alias work
# 创建/更新 ~/.local/bin/work

hermes profile alias work --name mywork
# 创建 ~/.local/bin/mywork

hermes profile alias work --remove
# 移除包装脚本
```

## `hermes profile rename`

```bash
hermes profile rename <旧名称> <新名称>
```

重命名 profile。更新目录和 shell 别名。

| 参数 | 描述 |
|----------|-------------|
| `<旧名称>` | 当前 profile 名称。 |
| `<新名称>` | 新 profile 名称。 |

**示例：**

```bash
hermes profile rename mybot assistant
# ~/.hermes/profiles/mybot → ~/.hermes/profiles/assistant
# ~/.local/bin/mybot → ~/.local/bin/assistant
```

## `hermes profile export`

```bash
hermes profile export <名称> [选项]
```

将 profile 导出为压缩 tar.gz 存档。

| 参数 / 选项 | 描述 |
|-------------------|-------------|
| `<名称>` | 要导出的 profile。 |
| `-o`, `--output <路径>` | 输出文件路径（默认：`<名称>.tar.gz`）。 |

**示例：**

```bash
hermes profile export work
# 在当前目录创建 work.tar.gz

hermes profile export work -o ./work-2026-03-29.tar.gz
```

## `hermes profile import`

```bash
hermes profile import <存档> [选项]
```

从 tar.gz 存档导入 profile。

| 参数 / 选项 | 描述 |
|-------------------|-------------|
| `<存档>` | 要导入的 tar.gz 存档路径。 |
| `--name <名称>` | 导入 profile 的名称（默认：从存档推断）。 |

**示例：**

```bash
hermes profile import ./work-2026-03-29.tar.gz
# 从存档推断 profile 名称

hermes profile import ./work-2026-03-29.tar.gz --name work-restored
```

## `hermes -p` / `hermes --profile`

```bash
hermes -p <名称> <命令> [选项]
hermes --profile <名称> <命令> [选项]
```

全局标志用于在特定 profile 下运行任何 Hermes 命令而不更改固定默认值。这会覆盖命令持续时间的活动 profile。

| 选项 | 描述 |
|--------|-------------|
| `-p <名称>`, `--profile <名称>` | 此次命令使用的 profile。 |

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

生成 shell 补全脚本。包括 profile 名称和 profile 子命令的补全。

| 参数 | 描述 |
|----------|-------------|
| `<shell>` | 为其生成补全的 shell：`bash` 或 `zsh`。 |

**示例：**

```bash
# 安装补全
hermes completion bash >> ~/.bashrc
hermes completion zsh >> ~/.zshrc

# 重新加载 shell
source ~/.bashrc
```

安装后，tab 补全适用于：
- `hermes profile <TAB>` — 子命令（list、use、create 等）
- `hermes profile use <TAB>` — profile 名称
- `hermes -p <TAB>` — profile 名称

## 另请参见

- [Profile 用户指南](../user-guide/profiles.md)
- [CLI 命令参考](./cli-commands.md)
- [常见问题 — Profile 部分](./faq.md#profiles)

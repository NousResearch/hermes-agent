---
title: "Watchers — 使用水印去重轮询 RSS、JSON API 和 GitHub"
sidebar_label: "Watchers"
description: "使用水印去重轮询 RSS、JSON API 和 GitHub"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Watchers

使用水印去重轮询 RSS、JSON API 和 GitHub。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/devops/watchers` 安装 |
| 路径 | `optional-skills/devops/watchers` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos |
| 标签 | `cron`, `polling`, `rss`, `github`, `http`, `automation`, `monitoring` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Watchers

按固定间隔轮询（polling）外部数据源，仅对新条目作出响应。提供三个现成脚本及一个共享水印（watermark）辅助模块；可将其接入 cron 任务，也可从终端临时运行。

## 使用场景

- 用户希望监控 RSS/Atom feed 并在有新条目时收到通知
- 用户希望监控 GitHub 仓库的 issues / pulls / releases / commits
- 用户希望轮询任意 JSON 端点并在有新条目时收到通知
- 用户请求"为 X 创建一个 watcher"或"当 X 变化时通知我"

## 工作原理

一个 watcher 本质上是一个脚本，执行以下操作：

1. 从外部数据源获取数据
2. 与记录已处理 ID 的水印文件进行比对
3. 将新水印写回文件
4. 将新条目打印到 stdout（无变化则不输出）

以下三个脚本均实现了上述逻辑。agent 通过终端工具运行它们——来自 cron 任务、webhook 或交互式对话——并报告新内容。

## 现成脚本

安装 skill 后，三个脚本均位于 `$HERMES_HOME/skills/devops/watchers/scripts/`。每个脚本读取 `WATCHER_STATE_DIR`（默认为 `$HERMES_HOME/watcher-state/`）作为状态文件目录，以 `--name` 参数作为键名。

| 脚本 | 监控对象 | 去重键 |
|---|---|---|
| `watch_rss.py` | RSS 2.0 或 Atom feed URL | `<guid>` / `<id>` |
| `watch_http_json.py` | 任意返回对象列表的 JSON 端点 | 可配置的 id 字段 |
| `watch_github.py` | GitHub 仓库的 issues / pulls / releases / commits | `id` / `sha` |

三个脚本的共同特性：

- 首次运行记录基线——不会重放已有 feed 内容
- 水印为有界 ID 集合（最多 500 条），以限制内存占用
- 输出格式：每条条目为 `## <title>\n<url>\n\n<optional body>`
- 无新内容时 stdout 为空——调用方将此视为静默
- 获取出错时返回非零退出码

## 用法

直接从终端工具运行 watcher：

```bash
python $HERMES_HOME/skills/devops/watchers/scripts/watch_rss.py \
  --name hn --url https://news.ycombinator.com/rss --max 5
```

监控 GitHub 仓库（在 `~/.hermes/.env` 中设置 `GITHUB_TOKEN` 以避免匿名请求限制 60 次/小时）：

```bash
python $HERMES_HOME/skills/devops/watchers/scripts/watch_github.py \
  --name hermes-issues --repo NousResearch/hermes-agent --scope issues
```

轮询任意 JSON API：

```bash
python $HERMES_HOME/skills/devops/watchers/scripts/watch_http_json.py \
  --name api --url https://api.example.com/events \
  --id-field event_id --items-path data.events
```

## 接入 cron

向 agent 发送如下 prompt（提示词）以调度 cron 任务：

> 每 15 分钟运行一次 `watch_rss.py --name hn --url https://news.ycombinator.com/rss`。如果有输出，则汇总标题并推送；如果没有输出，则保持静默。

agent 在 cron 任务的 agent 循环中通过终端工具调用脚本，无需修改 cron 内置的 `--script` 标志。

## 状态文件

每个 watcher 将状态写入 `$HERMES_HOME/watcher-state/<name>.json`。查看状态：

```bash
cat $HERMES_HOME/watcher-state/hn.json
```

强制重放（下次运行视为首次轮询）：

```bash
rm $HERMES_HOME/watcher-state/hn.json
```

## 自定义 watcher

三个脚本使用相同的模板：加载水印、获取数据、差异比对、保存、输出。`scripts/_watermark.py` 是共享辅助模块；导入它即可免费获得原子写入、有界 ID 集合及首次运行基线功能。参考任意一个脚本，即可了解所需的样板代码有多少。

## 常见问题

1. **每次 tick 都打印"无新条目"的标题。** 调用方依赖 stdout 为空来判断静默。若在空 delta 时打印任何内容，将导致频道被刷屏。已提供的脚本已处理此问题；自定义脚本也必须如此。
2. **期望首次运行就输出条目。** 首次运行只记录基线，不会输出内容。如需初始摘要，可在首次运行后删除状态文件，或在自定义脚本中添加 `--prime-with-latest N` 标志。
3. **水印无限增长。** 共享辅助模块上限为 500 个 ID。对于高频更新的 feed 可适当提高；在存储受限的文件系统上可适当降低。
4. **状态目录位于 agent 沙箱无法写入的位置。** `$HERMES_HOME/watcher-state/` 始终可写。Docker/Modal 后端可能无法访问任意宿主机路径。
---
title: "Qmd"
sidebar_label: "Qmd"
description: "使用 qmd 在本地搜索个人知识库、笔记、文档和会议记录 — 一个集成 BM25、向量搜索和 LLM 重排序的混合检索引擎"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Qmd

使用 qmd 在本地搜索个人知识库、笔记、文档和会议记录 — 一个集成 BM25、向量搜索和 LLM 重排序的混合检索引擎。支持 CLI 和 MCP 集成。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/research/qmd` 安装 |
| 路径 | `optional-skills/research/qmd` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent + Teknium |
| 许可证 | MIT |
| 平台 | macos, linux |
| 标签 | `Search`, `Knowledge-Base`, `RAG`, `Notes`, `MCP`, `Local-AI` |
| 相关 skill | [`obsidian`](/user-guide/skills/bundled/note-taking/note-taking-obsidian), [`native-mcp`](/user-guide/skills/bundled/mcp/mcp-native-mcp), [`arxiv`](/user-guide/skills/bundled/research/research-arxiv) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# QMD — Query Markup Documents

本地设备上的个人知识库搜索引擎。可索引 markdown 笔记、会议记录、文档及任何基于文本的文件，并提供结合关键词匹配、语义理解和 LLM 重排序的混合搜索 — 全部在本地运行，无需云端依赖。

由 [Tobi Lütke](https://github.com/tobi/qmd) 创建。MIT 许可证。

## 使用场景

- 用户要求搜索其笔记、文档、知识库或会议记录
- 用户希望在大量 markdown/文本文件中查找内容
- 用户需要语义搜索（"查找关于 X 概念的笔记"），而非仅仅是关键词 grep
- 用户已设置 qmd 集合并希望查询
- 用户要求搭建本地知识库或文档搜索系统
- 关键词："search my notes"、"find in my docs"、"knowledge base"、"qmd"

## 前置条件

### Node.js >= 22（必需）

```bash
# 检查版本
node --version  # must be >= 22

# macOS — install or upgrade via Homebrew
brew install node@22

# Linux — use NodeSource or nvm
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs
# or with nvm:
nvm install 22 && nvm use 22
```

### SQLite 扩展支持（仅 macOS）

macOS 系统自带的 SQLite 不支持扩展加载。请通过 Homebrew 安装：

```bash
brew install sqlite
```

### 安装 qmd

```bash
npm install -g @tobilu/qmd
# or with Bun:
bun install -g @tobilu/qmd
```

首次运行会自动下载 3 个本地 GGUF 模型（共约 2GB）：

| 模型 | 用途 | 大小 |
|-------|---------|------|
| embeddinggemma-300M-Q8_0 | 向量 embedding（嵌入） | ~300MB |
| qwen3-reranker-0.6b-q8_0 | 结果重排序 | ~640MB |
| qmd-query-expansion-1.7B | 查询扩展 | ~1.1GB |

### 验证安装

```bash
qmd --version
qmd status
```

## 快速参考

| 命令 | 功能 | 速度 |
|---------|-------------|-------|
| `qmd search "query"` | BM25 关键词搜索（无需模型） | ~0.2s |
| `qmd vsearch "query"` | 语义向量搜索（1 个模型） | ~3s |
| `qmd query "query"` | 混合搜索 + 重排序（全部 3 个模型） | 热启动 ~2-3s，冷启动 ~19s |
| `qmd get <docid>` | 获取完整文档内容 | 即时 |
| `qmd multi-get "glob"` | 批量获取文件 | 即时 |
| `qmd collection add <path> --name <n>` | 将目录添加为集合 | 即时 |
| `qmd context add <path> "description"` | 添加上下文元数据以提升检索效果 | 即时 |
| `qmd embed` | 生成/更新向量 embedding | 不定 |
| `qmd status` | 显示索引健康状态和集合信息 | 即时 |
| `qmd mcp` | 启动 MCP 服务器（stdio） | 持久运行 |
| `qmd mcp --http --daemon` | 启动 MCP 服务器（HTTP，模型保持热启动） | 持久运行 |

## 设置流程

### 1. 添加集合

将 qmd 指向包含文档的目录：

```bash
# Add a notes directory
qmd collection add ~/notes --name notes

# Add project docs
qmd collection add ~/projects/myproject/docs --name project-docs

# Add meeting transcripts
qmd collection add ~/meetings --name meetings

# List all collections
qmd collection list
```

### 2. 添加上下文描述

上下文元数据帮助搜索引擎理解每个集合的内容，可显著提升检索质量：

```bash
qmd context add qmd://notes "Personal notes, ideas, and journal entries"
qmd context add qmd://project-docs "Technical documentation for the main project"
qmd context add qmd://meetings "Meeting transcripts and action items from team syncs"
```

### 3. 生成 Embedding

```bash
qmd embed
```

此命令处理所有集合中的所有文档并生成向量 embedding。添加新文档或集合后需重新运行。

### 4. 验证

```bash
qmd status   # shows index health, collection stats, model info
```

## 搜索模式

### 快速关键词搜索（BM25）

适用场景：精确词语、代码标识符、名称、已知短语。
无需加载模型 — 近乎即时返回结果。

```bash
qmd search "authentication middleware"
qmd search "handleError async"
```

### 语义向量搜索

适用场景：自然语言问题、概念性查询。
首次查询时加载 embedding 模型（约 3s）。

```bash
qmd vsearch "how does the rate limiter handle burst traffic"
qmd vsearch "ideas for improving onboarding flow"
```

### 混合搜索 + 重排序（最佳质量）

适用场景：对质量要求最高的重要查询。
使用全部 3 个模型 — 查询扩展、并行 BM25+向量搜索、重排序。

```bash
qmd query "what decisions were made about the database migration"
```

### 结构化多模式查询

在单次查询中组合不同搜索类型以提升精度：

```bash
# BM25 for exact term + vector for concept
qmd query $'lex: rate limiter\nvec: how does throttling work under load'

# With query expansion
qmd query $'expand: database migration plan\nlex: "schema change"'
```

### 查询语法（lex/BM25 模式）

| 语法 | 效果 | 示例 |
|--------|--------|---------|
| `term` | 前缀匹配 | `perf` 匹配 "performance" |
| `"phrase"` | 精确短语 | `"rate limiter"` |
| `-term` | 排除词语 | `performance -sports` |

### HyDE（假设文档 Embedding）

对于复杂主题，可描述你期望答案的样子：

```bash
qmd query $'hyde: The migration plan involves three phases. First, we add the new columns without dropping the old ones. Then we backfill data. Finally we cut over and remove legacy columns.'
```

### 限定集合范围

```bash
qmd search "query" --collection notes
qmd query "query" --collection project-docs
```

### 输出格式

```bash
qmd search "query" --json        # JSON output (best for parsing)
qmd search "query" --limit 5     # Limit results
qmd get "#abc123"                # Get by document ID
qmd get "path/to/file.md"       # Get by file path
qmd get "file.md:50" -l 100     # Get specific line range
qmd multi-get "journals/*.md" --json  # Batch retrieve by glob
```

## MCP 集成（推荐）

qmd 提供 MCP 服务器，可通过原生 MCP 客户端直接向 Hermes Agent 提供搜索工具。这是推荐的集成方式 — 配置完成后，agent 无需每次加载此 skill 即可自动获得 qmd 工具。

### 方案 A：Stdio 模式（简单）

在 `~/.hermes/config.yaml` 中添加：

```yaml
mcp_servers:
  qmd:
    command: "qmd"
    args: ["mcp"]
    timeout: 30
    connect_timeout: 45
```

此配置注册以下工具：`mcp_qmd_search`、`mcp_qmd_vsearch`、`mcp_qmd_deep_search`、`mcp_qmd_get`、`mcp_qmd_status`。

**权衡：** 模型在首次搜索调用时加载（冷启动约 19s），之后在会话期间保持热启动状态。偶尔使用时可接受。

### 方案 B：HTTP Daemon 模式（快速，重度使用推荐）

单独启动 qmd daemon — 它会将模型保持在内存中：

```bash
# Start daemon (persists across agent restarts)
qmd mcp --http --daemon

# Runs on http://localhost:8181 by default
```

然后配置 Hermes Agent 通过 HTTP 连接：

```yaml
mcp_servers:
  qmd:
    url: "http://localhost:8181/mcp"
    timeout: 30
```

**权衡：** 运行时占用约 2GB 内存，但每次查询都很快（约 2-3s）。适合频繁搜索的用户。

### 保持 Daemon 持续运行

#### macOS（launchd）

```bash
cat > ~/Library/LaunchAgents/com.qmd.daemon.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.qmd.daemon</string>
  <key>ProgramArguments</key>
  <array>
    <string>qmd</string>
    <string>mcp</string>
    <string>--http</string>
    <string>--daemon</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/tmp/qmd-daemon.log</string>
  <key>StandardErrorPath</key>
  <string>/tmp/qmd-daemon.log</string>
</dict>
</plist>
EOF

launchctl load ~/Library/LaunchAgents/com.qmd.daemon.plist
```

#### Linux（systemd 用户服务）

```bash
mkdir -p ~/.config/systemd/user

cat > ~/.config/systemd/user/qmd-daemon.service << 'EOF'
[Unit]
Description=QMD MCP Daemon
After=network.target

[Service]
ExecStart=qmd mcp --http --daemon
Restart=on-failure
RestartSec=10
Environment=PATH=/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now qmd-daemon
systemctl --user status qmd-daemon
```

### MCP 工具参考

连接后，以下工具以 `mcp_qmd_*` 形式可用：

| MCP 工具 | 对应命令 | 描述 |
|----------|---------|-------------|
| `mcp_qmd_search` | `qmd search` | BM25 关键词搜索 |
| `mcp_qmd_vsearch` | `qmd vsearch` | 语义向量搜索 |
| `mcp_qmd_deep_search` | `qmd query` | 混合搜索 + 重排序 |
| `mcp_qmd_get` | `qmd get` | 通过 ID 或路径获取文档 |
| `mcp_qmd_status` | `qmd status` | 索引健康状态和统计信息 |

MCP 工具接受结构化 JSON 查询以支持多模式搜索：

```json
{
  "searches": [
    {"type": "lex", "query": "authentication middleware"},
    {"type": "vec", "query": "how user login is verified"}
  ],
  "collections": ["project-docs"],
  "limit": 10
}
```

## CLI 用法（不使用 MCP）

未配置 MCP 时，直接通过终端使用 qmd：

```
terminal(command="qmd query 'what was decided about the API redesign' --json", timeout=30)
```

设置和管理任务始终使用终端：

```
terminal(command="qmd collection add ~/Documents/notes --name notes")
terminal(command="qmd context add qmd://notes 'Personal research notes and ideas'")
terminal(command="qmd embed")
terminal(command="qmd status")
```

## 搜索流水线工作原理

了解内部机制有助于选择合适的搜索模式：

1. **查询扩展** — 一个经过微调的 1.7B 模型生成 2 个备选查询。原始查询在融合中获得 2 倍权重。
2. **并行检索** — BM25（SQLite FTS5）和向量搜索跨所有查询变体并行运行。
3. **RRF 融合** — 倒数排名融合（k=60）合并结果。顶部排名加成：第 1 名 +0.05，第 2-3 名 +0.02。
4. **LLM 重排序** — qwen3-reranker 对前 30 个候选结果评分（0.0-1.0）。
5. **位置感知混合** — 排名 1-3：75% 检索 / 25% 重排序。排名 4-10：60/40。排名 11+：40/60（对长尾结果更信任重排序）。

**智能分块：** 文档在自然断点处分割（标题、代码块、空行），目标约 900 个 token，重叠率 15%。代码块不会在中间被截断。

## 最佳实践

1. **始终添加上下文描述** — `qmd context add` 可显著提升检索准确性。描述每个集合包含的内容。
2. **添加文档后重新 embed** — 向集合添加新文件后必须重新运行 `qmd embed`。
3. **速度优先用 `qmd search`** — 需要快速关键词查找（代码标识符、精确名称）时，BM25 即时响应且无需模型。
4. **质量优先用 `qmd query`** — 问题具有概念性或用户需要最佳结果时，使用混合搜索。
5. **优先使用 MCP 集成** — 配置完成后，agent 无需每次加载此 skill 即可获得原生工具。
6. **频繁用户使用 daemon 模式** — 如果用户经常搜索知识库，建议设置 HTTP daemon。
7. **结构化搜索中第一个查询获得 2 倍权重** — 组合 lex 和 vec 时，将最重要/最确定的查询放在首位。

## 故障排查

### "首次运行时模型正在下载"
正常现象 — qmd 首次使用时会自动下载约 2GB 的 GGUF 模型。
这是一次性操作。

### 冷启动延迟（约 19s）
模型未加载到内存时会出现此情况。解决方案：
- 使用 HTTP daemon 模式（`qmd mcp --http --daemon`）保持热启动
- 不需要模型时使用 `qmd search`（仅 BM25）
- MCP stdio 模式在首次搜索时加载模型，会话期间保持热启动

### macOS："unable to load extension"
安装 Homebrew SQLite：`brew install sqlite`
然后确保其在系统 SQLite 之前出现在 PATH 中。

### "未找到集合"
运行 `qmd collection add <path> --name <name>` 添加目录，
然后运行 `qmd embed` 进行索引。

### Embedding 模型覆盖（CJK/多语言）
为非英语内容设置 `QMD_EMBED_MODEL` 环境变量：
```bash
export QMD_EMBED_MODEL="your-multilingual-model"
```

## 数据存储

- **索引与向量：** `~/.cache/qmd/index.sqlite`
- **模型：** 首次运行时自动下载到本地缓存
- **无云端依赖** — 全部在本地运行

## 参考资料

- [GitHub: tobi/qmd](https://github.com/tobi/qmd)
- [QMD 更新日志](https://github.com/tobi/qmd/blob/main/CHANGELOG.md)
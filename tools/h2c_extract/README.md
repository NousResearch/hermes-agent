# H2C Extract

从 Claude Code 和 Codex CLI 的会话文件中提取对话骨架，供 Hermes 消费。

## 原理

两个 AI CLI 都会在本地保存 `.jsonl` 格式的会话记录。这个脚本直接读取这些文件，过滤掉 97% 的噪音（工具输出、图片、系统提示），只保留用户问了什么、AI 回答了什么、改了哪些文件。

**不需要修改任何 CLI 配置，不需要 hook，不需要 AI 主动输出任何东西。**

## 前置条件

- Python 3.10+
- 装过 Claude Code（`~/.claude/` 目录存在）和/或 Codex CLI（`~/.codex/` 目录存在）

不需要安装任何依赖，纯标准库。

## 快速开始

```bash
# 查看状态（有多少会话待处理）
python3 ~/.hermes/skills/h2c-protocol/h2c_extract.py status

# 同步所有源（Claude Code + Codex）
python3 ~/.hermes/skills/h2c-protocol/h2c_extract.py sync

# 查看生成的骨架文件
ls ~/.hermes/inbox/
```

## 命令

```bash
# 同步所有源
python3 h2c_extract.py sync

# 只同步 Claude Code
python3 h2c_extract.py sync --source cc

# 只同步 Codex
python3 h2c_extract.py sync --source codex

# 强制重新处理所有文件（忽略已处理记录）
python3 h2c_extract.py sync --force

# 查看同步状态
python3 h2c_extract.py status
```

## 输出

骨架文件输出到 `~/.hermes/inbox/`，按来源区分：

```
~/.hermes/inbox/
├── 2026-04-07_cc_86fcda92.md   ← Claude Code 会话
├── 2026-04-07_cx_019d0e55.md   ← Codex 会话
└── ...
```

每个文件是一段可读的 markdown 对话摘要：

```markdown
---
session: 86fcda92
date: 2026-04-07
project: my-project
source: cc
tags: [code-change, debugging]
files_touched: [src/app.py, tests/test_app.py]
---

**User**: 这个数据加载器有 bug，xyz 格式解析出错
**CC**: 看了一下代码，问题在 parse_xyz() 函数的边界处理...
*[Tools: Edit(dataloader.py), Bash x2]*
**CC**: 修复了，主要改动是...
```

### 自动标签

| 标签 | 含义 |
|---|---|
| `code-change` | 会话中有文件编辑 |
| `discussion-only` | 纯讨论，没有工具调用（决策密度通常最高） |
| `debugging` | 包含 error/bug/fix 等关键词 |
| `multi-agent` | 使用了 Agent 工具（仅 CC） |

## 自动触发

### 方式一：macOS launchd（推荐）

创建 `~/Library/LaunchAgents/com.hermes.h2c-extract.plist`（将 `YOUR_USERNAME` 替换为实际用户名）：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.hermes.h2c-extract</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>/Users/YOUR_USERNAME/.hermes/skills/h2c-protocol/h2c_extract.py</string>
        <string>sync</string>
    </array>
    <key>StartInterval</key>
    <integer>600</integer>
    <key>StandardOutPath</key>
    <string>/Users/YOUR_USERNAME/.hermes/logs/h2c-extract.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/YOUR_USERNAME/.hermes/logs/h2c-extract.log</string>
</dict>
</plist>
```

```bash
# 加载
mkdir -p ~/.hermes/logs
launchctl load ~/Library/LaunchAgents/com.hermes.h2c-extract.plist

# 卸载
launchctl unload ~/Library/LaunchAgents/com.hermes.h2c-extract.plist

# 手动触发一次
launchctl start com.hermes.h2c-extract
```

### 方式二：Claude Code Stop Hook（即时同步）

在 `~/.claude/settings.json` 的 hooks 中添加：

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 ~/.hermes/skills/h2c-protocol/h2c_extract.py sync --source cc"
          }
        ]
      }
    ]
  }
}
```

每次 CC 会话结束时自动同步。配合 launchd 使用，launchd 兜底 crash 和 Codex 的同步。

## Hermes 消费

Hermes 读取 `~/.hermes/inbox/` 下的骨架文件，处理完后移到 `~/.hermes/inbox-archive/`：

```bash
# Hermes 侧示例（伪代码）
for file in ~/.hermes/inbox/*.md:
    summary = hermes.summarize(file)
    hermes.update_memory(summary)
    mv file ~/.hermes/inbox-archive/
```

## 文件说明

```
~/.hermes/skills/h2c-protocol/
├── h2c_extract.py   ← 提取脚本（本工具）
├── DESIGN.md        ← 设计文档（架构决策和技术规格）
├── RFC-001.md       ← 场景 1 协议（Hermes 指挥 CC，与本工具无关）
└── README.md        ← 本文件

~/.hermes/
├── inbox/           ← 输出目录（骨架文件）
├── inbox-archive/   ← 归档目录（Hermes 处理后移入）
└── h2c-state.json   ← 同步状态（记录已处理文件）
```

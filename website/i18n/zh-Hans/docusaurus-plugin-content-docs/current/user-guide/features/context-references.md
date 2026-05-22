---
sidebar_position: 9
sidebar_label: "上下文引用"
title: "上下文引用"
description: "用于将文件、文件夹、git 差异和 URL 直接附加到消息中的内联 @-语法"
---

# 上下文引用

输入 `@` 后跟引用，将内容直接注入到您的消息中。Hermes 会内联展开引用，并将内容附加在 `--- Attached Context ---` 部分下。

## 支持的引用

| 语法 | 描述 |
|--------|-------------|
| `@file:path/to/file.py` | 注入文件内容 |
| `@file:path/to/file.py:10-25` | 注入特定行范围（1 开始，包含） |
| `@folder:path/to/dir` | 注入目录树列表及文件元数据 |
| `@diff` | 注入 `git diff`（未暂存的工作树更改） |
| `@staged` | 注入 `git diff --staged`（已暂存更改） |
| `@git:5` | 注入最近 N 个提交及补丁（最多 10 个） |
| `@url:https://example.com` | 获取并注入网页内容 |

## 使用示例

```text
Review @file:src/main.py and suggest improvements

What changed? @diff

Compare @file:old_config.yaml and @file:new_config.yaml

What's in @folder:src/components?

Summarize this article @url:https://arxiv.org/abs/2301.00001
```

一条消息中可以使用多个引用：

```text
Check @file:main.py, and also @file:test.py.
```

尾部标点符号（`,`, `.`, `;`, `!`, `?`）会自动从引用值中剥离。

## CLI Tab 补全

在交互式 CLI 中，键入 `@` 会触发自动补全：

- `@` 显示所有引用类型（`@diff`, `@staged`, `@file:`, `@folder:`, `@git:`, `@url:`）
- `@file:` 和 `@folder:` 触发文件系统路径补全及文件大小元数据
- 裸 `@` 后跟部分文本会显示当前目录中匹配的文件和文件夹

## 行范围

`@file:` 引用支持行范围以进行精确的内容注入：

```text
@file:src/main.py:42        # 第 42 行
@file:src/main.py:10-25     # 第 10 到 25 行（包含）
```

行是 1 开始的。无效范围会被静默忽略（返回完整文件）。

## 大小限制

上下文引用受限制以防止超出模型的上下文窗口：

| 阈值 | 值 | 行为 |
|-----------|-------|----------|
| 软限制 | 上下文长度的 25% | 附加警告，继续展开 |
| 硬限制 | 上下文长度的 50% | 拒绝展开，原始消息保持不变返回 |
| 文件夹条目 | 最多 200 个文件 | 超出条目替换为 `- ...` |
| Git 提交 | 最多 10 个 | `@git:N` 限制在 [1, 10] 范围内 |

## 安全性

### 敏感路径阻止

这些路径始终被阻止通过 `@file:` 引用，以防止凭证暴露：

- SSH 密钥和配置：`~/.ssh/id_rsa`, `~/.ssh/id_ed25519`, `~/.ssh/authorized_keys`, `~/.ssh/config`
- Shell 配置文件：`~/.bashrc`, `~/.zshrc`, `~/.profile`, `~/.bash_profile`, `~/.zprofile`
- 凭证文件：`~/.netrc`, `~/.pgpass`, `~/.npmrc`, `~/.pypirc`
- Hermes 环境：`$HERMES_HOME/.env`

这些目录被完全阻止（内部的任何文件）：
`~/.ssh/`, `~/.aws/`, `~/.gnupg/`, `~/.kube/`, `$HERMES_HOME/skills/.hub/`

### 路径遍历保护

所有路径都相对于工作目录解析。解析到允许的工作区根目录之外的引用会被拒绝。

### 二进制文件检测

二进制文件通过 MIME 类型和空字节扫描检测。已知文本扩展名（`.py`, `.md`, `.json`, `.yaml`, `.toml`, `.js`, `.ts` 等）绕过基于 MIME 的检测。二进制文件会被拒绝并带有警告。

## 平台可用性

上下文引用主要是一个 **CLI 功能**。它们在交互式 CLI 中工作，其中 `@` 触发 tab 补全，引用在消息发送到智能体之前展开。

在 **消息平台**（Telegram, Discord 等）中，`@` 语法不会被网关展开 —— 消息按原样传递。智能体本身仍然可以通过 `read_file`, `search_files`, 和 `web_extract` 工具引用文件。

## 与上下文压缩的交互

当对话上下文被压缩时，展开的引用内容会包含在压缩摘要中。这意味着：

- 通过 `@file:` 注入的大文件内容会贡献到上下文使用
- 如果对话稍后压缩，文件内容会被摘要（不会逐字保留）
- 对于非常大的文件，考虑使用行范围（`@file:main.py:100-200`）仅注入相关部分

## 常见模式

```text
# 代码审查工作流
Review @diff and check for security issues

# 带上下文的调试
This test is failing. Here's the test @file:tests/test_auth.py
and the implementation @file:src/auth.py:50-80

# 项目探索
What does this project do? @folder:src @file:README.md

# 研究
Compare the approaches in @url:https://arxiv.org/abs/2301.00001
and @url:https://arxiv.org/abs/2301.00002
```

## 错误处理

无效引用会产生内联警告而非失败：

| 条件 | 行为 |
|-----------|----------|
| 文件未找到 | 警告："file not found" |
| 二进制文件 | 警告："binary files are not supported" |
| 文件夹未找到 | 警告："folder not found" |
| Git 命令失败 | 带有 git stderr 的警告 |
| URL 返回无内容 | 警告："no content extracted" |
| 敏感路径 | 警告："path is a sensitive credential file" |
| 路径在工作区外 | 警告："path is outside the allowed workspace" |

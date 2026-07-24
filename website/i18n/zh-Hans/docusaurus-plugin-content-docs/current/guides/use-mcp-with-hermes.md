---
sidebar_position: 6
title: "在 Hermes 中使用 MCP"
description: "将 MCP 服务器连接到 Hermes Agent、过滤其工具并在实际工作流中安全使用的实践指南"
---

# 在 Hermes 中使用 MCP

本指南介绍如何在日常工作流中实际使用 Hermes Agent 的 MCP 功能。

如果功能页面解释的是 MCP 是什么，本指南则关注如何快速、安全地从中获取价值。

## 何时应该使用 MCP？

在以下情况下使用 MCP：
- 工具已以 MCP 形式存在，且你不想构建原生 Hermes 工具
- 你希望 Hermes 通过干净的 RPC 层操作本地或远程系统
- 你需要细粒度的按服务器暴露控制
- 你希望将 Hermes 连接到内部 API、数据库或公司系统，而无需修改 Hermes 核心

在以下情况下不要使用 MCP：
- 内置 Hermes 工具已能很好地完成该工作
- 服务器暴露了大量危险工具，而你没有准备好对其进行过滤
- 你只需要一个非常窄的集成，原生工具会更简单、更安全

## 心智模型

将 MCP 视为一个适配器层：

- Hermes 仍然是 agent
- MCP 服务器提供工具
- Hermes 在启动或重新加载时发现这些工具
- 模型可以像使用普通工具一样使用它们
- 你控制每个服务器有多少内容可见

最后一点很重要。良好的 MCP 使用不是"连接一切"，而是"以最小的有效范围连接正确的东西"。

## 第一步：安装 MCP 支持

如果你使用标准安装脚本安装了 Hermes，MCP 支持已包含在内（安装程序会运行 `uv pip install -e ".[all]"`）。

如果你在没有附加组件的情况下安装，需要单独添加 MCP：

```bash
cd ~/.hermes/hermes-agent
uv pip install -e ".[mcp]"
```

对于基于 npm 的服务器，请确保 Node.js 和 `npx` 可用。

对于许多 Python MCP 服务器，`uvx` 是一个不错的默认选择。

## 第二步：先添加一个服务器

从单个、安全的服务器开始。

示例：仅访问一个项目目录的文件系统。

```yaml
mcp_servers:
  project_fs:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/my-project"]
```

然后启动 Hermes：

```bash
hermes chat
```

现在提出一个具体问题：

```text
Inspect this project and summarize the repo layout.
```

## 第三步：验证 MCP 已加载

你可以通过以下几种方式验证 MCP：

- 配置后 Hermes 横幅/状态应显示 MCP 集成
- 询问 Hermes 当前有哪些可用工具
- 配置更改后使用 `/reload-mcp`
- 如果服务器连接失败，检查日志

一个实用的测试 prompt（提示词）：

```text
Tell me which MCP-backed tools are available right now.
```

## 第四步：立即开始过滤

如果服务器暴露了大量工具，不要等到以后再过滤。

### 示例：仅白名单你需要的内容

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [list_issues, create_issue, search_code]
```

对于敏感系统，这通常是最佳默认设置。

## WSL2：将 WSL 中的 Hermes 桥接到 Windows Chrome

以下是适用场景的实际配置：

- Hermes 在 WSL2 内运行
- 你想控制的浏览器是 Windows 上已登录的普通 Chrome
- 从 WSL 使用 `/browser connect` 不稳定或不可靠

在此配置中，Hermes **不**直接连接到 Chrome，而是：

- Hermes 在 WSL 中运行
- Hermes 启动一个本地 stdio MCP 服务器
- 该 MCP 服务器通过 Windows 互操作（`cmd.exe` 或 `powershell.exe`）启动
- MCP 服务器附加到你的实时 Windows Chrome 会话

心智模型：

```text
Hermes (WSL) -> MCP stdio bridge -> Windows Chrome
```

### 为什么此模式有用

- 你保留真实的 Windows 浏览器配置文件、Cookie 和登录状态
- Hermes 保持在其支持的 Unix 环境（WSL2）中
- 浏览器控制以 MCP 工具的形式暴露，而不依赖 Hermes 核心浏览器传输

### 推荐服务器

使用 `chrome-devtools-mcp`。

如果你的 Windows Chrome 已通过 `chrome://inspect/#remote-debugging` 启用了实时远程调试，在 WSL 中按如下方式添加：

```bash
hermes mcp add chrome-devtools-win --command cmd.exe --args /c npx -y chrome-devtools-mcp@latest --autoConnect --no-usage-statistics
```

保存服务器后：

```bash
hermes mcp test chrome-devtools-win
```

然后启动一个新的 Hermes 会话或运行：

```text
/reload-mcp
```

### 典型 prompt

加载后，Hermes 可以直接使用带 MCP 前缀的浏览器工具。例如：

```text
调用 MCP 工具 mcp_chrome_devtools_win_list_pages，列出当前浏览器标签页。
```

### 何时 `/browser connect` 不适用

如果 Hermes 在 WSL 中运行而 Chrome 在 Windows 上运行，即使 Chrome 已打开且可调试，`/browser connect` 也可能失败。

常见原因：

- WSL 无法访问 Chrome 向 Windows 工具暴露的同一主机本地端点
- 较新的 Chrome 实时调试流程与经典的 `ws://localhost:9222` 不同
- 从 Windows 端辅助工具（如 `chrome-devtools-mcp`）附加浏览器更容易

在这些情况下，将 `/browser connect` 用于同环境配置，使用 MCP 进行 WSL 到 Windows 的浏览器桥接。

### 已知问题

- 通过 MCP 使用 Windows stdio 可执行文件时，从 `/mnt/c/Users/<you>` 或 `/mnt/c/workspace/...` 等 Windows 挂载路径启动 Hermes。
- 如果从 `/root` 或 `/home/...` 启动 Hermes，Windows 可能在 MCP 服务器启动前发出 `UNC` 当前目录警告。
- 如果 `chrome-devtools-mcp --autoConnect` 在枚举页面时超时，请减少 Chrome 中的后台/冻结标签页并重试。

### 示例：黑名单危险操作

```yaml
mcp_servers:
  stripe:
    url: "https://mcp.stripe.com"
    headers:
      Authorization: "Bearer ***"
    tools:
      exclude: [delete_customer, refund_payment]
```

### 示例：同时禁用实用工具包装器

```yaml
mcp_servers:
  docs:
    url: "https://mcp.docs.example.com"
    tools:
      prompts: false
      resources: false
```

## 过滤实际影响什么？

Hermes 中 MCP 暴露的功能分为两类：

1. 服务器原生 MCP 工具
- 通过以下方式过滤：
  - `tools.include`
  - `tools.exclude`

2. Hermes 添加的实用工具包装器
- 通过以下方式过滤：
  - `tools.resources`
  - `tools.prompts`

### 你可能看到的实用工具包装器

Resources（资源）：
- `list_resources`
- `read_resource`

Prompts（提示词）：
- `list_prompts`
- `get_prompt`

这些包装器仅在以下情况下出现：
- 你的配置允许它们，且
- MCP 服务器会话实际支持这些能力

因此，如果服务器不支持 resources/prompts，Hermes 不会假装它支持。

## 常见模式

### 模式 1：本地项目助手

当你希望 Hermes 在有界工作区内推理时，使用 MCP 连接仓库本地的文件系统或 git 服务器。

```yaml
mcp_servers:
  fs:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/project"]

  git:
    command: "uvx"
    args: ["mcp-server-git", "--repository", "/home/user/project"]
```

好的 prompt：

```text
Review the project structure and identify where configuration lives.
```

```text
Check the local git state and summarize what changed recently.
```

### 模式 2：GitHub 分类助手

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [list_issues, create_issue, update_issue, search_code]
      prompts: false
      resources: false
```

好的 prompt：

```text
List open issues about MCP, cluster them by theme, and draft a high-quality issue for the most common bug.
```

```text
Search the repo for uses of _discover_and_register_server and explain how MCP tools are registered.
```

### 模式 3：内部 API 助手

```yaml
mcp_servers:
  internal_api:
    url: "https://mcp.internal.example.com"
    headers:
      Authorization: "Bearer ***"
    tools:
      include: [list_customers, get_customer, list_invoices]
      resources: false
      prompts: false
```

好的 prompt：

```text
Look up customer ACME Corp and summarize recent invoice activity.
```

在这类场景中，严格的白名单远优于排除列表。

### 模式 4：文档/知识服务器

某些 MCP 服务器暴露的 prompts 或 resources 更像是共享知识资产，而非直接操作。

```yaml
mcp_servers:
  docs:
    url: "https://mcp.docs.example.com"
    tools:
      prompts: true
      resources: true
```

好的 prompt：

```text
List available MCP resources from the docs server, then read the onboarding guide and summarize it.
```

```text
List prompts exposed by the docs server and tell me which ones would help with incident response.
```

## 教程：带过滤的端到端配置

以下是一个实际的渐进式流程。

### 阶段 1：使用严格白名单添加 GitHub MCP

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [list_issues, create_issue, search_code]
      prompts: false
      resources: false
```

启动 Hermes 并询问：

```text
Search the codebase for references to MCP and summarize the main integration points.
```

### 阶段 2：仅在需要时扩展

如果之后还需要更新 issue：

```yaml
tools:
  include: [list_issues, create_issue, update_issue, search_code]
```

然后重新加载：

```text
/reload-mcp
```

### 阶段 3：添加具有不同策略的第二个服务器

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [list_issues, create_issue, update_issue, search_code]
      prompts: false
      resources: false

  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/project"]
```

现在 Hermes 可以组合使用它们：

```text
Inspect the local project files, then create a GitHub issue summarizing the bug you find.
```

MCP 好用的地方就在这里：无需修改 Hermes 核心即可实现多系统工作流。

## 安全使用建议

### 对危险系统优先使用白名单

对于任何涉及财务、面向客户或具有破坏性的系统：
- 使用 `tools.include`
- 从尽可能小的集合开始

### 禁用未使用的实用工具

如果你不希望模型浏览服务器提供的 resources/prompts，请将其关闭：

```yaml
tools:
  resources: false
  prompts: false
```

### 保持服务器范围狭窄

示例：
- 文件系统服务器根目录指向一个项目目录，而非整个主目录
- git 服务器指向一个仓库
- 内部 API 服务器默认以读取为主的工具暴露

### 配置更改后重新加载

```text
/reload-mcp
```

在更改以下内容后执行此操作：
- include/exclude 列表
- enabled 标志
- resources/prompts 开关
- 认证 header / env

## 按症状排查问题

### "服务器已连接，但我期望的工具不见了"

可能原因：
- 被 `tools.include` 过滤
- 被 `tools.exclude` 排除
- 实用工具包装器通过 `resources: false` 或 `prompts: false` 禁用
- 服务器实际上不支持 resources/prompts

### "服务器已配置，但什么都没加载"

检查：
- 配置中是否遗留了 `enabled: false`
- 命令/运行时是否存在（`npx`、`uvx` 等）
- HTTP 端点是否可达
- 认证 env 或 header 是否正确

### "为什么我看到的工具比 MCP 服务器公告的少？"

因为 Hermes 现在遵守你的按服务器策略和能力感知注册。这是预期行为，通常也是期望的结果。

### "如何在不删除配置的情况下移除 MCP 服务器？"

使用：

```yaml
enabled: false
```

这会保留配置，但阻止连接和注册。

## 推荐的首批 MCP 配置

适合大多数用户的首选服务器：
- filesystem
- git
- GitHub
- fetch / 文档 MCP 服务器
- 一个范围窄的内部 API

不适合作为首选的服务器：
- 具有大量破坏性操作且未经过滤的大型业务系统
- 任何你不够了解、无法加以约束的系统

## 相关文档

- [MCP（模型上下文协议）](/user-guide/features/mcp)
- [FAQ](/reference/faq)
- [斜杠命令](/reference/slash-commands)
---
sidebar_position: 6
title: "在 Hermes 中使用 MCP"
description: "将 MCP 服务器连接到 Hermes Agent、过滤工具以及在真实工作流程中安全使用它们的实用指南"
---

# 在 Hermes 中使用 MCP

本指南展示如何在日常工作中实际使用 MCP 与 Hermes Agent。

如果功能页面解释了 MCP 是什么，本指南是关于如何快速安全地从它获取价值。

## 何时应该使用 MCP？

使用 MCP 当：
- 一个工具已经以 MCP 形式存在，你不想构建原生 Hermes 工具
- 你希望 Hermes 通过干净的 RPC 层对本地或远程系统进行操作
- 你想要精细的每个服务器暴露控制
- 你想将 Hermes 连接到内部 API、数据库或公司系统，而不修改 Hermes 核心

不使用 MCP 当：
- 内置 Hermes 工具已经很好地解决了工作
- 服务器暴露了大量危险工具面，而你没准备好过滤它
- 你只需要一个非常狭窄的集成，而原生工具会更简单更安全

## 心理模型

将 MCP 视为适配器层：

- Hermes 保持为智能体
- MCP 服务器贡献工具
- Hermes 在启动或重载时发现这些工具
- 模型可以像普通工具一样使用它们
- 你控制每个服务器暴露多少

最后一部分很重要。好的 MCP 使用不是"连接一切"。而是"连接正确的东西，用最小的有用表面"。

## 步骤 1：安装 MCP 支持

如果你用标准安装脚本安装了 Hermes，MCP 支持已包含（安装程序运行 `uv pip install -e ".[all]"`）。

如果你没有附加组件安装，需要单独添加 MCP：

```bash
cd ~/.hermes/hermes-agent
uv pip install -e ".[mcp]"
```

对于基于 npm 的服务器，确保 Node.js 和 `npx` 可用。

对于许多 Python MCP 服务器，`uvx` 是一个不错的默认选择。

## 步骤 2：首先添加一个服务器

从一个单一、安全的服务器开始。

示例：仅限一个项目目录的文件系统访问。

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

现在问一些具体的问题：

```text
检查这个项目并总结仓库布局。
```

## 步骤 3：验证 MCP 已加载

你可以通过几种方式验证 MCP：

- Hermes 横幅/状态应该在配置时显示 MCP 集成
- 问 Hermes 它有什么可用工具
- 配置更改后使用 `/reload-mcp`
- 检查服务器是否连接失败的日志

一个实用的测试提示：

```text
告诉我现在有哪些 MCP 支持的工具可用。
```

## 步骤 4：立即开始过滤

如果服务器暴露了很多工具，不要等到以后。

### 示例：白名单仅你想要的内容

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

对于敏感系统，这通常是最好的默认。

## WSL2：将 WSL 中的 Hermes 桥接到 Windows Chrome

这是以下情况的实用设置：

- Hermes 在 WSL2 内运行
- 你想控制的浏览器是你正常的已登录 Windows Chrome
- `/browser connect` 从 WSL 使用很别扭或不可靠

在此设置中，Hermes **不**直接连接到 Chrome。相反：

- Hermes 在 WSL 中运行
- Hermes 启动本地 stdio MCP 服务器
- 该 MCP 服务器通过 Windows 互操作（`cmd.exe` 或 `powershell.exe`）启动
- MCP 服务器附加到你活动的 Windows Chrome 会话

心理模型：

```text
Hermes (WSL) -> MCP stdio 桥接 -> Windows Chrome
```

### 为什么此模式有用

- 你保留真实的 Windows 浏览器配置、cookies 和登录
- Hermes 保持在支持的 Unix 环境（WSL2）中
- 浏览器控制作为 MCP 工具暴露，而不依赖 Hermes 核心浏览器传输

### 推荐服务器

使用 `chrome-devtools-mcp`。

如果你的 Windows Chrome 已经从 `chrome://inspect/#remote-debugging` 启用了活动远程调试，从 WSL 这样添加：

```bash
hermes mcp add chrome-devtools-win --command cmd.exe --args /c "npx -y chrome-devtools-mcp@latest --autoConnect --no-usage-statistics"
```

保存服务器后：

```bash
hermes mcp test chrome-devtools-win
```

然后开启新的 Hermes 会话或运行：

```text
/reload-mcp
```

### 典型提示

加载后，Hermes 可以直接使用 MCP 前缀的浏览器工具。例如：

```text
调用 MCP 工具 mcp_chrome_devtools_win_list_pages，列出当前浏览器标签页。
```

### 何时 `/browser connect` 是错误的工具

如果 Hermes 在 WSL 运行而 Chrome 在 Windows 上运行，`/browser connect` 可能失败，即使 Chrome 打开且可调试。

常见原因：

- WSL 无法到达 Chrome 向 Windows 工具暴露的同一主机本地端点
- 较新的 Chrome 活动调试流程与经典 `ws://localhost:9222` 不同
- 浏览器更容易通过 `chrome-devtools-mcp` 从 Windows 端助手附加

在这些情况下，对同环境设置保留 `/browser connect`，对 WSL 到 Windows 浏览器桥接使用 MCP。

### 已知陷阱

- 从 Windows 挂载路径（如 `/mnt/c/Users/<you>` 或 `/mnt/c/workspace/...`）启动 Hermes，当通过 MCP 使用 Windows stdio 可执行文件时。
- 如果你从 `/root` 或 `/home/...` 启动 Hermes，Windows 可能在 MCP 服务器启动前发出 `UNC` 当前目录警告。
- 如果 `chrome-devtools-mcp --autoConnect` 在枚举页面时超时，减少 Chrome 中的后台/冻结标签页并重试。

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

### 示例：也禁用实用工具包装器

```yaml
mcp_servers:
  docs:
    url: "https://mcp.docs.example.com"
    tools:
      prompts: false
      resources: false
```

## 过滤实际影响什么？

Hermes 中有两类 MCP 暴露功能：

1. 服务器原生 MCP 工具
- 用以下方式过滤：
  - `tools.include`
  - `tools.exclude`

2. Hermes 添加的实用工具包装器
- 用以下方式过滤：
  - `tools.resources`
  - `tools.prompts`

### 你可能看到的实用工具包装器

资源：
- `list_resources`
- `read_resource`

提示：
- `list_prompts`
- `get_prompt`

这些包装器仅在以下情况下出现：
- 你的配置允许它们，且
- MCP 服务器会话实际支持这些能力

所以如果服务器没有资源/提示，Hermes 不会假装它有。

## 常见模式

### 模式 1：本地项目助手

当你想让 Hermes 对有界工作空间进行推理时，对仓库本地文件系统或 git 服务器使用 MCP。

```yaml
mcp_servers:
  fs:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/project"]

  git:
    command: "uvx"
    args: ["mcp-server-git", "--repository", "/home/user/project"]
```

好的提示：

```text
审查项目结构并识别配置所在位置。
```

```text
检查本地 git 状态并总结最近更改了什么。
```

### 模式 2：GitHub 分流助手

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

好的提示：

```text
列出关于 MCP 的开放 issues，按主题聚类，并为最常见的 bug 起草一个高质量 issue。
```

```text
在仓库中搜索 _discover_and_register_server 的使用，并解释 MCP 工具是如何注册的。
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

好的提示：

```text
查找客户 ACME Corp 并总结最近的发票活动。
```

这正是严格白名单比黑名单好得多的地方。

### 模式 4：文档/知识服务器

某些 MCP 服务器暴露的提示或资源更像共享知识资产，而非直接操作。

```yaml
mcp_servers:
  docs:
    url: "https://mcp.docs.example.com"
    tools:
      prompts: true
      resources: true
```

好的提示：

```text
列出文档服务器中可用的 MCP 资源，然后阅读入职指南并总结。
```

```text
列出文档服务器暴露的提示，告诉我哪些有助于事件响应。
```

## 教程：带过滤的端到端设置

这里是实用的进展。

### 阶段 1：添加 GitHub MCP 并设置严格白名单

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

启动 Hermes 并问：

```text
搜索代码库中对 MCP 的引用，并总结主要集成点。
```

### 阶段 2：仅在需要时扩展

如果你之后也需要 issue 更新：

```yaml
tools:
  include: [list_issues, create_issue, update_issue, search_code]
```

然后重载：

```text
/reload-mcp
```

### 阶段 3：添加第二个服务器并设置不同策略

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

现在 Hermes 可以组合它们：

```text
检查本地项目文件，然后创建一个 GitHub issue 总结你发现的 bug。
```

这就是 MCP 变得强大的地方：多系统工作流程而不修改 Hermes 核心。

## 安全使用建议

### 对危险系统优先使用白名单

对于任何金融、客户面向或破坏性的：
- 使用 `tools.include`
- 从尽可能小的集合开始

### 禁用未使用的实用工具

如果你不想要模型浏览服务器提供的资源/提示，关闭它们：

```yaml
tools:
  resources: false
  prompts: false
```

### 保持服务器范围狭窄

示例：
- 文件系统服务器根目录到一个项目目录，而不是你整个主目录
- git 服务器指向一个仓库
- 内部 API 服务器默认只暴露读取为主的工具

### 配置更改后重载

```text
/reload-mcp
```

在更改后执行：
- include/exclude 列表
- 启用标志
- resources/prompts 切换
- 认证头/环境

## 按症状故障排除

### "服务器连接但我期望的工具缺失"

可能原因：
- 被 `tools.include` 过滤
- 被 `tools.exclude` 排除
- 通过 `resources: false` 或 `prompts: false` 禁用了实用工具包装器
- 服务器实际不支持 resources/prompts

### "服务器已配置但什么都没加载"

检查：
- `enabled: false` 没有留在配置中
- 命令/运行时存在（`npx`、`uvx` 等）
- HTTP 端点可访问
- 认证环境或头正确

### "为什么我看到的工具比 MCP 服务器宣传的少？"

因为 Hermes 现在尊重你的每个服务器策略和感知能力注册。这是预期的，通常是期望的。

### "如何在不删除配置的情况下移除 MCP 服务器？"

使用：

```yaml
enabled: false
```

这保留配置但阻止连接和注册。

## 推荐的首个 MCP 设置

对大多数用户好的首个服务器：
- filesystem
- git
- GitHub
- fetch / documentation MCP 服务器
- 一个狭窄的内部 API

不太好的首个服务器：
- 有大量破坏性操作且无过滤的巨大业务系统
- 你不够了解而无法约束的任何东西

## 相关文档

- [MCP (Model Context Protocol)](/docs/user-guide/features/mcp)
- [常见问题](/docs/reference/faq)
- [斜杠命令](/docs/reference/slash-commands)

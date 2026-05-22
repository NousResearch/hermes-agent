---
sidebar_position: 11
title: "ACP 编辑器集成"
description: "在 VS Code、Zed 和 JetBrains 等兼容 ACP 的编辑器中使用 Hermes Agent"
---

# ACP 编辑器集成

Hermes Agent 可以作为 ACP 服务器运行，让兼容 ACP 的编辑器通过 stdio 与 Hermes 通信并渲染：

- 聊天消息
- 工具活动
- 文件差异
- 终端命令
- 审批提示
- 流式思考 / 响应片段

当您希望 Hermes 表现得像一个编辑器原生的编码智能体，而不是独立的 CLI 或消息机器人时，ACP 是一个很好的选择。

## Hermes 在 ACP 模式下暴露的内容

Hermes 运行一个为编辑器工作流精心策划的 `hermes-acp` 工具集。它包括：

- 文件工具：`read_file`、`write_file`、`patch`、`search_files`
- 终端工具：`terminal`、`process`
- 网页/浏览器工具
- 记忆、待办、会话搜索
- 技能
- execute_code 和 delegate_task
- 视觉

它有意排除了不适合典型编辑器用户体验的内容，例如消息传递和 cronjob 管理。

## 安装

正常安装 Hermes，然后添加 ACP 额外组件：

```bash
pip install -e '.[acp]'
```

这会安装 `agent-client-protocol` 依赖并启用：

- `hermes acp`
- `hermes-acp`
- `python -m acp_adapter`

## 启动 ACP 服务器

以下任一方式都会以 ACP 模式启动 Hermes：

```bash
hermes acp
```

```bash
hermes-acp
```

```bash
python -m acp_adapter
```

Hermes 记录到 stderr，因此 stdout 保留给 ACP JSON-RPC 流量。

## 编辑器设置

### VS Code

安装 [ACP Client](https://marketplace.visualstudio.com/items?itemName=formulahendry.acp-client) 扩展。

连接方式：

1. 从活动栏打开 ACP Client 面板。
2. 从内置智能体列表中选择 **Hermes Agent**。
3. 连接并开始聊天。

如果您想手动定义 Hermes，请通过 VS Code 设置在 `acp.agents` 下添加：

```json
{
  "acp.agents": {
    "Hermes Agent": {
      "command": "hermes",
      "args": ["acp"]
    }
  }
}
```

### Zed

示例设置片段：

```json
{
  "agent_servers": {
    "hermes-agent": {
      "type": "custom",
      "command": "hermes",
      "args": ["acp"],
    },
  },
}
```

### JetBrains

使用兼容 ACP 的插件并将其指向：

```text
/path/to/hermes-agent/acp_registry
```

## 注册表清单

ACP 注册表清单位于：

```text
acp_registry/agent.json
```

它宣传一个基于命令的智能体，其启动命令为：

```text
hermes acp
```

## 配置和凭证

ACP 模式使用与 CLI 相同的 Hermes 配置：

- `~/.hermes/.env`
- `~/.hermes/config.yaml`
- `~/.hermes/skills/`
- `~/.hermes/state.db`

提供商解析使用 Hermes 的正常运行时解析器，因此 ACP 继承当前配置的提供商和凭证。

## 会话行为

ACP 会话由 ACP 适配器的内存会话管理器在服务器运行时跟踪。

每个会话存储：

- 会话 ID
- 工作目录
- 选定的模型
- 当前对话历史
- 取消事件

底层 `AIAgent` 仍然使用 Hermes 的正常持久化/日志路径，但 ACP `list/load/resume/fork` 的范围限定在当前运行的 ACP 服务器进程内。

## 工作目录行为

ACP 会话将编辑器的 cwd 绑定到 Hermes 任务 ID，因此文件和终端工具相对于编辑器工作区运行，而不是服务器进程 cwd。

## 审批

危险的终端命令可以作为审批提示路由回编辑器。ACP 审批选项比 CLI 流程更简单：

- 允许一次
- 始终允许
- 拒绝

超时或错误时，审批桥拒绝请求。

## 故障排除

### ACP 智能体未出现在编辑器中

检查：

- 编辑器指向正确的 `acp_registry/` 路径
- Hermes 已安装并在您的 PATH 上
- ACP 额外组件已安装（`pip install -e '.[acp]'`）

### ACP 启动但立即报错

尝试以下检查：

```bash
hermes doctor
hermes status
hermes acp
```

### 缺少凭证

ACP 模式没有自己的登录流程。它使用 Hermes 现有的提供商设置。使用以下方式配置凭证：

```bash
hermes model
```

或通过编辑 `~/.hermes/.env`。

## 另请参阅

- [ACP 内部机制](/developer-guide/acp-internals)
- [提供商运行时解析](/developer-guide/provider-runtime)
- [工具运行时](/developer-guide/tools-runtime)

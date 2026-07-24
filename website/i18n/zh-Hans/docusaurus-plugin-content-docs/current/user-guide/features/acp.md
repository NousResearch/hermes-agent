---
sidebar_position: 11
title: "ACP 编辑器集成"
description: "在 VS Code、Zed 和 JetBrains 等兼容 ACP 的编辑器中使用 Hermes Agent"
---

# ACP 编辑器集成

Hermes Agent 可作为 ACP 服务器运行，让兼容 ACP 的编辑器通过 stdio 与 Hermes 通信并渲染：

- 聊天消息
- 工具活动
- 文件差异
- 终端命令
- 审批 prompt（提示词）
- 流式思考 / 响应块

当你希望 Hermes 表现得像编辑器原生的编码 agent，而非独立 CLI 或消息机器人时，ACP 是合适的选择。

## Hermes 在 ACP 模式下暴露的内容

Hermes 使用专为编辑器工作流设计的精选 `hermes-acp` 工具集运行，包括：

- 文件工具：`read_file`、`write_file`、`patch`、`search_files`
- 终端工具：`terminal`、`process`
- 网页/浏览器工具
- 记忆、待办事项、会话搜索
- skills
- `execute_code` 和 `delegate_task`
- 视觉

它有意排除了不适合典型编辑器 UX 的功能，例如消息投递和 cronjob 管理。

## 安装

正常安装 Hermes 后，从安装检出目录添加 ACP 扩展：

```bash
cd ~/.hermes/hermes-agent && uv pip install -e '.[acp]'
```

这将安装 `agent-client-protocol` 依赖并启用：

- `hermes acp`
- `hermes-acp`
- `python -m acp_adapter`

## 启动 ACP 服务器

以下任意命令均可以 ACP 模式启动 Hermes：

```bash
hermes acp
```

```bash
hermes-acp
```

```bash
python -m acp_adapter
```

Hermes 将日志输出到 stderr，以保留 stdout 用于 ACP JSON-RPC 流量。

非交互式检查：

```bash
hermes acp --version
hermes acp --check
```

### 浏览器工具（可选）

浏览器工具（`browser_navigate`、`browser_click` 等）依赖 `agent-browser` npm 包和 Chromium，这些不包含在 Python wheel 中。通过以下命令安装：

```bash
hermes acp --setup-browser           # 交互式（下载约 400 MB 前会提示确认）
hermes acp --setup-browser --yes     # 非交互式接受下载
```

这是独立命令。终端认证流程（`hermes acp --setup`）在模型选择后也会将浏览器引导作为后续问题提供，因此大多数用户无需直接运行 `--setup-browser`。

具体操作：

- 若缺少 Node.js 22 LTS，将其安装到 `~/.hermes/node/`
- 将 `npm install -g agent-browser @askjo/camofox-browser` 安装到该前缀（无需 sudo — `npm` 的 `--prefix` 指向用户可写的 Hermes 管理 Node）
- 安装 Playwright Chromium，或在检测到系统 Chrome/Chromium 时使用已有版本

该引导过程是幂等的——重复运行速度很快，已完成的步骤会被跳过。

## 编辑器设置

### VS Code

安装 [ACP Client](https://marketplace.visualstudio.com/items?itemName=formulahendry.acp-client) 扩展。

连接步骤：

1. 从活动栏打开 ACP Client 面板。
2. 从内置 agent 列表中选择 **Hermes Agent**。
3. 连接并开始聊天。

如需手动定义 Hermes，通过 VS Code 设置在 `acp.agents` 下添加：

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

在 Zed 设置中将 Hermes 配置为自定义 agent 服务器：

1. 打开 Agent 面板。
2. 使用以下配置添加自定义 agent 服务器：

```json
{
  "agent_servers": {
    "hermes-agent": {
      "type": "custom",
      "command": "hermes",
      "args": ["acp"]
    }
  }
}
```

3. 启动新的 Hermes 外部 agent 线程。

前提条件：

- 先通过 `hermes model` 配置 Hermes provider 凭据，或在 `~/.hermes/.env` / `~/.hermes/config.yaml` 中设置。

### JetBrains

使用兼容 ACP 的插件并将其指向 `hermes acp` 或 `hermes-acp`。

## 配置与凭据

ACP 模式使用与 CLI 相同的 Hermes 配置：

- `~/.hermes/.env`
- `~/.hermes/config.yaml`
- `~/.hermes/skills/`
- `~/.hermes/state.db`

Provider 解析使用 Hermes 的正常运行时解析器，因此 ACP 继承当前配置的 provider 和凭据。Hermes 还为首次运行的 ACP 客户端提供终端认证方法（`--setup`）；这将打开 Hermes 的交互式模型/provider 设置。

## 会话行为

ACP 会话在服务器运行期间由 ACP 适配器的内存会话管理器跟踪。

每个会话存储：

- 会话 ID
- 工作目录
- 已选模型
- 当前对话历史
- 取消事件

底层 `AIAgent` 仍使用 Hermes 的正常持久化/日志路径，但 ACP 的 `list/load/resume/fork` 仅限于当前运行的 ACP 服务器进程。

## 工作目录行为

ACP 会话将编辑器的 cwd 绑定到 Hermes 任务 ID，使文件和终端工具相对于编辑器工作区运行，而非服务器进程的 cwd。

## 审批

危险的终端命令可作为审批 prompt 路由回编辑器。ACP 审批选项比 CLI 流程更简单：

- 允许一次
- 始终允许
- 拒绝

超时或出错时，审批桥接会拒绝请求。

### 会话范围的编辑自动审批

ACP 在*允许一次*和*始终允许*之间提供第三层：**允许本次会话**。在编辑器的权限提示中选择此选项，会将审批记录在当前 ACP 会话内——该会话中所有后续匹配命令无需提示即可通过，但新的 ACP 会话（或重启编辑器）会重置状态，并在第一次时重新提示。

| 选项 | 编辑器标签 | 范围 | 重启后是否持久化 |
|---|---|---|---|
| `allow_once` | 允许一次 | 本次工具调用 | 否 |
| `allow_session` | 允许本次会话 | 本 ACP 会话中所有匹配调用 | 否——会话结束时清除 |
| `allow_always` | 始终允许 | 所有未来会话 | 是（写入 Hermes 永久允许列表） |
| `deny` | 拒绝 | 本次工具调用 | 否 |

`allow_session` 是编辑器工作流的正确默认选项——你在任务期间信任 agent，但不想授予长期允许列表条目。安全权衡很直接：范围越广，编辑器打断你的次数越少，行为异常的 agent（或 prompt 注入）在被发现前能造成的损害也越大。对不熟悉的命令从 `allow_once` 开始；在看到 agent 多次正确运行相同模式后升级为 `allow_session`；将 `allow_always` 保留给你永远信任的真正幂等命令（例如 `git status`）。

ACP 桥接将这些选项映射到 Hermes 的内部审批语义——`allow_always` 与 CLI 相同地写入永久允许列表条目，而 `allow_session` 仅影响当前 ACP 会话的进程内审批缓存。

## 故障排查

### ACP agent 未出现在编辑器中

检查：

- 对于手动/本地开发，验证自定义 `agent_servers` 命令是否指向 `hermes acp`。
- Hermes 已安装且在 PATH 中。
- ACP 扩展已安装（`cd ~/.hermes/hermes-agent && uv pip install -e '.[acp]'`）。

### ACP 启动后立即报错

尝试以下检查：

```bash
hermes acp --version
hermes acp --check
hermes doctor
hermes status
```

### 缺少凭据

ACP 模式使用 Hermes 现有的 provider 设置。通过以下方式配置凭据：

```bash
hermes model
```

或编辑 `~/.hermes/.env`。终端认证流程（`hermes acp --setup`）也可以触发交互式 provider/模型设置。

## 另请参阅

- [ACP 内部机制](../../developer-guide/acp-internals.md)
- [Provider 运行时解析](../../developer-guide/provider-runtime.md)
- [工具运行时](../../developer-guide/tools-runtime.md)
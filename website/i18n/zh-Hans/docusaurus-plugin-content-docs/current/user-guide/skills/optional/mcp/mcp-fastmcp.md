---
title: "Fastmcp — 使用 FastMCP 在 Python 中构建、测试、检查、安装和部署 MCP 服务器"
sidebar_label: "Fastmcp"
description: "使用 FastMCP 在 Python 中构建、测试、检查、安装和部署 MCP 服务器"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Fastmcp

使用 FastMCP 在 Python 中构建、测试、检查、安装和部署 MCP 服务器。适用于创建新的 MCP 服务器、将 API 或数据库封装为 MCP 工具、暴露资源或 prompt（提示词）、或为 Claude Code、Cursor 或 HTTP 部署准备 FastMCP 服务器。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/mcp/fastmcp` 安装 |
| 路径 | `optional-skills/mcp/fastmcp` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `MCP`, `FastMCP`, `Python`, `Tools`, `Resources`, `Prompts`, `Deployment` |
| 相关 skill | [`hermes-agent`](/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent), [`mcporter`](/user-guide/skills/optional/mcp/mcp-mcporter) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# FastMCP

使用 FastMCP 在 Python 中构建 MCP 服务器，在本地验证，安装到 MCP 客户端，并部署为 HTTP 端点。

## 使用时机

在以下任务中使用此 skill：

- 在 Python 中创建新的 MCP 服务器
- 将 API、数据库、CLI 或文件处理工作流封装为 MCP 工具
- 除工具外还需暴露资源或 prompt
- 在接入 Hermes 或其他客户端之前，使用 FastMCP CLI 对服务器进行冒烟测试
- 将服务器安装到 Claude Code、Claude Desktop、Cursor 或类似的 MCP 客户端
- 为 HTTP 部署准备 FastMCP 服务器仓库

若服务器已存在且只需连接到 Hermes，请使用 `native-mcp`。若目标是对现有 MCP 服务器进行临时 CLI 访问而非构建新服务器，请使用 `mcporter`。

## 前置条件

首先在工作环境中安装 FastMCP：

```bash
pip install fastmcp
fastmcp version
```

如需使用 API 模板，且 `httpx` 尚未安装，请先安装：

```bash
pip install httpx
```

## 包含文件

### 模板

- `templates/api_wrapper.py` - 支持 auth header 的 REST API 封装
- `templates/database_server.py` - 只读 SQLite 查询服务器
- `templates/file_processor.py` - 文本文件检查与搜索服务器

### 脚本

- `scripts/scaffold_fastmcp.py` - 复制入门模板并替换服务器名称占位符

### 参考资料

- `references/fastmcp-cli.md` - FastMCP CLI 工作流、安装目标及部署检查

## 工作流

### 1. 选择最小可行的服务器形态

优先选择最窄的有用接口：

- API 封装：从 1-3 个高价值端点开始，而非整个 API
- 数据库服务器：暴露只读自省能力和受约束的查询路径
- 文件处理器：暴露带有明确路径参数的确定性操作
- prompt/资源：仅在客户端需要可复用 prompt 模板或可发现文档时添加

优先选择接口精简、名称清晰、有 docstring 和 schema 的服务器，而非工具繁多但含义模糊的服务器。

### 2. 从模板脚手架生成

直接复制模板或使用脚手架辅助工具：

```bash
python ~/.hermes/skills/mcp/fastmcp/scripts/scaffold_fastmcp.py \
  --template api_wrapper \
  --name "Acme API" \
  --output ./acme_server.py
```

可用模板：

```bash
python ~/.hermes/skills/mcp/fastmcp/scripts/scaffold_fastmcp.py --list
```

如手动复制，请将 `__SERVER_NAME__` 替换为实际服务器名称。

### 3. 优先实现工具

在添加资源或 prompt 之前，先实现 `@mcp.tool` 函数。

工具设计规则：

- 为每个工具起一个具体的动词式名称
- 将 docstring 作为面向用户的工具描述
- 保持参数明确且有类型注解
- 尽可能返回结构化的 JSON 安全数据
- 尽早验证不安全的输入
- 第一版默认采用只读行为

良好的工具示例：

- `get_customer`
- `search_tickets`
- `describe_table`
- `summarize_text_file`

不佳的工具示例：

- `run`
- `process`
- `do_thing`

### 4. 仅在有帮助时添加资源和 Prompt

当客户端需要获取稳定的只读内容（如 schema、策略文档或生成的报告）时，添加 `@mcp.resource`。

当服务器应为已知工作流提供可复用 prompt 模板时，添加 `@mcp.prompt`。

不要将每个文档都变成 prompt。优先原则：

- 工具用于操作
- 资源用于数据/文档检索
- prompt 用于可复用的 LLM 指令

### 5. 集成前先测试服务器

使用 FastMCP CLI 进行本地验证：

```bash
fastmcp inspect acme_server.py:mcp
fastmcp list acme_server.py --json
fastmcp call acme_server.py search_resources query=router limit=5 --json
```

如需快速迭代调试，在本地运行服务器：

```bash
fastmcp run acme_server.py:mcp
```

如需在本地测试 HTTP transport：

```bash
fastmcp run acme_server.py:mcp --transport http --host 127.0.0.1 --port 8000
fastmcp list http://127.0.0.1:8000/mcp --json
fastmcp call http://127.0.0.1:8000/mcp search_resources query=router --json
```

在声明服务器可用之前，务必对每个新工具至少执行一次真实的 `fastmcp call`。

### 6. 本地验证通过后安装到客户端

FastMCP 可将服务器注册到支持的 MCP 客户端：

```bash
fastmcp install claude-code acme_server.py
fastmcp install claude-desktop acme_server.py
fastmcp install cursor acme_server.py -e .
```

使用 `fastmcp discover` 检查机器上已配置的命名 MCP 服务器。

若目标是集成到 Hermes，可选择：

- 使用 `native-mcp` skill，在 `~/.hermes/config.yaml` 中配置服务器，或
- 在接口稳定之前，在开发阶段继续使用 FastMCP CLI 命令

### 7. 本地契约稳定后再部署

对于托管部署，Prefect Horizon 是 FastMCP 文档中最直接的路径。部署前执行：

```bash
fastmcp inspect acme_server.py:mcp
```

确保仓库包含：

- 含有 FastMCP 服务器对象的 Python 文件
- `requirements.txt` 或 `pyproject.toml`
- 部署所需的环境变量文档

对于通用 HTTP 托管，先在本地验证 HTTP transport，然后在任何能暴露服务器端口的 Python 兼容平台上部署。

## 常见模式

### API 封装模式

适用于将 REST 或 HTTP API 暴露为 MCP 工具。

推荐的第一个切片：

- 一个读取路径
- 一个列表/搜索路径
- 可选的健康检查

实现注意事项：

- 将认证信息保存在环境变量中，不要硬编码
- 将请求逻辑集中在一个辅助函数中
- 以简洁的上下文暴露 API 错误
- 在返回前对上游不一致的 payload 进行规范化

从 `templates/api_wrapper.py` 开始。

### 数据库模式

适用于暴露安全的查询和自省能力。

推荐的第一个切片：

- `list_tables`
- `describe_table`
- 一个受约束的只读查询工具

实现注意事项：

- 默认使用只读数据库访问
- 早期版本拒绝非 `SELECT` SQL
- 限制返回行数
- 同时返回行数据和列名

从 `templates/database_server.py` 开始。

### 文件处理器模式

适用于服务器需要按需检查或转换文件的场景。

推荐的第一个切片：

- 汇总文件内容
- 在文件中搜索
- 提取确定性元数据

实现注意事项：

- 接受明确的文件路径
- 检查文件缺失和编码失败
- 限制预览和结果数量
- 除非需要特定外部工具，否则避免调用 shell

从 `templates/file_processor.py` 开始。

## 质量标准

在交付 FastMCP 服务器之前，验证以下所有内容：

- 服务器可以干净地导入
- `fastmcp inspect <file.py:mcp>` 成功
- `fastmcp list <server spec> --json` 成功
- 每个新工具至少有一次真实的 `fastmcp call`
- 环境变量已有文档说明
- 工具接口足够精简，无需猜测即可理解

## 故障排查

### FastMCP 命令缺失

在当前激活的环境中安装该包：

```bash
pip install fastmcp
fastmcp version
```

### `fastmcp inspect` 失败

检查：

- 文件导入时不存在导致崩溃的副作用
- FastMCP 实例在 `<file.py:object>` 中命名正确
- 模板所需的可选依赖已安装

### 工具在 Python 中正常但通过 CLI 不工作

运行：

```bash
fastmcp list server.py --json
fastmcp call server.py your_tool_name --json
```

这通常会暴露命名不匹配、缺少必填参数或返回值无法序列化等问题。

### Hermes 无法看到已部署的服务器

服务器构建部分可能正确，但 Hermes 配置有误。加载 `native-mcp` skill 并在 `~/.hermes/config.yaml` 中配置服务器，然后重启 Hermes。

## 参考资料

有关 CLI 详情、安装目标和部署检查，请阅读 `references/fastmcp-cli.md`。
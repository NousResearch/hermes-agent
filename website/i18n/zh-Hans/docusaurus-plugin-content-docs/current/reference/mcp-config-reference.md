---
sidebar_position: 8
title: "MCP 配置参考"
description: "Hermes Agent MCP 配置键、过滤语义与工具策略参考"
---

# MCP 配置参考

本页是 MCP 文档的参数速查版。

相关概念文档：

- [MCP（Model Context Protocol）](/user-guide/features/mcp)
- [将 MCP 与 Hermes 一起使用](/guides/use-mcp-with-hermes)

## 根配置结构

```yaml
mcp_servers:
  <server_name>:
    command: "..."      # stdio server
    args: []
    env: {}

    # 或 HTTP 模式
    url: "..."
    headers: {}

    enabled: true
    timeout: 120
    connect_timeout: 60
    tools:
      include: []
      exclude: []
      resources: true
      prompts: true
```

## 关键字段

| 键 | 类型 | 适用 | 说明 |
|---|---|---|---|
| `command` | string | stdio | 启动命令 |
| `args` | list | stdio | 启动参数 |
| `env` | mapping | stdio | 子进程环境变量 |
| `url` | string | HTTP | MCP 服务地址 |
| `headers` | mapping | HTTP | 请求头 |
| `enabled` | bool | both | 是否启用 |
| `timeout` | number | both | 工具调用超时 |
| `connect_timeout` | number | both | 首次连接超时 |
| `tools` | mapping | both | 工具过滤与策略 |
| `auth` | string | HTTP | 认证方式，可设 `oauth` |

## `tools` 过滤语义

| 键 | 类型 | 说明 |
|---|---|---|
| `include` | string 或 list | 白名单，只保留这些工具 |
| `exclude` | string 或 list | 黑名单，排除这些工具 |
| `resources` | bool-like | 开启/关闭 `list_resources`、`read_resource` |
| `prompts` | bool-like | 开启/关闭 `list_prompts`、`get_prompt` |

优先级规则：若 `include` 与 `exclude` 同时存在，`include` 优先。

## 重载配置

修改配置后，在会话中执行：

```text
/reload-mcp
```

## 工具命名

MCP 工具会注册为：

```text
mcp_<server>_<tool>
```

示例：`mcp_github_create_issue`。

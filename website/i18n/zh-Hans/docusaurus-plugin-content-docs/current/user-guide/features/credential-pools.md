---
title: 凭证池
description: "通过凭证池为同一提供商聚合多个 API 密钥或 OAuth 令牌，实现自动轮换和限流恢复。"
sidebar_label: 凭证池
sidebar_position: 9
---

# 凭证池

凭证池允许您为同一提供商注册多个 API 密钥或 OAuth 令牌。当一个密钥遇到限流或配额限额时，Hermes 会自动轮换到下一个健康的密钥——让会话保持运行而无需切换提供商。

这与[备用提供商](./fallback-providers.md)不同，后者切换到*完全不同*的提供商。凭证池是同提供商轮换；备用提供商是跨提供商故障转移。凭证池优先尝试——如果所有池密钥都耗尽，*然后*才激活备用提供商。

## 工作原理

```
您的请求
  → 从池中选取密钥（round_robin / least_used / fill_first / random）
  → 发送到提供商
  → 429 限流？
      → 重试同一密钥一次（瞬时抖动）
      → 第二次 429 → 轮换到下一个池密钥
      → 所有密钥耗尽 → fallback_model（不同提供商）
  → 402 计费错误？
      → 立即轮换到下一个池密钥（24小时冷却）
  → 401 认证过期？
      → 尝试刷新令牌（OAuth）
      → 刷新失败 → 轮换到下一个池密钥
  → 成功 → 正常继续
```

## 快速开始

如果您已在 `.env` 中设置了 API 密钥，Hermes 会自动将其发现为 1 密钥池。要从池化中获益，请添加更多密钥：

```bash
# 添加第二个 OpenRouter 密钥
hermes auth add openrouter --api-key sk-or-v1-your-second-key

# 添加第二个 Anthropic 密钥
hermes auth add anthropic --type api-key --api-key sk-ant-api03-your-second-key

# 添加 Anthropic OAuth 凭证（需要 Claude Max 计划 + 额外使用额度）
hermes auth add anthropic --type oauth
# 打开浏览器进行 OAuth 登录
```

查看您的池：

```bash
hermes auth list
```

输出：
```
openrouter (2 credentials):
  #1  OPENROUTER_API_KEY   api_key env:OPENROUTER_API_KEY ←
  #2  backup-key           api_key manual

anthropic (3 credentials):
  #1  hermes_pkce          oauth   hermes_pkce ←
  #2  claude_code          oauth   claude_code
  #3  ANTHROPIC_API_KEY    api_key env:ANTHROPIC_API_KEY
```

`←` 标记当前选中的凭证。

## 交互式管理

运行不带子命令的 `hermes auth` 进入交互式向导：

```bash
hermes auth
```

这会显示完整的池状态并提供菜单：

```
What would you like to do?
  1. Add a credential
  2. Remove a credential
  3. Reset cooldowns for a provider
  4. Set rotation strategy for a provider
  5. Exit
```

对于同时支持 API 密钥和 OAuth 的提供商（Anthropic、Nous、Codex），添加流程会询问类型：

```
anthropic supports both API keys and OAuth login.
  1. API key (paste a key from the provider dashboard)
  2. OAuth login (authenticate via browser)
Type [1/2]:
```

## CLI 命令

| 命令 | 描述 |
|---------|-------------|
| `hermes auth` | 交互式凭证池管理向导 |
| `hermes auth list` | 显示所有池和凭证 |
| `hermes auth list <provider>` | 显示特定提供商的池 |
| `hermes auth add <provider>` | 添加凭证（提示类型和密钥） |
| `hermes auth add <provider> --type api-key --api-key <key>` | 非交互式添加 API 密钥 |
| `hermes auth add <provider> --type oauth` | 通过浏览器登录添加 OAuth 凭证 |
| `hermes auth remove <provider> <index>` | 按 1-based 索引删除凭证 |
| `hermes auth reset <provider>` | 清除所有冷却/耗尽状态 |

## 轮换策略

通过 `hermes auth` → "Set rotation strategy" 或在 `config.yaml` 中配置：

```yaml
credential_pool_strategies:
  openrouter: round_robin
  anthropic: least_used
```

| 策略 | 行为 |
|----------|----------|
| `fill_first`（默认） | 使用第一个健康密钥直到耗尽，然后移到下一个 |
| `round_robin` | 均匀轮换密钥，每次选择后轮换 |
| `least_used` | 始终选择请求计数最低的密钥 |
| `random` | 在健康密钥中随机选择 |

## 错误恢复

凭证池对不同错误的处理方式：

| 错误 | 行为 | 冷却时间 |
|-------|----------|----------|
| **429 限流** | 重试同一密钥一次（瞬时）。第二次连续 429 轮换到下一个密钥 | 1 小时 |
| **402 计费/配额** | 立即轮换到下一个密钥 | 24 小时 |
| **401 认证过期** | 先尝试刷新 OAuth 令牌。只有刷新失败才轮换 | — |
| **所有密钥耗尽** | 如果配置了 `fallback_model`，则回退到该模型 | — |

每次成功的 API 调用都会重置 `has_retried_429` 标志，因此单个瞬时 429 不会触发轮换。

## 自定义端点池

自定义 OpenAI 兼容端点（Together.ai、RunPod、本地服务器）有各自的池，以 config.yaml 中 `custom_providers` 的端点名称作为池键。

当通过 `hermes model` 设置自定义端点时，它会自动生成名称，如 "Together.ai" 或 "Local (localhost:8080)"。此名称成为池键。

```bash
# 通过 hermes model 设置自定义端点后：
hermes auth list
# 显示：
#   Together.ai (1 credential):
#     #1  config key    api_key config:Together.ai ←

# 为同一端点添加第二个密钥：
hermes auth add Together.ai --api-key sk-together-second-key
```

自定义端点池存储在 `auth.json` 的 `credential_pool` 下，带有 `custom:` 前缀：

```json
{
  "credential_pool": {
    "openrouter": [...],
    "custom:together.ai": [...]
  }
}
```

## 自动发现

Hermes 在启动时从多个来源自动发现凭证并填充池：

| 来源 | 示例 | 自动填充？ |
|--------|---------|-------------|
| 环境变量 | `OPENROUTER_API_KEY`、`ANTHROPIC_API_KEY` | 是 |
| OAuth 令牌（auth.json） | Codex device code、Nous device code | 是 |
| Claude Code 凭证 | `~/.claude/.credentials.json` | 是（Anthropic） |
| Hermes PKCE OAuth | `~/.hermes/auth.json` | 是（Anthropic） |
| 自定义端点配置 | config.yaml 中的 `model.api_key` | 是（自定义端点） |
| 手动条目 | 通过 `hermes auth add` 添加 | 持久化到 auth.json |

自动填充的条目在每次池加载时更新——如果您删除环境变量，其池条目会自动修剪。通过 `hermes auth add` 添加的手动条目不会被自动修剪。

## 委托与子代理共享

当 agent 通过 `delegate_task` 生成子代理时，父代理的凭证池会自动与子代理共享：

- **同一提供商** — 子代理接收父代理的完整池，实现限流时的密钥轮换
- **不同提供商** — 子代理加载该提供商自己的池（如果已配置）
- **未配置池** — 子代理回退到继承的单个 API 密钥

这意味着子代理受益于与父代理相同的限流恢复能力，无需额外配置。任务级凭证租用确保子代理在并发轮换密钥时不会相互冲突。

## 线程安全

凭证池对所有状态变更（`select()`、`mark_exhausted_and_rotate()`、`try_refresh_current()`、`mark_used()`）使用线程锁。这确保网关同时处理多个聊天会话时的安全并发访问。

## 架构

完整数据流图参见仓库中的 [`docs/credential-pool-flow.excalidraw`](https://excalidraw.com/#json=2Ycqhqpi6f12E_3ITyiwh,c7u9jSt5BwrmiVzHGbm87g)。

凭证池集成在提供商解析层：

1. **`agent/credential_pool.py`** — 池管理器：存储、选择、轮换、冷却
2. **`hermes_cli/auth_commands.py`** — CLI 命令和交互式向导
3. **`hermes_cli/runtime_provider.py`** — 池感知的凭证解析
4. **`run_agent.py`** — 错误恢复：429/402/401 → 池轮换 → 备用

## 存储

池状态存储在 `~/.hermes/auth.json` 的 `credential_pool` 键下：

```json
{
  "version": 1,
  "credential_pool": {
    "openrouter": [
      {
        "id": "abc123",
        "label": "OPENROUTER_API_KEY",
        "auth_type": "api_key",
        "priority": 0,
        "source": "env:OPENROUTER_API_KEY",
        "access_token": "sk-or-v1-...",
        "last_status": "ok",
        "request_count": 142
      }
    ]
  },
}
```

策略存储在 `config.yaml`（不是 `auth.json`）：

```yaml
credential_pool_strategies:
  openrouter: round_robin
  anthropic: least_used
```

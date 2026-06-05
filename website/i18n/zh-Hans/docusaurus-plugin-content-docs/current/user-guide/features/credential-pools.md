---
title: 凭证池
description: 为每个提供商池化多个 API 密钥或 OAuth 令牌，实现自动轮换和速率限制恢复。
sidebar_label: 凭证池
sidebar_position: 9
---

# 凭证池

凭证池允许你为同一提供商注册多个 API 密钥或 OAuth 令牌。当某个密钥触达速率限制或计费配额时，Hermes 会自动轮换到下一个健康密钥——在不切换提供商的情况下保持会话持续运行。

这与[备用提供商](./fallback-providers.md)不同，后者会切换到*另一个*提供商。凭证池是同一提供商内的轮换；备用提供商是跨提供商的故障转移。池会优先尝试——如果池中所有密钥都耗尽，*才会*激活备用提供商。

## 工作原理

```
Your request
  → Pick key from pool (round_robin / least_used / fill_first / random)
  → Send to provider
  → 429 rate limit?
      → Plan/usage limit reached (e.g. ChatGPT/Codex "usage limit reached")?
          → Rotate to next pool key immediately (no retry — the cap won't clear on retry)
      → Generic / transient 429?
          → Retry same key once (transient blip)
          → Second 429 → rotate to next pool key
      → All keys exhausted → fallback_model (different provider)
  → 402 billing error?
      → Immediately rotate to next pool key (24h cooldown)
  → 401 auth expired?
      → Try refreshing the token (OAuth)
      → Refresh failed → rotate to next pool key
  → Success → continue normally
```

## 快速开始

如果你已在 `.env` 中设置了 API 密钥，Hermes 会自动将其识别为单密钥池。要充分利用池化功能，请添加更多密钥：

```bash
# Add a second OpenRouter key
hermes auth add openrouter --api-key sk-or-v1-your-second-key

# Add a second Anthropic key
hermes auth add anthropic --type api-key --api-key sk-ant-api03-your-second-key

# Add an Anthropic OAuth credential (requires Claude Max plan + extra usage credits)
hermes auth add anthropic --type oauth
# Opens browser for OAuth login
```

查看你的池：

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

不带子命令运行 `hermes auth` 以进入交互式向导：

```bash
hermes auth
```

这会显示完整的池状态并提供操作菜单：

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

| 命令 | 说明 |
|---------|-------------|
| `hermes auth` | 交互式池管理向导 |
| `hermes auth list` | 显示所有池和凭证 |
| `hermes auth list <provider>` | 显示指定提供商的池 |
| `hermes auth add <provider>` | 添加凭证（提示选择类型和密钥） |
| `hermes auth add <provider> --type api-key --api-key <key>` | 非交互式添加 API 密钥 |
| `hermes auth add <provider> --type oauth` | 通过浏览器登录添加 OAuth 凭证 |
| `hermes auth remove <provider> <index>` | 按从 1 开始的索引删除凭证 |
| `hermes auth reset <provider>` | 清除所有冷却时间/耗尽状态 |

## 轮换策略

通过 `hermes auth` → "Set rotation strategy" 配置，或在 `config.yaml` 中设置：

```yaml
credential_pool_strategies:
  openrouter: round_robin
  anthropic: least_used
```

| 策略 | 行为 |
|----------|----------|
| `fill_first`（默认） | 持续使用第一个健康密钥直至耗尽，然后切换到下一个 |
| `round_robin` | 均匀循环遍历所有密钥，每次选择后轮换 |
| `least_used` | 始终选择请求次数最少的密钥 |
| `random` | 在健康密钥中随机选择 |

## 错误恢复

池对不同错误的处理方式不同：

| 错误 | 行为 | 冷却时间 |
|-------|----------|----------|
| **429 速率限制** | 对同一密钥重试一次（瞬时错误）。连续第二次 429 则轮换到下一个密钥 | 1 小时 |
| **402 计费/配额** | 立即轮换到下一个密钥 | 24 小时 |
| **401 认证过期** | 先尝试刷新 OAuth 令牌。仅在刷新失败时才轮换 | — |
| **所有密钥耗尽** | 若已配置则转入 `fallback_model` | — |

`has_retried_429` 标志在每次成功的 API 调用后重置，因此单次瞬时 429 不会触发轮换。

## 自定义端点池

自定义 OpenAI 兼容端点（Together.ai、RunPod、本地服务器）拥有各自的池，以 `config.yaml` 中 `custom_providers` 的端点名称作为键。

通过 `hermes model` 设置自定义端点时，会自动生成类似 "Together.ai" 或 "Local (localhost:8080)" 的名称，该名称即成为池的键。

```bash
# After setting up a custom endpoint via hermes model:
hermes auth list
# Shows:
#   Together.ai (1 credential):
#     #1  config key    api_key config:Together.ai ←

# Add a second key for the same endpoint:
hermes auth add Together.ai --api-key sk-together-second-key
```

自定义端点池以 `custom:` 前缀存储在 `auth.json` 的 `credential_pool` 下：

```json
{
  "credential_pool": {
    "openrouter": [...],
    "custom:together.ai": [...]
  }
}
```

## 自动发现

Hermes 在启动时自动从多个来源发现凭证并初始化池：

| 来源 | 示例 | 自动初始化？ |
|--------|---------|-------------|
| 环境变量 | `OPENROUTER_API_KEY`、`ANTHROPIC_API_KEY` | 是 |
| OAuth 令牌（auth.json） | Codex device code、Nous device code | 是 |
| Claude Code 凭证 | `~/.claude/.credentials.json` | 是（Anthropic） |
| Hermes PKCE OAuth | `~/.hermes/auth.json` | 是（Anthropic） |
| 自定义端点配置 | `config.yaml` 中的 `model.api_key` | 是（自定义端点） |
| 手动条目 | 通过 `hermes auth add` 添加 | 持久化至 auth.json |

自动初始化的条目在每次池加载时更新——如果你删除了某个环境变量，其池条目会自动清除。通过 `hermes auth add` 添加的手动条目永远不会被自动清除。

## 委托与子代理共享

当代理通过 `delegate_task` 派生子代理时，父代理的凭证池会自动共享给子代理：

- **相同提供商** — 子代理接收父代理的完整池，在触达速率限制时可进行密钥轮换
- **不同提供商** — 子代理加载该提供商自己的池（如已配置）
- **未配置池** — 子代理回退到继承的单个 API 密钥

这意味着子代理无需额外配置即可获得与父代理相同的速率限制弹性。按任务的凭证租用机制确保子代理在并发轮换密钥时不会相互冲突。

## 线程安全

凭证池对所有状态变更操作（`select()`、`mark_exhausted_and_rotate()`、`try_refresh_current()`、`mark_used()`）使用线程锁，确保 gateway（网关）同时处理多个聊天会话时的并发访问安全。

## 架构

完整的数据流图请参见仓库中的 [`docs/credential-pool-flow.excalidraw`](https://excalidraw.com/#json=2Ycqhqpi6f12E_3ITyiwh,c7u9jSt5BwrmiVzHGbm87g)。

凭证池集成于提供商解析层：

1. **`agent/credential_pool.py`** — 池管理器：存储、选择、轮换、冷却时间
2. **`hermes_cli/auth_commands.py`** — CLI 命令和交互式向导
3. **`hermes_cli/runtime_provider.py`** — 感知池的凭证解析
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

策略存储在 `config.yaml` 中（而非 `auth.json`）：

```yaml
credential_pool_strategies:
  openrouter: round_robin
  anthropic: least_used
```
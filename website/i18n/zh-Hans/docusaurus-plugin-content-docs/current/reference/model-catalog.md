---
sidebar_position: 11
title: 模型目录
description: 远程托管的清单文件，驱动 OpenRouter 和 Nous Portal 的精选模型选择器列表。
---

# 模型目录

Hermes 从托管于文档站点旁的 JSON 清单中获取 **OpenRouter** 和 **Nous Portal** 的精选模型列表。这样维护者无需发布新的 `hermes-agent` 版本即可更新选择器列表。

当清单不可达时（离线、网络受阻、托管故障），Hermes 会静默回退到随 CLI 一同发布的仓库内置快照。清单永远不会导致选择器崩溃——最坏情况下，你看到的是与已安装版本捆绑的列表。

## 线上清单 URL

```
https://hermes-agent.nousresearch.com/docs/api/model-catalog.json
```

每次合并到 `main` 时，通过现有的 `deploy-site.yml` GitHub Pages 流水线发布。真实来源位于仓库的 `website/static/api/model-catalog.json`。

## Schema（模式）

```json
{
  "version": 1,
  "updated_at": "2026-04-25T22:00:00Z",
  "metadata": {},
  "providers": {
    "openrouter": {
      "metadata": {},
      "models": [
        {"id": "moonshotai/kimi-k2.6", "description": "recommended", "metadata": {}},
        {"id": "openai/gpt-5.4",       "description": ""}
      ]
    },
    "nous": {
      "metadata": {},
      "models": [
        {"id": "anthropic/claude-opus-4.7"},
        {"id": "moonshotai/kimi-k2.6"}
      ]
    }
  }
}
```

字段说明：

- **`version`** — 整数类型的 schema 版本号。未来的 schema 会递增此值；Hermes 拒绝处理版本号未知的清单，并回退到硬编码快照。
- **`metadata`** — 清单、provider 及模型级别的自由格式字典，支持任意键。Hermes 会忽略未知字段，因此你可以为条目添加注解（如 `"tier": "paid"`、`"tags": [...]` 等），无需协调 schema 变更。
- **`description`** — 仅限 OpenRouter。驱动选择器徽章文本（`"recommended"`、`"free"` 或空字符串）。Nous Portal 不使用此字段——免费层级的限制由 Portal 的定价端点实时决定。
- **定价和上下文长度**不在清单中。这些数据在获取时来自各 provider 的实时 API（`/v1/models` 端点、models.dev）。

## 获取行为

| 时机 | 行为 |
|---|---|
| `/model` 或 `hermes model` | 若磁盘缓存已过期则重新获取，否则使用缓存 |
| 磁盘缓存新鲜（< TTL） | 不发起网络请求 |
| 网络故障且有缓存 | 静默回退到缓存，输出一行日志 |
| 网络故障且无缓存 | 静默回退到仓库内置快照 |
| 清单未通过 schema 校验 | 视为不可达 |

缓存位置：`~/.hermes/cache/model_catalog.json`。

## 配置

```yaml
model_catalog:
  enabled: true
  url: https://hermes-agent.nousresearch.com/docs/api/model-catalog.json
  ttl_hours: 24
  providers: {}
```

将 `enabled` 设为 `false` 可完全禁用远程获取，始终使用仓库内置快照。

### 按 provider 覆盖 URL

第三方可使用相同 schema 自托管自己的精选列表。将某个 provider 指向自定义 URL：

```yaml
model_catalog:
  providers:
    openrouter:
      url: https://example.com/my-openrouter-curation.json
```

覆盖清单只需填充其关心的 provider 块，其他 provider 继续从主 URL 解析。

## 更新清单

维护者操作：

```bash
# 从仓库内硬编码列表重新生成（在编辑 hermes_cli/models.py 中的
# OPENROUTER_MODELS 或 _PROVIDER_MODELS["nous"] 后保持清单同步）。
python scripts/build_model_catalog.py
```

然后将 `website/static/api/model-catalog.json` 的变更提交 PR 到 `main`。文档站点在合并后自动部署，新清单将在几分钟内生效。

你也可以直接手动编辑 JSON，用于不适合放入仓库内置快照的细粒度元数据变更——生成脚本只是便捷工具，并非唯一的真实来源。
---
sidebar_position: 11
title: 模型目录
description: 驱动 OpenRouter 和 Nous Portal 精选模型选择器列表的远程托管清单。
---

# 模型目录

Hermes 从与文档站点一起托管的 JSON 清单中获取 **OpenRouter** 和 **Nous Portal** 的精选模型列表。这让维护者可以在不发布新 `hermes-agent` 版本的情况下更新选择器列表。

当清单无法访问（离线、网络被阻止、托管失败）时，Hermes 无声地回退到 CLI 附带的内置快照。清单永远不会破坏选择器 — 最坏情况下你看到的是与你安装版本捆绑的列表。

## 实时清单 URL

```
https://hermes-agent.nousresearch.com/docs/api/model-catalog.json
```

每次合并到 `main` 时通过现有的 `deploy-site.yml` GitHub Pages 管道发布。真相来源位于仓库 `website/static/api/model-catalog.json`。

## 模式

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

- **`version`** — 整数模式版本。未来模式会递增此值；Hermes 拒绝它不理解的版本的清单并回退到硬编码快照。
- **`metadata`** — 在清单、provider 和模型级别的自由格式字典。任何键。Hermes 忽略未知字段，因此你可以注释条目（`"tier": "paid"`、`"tags": [...]` 等）而不需要协调模式更改。
- **`description`** — 仅 OpenRouter。驱动选择器徽章文本（`"recommended"`、`"free"` 或空）。Nous Portal 不使用此字段 — 免费层门控由 Portal 的定价端点实时确定。
- **定价和上下文长度** 不在清单中。这些在获取时从实时 provider API（`/v1/models` 端点、models.dev）获取。

## 获取行为

| 时间 | 发生什么 |
|---|---|
| `/model` 或 `hermes model` | 如果磁盘缓存过期则获取，否则使用缓存 |
| 磁盘缓存新鲜（< TTL） | 无网络请求 |
| 网络失败但有缓存 | 无声回退到缓存，一条日志行 |
| 网络失败，无缓存 | 无声回退到内置快照 |
| 清单模式验证失败 | 视为无法访问 |

缓存位置：`~/.hermes/cache/model_catalog.json`。

## 配置

```yaml
model_catalog:
  enabled: true
  url: https://hermes-agent.nousresearch.com/docs/api/model-catalog.json
  ttl_hours: 24
  providers: {}
```

设置为 `enabled: false` 完全禁用远程获取并始终使用内置快照。

### 每个 provider 覆盖 URL

第三方可以使用相同模式自托管自己的精选列表。将 provider 指向自定义 URL：

```yaml
model_catalog:
  providers:
    openrouter:
      url: https://example.com/my-openrouter-curation.json
```

覆盖清单只需要填充它关心的 provider 块。其他 provider 继续针对主 URL 解析。

## 更新清单

维护者：

```bash
# 从内置硬编码列表重新生成（在编辑
# hermes_cli/models.py 中的 OPENROUTER_MODELS 或 _PROVIDER_MODELS["nous"] 后保持清单同步）
python scripts/build_model_catalog.py
```

然后将生成的 `website/static/api/model-catalog.json` 更改 PR 到 `main`。文档站点在合并时自动部署，新清单在几分钟内生效。

你也可以直接手动编辑 JSON 以进行不属于内置快照的细粒度元数据更改 — 生成器脚本是一种便利，而不是唯一真相来源。

---
sidebar_position: 15
title: "Google Vertex AI"
description: "在 Google Cloud Vertex AI 上通过 Gemini 使用 Hermes Agent —— OAuth2 服务账号或 ADC，GCP 计费和配额，无需静态 API 密钥"
---

# Google Vertex AI

Hermes Agent 通过 Vertex 的 OpenAI 兼容端点支持 **Google Cloud Vertex AI 上的 Gemini 模型**。与 [Google AI Studio Provider](/guides/google-gemini)（使用静态 API 密钥访问 `generativelanguage.googleapis.com`）不同，Vertex 提供**企业级速率限制和 GCP 计费/额度**，适合你希望 Gemini 用量从 Google Cloud 账户扣费而非 AI Studio 密钥的场景。

:::info Vertex 使用 OAuth2 认证，而非 API 密钥
Vertex 的标准端点**没有静态 API 密钥**。每个请求都需要一个短生命周期的 **OAuth2 访问令牌**（约 1 小时 TTL），由服务账号 JSON 或 Application Default Credentials（ADC）生成。Hermes 会为你自动铸造和**刷新**这些令牌 —— 你永远不需要手动粘贴令牌。这就是为什么将临时令牌粘贴到自定义 Provider 的 `api_key` 字段行不通：它会在会话中途过期。
:::

## 前置条件

- **一个 Google Cloud 项目**，已启用 **Vertex AI API** 并开通了计费。
- **凭据**，以下之一：
  - 具有 `roles/aiplatform.user` 角色的**服务账号 JSON** 密钥文件，或
  - 通过 `gcloud auth application-default login`（或在 GCP VM 上运行时使用元数据服务器）获取的 **Application Default Credentials**。
- **`google-auth`** —— 首次选择 Vertex 时自动安装（延迟安装），或通过 `pip install 'hermes-agent[vertex]'` 显式安装。

## 快速开始

```bash
# 选项 A —— 服务账号 JSON（推荐用于服务器/网关）
echo "VERTEX_CREDENTIALS_PATH=/path/to/service-account.json" >> ~/.hermes/.env

# 选项 B —— Application Default Credentials（适合本地开发）
gcloud auth application-default login

# 选择 Vertex 作为 Provider
hermes model
# → 选择 "More providers..." → "Google Vertex AI"
# → 输入你的 GCP 项目 ID（或留空使用凭据中的项目）
# → 选择区域（默认：global）
# → 选择一个 Gemini 模型

# 开始聊天
hermes chat
```

## 配置

Vertex 按敏感度拆分配置：

- **凭据路径**是密钥的指针，存放在 `~/.hermes/.env` 中。
- **项目 ID 和区域**是非密钥的路由设置，存放在 `~/.hermes/config.yaml` 中。

`~/.hermes/.env`：

```bash
# 以下二者之一（按此顺序检查）；都省略则使用 ADC：
VERTEX_CREDENTIALS_PATH=/path/to/service-account.json
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

`~/.hermes/config.yaml`：

```yaml
model:
  default: google/gemini-3-flash-preview
  provider: vertex

vertex:
  project_id: my-gcp-project   # 留空 → 使用凭据中嵌入的项目
  region: global               # Gemini 3.x 预览版需要 "global"
```

:::tip 环境变量优先于 config.yaml
`VERTEX_PROJECT_ID` 和 `VERTEX_REGION` 会覆盖 `config.yaml` 中的 `vertex.project_id` / `vertex.region`。用于单次 shell 的临时覆盖；持久设置放在 `config.yaml` 中。
:::

### 认证原理

1. Hermes 按以下顺序解析凭据：`VERTEX_CREDENTIALS_PATH` → `GOOGLE_APPLICATION_CREDENTIALS` → ADC。
2. 它铸造一个 OAuth2 访问令牌（`cloud-platform` 作用域）并缓存，在令牌过期前 5 分钟自动刷新。
3. 令牌被传递给指向 Vertex 端点的标准 OpenAI 客户端：
   ```text
   https://aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{region}/endpoints/openapi
   ```
   区域性端点使用 `{region}-aiplatform.googleapis.com` 主机名。
4. 如果会话运行时间超过令牌寿命且请求返回 `401`，Hermes 会重新铸造令牌并自动重试。在长时间运行的网关上，如果 ADC 的刷新令牌本身已过期，Hermes 会回退到已配置的服务账号 JSON。

## 可用模型

Vertex 要求模型 ID 带 `google/` 厂商前缀。`hermes model` 选择器提供：

| 模型 | ID |
|-------|----|
| Gemini 3.1 Pro Preview | `google/gemini-3.1-pro-preview` |
| Gemini 3 Pro Preview | `google/gemini-3-pro-preview` |
| Gemini 3 Flash Preview | `google/gemini-3-flash-preview` |
| Gemini 3.1 Flash Lite Preview | `google/gemini-3.1-flash-lite-preview` |
| Gemini 2.5 Pro | `google/gemini-2.5-pro` |
| Gemini 2.5 Flash | `google/gemini-2.5-flash` |

:::note Gemini 3.x 需要 `global` 区域
Gemini 3.x 预览模型通过 `global` 端点提供服务。区域性端点（`us-central1` 等）可能会返回 404。除非有特定原因需要锁定区域，否则请保持 `region: global`。
:::

## 会话中切换模型

```text
/model google/gemini-3-pro-preview
/model google/gemini-3-flash-preview
```

`/model` 在已配置的 Provider 和模型之间切换；它不会收集新的凭据。先用 `hermes model` 配置 Vertex。

## 推理 / 思考

Vertex 通过 OpenAI 兼容接口暴露 Gemini 的思考预算。Hermes 会自动将其推理力度设置映射到 `extra_body.google.thinking_config`，因此 `reasoning_effort` 的用法与其他 Gemini 接口相同。

## 诊断

```bash
hermes doctor
```

Doctor 会报告 Vertex 凭据是否能被解析（服务账号路径或 ADC）以及 Provider 是否已配置。

## 故障排除

### "Vertex AI credentials could not be resolved"

Hermes 既没有找到服务账号 JSON，也没有找到可用的 ADC。请在 `~/.hermes/.env` 中设置 `VERTEX_CREDENTIALS_PATH`，或运行 `gcloud auth application-default login`。如果你的项目没有嵌入凭据中，请在 `config.yaml` 中设置 `vertex.project_id`。

### `google-auth` 未安装

安装扩展：`pip install 'hermes-agent[vertex]'`。Hermes 也会在你首次选择 Vertex Provider 时自动安装它。

### Gemini 3.x 模型返回 404

你可能在使用区域性端点。在 `config.yaml` 的 `vertex:` 部分设置 `region: global`（或取消设置 `VERTEX_REGION`）。

### 403 / 权限被拒绝

服务账号（或你的 ADC 身份）需要在项目上具有 `roles/aiplatform.user` 角色，并且该项目必须已启用 Vertex AI API。

## 相关内容

- [Google Gemini（AI Studio）](/guides/google-gemini) —— 不依赖 GCP 的静态 API 密钥 Gemini
- [AWS Bedrock](/guides/aws-bedrock) —— 另一个原生云 Provider 集成
- [AI Provider](/integrations/providers)
- [配置](/user-guide/configuration)

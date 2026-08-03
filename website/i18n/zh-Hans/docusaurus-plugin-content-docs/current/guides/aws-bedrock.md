---
sidebar_position: 14
title: "AWS Bedrock"
description: "将 Hermes Agent 与 Amazon Bedrock 配合使用——原生 Converse API、IAM 身份验证、Guardrails 及跨区域推理"
---

# AWS Bedrock

Hermes Agent 通过 **Converse API** 原生支持 Amazon Bedrock——而非 OpenAI 兼容端点。这让你可以完整访问 Bedrock 生态系统：IAM 身份验证、Guardrails、跨区域推理配置文件以及所有基础模型。

## 前提条件

- **AWS 凭证** — [boto3 凭证链](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html)支持的任意来源：
  - IAM 实例角色（EC2、ECS、Lambda — 零配置）
  - `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` 环境变量
  - `AWS_PROFILE`（用于 SSO 或命名配置文件）
  - `aws configure`（用于本地开发）
- **boto3** — 通过 `cd ~/.hermes/hermes-agent && uv pip install -e ".[bedrock]"` 安装
- **IAM 权限** — 至少需要：
  - `bedrock:InvokeModel` 和 `bedrock:InvokeModelWithResponseStream`（用于推理）
  - `bedrock:ListFoundationModels` 和 `bedrock:ListInferenceProfiles`（用于模型发现）

:::tip EC2 / ECS / Lambda
在 AWS 计算环境中，为实例附加带有 `AmazonBedrockFullAccess` 的 IAM 角色即可。无需 API 密钥，无需 `.env` 配置——Hermes 会自动检测实例角色。
:::

## 快速开始

```bash
# 安装并启用 Bedrock 支持
cd ~/.hermes/hermes-agent && uv pip install -e ".[bedrock]"

# 选择 Bedrock 作为提供商
hermes model
# → 选择 "More providers..." → "AWS Bedrock"
# → 选择你的区域和模型

# 开始对话
hermes chat
```

## 配置

运行 `hermes model` 后，你的 `~/.hermes/config.yaml` 将包含以下内容：

```yaml
model:
  default: us.anthropic.claude-sonnet-4-6
  provider: bedrock
  base_url: https://bedrock-runtime.us-east-2.amazonaws.com

bedrock:
  region: us-east-2
```

### 区域

通过以下任意方式设置 AWS 区域（优先级从高到低）：

1. `config.yaml` 中的 `bedrock.region`
2. `AWS_REGION` 环境变量
3. `AWS_DEFAULT_REGION` 环境变量
4. 默认值：`us-east-1`

### Guardrails

要对所有模型调用应用 [Amazon Bedrock Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html)：

```yaml
bedrock:
  region: us-east-2
  guardrail:
    guardrail_identifier: "abc123def456"  # 来自 Bedrock 控制台
    guardrail_version: "1"                # 版本号或 "DRAFT"
    stream_processing_mode: "async"       # "sync" 或 "async"
    trace: "disabled"                     # "enabled"、"disabled" 或 "enabled_full"
```

### 模型发现

Hermes 通过 Bedrock 控制平面自动发现可用模型。你可以自定义发现行为：

```yaml
bedrock:
  discovery:
    enabled: true
    provider_filter: ["anthropic", "amazon"]  # 仅显示这些提供商
    refresh_interval: 3600                     # 缓存 1 小时
```

## 可用模型

Bedrock 模型使用**推理配置文件 ID** 进行按需调用。`hermes model` 选择器会自动显示这些 ID，并将推荐模型置于顶部：

| 模型 | ID | 备注 |
|-------|-----|-------|
| Claude Sonnet 4.6 | `us.anthropic.claude-sonnet-4-6` | 推荐——速度与能力的最佳平衡 |
| Claude Opus 4.6 | `us.anthropic.claude-opus-4-6-v1` | 能力最强 |
| Claude Haiku 4.5 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | 最快的 Claude |
| Amazon Nova Pro | `us.amazon.nova-pro-v1:0` | Amazon 旗舰模型 |
| Amazon Nova Micro | `us.amazon.nova-micro-v1:0` | 最快、最经济 |
| DeepSeek V3.2 | `deepseek.v3.2` | 强大的开源模型 |
| Llama 4 Scout 17B | `us.meta.llama4-scout-17b-instruct-v1:0` | Meta 最新模型 |

:::info 跨区域推理
以 `us.` 为前缀的模型使用跨区域推理配置文件，可在多个 AWS 区域间提供更好的容量保障和自动故障转移。以 `global.` 为前缀的模型则在全球所有可用区域间路由。
:::

## 会话中途切换模型

在对话过程中使用 `/model` 命令：

```
/model us.amazon.nova-pro-v1:0
/model deepseek.v3.2
/model us.anthropic.claude-opus-4-6-v1
```

## 诊断

```bash
hermes doctor
```

诊断工具会检查：
- AWS 凭证是否可用（环境变量、IAM 角色、SSO）
- `boto3` 是否已安装
- Bedrock API 是否可达（ListFoundationModels）
- 你所在区域的可用模型数量

## Gateway（消息平台）

Bedrock 可与所有 Hermes gateway 平台配合使用（Telegram、Discord、Slack、飞书等）。将 Bedrock 配置为提供商后，正常启动 gateway 即可：

```bash
hermes gateway setup
hermes gateway start
```

Gateway 读取 `config.yaml` 并使用相同的 Bedrock 提供商配置。

## 故障排查

### "No API key found" / "No AWS credentials"

Hermes 按以下顺序检查凭证：
1. `AWS_BEARER_TOKEN_BEDROCK`
2. `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`
3. `AWS_PROFILE`
4. EC2 实例元数据（IMDS）
5. ECS 容器凭证
6. Lambda 执行角色

若均未找到，请运行 `aws configure` 或为你的计算实例附加 IAM 角色。

### "Invocation of model ID ... with on-demand throughput isn't supported"

请使用**推理配置文件 ID**（以 `us.` 或 `global.` 为前缀），而非裸基础模型 ID。例如：
- ❌ `anthropic.claude-sonnet-4-6`
- ✅ `us.anthropic.claude-sonnet-4-6`

### "ThrottlingException"

你已触及 Bedrock 单模型速率限制。Hermes 会自动进行退避重试。如需提高限额，请在 [AWS Service Quotas 控制台](https://console.aws.amazon.com/servicequotas/)申请配额提升。

## 一键 AWS 部署

如需在 EC2 上通过 CloudFormation 进行全自动部署：

**[sample-hermes-agent-on-aws-with-bedrock](https://github.com/JiaDe-Wu/sample-hermes-agent-on-aws-with-bedrock)** — 自动创建 VPC、IAM 角色、EC2 实例并配置 Bedrock。一键即可在任意区域完成部署。
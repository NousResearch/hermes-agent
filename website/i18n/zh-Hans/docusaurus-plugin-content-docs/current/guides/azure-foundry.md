---
sidebar_position: 15
title: "Microsoft Foundry"
description: "将 Hermes Agent 与 Microsoft Foundry 配合使用——OpenAI 风格与 Anthropic 风格端点、传输协议与已部署模型的自动检测"
---

# Microsoft Foundry

Hermes Agent 的 `azure-foundry` provider 支持 Microsoft Foundry（原 Azure AI Foundry）和 Azure OpenAI。单个 Foundry 资源可以托管两种不同传输格式的模型：

- **OpenAI 风格** — 在 `https://<resource>.openai.azure.com/openai/v1` 等端点上执行 `POST /v1/chat/completions`。用于 GPT-4.x、GPT-5.x、Llama、Mistral 及大多数开放权重模型。
- **Anthropic 风格** — 在 `https://<resource>.services.ai.azure.com/anthropic` 等端点上执行 `POST /v1/messages`。当 Microsoft Foundry 通过 Anthropic Messages API 格式提供 Claude 模型时使用。

设置向导会探测你的端点并自动检测所使用的传输协议、可用的部署以及每个模型的上下文长度。

## 前提条件

- 一个至少包含一个部署的 Microsoft Foundry 或 Azure OpenAI 资源
- 该部署的端点 URL
- **以下之一**：API 密钥（从 Azure Portal 的"Keys and Endpoint"获取），**或者**在 Foundry 资源上拥有 **Azure AI User** RBAC 角色（如果你计划使用 Microsoft Entra ID——即 Microsoft 推荐的无密钥方式）。某些租户在 Microsoft 重命名推出期间可能将该角色显示为 **Foundry User**。

## 快速开始

```bash
hermes model
# → 选择 "Azure Foundry"
# → 输入你的端点 URL
# → 选择认证方式：
#     1. API key
#     2. Microsoft Entra ID（托管标识 / 工作负载标识 / az login）
# → （Entra）Hermes 探测 DefaultAzureCredential；成功后不再询问密钥
# → （API key）输入你的 API 密钥
# Hermes 探测端点并自动检测传输协议 + 模型
# → 从列表中选择模型（或手动输入部署名称）
```

向导将执行以下操作：

1. **嗅探 URL 路径** — 以 `/anthropic` 结尾的 URL 被识别为 Microsoft Foundry Claude 路由。
2. **探测 `GET <base>/models`** — 如果端点返回 OpenAI 格式的模型列表，Hermes 切换到 `chat_completions` 并用返回的部署 ID 预填选择器。
3. **探测 Anthropic Messages 格式** — 针对不暴露 `/models` 但接受 Anthropic Messages 格式的端点的回退方案。
4. **回退到手动输入** — 拒绝所有探测的私有/受限端点仍然可用；你手动选择 API 模式并输入部署名称。

所选模型的上下文长度通过 Hermes 的标准元数据链（`models.dev`、provider 元数据及硬编码的系列回退）解析，并存储在 `config.yaml` 中，以便模型正确确定自身的上下文窗口大小。

## Microsoft Entra ID（无密钥，RBAC）——推荐

Microsoft 推荐在生产 Foundry 工作负载中使用 [Microsoft Entra ID 无密钥认证](https://learn.microsoft.com/azure/ai-foundry/foundry-models/how-to/configure-entra-id)。Hermes 对**两种** API 接口均支持 Entra ID：

- **OpenAI 风格**（`api_mode: chat_completions` / `codex_responses`）— GPT-4/5、Llama、Mistral、DeepSeek 等。
- **Anthropic 风格**（`api_mode: anthropic_messages`）— Microsoft Foundry 上的 Claude 模型。

Foundry 的 RBAC 是按资源级别的（`Azure AI User` 授予两种接口的访问权限；某些租户可能显示为 `Foundry User`），Microsoft 文档对两者使用相同的推理 scope（`https://ai.azure.com/.default`）。底层实现：

- OpenAI 风格使用 OpenAI Python SDK 原生的可调用 `api_key=` 契约——SDK 每次请求自动生成新的 JWT。
- Anthropic 风格使用带有请求事件 hook 的 `httpx.Client`，该 hook 由 `agent.azure_identity_adapter.build_bearer_http_client` 安装，因为 Anthropic SDK 原生不接受可调用的 `auth_token`。该 hook 在每次出站请求时重写 `Authorization: Bearer <fresh-jwt>`。RBAC 和 Foundry scope 相同——唯一的区别在于 SDK 契约。

### 为什么使用 Entra ID？

- 无需轮换或吊销长期有效的 API 密钥。
- RBAC 驱动的访问控制——在 Foundry 资源上授予或移除 `Azure AI User`，无需重写配置。
- 访问和审计日志按被分配者分段，而非所有调用者共享一个静态密钥。
- 通过托管标识，为 Azure VM、ASK Pod、App Service、Functions、Container Apps 和 Foundry Agent Service 提供统一的认证接口。
- 支持 CI/CD 流水线的工作负载标识和服务主体流程。

### 一次性设置（Azure 侧）

1. 在 Azure Portal 中，打开你的 Foundry 资源 → **访问控制 (IAM)** → **添加 → 添加角色分配**。
2. 选择 **Azure AI User** 角色（如果你的租户已重命名，则选择 **Foundry User**）。
3. 将其分配给：
   - **你的用户账户**，用于通过 `az login` 进行本地开发。
   - **托管标识或工作负载标识**，用于 Azure 托管计算（生产环境推荐）。
   - **Foundry Agent Service 托管 Agent 的 Agent 标识**，当 Hermes 在托管 Agent 内运行时。
   - **服务主体**，用于工作负载标识不可用时的 CI/CD 流水线。
4. 等待约 5 分钟以使角色生效。

Azure CLI 等效命令：

```bash
az role assignment create \
  --assignee <principal-or-agent-identity-client-id> \
  --role "Azure AI User" \
  --scope <foundry-resource-id>
```

### 一次性设置（Hermes 侧）

```bash
hermes model
# → 选择 "Azure Foundry"
# → 输入你的端点 URL
# → 认证方式：2（Microsoft Entra ID）
# → （可选）用户分配的托管标识客户端 ID
# → （可选）Azure 租户 ID
# → Hermes 探测 DefaultAzureCredential() 并报告哪个内部凭据成功
#    （例如 AzureCliCredential、ManagedIdentityCredential）
```

向导运行一个有时间限制的预检探测（10 秒超时）。失败时提供"仍然保存，稍后验证"选项——适用于在当前机器上尚无凭据但运行时会有凭据的场景（例如为托管标识部署准备配置）。

`azure-identity` 在首次使用时通过 Hermes 的懒加载安装路径自动安装。如需预先安装：

```bash
pip install azure-identity
```

### 写入 `config.yaml` 的配置

```yaml
model:
  provider: azure-foundry
  base_url: https://my-resource.openai.azure.com/openai/v1
  api_mode: chat_completions
  auth_mode: entra_id
  default: gpt-4o
  context_length: 128000
  entra:
    scope: https://ai.azure.com/.default        # 仅在覆盖默认值时使用
```

Hermes 在 `config.yaml` 中只管理一个 Entra 专属配置项：

- **`scope`** — OAuth 资源 scope。默认为 Microsoft 文档中的推理 scope（`https://ai.azure.com/.default`）。仅在你的资源针对非标准 audience 进行了预配时才需要覆盖。

其他所有内容（租户、服务主体密钥、联合令牌文件、主权云 authority、broker 偏好）均由 `azure-identity` 直接从标准 `AZURE_*` 环境变量读取——参见下方的[凭据解析顺序](#credential-resolution-order)。在 `~/.hermes/.env` 或你的部署环境中设置这些变量，与 Microsoft SDK 参考文档的描述完全一致。

Entra 模式下不会将任何密钥写入 `~/.hermes/.env`——`azure-identity` 在进程内缓存令牌（在可用时也会使用操作系统密钥链 / `~/.IdentityService`）。

### 凭据解析顺序

`azure-identity` 的 `DefaultAzureCredential` 在每次令牌请求时按以下链路逐一尝试，在第一个返回令牌的凭据处停止：

1. **环境凭据** — `AZURE_TENANT_ID` + `AZURE_CLIENT_ID` + `AZURE_CLIENT_SECRET`（或 `AZURE_CLIENT_CERTIFICATE_PATH` / `AZURE_FEDERATED_TOKEN_FILE`）。
2. **工作负载标识** — `AZURE_FEDERATED_TOKEN_FILE`（ASK 联合令牌 / OIDC）。
3. **托管标识** — 虚拟机使用 IMDS 端点（`169.254.169.254`）；App Service / Functions / Container Apps 使用 `IDENTITY_ENDPOINT`。Foundry Agent Service 托管 Agent 使用托管 Agent 的 Agent 标识。
4. **Visual Studio Code** — Azure 账户扩展。
5. **Azure CLI** — `az login` 会话。
6. **Azure Developer CLI** — `azd auth login`。
7. **Azure PowerShell** — `Connect-AzAccount`。
8. **Broker**（仅限 Windows / WSL）— Web Account Manager。

交互式浏览器凭据在无人值守的 Hermes 运行中默认被排除；请改用 Azure CLI、Azure Developer CLI、托管标识、工作负载标识或服务主体凭据。

### 部署模式

**本地开发：**
```bash
az login
hermes model   # 选择 Azure Foundry → Entra ID
hermes         # 使用你的 az login 令牌
```

**Azure VM / Functions / App Service / Container Apps（系统分配的托管标识）：**
1. 在计算资源上启用系统分配的标识。
2. 在 Foundry 资源上为该标识授予 `Azure AI User`（或 `Foundry User`）角色。
3. 在 config.yaml 中设置 `model.auth_mode: entra_id`——无需环境变量。

**Azure VM / Functions / App Service / Container Apps（用户分配的托管标识）：**
- 将 `AZURE_CLIENT_ID` 设置为用户分配标识的客户端 ID，以便 `DefaultAzureCredential` 选择正确的标识。

**Foundry Agent Service 托管 Agent：**
- 创建托管 Agent 并在 Foundry 资源上为该 Agent 的标识授予 `Azure AI User`（或 `Foundry User`）角色。Hermes 在托管 Agent 内部使用 `ManagedIdentityCredential`；角色分配应针对 Agent 标识，而非仅针对父项目或你的用户。

**ASK 工作负载标识（替代 AAD Pod Identity）：**
- 使用工作负载标识客户端 ID 注解 Pod 的服务账户。
- Pod 的联合令牌文件通过 `AZURE_FEDERATED_TOKEN_FILE` 自动检测。
- `model.auth_mode: entra_id` 无需进一步修改配置即可使用。

**CI 中的服务主体：**
- 在 runner 环境中设置 `AZURE_TENANT_ID`、`AZURE_CLIENT_ID`、`AZURE_CLIENT_SECRET`。

#### 主权云（政府云、中国云）

导出 `AZURE_AUTHORITY_HOST`（例如 Azure Government 使用 `https://login.microsoftonline.us`，Azure China 使用 `https://login.partner.microsoftonline.cn`）。`azure-identity` 会直接读取该变量。

### 健康检查

当 `model.auth_mode: entra_id` 时，`hermes doctor` 会对 `DefaultAzureCredential` 运行 10 秒探测，报告哪个内部凭据成功（环境变量是否存在、托管标识端点是否可达等）。

`hermes auth` 显示结构化状态块：

```
azure-foundry (Microsoft Entra ID):
  Endpoint: https://my-resource.openai.azure.com/openai/v1
  Scope: https://ai.azure.com/.default
  Status: configured; live token probe is skipped here
```

### 限制

- **Anthropic 风格端点使用 httpx 事件 hook。** Anthropic Python SDK（≤ 0.86.0）原生不接受可调用的 `auth_token`。Hermes 在自定义 `httpx.Client` 上安装请求事件 hook，每次出站请求时生成新的 JWT 并重写 `Authorization: Bearer <jwt>`。这在功能上等同于 OpenAI SDK 原生的 `Callable[[], str]` 契约，但多了一层间接调用。如果 Anthropic SDK 在未来版本中添加对可调用认证的原生支持，Hermes 将透明地切换到该方式。
- **批处理任务与 `multiprocessing.Pool`。** Entra 令牌 provider 是一个闭包，无法跨进程边界序列化。`batch_runner.py` 会自动从 worker 配置中移除该可调用对象，让每个 worker 进程从 `config.yaml` 重建自己的 provider——无需用户操作，但每个 worker 在启动时需要执行一次凭据链遍历。
- **不在 `auth.json` 中持久化 Bearer JWT。** Hermes 不复制 `azure-identity` 的内部令牌缓存；冷启动时会在首次推理时遍历凭据链。

## 配置（写入 `config.yaml`）

运行向导后，你将看到类似如下的内容：

```yaml
model:
  provider: azure-foundry
  base_url: https://my-resource.openai.azure.com/openai/v1
  api_mode: chat_completions         # 或 "anthropic_messages"
  default: gpt-5.4-mini              # 你的部署 / 模型名称
  context_length: 400000             # 自动检测
```

以及在 `~/.hermes/.env` 中：

```
AZURE_FOUNDRY_API_KEY=<your-azure-key>
```

## OpenAI 风格端点（GPT、Llama 等）

Azure OpenAI 的 v1 GA 端点接受标准 `openai` Python 客户端，改动极少：

```yaml
model:
  provider: azure-foundry
  base_url: https://my-resource.openai.azure.com/openai/v1
  api_mode: chat_completions
  default: gpt-5.4
```

重要行为：

- **GPT-5.x、codex 和 o 系列自动路由到 Responses API。** Microsoft Foundry 将 GPT-5 / codex / o1 / o3 / o4 模型部署为仅支持 Responses API——对其调用 `/chat/completions` 会返回 `400 "The requested operation is unsupported."`。Hermes 通过名称检测这些模型系列，并透明地将 `api_mode` 升级为 `codex_responses`，即使 `config.yaml` 中仍写着 `api_mode: chat_completions`。GPT-4、GPT-4o、Llama、Mistral 及其他部署保持使用 `/chat/completions`。
- **自动使用 `max_completion_tokens`。** Azure OpenAI（与直接使用 OpenAI 一样）对 gpt-4o、o 系列和 gpt-5.x 模型要求使用 `max_completion_tokens`。Hermes 根据端点发送正确的参数。
- **需要 `api-version` 的旧版端点。** 如果你有类似 `https://<resource>.openai.azure.com/openai?api-version=2025-04-01-preview` 的旧版 base URL，Hermes 会提取查询字符串并通过每次请求的 `default_query` 转发（否则 OpenAI SDK 在拼接路径时会丢弃它）。

## Anthropic 风格端点（通过 Microsoft Foundry 使用 Claude）

对于 Claude 部署，使用 Anthropic 风格路由：

```yaml
model:
  provider: azure-foundry
  base_url: https://my-resource.services.ai.azure.com/anthropic
  api_mode: anthropic_messages
  default: claude-sonnet-4-6
```

重要行为：

- **从 base URL 中去除 `/v1`。** Anthropic SDK 在每次请求 URL 后追加 `/v1/messages`——Hermes 在将 URL 传递给 SDK 之前移除末尾的 `/v1`，以避免出现双重 `/v1` 路径。
- **`api-version` 通过 `default_query` 传递，而非追加到 URL。** Azure Anthropic 要求 `api-version` 查询字符串。将其嵌入 base URL 会产生类似 `/anthropic?api-version=.../v1/messages` 的畸形路径并返回 404。Hermes 通过 Anthropic SDK 的 `default_query` 传递 `api-version=2025-04-15`。
- **使用 Bearer 认证而非 `x-api-key`。** Azure 的 Anthropic 兼容路由要求 `Authorization: Bearer <key>`，而非 Anthropic 原生的 `x-api-key` 头。Hermes 检测到 base URL 中包含 `azure.com` 时，通过 SDK 的 `auth_token` 字段路由 API 密钥，确保正确的头部到达上游。
- **保留 1M 上下文窗口 beta 头。** Azure 仍通过 `anthropic-beta: context-1m-2025-08-07` 头控制 1M token Claude 上下文（Opus 4.6/4.7、Sonnet 4.6）的访问。Hermes 在 Azure 路径上保留该 beta 头（在原生 Anthropic OAuth 请求中会被去除，因为某些订阅会拒绝它，但 Azure 要求它）。
- **禁用 OAuth 令牌刷新。** Azure 部署使用静态 API 密钥。适用于 Anthropic Console 的 `~/.claude/.credentials.json` OAuth 令牌刷新循环对 Azure 端点明确跳过，以防止 Claude Code OAuth 令牌在会话中途覆盖你的 Azure 密钥。

## 替代方案：`provider: anthropic` + Azure base URL

如果你已配置 `provider: anthropic` 并只想将其指向 Microsoft Foundry 以使用 Claude，可以完全跳过 `azure-foundry` provider：

```yaml
model:
  provider: anthropic
  base_url: https://my-resource.services.ai.azure.com/anthropic
  key_env: AZURE_ANTHROPIC_KEY
  default: claude-sonnet-4-6
```

在 `~/.hermes/.env` 中设置 `AZURE_ANTHROPIC_KEY`。Hermes 检测到 base URL 中包含 `azure.com` 时，会绕过 Claude Code OAuth 令牌链，直接使用 Azure 密钥进行 `x-api-key` 认证。

`key_env` 是规范的 snake_case 字段名；`api_key_env`（以及驼峰式 `keyEnv` / `apiKeyEnv`）作为别名被接受。如果同时设置了 `key_env` 和 `AZURE_ANTHROPIC_KEY`/`ANTHROPIC_API_KEY`，`key_env` 指定的环境变量优先。

## 模型发现

Azure **不**暴露纯 API 密钥端点来列出你的*已部署*模型部署。部署枚举需要 Azure Resource Manager 认证（`az cognitiveservices account deployment list`）和 Azure AD 主体，而非推理 API 密钥。

Hermes 能做的：

- Azure OpenAI v1 端点（`<resource>.openai.azure.com/openai/v1`）通过 `GET /models` 暴露资源的**可用**模型目录。Hermes 使用此列表预填模型选择器。
- Microsoft Foundry `/anthropic` 路由：通过 URL 路径检测，模型名称手动输入。
- 私有 / 防火墙后的端点：手动输入，并显示友好的"无法探测"提示。

你始终可以直接输入部署名称——Hermes 不会对返回的列表进行验证。

## 环境变量

| 变量 | 用途 |
|----------|---------|
| `AZURE_FOUNDRY_API_KEY` | Microsoft Foundry / Azure OpenAI 的主 API 密钥（api_key 模式） |
| `AZURE_FOUNDRY_BASE_URL` | 端点 URL（通过 `hermes model` 设置；环境变量作为回退） |
| `AZURE_ANTHROPIC_KEY` | 由 `provider: anthropic` + Azure base URL 使用（`ANTHROPIC_API_KEY` 的替代） |
| `AZURE_TENANT_ID` | 服务主体流程的 Entra ID 租户 |
| `AZURE_CLIENT_ID` | Entra ID 客户端 ID（服务主体、工作负载标识或用户分配的托管标识） |
| `AZURE_CLIENT_SECRET` | 服务主体密钥 |
| `AZURE_CLIENT_CERTIFICATE_PATH` | 服务主体证书（密钥的替代方案） |
| `AZURE_FEDERATED_TOKEN_FILE` | 工作负载标识联合令牌路径（ASK） |
| `AZURE_AUTHORITY_HOST` | 主权云 authority 主机覆盖 |
| `IDENTITY_ENDPOINT` / `MSI_ENDPOINT` | App Service、Functions 和 Container Apps 的托管标识端点；VM 通常改用 IMDS |

Azure SDK 直接读取 `AZURE_*` 环境变量。Hermes 除在 `hermes doctor` 输出中报告哪些来源存在外，不会检查这些变量。

## 故障排查

**gpt-5.x 部署返回 401 Unauthorized。**
Azure 在 `/chat/completions` 上提供 gpt-5.x，而非 `/responses`。当 URL 包含 `openai.azure.com` 时，Hermes 会自动处理此问题，但如果你看到带有 `Invalid API key` 正文的 401，请检查 `config.yaml` 中的 `api_mode` 是否为 `chat_completions`。

**`/v1/messages?api-version=.../v1/messages` 返回 404。**
这是修复前 Azure Anthropic 设置中的畸形 URL 问题。升级 Hermes——`api-version` 参数现在通过 `default_query` 传递，而非嵌入 base URL，因此 SDK 在 URL 拼接时不会破坏它。

**向导提示"自动检测不完整"。**
端点拒绝了 `/models` 探测和 Anthropic Messages 探测。这对于防火墙后或设有 IP 白名单的私有端点是正常现象。回退到手动选择 API 模式并输入部署名称——一切仍然正常工作，Hermes 只是无法预填选择器。

**选择了错误的传输协议。**
再次运行 `hermes model`，向导将重新探测。如果探测仍然选择了错误的模式，可以直接编辑 `config.yaml`：

```yaml
model:
  provider: azure-foundry
  api_mode: anthropic_messages   # 或 chat_completions
```

**Entra ID："credential chain exhausted" 或切换到 `auth_mode: entra_id` 后返回 401 Unauthorized。**
- 运行 `az login` 刷新你的开发者会话（缓存的令牌可能已过期）。
- 验证 `Azure AI User`（或 `Foundry User`）角色分配是否已生效：`az role assignment list --assignee <user-or-identity-id>` 应在你的 Foundry 资源上列出该角色。角色传播最多需要 5 分钟。
- 对于用户分配的托管标识，请仔细检查 `AZURE_CLIENT_ID` 是否与附加到计算资源的标识匹配。
- 运行 `hermes doctor`——Azure Entra 探测会报告令牌获取是否成功，并提供修复提示。

**Entra ID：向导预检挂起或超时。**
10 秒预检是软性检查。选择"仍然保存，稍后验证"，部署到目标环境后运行 `hermes doctor`。常见原因包括令牌服务不可达或本地登录状态过期——在 CI 中优先使用工作负载标识，使用服务主体时设置 `AZURE_TENANT_ID`+`AZURE_CLIENT_ID`+`AZURE_CLIENT_SECRET`，或在本地开发时运行 `az login`。

**Anthropic 风格端点使用 Entra ID 时返回 401。**
验证同一 `Azure AI User`（或 `Foundry User`）角色是否已在 Foundry 资源上分配（它同时覆盖 `/openai/v1` 和 `/anthropic` 路径）。如果向导期间 OpenAI 风格探测成功，但运行时 `claude-*` 请求失败，最常见的原因是早期向导运行遗留的过时 `model.entra.scope`——从 `config.yaml` 中删除 `entra.scope` 行，使运行时回退到默认的 `https://ai.azure.com/.default` scope。

## 相关链接

- [环境变量](/reference/environment-variables)
- [配置](/user-guide/configuration)
- [AWS Bedrock](/guides/aws-bedrock) — 另一个主要的云 provider 集成
- [Microsoft：为 Foundry 配置 Entra ID](https://learn.microsoft.com/azure/ai-foundry/foundry-models/how-to/configure-entra-id) — 无密钥路径的上游文档
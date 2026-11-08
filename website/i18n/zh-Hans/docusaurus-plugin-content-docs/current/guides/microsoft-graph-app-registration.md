---
title: "注册 Microsoft Graph 应用程序"
description: "Azure 门户操作指南：创建为 Teams 会议流水线提供支持的应用注册"
---

# 注册 Microsoft Graph 应用程序

Teams 会议流水线使用**仅限应用**（daemon）身份验证从 Microsoft Graph 读取会议转录、录制及相关产物——无需用户登录，无需每次会议单独交互式授权。这需要一个经过管理员同意、具备应用程序权限的 Azure AD 应用注册。

本指南涵盖以下步骤：

1. 创建应用注册
2. 创建客户端密钥
3. 授予流水线所需的 Graph API 权限
4. 管理员同意这些权限
5. （可选）通过应用程序访问策略将应用限定到特定用户

完成本指南需要**租户管理员权限**（或由管理员代为授予同意）。请记录收集到的值——最终需要填入 `~/.hermes/.env`。

## 前提条件

- 一个具备 Teams Premium 或 Teams 许可证（可生成会议转录和录制）的 Microsoft 365 租户
- 可访问 Azure 门户 [entra.microsoft.com](https://entra.microsoft.com) 的管理员权限
- 一个可公开访问的 HTTPS 端点，用于接收 Graph 变更通知（在后续 webhook 监听器步骤中配置）

## 步骤 1：创建应用注册

1. 以租户管理员身份登录 [entra.microsoft.com](https://entra.microsoft.com)。
2. 导航至 **Identity → Applications → App registrations**。
3. 点击 **New registration**。
4. 填写以下内容：
   - **Name：**`Hermes Teams Meeting Pipeline`（或任何你能识别的名称）。
   - **Supported account types：***Accounts in this organizational directory only (Single tenant)*。
   - **Redirect URI：**留空——仅限应用的身份验证不需要此项。
5. 点击 **Register**。

页面将跳转至应用概览页。复制以下两个值：

- **Application (client) ID** → `MSGRAPH_CLIENT_ID`
- **Directory (tenant) ID** → `MSGRAPH_TENANT_ID`

## 步骤 2：创建客户端密钥

1. 在左侧导航栏中，打开 **Certificates & secrets**。
2. 点击 **New client secret**。
3. **Description：**`hermes-graph-secret`。**Expires：**根据你的轮换策略选择合适的值（通常为 6-24 个月）。
4. 点击 **Add**。
5. 立即复制 **Value** 列的值——该值仅显示一次。此值即为 `MSGRAPH_CLIENT_SECRET`。

> **Secret ID** 列不是密钥本身。你需要的是 **Value** 列。

## 步骤 3：授予 Graph API 权限

流水线使用最小化的应用程序权限集。仅添加所需权限；每项权限都会扩大应用在租户范围内的读取能力。

1. 在左侧导航栏中，打开 **API permissions**。
2. 点击 **Add a permission** → **Microsoft Graph** → **Application permissions**。
3. 根据下表添加流水线所需的权限。
4. 添加完成后，点击 **Grant admin consent for `<your tenant>`**。每项权限的 Status 列应变为绿色对勾。

### 转录优先摘要所需权限

| 权限 | 允许应用执行的操作 |
|------------|--------------------------|
| `OnlineMeetings.Read.All` | 读取 Teams 在线会议元数据（主题、参与者、加入 URL）。 |
| `OnlineMeetingTranscript.Read.All` | 读取 Teams 生成的会议转录。 |

### 录制回退所需权限（当转录不可用时）

| 权限 | 允许应用执行的操作 |
|------------|--------------------------|
| `OnlineMeetingRecording.Read.All` | 下载 Teams 会议录制以进行离线语音转文字处理。 |
| `CallRecords.Read.All` | 仅知道加入 URL 时，通过通话记录解析会议信息。 |

### 出站摘要投递所需权限（仅限 Graph 模式）

若 `platforms.teams.extra.delivery_mode` 设置为 `graph`，流水线将通过 Graph API 将摘要发布到 Teams 频道或聊天。如果使用 `incoming_webhook` 投递模式，可跳过这些权限。

| 权限 | 允许应用执行的操作 |
|------------|--------------------------|
| `ChannelMessage.Send` | 以应用身份向 Teams 频道发布消息。 |
| `Chat.ReadWrite.All` | 向一对一及群组聊天发布消息（仅在将 `chat_id` 设为投递目标时需要）。 |

### 不推荐的权限

- `OnlineMeetings.ReadWrite.All` / `Chat.ReadWrite`（不带 `.All`）——权限范围超出流水线所需。
- 委托权限——流水线使用仅限应用（客户端凭据）流程；委托权限在没有用户登录的情况下无法生效。

## 步骤 4：（推荐）通过应用程序访问策略限定应用范围

默认情况下，`OnlineMeetings.Read.All` 等应用程序权限会授予应用访问租户中**所有**会议的权限。对于合作伙伴演示和开发租户而言这没有问题；但在生产环境中，你几乎肯定需要限制应用可读取哪些用户的会议。

Microsoft 专门为 Teams 提供了**应用程序访问策略**（Application Access Policies）。该策略仅支持 PowerShell 操作，没有门户 UI。

在已安装并连接 MicrosoftTeams 模块的管理员 PowerShell 中（`Connect-MicrosoftTeams`）执行：

```powershell
# Create a policy scoped to the Hermes app
New-CsApplicationAccessPolicy `
  -Identity "Hermes-Meeting-Pipeline-Policy" `
  -AppIds "<MSGRAPH_CLIENT_ID>" `
  -Description "Restrict Hermes meeting pipeline to allow-listed users"

# Grant the policy to specific users whose meetings the pipeline may read
Grant-CsApplicationAccessPolicy `
  -PolicyName "Hermes-Meeting-Pipeline-Policy" `
  -Identity "alice@example.com"

Grant-CsApplicationAccessPolicy `
  -PolicyName "Hermes-Meeting-Pipeline-Policy" `
  -Identity "bob@example.com"
```

授权后策略生效最长需要 30 分钟。使用以下命令验证：

```powershell
Test-CsApplicationAccessPolicy -Identity "alice@example.com" -AppId "<MSGRAPH_CLIENT_ID>"
```

若不配置此策略，**任何**用户的会议均可被读取——这正是该权限在技术层面所授予的范围。生产租户请勿跳过此步骤。

## 步骤 5：将凭据写入环境文件

将收集到的三个值填入 `~/.hermes/.env`：

```bash
MSGRAPH_TENANT_ID=<directory-tenant-id>
MSGRAPH_CLIENT_ID=<application-client-id>
MSGRAPH_CLIENT_SECRET=<client-secret-value>
```

设置文件权限，确保只有你能读取密钥：

```bash
chmod 600 ~/.hermes/.env
```

## 步骤 6：验证令牌流程

Hermes 内置了 Graph 身份验证冒烟测试。在 Hermes 安装目录下执行：

```python
python -c "
import asyncio
from tools.microsoft_graph_auth import MicrosoftGraphTokenProvider
provider = MicrosoftGraphTokenProvider.from_env()
token = asyncio.run(provider.get_access_token())
print('Token acquired, length:', len(token))
print(provider.inspect_token_health())
"
```

成功执行后将打印一个较长的 token（令牌）字符串，以及一个健康状态字典，其中 `cached: True`，`expires_in_seconds` 值接近 3600。失败时将抛出 `MicrosoftGraphTokenError`，并附带 Azure 错误码——最常见的错误如下：

| Azure 错误码 | 含义 | 修复方法 |
|-------------|---------|-----|
| `AADSTS7000215: Invalid client secret` | 密钥值不匹配或已过期。 | 在步骤 2 中生成新密钥，并更新 `.env`。 |
| `AADSTS700016: Application not found` | `MSGRAPH_CLIENT_ID` 错误或租户不匹配。 | 确认步骤 1 中的值来自同一应用。 |
| `AADSTS90002: Tenant not found` | `MSGRAPH_TENANT_ID` 存在拼写错误。 | 重新从应用概览页复制 Directory (tenant) ID。 |
| `insufficient_claims`（调用时报错，非获取令牌时） | 令牌获取成功，但 Graph 返回 401/403。 | 跳过了步骤 3 的管理员同意，或添加权限后未重新同意。重新进入 API permissions 并点击 **Grant admin consent**。 |

## 轮换客户端密钥

Azure 客户端密钥有固定的过期时间。在密钥过期前：

1. 在步骤 2 中创建第二个客户端密钥，不要删除第一个。
2. 用新值更新 `~/.hermes/.env` 中的 `MSGRAPH_CLIENT_SECRET`。
3. 重启 gateway 以使新密钥生效：`hermes gateway restart`。
4. 使用上述冒烟测试进行验证。
5. 在 Azure 门户中删除旧密钥。

## 后续步骤

凭据验证通过后，继续完成以下配置：

- **Webhook 监听器配置**——部署接收 Graph 变更通知的 `msgraph_webhook` gateway 平台。
- **流水线配置**——配置 Teams 会议流水线运行时及操作员 CLI。
- **出站投递**——将摘要回传至 Teams 频道或聊天。

上述页面将随添加对应运行时的 PR 一并发布。本凭据配置是独立的前提步骤，可提前完成。
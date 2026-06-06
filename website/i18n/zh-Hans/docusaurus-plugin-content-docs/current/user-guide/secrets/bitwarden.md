# Bitwarden Secrets Manager

在进程启动时从 [Bitwarden Secrets Manager](https://bitwarden.com/products/secrets-manager/) 拉取 API 密钥，而不是以明文形式存储在 `~/.hermes/.env` 中。一个引导密钥（机器账户访问令牌）替代了 N 个提供商密钥，轮换凭据只需在 Bitwarden Web 应用中修改一次即可。

## 工作原理

1. 在 Bitwarden Secrets Manager 中创建一个**机器账户**，授予其对某个项目的读取权限，并生成一个**访问令牌**。
2. Hermes 将该单一令牌以 `BWS_ACCESS_TOKEN` 的形式存储在 `~/.hermes/.env` 中。
3. 每次 `hermes`（或 gateway，或 cron 任务）启动时，在加载 `~/.hermes/.env` 之后，Hermes 会调用 `bws secret list <project_id>` 并将返回的密钥写入 `os.environ`。
4. 默认情况下，Hermes **覆盖**环境中已有的值，因此 Bitwarden 是唯一可信来源——在 Web 应用中轮换一次密钥，每个 Hermes 进程在下次启动时即可获取最新值。如果希望 `.env` 优先，可在配置中将 `override_existing: false`。

`bws` 二进制文件在首次使用时会自动下载到 `~/.hermes/bin/`，无需 `apt`、`brew` 或 `sudo`。

## 为什么使用机器账户（以及为什么没有双因素认证提示）

Bitwarden Secrets Manager 专为非交互式工作负载设计：机器账户不能设置双因素认证（2FA）门控，因为流程中没有人工介入。访问令牌本身就是凭据。任何持有该令牌的人都可以读取机器账户有权访问的所有密钥，因此请将其视为高价值的 bearer token（持有者令牌）——将其存储在 `.env` 中（而非 `config.yaml`），如果泄露，请立即在 Bitwarden Web 应用中吊销并重新生成。

机器账户在 *Web 应用中*设置，此时你的正常双因素认证仍然有效。之后令牌即可自主运行。

## 设置

### 1. 创建机器账户和访问令牌

在 [Bitwarden Web 应用](https://vault.bitwarden.com)（欧盟账户请使用 [vault.bitwarden.eu](https://vault.bitwarden.eu)）中：

1. 通过产品切换器切换到 **Secrets Manager**。
2. 创建或选择一个**项目**（例如"Hermes keys"）。
3. 将提供商密钥添加为 secret。secret 的**名称**将成为环境变量名——使用 `OPENROUTER_API_KEY`、`ANTHROPIC_API_KEY` 等。
4. **Machine accounts → New machine account → My Hermes machine** → **Projects** 标签页 → 授予对你的项目的 Read 权限。
5. **Access tokens** 标签页 → **Create access token** → 选择**永不**过期（或指定日期）→ 复制令牌（以 `0.` 开头）。Bitwarden 无法再次检索该令牌——请妥善保存副本。

Secrets Manager 包含在 Bitwarden 免费套餐中（有使用限制）；无需付费计划即可试用。

### 2. 运行向导

```bash
hermes secrets bitwarden setup
```

该命令将：

1. 下载并验证 `bws v2.0.0`，存放至 `~/.hermes/bin/bws`。
2. 提示输入访问令牌（输入内容隐藏）。以 `BWS_ACCESS_TOKEN` 形式存储在 `~/.hermes/.env` 中。
3. 询问机器账户所属的 Bitwarden 区域——**US Cloud**、**EU Cloud** 或**自托管/自定义 URL**。以 `secrets.bitwarden.server_url` 形式存储在 `config.yaml` 中，并作为 `BWS_SERVER_URL` 传递给 `bws`。
4. 列出机器账户可见的项目，选择其中一个。以 `secrets.bitwarden.project_id` 形式存储在 `config.yaml` 中。
5. 测试拉取该项目的 secret，并显示将解析出哪些环境变量。
6. 将 `secrets.bitwarden.enabled` 设置为 `true`。

也支持通过参数进行非交互式设置：

```bash
hermes secrets bitwarden setup \
  --access-token "$BWS_ACCESS_TOKEN" \
  --server-url https://vault.bitwarden.eu \
  --project-id <project-uuid>
```

### 3. 确认

```bash
hermes secrets bitwarden status
```

此后，每次调用 `hermes` 都会在启动时拉取最新 secret。进程中首次应用 secret 时，stderr 会显示一行摘要信息。

## CLI

| 命令 | 功能 |
|---|---|
| `hermes secrets bitwarden setup` | 交互式向导（安装二进制文件、提示输入令牌、选择项目、测试拉取） |
| `hermes secrets bitwarden status` | 显示配置、二进制版本，以及令牌是否存在/是否通过校验 |
| `hermes secrets bitwarden sync` | 演习模式：立即拉取 secret 并显示将应用的内容 |
| `hermes secrets bitwarden sync --apply` | 拉取并导出到当前 shell 的环境中 |
| `hermes secrets bitwarden install` | 仅下载固定版本的 `bws` 二进制文件（无需认证） |
| `hermes secrets bitwarden disable` | 将 `enabled` 设为 `false`；保留令牌和项目 ID |

## 配置

`~/.hermes/config.yaml` 中的默认值：

```yaml
secrets:
  bitwarden:
    enabled: false
    access_token_env: BWS_ACCESS_TOKEN
    project_id: ""
    server_url: ""
    cache_ttl_seconds: 300
    override_existing: true
    auto_install: true
```

| 键 | 默认值 | 功能 |
|---|---|---|
| `enabled` | `false` | 主开关。为 false 时，永不联系 Bitwarden。 |
| `access_token_env` | `BWS_ACCESS_TOKEN` | 存储引导令牌的环境变量名。如果你已将 `BWS_ACCESS_TOKEN` 用于其他用途，可修改此项。 |
| `project_id` | `""` | 要同步的项目 UUID。 |
| `server_url` | `""` | Bitwarden 区域或自托管端点。为空时使用 `bws` 默认值（US Cloud，`https://vault.bitwarden.com`）。欧盟云设为 `https://vault.bitwarden.eu`，自托管则填写自己的 URL。以 `BWS_SERVER_URL` 形式传递给 `bws` 子进程。 |
| `cache_ttl_seconds` | `300` | 进程内拉取结果的复用时长。设为 `0` 可禁用缓存。缓存按进程隔离；新的 `hermes` 调用从头开始。 |
| `override_existing` | `true` | 为 true 时，Bitwarden 的值会覆盖环境中已有的任何值（使 Web 应用中的轮换真正生效）。如果希望本地 `.env` / shell 导出优先，设为 `false`。 |
| `auto_install` | `true` | 为 true 时，首次使用时自动将 `bws` 下载到 `~/.hermes/bin/`。 |

## 故障模式

Bitwarden 永远不会阻塞 Hermes 启动。如果出现任何问题，stderr 会显示一行警告，Hermes 继续使用 `.env` 中已有的凭据：

| 现象 | 原因 | 修复方法 |
|---|---|---|
| `BWS_ACCESS_TOKEN is not set` | 配置中已启用，但令牌已从 `.env` 中清除 | 重新运行 `hermes secrets bitwarden setup` |
| `bws exited 1: invalid access token` | 令牌已吊销或有误 | 生成新令牌，重新运行 setup |
| `[400 Bad Request] {"error":"invalid_client"}` | 令牌所属的 Bitwarden 区域与 `bws` 调用的区域不匹配（例如欧盟令牌访问了美国 identity 端点） | 重新运行 setup 并选择正确区域，或将 `secrets.bitwarden.server_url` 设为 `https://vault.bitwarden.eu`（或自托管 URL） |
| `bws timed out` | 网络受阻或 Bitwarden API 响应缓慢 | 检查到 `api.bitwarden.com`（或你的 `server_url`）的连通性 |
| `bws binary not available` | `auto_install: false` 且 `bws` 不在 PATH 中 | 从 [github.com/bitwarden/sdk-sm/releases](https://github.com/bitwarden/sdk-sm/releases) 手动安装，或重新开启 `auto_install` |
| `Checksum mismatch` | 下载内容损坏或被篡改 | 重新运行，将自动重试；如持续出现，请提交 issue |

## 安全说明

- 引导令牌（`BWS_ACCESS_TOKEN`）本身是敏感信息——任何持有它的人都可以读取机器账户有权访问的所有 secret。请与其他 API 密钥同等对待。
- 即使 `override_existing: true`，Hermes 也会拒绝让 Bitwarden 覆盖引导令牌本身。如果你将 `BWS_ACCESS_TOKEN` 作为 secret 存储在项目中，应用时会静默跳过。
- `bws` 二进制文件的下载会与同一 GitHub release 中发布的 SHA-256 校验和进行验证。不匹配时将中止安装。
- 固定版本（撰写本文时为 `bws v2.0.0`）通过向本仓库提交 PR 的方式更新——Hermes 不会将 `bws` 自动升级到"最新版本"，因为上游 release 的结构可能发生变化。

## 不适用场景

- **单机个人使用**，`~/.hermes/.env` 已经够用。你只是用一个凭据换了另一个，并在启动时增加了网络依赖。
- **无法访问 `api.bitwarden.com` 的隔离环境**。
- **CI/CD** 场景，已有现成的 secret 注入机制（GitHub Actions secrets、Vault 等）——选择一种方式，不要两者并用。

适合使用此功能的场景：多机器集群、共享开发机、gateway VPS，或任何需要跨多个 Hermes 安装进行集中轮换和吊销管理的场景。

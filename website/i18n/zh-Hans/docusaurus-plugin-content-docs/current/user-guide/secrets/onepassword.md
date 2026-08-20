# 1Password

在进程启动时从 [1Password](https://1password.com/) 解析 Provider API 密钥，而不是在 `~/.hermes/.env` 中明文存储。你将密钥保存为 1Password 项目，通过 `op://vault/item/field` 引用它们；轮换凭据只需在 1Password 中改一次。

## 工作原理

1. 安装官方 [1Password CLI](https://developer.1password.com/docs/cli/get-started/)（`op`）并认证 —— 使用**服务账号令牌**（无头服务器）或**交互式/桌面会话**（你的笔记本电脑）。
2. 在 `~/.hermes/config.yaml` 中将环境变量名映射到 `op://` 引用。
3. 每次 `hermes`（或网关、定时任务）启动时，在 `~/.hermes/.env` 加载后，Hermes 为每个引用运行 `op read` 并将解析后的值设置到 `os.environ`。
4. 默认情况下 Hermes **覆盖**环境中已有的值，因此 1Password 是真实来源 —— 轮换一次凭据，每个 Hermes 进程在下次启动时就会获取到。如果你想让 `.env` 优先，请翻转 `override_existing: false`。

Hermes 从不代替你认证，也从不下载 `op`：它调用你已安装、已信任的 CLI。如果 `op` 缺失、会话已锁定或引用错误，Hermes 打印一行警告并继续使用 `.env` 已有的凭据 —— 它绝不会阻止启动。

## 认证

`op` 支持两种非交互友好的模式；Hermes 对两种都有效：

- **服务账号**（推荐用于服务器/CI）：在 1Password 中创建服务账号，授予其对相关 Vault 的读取访问权限，并将其令牌导出为 `~/.hermes/.env` 中的 `OP_SERVICE_ACCOUNT_TOKEN`。令牌就是凭据 —— 像对待任何其他 bearer 令牌一样对待它。
- **桌面/交互式会话**（笔记本电脑）：运行 `op signin`（或在 1Password 应用中启用 CLI 集成）。Hermes 将你的 `OP_SESSION_*` 变量传递给 `op` 子进程。1Password 缓存键包含这些会话变量，因此登录不同账户永远不会返回缓存在之前身份下的值。

## 引导令牌（Bootstrap Token）

当你使用**服务账号令牌**认证时，该令牌本身就是 Hermes 在解析任何 `op://` 引用*之前*需要的引导凭据。它必须存在于解析密钥的每个进程的 `os.environ` 中 —— 包括定时任务（`kanban.dispatch_in_gateway: false`）、子进程调用、CLI 运行、macOS launchd 代理和 Docker 容器 —— 而不仅是交互式网关。有三种方式使其可用，按优先级排列：

1. **在 `~/.hermes/.env` 中（推荐）。** `hermes secrets onepassword setup --token <token>` 将令牌写入 `~/.hermes/.env`，与 Bitwarden 的 `BWS_ACCESS_TOKEN` 完全一样。因为 `load_hermes_dotenv()` 始终加载 `.env`，令牌在任何地方都可用，零额外设置。这是最简单可靠的选项。

2. **在 `~/.hermes/.op.env` 中（gitignore）。** 如果你想将服务账号令牌放在 `.env` 之外 —— 例如让 `.env` 可以检入私有 dotfiles 仓库而令牌不进入版本控制 —— 将其放在 `~/.hermes/.op.env`：

   ```bash
   echo 'OP_SERVICE_ACCOUNT_TOKEN=ops_...' > ~/.hermes/.op.env
   chmod 600 ~/.hermes/.op.env
   ```

   Hermes 在启动时自动加载 `.op.env`，**在** `.env` **之后**，且**绝不覆盖**环境中已有的令牌。`.op.env` 被 gitignore，令牌永远不会进入提交的文件。

3. **通过 systemd `EnvironmentFile`（Linux 网关）。** 如果你在 systemd 下运行网关，可以直接将令牌注入服务环境：

   ```ini
   [Service]
   EnvironmentFile=-/home/youruser/.hermes/.op.env
   ```

   通过这种方式注入的令牌具有优先级 —— Hermes 检测到 `OP_SERVICE_ACCOUNT_TOKEN` 已设置并完全跳过加载 `.op.env`。

如果令牌仅通过交互式 shell 可用（`op signin`、`.bashrc` 中的 `OP_SESSION_*` 导出等），定时任务或新生成的子进程**不会**继承它，这些上下文会记录警告并回退到 `.env` 已有的凭据。对于任何非交互式工作负载，请使用上述三种方式之一。

## 设置

### 1. 安装并登录 `op`

参阅 [1Password CLI 入门指南](https://developer.1password.com/docs/cli/get-started/)。验证它正常工作：

```bash
op whoami
```

### 2. 启用集成

```bash
hermes secrets onepassword setup
```

这会验证 `op` 在 `PATH` 上（或使用 `--binary-path`），记录你的账户/令牌设置，检查活跃会话，并翻转 `secrets.onepassword.enabled: true`。非交互标志：

```bash
hermes secrets onepassword setup \
  --account my.1password.com \
  --token-env OP_SERVICE_ACCOUNT_TOKEN \
  --token "$OP_SERVICE_ACCOUNT_TOKEN"
```

### 3. 映射你的凭据

引用格式为 `op://<vault>/<item>/<field>`：

```bash
hermes secrets onepassword set OPENAI_API_KEY    "op://Private/OpenAI/api key"
hermes secrets onepassword set ANTHROPIC_API_KEY "op://Private/Anthropic/credential"
```

### 4. 预览并确认

```bash
hermes secrets onepassword sync     # dry-run: resolve now, show what would apply
hermes secrets onepassword status   # config + binary + references + auth
```

从现在起，每次 `hermes` 调用在启动时解析引用。你会在 stderr 中看到一行摘要，当密钥在进程首次应用时。

## CLI

| 命令 | 功能 |
|---|---|
| `hermes secrets onepassword setup` | 验证 `op`，设置账户/令牌环境变量，启用 |
| `hermes secrets onepassword status` | 显示配置、二进制、认证和已配置的引用 |
| `hermes secrets onepassword set ENV_VAR "op://…"` | 映射环境变量到引用（存储时去除空白 + 验证） |
| `hermes secrets onepassword remove ENV_VAR` | 移除映射 |
| `hermes secrets onepassword sync` | 干运行：立即解析引用并显示将应用的内容 |
| `hermes secrets onepassword sync --apply` | 解析并导出到当前 shell 环境 |
| `hermes secrets onepassword disable` | 翻转 `enabled: false`；保留映射 |

`op` 和 `1password` 被接受为 `onepassword` 的别名。

## 配置

`~/.hermes/config.yaml` 中的默认值：

```yaml
secrets:
  onepassword:
    enabled: false
    env:
      OPENAI_API_KEY: "op://Private/OpenAI/api key"
      ANTHROPIC_API_KEY: "op://Private/Anthropic/credential"
    account: ""
    service_account_token_env: OP_SERVICE_ACCOUNT_TOKEN
    binary_path: ""
    cache_ttl_seconds: 300
    override_existing: true
```

| 键 | 默认值 | 功能 |
|---|---|---|
| `enabled` | `false` | 主开关。为 false 时，`op` 从不被调用。 |
| `env` | `{}` | 环境变量名 → `op://vault/item/field` 引用的映射。名称不是有效环境变量名或值不是 `op://` 引用的条目会被跳过并警告。 |
| `account` | `""` | 传递给 `op read --account` 的账户简称/登录地址。空值使用 `op` 的默认账户。 |
| `service_account_token_env` | `OP_SERVICE_ACCOUNT_TOKEN` | Hermes 读取服务账号令牌的环境变量。其值作为 `OP_SERVICE_ACCOUNT_TOKEN` 导出到 `op` 子进程（`op` 期望的名称）。不设置此变量则使用桌面/交互式会话。 |
| `binary_path` | `""` | `op` 的绝对路径。设置后按原样使用，**不查询** `PATH` —— 固定此值以避免信任 `PATH` 上出现的第一个 `op`。 |
| `cache_ttl_seconds` | `300` | 解析值的重用时间（进程内和磁盘上）。设为 `0` 禁用**两个**缓存层 —— 完全不写入磁盘。 |
| `override_existing` | `true` | 为 true 时，解析值覆盖环境中已有的值（使轮换生效）。翻转为 `false` 让 `.env` / shell 导出优先；那些引用在 `op` 调用*之前*被跳过。 |

## 失败模式

1Password 从不阻止 Hermes 启动。出错时你会在 stderr 中看到一行警告，Hermes 继续：

| 症状 | 原因 | 修复 |
|---|---|---|
| `the op CLI was not found on PATH` | `op` 未安装/不在 PATH 上 | 安装 CLI，或设置 `secrets.onepassword.binary_path` |
| `op read failed for 'op://…': …` | 会话已锁定、令牌过期或无 Vault 访问权限 | `op signin`，刷新令牌，或授予服务账号访问权限 |
| `op read returned an empty value for 'op://…'` | 引用的字段存在但为空 | 在 1Password 中修复项目/字段（空值永远不会被应用 —— 你现有的环境变量保持不变） |
| `… is not an op:// secret reference` | 映射值不是 `op://` 引用 | 使用正确的 `op://vault/item/field` 格式重新设置 |
| `op read timed out` | 网络被阻止或 1Password 响应慢 | 检查连接性/桌面应用集成 |

## 缓存

成功、完整的拉取在进程内和磁盘上缓存，位于 `<hermes_home>/cache/op_cache.json`（原子写入，权限 `0600`），因此连续的短生命周期 `hermes` 调用不会为每个引用重新 shell `op`。缓存：

- 仅存储解析后的密钥**值** —— 绝不存储服务账号令牌或任何原始认证材料（认证被指纹化到缓存键中）；
- 当令牌、账户、`OP_SESSION_*` 变量或引用集合变化时失效；
- 在拉取有任何逐引用错误时**不写入**，因此临时认证失败不会被冻结在 TTL 内；
- 在 `cache_ttl_seconds: 0` 时完全禁用 —— 读取*和*写入。

## 安全说明

- 1Password 服务账号令牌可以读取账户有权访问的每个密钥。将其存储在 `~/.hermes/.env` 中（不是 `config.yaml`），泄漏时从 1Password 撤销 + 重新生成。
- Hermes 拒绝让解析值覆盖令牌环境变量本身，即使在 `override_existing: true` 时。
- `op` 子进程获得最小化的白名单环境（认证/会话变量 + `PATH`/`HOME`），而不是 `os.environ` 的完整副本，因此 dotenv 后的 Provider 凭据不会全部被子进程继承。
- 引用被验证以 `op://` 开头，引用在 `--` 选项终止符后传递，因此精心构造的值无法被解析为 `op` 标志。

## 何时不该使用

- **单机个人设置**，`~/.hermes/.env` 就够了。
- **无法访问 1Password 的离线环境**。
- **CI/CD** 已有现成的密钥注入机制 —— 选择一条路径，不要两条。

适合使用的场景是多机集群、共享开发机、网关 VPS，或任何需要跨多个 Hermes 安装集中轮换和撤销的地方。

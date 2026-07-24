---
sidebar_position: 9
title: "Secret Source 插件"
description: "如何为 Hermes Agent 构建密钥管理后端插件"
---

# 构建 Secret Source 插件

Secret Source（密钥源）负责在进程启动时将 Provider 凭据从外部密钥管理器（Vault、密码管理器、操作系统密钥库、自定义脚本）解析为环境变量 —— 发生在 `~/.hermes/.env` 加载之后、Hermes 读取凭据之前。Bitwarden 和 1Password 已内置于源码中；**其他所有后端都以插件形式提供**。本指南将介绍如何构建一个。

:::tip
内置集合是刻意封闭的，与 [Memory Provider](/developer-guide/memory-provider-plugin) 采用相同策略：向 `agent/secret_sources/` 添加新 Vault 后端的 PR 会被关闭，并指向本指南。请将你的后端发布为独立插件仓库，并在 Nous Research Discord（`#plugins-skills-and-skins`）中分享。
:::

## 框架负责什么 vs. 你负责什么

编排器（`agent.secret_sources.registry.apply_all`）负责所有与安全性和优先级相关的逻辑，后端不需要也无法处理这些：

| 框架负责 | 你负责 |
|---|---|
| 源排序、显式映射 vs. 批量导入的优先级 | 从你的后端获取值 |
| 先到先得的冲突处理 + 警告 | 验证你的引用格式 |
| `override_existing` 语义（不会跨源覆盖） | 与你的 CLI/SDK/API 交互 |
| 受保护的引导令牌（bootstrap token） | 声明哪个环境变量是你的引导令牌 |
| 每个源的挂钟超时 | 保持 `fetch()` 合理的速度 |
| 每个变量的来源追踪 + `(from X)` 标签 | 一个人类可读的 `label` |
| `os.environ` 写入 | 什么都不做 —— 你永远不触碰环境变量 |

## 目录结构

```
~/.hermes/plugins/my-vault/
├── plugin.yaml      # name, description
└── __init__.py      # SecretSource 子类 + register(ctx)
```

## SecretSource ABC

实现 `agent.secret_sources.base.SecretSource`。只有一个方法是必需的：

```python
from pathlib import Path

from agent.secret_sources.base import (
    ErrorKind,
    FetchResult,
    SecretSource,
    run_secret_cli,
)


class MyVaultSource(SecretSource):
    name = "myvault"          # config section key: secrets.myvault
    label = "My Vault"        # used in startup lines + provenance labels
    shape = "mapped"          # "mapped" (explicit VAR→ref map) or "bulk" (project dump)
    scheme = "mv"             # optional: unique URI scheme you own (mv://...)

    def fetch(self, cfg: dict, home_path: Path) -> FetchResult:
        """Resolve secrets. MUST NOT raise. MUST NOT prompt."""
        result = FetchResult()
        token = os.environ.get("MYVAULT_TOKEN", "").strip()
        if not token:
            result.error = "secrets.myvault.enabled is true but MYVAULT_TOKEN is not set."
            result.error_kind = ErrorKind.NOT_CONFIGURED
            return result

        try:
            proc = run_secret_cli(
                ["myvault-cli", "export", "--json"],
                allow_env=["MYVAULT_TOKEN"],   # ONLY your auth vars — never full os.environ
                timeout=30,
            )
        except RuntimeError as exc:           # spawn failure / timeout
            result.error = str(exc)
            result.error_kind = ErrorKind.BINARY_MISSING
            return result

        if proc.returncode != 0:
            result.error = f"myvault-cli exited {proc.returncode}: {proc.stderr[:200]}"
            result.error_kind = ErrorKind.AUTH_FAILED
            return result

        result.secrets = parse_your_output(proc.stdout)  # {ENV_VAR: value}
        return result

    def protected_env_vars(self, cfg: dict):
        # Your bootstrap token — no source (including yours) may ever overwrite it.
        return frozenset({"MYVAULT_TOKEN"})
```

### 契约规则（强制执行，非建议）

- **`fetch()` 绝不抛出异常。** 错误放入 `result.error` + `result.error_kind`。如果 fetch 抛出了异常，会被编排器捕获并报告为 `INTERNAL` —— 这是契约违规，不是功能特性。
- **`fetch()` 绝不交互提示。** 启动过程运行在非 TTY 环境中（网关、定时任务、Docker）。`run_secret_cli()` 会关闭 stdin，使交互式提示工具快速失败。交互式认证应放在你的 CLI 安装流程中，绝不应在启动路径上。
- **同步执行，在预算内完成。** 编排器强制执行挂钟超时（默认 120 秒，用户可通过 `secrets.<name>.timeout_seconds` 调整）。超时会报告 `TIMEOUT`，你的结果会被丢弃。
- **你获取值；编排器应用值。** 返回你*打算*贡献的映射。永远不要自己写 `os.environ` —— 那会绕过优先级、冲突检测和来源追踪。
- **API 版本控制。** `SecretSource.api_version` 默认为当前的 `SECRET_SOURCE_API_VERSION`。注册表会跳过（带警告）构建于不同版本的源，而不是让启动崩溃。

### 选择你的 `shape`

- `mapped` —— 用户在配置中显式绑定环境变量名到引用（类似 1Password 的 `env:` 映射）。意图最强：在有冲突的变量上，mapped 声明优先于 bulk 声明。
- `bulk` —— 你隐式注入整个项目/文件夹的密钥（类似 Bitwarden BSM）。让位于 mapped 源。

### 可选钩子

| 方法 | 默认值 | 何时重写 |
|---|---|---|
| `is_enabled(cfg)` | `cfg.get("enabled")` | 自定义激活逻辑 |
| `override_existing(cfg)` | `cfg.get("override_existing", False)` | 你想用不同的默认值（两个内置源默认为 `True` 以支持轮换） |
| `protected_env_vars(cfg)` | 空集 | 你有引导令牌（你几乎肯定有） |
| `fetch_timeout_seconds(cfg)` | 120 秒 | 你的后端需要不同的超时 |
| `config_schema()` | `{}` | 为安装界面声明配置键 |

## 子进程安全：使用 `run_secret_cli()`

如果你的后端需要调用 CLI，请使用共享辅助函数而不是直接调用 `subprocess.run`。它免费为你提供审计级别的安全姿态：仅 argv 参数（无 `shell=True`）、**最小化的白名单子进程环境**（当源运行时，`os.environ` 持有 Hermes 已知的所有凭据 —— 绝不能将这些交给子进程）、`NO_COLOR` + ANSI 清洗的 stderr、关闭的 stdin、超时 → 干净的 `RuntimeError`。在 argv 中使用 `--` 终止符来传递用户提供的引用字符串，这样它们就不会被解析为命令行参数。

## 注册

```python
# __init__.py
def register(ctx):
    ctx.register_secret_source(MyVaultSource())
```

注册会被拒绝（带日志警告，不会崩溃）的情况包括：非 `SecretSource` 实例、无效/重复的名称、`scheme` 已被其他源占用、错误的 `api_version`、或 `shape` 不在 `mapped`/`bulk` 范围内。

:::note 时序说明
插件发现运行在启动时首次 `load_hermes_dotenv()` 调用之后，因此发现它的进程不会在首次 env 加载时查询插件源。但此后所有由 Hermes 派生的进程（网关子进程、定时任务会话、子代理）都会查询它。内置源覆盖了首次进程的引导。
:::

## 用户配置方式与其他源一样

```yaml
secrets:
  sources: [myvault, bitwarden]   # optional ordering
  myvault:
    enabled: true
    # ... your config_schema keys
```

多源优先级、冲突警告和 `(from My Vault)` 来源标签都会自动工作 —— 请参阅[用户端 secrets 文档](/user-guide/secrets/)了解优先级规则。

## 使用一致性测试套件验证

在插件的测试中继承 Hermes 仓库的测试套件（`tests/secret_sources/conformance.py`）：

```python
import pytest
from tests.secret_sources.conformance import SecretSourceConformance

class TestMyVaultConformance(SecretSourceConformance):
    @pytest.fixture
    def source(self):
        return MyVaultSource()
```

它会检查那些一旦违反就会影响他人的规则：畸形配置时不抛异常、机器可读的错误类型、默认禁用、正数超时、有效的受保护变量名，以及完整的 `apply_all()` 往返测试。绿色通过（全部合格）是认定一个后端符合契约的标准。

## ErrorKind 参考

| 类型 | 含义 |
|---|---|
| `NOT_CONFIGURED` | 已启用但缺少令牌/项目/映射 |
| `BINARY_MISSING` | 辅助 CLI 未找到或不可执行 |
| `AUTH_FAILED` / `AUTH_EXPIRED` | 凭据错误/过期 |
| `REF_INVALID` | 密钥引用验证失败 |
| `NETWORK` | 传输层错误 |
| `EMPTY_VALUE` | 后端对某个引用返回了空值 —— 永远不要用 `""` 覆盖有效的凭据 |
| `TIMEOUT` | 获取超时 |
| `INTERNAL` | 其他一切（Bug、意外的 shape） |

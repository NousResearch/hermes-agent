---
sidebar_position: 9
---

# 添加平台适配器

本指南涵盖向 Hermes 网关添加新的消息平台。平台适配器将 Hermes 连接到外部消息服务（Telegram、Discord、WeCom 等），以便用户可以通过该服务与代理交互。

:::tip
有两种方式添加平台：
- **插件**（推荐用于社区/第三方）：将插件目录放入 `~/.hermes/plugins/` — 无需修改核心代码。参见下面的[插件路径](#插件路径（推荐）)。
- **内置**：修改 20+ 个跨代码、配置和文档的文件。使用下面的[内置清单](#步骤清单）。
:::

## 架构概述

```
用户 ↔ 消息平台 ↔ 平台适配器 ↔ 网关运行器 ↔ AIAgent
```

每个适配器扩展自 `gateway/platforms/base.py` 的 `BasePlatformAdapter`，并实现：

- **`connect()`** — 建立连接（WebSocket、长轮询、HTTP 服务器等）*（抽象）*
- **`disconnect()`** — 干净关闭 *（抽象）*
- **`send()`** — 向聊天发送文本消息 *（抽象）*
- **`send_typing()`** — 显示打字指示器（可选覆盖）
- **`get_chat_info()`** — 返回聊天元数据（可选覆盖）

入站消息由适配器接收并通过 `self.handle_message(event)` 转发，基类将其路由到网关运行器。

## 插件路径（推荐）

插件系统允许您在不修改任何核心 Hermes 代码的情况下添加平台适配器。您的插件是一个包含两个文件的目录：

```
~/.hermes/plugins/my-platform/
  PLUGIN.yaml      # 插件元数据
  adapter.py       # 适配器类 + register() 入口点
```

### PLUGIN.yaml

插件元数据。`requires_env` 和 `optional_env` 块自动填充 `hermes config` UI 条目（参见下面的[在 Hermes Config 中暴露 Env 变量](#在-hermes-config-中暴露-env-变量)）。

```yaml
name: my-platform
label: My Platform
kind: platform
version: 1.0.0
description: My custom messaging platform adapter
author: Your Name
requires_env:
  - MY_PLATFORM_TOKEN          # bare string works
  - name: MY_PLATFORM_CHANNEL  # or rich dict for better UX
    description: "Channel to join"
    prompt: "Channel"
    password: false
optional_env:
  - name: MY_PLATFORM_HOME_CHANNEL
    description: "Default channel for cron delivery"
    password: false
```

### adapter.py

```python
import os
from gateway.platforms.base import (
    BasePlatformAdapter, SendResult, MessageEvent, MessageType,
)
from gateway.config import Platform, PlatformConfig


class MyPlatformAdapter(BasePlatformAdapter):
    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("my_platform"))
        extra = config.extra or {}
        self.token = os.getenv("MY_PLATFORM_TOKEN") or extra.get("token", "")

    async def connect(self) -> bool:
        # Connect to the platform API, start listeners
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        # Send message via platform API
        return SendResult(success=True, message_id="...")

    async def get_chat_info(self, chat_id):
        return {"name": chat_id, "type": "dm"}


def check_requirements() -> bool:
    return bool(os.getenv("MY_PLATFORM_TOKEN"))


def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    return bool(os.getenv("MY_PLATFORM_TOKEN") or extra.get("token"))


def _env_enablement() -> dict | None:
    token = os.getenv("MY_PLATFORM_TOKEN", "").strip()
    channel = os.getenv("MY_PLATFORM_CHANNEL", "").strip()
    if not (token and channel):
        return None
    seed = {"token": token, "channel": channel}
    home = os.getenv("MY_PLATFORM_HOME_CHANNEL")
    if home:
        seed["home_channel"] = {"chat_id": home, "name": "Home"}
    return seed


def register(ctx):
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="my_platform",
        label="My Platform",
        adapter_factory=lambda cfg: MyPlatformAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        required_env=["MY_PLATFORM_TOKEN"],
        install_hint="pip install my-platform-sdk",
        # Env-driven auto-configuration — seeds PlatformConfig.extra from
        # env vars before adapter construction. See "Env-Driven Auto-
        # Configuration" section below.
        env_enablement_fn=_env_enablement,
        # Cron home-channel delivery support. Lets deliver=my_platform cron
        # jobs route without editing cron/scheduler.py. See "Cron Delivery"
        # section below.
        cron_deliver_env_var="MY_PLATFORM_HOME_CHANNEL",
        # Per-platform user authorization env vars
        allowed_users_env="MY_PLATFORM_ALLOWED_USERS",
        allow_all_env="MY_PLATFORM_ALLOW_ALL_USERS",
        # Message length limit for smart chunking (0 = no limit)
        max_message_length=4000,
        # LLM guidance injected into system prompt
        platform_hint=(
            "You are chatting via My Platform. "
            "It supports markdown formatting."
        ),
        # Display
        emoji="💬",
    )

    # Optional: register platform-specific tools
    ctx.register_tool(
        name="my_platform_search",
        toolset="my_platform",
        schema={...},
        handler=my_search_handler,
    )
```

### 配置

用户在 `config.yaml` 中配置平台：

```yaml
gateway:
  platforms:
    my_platform:
      enabled: true
      extra:
        token: "..."
        channel: "#general"
```

或通过环境变量（适配器在 `__init__` 中读取）。

### 插件系统自动处理的内容

当您调用 `ctx.register_platform()` 时，以下集成点会为您自动处理 — 无需修改核心代码：

| 集成点 | 工作原理 |
|---|---|
| 网关适配器创建 | 如果/elif 链之前检查注册表 |
| 配置解析 | `Platform._missing_()` 接受任何平台名称 |
| 已连接平台验证 | 调用注册表 `validate_config()` |
| 用户授权 | 检查 `allowed_users_env` / `allow_all_env` |
| 仅 Env 自动启用 | `env_enablement_fn` 在适配器构造前填充 `PlatformConfig.extra` + `home_channel` |
| Cron 投递 | `cron_deliver_env_var` 使 `deliver=<name>` 生效 |
| `hermes config` UI 条目 | `plugin.yaml` 中的 `requires_env` / `optional_env` 自动填充 |
| send_message 工具 | 通过活动网关适配器路由 |
| Webhook 跨平台投递 | 检查已知平台的注册表 |
| `/update` 命令访问 | `allow_update_command` 标志 |
| 频道目录 | 插件平台包含在枚举中 |
| 系统提示词提示 | `platform_hint` 注入到 LLM 上下文中 |
| 消息分块 | `max_message_length` 用于智能分割 |
| PII 去标识化 | `pii_safe` 标志 |
| `hermes status` | 显示带 `(plugin)` 标签的插件平台 |
| `hermes gateway setup` | 插件平台出现在设置菜单中 |
| `hermes tools` / `hermes skills` | 每个平台的插件平台配置 |
| Token 锁（多配置文件） | 在 `connect()` 中使用 `acquire_scoped_lock()` |
| 孤立配置警告 | 插件缺失时的描述性日志 |

## 环境驱动的自动配置

大多数用户通过将 env 变量放入 `~/.hermes/.env` 而不是编辑 `config.yaml` 来设置平台。`env_enablement_fn` 钩子让您的插件**在适配器构造之前**获取这些 env 变量，这样 `hermes gateway status`、`get_connected_platforms()` 和 cron 投递可以看到正确的状态，而无需实例化平台 SDK。

```python
def _env_enablement() -> dict | None:
    """Seed PlatformConfig.extra from env vars.

    Called by the platform registry during load_gateway_config().
    Return None when the platform isn't minimally configured — the
    caller then skips auto-enabling. Return a dict to seed extras.

    The special 'home_channel' key is extracted and becomes a proper
    HomeChannel dataclass on the PlatformConfig; every other key is
    merged into PlatformConfig.extra.
    """
    token = os.getenv("MY_PLATFORM_TOKEN", "").strip()
    channel = os.getenv("MY_PLATFORM_CHANNEL", "").strip()
    if not (token and channel):
        return None
    seed = {"token": token, "channel": channel}
    home = os.getenv("MY_PLATFORM_HOME_CHANNEL")
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("MY_PLATFORM_HOME_CHANNEL_NAME", "Home"),
        }
    return seed


def register(ctx):
    ctx.register_platform(
        name="my_platform",
        label="My Platform",
        adapter_factory=lambda cfg: MyPlatformAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        env_enablement_fn=_env_enablement,
        # ... other fields
    )
```

## Cron 投递

要让 `deliver=my_platform` cron 作业路由到配置的主频道，请将 `cron_deliver_env_var` 设置为保存默认聊天/房间/频道 ID 的 env 变量名称：

```python
ctx.register_platform(
    name="my_platform",
    ...
    cron_deliver_env_var="MY_PLATFORM_HOME_CHANNEL",
)
```

调度器在解析 `deliver=my_platform` 作业的主目标时读取此 env 变量，并将平台视为 `_KNOWN_DELIVERY_PLATFORMS` 样式检查中的有效 cron 目标。如果您的 `env_enablement_fn` 填充了 `home_channel` 字典（见上文），则优先使用 — `cron_deliver_env_var` 是 env 填充之前运行的 cron 作业的后备。

## 在 `hermes config` 中暴露 Env 变量

`hermes_cli/config.py` 在导入时扫描 `plugins/platforms/*/plugin.yaml` 并从 `requires_env` 和（可选）`optional_env` 块自动填充 `OPTIONAL_ENV_VARS`。使用富字典形式贡献适当的描述、提示、密码标志和 URL — CLI 设置 UI 免费获取它们。

```yaml
# plugins/platforms/my_platform/plugin.yaml
name: my_platform-platform
label: My Platform
kind: platform
version: 1.0.0
description: >
  My Platform gateway adapter for Hermes Agent.
author: Your Name
requires_env:
  - name: MY_PLATFORM_TOKEN
    description: "Bot API token from the My Platform console"
    prompt: "My Platform bot token"
    url: "https://my-platform.example.com/bots"
    password: true
  - name: MY_PLATFORM_CHANNEL
    description: "Channel to join (e.g. #hermes)"
    prompt: "Channel"
    password: false
optional_env:
  - name: MY_PLATFORM_HOME_CHANNEL
    description: "Default channel for cron delivery (defaults to MY_PLATFORM_CHANNEL)"
    prompt: "Home channel (or empty)"
    password: false
  - name: MY_PLATFORM_ALLOWED_USERS
    description: "Comma-separated user IDs allowed to talk to the bot"
    prompt: "Allowed users (comma-separated)"
    password: false
```

**支持的字典键：** `name`（必需）、`description`、`prompt`、`url`、`password`（布尔值；省略时从 `*_TOKEN` / `*_SECRET` / `*_KEY` / `*_PASSWORD` / `*_JSON` 后缀自动检测）、`category`（默认为 `"messaging"`）。

裸字符串条目（`- MY_PLATFORM_TOKEN`）仍然有效 — 它们获得从插件 `label` 派生的通用描述。如果 `OPTIONAL_ENV_VARS` 中已存在同一变量的硬编码条目，则优先（向后兼容）；plugin.yaml 形式作为后备。

### 参考实现

参见仓库中的 `plugins/platforms/irc/` 获取完整的工作示例 — 一个没有外部依赖的完整异步 IRC 适配器。

---

## 步骤清单（内置路径）

:::note
此清单用于将平台直接添加到 Hermes 核心代码库 — 通常由核心贡献者为官方支持的平台执行。社区/第三方平台应使用上面的[插件路径](#插件路径（推荐）)。
:::

### 1. 平台枚举

在 `gateway/config.py` 的 `Platform` 枚举中添加您的平台：

```python
class Platform(str, Enum):
    # ... existing platforms ...
    NEWPLAT = "newplat"
```

### 2. 适配器文件

创建 `gateway/platforms/newplat.py`：

```python
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter, MessageEvent, MessageType, SendResult,
)

def check_newplat_requirements() -> bool:
    """Return True if dependencies are available."""
    return SOME_SDK_AVAILABLE

class NewPlatAdapter(BasePlatformAdapter):
    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.NEWPLAT)
        # Read config from config.extra dict
        extra = config.extra or {}
        self._api_key = extra.get("api_key") or os.getenv("NEWPLAT_API_KEY", "")

    async def connect(self) -> bool:
        # Set up connection, start polling/webhook
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        self._running = False
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        # Send message via platform API
        return SendResult(success=True, message_id="...")

    async def get_chat_info(self, chat_id):
        return {"name": chat_id, "type": "dm"}
```

对于入站消息，构建 `MessageEvent` 并调用 `self.handle_message(event)`：

```python
source = self.build_source(
    chat_id=chat_id,
    chat_name=name,
    chat_type="dm",  # or "group"
    user_id=user_id,
    user_name=user_name,
)
event = MessageEvent(
    text=content,
    message_type=MessageType.TEXT,
    source=source,
    message_id=msg_id,
)
await self.handle_message(event)
```

### 3. 网关配置（`gateway/config.py`）

三个接触点：

1. **`get_connected_platforms()`** — 添加对平台必需凭据的检查
2. **`load_gateway_config()`** — 添加 token env 映射条目：`Platform.NEWPLAT: "NEWPLAT_TOKEN"`
3. **`_apply_env_overrides()`** — 映射所有 `NEWPLAT_*` env 变量到配置

### 4. 网关运行器（`gateway/run.py`）

五个接触点：

1. **`_create_adapter()`** — 添加 `elif platform == Platform.NEWPLAT:` 分支
2. **`_is_user_authorized()` allowed_users map** — `Platform.NEWPLAT: "NEWPLAT_ALLOWED_USERS"`
3. **`_is_user_authorized()` allow_all map** — `Platform.NEWPLAT: "NEWPLAT_ALLOW_ALL_USERS"`
4. **早期 env 检查 `_any_allowlist` 元组** — 添加 `"NEWPLAT_ALLOWED_USERS"`
5. **早期 env 检查 `_allow_all` 元组** — 添加 `"NEWPLAT_ALLOW_ALL_USERS"`
6. **`_UPDATE_ALLOWED_PLATFORMS` frozenset** — 添加 `Platform.NEWPLAT`

### 5. 跨平台投递

1. **`gateway/platforms/webhook.py`** — 在投递类型元组中添加 `"newplat"`
2. **`cron/scheduler.py`** — 在 `_KNOWN_DELIVERY_PLATFORMS` frozenset 和 `_deliver_result()` 平台映射中添加

### 6. CLI 集成

1. **`hermes_cli/config.py`** — 将所有 `NEWPLAT_*` 变量添加到 `_EXTRA_ENV_KEYS`
2. **`hermes_cli/gateway.py`** — 在 `_PLATFORMS` 列表中添加条目，包含 key、label、emoji、token_var、setup_instructions 和 vars
3. **`hermes_cli/platforms.py`** — 添加 `PlatformInfo` 条目，包含 label 和 default_toolset（用于 `skills_config` 和 `tools_config` TUI）
4. **`hermes_cli/setup.py`** — 添加 `_setup_newplat()` 函数（可以委托给 `gateway.py`）并在消息平台列表中添加元组
5. **`hermes_cli/status.py`** — 添加平台检测条目：`"NewPlat": ("NEWPLAT_TOKEN", "NEWPLAT_HOME_CHANNEL")`
6. **`hermes_cli/dump.py`** — 在平台检测字典中添加 `"newplat": "NEWPLAT_TOKEN"`

### 7. 工具

1. **`tools/send_message_tool.py`** — 在平台映射中添加 `"newplat": Platform.NEWPLAT`
2. **`tools/cronjob_tools.py`** — 在投递目标描述字符串中添加 `newplat`

### 8. 工具集

1. **`toolsets.py`** — 添加带有 `_HERMES_CORE_TOOLS` 的 `"hermes-newplat"` 工具集定义
2. **`toolsets.py`** — 将 `"hermes-newplat"` 添加到 `"hermes-gateway"` includes 列表

### 9. 可选：平台提示

**`agent/prompt_builder.py`** — 如果您的平台有特定的渲染限制（不支持 markdown、消息长度限制等），在 `_PLATFORM_HINTS` 字典中添加条目。这会将特定于平台的指导注入系统提示：

```python
_PLATFORM_HINTS = {
    # ...
    "newplat": (
        "You are chatting via NewPlat. It supports markdown formatting "
        "but has a 4000-character message limit."
    ),
}
```

并非所有平台都需要提示 — 只有在代理行为应该不同时才添加。

### 10. 测试

创建 `tests/gateway/test_newplat.py` 覆盖：

- 从配置构造适配器
- 消息事件构建
- 发送方法（模拟外部 API）
- 平台特定功能（加密、路由等）

### 11. 文档

| 文件 | 添加内容 |
|------|---------|
| `website/docs/user-guide/messaging/newplat.md` | 完整的平台设置页面 |
| `website/docs/user-guide/messaging/index.md` | 平台比较表、架构图、工具集表、安全部分、下一步链接 |
| `website/docs/reference/environment-variables.md` | 所有 NEWPLAT_* env 变量 |
| `website/docs/reference/toolsets-reference.md` | hermes-newplat 工具集 |
| `website/docs/integrations/index.md` | 平台链接 |
| `website/sidebars.ts` | 文档页面的侧边栏条目 |
| `website/docs/developer-guide/architecture.md` | 适配器计数和列表 |
| `website/docs/developer-guide/gateway-internals.md` | 适配器文件列表 |

## 同等审计

在将新平台 PR 标记为完成之前，针对已建立的平台运行同等审计：

```bash
# Find every .py file mentioning the reference platform
search_files "bluebubbles" output_mode="files_only" file_glob="*.py"

# Find every .py file mentioning the new platform
search_files "newplat" output_mode="files_only" file_glob="*.py"

# Any file in the first set but not the second is a potential gap
```

对 `.md` 和 `.ts` 文件重复。调查每个差距 — 是平台枚举（需要更新）还是平台特定引用（跳过）？

## 常见模式

### 长轮询适配器

如果您的适配器使用长轮询（如 Telegram 或微信），使用轮询循环任务：

```python
async def connect(self):
    self._poll_task = asyncio.create_task(self._poll_loop())
    self._mark_connected()

async def _poll_loop(self):
    while self._running:
        messages = await self._fetch_updates()
        for msg in messages:
            await self.handle_message(self._build_event(msg))
```

### 回调/Webhook 适配器

如果平台向您的端点推送消息（如 WeCom 回调），运行 HTTP 服务器：

```python
async def connect(self):
    self._app = web.Application()
    self._app.router.add_post("/callback", self._handle_callback)
    # ... start aiohttp server
    self._mark_connected()

async def _handle_callback(self, request):
    event = self._build_event(await request.text())
    await self._message_queue.put(event)
    return web.Response(text="success")  # Acknowledge immediately
```

对于有严格响应期限的平台（如 WeCom 的 5 秒限制），始终立即确认，稍后通过 API 主动投递代理的回复。代理会话运行 3–30 分钟 — 在回调响应窗口内进行内联回复不可行。

### Token 锁

如果适配器使用唯一凭据保持持久连接，添加作用域锁以防止两个配置文件使用相同凭据：

```python
from gateway.status import acquire_scoped_lock, release_scoped_lock

async def connect(self):
    if not acquire_scoped_lock("newplat", self._token):
        logger.error("Token already in use by another profile")
        return False
    # ... connect

async def disconnect(self):
    release_scoped_lock("newplat", self._token)
```

## 参考实现

| 适配器 | 模式 | 复杂度 | 适合参考 |
|--------|------|--------|---------|
| `bluebubbles.py` | REST + webhook | 中等 | 简单的 REST API 集成 |
| `weixin.py` | 长轮询 + CDN | 高 | 媒体处理、加密 |
| `wecom_callback.py` | 回调/webhook | 中等 | HTTP 服务器、AES 加密、多应用 |
| `telegram.py` | 长轮询 + Bot API | 高 | 全功能适配器，带群组、线程 |

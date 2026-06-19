---
sidebar_position: 9
---

# 添加平台适配器

本指南介绍如何向 Hermes gateway 添加新的消息平台。平台适配器将 Hermes 连接到外部消息服务（Telegram、Discord、WeCom 等），使用户可以通过该服务与 agent 交互。

:::tip
添加平台有两种方式：
- **Plugin**（推荐用于社区/第三方）：将 plugin 目录放入 `~/.hermes/plugins/` — 无需修改任何核心代码。参见下方 [Plugin 路径](#plugin-path-recommended)。
- **内置**：需修改代码、配置和文档共 20+ 个文件。参见下方 [内置清单](#step-by-step-checklist)。
:::

## 架构概览

```
用户 ↔ 消息平台 ↔ 平台适配器 ↔ Gateway Runner ↔ AIAgent
```

每个适配器都继承自 `gateway/platforms/base.py` 中的 `BasePlatformAdapter`，并实现以下方法：

- **`connect()`** — 建立连接（WebSocket、长轮询、HTTP 服务器等）*(抽象方法)*
- **`disconnect()`** — 清理关闭 *(抽象方法)*
- **`send()`** — 向聊天发送文本消息 *(抽象方法)*
- **`send_typing()`** — 显示正在输入指示器（可选覆盖）
- **`get_chat_info()`** — 返回聊天元数据（可选覆盖）

适配器接收入站消息后，通过 `self.handle_message(event)` 转发，基类将其路由到 gateway runner。

## Plugin 路径（推荐）{#plugin-path-recommended}

Plugin 系统允许你在不修改任何 Hermes 核心代码的情况下添加平台适配器。你的 plugin 是一个包含两个文件的目录：

```
~/.hermes/plugins/my-platform/
  plugin.yaml      # Plugin 元数据
  adapter.py       # 适配器类 + register() 入口点
```

### plugin.yaml

Plugin 元数据。`requires_env` 和 `optional_env` 块会自动填充 `hermes config` UI 条目（参见下方[在 hermes config 中暴露环境变量](#surfacing-env-vars-in-hermes-config)）。

```yaml
name: my-platform
label: My Platform
kind: platform
version: 1.0.0
description: My custom messaging platform adapter
author: Your Name
requires_env:
  - MY_PLATFORM_TOKEN          # 裸字符串有效
  - name: MY_PLATFORM_CHANNEL  # 或使用富字典以获得更好的 UX
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
        # 连接到平台 API，启动监听器
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        # 通过平台 API 发送消息
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
    """Plugin 入口点 — 由 Hermes plugin 系统调用。"""
    ctx.register_platform(
        name="my_platform",
        label="My Platform",
        adapter_factory=lambda cfg: MyPlatformAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        required_env=["MY_PLATFORM_TOKEN"],
        install_hint="pip install my-platform-sdk",
        # 环境变量驱动的自动配置 — 在适配器构建前从环境变量
        # 填充 PlatformConfig.extra。参见下方"环境变量驱动的自动配置"章节。
        env_enablement_fn=_env_enablement,
        # Cron 主频道投递支持。允许 deliver=my_platform 的 cron 任务
        # 无需编辑 cron/scheduler.py 即可路由。参见下方"Cron 投递"章节。
        cron_deliver_env_var="MY_PLATFORM_HOME_CHANNEL",
        # 每平台用户授权环境变量
        allowed_users_env="MY_PLATFORM_ALLOWED_USERS",
        allow_all_env="MY_PLATFORM_ALLOW_ALL_USERS",
        # 智能分块的消息长度限制（0 = 无限制）
        max_message_length=4000,
        # 注入系统 prompt（提示词）的 LLM 指导
        platform_hint=(
            "You are chatting via My Platform. "
            "It supports markdown formatting."
        ),
        # 显示
        emoji="💬",
    )

    # 可选：注册平台专属工具
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

### Plugin 系统自动处理的内容

调用 `ctx.register_platform()` 时，以下集成点将自动处理 — 无需修改核心代码：

| 集成点 | 工作方式 |
|---|---|
| Gateway 适配器创建 | 在内置 if/elif 链之前检查注册表 |
| 配置解析 | `Platform._missing_()` 接受任意平台名称 |
| 已连接平台验证 | 调用注册表中的 `validate_config()` |
| 用户授权 | 检查 `allowed_users_env` / `allow_all_env` |
| 仅环境变量自动启用 | `env_enablement_fn` 填充 `PlatformConfig.extra` + `home_channel` |
| YAML 配置桥接 | `apply_yaml_config_fn` 将 `config.yaml` 键转换为环境变量/extras |
| Cron 投递 | `cron_deliver_env_var` 使 `deliver=<name>` 生效 |
| `hermes config` UI 条目 | `plugin.yaml` 中的 `requires_env` / `optional_env` 自动填充 |
| send_message 工具 | 通过实时 gateway 适配器路由 |
| Webhook 跨平台投递 | 检查注册表中的已知平台 |
| `/update` 命令访问 | `allow_update_command` 标志 |
| 频道目录 | Plugin 平台包含在枚举中 |
| 系统 prompt 提示 | `platform_hint` 注入 LLM 上下文 |
| 消息分块 | `max_message_length` 用于智能分割 |
| PII 脱敏 | `pii_safe` 标志 |
| `hermes status` | 显示带 `(plugin)` 标签的 plugin 平台 |
| `hermes gateway setup` | Plugin 平台出现在设置菜单中 |
| `hermes tools` / `hermes skills` | Plugin 平台出现在每平台配置中 |
| Token 锁（多配置文件） | 在 `connect()` 中使用 `acquire_scoped_lock()` |
| 孤立配置警告 | Plugin 缺失时输出描述性日志 |

## 环境变量驱动的自动配置

大多数用户通过将环境变量写入 `~/.hermes/.env` 来配置平台，而不是编辑 `config.yaml`。`env_enablement_fn` hook 允许你的 plugin 在适配器构建**之前**读取这些环境变量，使 `hermes gateway status`、`get_connected_platforms()` 和 cron 投递无需实例化平台 SDK 即可看到正确状态。

```python
def _env_enablement() -> dict | None:
    """从环境变量填充 PlatformConfig.extra。

    在 load_gateway_config() 期间由平台注册表调用。
    当平台未完成最低配置时返回 None — 调用方将跳过自动启用。
    返回字典以填充 extras。

    特殊键 'home_channel' 会被提取并成为 PlatformConfig 上的
    HomeChannel dataclass；其他所有键合并到 PlatformConfig.extra 中。
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
        # ... 其他字段
    )
```


## YAML→env 配置桥接

部分用户更倾向于设置 `config.yaml` 键（`my_platform.require_mention`、`my_platform.allowed_channels` 等）而非环境变量。`apply_yaml_config_fn` hook 允许你的 plugin 自行处理这一转换，而无需强制核心 `gateway/config.py` 了解你平台的 YAML schema。

```python
import os

def _apply_yaml_config(yaml_cfg: dict, platform_cfg: dict) -> dict | None:
    """将 config.yaml 中的 `my_platform:` 键转换为环境变量/extras。

    yaml_cfg     — 完整的顶层解析后 config.yaml 字典
    platform_cfg — 平台自身的子字典（yaml_cfg.get("my_platform", {})）

    可直接修改 os.environ（使用 `not os.getenv(...)` 守卫以保持
    环境变量 > YAML 的优先级），也可返回字典合并到 PlatformConfig.extra 中。
    返回 None 或 {} 表示无额外内容。
    """
    if "require_mention" in platform_cfg and not os.getenv("MY_PLATFORM_REQUIRE_MENTION"):
        os.environ["MY_PLATFORM_REQUIRE_MENTION"] = str(platform_cfg["require_mention"]).lower()
    allowed = platform_cfg.get("allowed_channels")
    if allowed is not None and not os.getenv("MY_PLATFORM_ALLOWED_CHANNELS"):
        if isinstance(allowed, list):
            allowed = ",".join(str(v) for v in allowed)
        os.environ["MY_PLATFORM_ALLOWED_CHANNELS"] = str(allowed)
    return None  # 无需合并到 PlatformConfig.extra 的额外内容

def register(ctx):
    ctx.register_platform(
        name="my_platform",
        ...,
        apply_yaml_config_fn=_apply_yaml_config,
    )
```

该 hook 在 `load_gateway_config()` 期间，于通用共享键循环（处理 `unauthorized_dm_behavior`、`notice_delivery`、`reply_prefix`、`require_mention` 等公共键）之后、`_apply_env_overrides()` 之前调用，因此你的 plugin 只需桥接**平台专属**键。

hook 内抛出的异常会被捕获并以 debug 级别记录 — 行为异常的 plugin 不会中止 gateway 配置加载。


## Cron 投递

要让 `deliver=my_platform` 的 cron 任务路由到已配置的主频道，将 `cron_deliver_env_var` 设置为持有默认聊天/房间/频道 ID 的环境变量名：

```python
ctx.register_platform(
    name="my_platform",
    ...
    cron_deliver_env_var="MY_PLATFORM_HOME_CHANNEL",
)
```

调度器在解析 `deliver=my_platform` 任务的主目标时会读取此环境变量，并将该平台视为 `_KNOWN_DELIVERY_PLATFORMS` 风格检查中的有效 cron 目标。如果你的 `env_enablement_fn` 填充了 `home_channel` 字典（见上文），则优先使用该值 — `cron_deliver_env_var` 是在环境变量填充之前运行的 cron 任务的回退方案。

### 进程外 cron 投递

`cron_deliver_env_var` 使你的平台成为可识别的 `deliver=` 目标。要在 cron 任务运行于独立进程（即 `hermes cron run` 与 `hermes gateway` 分离）时使实际发送成功，需注册 `standalone_sender_fn`：

```python
async def _standalone_send(
    pconfig,
    chat_id,
    message,
    *,
    thread_id=None,
    media_files=None,
    force_document=False,
):
    """建立临时连接/获取新 token，发送消息，然后关闭。"""
    # ... 建立连接，发送消息，返回结果 ...
    return {"success": True, "message_id": "..."}
    # 或 {"error": "..."}

ctx.register_platform(
    name="my_platform",
    ...
    cron_deliver_env_var="MY_PLATFORM_HOME_CHANNEL",
    standalone_sender_fn=_standalone_send,
)
```

为何需要此 hook：内置平台（Telegram、Discord、Slack 等）在 `tools/send_message_tool.py` 中内置了直接 REST 辅助函数，使 cron 无需在同一进程中持有 gateway 即可投递。Plugin 平台历史上依赖 `_gateway_runner_ref()`，该函数在 gateway 进程外返回 `None`，因此若没有 `standalone_sender_fn`，cron 端发送会失败并报 `No live adapter for platform '<name>'`。

该函数接收与实时适配器相同的 `pconfig` 和 `chat_id`，以及可选的 `thread_id`、`media_files` 和 `force_document` 关键字参数。返回 `{"success": True, "message_id": ...}` 视为成功投递；返回 `{"error": "..."}` 会将消息记录到 cron 的 `delivery_errors` 中。函数内抛出的异常由调度器捕获并报告为 `Plugin standalone send failed: <reason>`。参考实现位于 `plugins/platforms/{irc,teams,google_chat}/adapter.py`。

## 在 `hermes config` 中暴露环境变量 {#surfacing-env-vars-in-hermes-config}

`hermes_cli/config.py` 在导入时扫描 `plugins/platforms/*/plugin.yaml`，并从 `requires_env` 和（可选的）`optional_env` 块自动填充 `OPTIONAL_ENV_VARS`。使用富字典形式可提供完整的描述、prompt、password 标志和 URL — CLI 设置 UI 会自动识别。

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

**支持的字典键：** `name`（必填）、`description`、`prompt`、`url`、`password`（布尔值；当省略时根据 `*_TOKEN` / `*_SECRET` / `*_KEY` / `*_PASSWORD` / `*_JSON` 后缀自动检测）、`category`（默认为 `"messaging"`）。

裸字符串条目（`- MY_PLATFORM_TOKEN`）仍然有效 — 会根据 plugin 的 `label` 自动生成通用描述。如果 `OPTIONAL_ENV_VARS` 中已存在同名变量的硬编码条目，则以硬编码为准（向后兼容）；plugin.yaml 形式作为回退。

## 平台专属慢速 LLM 用户体验

某些平台存在约束，影响慢速 LLM 响应的呈现方式：

- **LINE** 发出单次使用的*回复 token*，在入站事件后约 60 秒过期。使用该 token 回复是免费的；回退到计费的 Push API 则不然。如果 LLM 在截止时间前未完成，选择是"消耗付费 Push 配额"或"在回复 token 过期前用它做些更聪明的事"。
- **WhatsApp** 在 24 小时不活跃后将会话标记为非活跃，此后只接受模板消息。
- **SMS** 没有正在输入指示器或渐进式更新的概念 — 长响应看起来就像 bot 离线了。

这些是 `BasePlatformAdapter` 无法预判的真实约束。Plugin 接口有意为适配器在基础输入循环之上叠加平台专属 UX 留出空间，而无需扩展 kwarg 列表。

### 模式：子类化 `_keep_typing` 以叠加飞行中 UX

`BasePlatformAdapter._keep_typing` 是正在输入指示器的心跳 — 它在 LLM 生成时作为后台任务运行，响应投递后被取消。要在某个阈值时叠加平台专属行为（例如在 45 秒时发送"仍在思考"气泡），在你的适配器中覆盖 `_keep_typing`，在 `super()._keep_typing()` 旁边调度你自己的任务，并在 `finally` 中清理：

```python
class LineAdapter(BasePlatformAdapter):
    async def _keep_typing(self, chat_id: str, *args, **kwargs) -> None:
        if self.slow_response_threshold <= 0:
            await super()._keep_typing(chat_id, *args, **kwargs)
            return

        async def _fire_at_threshold() -> None:
            try:
                await asyncio.sleep(self.slow_response_threshold)
            except asyncio.CancelledError:
                raise
            # 平台专属操作 — 对于 LINE，使用缓存的回复 token 发送
            # Template Buttons "获取答案"气泡，用户可通过 postback
            # 回调中的新（免费）回复 token 稍后获取缓存的响应。
            await self._send_slow_response_button(chat_id)

        side_task = asyncio.create_task(_fire_at_threshold())
        try:
            await super()._keep_typing(chat_id, *args, **kwargs)
        finally:
            if not side_task.done():
                side_task.cancel()
                try:
                    await side_task
                except (asyncio.CancelledError, Exception):
                    pass
```

关键点：

- **始终 `await super()._keep_typing(...)`。** 输入心跳本身有独立价值 — 不要替换它，而是在其上叠加。
- **在 `finally` 中清理副任务。** 当 LLM 完成（或 `/stop` 取消运行）时，gateway 会取消输入任务。你的副任务也必须响应该取消，否则它会残留并可能在响应已投递后触发。
- **配合 `interrupt_session_activity`** 在用户发出 `/stop` 时解决任何孤立 UX 状态。对于 LINE，这意味着将 postback 缓存条目从 `PENDING` 转换为 `ERROR`，使持久的"获取答案"按钮投递"运行已中断"消息而非循环。

### 模式：子类化 `send` 以通过缓存路由而非立即发送

如果你的慢速响应 UX 缓存响应以供稍后检索（LINE 的 postback 流程），你的 `send` 覆盖需要识别三种模式：

1. **此聊天存在待处理的 postback** → 将响应缓存在 request_id 下，不发送任何可见内容。
2. **系统忙碌确认**（`⚡ Interrupting`、`⏳ Queued`、`⏩ Steered`）→ 绕过缓存直接发送，使用户看到 gateway 对其输入的响应。
3. **正常响应** → 按常规通过回复 token 或 Push 发送。

```python
async def send(self, chat_id: str, content: str, **kw) -> SendResult:
    if _is_system_bypass(content):
        return await self._send_text_chunks(chat_id, content, force_push=False)
    pending_rid = self._pending_buttons.get(chat_id)
    if pending_rid:
        self._cache.set_ready(pending_rid, content)
        return SendResult(success=True, message_id=pending_rid)
    return await self._send_text_chunks(chat_id, content, force_push=False)
```

`_SYSTEM_BYPASS_PREFIXES` 是 gateway 自身的忙碌确认前缀（`⚡`、`⏳`、`⏩`、`💾`）。无论缓存 UX 状态如何，始终让这些前缀可见地通过。

### 何时适用此模式

在以下情况使用输入循环覆盖方式：

- 平台的出站 API 存在硬性时间窗口约束（单次使用回复 token、过期的粘性会话等），**且**
- 在该平台上*可见的飞行中气泡*是可接受的 UX。

在以下情况使用更简单的 `slow_response_threshold = 0` 始终 Push 路径：

- 平台没有有意义的免费与付费区别，**或**
- 用户社区更倾向于"加载中……加载中……完成"的静默后响应，而非交互式中间气泡。

LINE 两者都支持：阈值默认为 45 秒用于免费 postback 获取，`LINE_SLOW_RESPONSE_THRESHOLD=0` 恢复为"始终 Push 回退"。

### 参考实现

完整的 LINE postback 实现参见 `plugins/platforms/line/adapter.py` — 包含 `RequestCache` 状态机（`PENDING → READY → DELIVERED`，以及 `/stop` 的 `ERROR`）、在阈值时触发 Template Buttons 气泡的 `_keep_typing` 覆盖、通过缓存路由的 `send` 覆盖，以及解决孤立 PENDING 条目的 `interrupt_session_activity` 覆盖。

### 参考实现（Plugin 路径）

完整的工作示例参见仓库中的 `plugins/platforms/irc/` — 一个无外部依赖的完整异步 IRC 适配器。`plugins/platforms/teams/` 涵盖 Bot Framework / Adaptive Cards，`plugins/platforms/google_chat/` 涵盖基于 OAuth 的 REST API，`plugins/platforms/line/` 涵盖带平台专属慢速 LLM UX 的 webhook 驱动消息 API。

---

## 分步清单（内置路径）{#step-by-step-checklist}

:::note
此清单用于将平台直接添加到 Hermes 核心代码库 — 通常由核心贡献者为官方支持的平台执行。社区/第三方平台应使用上方的 [Plugin 路径](#plugin-path-recommended)。
:::

### 1. Platform 枚举

在 `gateway/config.py` 的 `Platform` 枚举中添加你的平台：

```python
class Platform(str, Enum):
    # ... 现有平台 ...
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
    """如果依赖可用则返回 True。"""
    return SOME_SDK_AVAILABLE

class NewPlatAdapter(BasePlatformAdapter):
    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.NEWPLAT)
        # 从 config.extra 字典读取配置
        extra = config.extra or {}
        self._api_key = extra.get("api_key") or os.getenv("NEWPLAT_API_KEY", "")

    async def connect(self) -> bool:
        # 建立连接，启动轮询/webhook
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        self._running = False
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        # 通过平台 API 发送消息
        return SendResult(success=True, message_id="...")

    async def get_chat_info(self, chat_id):
        return {"name": chat_id, "type": "dm"}
```

对于入站消息，构建 `MessageEvent` 并调用 `self.handle_message(event)`：

```python
source = self.build_source(
    chat_id=chat_id,
    chat_name=name,
    chat_type="dm",  # 或 "group"
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

### 3. Gateway 配置（`gateway/config.py`）

三个接触点：

1. **`get_connected_platforms()`** — 添加对你平台所需凭据的检查
2. **`load_gateway_config()`** — 添加 token 环境变量映射条目：`Platform.NEWPLAT: "NEWPLAT_TOKEN"`
3. **`_apply_env_overrides()`** — 将所有 `NEWPLAT_*` 环境变量映射到配置

### 4. Gateway Runner（`gateway/run.py`）

五个接触点：

1. **`_create_adapter()`** — 添加 `elif platform == Platform.NEWPLAT:` 分支
2. **`_is_user_authorized()` allowed_users 映射** — `Platform.NEWPLAT: "NEWPLAT_ALLOWED_USERS"`
3. **`_is_user_authorized()` allow_all 映射** — `Platform.NEWPLAT: "NEWPLAT_ALLOW_ALL_USERS"`
4. **早期环境检查 `_any_allowlist` 元组** — 添加 `"NEWPLAT_ALLOWED_USERS"`
5. **早期环境检查 `_allow_all` 元组** — 添加 `"NEWPLAT_ALLOW_ALL_USERS"`
6. **`_UPDATE_ALLOWED_PLATFORMS` frozenset** — 添加 `Platform.NEWPLAT`

### 5. 跨平台投递

1. **`gateway/platforms/webhook.py`** — 将 `"newplat"` 添加到投递类型元组
2. **`cron/scheduler.py`** — 添加到 `_KNOWN_DELIVERY_PLATFORMS` frozenset 和 `_deliver_result()` 平台映射

### 6. CLI 集成

1. **`hermes_cli/config.py`** — 将所有 `NEWPLAT_*` 变量添加到 `_EXTRA_ENV_KEYS`
2. **`hermes_cli/gateway.py`** — 在 `_PLATFORMS` 列表中添加条目，包含 key、label、emoji、token_var、setup_instructions 和 vars
3. **`hermes_cli/platforms.py`** — 添加带 label 和 default_toolset 的 `PlatformInfo` 条目（供 `skills_config` 和 `tools_config` TUI 使用）
4. **`hermes_cli/setup.py`** — 添加 `_setup_newplat()` 函数（可委托给 `gateway.py`）并将元组添加到消息平台列表
5. **`hermes_cli/status.py`** — 添加平台检测条目：`"NewPlat": ("NEWPLAT_TOKEN", "NEWPLAT_HOME_CHANNEL")`
6. **`hermes_cli/dump.py`** — 将 `"newplat": "NEWPLAT_TOKEN"` 添加到平台检测字典

### 7. 工具

1. **`tools/send_message_tool.py`** — 将 `"newplat": Platform.NEWPLAT` 添加到平台映射
2. **`tools/cronjob_tools.py`** — 将 `newplat` 添加到投递目标描述字符串

### 8. Toolset

1. **`toolsets.py`** — 添加带 `_HERMES_CORE_TOOLS` 的 `"hermes-newplat"` toolset 定义
2. **`toolsets.py`** — 将 `"hermes-newplat"` 添加到 `"hermes-gateway"` 的 includes 列表

### 9. 可选：平台提示

**`agent/prompt_builder.py`** — 如果你的平台有特定渲染限制（不支持 markdown、消息长度限制等），在 `_PLATFORM_HINTS` 字典中添加条目。这会将平台专属指导注入系统 prompt：

```python
_PLATFORM_HINTS = {
    # ...
    "newplat": (
        "You are chatting via NewPlat. It supports markdown formatting "
        "but has a 4000-character message limit."
    ),
}
```

并非所有平台都需要提示 — 仅在 agent 行为应有所不同时添加。

### 10. 测试

创建 `tests/gateway/test_newplat.py`，覆盖：

- 从配置构建适配器
- 消息事件构建
- 发送方法（mock 外部 API）
- 平台专属功能（加密、路由等）

### 11. 文档

| 文件 | 需添加内容 |
|------|-------------|
| `website/docs/user-guide/messaging/newplat.md` | 完整的平台设置页面 |
| `website/docs/user-guide/messaging/index.md` | 平台对比表、架构图、toolset 表、安全章节、下一步链接 |
| `website/docs/reference/environment-variables.md` | 所有 NEWPLAT_* 环境变量 |
| `website/docs/reference/toolsets-reference.md` | hermes-newplat toolset |
| `website/docs/integrations/index.md` | 平台链接 |
| `website/sidebars.ts` | 文档页面的侧边栏条目 |
| `website/docs/developer-guide/architecture.md` | 适配器数量 + 列表 |
| `website/docs/developer-guide/gateway-internals.md` | 适配器文件列表 |

## 一致性审计

在将新平台 PR 标记为完成之前，对照已有平台进行一致性审计：

```bash
# 查找所有提及参考平台的 .py 文件
search_files "bluebubbles" output_mode="files_only" file_glob="*.py"

# 查找所有提及新平台的 .py 文件
search_files "newplat" output_mode="files_only" file_glob="*.py"

# 在第一个集合中但不在第二个集合中的文件是潜在的遗漏点
```

对 `.md` 和 `.ts` 文件重复上述操作。逐一排查每个遗漏点 — 是平台枚举（需要更新）还是平台专属引用（可跳过）？

## 常见模式

### 长轮询适配器

如果你的适配器使用长轮询（如 Telegram 或 Weixin），使用轮询循环任务：

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

如果平台将消息推送到你的端点（如 WeCom 回调），运行 HTTP 服务器：

```python
async def connect(self):
    self._app = web.Application()
    self._app.router.add_post("/callback", self._handle_callback)
    # ... 启动 aiohttp 服务器
    self._mark_connected()

async def _handle_callback(self, request):
    event = self._build_event(await request.text())
    await self._message_queue.put(event)
    return web.Response(text="success")  # 立即确认
```

对于有严格响应截止时间的平台（例如 WeCom 的 5 秒限制），始终立即确认，稍后通过 API 主动投递 agent 的回复。Agent 会话运行 3–30 分钟 — 在回调响应窗口内内联回复是不可行的。

### Token 锁

如果适配器持有带唯一凭据的持久连接，添加作用域锁以防止两个配置文件使用相同凭据：

```python
from gateway.status import acquire_scoped_lock, release_scoped_lock

async def connect(self):
    if not acquire_scoped_lock("newplat", self._token):
        logger.error("Token already in use by another profile")
        return False
    # ... 连接

async def disconnect(self):
    release_scoped_lock("newplat", self._token)
```

## 参考实现

| 适配器 | 模式 | 复杂度 | 适合参考的场景 |
|---------|---------|------------|-------------------|
| `bluebubbles.py` | REST + webhook | 中 | 简单 REST API 集成 |
| `weixin.py` | 长轮询 + CDN | 高 | 媒体处理、加密 |
| `wecom_callback.py` | 回调/webhook | 中 | HTTP 服务器、AES 加密、多应用 |
| `telegram.py` | 长轮询 + Bot API | 高 | 支持群组、线程的全功能适配器 |
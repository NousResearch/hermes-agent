---
title: 电脑操控
sidebar_position: 16
---

# 电脑操控（macOS）

Hermes Agent 可以在**后台**驱动你的 Mac 桌面——点击、输入、滚动、拖拽。你的光标不会移动，键盘焦点不会改变，macOS 也不会切换 Spaces。你和 Agent 可以在同一台机器上协同工作。

与大多数电脑操控集成不同，这适用于**任何支持工具调用的模型**——Claude、GPT、Gemini，或本地 vLLM 端点上的开源模型。无需关心 Anthropic 原生 schema。

## 工作原理

`computer_use` 工具集通过 stdio 以 MCP 协议与 [`cua-driver`](https://github.com/trycua/cua) 通信。`cua-driver` 是一个 macOS 驱动，使用 SkyLight 私有 SPI（`SLEventPostToPid`、`SLPSPostEventRecordTo`）以及 `_AXObserverAddNotificationAndCheckRemote` 无障碍 SPI，实现以下功能：

- 直接向目标进程投递合成事件——无需 HID 事件 tap，无需光标跳转。
- 在不提升窗口的情况下切换 AppKit 激活状态——不触发 Space 切换。
- 在窗口被遮挡时保持 Chromium/Electron 无障碍树存活。

这一组合正是 OpenAI Codex「后台电脑操控」所采用的方案。cua-driver 是其开源等价实现。

## 启用

选择最方便的方式——两种方式运行的是同一个上游安装程序：

**方式一：使用专用 CLI 命令（最直接）。**

```
hermes computer-use install
```

此命令会获取并运行上游 cua-driver 安装脚本：
`curl -fsSL https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/scripts/install.sh`。
使用 `hermes computer-use status` 验证安装结果。

**方式二：通过交互式界面启用工具集。**

1. 运行 `hermes tools`，选择 `🖱️ Computer Use (macOS)` → `cua-driver (background)`。
2. 安装程序将运行上游安装脚本（与方式一相同）。

安装完成后，无论采用哪种方式，继续执行以下步骤：

3. 在提示时授予 macOS 权限：
   - **系统设置 → 隐私与安全性 → 辅助功能** → 允许终端（或 Hermes 应用）。
   - **系统设置 → 隐私与安全性 → 屏幕录制** → 允许同一应用。
4. 启动启用了该工具集的会话：
   ```
   hermes -t computer_use chat
   ```
   或在 `~/.hermes/config.yaml` 中将 `computer_use` 添加到已启用的工具集列表。

## 保持 cua-driver 最新

cua-driver 项目会定期发布修复（例如 v0.1.6 修复了 UTM 工作流中的 Safari 窗口焦点问题）。Hermes 在两处刷新二进制文件，避免你停留在过时版本：

- **`hermes update`** — 更新 Hermes 本身时，如果 `cua-driver` 在 PATH 中，更新结束时会重新运行上游安装程序。对非 macOS 用户及未安装 cua-driver 的用户无操作。
- **`hermes computer-use install --upgrade`** — 手动强制刷新。无论 cua-driver 是否已安装，都会重新运行上游安装程序。在不等待下次 Agent 更新的情况下获取最新修复时使用此命令。

`hermes computer-use status` 会在二进制路径旁显示已安装的版本号。

## 快速示例

用户 prompt（提示词）：*「找到我最近一封来自 Stripe 的邮件，总结他们希望我做什么。」*

Agent 的执行计划：

1. `computer_use(action="capture", mode="som", app="Mail")` — 获取 Mail 的截图，其中每个侧边栏项目、工具栏按钮和邮件行均已编号。
2. `computer_use(action="click", element=14)` — 点击搜索框（来自截图的第 #14 号元素）。
3. `computer_use(action="type", text="from:stripe")`
4. `computer_use(action="key", keys="return", capture_after=True)` — 提交并获取新截图。
5. 点击最顶部的结果，读取正文，进行总结。

整个过程中，你的光标保持原位，Mail 窗口始终不会切换到前台。

## 提供商兼容性

| 提供商 | 支持视觉？ | 可用？ | 备注 |
|---|---|---|---|
| Anthropic（Claude Sonnet/Opus 3+） | ✅ | ✅ | 综合表现最佳；支持 SOM 与原始坐标。 |
| OpenRouter（任意视觉模型） | ✅ | ✅ | 支持多部分工具消息。 |
| OpenAI（GPT-4+、GPT-5） | ✅ | ✅ | 同上。 |
| 本地 vLLM / LM Studio（视觉模型） | ✅ | ✅ | 需模型支持多部分工具内容。 |
| 纯文本模型 | ❌ | ✅（降级） | 使用 `mode="ax"` 仅通过无障碍树操作。 |

截图以 OpenAI 风格的 `image_url` 部分内联在工具结果中发送。对于 Anthropic，适配器会将其转换为原生 `tool_result` 图像块。

## 安全性

Hermes 应用多层防护机制：

- 破坏性操作（click、type、drag、scroll、key、focus_app）需要审批——通过 CLI 对话框交互确认，或通过消息平台审批按钮确认。
- 工具层面硬性屏蔽的按键组合：清空废纸篓、强制删除、锁定屏幕、注销、强制注销。
- 硬性屏蔽的输入模式：`curl | bash`、`sudo rm -rf /`、fork bomb 等。
- Agent 的系统 prompt 明确规定：不得点击权限对话框，不得输入密码，不得执行截图中嵌入的指令。

如需对每个操作进行确认，可在 `~/.hermes/config.yaml` 中配置 `approvals.mode: manual`。

## Token 效率

截图开销较大。Hermes 应用四层优化措施：

- **截图淘汰** — Anthropic 适配器在上下文中仅保留最近 3 张截图；较旧的截图替换为 `[screenshot removed to save context]` 占位符。
- **客户端压缩裁剪** — 上下文压缩器检测多模态工具结果，并从旧结果中剥离图像部分。
- **图像感知 token 估算** — 每张图像计为约 1500 个 token（Anthropic 的固定费率），而非其 base64 字符长度。
- **服务端上下文编辑（仅限 Anthropic）** — 激活后，适配器通过 `context_management` 启用 `clear_tool_uses_20250919`，由 Anthropic API 在服务端清除旧工具结果。

在 1568×900 分辨率下执行 20 个操作的会话，截图上下文通常消耗约 3 万个 token，而非约 60 万个。

## 限制

- **仅限 macOS。** cua-driver 使用的私有 Apple SPI 在 Linux 或 Windows 上不存在。跨平台 GUI 自动化请使用 `browser` 工具集。
- **私有 SPI 风险。** Apple 可能在任何 OS 更新中更改 SkyLight 的符号接口。如需在 macOS 版本升级时保持可复现性，请通过 `HERMES_CUA_DRIVER_VERSION` 环境变量固定驱动版本。
- **性能。** 后台模式比前台模式慢——SkyLight 路由事件耗时约 5–20ms，而直接 HID 投递更快。对于 Agent 速度的点击操作无明显影响；若尝试录制速通视频则会有感知。
- **不支持键盘输入密码。** `type` 对命令行 payload 有硬性屏蔽模式；密码请使用系统自动填充功能。

## 配置

覆盖驱动二进制路径（用于测试 / CI）：

```
HERMES_CUA_DRIVER_CMD=/opt/homebrew/bin/cua-driver
HERMES_CUA_DRIVER_VERSION=0.5.0    # optional pin
```

完全替换后端（用于测试）：

```
HERMES_COMPUTER_USE_BACKEND=noop   # records calls, no side effects
```

## 故障排查

**`computer_use backend unavailable: cua-driver is not installed`** — 运行 `hermes computer-use install` 获取 cua-driver 二进制文件，或运行 `hermes tools` 并启用 Computer Use 工具集。

**点击似乎没有效果** — 截图并验证。可能有一个你未注意到的模态框正在阻止输入。使用 `escape` 或关闭按钮将其关闭。

**元素索引已过期** — SOM 索引仅在下次 `capture` 之前有效。任何改变状态的操作后请重新截图。

**「blocked pattern in type text」** — 你尝试 `type` 的文本匹配了危险 shell 模式列表。请拆分命令或重新考虑操作方式。

## 另请参阅

- [通用技能：`macos-computer-use`](https://github.com/NousResearch/hermes-agent/blob/main/skills/apple/macos-computer-use/SKILL.md)
- [cua-driver 源码（trycua/cua）](https://github.com/trycua/cua)
- 跨平台 Web 任务请参阅[浏览器自动化](./browser.md)。
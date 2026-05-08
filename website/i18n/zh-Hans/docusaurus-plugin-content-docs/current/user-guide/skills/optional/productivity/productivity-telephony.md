---
title: "Telephony — 无需修改核心工具即可为 Hermes 添加电话功能"
sidebar_label: "Telephony"
description: "无需修改核心工具即可为 Hermes 添加电话功能"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Telephony

无需修改核心工具即可为 Hermes 添加电话功能。配置和持久化 Twilio 号码，发送和接收短信/彩信，直接拨打电话，以及通过 Bland.ai 或 Vapi 发起 AI 驱动的去电。

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/productivity/telephony` |
| Path | `optional-skills/productivity/telephony` |
| Version | `1.0.0` |
| Author | Nous Research |
| License | MIT |
| Tags | `telephony`, `phone`, `sms`, `mms`, `voice`, `twilio`, `bland.ai`, `vapi`, `calling`, `texting` |
| Related skills | [`maps`](/docs/user-guide/skills/bundled/productivity/productivity-maps), [`google-workspace`](/docs/user-guide/skills/bundled/productivity/productivity-google-workspace), [`agentmail`](/docs/user-guide/skills/optional/email/email-agentmail) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Telephony — 无需修改核心工具的号码、通话和短信功能

此可选技能为 Hermes 提供实用的电话功能，同时将电话功能排除在核心工具列表之外。

它附带一个辅助脚本 `scripts/telephony.py`，可以：
- 将提供商凭证保存到 `~/.hermes/.env`
- 搜索并购买 Twilio 电话号码
- 记住已拥有的号码以供后续会话使用
- 从已拥有的号码发送短信/彩信
- 通过轮询接收该号码的入站短信，无需 webhook 服务器
- 使用 TwiML `<Say>` 或 `<Play>` 进行直接 Twilio 通话
- 将已拥有的 Twilio 号码导入 Vapi
- 通过 Bland.ai 或 Vapi 发起 AI 去电

## 解决的问题

此技能旨在覆盖用户实际需要的实用电话任务：
- 去电
- 发短信
- 拥有一个可复用的代理号码
- 稍后检查发送到该号码的消息
- 在会话之间保留该号码和关联的 ID
- 为入站短信轮询和其他自动化提供面向未来的电话身份

它**不会**将 Hermes 变成实时入站电话网关。入站短信通过轮询 Twilio REST API 处理。这对于许多工作流程（包括通知和一些一次性验证码获取）已经足够，无需添加核心 webhook 基础设施。

## 安全规则——强制执行

1. 拨打电话或发送短信前务必确认。
2. 切勿拨打紧急号码。
3. 切勿将电话功能用于骚扰、垃圾信息、冒充或任何非法活动。
4. 将第三方电话号码视为敏感操作数据：
   - 不要将它们保存到 Hermes 记忆中
   - 不要将它们包含在技能文档、摘要或后续笔记中，除非用户明确要求
5. 持久化**代理拥有的 Twilio 号码**是可以的，因为它是用户配置的一部分。
6. VoIP 号码**不保证**在所有第三方 2FA 流程中都能工作。请谨慎使用并明确告知用户预期。

## 决策树——使用哪个服务？

请使用以下逻辑而非硬编码的提供商路由：

### 1) "我希望 Hermes 拥有一个真实电话号码"
使用 **Twilio**。

原因：
- 购买和保留号码最简单的途径
- 最佳的短信/彩信支持
- 最简单的入站短信轮询方案
- 最清晰的入站 webhook 或通话处理的未来路径

使用场景：
- 稍后接收短信
- 发送部署告警/定时通知
- 为代理维护可复用的电话身份
- 后续尝试基于电话的认证流程

### 2) "我现在只需要最简单的 AI 语音去电"
使用 **Bland.ai**。

原因：
- 最快的设置
- 一个 API 密钥
- 无需先购买/导入号码

权衡：
- 灵活性较低
- 语音质量尚可，但不是最好的

### 3) "我想要最好的对话式 AI 语音质量"
使用 **Twilio + Vapi**。

原因：
- Twilio 提供你拥有的号码
- Vapi 提供更好的对话式 AI 通话质量和更多语音/模型灵活性

推荐流程：
1. 购买/保存 Twilio 号码
2. 将其导入 Vapi
3. 保存返回的 `VAPI_PHONE_NUMBER_ID`
4. 使用 `ai-call --provider vapi`

### 4) "我想用自定义预录语音消息拨打电话"
使用 **Twilio 直接通话** 配合公开音频 URL。

原因：
- 播放自定义 MP3 最简单的方式
- 与 Hermes 的 `text_to_speech` 以及公开文件托管或隧道配合良好

## 文件和持久化状态

此技能在两个位置持久化电话状态：

### `~/.hermes/.env`
用于长期有效的提供商凭证和已拥有号码 ID，例如：
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_PHONE_NUMBER`
- `TWILIO_PHONE_NUMBER_SID`
- `BLAND_API_KEY`
- `VAPI_API_KEY`
- `VAPI_PHONE_NUMBER_ID`
- `PHONE_PROVIDER`（AI 通话提供商：bland 或 vapi）

### `~/.hermes/telephony_state.json`
用于仅限技能的状态，应跨会话保留，例如：
- 记住的默认 Twilio 号码/SID
- 记住的 Vapi 电话号码 ID
- 上次入站消息 SID/日期，用于收件箱轮询检查点

这意味着：
- 下次加载此技能时，`diagnose` 可以告诉你已配置了什么号码
- `twilio-inbox --since-last --mark-seen` 可以从上一个检查点继续

## 定位辅助脚本

安装此技能后，按以下方式定位脚本：

```bash
SCRIPT="$(find ~/.hermes/skills -path '*/telephony/scripts/telephony.py' -print -quit)"
```

如果 `SCRIPT` 为空，说明技能尚未安装。

## 安装

这是一个官方可选技能，从 Skills Hub 安装：

```bash
hermes skills search telephony
hermes skills install official/productivity/telephony
```

## 提供商设置

### Twilio — 拥有号码、短信/彩信、直接通话、入站短信轮询

注册地址：
- https://www.twilio.com/try-twilio

然后将凭证保存到 Hermes：

```bash
python3 "$SCRIPT" save-twilio ACXXXXXXXXXXXXXXXXXXXXXXXXXXXX your_auth_token_here
```

搜索可用号码：

```bash
python3 "$SCRIPT" twilio-search --country US --area-code 702 --limit 5
```

购买并记住一个号码：

```bash
python3 "$SCRIPT" twilio-buy "+17025551234" --save-env
```

列出已拥有的号码：

```bash
python3 "$SCRIPT" twilio-owned
```

稍后将其中一个设为默认：

```bash
python3 "$SCRIPT" twilio-set-default "+17025551234" --save-env
# 或
python3 "$SCRIPT" twilio-set-default PNXXXXXXXXXXXXXXXXXXXXXXXXXXXX --save-env
```

### Bland.ai — 最简单的 AI 语音去电

注册地址：
- https://app.bland.ai

保存配置：

```bash
python3 "$SCRIPT" save-bland your_bland_api_key --voice mason
```

### Vapi — 更好的对话式语音质量

注册地址：
- https://dashboard.vapi.ai

先保存 API 密钥：

```bash
python3 "$SCRIPT" save-vapi your_vapi_api_key
```

将已拥有的 Twilio 号码导入 Vapi 并持久化返回的电话号码 ID：

```bash
python3 "$SCRIPT" vapi-import-twilio --save-env
```

如果你已知 Vapi 电话号码 ID，可以直接保存：

```bash
python3 "$SCRIPT" save-vapi your_vapi_api_key --phone-number-id vapi_phone_number_id_here
```

## 诊断当前状态

随时检查此技能已知的信息：

```bash
python3 "$SCRIPT" diagnose
```

在后续会话恢复工作时，请首先使用此命令。

## 常用工作流程

### A. 购买代理号码并稍后继续使用

1. 保存 Twilio 凭证：
```bash
python3 "$SCRIPT" save-twilio AC... auth_token_here
```

2. 搜索号码：
```bash
python3 "$SCRIPT" twilio-search --country US --area-code 702 --limit 10
```

3. 购买并保存到 `~/.hermes/.env` 和状态中：
```bash
python3 "$SCRIPT" twilio-buy "+17025551234" --save-env
```

4. 下次会话运行：
```bash
python3 "$SCRIPT" diagnose
```
这将显示记住的默认号码和收件箱检查点状态。

### B. 从代理号码发送短信

```bash
python3 "$SCRIPT" twilio-send-sms "+15551230000" "Your deployment completed successfully."
```

带媒体：

```bash
python3 "$SCRIPT" twilio-send-sms "+15551230000" "Here is the chart." --media-url "https://example.com/chart.png"
```

### C. 稍后无需 webhook 服务器即可检查入站短信

轮询默认 Twilio 号码的收件箱：

```bash
python3 "$SCRIPT" twilio-inbox --limit 20
```

仅显示上一个检查点之后到达的消息，并在阅读完成后推进检查点：

```bash
python3 "$SCRIPT" twilio-inbox --since-last --mark-seen
```

这是"下次加载技能时如何访问号码收到的消息"的主要解决方案。

### D. 使用内置 TTS 进行直接 Twilio 通话

```bash
python3 "$SCRIPT" twilio-call "+15551230000" --message "Hello! This is Hermes calling with your status update." --voice Polly.Joanna
```

### E. 使用预录/自定义语音消息拨打电话

这是复用 Hermes 现有 `text_to_speech` 支持的主要路径。

适用于以下场景：
- 希望通话使用 Hermes 配置的 TTS 语音而非 Twilio `<Say>`
- 希望进行单向语音传递（简报、告警、笑话、提醒、状态更新）
- **不需要**实时对话式电话通话

单独生成或托管音频，然后：

```bash
python3 "$SCRIPT" twilio-call "+155****0000" --audio-url "https://example.com/briefing.mp3"
```

推荐的 Hermes TTS → Twilio Play 工作流程：

1. 使用 Hermes `text_to_speech` 生成音频。
2. 使生成的 MP3 可公开访问。
3. 使用 `--audio-url` 拨打 Twilio 电话。

示例代理流程：
- 让 Hermes 使用 `text_to_speech` 创建消息音频
- 如需要，通过临时静态主机/隧道/对象存储 URL 公开文件
- 使用 `twilio-call --audio-url ...` 通过电话投递

MP3 的推荐托管方案：
- 临时公开的对象/存储 URL
- 到本地静态文件服务器的短期隧道
- 电话提供商可直接获取的任何现有 HTTPS URL

重要说明：
- Hermes TTS 非常适合预录去电消息
- Bland/Vapi 更适合**实时对话式 AI 通话**，因为它们自行处理实时电话音频栈
- 这里并未将 Hermes STT/TTS 用作全双工电话对话引擎；那将需要比重型流式/webhook 集成多得多的基础设施

### F. 使用 Twilio 直接通话导航电话树/IVR

如果你需要在通话接通后按数字，请使用 `--send-digits`。
Twilio 将 `w` 解释为短等待。

```bash
python3 "$SCRIPT" twilio-call "+18005551234" --message "Connecting to billing now." --send-digits "ww1w2w3"
```

这在转接给人工或发送简短状态消息之前到达特定菜单分支时很有用。

### G. 使用 Bland.ai 进行 AI 语音去电

```bash
python3 "$SCRIPT" ai-call "+15551230000" "Call the dental office, ask for a cleaning appointment on Tuesday afternoon, and if they do not have Tuesday availability, ask for Wednesday or Thursday instead." --provider bland --voice mason --max-duration 3
```

检查状态：

```bash
python3 "$SCRIPT" ai-status <call_id> --provider bland
```

完成后向 Bland 提出分析问题：

```bash
python3 "$SCRIPT" ai-status <call_id> --provider bland --analyze "Was the appointment confirmed?,What date and time?,Any special instructions?"
```

### H. 使用 Vapi 通过你拥有的号码进行 AI 语音去电

1. 将 Twilio 号码导入 Vapi：
```bash
python3 "$SCRIPT" vapi-import-twilio --save-env
```

2. 拨打电话：
```bash
python3 "$SCRIPT" ai-call "+15551230000" "You are calling to make a dinner reservation for two at 7:30 PM. If that is unavailable, ask for the nearest time between 6:30 and 8:30 PM." --provider vapi --max-duration 4
```

3. 检查结果：
```bash
python3 "$SCRIPT" ai-status <call_id> --provider vapi
```

## 建议的代理流程

当用户要求打电话或发短信时：

1. 通过决策树确定哪种路径适合该请求。
2. 如果配置状态不清楚，运行 `diagnose`。
3. 收集完整的任务详情。
4. 在拨打电话或发送短信之前与用户确认。
5. 使用正确的命令。
6. 如需要则轮询结果。
7. 总结结果，不要将第三方电话号码持久化到 Hermes 记忆中。

## 此技能目前仍不支持的功能

- 实时入站通话接听
- 基于 webhook 的实时短信推送到代理循环
- 对任意第三方 2FA 提供商的保证支持

这些功能需要比纯可选技能更多的基础设施。

## 注意事项

- Twilio 试用账户和地区规则可能会限制你可以呼叫/发送短信的对象。
- 某些服务会拒绝 VoIP 号码用于 2FA。
- `twilio-inbox` 轮询 REST API；它不是即时推送。
- Vapi 去电仍然依赖有效的已导入号码。
- Bland 最简单，但声音不总是最好的。
- 请勿在 Hermes 记忆中存储任意第三方电话号码。

## 验证清单

设置完成后，你应该能够仅使用此技能完成以下所有操作：

1. `diagnose` 显示提供商就绪状态和已记住的状态
2. 搜索并购买 Twilio 号码
3. 将该号码持久化到 `~/.hermes/.env`
4. 从已拥有的号码发送短信
5. 稍后轮询已拥有号码的入站短信
6. 拨打直接 Twilio 通话
7. 通过 Bland 或 Vapi 拨打 AI 通话

## 参考资料

- Twilio 电话号码：https://www.twilio.com/docs/phone-numbers/api
- Twilio 消息：https://www.twilio.com/docs/messaging/api/message-resource
- Twilio 语音：https://www.twilio.com/docs/voice/api/call-resource
- Vapi 文档：https://docs.vapi.ai/
- Bland.ai：https://app.bland.ai/

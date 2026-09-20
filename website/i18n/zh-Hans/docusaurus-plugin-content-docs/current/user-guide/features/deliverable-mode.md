---
title: 可交付成果模式（聊天中的 Artifacts）
sidebar_label: 可交付成果模式
description: Agent 如何将生成的图表、PDF、电子表格及其他文件作为原生附件发送到消息平台。
---

# 可交付成果模式

当 Hermes Agent 在消息 gateway（Slack、Discord、Telegram、WhatsApp、Signal 等）中运行时，它可以将生成的文件直接发送到聊天中——不是让用户自行复制路径，而是作为原生附件。

图表以内联图片形式显示。PDF 报告以文件下载形式显示。电子表格以 `.xlsx` 格式上传。Agent 无需写入 `MEDIA:` 标签或进行任何特殊操作——只需生成文件并在回复中提及其绝对路径。Gateway 会从文本中提取路径，将其从可见消息中移除，并原生上传文件。

## 工作原理

三个部分协同配合：

1. **Agent 拥有可生成文件的工具。** `execute_code` 用于通过 matplotlib 生成图表，`latex-pdf-report` skill 用于生成 PDF，`powerpoint` skill 用于生成演示文稿，`image_generate` 用于生成图片，`text_to_speech` 用于生成音频，等等。

2. **Gateway 扫描 agent 回复中的文件路径。** 任何以支持扩展名结尾的绝对路径（`/tmp/...`）或相对主目录路径（`~/...`）都会被提取。代码块和内联代码中的路径会被忽略，以避免代码示例被破坏。

3. **Gateway 按文件类型分发。** 在平台支持的情况下，图片以内联方式嵌入；视频以内联方式嵌入；音频路由至语音/音频附件；其他所有内容作为文件附件上传。

## 支持的文件扩展名

| 类别 | 扩展名 | 发送方式 |
|---|---|---|
| 图片 | `.png .jpg .jpeg .gif .webp .bmp .tiff .svg` | 内联嵌入 |
| 视频 | `.mp4 .mov .avi .mkv .webm` | 内联嵌入（平台支持时） |
| 音频 | `.mp3 .wav .ogg .m4a .flac` | 语音/音频附件 |
| 文档 | `.pdf .docx .doc .odt .rtf .txt .md` | 文件上传 |
| 数据 | `.xlsx .xls .csv .tsv .json .xml .yaml .yml` | 文件上传 |
| 演示文稿 | `.pptx .ppt .odp` | 文件上传 |
| 压缩包 | `.zip .tar .gz .tgz .bz2 .7z` | 文件上传 |
| Web | `.html .htm` | 文件上传 |

`.py`、`.log` 及其他源文件扩展名被有意排除，以防 agent 自动发送任意源文件；如需向用户发送代码，请使用代码块。

## 引导 Agent 生成 Artifacts

Agent 默认不会主动生成 artifacts——需要明确告知。有两种方式：

**单次会话：** 明确提出请求（"以图表形式发给我对比结果"、"将数据以 CSV 格式返回"），或编写自定义指令/个性化条目，使其在消息平台上倾向于以 artifact 形式回复。

**项目级别：** 将偏好设置添加到项目中的 `AGENTS.md` / `CLAUDE.md` / `.cursorrules`（agent 从该项目工作），或添加到 `~/.hermes/config.yaml` 中 `agent.custom_instructions` 下的全局自定义指令。

Agent 需要使用的机制很简单：将文件渲染到绝对路径（例如 `/tmp/q3-revenue.png`），并在回复中以纯文本形式提及该路径。Gateway 负责其余工作。围栏代码块或反引号中的路径会被忽略，以避免代码示例被破坏。

## Kanban：Artifacts 随完成通知一并发送

如果使用 Hermes 的 kanban（看板）多 agent 工作流，worker 可以在调用 `kanban_complete` 时附加可交付文件：

```python
kanban_complete(
    summary="rendered Q3 revenue chart and report",
    artifacts=[
        "/tmp/q3-revenue.png",
        "/tmp/q3-report.pdf",
    ],
)
```

当 gateway 通知器将"任务完成"消息发送给在 Slack/Telegram 等平台订阅该任务的用户时，也会将每个 artifact 作为原生附件上传到对应聊天中。用户在同一位置获得可交付成果和摘要。

通知器运行时磁盘上不存在的文件会被静默跳过。

## 通过 MCP 连接更多服务

除 artifact 发送管道外，agent 还可以通过 MCP（Model Context Protocol，模型上下文协议）接入其他服务。MCP 生态系统为大多数主流工具提供了社区服务器——按需安装：

| 服务 | 解锁功能 |
|---|---|
| **Notion** | 读写 Notion 页面、数据库，查询工作区 |
| **GitHub** | Issues、PR、评论、超出 gh CLI 范围的仓库搜索 |
| **Linear** | 工单、项目、迭代周期 |
| **Slack** | 工作区全局搜索、读取其他频道 |
| **Gmail** | 收件箱整理、发送邮件、标签管理 |
| **Salesforce** | 线索、商机、账户数据 |
| **Snowflake / BigQuery** | 对数据仓库执行 SQL |
| **Google Drive** | 文件搜索、内容读取、共享管理 |

通过 `~/.hermes/config.yaml` 中的 `mcp_servers` 部分安装 MCP 服务器。完整配置指南请参阅 [MCP 集成](./mcp.md)。

## 与 Perplexity Computer in Slack 的对比

Perplexity Computer 的 Slack 集成基于相同理念：agent 生成可交付成果（图表、PDF、幻灯片），并将其作为原生附件发回线程。Hermes Agent 的可交付成果模式在本地提供相同的用户体验：

- 生成在用户自己的 venv/沙箱中进行（无远程租户）。
- 文件通过相同的 Slack `files.uploadV2` API 发送到聊天。
- 连接器广度通过 MCP 实现，而非精心策划的 400 个托管集成目录——按需安装所需的即可。

OAuth token 保存在用户本机的 `auth.json` / `.env` 中。无托管 token 存储。无多租户 microVM。最终效果相同。
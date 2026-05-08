---
title: "YouTube 内容 — YouTube 字幕转摘要、推文、博客"
sidebar_label: "YouTube 内容"
description: "YouTube 字幕转摘要、推文、博客"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# YouTube 内容

YouTube 字幕转摘要、推文、博客。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/media/youtube-content` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此技能时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# YouTube 内容工具

## 使用场景

当用户分享 YouTube URL 或视频链接、要求总结视频、请求字幕或想要从任何 YouTube 视频中提取和重新格式化内容时使用。将字幕转换为结构化内容（章节、摘要、推文、博客文章）。

从 YouTube 视频中提取字幕并转换为有用的格式。

## 设置

```bash
pip install youtube-transcript-api
```

## 辅助脚本

`SKILL_DIR` 是包含此 SKILL.md 文件的目录。脚本接受任何标准 YouTube URL 格式、短链接（youtu.be）、shorts、嵌入、直播链接或原始 11 字符视频 ID。

```bash
# JSON 输出带元数据
python3 SKILL_DIR/scripts/fetch_transcript.py "https://youtube.com/watch?v=VIDEO_ID"

# 纯文本（适合管道传给后续处理）
python3 SKILL_DIR/scripts/fetch_transcript.py "URL" --text-only

# 带时间戳
python3 SKILL_DIR/scripts/fetch_transcript.py "URL" --timestamps

# 指定语言并带回退链
python3 SKILL_DIR/scripts/fetch_transcript.py "URL" --language tr,en
```

## 输出格式

获取字幕后，根据用户要求格式化：

- **章节**：按主题转换分组，输出带时间戳的章节列表
- **摘要**：整个视频的简明 5-10 句概述
- **章节摘要**：每个章节带有简短段落摘要的章节
- **推文**：Twitter/X 推文格式 — 编号帖子，每条不超过 280 字符
- **博客文章**：包含标题、章节和关键要点的完整文章
- **引用**：带时间戳的值得注意的引用

### 示例 — 章节输出

```
00:00 引言 — 主持人开场提出问题
03:45 背景 — 先前的工作及为什么现有解决方案不足
12:20 核心方法 — 提议方法的详细讲解
24:10 结果 — 基准比较和关键要点
31:55 问答 — 关于可扩展性和后续步骤的观众问题
```

## 工作流

1. **获取**字幕，使用辅助脚本带 `--text-only --timestamps` 参数。
2. **验证**：确认输出非空且为预期语言。如果为空，尝试不带 `--language` 重新获取。如果仍然为空，告诉用户该视频可能已禁用字幕。
3. **需要时分块**：如果字幕超过约 50K 字符，拆分为重叠的块（约 40K 带 2K 重叠），分别总结后合并。
4. **转换**为请求的输出格式。如果用户未指定格式，默认使用摘要。
5. **验证**：重新阅读转换后的输出，检查连贯性、正确的时间戳和完整性后再展示。

## 错误处理

- **字幕已禁用**：告诉用户；建议他们在视频页面上检查是否有字幕可用。
- **私密/不可用视频**：转达错误并请用户验证 URL。
- **无匹配语言**：尝试不带 `--language` 重新获取，然后告知用户实际语言。
- **缺少依赖**：运行 `pip install youtube-transcript-api` 后重试。

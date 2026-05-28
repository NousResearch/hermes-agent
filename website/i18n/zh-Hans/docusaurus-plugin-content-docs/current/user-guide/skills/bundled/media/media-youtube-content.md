---
title: "Youtube Content — YouTube 视频转文字摘要、推文、博客"
sidebar_label: "Youtube Content"
description: "YouTube 视频转文字摘要、推文、博客"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Youtube Content

YouTube 视频转文字摘要、推文、博客。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/media/youtube-content` |
| 平台 | linux, macos, windows |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# YouTube Content Tool

## 使用时机

当用户分享 YouTube URL 或视频链接、要求总结视频、请求获取文字稿，或希望提取并重新格式化任意 YouTube 视频内容时使用。可将文字稿转换为结构化内容（章节、摘要、推文线程、博客文章）。

从 YouTube 视频中提取文字稿并将其转换为实用格式。

## 安装

```bash
pip install youtube-transcript-api
```

## 辅助脚本

`SKILL_DIR` 是包含此 SKILL.md 文件的目录。该脚本接受任何标准 YouTube URL 格式、短链接（youtu.be）、Shorts、嵌入链接、直播链接，或原始 11 位视频 ID。

```bash
# JSON 输出（含元数据）
python3 SKILL_DIR/scripts/fetch_transcript.py "https://youtube.com/watch?v=VIDEO_ID"

# 纯文本输出（适合管道传递给后续处理）
python3 SKILL_DIR/scripts/fetch_transcript.py "URL" --text-only

# 带时间戳
python3 SKILL_DIR/scripts/fetch_transcript.py "URL" --timestamps

# 指定语言并设置回退链
python3 SKILL_DIR/scripts/fetch_transcript.py "URL" --language tr,en
```

## 输出格式

获取文字稿后，根据用户需求选择以下格式：

- **章节（Chapters）**：按主题转换分组，输出带时间戳的章节列表
- **摘要（Summary）**：对整个视频进行 5–10 句的简洁概述
- **章节摘要（Chapter summaries）**：各章节附带简短段落摘要
- **推文线程（Thread）**：Twitter/X 线程格式——编号帖子，每条不超过 280 字符
- **博客文章（Blog post）**：含标题、各节及关键要点的完整文章
- **引用（Quotes）**：带时间戳的精彩引用

### 示例——章节输出

```
00:00 Introduction — host opens with the problem statement
03:45 Background — prior work and why existing solutions fall short
12:20 Core method — walkthrough of the proposed approach
24:10 Results — benchmark comparisons and key takeaways
31:55 Q&A — audience questions on scalability and next steps
```

## 工作流程

1. **获取**：使用辅助脚本并加上 `--text-only --timestamps` 参数获取文字稿。
2. **验证**：确认输出非空且语言符合预期。若为空，去掉 `--language` 参数重试以获取任意可用文字稿。若仍为空，告知用户该视频可能已禁用文字稿。
3. **分块（如需）**：若文字稿超过约 50K 字符，将其拆分为有重叠的块（约 40K，重叠 2K），逐块摘要后再合并。
4. **转换**：将内容转换为用户请求的输出格式。若用户未指定格式，默认输出摘要。
5. **校验**：重新阅读转换后的输出，在呈现前检查连贯性、时间戳准确性及完整性。

## 错误处理

- **文字稿已禁用**：告知用户；建议其在视频页面检查字幕是否可用。
- **视频不可用或为私密视频**：转达错误信息，请用户核实 URL。
- **无匹配语言**：去掉 `--language` 参数重试以获取任意可用文字稿，并向用户说明实际语言。
- **缺少依赖**：执行 `pip install youtube-transcript-api` 后重试。
---
title: "Memento Flashcards — 间隔重复抽认卡系统"
sidebar_label: "Memento Flashcards"
description: "间隔重复抽认卡系统"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Memento Flashcards

间隔重复抽认卡系统。从事实或文本创建卡片，通过代理评分的自由文本回答与抽认卡交互，从 YouTube 字幕生成测验，通过自适应调度复习到期卡片，以及以 CSV 格式导出/导入卡片组。

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/productivity/memento-flashcards` |
| Path | `optional-skills/productivity/memento-flashcards` |
| Version | `1.0.0` |
| Author | Memento AI |
| License | MIT |
| Platforms | macos, linux |
| Tags | `Education`, `Flashcards`, `Spaced Repetition`, `Learning`, `Quiz`, `YouTube` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Memento Flashcards — 间隔重复抽认卡技能

## 概述

Memento 提供了一个基于本地文件的抽认卡系统，支持间隔重复调度。用户可以通过自由文本回答问题，由代理评分后安排下次复习。

适用于以下场景：

- **记住事实** —— 将任何陈述转化为问答卡片
- **间隔重复学习** —— 通过自适应间隔和代理评分的自由文本答案复习到期卡片
- **从 YouTube 视频生成测验** —— 获取字幕并生成 5 道测验题
- **管理卡片组** —— 将卡片组织到集合中，导出/导入 CSV

所有卡片数据存储在单个 JSON 文件中。无需外部 API 密钥——由你（代理）直接生成抽认卡内容和测验问题。

Memento Flashcards 的用户响应风格：
- 仅使用纯文本。在给用户的回复中不要使用 Markdown 格式。
- 保持复习和测验反馈简短且中立。避免额外的表扬、鼓励或长篇解释。

## 使用时机

当用户需要以下操作时使用此技能：
- 将事实保存为抽认卡以备后续复习
- 使用间隔重复复习到期卡片
- 从 YouTube 视频字幕生成测验
- 导入、导出、检查或删除抽认卡数据

不要将此技能用于一般性问答、编码帮助或非记忆类任务。

## 快速参考

| 用户意图 | 操作 |
|---|---|
| "记住 X" / "把这个保存为抽认卡" | 生成问答卡片，调用 `memento_cards.py add` |
| 发送事实但未提及抽认卡 | 询问"要我把这个保存为 Memento 抽认卡吗？"——仅在确认后创建 |
| "创建一个抽认卡" | 询问问题、答案、集合名称；调用 `memento_cards.py add` |
| "复习我的卡片" | 调用 `memento_cards.py due`，逐张展示卡片 |
| "用 [YouTube URL] 考考我" | 调用 `youtube_quiz.py fetch VIDEO_ID`，生成 5 道问题，调用 `memento_cards.py add-quiz` |
| "导出我的卡片" | 调用 `memento_cards.py export --output PATH` |
| "从 CSV 导入卡片" | 调用 `memento_cards.py import --file PATH --collection NAME` |
| "显示我的统计" | 调用 `memento_cards.py stats` |
| "删除一张卡片" | 调用 `memento_cards.py delete --id ID` |
| "删除一个集合" | 调用 `memento_cards.py delete-collection --collection NAME` |

## 卡片存储

卡片存储在以下 JSON 文件中：

```
~/.hermes/skills/productivity/memento-flashcards/data/cards.json
```

**切勿直接编辑此文件。** 始终使用 `memento_cards.py` 子命令。脚本使用原子写入（先写入临时文件，再重命名）以防止数据损坏。

该文件在首次使用时自动创建。

## 操作流程

### 从事实创建卡片

### 激活规则

并非每个事实性陈述都应该成为抽认卡。请使用此三层检查：

1. **明确意图** —— 用户提及"memento"、"flashcard"、"记住这个"、"保存这张卡片"、"添加一张卡片"等明确请求创建抽认卡的表述 → **直接创建卡片**，无需确认。
2. **隐含意图** —— 用户发送了一个事实性陈述但未提及抽认卡（例如"光速是 299,792 km/s"）→ **先询问**："要我把这个保存为 Memento 抽认卡吗？"仅在用户确认后创建。
3. **无意图** —— 消息是编码任务、问题、指令、正常对话或任何明显不需要记忆的事实 → **完全不要激活此技能**。让其他技能或默认行为处理。

当激活确认后（第一层直接创建，第二层确认后创建），生成一张抽认卡：

**第一步：** 将陈述转化为问答对。使用以下内部格式：

```
将事实性陈述转化为正反面配对。
仅返回两行：
Q: <问题文本>
A: <答案文本>

陈述: "{statement}"
```

规则：
- 问题应测试对关键事实的回忆
- 答案应简洁明了

**第二步：** 调用脚本存储卡片：

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py add \
  --question "What year did World War 2 end?" \
  --answer "1945" \
  --collection "History"
```

如果用户未指定集合，默认使用 `"General"`。

脚本输出 JSON 确认已创建的卡片。

### 手动创建卡片

当用户明确要求创建抽认卡时，询问以下信息：
1. 问题（卡片正面）
2. 答案（卡片背面）
3. 集合名称（可选——默认为 `"General"`）

然后按上述方式调用 `memento_cards.py add`。

### 复习到期卡片

当用户想要复习时，获取所有到期卡片：

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py due
```

返回一个 JSON 数组，包含 `next_review_at <= now` 的卡片。如果需要按集合筛选：

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py due --collection "History"
```

**复习流程（自由文本评分）：**

以下是你必须遵循的精确交互模式示例。用户回答后，你评分、告知正确答案，然后对卡片评级。

**交互示例：**

> **代理：** 柏林墙是哪一年倒塌的？
>
> **用户：** 1991
>
> **代理：** 不太对。柏林墙于 1989 年倒塌。下次复习时间是明天。
> *(代理调用: memento_cards.py rate --id ABC --rating hard --user-answer "1991")*
>
> 下一题：谁第一个登上月球？

**规则：**

1. 只展示问题。等待用户回答。
2. 收到回答后，与预期答案进行比较并评分：
   - **正确** —— 用户答对了关键事实（即使措辞不同）
   - **部分正确** —— 方向正确但缺少核心细节
   - **错误** —— 答错或答非所问
3. **你必须告知用户正确答案及其表现。** 保持简短和纯文本。使用以下格式：
   - 正确："正确。答案：{答案}。下次复习在 7 天后。"
   - 部分正确："接近了。答案：{答案}。{遗漏的内容}。下次复习在 3 天后。"
   - 错误："不太对。答案：{答案}。下次复习在明天。"
4. 然后调用评分命令：正确→easy，部分正确→good，错误→hard。
5. 然后展示下一题。

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py rate \
  --id CARD_ID --rating easy --user-answer "what the user said"
```

**切勿跳过第 3 步。** 用户必须始终在进入下一张卡片之前看到正确答案和反馈。

如果没有到期卡片，告诉用户："目前没有需要复习的卡片。稍后再来查看！"

**退役覆盖：** 用户随时可以说"退役这张卡片"以将其永久移出复习。使用 `--rating retire` 实现。

### 间隔重复算法

评分决定下次复习间隔：

| 评分 | 间隔 | ease_streak | 状态变化 |
|---|---|---|---|
| **hard** | +1 天 | 重置为 0 | 保持学习中 |
| **good** | +3 天 | 重置为 0 | 保持学习中 |
| **easy** | +7 天 | +1 | 如果 ease_streak >= 3 → 已退役 |
| **retire** | 永久 | 重置为 0 | → 已退役 |

- **学习中**：卡片正在积极轮换
- **已退役**：卡片不再出现在复习中（用户已掌握或手动退役）
- 连续 3 次"easy"评分自动退役一张卡片

### YouTube 测验生成

当用户发送 YouTube URL 并希望生成测验时：

**第一步：** 从 URL 中提取视频 ID（例如从 `https://www.youtube.com/watch?v=dQw4w9WgXcQ` 中提取 `dQw4w9WgXcQ`）。

**第二步：** 获取字幕：

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/youtube_quiz.py fetch VIDEO_ID
```

返回 `{"title": "...", "transcript": "..."}` 或错误信息。

如果脚本报告 `missing_dependency`，告诉用户安装：
```bash
pip install youtube-transcript-api
```

**第三步：** 从字幕生成 5 道测验题。使用以下规则：

```
你正在为一个播客节目创建 5 道测验题。
仅返回一个包含恰好 5 个对象的 JSON 数组。
每个对象必须包含 'question' 和 'answer' 键。

选择标准：
- 优先选择重要的、令人惊讶的或基础性的事实。
- 跳过填充内容、明显细节和需要大量上下文的事实。
- 永远不要返回判断对错题。
- 不要仅询问日期。

问题规则：
- 每道题必须测试恰好一个离散事实。
- 使用清晰、无歧义的措辞。
- 优先使用"什么"、"谁"、"多少"、"哪个"。
- 避免开放式的"描述"或"解释"提示。

答案规则：
- 每个答案不超过 240 个字符。
- 以答案本身开头，不要有前缀。
- 仅在需要时添加最少的澄清细节。
```

使用字幕的前 15,000 个字符作为上下文。由你自己生成问题（你是 LLM）。

**第四步：** 验证输出是否为有效 JSON 且恰好包含 5 项，每项都有非空的 `question` 和 `answer` 字符串。如果验证失败，重试一次。

**第五步：** 存储测验卡片：

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py add-quiz \
  --video-id "VIDEO_ID" \
  --questions '[{"question":"...","answer":"..."},...]' \
  --collection "Quiz - Episode Title"
```

脚本按 `video_id` 去重——如果该视频的卡片已存在，将跳过创建并报告现有卡片。

**第六步：** 使用相同的自由文本评分流程逐题展示：
1. 展示"第 1/5 题：..."并等待用户回答。绝不包含答案或任何关于揭示答案的提示。
2. 等待用户用自己的话回答
3. 使用评分提示对回答评分（参见"复习到期卡片"部分）
4. **重要：你必须先向用户回复反馈，再做其他任何事情。** 展示评分、正确答案和下次复习时间。不要静默跳到下一题。保持简短和纯文本。示例："不太对。答案：{答案}。下次复习在明天。"
5. **展示反馈后**，调用评分命令，然后在同一条消息中展示下一题：
```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py rate \
  --id CARD_ID --rating easy --user-answer "what the user said"
```
6. 重复。每个回答必须在下一题之前获得可见的反馈。

### 导出/导入 CSV

**导出：**
```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py export \
  --output ~/flashcards.csv
```

生成 3 列 CSV：`question,answer,collection`（无表头行）。

**导入：**
```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py import \
  --file ~/flashcards.csv \
  --collection "Imported"
```

读取包含以下列的 CSV：问题、答案，以及可选的集合（第 3 列）。如果缺少集合列，使用 `--collection` 参数。

### 统计信息

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py stats
```

返回包含以下信息的 JSON：
- `total`：总卡片数
- `learning`：正在轮换中的卡片
- `retired`：已掌握的卡片
- `due_now`：当前需要复习的卡片
- `collections`：按集合名称分类的统计

## 注意事项

- **切勿直接编辑 `cards.json`** —— 始终使用脚本子命令以避免数据损坏
- **字幕获取失败** —— 部分 YouTube 视频没有英文字幕或字幕被禁用；请告知用户并建议其他视频
- **可选依赖** —— `youtube_quiz.py` 需要 `youtube-transcript-api`；如果缺少，请告诉用户运行 `pip install youtube-transcript-api`
- **大量导入** —— 包含数千行的 CSV 导入可以正常工作，但 JSON 输出可能很长；请为用户总结结果
- **视频 ID 提取** —— 支持两种 URL 格式：`youtube.com/watch?v=ID` 和 `youtu.be/ID`

## 验证

直接验证辅助脚本：

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py stats
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py add --question "Capital of France?" --answer "Paris" --collection "General"
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py due
```

如果从仓库检出代码进行测试，运行：

```bash
pytest tests/skills/test_memento_cards.py tests/skills/test_youtube_quiz.py -q
```

代理级别的验证：
- 开始一次复习并确认反馈为纯文本、简短，且始终在下一张卡片之前包含正确答案
- 运行 YouTube 测验流程并确认每个回答在下一题之前获得可见的反馈

---
title: "Memento Flashcards — 间隔重复闪卡系统"
sidebar_label: "Memento Flashcards"
description: "间隔重复闪卡系统"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Memento Flashcards

间隔重复（Spaced-repetition）闪卡系统。可从事实或文本创建卡片，通过自由文本回答与闪卡对话并由 agent 评分，从 YouTube 字幕生成测验，以自适应调度复习到期卡片，以及以 CSV 格式导出/导入卡组。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/productivity/memento-flashcards` 安装 |
| 路径 | `optional-skills/productivity/memento-flashcards` |
| 版本 | `1.0.0` |
| 作者 | Memento AI |
| 许可证 | MIT |
| 平台 | macos, linux |
| 标签 | `Education`, `Flashcards`, `Spaced Repetition`, `Learning`, `Quiz`, `YouTube` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Memento Flashcards — 间隔重复闪卡 Skill

## 概述

Memento 为你提供一个本地、基于文件的闪卡系统，具备间隔重复调度功能。
用户可以通过自由文本回答与闪卡互动，由 agent 在安排下次复习前对回答进行评分。
在以下情况下使用此 skill：

- **记住一个事实** — 将任意陈述转化为问答闪卡
- **间隔重复学习** — 以自适应间隔和 agent 评分的自由文本回答复习到期卡片
- **从 YouTube 视频生成测验** — 获取字幕并生成 5 道测验题
- **管理卡组** — 将卡片整理成集合，导出/导入 CSV

所有卡片数据存储在单个 JSON 文件中。无需外部 API 密钥 — 由你（agent）直接生成闪卡内容和测验题。

Memento Flashcards 的用户响应风格：
- 仅使用纯文本。回复用户时不使用 Markdown 格式。
- 复习和测验反馈保持简短、中立。避免额外的称赞、鼓励或冗长解释。

## 使用时机

在用户希望执行以下操作时使用此 skill：
- 将事实保存为闪卡以供后续复习
- 以间隔重复方式复习到期卡片
- 从 YouTube 视频字幕生成测验
- 导入、导出、查看或删除闪卡数据

不要将此 skill 用于通用问答、编程帮助或非记忆类任务。

## 快速参考

| 用户意图 | 操作 |
|---|---|
| "记住 X" / "将此保存为闪卡" | 生成问答卡片，调用 `memento_cards.py add` |
| 发送事实但未提及闪卡 | 询问"要将此保存为 Memento 闪卡吗？" — 仅在确认后创建 |
| "创建一张闪卡" | 询问问题、答案、集合；调用 `memento_cards.py add` |
| "复习我的卡片" | 调用 `memento_cards.py due`，逐张呈现卡片 |
| "用 [YouTube URL] 测验我" | 调用 `youtube_quiz.py fetch VIDEO_ID`，生成 5 道题，调用 `memento_cards.py add-quiz` |
| "导出我的卡片" | 调用 `memento_cards.py export --output PATH` |
| "从 CSV 导入卡片" | 调用 `memento_cards.py import --file PATH --collection NAME` |
| "显示我的统计" | 调用 `memento_cards.py stats` |
| "删除一张卡片" | 调用 `memento_cards.py delete --id ID` |
| "删除一个集合" | 调用 `memento_cards.py delete-collection --collection NAME` |

## 卡片存储

卡片存储在以下路径的 JSON 文件中：

```
~/.hermes/skills/productivity/memento-flashcards/data/cards.json
```

**切勿直接编辑此文件。** 始终使用 `memento_cards.py` 子命令。该脚本通过原子写入（先写入临时文件，再重命名）来防止数据损坏。

该文件在首次使用时自动创建。

## 操作流程

### 从事实创建卡片

### 激活规则

并非每个事实陈述都应成为闪卡。使用以下三级检查：

1. **明确意图** — 用户提到"memento"、"flashcard"、"记住这个"、"保存这张卡片"、"添加一张卡片"或类似明确请求闪卡的措辞 → **直接创建卡片**，无需确认。
2. **隐含意图** — 用户发送事实陈述但未提及闪卡（例如"光速是 299,792 km/s"）→ **先询问**："要将此保存为 Memento 闪卡吗？"仅在用户确认后创建卡片。
3. **无意图** — 消息是编程任务、问题、指令、普通对话，或明显不是需要记忆的事实 → **完全不激活此 skill**。让其他 skill 或默认行为处理。

当激活被确认（第 1 级直接确认，第 2 级经用户确认后），生成闪卡：

**第 1 步：** 将陈述转化为问答对。内部使用以下格式：

```
Turn the factual statement into a front-back pair.
Return exactly two lines:
Q: <question text>
A: <answer text>

Statement: "{statement}"
```

规则：
- 问题应测试对关键事实的回忆
- 答案应简洁直接

**第 2 步：** 调用脚本存储卡片：

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py add \
  --question "What year did World War 2 end?" \
  --answer "1945" \
  --collection "History"
```

如果用户未指定集合，使用 `"General"` 作为默认值。

脚本输出 JSON 确认已创建的卡片。

### 手动创建卡片

当用户明确要求创建闪卡时，询问：
1. 问题（卡片正面）
2. 答案（卡片背面）
3. 集合名称（可选 — 默认为 `"General"`）

然后如上所示调用 `memento_cards.py add`。

### 复习到期卡片

当用户想要复习时，获取所有到期卡片：

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py due
```

返回 `next_review_at <= now` 的卡片 JSON 数组。如需集合过滤：

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py due --collection "History"
```

**复习流程（自由文本评分）：**

以下是你必须遵循的确切交互模式示例。用户回答后，你评分，告知正确答案，然后对卡片评级。

**交互示例：**

> **Agent：** 柏林墙是哪年倒塌的？
>
> **用户：** 1991
>
> **Agent：** 不太对。柏林墙倒塌于 1989 年。下次复习是明天。
> *（agent 调用：memento_cards.py rate --id ABC --rating hard --user-answer "1991"）*
>
> 下一题：第一个登上月球的人是谁？

**规则：**

1. 只显示问题。等待用户回答。
2. 收到回答后，将其与预期答案对比并评分：
   - **correct（正确）** → 用户答对了关键事实（即使措辞不同）
   - **partial（部分正确）** → 方向正确但缺少核心细节
   - **incorrect（错误）** → 答错或偏题
3. **你必须告知用户正确答案及其表现。** 保持简短、纯文本。使用以下格式：
   - correct：「正确。答案：&#123;answer&#125;。下次复习在 7 天后。」
   - partial：「接近了。答案：&#123;answer&#125;。&#123;缺少的内容&#125;。下次复习在 3 天后。」
   - incorrect：「不太对。答案：&#123;answer&#125;。下次复习是明天。」
4. 然后调用评级命令：correct→easy，partial→good，incorrect→hard。
5. 然后显示下一题。

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py rate \
  --id CARD_ID --rating easy --user-answer "what the user said"
```

**绝不跳过第 3 步。** 用户必须在进入下一题前始终看到正确答案和反馈。

如果没有到期卡片，告知用户："现在没有到期的复习卡片。稍后再来查看！"

**退休覆盖：** 用户随时可以说"退休这张卡片"以将其永久从复习中移除。为此使用 `--rating retire`。

### 间隔重复算法

评级决定下次复习间隔：

| 评级 | 间隔 | ease_streak | 状态变化 |
|---|---|---|---|
| **hard** | +1 天 | 重置为 0 | 保持 learning |
| **good** | +3 天 | 重置为 0 | 保持 learning |
| **easy** | +7 天 | +1 | 若 ease_streak >= 3 → retired |
| **retire** | 永久 | 重置为 0 | → retired |

- **learning**：卡片在活跃轮换中
- **retired**：卡片不再出现在复习中（用户已掌握或手动退休）
- 连续三次"easy"评级自动退休卡片

### YouTube 测验生成

当用户发送 YouTube URL 并想要测验时：

**第 1 步：** 从 URL 中提取视频 ID（例如从 `https://www.youtube.com/watch?v=dQw4w9WgXcQ` 中提取 `dQw4w9WgXcQ`）。

**第 2 步：** 获取字幕：

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/youtube_quiz.py fetch VIDEO_ID
```

返回 `{"title": "...", "transcript": "..."}` 或错误信息。

如果脚本报告 `missing_dependency`，告知用户安装：
```bash
pip install youtube-transcript-api
```

**第 3 步：** 从字幕生成 5 道测验题。使用以下规则：

```
You are creating a 5-question quiz for a podcast episode.
Return ONLY a JSON array with exactly 5 objects.
Each object must contain keys 'question' and 'answer'.

Selection criteria:
- Prioritize important, surprising, or foundational facts.
- Skip filler, obvious details, and facts that require heavy context.
- Never return true/false questions.
- Never ask only for a date.

Question rules:
- Each question must test exactly one discrete fact.
- Use clear, unambiguous wording.
- Prefer What, Who, How many, Which.
- Avoid open-ended Describe or Explain prompts.

Answer rules:
- Each answer must be under 240 characters.
- Lead with the answer itself, not preamble.
- Add only minimal clarifying detail if needed.
```

使用字幕的前 15,000 个字符作为上下文。由你自己（作为 LLM）生成问题。

**第 4 步：** 验证输出是否为有效 JSON，且恰好包含 5 个条目，每个条目具有非空的 `question` 和 `answer` 字符串。如果验证失败，重试一次。

**第 5 步：** 存储测验卡片：

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py add-quiz \
  --video-id "VIDEO_ID" \
  --questions '[{"question":"...","answer":"..."},...]' \
  --collection "Quiz - Episode Title"
```

脚本通过 `video_id` 去重 — 如果该视频的卡片已存在，则跳过创建并报告现有卡片。

**第 6 步：** 使用相同的自由文本评分流程逐题呈现：
1. 显示"第 1/5 题：..."并等待用户回答。切勿包含答案或任何关于揭示答案的提示。
2. 等待用户用自己的话回答
3. 使用评分 prompt（见"复习到期卡片"部分）对回答评分
4. **重要：你必须先回复用户反馈，再做任何其他操作。** 显示评级、正确答案以及卡片下次到期时间。不要静默跳到下一题。保持简短、纯文本。示例："不太对。答案：&#123;answer&#125;。下次复习是明天。"
5. **显示反馈后**，调用评级命令，然后在同一消息中显示下一题：
```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py rate \
  --id CARD_ID --rating easy --user-answer "what the user said"
```
6. 重复。每个回答在进入下一题前必须收到可见反馈。

### 导出/导入 CSV

**导出：**
```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py export \
  --output ~/flashcards.csv
```

生成 3 列 CSV：`question,answer,collection`（无标题行）。

**导入：**
```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py import \
  --file ~/flashcards.csv \
  --collection "Imported"
```

读取包含以下列的 CSV：question、answer，以及可选的 collection（第 3 列）。如果缺少 collection 列，使用 `--collection` 参数值。

### 统计

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py stats
```

返回包含以下字段的 JSON：
- `total`：卡片总数
- `learning`：活跃轮换中的卡片
- `retired`：已掌握的卡片
- `due_now`：当前到期待复习的卡片
- `collections`：按集合名称的细分统计

## 注意事项

- **切勿直接编辑 `cards.json`** — 始终使用脚本子命令以避免数据损坏
- **字幕获取失败** — 部分 YouTube 视频没有英文字幕或字幕已禁用；告知用户并建议换一个视频
- **可选依赖** — `youtube_quiz.py` 需要 `youtube-transcript-api`；如果缺失，告知用户运行 `pip install youtube-transcript-api`
- **大量导入** — 包含数千行的 CSV 导入可正常工作，但 JSON 输出可能较冗长；为用户总结结果
- **视频 ID 提取** — 同时支持 `youtube.com/watch?v=ID` 和 `youtu.be/ID` 两种 URL 格式

## 验证

直接验证辅助脚本：

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py stats
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py add --question "Capital of France?" --answer "Paris" --collection "General"
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py due
```

如果从仓库检出进行测试，运行：

```bash
pytest tests/skills/test_memento_cards.py tests/skills/test_youtube_quiz.py -q
```

Agent 级别验证：
- 开始一次复习，确认反馈为纯文本、简短，且在进入下一张卡片前始终包含正确答案
- 运行 YouTube 测验流程，确认每个回答在进入下一题前收到可见反馈
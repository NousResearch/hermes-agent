---
title: "表情包生成 — 通过选择模板并使用 Pillow 叠加文字来生成真实的表情包图片"
sidebar_label: "表情包生成"
description: "通过选择模板并使用 Pillow 叠加文字来生成真实的表情包图片"
---

{/* 此页面由 website/scripts/generate-skill-docs.py 根据 skill 的 SKILL.md 自动生成。请编辑源 SKILL.md，而非此页面。 */}

# 表情包生成

通过选择模板并使用 Pillow 叠加文字来生成真实的表情包图片。生成实际的 .png 表情包文件。

## 技能元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/creative/meme-generation` 安装 |
| 路径 | `optional-skills/creative/meme-generation` |
| 版本 | `2.0.0` |
| 作者 | adanaleycio |
| 许可证 | MIT |
| 标签 | `creative`、`memes`、`humor`、`images` |
| 相关技能 | [`ascii-art`](/docs/user-guide/skills/bundled/creative/creative-ascii-art)、`generative-widgets` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 加载此技能时使用的完整技能定义。这是技能激活时代理看到的指令。
:::

# 表情包生成

根据主题生成真实的表情包图片。选择模板，编写标题，并使用文字叠加渲染真实的 .png 文件。

## 使用场景

- 用户请求您制作或生成表情包
- 用户想要关于特定主题、情况或挫折的表情包
- 用户说"meme this"或类似的话

## 可用模板

脚本支持按名称或 ID 使用**约 100 个流行的 imgflip 模板**，加上 10 个经过文字定位微调的精选模板。

### 精选模板（自定义文字位置）

| ID | Name | Fields | Best for |
|----|------|--------|----------|
| `this-is-fine` | This is Fine | top, bottom | chaos, denial |
| `drake` | Drake Hotline Bling | reject, approve | rejecting/preferring |
| `distracted-boyfriend` | Distracted Boyfriend | distraction, current, person | temptation, shifting priorities |
| `two-buttons` | Two Buttons | left, right, person | impossible choice |
| `expanding-brain` | Expanding Brain | 4 levels | escalating irony |
| `change-my-mind` | Change My Mind | statement | hot takes |
| `woman-yelling-at-cat` | Woman Yelling at Cat | woman, cat | arguments |
| `one-does-not-simply` | One Does Not Simply | top, bottom | deceptively hard things |
| `grus-plan` | Gru's Plan | step1-3, realization | plans that backfire |
| `batman-slapping-robin` | Batman Slapping Robin | robin, batman | shutting down bad ideas |

### 动态模板（来自 imgflip API）

任何不在精选列表中的模板都可以通过名称或 imgflip ID 使用。这些获取智能默认文字位置（2 字段为顶部/底部，3+ 字段均匀分布）。使用以下方式搜索：
```bash
python "$SKILL_DIR/scripts/generate_meme.py" --search "disaster"
```

## 步骤

### 模式 1：经典模板（默认）

1. 阅读用户的主题并识别核心动态（混乱、困境、偏好、反讽等）
2. 选择最匹配的模板。使用"Best for"列，或使用 `--search` 搜索。
3. 为每个字段写简短标题（每个字段最多 8-12 个词，越短越好）。
4. 找到技能的脚本目录：
   ```
   SKILL_DIR=$(dirname "$(find ~/.hermes/skills -path '*/meme-generation/SKILL.md' 2>/dev/null | head -1)")
   ```
5. 运行生成器：
   ```bash
   python "$SKILL_DIR/scripts/generate_meme.py" <template_id> /tmp/meme.png "caption 1" "caption 2" ...
   ```
6. 使用 `MEDIA:/tmp/meme.png` 返回图片

### 模式 2：自定义 AI 图片（当 image_generate 可用时）

当没有经典模板合适，或用户想要原创内容时使用此模式。

1. 首先编写标题。
2. 使用 `image_generate` 创建匹配表情包概念的场景。请勿在图片提示中包含任何文字 — 文字将由脚本添加。仅描述视觉场景。
3. 从 image_generate 结果 URL 获取生成的图片路径。如需要，下载到本地路径。
4. 使用 `--image` 运行脚本以叠加文字，选择模式：
   - **叠加**（文字直接放在图片上，白色带黑色描边）：
     ```bash
     python "$SKILL_DIR/scripts/generate_meme.py" --image /path/to/scene.png /tmp/meme.png "top text" "bottom text"
     ```
   - **条状**（黑色条在上/下，白色文字 — 更干净，始终可读）：
     ```bash
     python "$SKILL_DIR/scripts/generate_meme.py" --image /path/to/scene.png --bars /tmp/meme.png "top text" "bottom text"
     ```
   当图片复杂/细节丰富、文字难以阅读时使用 `--bars`。
5. **用视觉验证**（如果 `vision_analyze` 可用）：检查结果看起来不错：
   ```
   vision_analyze(image_url="/tmp/meme.png", question="Is the text legible and well-positioned? Does the meme work visually?")
   ```
   如果视觉模型标记问题（文字难以阅读、位置不佳等），尝试其他模式（在叠加和条状之间切换）或重新生成场景。
6. 使用 `MEDIA:/tmp/meme.png` 返回图片

## 示例

**"凌晨 2 点在生产环境调试"：**
```bash
python generate_meme.py this-is-fine /tmp/meme.png "SERVERS ARE ON FIRE" "This is fine"
```

**"在睡觉和再看一集之间选择"：**
```bash
python generate_meme.py drake /tmp/meme.png "Getting 8 hours of sleep" "One more episode at 3 AM"
```

**"周一早上的阶段"：**
```bash
python generate_meme.py expanding-brain /tmp/meme.png "Setting an alarm" "Setting 5 alarms" "Sleeping through all alarms" "Working from bed"
```

## 列出模板

查看所有可用模板：
```bash
python generate_meme.py --list
```

## 陷阱

- 标题要**简短**。文字长的表情包看起来很糟糕。
- 文本参数数量必须匹配模板的字段数。
- 选择适合笑话结构的模板，而非仅匹配主题。
- 不要生成仇恨、辱骂或针对个人的内容。
- 脚本在首次下载后将模板图片缓存在 `scripts/.cache/` 中。

## 验证

输出正确当且仅当：
- 在输出路径创建了 .png 文件
- 模板上的文字清晰可读（白色带黑色描边）
- 笑话有效 — 标题匹配模板的预期结构
- 文件可以通过 MEDIA: 路径传递

---
title: "Meme Generation — 使用 Pillow 选取模板并叠加文字，生成真实的表情包图片"
sidebar_label: "Meme Generation"
description: "使用 Pillow 选取模板并叠加文字，生成真实的表情包图片"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Meme Generation

使用 Pillow 选取模板并叠加文字，生成真实的表情包图片。输出实际的 .png 表情包文件。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/creative/meme-generation` 安装 |
| 路径 | `optional-skills/creative/meme-generation` |
| 版本 | `2.0.0` |
| 作者 | adanaleycio |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `creative`, `memes`, `humor`, `images` |
| 相关 skill | [`ascii-art`](/user-guide/skills/bundled/creative/creative-ascii-art), `generative-widgets` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Meme Generation

根据主题生成实际的表情包图片。选取模板、编写说明文字，并渲染带有文字叠加的真实 .png 文件。

## 使用时机

- 用户要求制作或生成表情包
- 用户想要关于某个话题、情境或吐槽的表情包
- 用户说"把这个做成表情包"或类似表达

## 可用模板

该脚本支持按名称或 ID 使用 **imgflip 上约 100 个热门模板**，另外还有 10 个经过精心调整文字位置的精选模板。

### 精选模板（自定义文字位置）

| ID | 名称 | 字段 | 最适合 |
|----|------|--------|----------|
| `this-is-fine` | This is Fine | top, bottom | 混乱、否认 |
| `drake` | Drake Hotline Bling | reject, approve | 拒绝/偏好 |
| `distracted-boyfriend` | Distracted Boyfriend | distraction, current, person | 诱惑、转移注意力 |
| `two-buttons` | Two Buttons | left, right, person | 两难抉择 |
| `expanding-brain` | Expanding Brain | 4 个层级 | 层层递进的讽刺 |
| `change-my-mind` | Change My Mind | statement | 热门观点 |
| `woman-yelling-at-cat` | Woman Yelling at Cat | woman, cat | 争论 |
| `one-does-not-simply` | One Does Not Simply | top, bottom | 出乎意料的难事 |
| `grus-plan` | Gru's Plan | step1-3, realization | 计划反噬 |
| `batman-slapping-robin` | Batman Slapping Robin | robin, batman | 驳斥烂主意 |

### 动态模板（来自 imgflip API）

不在精选列表中的任何模板均可通过名称或 imgflip ID 使用。这些模板会自动应用智能默认文字位置（2 个字段时为上/下，3 个及以上时均匀分布）。搜索方式：
```bash
python "$SKILL_DIR/scripts/generate_meme.py" --search "disaster"
```

## 操作流程

### 模式 1：经典模板（默认）

1. 读取用户的主题，识别核心动态（混乱、两难、偏好、讽刺等）。
2. 选取最匹配的模板。参考"最适合"列，或使用 `--search` 搜索。
3. 为每个字段编写简短说明文字（每个字段最多 8-12 个词，越短越好）。
4. 找到 skill 的脚本目录：
   ```
   SKILL_DIR=$(dirname "$(find ~/.hermes/skills -path '*/meme-generation/SKILL.md' 2>/dev/null | head -1)")
   ```
5. 运行生成器：
   ```bash
   python "$SKILL_DIR/scripts/generate_meme.py" <template_id> /tmp/meme.png "caption 1" "caption 2" ...
   ```
6. 使用 `MEDIA:/tmp/meme.png` 返回图片。

### 模式 2：自定义 AI 图片（当 image_generate 可用时）

当没有合适的经典模板，或用户想要原创内容时使用此模式。

1. 先编写说明文字。
2. 使用 `image_generate` 创建符合表情包概念的场景。图片 prompt（提示词）中**不要包含任何文字** — 文字将由脚本添加。仅描述视觉场景。
3. 从 image_generate 结果 URL 中找到生成图片的路径。如有需要，将其下载到本地路径。
4. 使用 `--image` 运行脚本叠加文字，选择一种模式：
   - **Overlay**（文字直接叠加在图片上，白色带黑色描边）：
     ```bash
     python "$SKILL_DIR/scripts/generate_meme.py" --image /path/to/scene.png /tmp/meme.png "top text" "bottom text"
     ```
   - **Bars**（图片上下方添加黑色条带显示白色文字 — 更整洁，始终可读）：
     ```bash
     python "$SKILL_DIR/scripts/generate_meme.py" --image /path/to/scene.png --bars /tmp/meme.png "top text" "bottom text"
     ```
   当图片内容复杂/细节丰富、文字叠加后难以辨认时，使用 `--bars`。
5. **使用视觉验证**（如果 `vision_analyze` 可用）：检查结果是否美观：
   ```
   vision_analyze(image_url="/tmp/meme.png", question="Is the text legible and well-positioned? Does the meme work visually?")
   ```
   如果视觉模型发现问题（文字难以辨认、位置不佳等），尝试切换另一种模式（在 overlay 和 bars 之间切换）或重新生成场景。
6. 使用 `MEDIA:/tmp/meme.png` 返回图片。

## 示例

**"凌晨 2 点调试生产环境"：**
```bash
python generate_meme.py this-is-fine /tmp/meme.png "SERVERS ARE ON FIRE" "This is fine"
```

**"在睡觉和再看一集之间做选择"：**
```bash
python generate_meme.py drake /tmp/meme.png "Getting 8 hours of sleep" "One more episode at 3 AM"
```

**"周一早晨的各个阶段"：**
```bash
python generate_meme.py expanding-brain /tmp/meme.png "Setting an alarm" "Setting 5 alarms" "Sleeping through all alarms" "Working from bed"
```

## 列出模板

查看所有可用模板：
```bash
python generate_meme.py --list
```

## 注意事项

- 说明文字要**简短**。文字过长的表情包效果很差。
- 文字参数数量须与模板的字段数量匹配。
- 根据笑点结构选择模板，而不仅仅是根据话题。
- 不得生成仇恨、辱骂或针对特定个人的内容。
- 脚本会在首次下载后将模板图片缓存至 `scripts/.cache/`。

## 验证

以下情况说明输出正确：
- 在输出路径创建了 .png 文件
- 文字在模板上清晰可读（白色带黑色描边）
- 笑点成立 — 说明文字与模板的预期结构相符
- 文件可通过 MEDIA: 路径传递
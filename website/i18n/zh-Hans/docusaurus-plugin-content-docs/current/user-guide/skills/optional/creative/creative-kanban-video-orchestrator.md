---
title: "Kanban Video Orchestrator — 规划、搭建并监控由 Hermes Kanban 支撑的多智能体视频制作流水线"
sidebar_label: "Kanban Video Orchestrator"
description: "规划、搭建并监控由 Hermes Kanban 支撑的多智能体视频制作流水线"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Kanban Video Orchestrator

规划、搭建并监控由 Hermes Kanban 支撑的多智能体视频制作流水线。当用户想要制作**任何**类型的视频时使用本技能——叙事短片、产品/营销视频、MV、解说视频、ASCII/终端艺术、抽象/生成循环、漫画、3D、实时/装置艺术——且工作需要分解为专业角色（编剧、设计师、动画师、渲染师、配音、剪辑等）并通过 kanban 看板协调。执行自适应探索以明确需求范围，为所请求的风格设计合适的团队，生成用于创建 Hermes profiles 和初始 kanban 任务的安装脚本，然后协助监控执行过程并在任务卡住或失败时介入。将场景路由到适合每个节拍的 Hermes 渲染/音频/设计技能（`ascii-video`、`manim-video`、`p5js`、`comfyui`、`touchdesigner-mcp`、`blender-mcp`、`pixel-art`、`baoyu-comic`、`claude-design`、`excalidraw`、`songsee`、`heartmula`……）以及用于 TTS、图像生成和图像转视频的外部 API。

## 技能元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/creative/kanban-video-orchestrator` 安装 |
| 路径 | `optional-skills/creative/kanban-video-orchestrator` |
| 版本 | `1.0.0` |
| 作者 | ['SHL0MS', 'alt-glitch'] |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `video`, `kanban`, `multi-agent`, `orchestration`, `production-pipeline` |
| 相关技能 | [`kanban-orchestrator`](/user-guide/skills/bundled/devops/devops-kanban-orchestrator)、[`kanban-worker`](/user-guide/skills/bundled/devops/devops-kanban-worker)、[`ascii-video`](/user-guide/skills/bundled/creative/creative-ascii-video)、[`manim-video`](/user-guide/skills/bundled/creative/creative-manim-video)、[`p5js`](/user-guide/skills/bundled/creative/creative-p5js)、[`comfyui`](/user-guide/skills/bundled/creative/creative-comfyui)、[`touchdesigner-mcp`](/user-guide/skills/bundled/creative/creative-touchdesigner-mcp)、[`blender-mcp`](/user-guide/skills/optional/creative/creative-blender-mcp)、[`pixel-art`](/user-guide/skills/bundled/creative/creative-pixel-art)、[`ascii-art`](/user-guide/skills/bundled/creative/creative-ascii-art)、[`songwriting-and-ai-music`](/user-guide/skills/bundled/creative/creative-songwriting-and-ai-music)、[`heartmula`](/user-guide/skills/bundled/media/media-heartmula)、[`songsee`](/user-guide/skills/bundled/media/media-songsee)、[`spotify`](/user-guide/skills/bundled/media/media-spotify)、[`youtube-content`](/user-guide/skills/bundled/media/media-youtube-content)、[`claude-design`](/user-guide/skills/bundled/creative/creative-claude-design)、[`excalidraw`](/user-guide/skills/bundled/creative/creative-excalidraw)、[`architecture-diagram`](/user-guide/skills/bundled/creative/creative-architecture-diagram)、[`concept-diagrams`](/user-guide/skills/optional/creative/creative-concept-diagrams)、[`baoyu-comic`](/user-guide/skills/bundled/creative/creative-baoyu-comic)、[`baoyu-infographic`](/user-guide/skills/bundled/creative/creative-baoyu-infographic)、[`humanizer`](/user-guide/skills/bundled/creative/creative-humanizer)、[`gif-search`](/user-guide/skills/bundled/media/media-gif-search)、[`meme-generation`](/user-guide/skills/optional/creative/creative-meme-generation) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发本技能时加载的完整技能定义。这是技能激活时智能体所看到的指令内容。
:::

# Kanban Video Orchestrator

将任何视频请求——从 15 秒产品预告到 5 分钟叙事短片，再到 MV 或 ASCII 循环——封装进 Hermes Kanban 流水线，将工作分解给专业智能体 profiles。

本技能**不**自行渲染任何内容。它是一个元流水线，负责：

1. **探索**——通过有针对性的发现问题明确需求范围
2. **设计**——根据风格设计合适的团队（哪些角色、每个角色使用哪些工具）
3. **生成**——生成安装脚本，创建 Hermes profiles、项目工作区和初始 kanban 任务
4. **交接**——移交给 director profile，由其通过 kanban 进行分解
5. **监控**——跟踪执行过程，在任务卡住或失败时协助介入

实际渲染在 kanban 运行后在其内部完成，使用适合各场景的现有技能和工具——`ascii-video`、`manim-video`、`p5js`、`comfyui`、`touchdesigner-mcp`、`blender-mcp`、`songwriting-and-ai-music`、`heartmula`、外部 API，或使用 PIL + ffmpeg 的纯 Python。

## 不适用本技能的情况

- 视频是一个无需专业分工的连续程序化项目。直接编写代码即可。
- 用户只需快速一次性转换（例如"把这个 mp4 转成 GIF"）——直接使用 ffmpeg。
- 输出是静态图片、GIF 或纯音频产物——使用对应的专项技能（`ascii-art`、`gifs`、`meme-generation`、`songwriting-and-ai-music`）。
- 工作完全适合某个现有技能（例如纯 ASCII 视频——直接使用 `ascii-video`）。

## 工作流程

```
DISCOVER  →  BRIEF  →  TEAM DESIGN  →  SETUP  →  EXECUTE  →  MONITOR
```

### 第一步 — 探索（提出正确的问题）

探索过程是**自适应的**：只问真正需要的问题。始终从三个问题开始，以识别大致轮廓：

- **视频是什么？**（一句话简介）
- **时长多少？**（5-30 秒预告 / 30-90 秒短片 / 90 秒-3 分钟解说 / 3-10 分钟影片 / 更长）
- **宽高比和目标平台？**（1:1 / 9:16 / 16:9；X、IG、YouTube、内部使用等）

根据回答，对风格类别进行分类。风格决定后续需要提问的问题。**不要一次性问所有问题。** 每次问 2-4 个，倾听回答，然后继续。当用户的回答隐含某个答案时，做出合理假设。

完整的收集模式和各风格问题库，参见
**[references/intake.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/kanban-video-orchestrator/references/intake.md)**。

### 第二步 — 简报

掌握足够信息后，使用 `assets/brief.md.tmpl` 中的模板生成结构化的 `brief.md`。阶段如下：

1. **概念** — 一句话 pitch + 情感北极星
2. **范围** — 时长、宽高比、平台、截止日期
3. **风格** — 视觉参考、品牌约束、基调
4. **场景** — 逐拍分解（时长、内容、目标工具）
5. **音频** — 旁白 / 音乐 / 音效 / 静音（如需可按场景细分）
6. **交付物** — 文件格式、分辨率、可选备选版本（竖版剪辑、GIF 等）

在设计团队之前，将简报展示给用户确认。**简报即合同**——所有下游任务均以其为参考。

### 第三步 — 团队设计

从角色库中挑选适合本视频的角色原型。**组合，而非复制。** 大多数视频需要 4-7 个 profiles。director 始终存在；其余角色根据简报的实际需求选取。

角色库和各风格团队组合，参见
**[references/role-archetypes.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/kanban-video-orchestrator/references/role-archetypes.md)**。

角色与 Hermes 技能及工具集的映射关系，参见
**[references/tool-matrix.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/kanban-video-orchestrator/references/tool-matrix.md)**。

### 第四步 — 安装

生成安装脚本（`setup.sh`）并运行。脚本将：

1. 创建项目工作区（`~/projects/video-pipeline/<slug>/`）
2. 将提供的资产复制到 `taste/`、`audio/`、`assets/`
3. 通过 `hermes profile create --clone` 创建每个 Hermes profile
4. 编写各 profile 的 `SOUL.md`（个性 + 角色定义）
5. 配置 profile YAML（工具集、always_load 技能、cwd）
6. 编写 `brief.md`、`TEAM.md` 和 `taste/` 内容
7. 触发分配给 director 的初始 `hermes kanban create` 任务

使用 `scripts/bootstrap_pipeline.py` 从简报 + 团队设计 JSON 生成 setup.sh。安装脚本结构、profile 配置模式和关键的"共享工作区"规则，参见 **[references/kanban-setup.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/kanban-video-orchestrator/references/kanban-setup.md)**。

### 第五步 — 执行

运行 `setup.sh`。然后向用户提供监控命令：

```bash
hermes kanban watch --tenant <project-tenant>     # 实时事件
hermes kanban list  --tenant <project-tenant>     # 看板快照
hermes dashboard                                   # 可视化看板 UI
```

director profile 从此接管，通过 kanban 工具集将工作分解并路由给专业 profiles。

### 第六步 — 监控与介入

保持参与——kanban 自主运行，但卡住的任务或不良输出需要人工（或 AI）判断。

监控模式：定期轮询 `kanban list`，用 `kanban show <id>` 检查任何超出预期时长的 RUNNING 任务，并检查心跳。当某个 worker 的输出未通过审核时，标准介入方式为：

1. 在 worker 的任务上附上具体反馈评论（`kanban_comment`）
2. 以原任务为父任务创建重新运行任务
3. 调整简报范围，让 director 重新分解

诊断模式、介入方案和"任务卡住"处理手册，参见 **[references/monitoring.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/kanban-video-orchestrator/references/monitoring.md)**。

## 参考：实际案例

六个涵盖截然不同视频风格的具体流水线——叙事短片、产品/营销视频、MV、数学/算法解说、ASCII 视频、实时装置——展示相同工作流程如何产生截然不同的团队和任务图。参见 **[references/examples.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/kanban-video-orchestrator/references/examples.md)**。

## 关键规则

1. **行动前先探索。** 在至少提出三个基线问题之前，绝不开始生成简报或团队设计。糟糕的简报会在整个流水线中产生连锁反应。

2. **团队要匹配视频。** 不要对每个项目都复用同一套 4-profile 配置。没有节拍分析 profile 的 MV 会出错。没有编剧 profile 的叙事短片会产生不连贯的场景。参见 `references/role-archetypes.md`。

3. **每个项目一个工作区。** 同一视频的所有 profiles 共享同一个 `dir:` 工作区。任务通过共享文件系统和结构化交接传递产物。**每个** `kanban_create` 调用都传入 `workspace_kind="dir"` + `workspace_path="<绝对项目路径>"`。

4. **每个项目使用独立 tenant。** 使用项目专属 tenant（`--tenant <project-slug>`）。保持 dashboard 范围清晰，防止与其他正在进行的 kanban 交叉污染。

5. **尊重现有技能。** 当某个场景适合现有技能时，相关渲染器应通过任务上的 `--skill <name>` 或 profile 中的 `always_load` 加载该技能。不要重新推导技能已提供的内容。

6. **director 绝不执行。** 即使拥有完整的 `kanban + terminal + file` 工具集，director 的 `SOUL.md` 规则也禁止其自行执行工作。它只负责分解和路由——每个具体任务都变成对专业 profile 的 `hermes kanban create` 调用。`kanban-orchestrator` 技能对此有进一步说明。

7. **不要过度分解。** 一个 30 秒的产品视频**不需要** 20 个任务。目标是最小任务图，同时仍能良好并行化并暴露正确的人工审核节点。

8. **触发前验证 API 密钥。** 外部 API（TTS、图像生成、图像转视频）需要在 `~/.hermes/.env` 或用户密钥存储中配置密钥。遇到缺少密钥错误的 worker 会浪费一个任务槽。安装脚本的 `check_key` 辅助函数在缺少必要密钥时会干净地中止。

## 文件结构

```
SKILL.md                            ← 本文件（工作流程 + 规则）
references/
  intake.md                         ← 各风格的探索问题库
  role-archetypes.md                ← 角色库（编剧、设计师、动画师……）
  tool-matrix.md                    ← 各角色的技能 + 工具集映射
  kanban-setup.md                   ← 安装脚本结构与 profile 配置
  monitoring.md                     ← 监控 + 介入模式
  examples.md                       ← 六个实际流水线案例
assets/
  brief.md.tmpl                     ← 简报骨架
  setup.sh.tmpl                     ← 安装脚本骨架
  soul.md.tmpl                      ← profile 个性骨架
scripts/
  bootstrap_pipeline.py             ← 从简报 + 团队 JSON 生成 setup.sh
  monitor.py                        ← 轮询 + 介入辅助工具
```
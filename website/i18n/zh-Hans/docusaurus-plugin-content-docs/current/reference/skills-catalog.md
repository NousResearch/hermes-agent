---
sidebar_position: 5
title: "内置技能目录"
description: "随 Hermes Agent 附带的内置技能目录"
---

# 内置技能目录

Hermes 附带一个大型内置技能库，安装时会复制到 `~/.hermes/skills/`。下方每个技能均链接至专属页面，包含完整定义、配置和用法说明。

Hermes 在执行 `hermes update` 时也会同步内置技能，但同步清单会尊重本地删除和用户编辑。如果此处列出的某个技能在你的 `~/.hermes/skills/` 目录树中缺失，它仍随 Hermes 一同发布；可通过 `hermes skills reset <name> --restore` 恢复。

如果某个技能未出现在此列表中但存在于仓库中，目录由 `website/scripts/generate-skill-docs.py` 重新生成。

## apple

| 技能 | 描述 | 路径 |
|-------|-------------|------|
| [`apple-notes`](/user-guide/skills/bundled/apple/apple-apple-notes) | 通过 memo CLI 管理 Apple Notes：创建、搜索、编辑。 | `apple/apple-notes` |
| [`apple-reminders`](/user-guide/skills/bundled/apple/apple-apple-reminders) | 通过 remindctl 操作 Apple Reminders：添加、列出、完成。 | `apple/apple-reminders` |
| [`findmy`](/user-guide/skills/bundled/apple/apple-findmy) | 在 macOS 上通过 FindMy.app 追踪 Apple 设备/AirTag。 | `apple/findmy` |
| [`imessage`](/user-guide/skills/bundled/apple/apple-imessage) | 在 macOS 上通过 imsg CLI 发送和接收 iMessage/SMS。 | `apple/imessage` |
| [`macos-computer-use`](/user-guide/skills/bundled/apple/apple-macos-computer-use) | 在后台驱动 macOS 桌面——截图、鼠标、键盘、滚动、拖拽——不抢占用户的光标、键盘焦点或 Space。适用于任何支持工具调用的模型。每当需要 `computer_use` 工具时加载此技能。 | `apple/macos-computer-use` |

## autonomous-ai-agents

| 技能 | 描述 | 路径 |
|-------|-------------|------|
| [`claude-code`](/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code) | 将编码任务委托给 Claude Code CLI（功能开发、PR）。 | `autonomous-ai-agents/claude-code` |
| [`codex`](/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex) | 将编码任务委托给 OpenAI Codex CLI（功能开发、PR）。 | `autonomous-ai-agents/codex` |
| [`hermes-agent`](/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) | 配置、扩展或贡献 Hermes Agent。 | `autonomous-ai-agents/hermes-agent` |
| [`opencode`](/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-opencode) | 将编码任务委托给 OpenCode CLI（功能开发、PR 审查）。 | `autonomous-ai-agents/opencode` |

## creative

| 技能 | 描述 | 路径 |
|-------|-------------|------|
| [`architecture-diagram`](/user-guide/skills/bundled/creative/creative-architecture-diagram) | 以 HTML 形式生成深色主题的 SVG 架构/云/基础设施图。 | `creative/architecture-diagram` |
| [`ascii-art`](/user-guide/skills/bundled/creative/creative-ascii-art) | ASCII 艺术：pyfiglet、cowsay、boxes、图像转 ASCII。 | `creative/ascii-art` |
| [`ascii-video`](/user-guide/skills/bundled/creative/creative-ascii-video) | ASCII 视频：将视频/音频转换为彩色 ASCII MP4/GIF。 | `creative/ascii-video` |
| [`baoyu-infographic`](/user-guide/skills/bundled/creative/creative-baoyu-infographic) | 信息图（可视化）：21 种布局 × 21 种风格。 | `creative/baoyu-infographic` |
| [`claude-design`](/user-guide/skills/bundled/creative/creative-claude-design) | 设计一次性 HTML 制品（落地页、幻灯片、原型）。 | `creative/claude-design` |
| [`comfyui`](/user-guide/skills/bundled/creative/creative-comfyui) | 使用 ComfyUI 生成图像、视频和音频——安装、启动、管理节点/模型、运行带参数注入的工作流。使用官方 comfy-cli 管理生命周期，通过 REST/WebSocket API 直接执行。 | `creative/comfyui` |
| [`design-md`](/user-guide/skills/bundled/creative/creative-design-md) | 编写/验证/导出 Google 的 DESIGN.md token 规范文件。 | `creative/design-md` |
| [`excalidraw`](/user-guide/skills/bundled/creative/creative-excalidraw) | 手绘风格的 Excalidraw JSON 图表（架构、流程、时序）。 | `creative/excalidraw` |
| [`humanizer`](/user-guide/skills/bundled/creative/creative-humanizer) | 人性化文本：去除 AI 腔，加入真实语气。 | `creative/humanizer` |
| [`manim-video`](/user-guide/skills/bundled/creative/creative-manim-video) | Manim CE 动画：3Blue1Brown 风格数学/算法视频。 | `creative/manim-video` |
| [`p5js`](/user-guide/skills/bundled/creative/creative-p5js) | p5.js 草图：生成艺术、着色器、交互、3D。 | `creative/p5js` |
| [`popular-web-designs`](/user-guide/skills/bundled/creative/creative-popular-web-designs) | 54 种真实设计系统（Stripe、Linear、Vercel）的 HTML/CSS 实现。 | `creative/popular-web-designs` |
| [`pretext`](/user-guide/skills/bundled/creative/creative-pretext) | 使用 @chenglou/pretext 构建创意浏览器 demo——无 DOM 的文本布局，支持 ASCII 艺术、绕障碍物的排版流、文字即几何游戏、动态排版和文字驱动的生成艺术。生成单文件 HTML。 | `creative/pretext` |
| [`sketch`](/user-guide/skills/bundled/creative/creative-sketch) | 一次性 HTML 原型：生成 2-3 个设计变体供对比。 | `creative/sketch` |
| [`songwriting-and-ai-music`](/user-guide/skills/bundled/creative/creative-songwriting-and-ai-music) | 歌曲创作技巧与 Suno AI 音乐 prompt（提示词）。 | `creative/songwriting-and-ai-music` |
| [`touchdesigner-mcp`](/user-guide/skills/bundled/creative/creative-touchdesigner-mcp) | 通过 twozero MCP 控制运行中的 TouchDesigner 实例——创建算子、设置参数、连接节点、执行 Python、构建实时视觉效果。36 个原生工具。 | `creative/touchdesigner-mcp` |

## devops

| 技能 | 描述 | 路径 |
|-------|-------------|------|


## email

| 技能 | 描述 | 路径 |
|-------|-------------|------|
| [`himalaya`](/user-guide/skills/bundled/email/email-himalaya) | Himalaya CLI：在终端中收发 IMAP/SMTP 邮件。 | `email/himalaya` |

## gaming

| 技能 | 描述 | 路径 |
|-------|-------------|------|

## github

| 技能 | 描述 | 路径 |
|-------|-------------|------|
| [`codebase-inspection`](/user-guide/skills/bundled/github/github-codebase-inspection) | 使用 pygount 检查代码库：代码行数、语言、占比。 | `github/codebase-inspection` |
| [`github-auth`](/user-guide/skills/bundled/github/github-github-auth) | GitHub 认证配置：HTTPS token、SSH 密钥、gh CLI 登录。 | `github/github-auth` |
| [`github-code-review`](/user-guide/skills/bundled/github/github-github-code-review) | 审查 PR：通过 gh 或 REST API 查看 diff、添加行内评论。 | `github/github-code-review` |
| [`github-issues`](/user-guide/skills/bundled/github/github-github-issues) | 通过 gh 或 REST API 创建、分类、标记、分配 GitHub issue。 | `github/github-issues` |
| [`github-pr-workflow`](/user-guide/skills/bundled/github/github-github-pr-workflow) | GitHub PR 生命周期：分支、提交、开启、CI、合并。 | `github/github-pr-workflow` |
| [`github-repo-management`](/user-guide/skills/bundled/github/github-github-repo-management) | 克隆/创建/fork 仓库；管理远程、发布版本。 | `github/github-repo-management` |

## mcp

| 技能 | 描述 | 路径 |
|-------|-------------|------|

## media

| 技能 | 描述 | 路径 |
|-------|-------------|------|
| [`gif-search`](/user-guide/skills/bundled/media/media-gif-search) | 通过 curl + jq 从 Tenor 搜索/下载 GIF。 | `media/gif-search` |
| [`heartmula`](/user-guide/skills/bundled/media/media-heartmula) | HeartMuLa：根据歌词 + 标签生成类 Suno 风格的歌曲。 | `media/heartmula` |
| [`songsee`](/user-guide/skills/bundled/media/media-songsee) | 通过 CLI 生成音频频谱图/特征（mel、chroma、MFCC）。 | `media/songsee` |
| [`youtube-content`](/user-guide/skills/bundled/media/media-youtube-content) | 将 YouTube 字幕转换为摘要、推文串、博客文章。 | `media/youtube-content` |

## mlops

| 技能 | 描述 | 路径 |
|-------|-------------|------|
| [`audiocraft-audio-generation`](/user-guide/skills/bundled/mlops/mlops-models-audiocraft) | AudioCraft：MusicGen 文本转音乐、AudioGen 文本转音效。 | `mlops/models/audiocraft` |
| [`huggingface-hub`](/user-guide/skills/bundled/mlops/mlops-huggingface-hub) | HuggingFace hf CLI：搜索/下载/上传模型、数据集。 | `mlops/huggingface-hub` |
| [`llama-cpp`](/user-guide/skills/bundled/mlops/mlops-inference-llama-cpp) | llama.cpp 本地 GGUF 推理 + HF Hub 模型发现。 | `mlops/inference/llama-cpp` |
| [`evaluating-llms-harness`](/user-guide/skills/bundled/mlops/mlops-evaluation-lm-evaluation-harness) | lm-eval-harness：对 LLM 进行基准测试（MMLU、GSM8K 等）。 | `mlops/evaluation/lm-evaluation-harness` |
| [`segment-anything-model`](/user-guide/skills/bundled/mlops/mlops-models-segment-anything) | SAM：通过点、框、掩码进行零样本图像分割。 | `mlops/models/segment-anything` |
| [`serving-llms-vllm`](/user-guide/skills/bundled/mlops/mlops-inference-vllm) | vLLM：高吞吐量 LLM 服务、OpenAI API 兼容、量化支持。 | `mlops/inference/vllm` |
| [`weights-and-biases`](/user-guide/skills/bundled/mlops/mlops-evaluation-weights-and-biases) | W&B：记录 ML 实验、超参数搜索、模型注册表、仪表盘。 | `mlops/evaluation/weights-and-biases` |

## note-taking

| 技能 | 描述 | 路径 |
|-------|-------------|------|
| [`obsidian`](/user-guide/skills/bundled/note-taking/note-taking-obsidian) | 在 Obsidian 知识库中读取、搜索、创建和编辑笔记。 | `note-taking/obsidian` |

## productivity

| 技能 | 描述 | 路径 |
|-------|-------------|------|
| [`airtable`](/user-guide/skills/bundled/productivity/productivity-airtable) | 通过 curl 调用 Airtable REST API：记录增删改查、过滤、upsert。 | `productivity/airtable` |
| [`google-workspace`](/user-guide/skills/bundled/productivity/productivity-google-workspace) | 通过 gws CLI 或 Python 操作 Gmail、Calendar、Drive、Docs、Sheets。 | `productivity/google-workspace` |
| [`maps`](/user-guide/skills/bundled/productivity/productivity-maps) | 通过 OpenStreetMap/OSRM 进行地理编码、POI 查询、路线规划、时区查询。 | `productivity/maps` |
| [`nano-pdf`](/user-guide/skills/bundled/productivity/productivity-nano-pdf) | 通过 nano-pdf CLI 编辑 PDF 文本/错别字/标题（自然语言 prompt）。 | `productivity/nano-pdf` |
| [`notion`](/user-guide/skills/bundled/productivity/productivity-notion) | Notion API + ntn CLI：页面、数据库、Markdown、Workers。 | `productivity/notion` |
| [`ocr-and-documents`](/user-guide/skills/bundled/productivity/productivity-ocr-and-documents) | 从 PDF/扫描件中提取文本（pymupdf、marker-pdf）。 | `productivity/ocr-and-documents` |
| [`powerpoint`](/user-guide/skills/bundled/productivity/productivity-powerpoint) | 创建、读取、编辑 .pptx 演示文稿、幻灯片、备注、模板。 | `productivity/powerpoint` |
| [`teams-meeting-pipeline`](/user-guide/skills/bundled/productivity/productivity-teams-meeting-pipeline) | 通过 Hermes CLI 操作 Teams 会议摘要流水线——汇总会议、检查流水线状态、重放任务、管理 Microsoft Graph 订阅。 | `productivity/teams-meeting-pipeline` |

## research

| 技能 | 描述 | 路径 |
|-------|-------------|------|
| [`arxiv`](/user-guide/skills/bundled/research/research-arxiv) | 按关键词、作者、分类或 ID 搜索 arXiv 论文。 | `research/arxiv` |
| [`blogwatcher`](/user-guide/skills/bundled/research/research-blogwatcher) | 通过 blogwatcher-cli 工具监控博客和 RSS/Atom 订阅源。 | `research/blogwatcher` |
| [`llm-wiki`](/user-guide/skills/bundled/research/research-llm-wiki) | Karpathy 的 LLM Wiki：构建/查询互联 Markdown 知识库。 | `research/llm-wiki` |
| [`polymarket`](/user-guide/skills/bundled/research/research-polymarket) | 查询 Polymarket：市场、价格、订单簿、历史数据。 | `research/polymarket` |
| [`research-paper-writing`](/user-guide/skills/bundled/research/research-research-paper-writing) | 为 NeurIPS/ICML/ICLR 撰写 ML 论文：从设计到投稿。 | `research/research-paper-writing` |

## smart-home

| 技能 | 描述 | 路径 |
|-------|-------------|------|
| [`openhue`](/user-guide/skills/bundled/smart-home/smart-home-openhue) | 通过 OpenHue CLI 控制 Philips Hue 灯光、场景、房间。 | `smart-home/openhue` |

## social-media

| 技能 | 描述 | 路径 |
|-------|-------------|------|
| [`xurl`](/user-guide/skills/bundled/social-media/social-media-xurl) | 通过 xurl CLI 操作 X/Twitter：发帖、搜索、私信、媒体、v2 API。 | `social-media/xurl` |

## software-development

| 技能 | 描述 | 路径 |
|-------|-------------|------|
| [`dogfood`](/user-guide/skills/bundled/software-development/software-development-dogfood) | Web 应用探索性 QA：发现 bug、收集证据、生成报告。 | `software-development/dogfood` |
| [`hermes-agent-skill-authoring`](/user-guide/skills/bundled/software-development/software-development-hermes-agent-skill-authoring) | 编写仓库内 SKILL.md：frontmatter、验证器、结构规范。 | `software-development/hermes-agent-skill-authoring` |
| [`node-inspect-debugger`](/user-guide/skills/bundled/software-development/software-development-node-inspect-debugger) | 通过 --inspect + Chrome DevTools Protocol CLI 调试 Node.js。 | `software-development/node-inspect-debugger` |
| [`plan`](/user-guide/skills/bundled/software-development/software-development-plan) | 计划模式：将 Markdown 计划写入 `.hermes/plans/`，不执行。 | `software-development/plan` |
| [`python-debugpy`](/user-guide/skills/bundled/software-development/software-development-python-debugpy) | 调试 Python：pdb REPL + debugpy 远程调试（DAP）。 | `software-development/python-debugpy` |
| [`requesting-code-review`](/user-guide/skills/bundled/software-development/software-development-requesting-code-review) | 提交前审查：安全扫描、质量门控、自动修复。 | `software-development/requesting-code-review` |
| [`spike`](/user-guide/skills/bundled/software-development/software-development-spike) | 一次性实验，在正式构建前验证想法。 | `software-development/spike` |
| [`systematic-debugging`](/user-guide/skills/bundled/software-development/software-development-systematic-debugging) | 四阶段根因调试：先理解 bug，再修复。 | `software-development/systematic-debugging` |
| [`test-driven-development`](/user-guide/skills/bundled/software-development/software-development-test-driven-development) | TDD：强制执行红-绿-重构流程，先写测试再写代码。 | `software-development/test-driven-development` |

## yuanbao

| 技能 | 描述 | 路径 |
|-------|-------------|------|
| [`yuanbao`](/user-guide/skills/bundled/yuanbao/yuanbao-yuanbao) | 元宝（Yuanbao）群组：@提及用户、查询信息/成员。 | `yuanbao` |
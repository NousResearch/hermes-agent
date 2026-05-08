---
sidebar_position: 9
title: "可选技能目录"
description: "与 hermes-agent 一起提供的官方可选技能 — 通过 hermes skills install official/<category>/<skill> 安装"
---

# 可选技能目录

可选技能与 hermes-agent 一起发运在 `optional-skills/` 下，但**默认不激活**。显式安装它们：

```bash
hermes skills install official/<category>/<skill>
```

例如：

```bash
hermes skills install official/blockchain/solana
hermes skills install official/mlops/flash-attention
```

每个下面的技能链接到包含其完整定义、设置和用法的专属页面。

卸载：

```bash
hermes skills uninstall <skill-name>
```

## autonomous-ai-agents

| 技能 | 描述 |
|-------|-------------|
| [**blackbox**](/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-blackbox) | 将编码任务委托给 Blackbox AI CLI 代理。具有内置评判的多模型代理，通过多个 LLM 运行任务并选择最佳结果。需要 blackbox CLI 和 Blackbox AI API 密钥。 |
| [**honcho**](/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-honcho) | 使用 Honcho 记忆配置和运行 Hermes — 跨会话用户建模、多 profile 对等隔离、观察配置、辩证推理、会话摘要和上下文预算执行。使用 Honcho 设置、故障排除... |

## blockchain

| 技能 | 描述 |
|-------|-------------|
| [**base**](/docs/user-guide/skills/optional/blockchain/blockchain-base) | 查询 Base（Ethereum L2）区块链数据，带 USD 定价 — 钱包余额、代币信息、交易详情、gas 分析、合约检查、鲸鱼检测和实时网络统计。使用 Base RPC + CoinGecko。无需 API 密钥。 |
| [**solana**](/docs/user-guide/skills/optional/blockchain/blockchain-solana) | 查询 Solana 区块链数据，带 USD 定价 — 钱包余额、带价值的代币组合、交易详情、NFT、鲸鱼检测和实时网络统计。使用 Solana RPC + CoinGecko。无需 API 密钥。 |

## communication

| 技能 | 描述 |
|-------|-------------|
| [**one-three-one-rule**](/docs/user-guide/skills/optional/communication/communication-one-three-one-rule) | 用于技术提案和权衡分析的结构化决策框架。当用户面临多种方法之间的选择时（架构决策、工具选择、重构策略、迁移路径），此技能 p... |

## creative

| 技能 | 描述 |
|-------|-------------|
| [**blender-mcp**](/docs/user-guide/skills/optional/creative/creative-blender-mcp) | 通过 socket 连接 blender-mcp 插件直接从 Hermes 控制 Blender。创建 3D 对象、材质、动画和运行任意 Blender Python（bpy）代码。用户想要在 Blender 中创建或修改任何内容时使用。 |
| [**concept-diagrams**](/docs/user-guide/skills/optional/creative/creative-concept-diagrams) | 生成平面、最小化支持亮/暗模式的 SVG 图表作为独立 HTML 文件，使用统一的教育视觉语言、9 个语义色阶、句子大小写排版和自动暗模式。最适合教育... |
| [**meme-generation**](/docs/user-guide/skills/optional/creative/creative-meme-generation) | 通过选择模板并用 Pillow 叠加文字生成真实 meme 图像。生成实际的 .png meme 文件。 |

## devops

| 技能 | 描述 |
|-------|-------------|
| [**inference-sh-cli**](/docs/user-guide/skills/optional/devops/devops-cli) | 通过 inference.sh CLI（infsh）运行 150+ AI 应用 — 图像生成、视频创建、LLM、搜索、3D、社交自动化。使用终端工具。触发：inference.sh、infsh、AI 应用、flux、veo、图像生成、视频生成、seedrea... |
| [**docker-management**](/docs/user-guide/skills/optional/devops/devops-docker-management) | 管理 Docker 容器、镜像、卷、网络和 Compose 栈 — 生命周期操作、调试、清理和 Dockerfile 优化。 |

## dogfood

| 技能 | 描述 |
|-------|-------------|
| [**adversarial-ux-test**](/docs/user-guide/skills/optional/dogfood/dogfood-adversarial-ux-test) | 以产品最难、技术最抗拒的用户角色浏览应用。作为那个 persona 找到每个 UX 痛点，然后用实用主义层过滤投诉以区分真实问题... |

## email

| 技能 | 描述 |
|-------|-------------|
| [**agentmail**](/docs/user-guide/skills/optional/email/email-agentmail) | 通过 AgentMail 给代理自己的专属邮箱。使用代理拥有的邮箱地址（如 hermes-agent@agentmail.to）自主发送、接收和管理邮件。 |

## health

| 技能 | 描述 |
|-------|-------------|
| [**fitness-nutrition**](/docs/user-guide/skills/optional/health/health-fitness-nutrition) | 健身房锻炼计划生成器和营养追踪器。通过 wger 按肌肉、设备或类别搜索 690+ 锻炼。通过 USDA FoodData Central 查找 380,000+ 食物的宏量和卡路里。计算 BMI、TDEE、单次最大重复、宏量分配和身体... |
| [**neuroskill-bci**](/docs/user-guide/skills/optional/health/health-neuroskill-bci) | 连接到运行中的 NeuroSkill 实例并将用户实时认知和情绪状态（专注、放松、情绪、认知负荷、嗜睡、心率、HRV、睡眠分阶段和 40+ 衍生 EXG 分数）纳入响应... |

## mcp

| 技能 | 描述 |
|-------|-------------|
| [**fastmcp**](/docs/user-guide/skills/optional/mcp/mcp-fastmcp) | 用 FastMCP 在 Python 中构建、测试、检查、安装和部署 MCP 服务器。需要创建新 MCP 服务器、将 API 或数据库包装为 MCP 工具、暴露资源或提示，或准备用于 Claude Code、Cur... |
| [**mcporter**](/docs/user-guide/skills/optional/mcp/mcp-mcporter) | 使用 mcporter CLI 直接列出、配置、认证和调用 MCP 服务器/工具（HTTP 或 stdio），包括临时服务器、配置编辑和 CLI/类型生成。 |

## migration

| 技能 | 描述 |
|-------|-------------|
| [**openclaw-migration**](/docs/user-guide/skills/optional/migration/migration-openclaw-migration) | 将用户的 OpenClaw 自定义足迹迁移到 Hermes Agent。从 ~/.openclaw 导入 Hermes 兼容的记忆、SOUL.md、命令允许列表、用户技能和选定工作区资产，然后准确报告无法迁移的内容... |

## mlops

| 技能 | 描述 |
|-------|-------------|
| [**huggingface-accelerate**](/docs/user-guide/skills/optional/mlops/mlops-accelerate) | 最简单的分布式训练 API。4 行代码为任何 PyTorch 脚本添加分布式支持。DeepSpeed/FSDP/Megatron/DDP 统一 API。自动设备放置、混合精度（FP16/BF16/FP8）。交互式配置、单次启动通信... |
| [**chroma**](/docs/user-guide/skills/optional/mlops/mlops-chroma) | 用于 AI 应用的开源嵌入数据库。存储嵌入和元数据、执行向量和全文搜索、按元数据过滤。简单 4 函数 API。从笔记本扩展到生产集群。用于语义搜索、RAG... |
| [**clip**](/docs/user-guide/skills/optional/mlops/mlops-clip) | 连接视觉和语言的 OpenAI 模型。实现零样本图像分类、图像文本匹配和跨模态检索。训练于 4 亿图像文本对。用于图像搜索、内容审核或视觉语言任务 w... |
| [**faiss**](/docs/user-guide/skills/optional/mlops/mlops-faiss) | Facebook 用于高效相似性搜索和密集向量聚类的库。支持数十亿向量、GPU 加速和各种索引类型（Flat、IVF、HNSW）。用于快速 k-NN 搜索、大规模向量检索或 whe... |
| [**optimizing-attention-flash**](/docs/user-guide/skills/optional/mlops/mlops-flash-attention) | 使用 Flash Attention 优化 Transformer 注意力，2-4 倍加速和 10-20 倍内存减少。在长序列（>512 token）上训练/运行 Transformer、遇到 GPU 内存问题时使用... |
| [**guidance**](/docs/user-guide/skills/optional/mlops/mlops-guidance) | 用 regex 和语法控制 LLM 输出，保证有效 JSON/XML/代码生成、强制结构化格式并使用 Microsoft Research 的约束生成框架构建多步工作流 |
| [**hermes-atropos-environments**](/docs/user-guide/skills/optional/mlops/mlops-hermes-atropos-environments) | 构建、测试和调试 Atropos 训练的 Hermes Agent RL 环境。涵盖 HermesAgentBaseEnv 接口、奖励函数、代理循环集成、带工具评估、wandb 日志和三种 CLI 模式（serve/process/eva... |
| [**huggingface-tokenizers**](/docs/user-guide/skills/optional/mlops/mlops-huggingface-tokenizers) | 为研究和生产优化的快速分词器。基于 Rust 的实现 20 秒内分词 1GB。支持 BPE、WordPiece 和 Unigram 算法。训练自定义词汇表、跟踪对齐、处理填充/截断。集成... |
| [**instructor**](/docs/user-guide/skills/optional/mlops/mlops-instructor) | 通过 Pydantic 验证从 LLM 响应中提取结构化数据，自动重试失败的提取、解析带类型安全的复杂 JSON 并使用 Instructor 流式传输部分结果 — 经过战斗测试的结构化输出库 |
| [**lambda-labs-gpu-cloud**](/docs/user-guide/skills/optional/mlops/mlops-lambda-labs) | 用于 ML 训练和推理的预留和按需 GPU 云实例。需要带简单 SSH 访问、持久文件系统或高性能多节点集群进行大规模训练的专用 GPU 实例时使用。 |
| [**llava**](/docs/user-guide/skills/optional/mlops/mlops-llava) | 大型语言和视觉助手。实现视觉指令调整和基于图像的对话。结合 CLIP 视觉编码器和 Vicuna/LLaMA 语言模型。支持多轮图像聊天、视觉问答和教学... |
| [**modal-serverless-gpu**](/docs/user-guide/skills/optional/mlops/mlops-modal) | 用于运行 ML 工作负载的无服务器 GPU 云平台。需要无需基础设施管理即可按需 GPU 访问、将 ML 模型部署为 API 或运行具有自动扩展的批处理作业时使用。 |
| [**nemo-curator**](/docs/user-guide/skills/optional/mlops/mlops-nemo-curator) | 用于 LLM 训练的数据整理加速。支持文本/图像/视频/音频。功能包括模糊去重（16 倍快）、质量过滤（30+ 启发式）、语义去重、PII 清理、NSFW 检测。跨 GPU 扩展... |
| [**peft-fine-tuning**](/docs/user-guide/skills/optional/mlops/mlops-peft) | 使用 LoRA、QLoRA 和 25+ 方法对 LLM 进行参数高效微调。用有限 GPU 内存微调大模型（7B-70B）、需要用 <1% 参数训练且准确度损失最小或用于多适配器 se... |
| [**pinecone**](/docs/user-guide/skills/optional/mlops/mlops-pinecone) | 用于生产 AI 应用的托管向量数据库。完全托管、自动扩展、混合搜索（密集 + 稀疏）、元数据过滤和命名空间。低延迟（<100ms p95）。用于生产 RAG、推荐系统或 se... |
| [**pytorch-fsdp**](/docs/user-guide/skills/optional/mlops/mlops-pytorch-fsdp) | 使用 PyTorch FSDP 进行完全分片数据并行训练的专家指导 — 参数分片、混合精度、CPU 卸载、FSDP2 |
| [**pytorch-lightning**](/docs/user-guide/skills/optional/mlops/mlops-pytorch-lightning) | 带 Trainer 类的高级 PyTorch 框架，自动分布式训练（DDP/FSDP/DeepSpeed）、回调系统和最小样板。从笔记本扩展到超级计算机，代码相同。需要干净训练循环 w... |
| [**qdrant-vector-search**](/docs/user-guide/skills/optional/mlops/mlops-qdrant) | 用于 RAG 和语义搜索的高性能向量相似性搜索引擎。构建需要快速最近邻搜索、带有过滤的混合搜索或具有 Rust 驱动持续存在的可扩展向量存储的生产 RAG 系统时使用。 |
| [**sparse-autoencoder-training**](/docs/user-guide/skills/optional/mlops/mlops-saelens) | 提供使用 SAELens 训练和分析稀疏自编码器的指导，将神经网络激活分解为可解释特征。用于发现可解释特征、分析叠加或研究... |
| [**simpo-training**](/docs/user-guide/skills/optional/mlops/mlops-simpo) | 用于 LLM 对齐的简单偏好优化。DPO 的无参考替代方案，性能更好（AlpacaEval 2.0 上 +6.4 点）。不需要参考模型，比 DPO 更高效。需要简单偏好对齐时使用... |
| [**slime-rl-training**](/docs/user-guide/skills/optional/mlops/mlops-slime) | 提供使用 slime（Megatron+SGLang 框架）进行 LLM 后训练的 RL 指导。用于训练 GLM 模型、实现自定义数据生成工作流或需要与 RL 扩展的紧密 Megatron-LM 集成时。 |
| [**stable-diffusion-image-generation**](/docs/user-guide/skills/optional/mlops/mlops-stable-diffusion) | 通过 HuggingFace Diffusers 使用 Stable Diffusion 模型进行最先进的文本到图像生成。用于从文本提示生成图像、执行图像到图像转换、修复或构建自定义扩散管道时。 |
| [**tensorrt-llm**](/docs/user-guide/skills/optional/mlops/mlops-tensorrt-llm) | 使用 NVIDIA TensorRT 优化 LLM 推理以获得最大吞吐量和最低延迟。在 NVIDIA GPU（A100/H100）上进行生产部署、需要比 PyTorch 快 10-100 倍推理或使用量化的服务模型时使用... |
| [**distributed-llm-pretraining-torchtitan**](/docs/user-guide/skills/optional/mlops/mlops-torchtitan) | 使用 torchtitan 通过 4D 并行性（FSDP2、TP、PP、CP）提供 PyTorch 原生分布式 LLM 预训练。使用 Float8、torch.compile 和 dist... 从 8 到 512+ GPU 预训练 Llama 3.1、DeepSeek V3 或自定义模型。 |
| [**whisper**](/docs/user-guide/skills/optional/mlops/mlops-whisper) | OpenAI 的通用语音识别模型。支持 99 种语言、转录、翻译成英语和语言识别。六个模型大小，从小（39M 参数）到大（1550M 参数）。用于语音转文本、播客... |

## productivity

| 技能 | 描述 |
|-------|-------------|
| [**canvas**](/docs/user-guide/skills/optional/productivity/productivity-canvas) | Canvas LMS 集成 — 使用 API 令牌认证获取已注册课程和作业。 |
| [**here.now**](/docs/user-guide/skills/optional/productivity/productivity-here-now) | 将静态站点发布到 &#123;slug&#125;.here.now 并将私有文件存储在云驱动器中用于代理间交接。 |
| [**memento-flashcards**](/docs/user-guide/skills/optional/productivity/productivity-memento-flashcards) | 间隔重复抽认卡系统。从事实或文本创建卡片、使用免费文本答案聊天抽认卡由代理评分、从 YouTube 成绩单生成测验、复习到期卡片与自适应调度以及导出/导入... |
| [**shopify**](/docs/user-guide/skills/optional/productivity/productivity-shopify) | 通过 curl 使用 Shopify Admin & Storefront GraphQL API。产品、订单、客户、库存、元字段。 |
| [**siyuan**](/docs/user-guide/skills/optional/productivity/productivity-siyuan) | 通过 curl 使用 SiYuan Note API 在自托管知识库中搜索、读取、创建和管理块和文档。 |
| [**telephony**](/docs/user-guide/skills/optional/productivity/productivity-telephony) | 无需核心工具更改即可赋予 Hermes 电话能力。配置和持久化 Twilio 号码、发送和接收 SMS/MMS、进行直接通话以及通过 Bland.ai 或 Vapi 放置 AI 驱动的外呼电话。 |

## research

| 技能 | 描述 |
|-------|-------------|
| [**bioinformatics**](/docs/user-guide/skills/optional/research/research-bioinformatics) | 通往 bioSkills 和 ClawBio 400+ 生物信息学技能的门户。涵盖基因组学、转录组学、单细胞、变体调用、药物基因组学、元基因组学、结构生物学等。获取领域特定参考材料... |
| [**domain-intel**](/docs/user-guide/skills/optional/research/research-domain-intel) | 使用 Python stdlib 进行被动域侦察。子域发现、SSL 证书检查、WHOIS 查询、DNS 记录、域可用性检查和批量多域分析。无需 API 密钥。 |
| [**drug-discovery**](/docs/user-guide/skills/optional/research/research-drug-discovery) | 用于药物发现工作流的药物研究助手。在 ChEMBL 上搜索生物活性化合物、计算类药性（Lipinski Ro5、QED、TPSA、合成可及性）、通过 OpenFDA 查找药物相互作用、解释 ADMET... |
| [**duckduckgo-search**](/docs/user-guide/skills/optional/research/research-duckduckgo-search) | 通过 DuckDuckGo 免费网络搜索 — 文本、新闻、图像、视频。无需 API 密钥。优先在当前运行时安装了 `ddgs` CLI 时使用；只有在验证 `ddgs` 可用后才使用 Python DDGS 库。 |
| [**searxng-search**](/docs/user-guide/skills/optional/research/research-searxng-search) | 通过 SearXNG 免费元搜索 — 聚合 70+ 搜索引擎结果。自托管或使用公共实例。无需 API 密钥。当网络搜索工具集不可用时自动回退。 |
| [**gitnexus-explorer**](/docs/user-guide/skills/optional/research/research-gitnexus-explorer) | 使用 GitNexus 索引代码库并通过 Web UI + Cloudflare 隧道提供交互式知识图谱服务。 |
| [**parallel-cli**](/docs/user-guide/skills/optional/research/research-parallel-cli) | Parallel CLI 的可选 vendor 技能 — 代理原生网络搜索、提取、深度研究、丰富、FindAll 和监控。优先使用 JSON 输出和非交互式流程。 |
| [**qmd**](/docs/user-guide/skills/optional/research/research-qmd) | 使用 qmd 本地搜索个人知识库、笔记、文档和会议记录 — 结合 BM25、向量搜索和 LLM 重排的混合检索引擎。支持 CLI 和 MCP 集成。 |
| [**scrapling**](/docs/user-guide/skills/optional/research/research-scrapling) | 使用 Scrapling 进行网络抓取 — HTTP 获取、隐身浏览器自动化、Cloudflare 绕过和通过 CLI 和 Python 进行蜘蛛爬取。 |

## security

| 技能 | 描述 |
|-------|-------------|
| [**1password**](/docs/user-guide/skills/optional/security/security-1password) | 设置和使用 1Password CLI（op）。安装 CLI、启用桌面应用集成、登录以及读取/注入命令秘密时使用。 |
| [**oss-forensics**](/docs/user-guide/skills/optional/security/security-oss-forensics) | GitHub 仓库的供应链调查、证据恢复和取证分析。涵盖删除提交恢复、强制推送检测、IOC 提取、多源证据收集、假设形成/验证和 st... |
| [**sherlock**](/docs/user-guide/skills/optional/security/security-sherlock) | 跨 400+ 社交网络进行 OSINT 用户名搜索。按用户名追踪社交媒体账户。 |

## web-development

| 技能 | 描述 |
|-------|-------------|
| [**page-agent**](/docs/user-guide/skills/optional/web-development/web-development-page-agent) | 将 alibaba/page-agent 嵌入到你自己的网络应用中 — 纯 JavaScript 页面内 GUI 代理，作为单个 &lt;script> 标签或 npm 包提供，让最终用户可以使用自然语言驱动 UI（"点击登录，填写用户名... |

---

## 贡献可选技能

将新的可选技能添加到仓库：

1. 在 `optional-skills/<category>/<skill-name>/` 下创建目录
2. 添加包含标准 frontmatter（name、description、version、author）的 `SKILL.md`
3. 在 `references/`、`templates/` 或 `scripts/` 子目录中包含任何支持文件
4. 提交拉取请求 — 技能将在合并后出现在此目录中并获得自己的文档页面

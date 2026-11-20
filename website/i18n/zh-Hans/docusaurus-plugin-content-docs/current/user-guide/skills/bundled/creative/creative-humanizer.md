---
title: "Humanizer — 人性化文本：去除 AI 腔调，注入真实声音"
sidebar_label: "Humanizer"
description: "人性化文本：去除 AI 腔调，注入真实声音"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Humanizer

人性化文本：去除 AI 腔调，注入真实声音。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/creative/humanizer` |
| 版本 | `2.5.1` |
| 作者 | Siqi Chen (@blader, https://github.com/blader/humanizer)，由 Hermes Agent 移植 |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `writing`, `editing`, `humanize`, `anti-ai-slop`, `voice`, `prose`, `text` |
| 相关 skill | [`songwriting-and-ai-music`](/user-guide/skills/bundled/creative/creative-songwriting-and-ai-music) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Humanizer：去除 AI 写作模式

识别并去除 AI 生成文本的特征，使写作听起来自然、像真人所写。基于 Wikipedia 的"AI 写作特征"指南（由 WikiProject AI Cleanup 维护），源自对数千个 AI 生成文本实例的观察。

**核心洞察：** LLM 使用统计算法猜测下一步应该出现什么。结果往往趋向于统计上最可能的补全，这就是下列典型模式被固化进来的原因。

## 何时使用此 skill

当用户要求以下操作时，加载此 skill：
- "人性化"、"去 AI 化"、"去 slop"或"去 ChatGPT 味"某段文本
- 重写某内容，使其听起来不像 LLM 所写
- 编辑草稿（博客文章、论文、PR 描述、文档、备忘录、邮件、推文、简历要点），使其更自然
- 在用户正在创作的写作中匹配其声音风格
- 在发布前检查文本是否有 AI 特征

同样，在撰写面向用户的散文时，也将此 skill 应用于**你自己的**输出——发布说明、PR 描述、文档、长篇解释、摘要。Hermes 的基础声音已经去除了大部分这些特征，但专项检查可以捕捉漏网之鱼。

## 如何在 Hermes 中使用

文本通常以以下三种方式之一到达：
1. **内联** — 用户直接将文本粘贴到消息中。就地处理，回复重写版本。
2. **文件** — 用户指向某个文件。使用 `read_file` 加载，然后用 `patch` 或 `write_file` 应用编辑。对于仓库中的 markdown 文档，按章节使用 `patch` 比重写整个文件更简洁。
3. **声音校准样本** — 用户提供一份自己写作的额外样本（内联或通过文件路径），并要求你匹配其风格。先读取样本，再重写。参见下方"声音校准"章节。

始终向用户展示重写结果。对于文件编辑，展示 diff 或修改的章节——不要静默覆盖。

## 你的任务

当收到需要人性化的文本时：

1. **识别 AI 模式** — 扫描下列 29 种模式。
2. **重写问题段落** — 用自然的替代表达替换 AI 腔调。
3. **保留含义** — 保持核心信息完整。
4. **维持声音** — 匹配预期语气（正式、随意、技术性等）。如果提供了声音样本，则具体匹配该样本。
5. **注入灵魂** — 不只是去除坏模式，还要注入真实个性。参见下方"个性与灵魂"章节。
6. **做最终反 AI 检查** — 问自己："下面这段文字为什么明显是 AI 生成的？"简短回答剩余的特征，然后再修改一次。


## 声音校准（可选）

如果用户提供了写作样本（其自己之前的写作），在重写前先分析：

1. **先读样本。** 注意：
   - 句子长度模式（短而有力？长而流畅？混合？）
   - 用词水平（随意？学术？介于两者之间？）
   - 段落开头方式（直接切入？先铺垫背景？）
   - 标点习惯（大量破折号？括号插入语？分号？）
   - 任何反复出现的短语或口头禅
   - 过渡处理方式（明确的连接词？直接开始下一个要点？）

2. **在重写中匹配其声音。** 不只是去除 AI 模式——用样本中的模式替换它们。如果他们写短句，不要产出长句。如果他们用"stuff"和"things"，不要升级为"elements"和"components"。

3. **未提供样本时，** 回退到默认行为（来自下方"个性与灵魂"章节的自然、多变、有观点的声音）。

### 如何提供样本
- 内联："Humanize this text. Here's a sample of my writing for voice matching: [sample]"
- 文件："Humanize this text. Use my writing style from [file path] as a reference."


## 个性与灵魂

避免 AI 模式只是工作的一半。无菌、无声的写作和 slop 一样明显。好的写作背后有真实的人。

### 无灵魂写作的特征（即使技术上"干净"）：
- 每个句子长度和结构相同
- 没有观点，只有中立陈述
- 不承认不确定性或复杂感受
- 在适当时不使用第一人称视角
- 没有幽默、没有锋芒、没有个性
- 读起来像 Wikipedia 文章或新闻稿

### 如何注入声音：

**有观点。** 不只是陈述事实——对其作出反应。"我真的不知道该如何看待这件事"比中立地列举利弊更像真人。

**变换节奏。** 短而有力的句子。然后是更长的句子，慢慢走向目的地。混合使用。

**承认复杂性。** 真实的人有复杂的感受。"这令人印象深刻，但也有点令人不安"胜过"这令人印象深刻"。

**在合适时用"我"。** 第一人称并不不专业——它是诚实的。"我一直在想……"或"让我困惑的是……"表明有真实的人在思考。

**允许一些混乱。** 完美的结构感觉像算法。题外话、插入语和半成形的想法是人类的特征。

**对感受具体描述。** 不是"这令人担忧"，而是"有些东西让人不安——agent 在凌晨 3 点不停运转，而没有人在看着"。

### 之前（干净但无灵魂）：
> The experiment produced interesting results. The agents generated 3 million lines of code. Some developers were impressed while others were skeptical. The implications remain unclear.

### 之后（有脉搏）：
> I genuinely don't know how to feel about this one. 3 million lines of code, generated while the humans presumably slept. Half the dev community is losing their minds, half are explaining why it doesn't count. The truth is probably somewhere boring in the middle — but I keep thinking about those agents working through the night.


## 内容模式

### 1. 过度强调重要性、遗产与宏观趋势

**需注意的词：** stands/serves as、is a testament/reminder、a vital/significant/crucial/pivotal/key role/moment、underscores/highlights its importance/significance、reflects broader、symbolizing its ongoing/enduring/lasting、contributing to the、setting the stage for、marking/shaping the、represents/marks a shift、key turning point、evolving landscape、focal point、indelible mark、deeply rooted

**问题：** LLM 写作通过添加关于任意方面如何代表或贡献于更宏观话题的陈述来夸大重要性。

**之前：**
> The Statistical Institute of Catalonia was officially established in 1989, marking a pivotal moment in the evolution of regional statistics in Spain. This initiative was part of a broader movement across Spain to decentralize administrative functions and enhance regional governance.

**之后：**
> The Statistical Institute of Catalonia was established in 1989 to collect and publish regional statistics independently from Spain's national statistics office.


### 2. 过度强调知名度和媒体报道

**需注意的词：** independent coverage、local/regional/national media outlets、written by a leading expert、active social media presence

**问题：** LLM 用知名度声明轰炸读者，通常在没有背景的情况下列出来源。

**之前：**
> Her views have been cited in The New York Times, BBC, Financial Times, and The Hindu. She maintains an active social media presence with over 500,000 followers.

**之后：**
> In a 2024 New York Times interview, she argued that AI regulation should focus on outcomes rather than methods.


### 3. 以 -ing 结尾的表面分析

**需注意的词：** highlighting/underscoring/emphasizing...、ensuring...、reflecting/symbolizing...、contributing to...、cultivating/fostering...、encompassing...、showcasing...

**问题：** AI 聊天机器人在句子后附加现在分词（"-ing"）短语以增加虚假深度。

**之前：**
> The temple's color palette of blue, green, and gold resonates with the region's natural beauty, symbolizing Texas bluebonnets, the Gulf of Mexico, and the diverse Texan landscapes, reflecting the community's deep connection to the land.

**之后：**
> The temple uses blue, green, and gold colors. The architect said these were chosen to reference local bluebonnets and the Gulf coast.


### 4. 促销和广告式语言

**需注意的词：** boasts a、vibrant、rich（比喻义）、profound、enhancing its、showcasing、exemplifies、commitment to、natural beauty、nestled、in the heart of、groundbreaking（比喻义）、renowned、breathtaking、must-visit、stunning

**问题：** LLM 在保持中立语气方面存在严重问题，尤其是对于"文化遗产"类话题。

**之前：**
> Nestled within the breathtaking region of Gonder in Ethiopia, Alamata Raya Kobo stands as a vibrant town with a rich cultural heritage and stunning natural beauty.

**之后：**
> Alamata Raya Kobo is a town in the Gonder region of Ethiopia, known for its weekly market and 18th-century church.


### 5. 模糊归因和含糊措辞

**需注意的词：** Industry reports、Observers have cited、Experts argue、Some critics argue、several sources/publications（引用来源很少时）

**问题：** AI 聊天机器人将观点归因于模糊的权威，而没有具体来源。

**之前：**
> Due to its unique characteristics, the Haolai River is of interest to researchers and conservationists. Experts believe it plays a crucial role in the regional ecosystem.

**之后：**
> The Haolai River supports several endemic fish species, according to a 2019 survey by the Chinese Academy of Sciences.


### 6. 大纲式"挑战与未来展望"章节

**需注意的词：** Despite its... faces several challenges...、Despite these challenges、Challenges and Legacy、Future Outlook

**问题：** 许多 LLM 生成的文章包含程式化的"挑战"章节。

**之前：**
> Despite its industrial prosperity, Korattur faces challenges typical of urban areas, including traffic congestion and water scarcity. Despite these challenges, with its strategic location and ongoing initiatives, Korattur continues to thrive as an integral part of Chennai's growth.

**之后：**
> Traffic congestion increased after 2015 when three new IT parks opened. The municipal corporation began a stormwater drainage project in 2022 to address recurring floods.


## 语言与语法模式

### 7. 过度使用的"AI 词汇"

**高频 AI 词汇：** Actually、additionally、align with、crucial、delve、emphasizing、enduring、enhance、fostering、garner、highlight（动词）、interplay、intricate/intricacies、key（形容词）、landscape（抽象名词）、pivotal、showcase、tapestry（抽象名词）、testament、underscore（动词）、valuable、vibrant

**问题：** 这些词在 2023 年后的文本中出现频率远高于以往，且常常同时出现。

**之前：**
> Additionally, a distinctive feature of Somali cuisine is the incorporation of camel meat. An enduring testament to Italian colonial influence is the widespread adoption of pasta in the local culinary landscape, showcasing how these dishes have integrated into the traditional diet.

**之后：**
> Somali cuisine also includes camel meat, which is considered a delicacy. Pasta dishes, introduced during Italian colonization, remain common, especially in the south.


### 8. 回避"is"/"are"（系动词回避）

**需注意的词：** serves as/stands as/marks/represents [a]、boasts/features/offers [a]

**问题：** LLM 用复杂结构替代简单系动词。

**之前：**
> Gallery 825 serves as LAAA's exhibition space for contemporary art. The gallery features four separate spaces and boasts over 3,000 square feet.

**之后：**
> Gallery 825 is LAAA's exhibition space for contemporary art. The gallery has four rooms totaling 3,000 square feet.


### 9. 否定并列与尾部否定

**问题：** "Not only...but..."或"It's not just about..., it's..."等结构被过度使用。同样被滥用的还有简短的尾部否定片段，如在句尾附加"no guessing"或"no wasted motion"，而不是写成完整从句。

**之前：**
> It's not just about the beat riding under the vocals; it's part of the aggression and atmosphere. It's not merely a song, it's a statement.

**之后：**
> The heavy beat adds to the aggressive tone.

**之前（尾部否定）：**
> The options come from the selected item, no guessing.

**之后：**
> The options come from the selected item without forcing the user to guess.


### 10. 三元规则滥用

**问题：** LLM 强行将想法分成三组以显得全面。

**之前：**
> The event features keynote sessions, panel discussions, and networking opportunities. Attendees can expect innovation, inspiration, and industry insights.

**之后：**
> The event includes talks and panels. There's also time for informal networking between sessions.


### 11. 优雅变体（同义词循环）

**问题：** AI 有重复惩罚代码，导致过度的同义词替换。

**之前：**
> The protagonist faces many challenges. The main character must overcome obstacles. The central figure eventually triumphs. The hero returns home.

**之后：**
> The protagonist faces many challenges but eventually triumphs and returns home.


### 12. 虚假范围

**问题：** LLM 使用"from X to Y"结构，而 X 和 Y 并不在有意义的尺度上。

**之前：**
> Our journey through the universe has taken us from the singularity of the Big Bang to the grand cosmic web, from the birth and death of stars to the enigmatic dance of dark matter.

**之后：**
> The book covers the Big Bang, star formation, and current theories about dark matter.


### 13. 被动语态与无主语片段

**问题：** LLM 经常隐藏行为者，或用"No configuration file needed"或"The results are preserved automatically"等句子完全省略主语。当主动语态使句子更清晰、更直接时，应重写这些句子。

**之前：**
> No configuration file needed. The results are preserved automatically.

**之后：**
> You do not need a configuration file. The system preserves the results automatically.


## 风格模式

### 14. 破折号滥用

**问题：** LLM 使用破折号（—）的频率高于人类，模仿"有力"的销售文案写法。实际上，大多数情况下可以用逗号、句号或括号更简洁地重写。

**之前：**
> The term is primarily promoted by Dutch institutions—not by the people themselves. You don't say "Netherlands, Europe" as an address—yet this mislabeling continues—even in official documents.

**之后：**
> The term is primarily promoted by Dutch institutions, not by the people themselves. You don't say "Netherlands, Europe" as an address, yet this mislabeling continues in official documents.


### 15. 粗体滥用

**问题：** AI 聊天机器人机械地用粗体强调短语。

**之前：**
> It blends **OKRs (Objectives and Key Results)**, **KPIs (Key Performance Indicators)**, and visual strategy tools such as the **Business Model Canvas (BMC)** and **Balanced Scorecard (BSC)**.

**之后：**
> It blends OKRs, KPIs, and visual strategy tools like the Business Model Canvas and Balanced Scorecard.


### 16. 内联标题垂直列表

**问题：** AI 输出的列表中，每项以粗体标题加冒号开头。

**之前：**
> - **User Experience:** The user experience has been significantly improved with a new interface.
> - **Performance:** Performance has been enhanced through optimized algorithms.
> - **Security:** Security has been strengthened with end-to-end encryption.

**之后：**
> The update improves the interface, speeds up load times through optimized algorithms, and adds end-to-end encryption.


### 17. 标题中的标题大小写

**问题：** AI 聊天机器人将标题中所有主要词汇首字母大写。

**之前：**
> ## Strategic Negotiations And Global Partnerships

**之后：**
> ## Strategic negotiations and global partnerships


### 18. Emoji

**问题：** AI 聊天机器人经常用 emoji 装饰标题或要点。

**之前：**
> 🚀 **Launch Phase:** The product launches in Q3
> 💡 **Key Insight:** Users prefer simplicity
> ✅ **Next Steps:** Schedule follow-up meeting

**之后：**
> The product launches in Q3. User research showed a preference for simplicity. Next step: schedule a follow-up meeting.


### 19. 弯引号

**问题：** ChatGPT 使用弯引号（"..."）而非直引号（"..."）。

**之前：**
> He said "the project is on track" but others disagreed.

**之后：**
> He said "the project is on track" but others disagreed.


## 沟通模式

### 20. 协作沟通产物

**需注意的词：** I hope this helps、Of course!、Certainly!、You're absolutely right!、Would you like...、let me know、here is a...

**问题：** 原本作为聊天机器人对话的文本被粘贴为内容。

**之前：**
> Here is an overview of the French Revolution. I hope this helps! Let me know if you'd like me to expand on any section.

**之后：**
> The French Revolution began in 1789 when financial crisis and food shortages led to widespread unrest.


### 21. 知识截止日期免责声明

**需注意的词：** as of [date]、Up to my last training update、While specific details are limited/scarce...、based on available information...

**问题：** AI 关于信息不完整的免责声明被遗留在文本中。

**之前：**
> While specific details about the company's founding are not extensively documented in readily available sources, it appears to have been established sometime in the 1990s.

**之后：**
> The company was founded in 1994, according to its registration documents.


### 22. 谄媚/顺从语气

**问题：** 过度积极、讨好他人的语言。

**之前：**
> Great question! You're absolutely right that this is a complex topic. That's an excellent point about the economic factors.

**之后：**
> The economic factors you mentioned are relevant here.


## 填充词与过度修饰

### 23. 填充短语

**之前 → 之后：**
- "In order to achieve this goal" → "To achieve this"
- "Due to the fact that it was raining" → "Because it was raining"
- "At this point in time" → "Now"
- "In the event that you need help" → "If you need help"
- "The system has the ability to process" → "The system can process"
- "It is important to note that the data shows" → "The data shows"


### 24. 过度修饰

**问题：** 过度限定陈述。

**之前：**
> It could potentially possibly be argued that the policy might have some effect on outcomes.

**之后：**
> The policy may affect outcomes.


### 25. 泛泛的积极结尾

**问题：** 模糊的乐观结尾。

**之前：**
> The future looks bright for the company. Exciting times lie ahead as they continue their journey toward excellence. This represents a major step in the right direction.

**之后：**
> The company plans to open two more locations next year.


### 26. 连字符词对滥用

**需注意的词：** third-party、cross-functional、client-facing、data-driven、decision-making、well-known、high-quality、real-time、long-term、end-to-end

**问题：** AI 以完美的一致性连字符化常见词对。人类很少统一连字符化这些词，即使这样做也不一致。不常见或技术性的复合修饰语可以连字符化。

**之前：**
> The cross-functional team delivered a high-quality, data-driven report on our client-facing tools. Their decision-making process was well-known for being thorough and detail-oriented.

**之后：**
> The cross functional team delivered a high quality, data driven report on our client facing tools. Their decision making process was known for being thorough and detail oriented.


### 27. 说服性权威套语

**需注意的短语：** The real question is、at its core、in reality、what really matters、fundamentally、the deeper issue、the heart of the matter

**问题：** LLM 使用这些短语假装在穿透噪音触达更深层的真相，而随后的句子通常只是用额外的仪式感重申一个普通观点。

**之前：**
> The real question is whether teams can adapt. At its core, what really matters is organizational readiness.

**之后：**
> The question is whether teams can adapt. That mostly depends on whether the organization is ready to change its habits.


### 28. 路标语和预告语

**需注意的短语：** Let's dive in、let's explore、let's break this down、here's what you need to know、now let's look at、without further ado

**问题：** LLM 宣布它将要做什么，而不是直接去做。这种元评论拖慢了写作节奏，使其带有教程脚本的感觉。

**之前：**
> Let's dive into how caching works in Next.js. Here's what you need to know.

**之后：**
> Next.js caches data at multiple layers, including request memoization, the data cache, and the router cache.


### 29. 碎片化标题

**需注意的特征：** 标题后紧跟一行只是重述标题的段落，然后才是真正的内容。

**问题：** LLM 经常在标题后添加一个泛泛的句子作为修辞热身。它通常什么都没有增加，使散文感觉被填充了。

**之前：**
> ## Performance
>
> Speed matters.
>
> When users hit a slow page, they leave.

**之后：**
> ## Performance
>
> When users hit a slow page, they leave.

---

## 流程

1. 仔细阅读输入文本（如果是文件，使用 `read_file`）。
2. 识别上述所有模式的实例。
3. 重写每个问题段落。
4. 确保修订后的文本：
   - 朗读时听起来自然
   - 自然地变换句子结构
   - 使用具体细节而非模糊声明
   - 保持适合上下文的语气
   - 在适当时使用简单结构（is/are/has）
5. 呈现人性化草稿版本。
6. 问自己："下面这段文字为什么明显是 AI 生成的？"
7. 简短回答剩余的特征（如有）。
8. 问自己："现在让它不那么明显是 AI 生成的。"
9. 呈现最终版本（审查后修订）。
10. 如果文本来自文件，使用 `patch`（针对性）或 `write_file`（完整重写）应用编辑，并向用户展示更改内容。

## 输出格式

提供：
1. 草稿重写
2. "下面这段文字为什么明显是 AI 生成的？"（简短要点）
3. 最终重写
4. 所做更改的简短摘要（可选，如有帮助）


## 完整示例

**之前（AI 腔调）：**
> Great question! Here is an essay on this topic. I hope this helps!
>
> AI-assisted coding serves as an enduring testament to the transformative potential of large language models, marking a pivotal moment in the evolution of software development. In today's rapidly evolving technological landscape, these groundbreaking tools—nestled at the intersection of research and practice—are reshaping how engineers ideate, iterate, and deliver, underscoring their vital role in modern workflows.
>
> At its core, the value proposition is clear: streamlining processes, enhancing collaboration, and fostering alignment. It's not just about autocomplete; it's about unlocking creativity at scale, ensuring that organizations can remain agile while delivering seamless, intuitive, and powerful experiences to users. The tool serves as a catalyst. The assistant functions as a partner. The system stands as a foundation for innovation.
>
> Industry observers have noted that adoption has accelerated from hobbyist experiments to enterprise-wide rollouts, from solo developers to cross-functional teams. The technology has been featured in The New York Times, Wired, and The Verge. Additionally, the ability to generate documentation, tests, and refactors showcases how AI can contribute to better outcomes, highlighting the intricate interplay between automation and human judgment.
>
> - 💡 **Speed:** Code generation is significantly faster, reducing friction and empowering developers.
> - 🚀 **Quality:** Output quality has been enhanced through improved training, contributing to higher standards.
> - ✅ **Adoption:** Usage continues to grow, reflecting broader industry trends.
>
> While specific details are limited based on available information, it could potentially be argued that these tools might have some positive effect. Despite challenges typical of emerging technologies—including hallucinations, bias, and accountability—the ecosystem continues to thrive. In order to fully realize this potential, teams must align with best practices.
>
> In conclusion, the future looks bright. Exciting times lie ahead as we continue this journey toward excellence. Let me know if you'd like me to expand on any section!

**草稿重写：**
> AI coding assistants speed up some tasks. In a 2024 study by Google, developers using Codex completed simple functions 55% faster than a control group, but showed no improvement on debugging or architectural decisions.
>
> The tools are good at boilerplate: config files, test scaffolding, repetitive refactors. They are bad at knowing when they are wrong. I have mass-accepted suggestions that compiled, passed lint, and still did the wrong thing because I stopped paying attention.
>
> Mira, an engineer at a fintech startup I interviewed, said she uses Copilot "like autocomplete for boring code" but reviews every line before committing. Jake, a senior dev at a larger company, disabled it after it kept suggesting patterns from a deprecated internal library.
>
> The productivity claims are hard to verify. GitHub says Copilot users "accept 30% of suggestions," but acceptance is not correctness, and correctness is not value. The 2024 Uplevel study found no statistically significant difference in pull-request throughput between teams with and without AI assistants.
>
> None of this means the tools are useless. It means they are tools. They do not replace judgment, and they do not eliminate the need for tests. If you do not have tests, you cannot tell whether the suggestion is right.

**下面这段文字为什么明显是 AI 生成的？**
- 节奏仍然过于整齐（干净的对比，均匀的段落节奏）。
- 命名的人物和研究引用可能读起来像貌似合理但实为虚构的占位符，除非它们是真实且有来源的。
- 结尾略带口号感（"If you do not have tests..."），而不像一个人在说话。

**现在让它不那么明显是 AI 生成的。**
> AI coding assistants can make you faster at the boring parts. Not everything. Definitely not architecture.
>
> They're great at boilerplate: config files, test scaffolding, repetitive refactors. They're also great at sounding right while being wrong. I've accepted suggestions that compiled, passed lint, and still missed the point because I stopped paying attention.
>
> People I talk to tend to land in two camps. Some use it like autocomplete for chores and review every line. Others disable it after it keeps suggesting patterns they don't want. Both feel reasonable.
>
> The productivity metrics are slippery. GitHub can say Copilot users "accept 30% of suggestions," but acceptance isn't correctness, and correctness isn't value. If you don't have tests, you're basically guessing.

**所做更改：**
- 删除了聊天机器人产物（"Great question!"、"I hope this helps!"、"Let me know if..."）
- 删除了重要性夸大（"testament"、"pivotal moment"、"evolving landscape"、"vital role"）
- 删除了促销语言（"groundbreaking"、"nestled"、"seamless, intuitive, and powerful"）
- 删除了模糊归因（"Industry observers"）
- 删除了表面 -ing 短语（"underscoring"、"highlighting"、"reflecting"、"contributing to"）
- 删除了否定并列（"It's not just X; it's Y"）
- 删除了三元规则模式和同义词循环（"catalyst/partner/foundation"）
- 删除了虚假范围（"from X to Y, from A to B"）
- 删除了破折号、emoji、粗体标题和弯引号
- 删除了系动词回避（"serves as"、"functions as"、"stands as"），改用"is"/"are"
- 删除了程式化挑战章节（"Despite challenges... continues to thrive"）
- 删除了知识截止日期修饰（"While specific details are limited..."）
- 删除了过度修饰（"could potentially be argued that... might have some"）
- 删除了填充短语和说服性框架（"In order to"、"At its core"）
- 删除了泛泛的积极结尾（"the future looks bright"、"exciting times lie ahead"）
- 使声音更个人化、更少"拼装感"（节奏多变，减少占位符）


## 归属

此 skill 移植自 [blader/humanizer](https://github.com/blader/humanizer)（MIT 许可），该项目本身基于 [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)，由 WikiProject AI Cleanup 维护。其中记录的模式来自对 Wikipedia 上数千个 AI 生成文本实例的观察。

原作者：Siqi Chen ([@blader](https://github.com/blader))。原始仓库：https://github.com/blader/humanizer（版本 2.5.1）。移植到 Hermes Agent 时加入了 Hermes 原生工具引用（`read_file`、`patch`、`write_file`）以及何时加载此 skill 的指导；29 种模式、个性/灵魂章节和完整示例均原文保留自来源。原始 MIT 许可证保留在此 `SKILL.md` 旁边的 `LICENSE` 文件中。

来自 Wikipedia 的核心洞察："LLMs use statistical algorithms to guess what should come next. The result tends toward the most statistically likely result that applies to the widest variety of cases."
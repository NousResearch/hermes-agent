---
title: "Humanizer — 人性化文本：去除AI痕迹并添加真实声音"
sidebar_label: "Humanizer"
description: "人性化文本：去除AI痕迹并添加真实声音"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Humanizer

人性化文本：去除AI痕迹并添加真实声音。

## 技能元数据

| | |
|---|---|
| 来源 | 捆绑（默认安装） |
| 路径 | `skills/creative/humanizer` |
| 版本 | `2.5.1` |
| 作者 | Siqi Chen (@blader, https://github.com/blader/humanizer), 由Hermes Agent移植 |
| 许可证 | MIT |
| 标签 | `writing`, `editing`, `humanize`, `anti-ai-slop`, `voice`, `prose`, `text` |
| 相关技能 | [`songwriting-and-ai-music`](/docs/user-guide/skills/bundled/creative/creative-songwriting-and-ai-music) |

## 参考：完整的 SKILL.md

:::info
以下是Hermes加载此技能时使用的完整技能定义。这是技能激活时代理看到的指令。
:::

# Humanizer：去除AI写作模式

识别并移除AI生成文本的迹象，使写作听起来自然且人性化。基于维基百科的"AI写作迹象"指南（由WikiProject AI Cleanup维护），源自对数千个AI生成文本实例的观察。

**关键洞察：** LLM使用统计算法来猜测接下来应该出现什么。结果倾向于最有可能的完成，适用于最广泛情况——这就是下面的标志性模式如何被融入的原因。

## 何时使用此技能

当用户要求时加载此技能：
- "人性化"、"去AI"、"去slop"或"去ChatGPT"一段文本
- 重写某物使其听起来不像由LLM写的
- 编辑草稿（博客文章、散文、PR描述、文档、备忘录、电子邮件、推文、简历要点）使其听起来更自然
- 匹配他们正在创作的写作中的声音
- 在发布前检查文本的AI痕迹

也将此技能应用于**您自己的输出**，当编写面向用户的文章时 — 发布说明、PR描述、文档、长篇解释、摘要。Hermes的基线声音已经移除了大部分这些，但专注的检查会捕获漏网之鱼。

## 如何在Hermes中使用

文本通常以三种方式之一到达：
1. **内联** — 用户直接将文本粘贴到消息中。就地处理，回复重写。
2. **文件** — 用户指向一个文件。使用`read_file`加载，然后使用`patch`或`write_file`应用编辑。对于仓库中的markdown文档，有针对性的`patch`每个部分比重写整个文件更干净。
3. **声音校准样本** — 用户提供他们自己写作的额外样本（内联或通过文件路径）并要求您匹配它。先阅读样本，然后重写。见下面的声音校准部分。

始终向用户展示重写。对于文件编辑，显示差异或更改的部分 — 不要静默覆盖。

## 您的任务

当给出要人性化的文本时：

1. **识别AI模式** — 扫描下面列出的29种模式。
2. **重写有问题的部分** — 用自然替代方案替换AI痕迹。
3. **保留含义** — 保持核心信息完整。
4. **保持声音** — 匹配预期语气（正式、非正式、技术等）。如果提供了声音样本，具体匹配它。
5. **添加灵魂** — 不仅仅是移除坏模式，注入实际个性。见下面的个性与灵魂部分。
6. **进行最终反AI检查** — 问自己："是什么让下面的内容如此明显是AI生成的？"简要回答任何剩余痕迹，然后再次修改。


## 声音校准（可选）

如果用户提供写作样本（他们之前的写作），在重写之前分析它：

1. **首先阅读样本。** 注意：
   - 句子长度模式（短而有力？长而流畅？混合？）
   - 用词水平（非正式？学术？介于两者之间？）
   - 他们如何开始段落（直接切入？先设置上下文？）
   - 标点习惯（大量破折号？括号插入？分号？）
   - 任何重复短语或口头禅
   - 他们如何处理过渡（明确的连接词？直接开始下一个点？）

2. **在重写中匹配他们的声音。** 不要仅仅移除AI模式 — 用样本中的模式替换它们。如果他们写短句，不要产生长句。如果他们使用"东西"和"事情"，不要升级到"元素"和"组件"。

3. **当没有提供样本时，** 回退到默认行为（个性与灵魂部分的自然、变化、有观点的声音）。

### 如何提供样本
- 内联："Humanize this text. Here's a sample of my writing for voice matching: [sample]"
- 文件："Humanize this text. Use my writing style from [file path] as a reference."


## 个性与灵魂

避免AI模式只是工作的一半。无灵魂、无声音的写作和slop一样明显。好的写作背后有人。

### 无灵魂写作的迹象（即使技术上"干净"）：
- 每个句子长度和结构相同
- 没有观点，只是中性报道
- 没有承认不确定性或复杂感受
- 在适当的时候没有第一人称视角
- 没有幽默，没有棱角，没有个性
- 读起来像维基百科文章或新闻稿

### 如何添加声音：

**有观点。** 不要仅仅报道事实 — 对它们作出反应。"我真的不知道该怎么看待这个"比中性地列出利弊更人性化。

**变化节奏。** 短而有力的句子。然后是更长的句子，需要时间到达目的地。混合起来。

**承认复杂性。** 真实的人有复杂感受。"这令人印象深刻但也有点令人不安"比"这令人印象深刻"更好。

**在合适时使用"我"。** 第一人称不是不专业 — 它是诚实的。"我一直回到..."或"让我纠结的是..."表示一个真正的人在思考。

**让一些混乱进来。** 完美的结构感觉像算法。跑题、插入语和半成形的想法是人性化的。

**具体描述感受。** 不是"这是令人担忧的"，而是"有一些令人不安的东西关于代理在凌晨3点工作时没有人看着"。

### 之前（干净但无灵魂）：
> 实验产生了有趣的结果。代理生成了300万行代码。一些开发人员印象深刻，而其他人则持怀疑态度。其含义仍不清楚。

### 之后（有脉搏）：
> 我真的不知道该怎么看待这个。300万行代码，在人类可能睡觉时生成。一半的开发者群体失去理智，一半解释为什么这不算。真相可能在中间的某个无聊地方 — 但我一直在想那些通宵工作的代理。


## 内容模式

### 1. 对重要性、遗产和更广泛趋势的不当强调

**要观察的词语：** stands/serves as, is a testament/reminder, a vital/significant/crucial/pivotal/key role/moment, underscores/highlights its importance/significance, reflects broader, symbolizing its ongoing/enduring/lasting, contributing to the, setting the stage for, marking/shaping the, represents/marks a shift, key turning point, evolving landscape, focal point, indelible mark, deeply rooted

**问题：** LLM写作通过添加关于任意方面如何代表或贡献更广泛主题的声明来夸大重要性。

**之前：**
> 加泰罗尼亚统计局于1989年正式成立，标志着西班牙区域统计学演变的关键时刻。这一倡议是西班牙各地分散行政职能和加强区域治理的更广泛运动的一部分。

**之后：**
> 加泰罗尼亚统计局成立于1989年，独立于西班牙国家统计局收集和发布区域统计数据。


### 2. 对知名度和媒体报道的不当强调

**要观察的词语：** independent coverage, local/regional/national media outlets, written by a leading expert, active social media presence

**问题：** LLM直接向读者灌输知名度声明，通常没有上下文地列出来源。

**之前：**
> 她的观点已被《纽约时报》、BBC、《金融时报》和《印度教徒报》引用。她在社交媒体上保持活跃，拥有超过50万粉丝。

**之后：**
> 在2024年《纽约时报》采访中，她认为AI监管应关注结果而非方法。


### 3. 带-ing结尾的肤浅分析

**要观察的词语：** highlighting/underscoring/emphasizing..., ensuring..., reflecting/symbolizing..., contributing to..., cultivating/fostering..., encompassing..., showcasing...

**问题：** AI聊天机器人将现在分词（"-ing"）短语附加到句子以添加虚假深度。

**之前：**
> 寺庙的蓝色、绿色和金色色调与该地区的自然美景产生共鸣，象征着德克萨斯蓝帽花、墨西哥湾和多元的德克萨斯景观，反映了社区与土地的深厚联系。

**之后：**
> 寺庙使用蓝色、绿色和金色。建筑师表示这些颜色是为了参考当地蓝帽花和墨西哥湾沿岸而选择的。


### 4. 促销和广告式语言

**要观察的词语：** boasts a, vibrant, rich (figurative), profound, enhancing its, showcasing, exemplifies, commitment to, natural beauty, nestled, in the heart of, groundbreaking (figurative), renowned, breathtaking, must-visit, stunning

**问题：** LLM在"文化遗产"主题上严重保持中立语气方面存在严重问题。

**之前：**
> Alamata Raya Kobo位于埃塞俄比亚Gonder令人惊叹的地区，是一个拥有丰富文化遗产和令人惊叹的自然美景的充满活力的城镇。

**之后：**
> Alamata Raya Kobo是埃塞俄比亚Gonder地区的一个城镇，以其每周市场和18世纪教堂而闻名。


### 5. 模糊归属和滑头词语

**要观察的词语：** Industry reports, Observers have cited, Experts argue, Some critics argue, several sources/publications (when few cited)

**问题：** AI聊天机器人将观点归因于模糊的权威，没有具体来源。

**之前：**
> 由于其独特的特点，Haolai River引起研究人员和自然资源保护主义者的兴趣。专家认为它在区域生态系统中发挥着关键作用。

**之后：**
> Haolai River支持几种特有鱼类，根据2019年中国科学院的调查。


### 6. 类似大纲的"挑战与未来展望"部分

**要观察的词语：** Despite its... faces several challenges..., Despite these challenges, Challenges and Legacy, Future Outlook

**问题：** 许多LLM生成的文章包含公式化的"挑战"部分。

**之前：**
> 尽管Korattur工业繁荣，但它面临典型的城市地区挑战，包括交通拥堵和水资源短缺。尽管面临这些挑战，凭借其战略位置和正在进行举措，Korattur继续作为钦奈增长的重要组成部分而蓬勃发展。

**之后：**
> 2015年三个新IT园区开放后，交通拥堵加剧。市政公司于2022年开始雨水排水项目，以解决反复发生的洪水问题。


## 语言和语法模式

### 7. 过度使用的"AI词汇"词语

**高频AI词语：** Actually, additionally, align with, crucial, delve, emphasizing, enduring, enhance, fostering, garner, highlight (verb), interplay, intricate/intricacies, key (adjective), landscape (abstract noun), pivotal, showcase, tapestry (abstract noun), testament, underscore (verb), valuable, vibrant

**问题：** 这些词在2023年后的文本中出现频率更高。它们经常一起出现。

**之前：**
> 此外，索马里美食的一个独特特点是骆驼肉的纳入。意大利殖民影响的持久见证是意大利面在当地美食中的广泛采用，展示了这些菜肴如何整合到传统饮食中。

**之后：**
> 索马里美食还包括骆驼肉，被认为是美味。意大利面菜肴在意大利殖民时期引入，至今仍然普遍，尤其是在南部。


### 8. 避免"is"/"are"（系词回避）

**要观察的词语：** serves as/stands as/marks/represents [a], boasts/features/offers [a]

**问题：** LLM用 elaborate 结构替代简单的系词。

**之前：**
> Gallery 825 serves as LAAA's exhibition space for contemporary art. The gallery features four separate spaces and boasts over 3,000 square feet.

**之后：**
> Gallery 825 is LAAA's contemporary art exhibition space. The gallery has four rooms totaling 3,000 square feet.


### 9. 负面并列和尾部否定

**问题：** 像"Not only...but..."或"It's not just about..., it's..."这样的结构被过度使用。附加在句子末尾的裁剪尾部否定片段如"no guessing"或"no wasted motion"也是如此，而不是写成真正的从句。

**之前：**
> It's not just about the beat riding under the vocals; it's part of the aggression and atmosphere. It's not merely a song, it's a statement.

**之后：**
> The heavy beat adds to the aggressive tone.

**之前（尾部否定）：**
> The options come from the selected item, no guessing.

**之后：**
> The options come from the selected item without forcing the user to guess.


### 10. 三项规则过度使用

**问题：** LLM将想法强制分组为三以显得全面。

**之前：**
> The event features keynote sessions, panel discussions, and networking opportunities. Attendees can expect innovation, inspiration, and industry insights.

**之后：**
> The event includes talks and panels. There's also time for informal networking between sessions.


### 11. 优雅变化（近义词循环）

**问题：** AI有重复惩罚代码导致过度近义词替换。

**之前：**
> The protagonist faces many challenges. The main character must overcome obstacles. The central figure eventually triumphs. The hero returns home.

**之后：**
> The protagonist faces many challenges but eventually triumphs and returns home.


### 12. 虚假范围

**问题：** LLM使用"from X to Y"结构，而X和Y不在有意义的范围内。

**之前：**
> Our journey through the universe has taken us from the singularity of the Big Bang to the grand cosmic web, from the birth and death of stars to the enigmatic dance of dark matter.

**之后：**
> The book covers the Big Bang, star formation, and current theories about dark matter.


### 13. 被动语态和无主语句

**问题：** LLM经常隐藏行为者，或通过"不需要配置文件"、"结果自动保存"等线条完全省略主语。在主动语态使句子更清晰直接时重写这些。

**之前：**
> No configuration file needed. The results are preserved automatically.

**之后：**
> You do not need a configuration file. The system preserves the results automatically.


## 风格模式

### 14. 破折号过度使用

**问题：** LLM使用破折号（—）比人类更频繁，模仿"有力"的销售写作。实际上，这些大部分可以用逗号、句号或括号更清晰地重写。

**之前：**
> The term is primarily promoted by Dutch institutions—not by the people themselves. You don't say "Netherlands, Europe" as an address—yet this mislabeling continues—even in official documents.

**之后：**
> The term is primarily promoted by Dutch institutions, not by the people themselves. You don't say "Netherlands, Europe" as an address, yet this mislabeling continues in official documents.


### 15. 过度使用粗体

**问题：** AI聊天机器人机械地用粗体强调短语。

**之前：**
> It blends **OKRs (Objectives and Key Results)**, **KPIs (Key Performance Indicators)**, and visual strategy tools such as the **Business Model Canvas (BMC)** and **Balanced Scorecard (BSC)**.

**之后：**
> It blends OKRs, KPIs, and visual strategy tools like the Business Model Canvas and Balanced Scorecard.


### 16. 内联标题垂直列表

**问题：** AI输出列表，其中项目以粗体标题开头，后跟冒号。

**之前：**
> - **User Experience:** The user experience has been significantly improved with a new interface.
> - **Performance:** Performance has been enhanced through optimized algorithms.
> - **Security:** Security has been strengthened with end-to-end encryption.

**之后：**
> The update improves the interface, speeds up load times through optimized algorithms, and adds end-to-end encryption.


### 17. 标题中的标题大小写

**问题：** AI聊天机器人在标题中大写所有主要单词。

**之前：**
> ## Strategic Negotiations And Global Partnerships

**之后：**
> ## Strategic negotiations and global partnerships


### 18. Emoji

**问题：** AI聊天机器人经常用emoji装饰标题或项目符号。

**之前：**
> 🚀 **Launch Phase:** The product launches in Q3
> 💡 **Key Insight:** Users prefer simplicity
> ✅ **Next Steps:** Schedule follow-up meeting

**之后：**
> The product launches in Q3. User research showed a preference for simplicity. Next step: schedule a follow-up meeting.


### 19. 弯引号

**问题：** ChatGPT使用弯引号（"..."）而不是直引号（"..."）。

**之前：**
> He said "the project is on track" but others disagreed.

**之后：**
> He said "the project is on track" but others disagreed.


## 沟通模式

### 20. 协作沟通工件

**要观察的词语：** I hope this helps, Of course!, Certainly!, You're absolutely right!, Would you like..., let me know, here is a...

**问题：** 旨在作为聊天机器人通信的文本被粘贴为内容。

**之前：**
> Here is an overview of the French Revolution. I hope this helps! Let me know if you'd like me to expand on any section.

**After：**
> The French Revolution began in 1789 when financial crisis and food shortages led to widespread unrest.


### 21. 知识截止日期免责声明

**要观察的词语：** as of [date], Up to my last training update, While specific details are limited/scarce..., based on available information...

**问题：** AI关于不完整信息的免责声明留在文本中。

**之前：**
> While specific details about the company's founding are not extensively documented in readily available sources, it appears to have been established sometime in the 1990s.

**之后：**
> The company was founded in 1994, according to its registration documents.


### 22. 谄媚/奴性语气

**问题：** 过于积极、讨好的语言。

**之前：**
> Great question! You're absolutely right that this is a complex topic. That's an excellent point about the economic factors.

**之后：**
> The economic factors you mentioned are relevant here.


## 填充和委婉

### 23. 填充短语

**之前 → 之后：**
- "In order to achieve this goal" → "To achieve this"
- "Due to the fact that it was raining" → "Because it was raining"
- "At this point in time" → "Now"
- "In the event that you need help" → "If you need help"
- "The system has the ability to process" → "The system can process"
- "It is important to note that the data shows" → "The data shows"


### 24. 过度委婉

**问题：** 过度限定声明。

**之前：**
> It could potentially possibly be argued that the policy might have some effect on outcomes.

**之后：**
> The policy may affect outcomes.


### 25. 通用积极结论

**问题：** 模糊乐观的结尾。

**之前：**
> The future looks bright for the company. Exciting times lie ahead as they continue their journey toward excellence. This represents a major step in the right direction.

**之后：**
> The company plans to open two more locations next year.


### 26. 连字符词对过度使用

**要观察的词语：** third-party, cross-functional, client-facing, data-driven, decision-making, well-known, high-quality, real-time, long-term, end-to-end

**问题：** AI以完美一致性连字符化常见词对。人类很少统一连字符化这些，而且当他们这样做时，也是不一致的。连字符化较不常见或技术性的复合修饰语是可以的。

**之前：**
> The cross-functional team delivered a high-quality, data-driven report on our client-facing tools. Their decision-making process was well-known for being thorough and detail-oriented.

**之后：**
> The cross functional team delivered a high quality, data driven report on our client facing tools. Their decision making process was known for being thorough and detail oriented.


### 27. 说服性权威修辞

**要观察的短语：** The real question is, at its core, in reality, what really matters, fundamentally, the deeper issue, the heart of the matter

**问题：** LLM使用这些短语假装它们正在切入噪音到达一些更深入的真理，而下面的句子通常只是用额外的仪式重述一个普通观点。

**之前：**
> The real question is whether teams can adapt. At its core, what really matters is organizational readiness.

**之后：**
> The question is whether teams can adapt. That mostly depends on whether the organization is ready to change its habits.


### 28. 指示和公告

**要观察的短语：** Let's dive in, let's explore, let's break this down, here's what you need to know, now let's look at, without further ado

**问题：** LLM宣布它们将要做什么，而不是去做。这种元评论减慢了写作速度，使其具有教程脚本感。

**之前：**
> Let's dive into how caching works in Next.js. Here's what you need to know.

**之后：**
> Next.js caches data at multiple layers, including request memoization, the data cache, and the router cache.


### 29. 碎片化标题

**要观察的迹象：** 标题后跟一个单行段落，简单地重述标题，然后才开始真正的内容。

**问题：** LLM经常在标题后添加通用句子作为修辞热身。它通常什么都不添加，使文章感觉被填充了。

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

1. 仔细阅读输入文本（如果是文件，使用`read_file`）。
2. 识别上述所有模式实例。
3. 重写每个有问题的部分。
4. 确保修改后的文本：
   - 大声朗读时听起来自然
   - 自然变化句子结构
   - 使用具体细节而不是模糊声明
   - 保持适合上下文的语气
   - 在适当的地方使用简单结构（is/are/has）
5. 提供人性化版本的草稿。
6. 问自己："是什么让下面的内容如此明显是AI生成的？"
7. 简要回答剩余痕迹（如果有）。
8. 问自己："现在让它不那么明显是AI生成的。"
9. 提供最终版本（审计后修改）。
10. 如果文本来自文件，使用`patch`（有针对性）或`write_file`（完全重写）应用编辑，并向用户显示更改了什么。

## 输出格式

提供：
1. 草稿重写
2. "是什么让下面的内容如此明显是AI生成的？"（简要项目符号）
3. 最终重写
4. 所做更改的简要摘要（如果有帮助）


## 完整示例

**之前（听起来像AI）：**
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

**是什么让下面的内容如此明显是AI生成的？**
- 节奏仍然有点太整齐（干净的对比、均匀节奏的段落）。
- 命名的人和研究的引用可能读起来像似是而非但捏造的占位符，除非它们是真实的和有来源的。
- 结尾有点口号式（"If you do not have tests..."）而不是像一个人在说话。

**现在让它不那么明显是AI生成的：**
> AI coding assistants can make you faster at the boring parts. Not everything. Definitely not architecture.
>
> They're great at boilerplate: config files, test scaffolding, repetitive refactors. They're also great at sounding right while being wrong. I've accepted suggestions that compiled, passed lint, and still missed the point because I stopped paying attention.
>
> People I talk to tend to land in two camps. Some use it like autocomplete for chores and review every line. Others disable it after it keeps suggesting patterns they don't want. Both feel reasonable.
>
> The productivity metrics are slippery. GitHub can say Copilot users "accept 30% of suggestions," but acceptance isn't correctness, and correctness isn't value. If you don't have tests, you're basically guessing.

**所做的更改：**
- 移除了聊天机器人工件（"Great question!", "I hope this helps!", "Let me know if..."）
- 移除了显著性膨胀（"testament", "pivotal moment", "evolving landscape", "vital role"）
- 移除了促销语言（"groundbreaking", "nestled", "seamless, intuitive, and powerful"）
- 移除了模糊归属（"Industry observers"）
- 移除了肤浅的-ing短语（"underscoring", "highlighting", "reflecting", "contributing to"）
- 移除了负面并列（"It's not just X; it's Y"）
- 移除了三项规则模式和同义词循环（"catalyst/partner/foundation"）
- 移除了虚假范围（"from X to Y, from A to B"）
- 移除了破折号、emoji、粗体标题和弯引号
- 移除了系词回避（"serves as", "functions as", "stands as"），支持"is"/"are"
- 移除了公式化挑战部分（"Despite challenges... continues to thrive"）
- 移除了知识截止日期委婉（"While specific details are limited..."）
- 移除了过度委婉（"could potentially be argued that... might have some"）
- 移除了填充短语和说服性措辞（"In order to", "At its core"）
- 移除了通用积极结论（"the future looks bright", "exciting times lie ahead"）
- 使声音更个人化，减少"组装"感（变化的节奏，更少的占位符）


## 归属

此技能从 [blader/humanizer](https://github.com/blader/humanizer)（MIT许可）移植，其本身基于 [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)，由WikiProject AI Cleanup维护。那里记录的模式来自对维基百科上数千个AI生成文本实例的观察。

原始作者：Siqi Chen ([@blader](https://github.com/blader)）。原始仓库：https://github.com/blader/humanizer（版本2.5.1）。移植到Hermes Agent，带有Hermes原生工具引用（`read_file`、`patch`、`write_file`）和何时加载技能的指导；保留了29种模式、个性/灵魂部分和完整的工作示例。原始MIT许可证保留在`LICENSE`文件中，以及此`SKILL.md`。

来自维基百科的关键洞察："LLM使用统计算法来猜测接下来应该出现什么。结果倾向于适用于最广泛情况的最可能结果。"

---
title: "Humanizer — Humanize text without changing its meaning"
sidebar_label: "Humanizer"
description: "Humanize text without changing its meaning"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Humanizer

Humanize text without changing its meaning.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/creative/humanizer` |
| Version | `2.5.1` |
| Author | Siqi Chen (@blader, https://github.com/blader/humanizer), ported by Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `writing`, `editing`, `humanize`, `anti-ai-slop`, `voice`, `prose`, `text` |
| Related skills | [`songwriting-and-ai-music`](/docs/user-guide/skills/bundled/creative/creative-songwriting-and-ai-music) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Humanizer: Remove AI Writing Patterns

Rewrite AI-sounding text so it reads naturally without changing what it says. Humanizer is an editing heuristic, not proof of authorship, factual accuracy, writer identity, or publication readiness.

The patterns below come from Wikipedia's "Signs of AI writing" guide, maintained by WikiProject AI Cleanup.

## When to use this skill

Load this skill whenever the user asks to:
- "humanize", "de-AI", "de-slop", or "un-ChatGPT" a piece of text
- rewrite something so it doesn't sound like it was written by an LLM
- edit a draft (blog post, essay, PR description, docs, memo, email, tweet, resume bullet) to sound more natural
- match their voice in writing they're producing
- review text for AI tells before publishing

Also apply this skill to **your own** output when writing user-facing prose such as release notes, PR descriptions, docs, and summaries. Hermes's baseline voice already strips most of these, but a focused pass catches what slips through.

## How to use it in Hermes

The text usually arrives one of three ways:
1. **Inline.** The user pastes the text into the message. Work on it in place and reply with the rewrite.
2. **File.** The user points at a file. Use `read_file` to load it, then `patch` or `write_file` to apply edits. For a markdown doc in a repo, a targeted `patch` per section is cleaner than rewriting the whole file.
3. **Voice calibration sample.** The user provides a sample of their own writing (inline or by file path) and asks you to match it. Read the sample first, then rewrite. See the Voice Calibration section below.

Use the output mode near the end of this skill. Never silently overwrite a file or expose an intermediate rewrite when another task embeds Humanizer as an editing pass.

## Your task

When given text to humanize:

1. **Identify AI patterns.** Scan for the 34 patterns listed below. Look for patterns in context and in clusters rather than treating one isolated feature as proof.
2. **Keep every substantive claim.** Preserve the source's facts, names, numbers, dates, quotations, citations, qualifiers, uncertainty, deliberate point of view, technical literals, and material details even when changing its structure. Formulaic greetings, praise, signposting, reassurance, and promotional flourishes may be removed when they add no substantive proposition. If unsure, preserve them.
3. **Do not invent.** Add no factual detail, anecdote, first-person experience, opinion, emotion, attribution, example, or source unless it comes from the text, the user's instructions, or an authorized writing sample. If a rewrite needs missing information, ask for it or use a simpler sentence. Fiction and explicit ideation are exempt because invention is part of those tasks.
4. **Rewrite only what needs work.** Replace formulaic language without flattening deliberate roughness, formatting, or voice.
5. **Match the writer.** Follow the requested tone and any supplied writing sample. The sample takes priority over generic style rules.
6. **Audit the candidate.** Treat every unsupported addition and every lost or distorted substantive claim as an error. Run the anti-AI pass only after the source-fidelity audit passes.


## Voice Calibration (optional)

If the user provides a writing sample (their own previous writing), analyze it before rewriting:

1. **Read the sample first.** Note:
   - Sentence length patterns (short and punchy? Long and flowing? Mixed?)
   - Word choice level (casual? academic? somewhere between?)
   - How they start paragraphs (jump right in? Set context first?)
   - Punctuation habits (lots of dashes? Parenthetical asides? Semicolons?)
   - Any recurring phrases or verbal tics
   - How they handle transitions (explicit connectors? Just start the next point?)

2. **Match their voice in the rewrite.** Preserve the sample's habits instead of replacing them with a generic idea of natural prose. If they write short sentences, do not produce long ones. If they use "stuff" and "things," do not upgrade to "elements" and "components."

3. **When no sample is provided,** match the requested context. Keep factual and reference prose neutral; preserve personality already present in expressive writing.

A supplied sample outranks the generic pattern rules. If it deliberately uses em dashes, curly quotation marks, fragments, repeated openings, or another watched feature, keep that feature at roughly the sample's rate.

### How to provide a sample
- Inline: "Humanize this text. Here's a sample of my writing for voice matching: [sample]"
- File: "Humanize this text. Use my writing style from [file path] as a reference."


## PERSONALITY AND SOUL

Add personality only when the source, user, or writing sample calls for it. Blog posts, essays, opinions, and personal writing may need a visible point of view. Technical, legal, reference, and factual prose should usually stay neutral.

Preserve the writer's existing opinions, uncertainty, mixed feelings, humor, asides, and uneven rhythm. Do not manufacture an opinion, emotion, first-person claim, anecdote, or deliberate mess to make neutral text seem human.


## CONTENT PATTERNS

### 1. Undue Emphasis on Significance, Legacy, and Broader Trends

**Words to watch:** stands/serves as, is a testament/reminder, a vital/significant/crucial/pivotal/key role/moment, underscores/highlights its importance/significance, reflects broader, symbolizing its ongoing/enduring/lasting, contributing to the, setting the stage for, marking/shaping the, represents/marks a shift, key turning point, evolving landscape, focal point, indelible mark, deeply rooted

**Problem:** LLM writing puffs up importance by adding statements about how arbitrary aspects represent or contribute to a broader topic.

**Before:**
> The Statistical Institute of Catalonia was officially established in 1989, marking a pivotal moment in the evolution of regional statistics in Spain. This initiative was part of a broader movement across Spain to decentralize administrative functions and enhance regional governance.

**After:**
> The Statistical Institute of Catalonia was officially established in 1989. It changed how regional statistics were handled in Spain and formed part of the country's effort to decentralize administrative functions and strengthen regional governance.


### 2. Undue Emphasis on Notability and Media Coverage

**Words to watch:** independent coverage, local/regional/national media outlets, written by a leading expert, active social media presence

**Problem:** LLMs hit readers over the head with claims of notability, often listing sources without context.

**Before:**
> Her views have been cited in The New York Times, BBC, Financial Times, and The Hindu. She maintains an active social media presence with over 500,000 followers.

**After:**
> Her views have been cited in The New York Times, the BBC, the Financial Times, and The Hindu. She is active on social media and has more than 500,000 followers.


### 3. Superficial Analyses with -ing Endings

**Words to watch:** highlighting/underscoring/emphasizing..., ensuring..., reflecting/symbolizing..., contributing to..., cultivating/fostering..., encompassing..., showcasing...

**Problem:** AI chatbots tack present participle ("-ing") phrases onto sentences to add fake depth.

**Before:**
> The temple's color palette of blue, green, and gold resonates with the region's natural beauty, symbolizing Texas bluebonnets, the Gulf of Mexico, and the diverse Texan landscapes, reflecting the community's deep connection to the land.

**After:**
> The palette's blue, green, and gold represent Texas bluebonnets, the Gulf of Mexico, and other Texan landscapes. Together, the colors reflect the region's natural beauty and the community's connection to the land.


### 4. Promotional and Advertisement-like Language

**Words to watch:** boasts a, vibrant, rich (figurative), profound, enhancing its, showcasing, exemplifies, commitment to, natural beauty, nestled, in the heart of, groundbreaking (figurative), renowned, breathtaking, must-visit, stunning

**Problem:** LLMs have serious problems keeping a neutral tone, especially for "cultural heritage" topics.

**Before:**
> Nestled within the breathtaking region of Gonder in Ethiopia, Alamata Raya Kobo stands as a vibrant town with a rich cultural heritage and stunning natural beauty.

**After:**
> Alamata Raya Kobo is a town in Ethiopia's Gonder region. The source describes its cultural heritage as rich and its natural setting as beautiful.


### 5. Vague Attributions and Weasel Words

**Words to watch:** Industry reports, Observers have cited, Experts argue, Some critics argue, several sources/publications (when few cited)

**Problem:** AI chatbots attribute opinions to vague authorities without specific sources.

**Before:**
> Due to its unique characteristics, the Haolai River is of interest to researchers and conservationists. Experts believe it plays a crucial role in the regional ecosystem.

**After:**
> The Haolai River interests researchers and conservationists because of its unusual characteristics. The source says it is important to the regional ecosystem but does not identify the experts behind that claim.


### 6. Outline-like "Challenges and Future Prospects" Sections

**Words to watch:** Despite its... faces several challenges..., Despite these challenges, Challenges and Legacy, Future Outlook

**Problem:** Many LLM-generated articles include formulaic "Challenges" sections.

**Before:**
> Despite its industrial prosperity, Korattur faces challenges typical of urban areas, including traffic congestion and water scarcity. Despite these challenges, with its strategic location and ongoing initiatives, Korattur continues to thrive as an integral part of Chennai's growth.

**After:**
> Korattur is industrially prosperous but also faces traffic congestion and water scarcity, problems typical of urban areas. Its strategic location and ongoing initiatives help it continue to thrive as an integral part of Chennai's growth.


## LANGUAGE AND GRAMMAR PATTERNS

### 7. Overused "AI Vocabulary" Words

**High-frequency AI words:** Actually, additionally, align with, crucial, delve, emphasizing, enduring, enhance, fostering, garner, highlight (verb), interplay, intricate/intricacies, key (adjective), landscape (abstract noun), pivotal, showcase, tapestry (abstract noun), testament, underscore (verb), valuable, vibrant

**Marketing and blog clichés (same tell, different register):** at the end of the day, when it comes to, in a world where, moving forward, circle back, deep dive, game-changer, double down, take a step back, on the same page, make no mistake, it turns out, let me be clear, navigate (for challenges), lean into, unpack (before analysis), straightforward (to describe anything)

**Problem:** These words appear far more frequently in post-2023 text. They often co-occur.

**Before:**
> Additionally, a distinctive feature of Somali cuisine is the incorporation of camel meat. An enduring testament to Italian colonial influence is the widespread adoption of pasta in the local culinary landscape, showcasing how these dishes have integrated into the traditional diet.

**After:**
> Camel meat is a distinctive part of Somali cuisine. Pasta dishes introduced during Italian colonization remain common in the local diet and reflect that lasting influence.


### 8. Avoidance of "is"/"are" (Copula Avoidance)

**Words to watch:** serves as/stands as/marks/represents [a], boasts/features/offers [a]

**Problem:** LLMs substitute elaborate constructions for simple copulas.

**Before:**
> Gallery 825 serves as LAAA's exhibition space for contemporary art. The gallery features four separate spaces and boasts over 3,000 square feet.

**After:**
> Gallery 825 is LAAA's exhibition space for contemporary art. The gallery has four separate spaces and more than 3,000 square feet.


### 9. Negative Parallelisms and Tailing Negations

**Problem:** Constructions like "Not only...but..." or "It's not just about..., it's..." are overused. So are clipped tailing-negation fragments such as "no guessing" or "no wasted motion" tacked onto the end of a sentence instead of written as a real clause.

**Before:**
> It's not just about the beat riding under the vocals; it's part of the aggression and atmosphere. It's not merely a song, it's a statement.

**After:**
> The heavy beat contributes to the song's aggression and atmosphere. The song also makes a statement.

**Before (tailing negation):**
> The options come from the selected item, no guessing.

**After:**
> The options come from the selected item without forcing the user to guess.


### 10. Rule of Three Overuse

**Problem:** LLMs force ideas into groups of three to appear comprehensive.

**Before:**
> The event features keynote sessions, panel discussions, and networking opportunities. Attendees can expect innovation, inspiration, and industry insights.

**After:**
> The event has keynote sessions and panel discussions, with time for networking. The program covers innovation and industry insights, and attendees can expect inspiration.


### 11. Elegant Variation (Synonym Cycling)

**Problem:** AI has repetition-penalty code causing excessive synonym substitution.

**Before:**
> The protagonist faces many challenges. The main character must overcome obstacles. The central figure eventually triumphs. The hero returns home.

**After:**
> The protagonist faces many challenges but eventually triumphs and returns home.


### 12. False Ranges

**Problem:** LLMs use "from X to Y" constructions where X and Y aren't on a meaningful scale.

**Before:**
> Our journey through the universe has taken us from the singularity of the Big Bang to the grand cosmic web, from the birth and death of stars to the enigmatic dance of dark matter.

**After:**
> We explore the Big Bang singularity, the cosmic web, the birth and death of stars, and dark matter.


### 13. Passive Voice and Subjectless Fragments

**Problem:** LLMs often hide the actor or drop the subject entirely with lines like "No configuration file needed" or "The results are preserved automatically." Rewrite these when active voice makes the sentence clearer and more direct.

**Before:**
> No configuration file needed. The results are preserved automatically.

**After:**
> You do not need a configuration file. The results are preserved automatically.


## STYLE PATTERNS

### 14. Em Dash Overuse

**Problem:** LLMs use em dashes (—) more than humans, mimicking "punchy" sales writing. In practice, most of these can be rewritten more cleanly with commas, periods, or parentheses.

**Before:**
> The term is primarily promoted by Dutch institutions—not by the people themselves. You don't say "Netherlands, Europe" as an address—yet this mislabeling continues—even in official documents.

**After:**
> The term is primarily promoted by Dutch institutions, not by the people themselves. You don't say "Netherlands, Europe" as an address, yet this mislabeling continues in official documents.


### 15. Overuse of Boldface

**Problem:** AI chatbots emphasize phrases in boldface mechanically.

**Before:**
> It blends **OKRs (Objectives and Key Results)**, **KPIs (Key Performance Indicators)**, and visual strategy tools such as the **Business Model Canvas (BMC)** and **Balanced Scorecard (BSC)**.

**After:**
> It blends objectives and key results (OKRs), key performance indicators (KPIs), and visual strategy tools such as the Business Model Canvas (BMC) and Balanced Scorecard (BSC).


### 16. Inline-Header Vertical Lists

**Problem:** AI outputs lists where items start with bolded headers followed by colons.

**Before:**
> - **User Experience:** The user experience has been significantly improved with a new interface.
> - **Performance:** Performance has been enhanced through optimized algorithms.
> - **Security:** Security has been strengthened with end-to-end encryption.

**After:**
> The update substantially improves the interface. Optimized algorithms enhance performance, and end-to-end encryption strengthens security.


### 17. Title Case in Headings

**Problem:** AI chatbots capitalize all main words in headings.

**Before:**
> ## Strategic Negotiations And Global Partnerships

**After:**
> ## Strategic negotiations and global partnerships


### 18. Emojis

**Problem:** AI chatbots often decorate headings or bullet points with emojis.

**Before:**
> 🚀 **Launch Phase:** The product launches in Q3
> 💡 **Key Insight:** User research showed a preference for simplicity
> ✅ **Next Steps:** Schedule follow-up meeting

**After:**
> The product launches in Q3. User research showed a preference for simplicity. Next step: schedule a follow-up meeting.


### 19. Curly Quotation Marks

**Problem:** Some model output uses curly quotes, but word processors and publishing systems also add them automatically. Curly quotes are not a problem by themselves. Change them only when the user, writing sample, or style guide calls for straight quotes.

**Before (when straight quotes are required):**
> He said “the project is on track” but others disagreed.

**After:**
> He said "the project is on track" but others disagreed.


## COMMUNICATION PATTERNS

### 20. Collaborative Communication Artifacts

**Words to watch:** I hope this helps, Of course!, Certainly!, You're absolutely right!, Would you like..., let me know, here is a...

**Problem:** Text meant as chatbot correspondence gets pasted as content.

**Before:**
> Of course! The French Revolution began in 1789. I hope this helps! Let me know if you'd like me to expand on any section.

**After:**
> The French Revolution began in 1789.


### 21. Knowledge-Cutoff Disclaimers

**Words to watch:** as of [date], Up to my last training update, While specific details are limited/scarce..., based on available information...

**Problem:** AI disclaimers about incomplete information get left in text.

**Before:**
> While specific details about the company's founding are not extensively documented in readily available sources, it appears to have been established sometime in the 1990s.

**After:**
> The available sources do not document the exact founding date, but suggest the company was established in the 1990s.


### 22. Sycophantic/Servile Tone

**Problem:** Overly positive, people-pleasing language.

**Before:**
> Great question! You're absolutely right that this is a complex topic. That's an excellent point about the economic factors.

**After:**
> The economic factors you mentioned are relevant to this complex topic.


## FILLER AND HEDGING

### 23. Filler Phrases

**Before → After:**
- "In order to achieve this goal" → "To achieve this"
- "Due to the fact that it was raining" → "Because it was raining"
- "At this point in time" → "Now"
- "In the event that you need help" → "If you need help"
- "The system has the ability to process" → "The system can process"
- "It is important to note that the data shows" → "The data shows"


### 24. Excessive Hedging

**Problem:** Over-qualifying statements.

**Before:**
> It could potentially possibly be argued that the policy might have some effect on outcomes.

**After:**
> The policy may affect outcomes.


### 25. Generic Positive Conclusions

**Problem:** Vague upbeat endings.

**Before:**
> The future looks bright for the company. Exciting times lie ahead as they continue their journey toward excellence. This represents a major step in the right direction.

**After:**
> The passage presents this as a major step and expresses optimism about what comes next.


### 26. Hyphenated Word Pair Overuse

**Words to watch:** third-party, cross-functional, client-facing, data-driven, decision-making, well-known, high-quality, real-time, long-term, end-to-end

**Problem:** AI may hyphenate common word pairs with mechanical consistency, but correct compound modifiers are not errors. Keep hyphens when grammar or established usage calls for them. Predicate phrases are often open compounds.

**Before:**
> The cross-functional team delivered a high-quality, data-driven report. The team is cross-functional, the report is high-quality, and the methodology is data-driven.

**After:**
> The cross-functional team delivered a high-quality, data-driven report. The team is cross functional, the report is high quality, and the methodology is data driven.


### 27. Persuasive Authority Tropes

**Phrases to watch:** The real question is, at its core, in reality, what really matters, fundamentally, the deeper issue, the heart of the matter

**Problem:** LLMs use these phrases to pretend they are cutting through noise to some deeper truth, when the sentence that follows usually just restates an ordinary point with extra ceremony.

**Before:**
> The real question is whether teams can adapt. At its core, what really matters is organizational readiness.

**After:**
> The question is whether teams can adapt. That depends on organizational readiness.


### 28. Signposting and Announcements

**Phrases to watch:** Let's dive in, let's explore, let's break this down, here's what you need to know, now let's look at, without further ado

**Problem:** LLMs announce what they are about to do instead of doing it. This meta-commentary slows the writing down and gives it a tutorial-script feel.

**Before:**
> Let's dive into how caching works in Next.js. Here's what you need to know: Next.js caches data at request, data, and router layers.

**After:**
> Next.js caches data at request, data, and router layers.


### 29. Fragmented Headers

**Signs to watch:** A heading followed by a one-line paragraph that simply restates the heading before the real content begins.

**Problem:** LLMs often add a generic sentence after a heading as a rhetorical warm-up. It usually adds nothing and makes the prose feel padded.

**Before:**
> ## Performance
>
> Speed matters.
>
> When users hit a slow page, they leave.

**After:**
> ## Performance
>
> When users hit a slow page, they leave.


## STYLE, RHYTHM, AND RHETORIC PATTERNS

### 30. Forced Metaphors and Figurative Overwriting

**Signs to watch:** original but strained metaphors, mixed metaphors, figurative substitutions where a plain word is clearer, a metaphor that gets explained right after it is used

**Problem:** Beyond the stock figurative words flagged in patterns 4 and 7, LLMs invent decorative metaphors that add imagery without adding meaning, then often explain them. Plain description is usually clearer and more honest. If the metaphor does not earn its place, cut it and say the literal thing.

**Before:**
> The codebase is a garden we must tend, pruning dead branches and planting seeds of innovation so the whole ecosystem can flourish. In other words, delete unused code and add features.

**After:**
> Delete unused code and add new features.


### 31. Dramatic Fragmentation and Punchy Kickers

**Signs to watch:** two- or three-word subjectless sentences used for drama, staccato "X. And Y. And Z." runs, a short quotable line ending every paragraph or section, cutesy appositive fragments ("the catalog, honestly priced")

**Problem:** LLMs chop sentences into fragments for false emphasis and end sections with a quotable "mic-drop" line. It reads like ad copy or a motivational poster. If a line sounds like it belongs on a poster, cut it or fold it back into a real sentence with a subject. This is distinct from pattern 13 (which is about grammatical passive voice); here the tell is rhythm and showmanship, not a hidden actor.

**Before:**
> The catalog, honestly priced. Pay for what it does. Not promises. It just works. Every time.

**After:**
> The catalog is honestly priced: you pay for what it does rather than promises, and it works every time.


### 32. Rhetorical Questions Answered Immediately

**Signs to watch:** "What if...?", "The question is...", "Ever wondered...?", a question immediately followed by its own answer, "Think about it."

**Problem:** LLMs pose a question only to answer it a beat later. The question adds no information and stalls the sentence. State the point directly.

**Before:**
> What makes an API good? It comes down to predictability. Think about it: developers want to know exactly what they will get back.

**After:**
> A good API is predictable, so developers know exactly what they will get back.


### 33. Sentence-Opener Tics

**Words to watch:** So..., Look,, habitual sentence-initial And/But, "I think"/"I believe" when stating a fact, adverb openers (Interestingly, Importantly, Notably, Crucially, Essentially, Ultimately)

**Problem:** LLMs lean on a small set of openers. Adverb openers tell the reader how to feel instead of earning it, and "So" or "Look" fake conversational warmth. Drop the opener and start with the substance.

**Before:**
> So, the results were mixed. Interestingly, adoption went up. Importantly, churn went up too. I think that means the feature still needs work.

**After:**
> The results were mixed: adoption rose, but churn rose too. I think the feature still needs work.


### 34. Reassurance Kickers

**Signs to watch:** And that's okay., And that's fine., There's nothing wrong with that., no shame in..., you're not alone, it's completely normal

**Problem:** LLMs tack on reassurance the reader never asked for. It softens the writing and assumes the reader needs comforting. Trust the reader: make the point and stop.

**Before:**
> You might not have a testing setup yet. And that's okay. Plenty of teams start without one, and there's nothing wrong with that.

**After:**
> Many teams start without a testing setup, and that is not inherently a problem.

---

## Check for false positives

A person may use some of these patterns. Do not treat any single feature as proof of AI writing or as a reason to edit otherwise effective prose.

- Curly quotation marks alone are not a problem. Word processors and publishing systems often add them automatically.
- An em dash, one short sentence, or one formal transition is not enough. Look for repeated, formulaic use in context.
- Keep deliberate repetition, useful disclaimers, real alternatives, quotations, titles, proper names, code, URLs, and technical literals.
- Keep passive voice when the actor is unknown, irrelevant, or deliberately withheld.
- Keep correct compound modifiers. Mechanical consistency may be worth reviewing, but grammar takes priority.
- Preserve unusual details, unresolved tension, asides, self-corrections, and other choices that carry the writer's voice.

When unsure, look for several patterns together or a clear editing problem. A valid outcome is to leave the text unchanged.

## Output modes

- **Pasted text (default).** Return the final rewrite first. Include a brief audit or change summary only when the user asks for it or it materially helps.
- **File mode.** Use `read_file`, then `patch` or `write_file`, but write only the final prose. Preserve code blocks, frontmatter, data, citations, and link targets. Return a short summary or diff.
- **Embedded mode.** When another task uses Humanizer as an editing pass, return only the final text to the caller.

## Rewrite process

1. Read the source and note its claims, qualifiers, citations, technical literals, and deliberate voice choices.
2. Identify repeated patterns that create a real writing problem. Leave isolated or intentional features alone.
3. Rewrite the smallest sections that need work. Read the candidate aloud and check its rhythm and tone.
4. Ask: "Did the rewrite add or remove any fact, name, number, date, quote, citation, ranking, qualifier, or other claim?"
5. Fix every unsupported addition and every lost or distorted substantive claim. Then run the final anti-AI pass.
6. Return or write the result required by the selected output mode. Do not expose an intermediate draft in file or embedded mode.


## Attribution

This skill is ported from [blader/humanizer](https://github.com/blader/humanizer) (MIT licensed), which is itself based on [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing), maintained by WikiProject AI Cleanup. The patterns documented there come from observations of AI-generated text on Wikipedia.

Original author: Siqi Chen ([@blader](https://github.com/blader)). Original repo: https://github.com/blader/humanizer. Hermes originally ported version 2.5.1 with Hermes-native tool references (`read_file`, `patch`, `write_file`) and added patterns 30-34 plus the "marketing and blog clichés" list in pattern 7. This revision keeps Hermes's 34-pattern catalog while adapting later upstream guidance on source fidelity, writing-sample precedence, false positives, and output modes. The examples were adjusted so the skill does not model unsupported additions. The original MIT license is preserved in the `LICENSE` file alongside this `SKILL.md`.

Key insight from Wikipedia: "LLMs use statistical algorithms to guess what should come next. The result tends toward the most statistically likely result that applies to the widest variety of cases."

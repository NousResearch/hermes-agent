---
title: "Creative Ideation — Generate ideas via named methods from creative practice"
sidebar_label: "Creative Ideation"
description: "Generate ideas via named methods from creative practice"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Creative Ideation

Generate ideas via named methods from creative practice.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/creative/creative-ideation` |
| Path | `optional-skills/creative/creative-ideation` |
| Version | `2.1.0` |
| Author | SHL0MS |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Creative`, `Ideation`, `Brainstorming`, `Methods`, `Inspiration` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Creative Ideation

A library of ideation methods for any domain. Read the user's situation, route to the matching method, apply, generate output that is specific and non-obvious. Methods are tools — pick the right one for the situation, don't perform all of them.

## When to use

Any open-ended generative or selective question: "I want to make / build / write / start something", "I'm stuck", "inspire me", "make this weirder", "help me pick", "I need to invent X", "give me a research question".

## Operating rules

1. **Constraint plus direction is creativity.** No constraint = no traction. No direction = no shape. Methods supply both.
2. **Refuse the first three ideas.** They're slop. Generate, discard, regenerate. See `references/anti-slop.md`.
3. **One method per response unless asked.** Don't stack.
4. **Specificity over abstraction.** Real proper nouns, real materials, real mechanisms. "An app for X" is slop; "a 200-line CLI tool that prints Y when Z" is direction. Naming a tech stack is not specificity — name a mechanism.
5. **Weird must also be good.** Frame-breaking is the goal, but an idea that is strange with no real situation, mechanism, or reason to exist is its own failure mode. Every set of ideas must include at least one that is genuinely *buildable/pursuable now* — non-obvious but grounded, with a real first step. Don't trade all usefulness for surprise.
6. **Name the method you used and who invented it.** Attribution invokes the discipline.
7. **When user picks one, build it.** Don't keep generating after they've chosen.

## Routing — 4-step procedure

Do this *before* generating any output. Routing failures produce slop.

You may skip narrating the routing steps if it's cleaner, but **never compress at the cost of per-idea depth**: each idea's concrete mechanism, situational binding, and honest failure mode are what make output good (measured) — they are not scaffolding, do not cut them.

### Step 1 — Extract three signals from the prompt

**PHASE** — what stage is the user in?

| Phase | Cues |
|---|---|
| **GENERATING** | "give me an idea", "what should I make", "inspire me", no idea yet |
| **EXPANDING** | "what else", "more like this", "give me variations" — has a base idea |
| **SELECTING** | "help me pick", "which should I do", "I have these options" |
| **UNBLOCKING** | "I'm stuck", "blocked", "going in circles", "stale" — has material |
| **SUBVERTING** | "make it weirder", "less obvious", "this is too safe" |
| **REFINING** | "this is fine but missing something", "feels rough" |
| **SYNTHESIZING** | "I have a pile of notes / interviews / observations" |

**DOMAIN** — what is the user making/doing?

| Domain | Cues |
|---|---|
| **TEXT** | fiction, essay, poem, lyric, script, copy |
| **OBJECT** | visual art, music, sound, performance, installation, sculpture |
| **ARTIFACT** | software, hardware, mechanism, device |
| **SYSTEM** | org, civic, institution, ecology, community |
| **SELF** | life decision, career, personal practice |
| **RESEARCH** | paper, thesis, scholarly question |
| **PRODUCT** | business, market, service |

**SPECIFICITY** — how much constraint is in the prompt?

| Level | Cues |
|---|---|
| **NONE** | "I'm bored", "inspire me" — no domain, no project |
| **DOMAIN** | "I want to write something" — knows the field, no project |
| **PROJECT** | "I'm working on this specific X" |
| **PROBLEM** | "I have this specific friction within X" |

### Step 2 — Apply overrides (highest priority, fire first)

Override rules beat the routing table:

- **Mood signal** — user says "weird", "strange", "surprising", "less obvious", "more interesting" → `references/methods/lateral-provocations.md` or `references/methods/pataphysics.md`, regardless of domain.
- **User names a method** — use it.
- **User asks for a method recommendation** ("which method") → surface 2–3 candidates with one-line each, ask which to apply. Don't silently default.
- **High-slop terrain** — "AI ideas", "startup ideas", "habit tracker", "productivity / wellness / fitness / food / travel app" → force `references/methods/lateral-provocations.md` or `references/methods/pataphysics.md` over the obvious method. Refuse the first **5** ideas, not 3.

### Step 3 — Route by phase first, then domain

**By phase (applies regardless of domain):**

| Phase | Default route |
|---|---|
| GENERATING + SPECIFICITY=NONE | `references/full-prompt-library.md` **General** section (constraint dispatch) |
| GENERATING + DOMAIN known | route by domain (next table) |
| EXPANDING | `references/methods/scamper.md` |
| SELECTING | `references/methods/premortem-and-inversion.md` (or `references/methods/compression-progress.md` for upside) |
| UNBLOCKING | `references/methods/oblique-strategies.md` |
| SUBVERTING | `references/methods/lateral-provocations.md` (fallback `references/methods/pataphysics.md`) |
| REFINING (text) | `references/methods/defamiliarization.md` |
| REFINING (other) | `references/methods/creative-discipline.md` (Tharp's spine) |
| SYNTHESIZING | `references/methods/affinity-diagrams.md` |
| Volume needed fast | `references/methods/volume-generation.md` |

**By domain (when GENERATING with DOMAIN known):**

| Domain | Default route |
|---|---|
| TEXT — formal / poetry | `references/methods/oulipo.md` |
| TEXT — narrative | `references/methods/story-skeletons.md` |
| TEXT — has source material to remix | `references/methods/chance-and-remix.md` |
| OBJECT (music, visual, performance) | `references/methods/oblique-strategies.md` |
| OBJECT — physical maker / wants a starting constraint | `references/full-prompt-library.md` **Physical / object** section |
| ARTIFACT — wants a starting constraint | `references/full-prompt-library.md` **Software / artifact** section |
| ARTIFACT — engineering invention with parameter conflict | `references/methods/triz-principles.md` |
| ARTIFACT — software architecture | `references/methods/pattern-languages.md` |
| ARTIFACT — has natural-system analog | `references/methods/biomimicry.md` |
| ARTIFACT — accumulated assumptions to question | `references/methods/first-principles.md` |
| SYSTEM (civic, org, institutional) | `references/methods/leverage-points.md` |
| SYSTEM — collective / participatory | `references/full-prompt-library.md` **Social / collective** section |
| SELF (life, career, what-to-study) | `references/methods/derive-and-mapping.md` |
| RESEARCH — picking a question | `references/methods/compression-progress.md` |
| RESEARCH — attacking a known problem | `references/methods/polya.md` |
| PRODUCT (business, service) | `references/methods/jobs-to-be-done.md` |
| Need to break a frame / find analogy | `references/methods/analogy-and-blending.md` |

### Step 4 — Handle ambiguity and contradiction

- **Multiple paths plausible** → pick the one closest to the user's actual phrasing. Don't pick the most interesting method to seem sophisticated.
- **Genuinely ambiguous** → ask ONE clarifying question, don't silently guess. Examples: *"Are you generating ideas or picking between ones you have?"* / *"Is this for fiction, essay, or something else?"*
- **Signals contradict** (e.g., "weird startup ideas" → product domain + weird mood) → **stack two methods explicitly**. State what you're doing: *"Using `jobs-to-be-done` for the product framing + `lateral-provocations` to break the obvious shape."*
- **No match** → constraint dispatch (`references/full-prompt-library.md`) is the safe fallback.
- **Same question asked again** → switch methods. Variation in method = variation in idea distribution.

### Anti-default check (run before generating)

- About to write "Here are 5 ideas:" or a bare numbered list? → STOP. Pick a method first.
- About to default to generic LLM-mode brainstorming? → STOP. Pick a path above.
- Output looks like what an unrouted LLM would produce? → routing failed, redo.

The default LLM mode is exactly what this skill exists to displace. If you generate without routing, you've defeated the skill.

For deeper edge cases (mood signals, stacking, anti-patterns) see `references/heuristics.md`.

## Output format

For the constraint-dispatch default path:

```
## Constraint: [Name] — from [Source]
> [The constraint, one sentence]

### Ideas

1. **[One-line pitch]**
   [2-3 sentences — what specifically is made, why it's interesting]
   ⏱ [weekend/week/month]  •  🔧 [stack/medium/materials]

2. ...
3. ...
```

For other methods, use the format the method specifies (TRIZ produces a contradiction analysis; OuLiPo produces constrained text; Oblique Strategies produces a single applied card → next move). Don't force every method into the constraint template.

**Every idea set, regardless of method:**
- Name the method used. On slop terrain, name the obvious ideas you refused.
- Give each idea its concrete mechanism and its honest failure mode / tradeoff / who-it's-for. This depth is what makes ideas land — measured, not decorative.
- Mark at least one idea as the **grounded** one — buildable/pursuable now, non-obvious but with a real first step. The others can run further toward the strange; this one has to be genuinely doable. Don't let the whole set be weird-but-impractical.

## File map

- `references/full-prompt-library.md` — constraint library, sectioned by domain (General, Software, Physical, Social, Lists). Default path for SPECIFICITY=NONE.
- `references/method-catalog.md` — one-line summary + when-to-use per method
- `references/heuristics.md` — extended decision tree for edge cases
- `references/anti-slop.md` — anti-slop rules; apply to every output
- `references/exercises.md` — time-boxed exercises (5min / 30min / 1hr / day / week)
- `references/methods/` — 22 named methods, one file each, load only the one you're using

## Attribution

Constraint-dispatch core adapted from [wttdotm.com/prompts.html](https://wttdotm.com/prompts.html). Methods drawn from primary sources cited in each method file.

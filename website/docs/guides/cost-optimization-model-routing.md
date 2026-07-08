---
sidebar_position: 35
title: "Cost Optimization with Model Routing"
description: "Dramatically reduce inference costs by routing cheap models for drafting and expensive models for verification — the dual-model loop that saved GenTech 92% on monthly inference."
---

# Cost Optimization with Model Routing

Most Hermes users pick one model and stick with it. That's fine for getting started, but if you're running automated workflows, cron jobs, or development loops, you're almost certainly overpaying.

The pattern: **use a cheap, fast model for the heavy lifting, then route only critical steps to an expensive model for verification.** It's the same principle as unit tests on a fast runner and integration tests on a staging environment — catch the cheap stuff cheap, validate the important stuff thorough.

## The Dual-Model Loop

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Draft      │ ──→ │  Iterate     │ ──→ │  Audit      │
│  (cheap)    │     │  (cheap)     │     │  (expensive) │
│  $0.15/hr   │     │  $0.15/hr    │     │  $0.50/hr   │
└─────────────┘     └──────────────┘     └─────────────┘
                                                  │
                                          ┌───────┘
                                          ▼
                                   ┌─────────────┐
                                   │  Ship / Fix  │
                                   │  (human ok)  │
                                   └─────────────┘
```

The loop:
1. **Draft** on a cheap model (DeepSeek V3, GLM-4.7, Gemini 2.0 Flash)
2. **Iterate** on the same cheap model — refine, expand, test
3. **Audit** on an expensive model (Claude Sonnet, GPT-4o, GLM-5.2) — review, verify, catch edge cases
4. **Ship** or loop back to step 2

### Real Numbers from GenTech

GenTech runs a dual-agent setup (VPS 24/7 + desktop for local builds) with heavy cron workloads:

| Metric | Single Model (GPT-4o) | Dual-Model Loop | Savings |
|--------|----------------------|-----------------|---------|
| Monthly inference | ~$480 | ~$37 | **92%** |
| Dev loop (full day) | ~$16 | ~$1.25 | **92%** |
| PR audit (per review) | ~$3 | ~$0.25 | **91%** |
| Quality (user-reported) | Baseline | Equivalent | — |

The audit pass catches the same issues it would on a single-expensive-model workflow — because the expensive model is still making the critical judgment calls. The cheap model just does the volume work.

## Configuration

### Per-Task Model Routing

In your `config.yaml`, define named model tiers:

```yaml
models:
  drafting:
    provider: deepseek
    model: deepseek-chat
    temperature: 0.7

  auditing:
    provider: anthropic
    model: claude-sonnet-4
    temperature: 0.3
```

Then use them in your prompts:

```
Draft a Solana token transfer script on the drafting model.
When done, audit the result on the auditing model.
```

### Cron Job Model Pinning

Cron jobs should pin their model explicitly so they don't drift when you switch your interactive model:

```bash
hermes cron create "0 9 * * *" \
  --prompt "Generate weekly market summary. Draft on drafting, audit on auditing." \
  --model-provider deepseek \
  --model deepseek-chat
```

### Pipeline with Context Chaining

Chain cron jobs so a cheap model collects data and an expensive model only processes it when there's something to analyze:

```yaml
# Job 1: Cheap data collection (script-only, no LLM)
job-1:
  schedule: "*/30 9-17 * * *"
  script: "fetch-market-data.py"
  no_agent: true

# Job 2: Expensive analysis (only when job 1 has output)
job-2:
  schedule: "0 18 * * *"
  prompt: "Analyze today's market data from context..."
  context_from: ["job-1"]
  model:
    provider: anthropic
    model: claude-sonnet-4
```

## When to Use This Pattern

### ✅ Great for:
- **Development loops** — write code on fast models, audit on strong models
- **Content generation** — draft posts/emails on cheap models, polish on premium
- **Research synthesis** — gather and summarize cheaply, verify conclusions expensively
- **Cron pipelines** — data collection on scripts, analysis on LLMs, audit on strong LLMs
- **PR reviews** — first pass on cheap model, second pass on strong model
- **Batch processing** — process 100 items on cheap, route edge cases to expensive

### ❌ Skip when:
- **Single-shot tasks** — one-off questions don't benefit from multi-pass routing
- **Real-time conversation** — latency matters more than cost
- **The cheap model is good enough alone** — if your task doesn't need a stronger model,
  don't add one
- **Context window is the bottleneck** — some expensive models have larger contexts

## Advanced: Three-Tier Pipeline

For complex workflows, use three tiers:

```yaml
tiers:
  scout:     # Ultra-cheap, high throughput
    model: gemini-2.0-flash
    provider: google
  worker:    # Mid-range, balanced
    model: deepseek-chat
    provider: deepseek  
  reviewer:  # Premium, judgment calls
    model: claude-sonnet-4
    provider: anthropic
```

Application: a code review pipeline

1. **Scout** detects changed files and potential issues (fast scan)
2. **Worker** analyzes each file in detail and drafts inline comments
3. **Reviewer** reads the full diff + worker output, makes final call, approves or requests changes

Each tier costs more per-call but runs on fewer items. The reviewer might see 1 file for every 20 the scout scanned — so 95% of the work runs at scout pricing.

---

:::tip Start simple
Begin with the two-tier loop (draft + audit) on one workflow before scaling to pipelines. The savings are immediate and the setup is minimal.
:::

## Related

- [Model Configuration Reference](/reference/model-configuration)
- [Cron Jobs Guide](/guides/automate-with-cron)
- [Delegation Patterns](/guides/delegation-patterns)

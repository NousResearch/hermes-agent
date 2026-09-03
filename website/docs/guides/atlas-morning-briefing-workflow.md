---
sidebar_position: 4
title: "Atlas Morning Briefing Workflow"
description: "How the morning brief is pulled, what sources it uses, and how the Mac side should reproduce it"
---

# Atlas Morning Briefing Workflow

This page is the handoff for the Mac side.

## What the morning brief is

The morning brief is a *self-contained premarket workflow* that answers:

- What is in play today?
- Why is it in play?
- Is it tradable?
- What can price do today?

It is *not* just a generic news summary. It is a catalyst-first briefing built from repeatable sources, then filtered for liquidity, spreads, and price action.

## Core rules

- Run it before the US market opens on weekdays.
- Check for market holidays and early closes first.
- If the market is closed, send only a short closed-market note.
- Use public, accessible sources when a paid source is unavailable.
- Never invent catalysts, prices, or spreads.
- Keep the output Telegram-friendly:
  - bold labels
  - bullets
  - no tables

## Timing

Default target:

- **8:00 AM America/Chicago**
- **9:00 AM ET**
- weekdays only

If the schedule is late or the open is near, prioritize ticker extraction and chart review over long-form reading.

## Source flow

Pull sources in this order:

1. **Market status / holiday check**
   - Verify whether NYSE/Nasdaq are open.
   - If closed, stop after the closure note.

2. **Macro context**
   - High-impact USD events
   - Fed speakers / FOMC items
   - White House / economy remarks
   - SPY / QQQ / IWM / VIX context

3. **Catalyst sweep**
   - Seeking Alpha **Catalyst Watch** for the weekly map
   - Seeking Alpha **Stocks to Watch** for the day-of movers
   - CNBC **Market Insider** / biggest moves before the bell
   - Unusual Whales earnings calendar sorted by option volume
   - Benzinga keywords: upgrades, downgrades, AI
   - MarketWatch premarket screener
   - Market Chameleon company events
   - StockTwits trending as discovery only

4. **Ticker filtering**
   - Keep recognizable / liquid names
   - Reject wide spreads or untradable names
   - Count how many sources confirm each ticker
   - Treat trending alone as insufficient

## What to include in the final brief

For each ticker:

- **Ticker**
- **Catalyst**
- **Why it is in play**
- **Source confirmation count**
- **Earnings timing**
- **Spread quality**
- **Premarket move**
- **Price action / chart read**
- **Plan type**: watch / avoid / needs confirmation
- **Invalidation / no-trade condition**

Then end with:

- **Top 3 priorities**
- **Do-not-chase list**
- **One-line market read**

## Self-contained prompt for the Mac side

Use this exact structure when creating the cron job or running the brief manually:

```text
Create a JR-style catalyst morning briefing for Atlas.

First check whether the US market is open, closed, or on holiday/early close.
If the market is closed, send only a short closure note with the reason.

If the market is open, build a premarket briefing using:
- ForexFactory or equivalent macro calendar for high-impact USD events
- SPY, QQQ, IWM, and VIX context
- Seeking Alpha Catalyst Watch
- Seeking Alpha Stocks to Watch
- CNBC Market Insider / biggest moves before the bell
- Unusual Whales earnings calendar sorted by option volume
- Benzinga upgrades, downgrades, and AI headlines
- MarketWatch premarket screener
- Market Chameleon company events
- StockTwits trending only as discovery

For each ticker, include:
- ticker
- catalyst
- why it is in play
- source confirmation count
- earnings timing if relevant
- spread quality
- premarket move
- price action read
- watch / avoid / needs confirmation
- invalidation or no-trade condition

Be concise, use bold labels and bullets, no tables, and never invent data if a source is inaccessible.
```

## Practical note for Mac sync

If the Mac side needs to mirror the same workflow, it only needs to know:

- what sources to pull
- when to pull them
- how to stop on closed-market days
- how to format the output
- how to avoid hallucinating missing details

That is the whole handoff.

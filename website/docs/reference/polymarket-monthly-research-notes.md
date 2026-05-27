---
sidebar_position: 7
title: "Polymarket Monthly Research Notes"
description: "Execution-aware notes and hypotheses for monthly BTC / Polymarket research"
---

# Polymarket Monthly Research Notes

These are research notes from practical discussion around PM monthly markets and adjacent execution questions.

They are **not** part of the `onchaindivers-w3e` skill contract.
They belong in docs / knowledge base because they are strategy and microstructure notes, not instructions for how to use W3E itself.

## Status

Treat everything here as:

- a working hypothesis,
- a research constraint,
- or an execution caveat.

Do **not** treat these notes as validated alpha without fill-level evidence.

## Core idea

One possible direction is to monitor many BTC monthly / expiry / time-window markets instead of reasoning from one isolated market.

Why this matters:

- more events may create more placement opportunities,
- strategy viability may depend on breadth, not one-off setups,
- opportunity count and execution quality have to be measured together.

## Main execution lesson

The biggest warning from the discussion is simple:

**paper edge is not real edge unless the fill survives execution.**

Example framing:

- intended entry: `0.70`
- realized fill: `0.73`
- nominal edge: about `5%`
- realized edge after bad fill: maybe only `1–2%`

So the correct object to model is not just:

- market probability,
- or top-of-book quote,

but instead:

- actual reachable fill,
- blended fill after partial execution,
- fees,
- and inventory left resting in the book.

## Practical execution hypotheses

### Marketable limit orders may dominate pure market orders

For these setups, speed is often less important than preserving price.

A useful pattern to test:

1. place a **marketable limit** at the worst acceptable entry,
2. let immediately available size fill,
3. leave the remainder resting,
4. monitor whether the alpha still exists,
5. cancel or unwind if the thesis degrades.

This is especially important around balanced levels like `0.45–0.55`, where a few cents of slippage can erase much of the expected edge.

### Partial-maker execution may improve realized economics

A marketable limit can outperform a blunt market order because:

- some size may execute immediately,
- some size may rest as maker,
- effective fees may improve,
- blended entry may stay closer to the target threshold.

## Data-quality warning

If research depends on book updates or event-stream reconstruction, assume duplicates are possible.

Potential duplicate dimensions:

- event hash / update key,
- server timestamp,
- repeated book snapshots carrying the same economic state.

Deduplication is part of the research problem, not optional cleanup.

## Structure ideas that still need validation

One strategy framing mentioned in discussion:

- shape straddle / vol exposure so that losses are concentrated in a narrow central band,
- then rebalance the rest.

This is not enough on its own.
It must be checked against:

- real PM fills,
- available size,
- duplicate-book noise,
- real execution latency,
- and blended post-fee entry.

If the analysis cannot show realized fills, the structure is still unproven.

## What to measure in research notebooks

Keep these fields separate:

- theoretical edge from model / probability mismatch,
- visible top-of-book price,
- actual reachable fill,
- blended realized fill,
- fees,
- remaining unfilled inventory,
- post-entry thesis drift.

## Generic dedup query pattern

```sql
WITH grouped AS (
  SELECT
    <event_hash_or_key> AS k,
    <server_timestamp_column> AS ts,
    count() AS n
  FROM <book_or_event_table>
  GROUP BY 1, 2
)
SELECT *
FROM grouped
WHERE n > 1
ORDER BY n DESC
LIMIT 50
```

Adapt placeholders to the actual schema.

## Bottom line

The main takeaway is:

**for PM monthly research, execution-aware fill data is the unit of truth.**

Without real fill analysis, apparent edge may just be microstructure illusion.

---
title: "Travel Designer — Plan trips around traveler style and current facts"
sidebar_label: "Travel Designer"
description: "Plan trips around traveler style and current facts"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Travel Designer

Plan trips around traveler style and current facts.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/productivity/travel-designer` |
| Path | `optional-skills/productivity/travel-designer` |
| Version | `0.1.0` |
| Author | Nico Allen (Nico-AK), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Travel`, `Itinerary`, `Budget`, `Research`, `Planning` |
| Related skills | [`maps`](../../bundled/productivity/productivity-maps.md), [`product-price-monitor`](../../bundled/productivity/productivity-product-price-monitor.md), [`grounded-citations`](../../bundled/research/research-grounded-citations.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Travel Designer Skill

Build trip recommendations, budgets, itineraries, packing lists, and on-the-ground pivots around how the traveler actually likes to travel. This skill does not replace live booking systems or the traveler's final judgment; it turns stated preferences, constraints, current facts, and past trip lessons into practical travel plans.

## When to Use

- "Help me pick a destination."
- "Plan a trip to &lt;place> for &lt;dates>."
- "Build a budget for this trip."
- "What should I pack for &lt;destination>?"
- "We are here and the plan broke; what should we do instead?"
- The user asks for a trip plan that should use their saved traveler profile, travel lessons, or past travel patterns.

Don't use for: booking purchases without approval, visa/legal advice beyond cited official sources, medical clearance, or travel safety claims without current sourcing.

## Prerequisites

- Traveler inputs: traveler profile, travel lessons, trip party, dates or season, departure city, destination candidates or desired trip type, budget posture, and hard constraints.
- Current facts: use `web_search`, `web_extract`, `weather`, or relevant provider tools for live prices, opening hours, closures, weather, booking requirements, transit, and safety conditions. Do not rely on memory for facts that can change.
- Source discipline: load `grounded-citations` when the answer needs citations, direct links, or source-backed claims.
- Location work: load `maps` when comparing neighborhoods, drive times, transit feasibility, or route clusters.
- Price watching: load `product-price-monitor` when the user wants ongoing fare, lodging, ticket, or listing alerts.

## How to Run

1. Identify the operating mode: destination selection, trip planning, budget, packing, or on-the-ground recovery.
2. Retrieve or request the traveler's `TRAVELER PROFILE` and `TRAVEL LESSONS`; if either is missing, ask only the minimum questions needed for the active mode.
3. Verify current facts before recommending concrete places, venues, routes, prices, weather-sensitive activities, lodging areas, or reservations.
4. Produce the plan in the mode's output shape with clear tradeoffs, risks, and next actions.

## Quick Reference

- Destination choice: compare fit, budget, logistics, seasonality, and trip-killer risks.
- Trip plan: group days by neighborhood or area; keep the day pace within the traveler profile.
- Budget: separate accommodation, food, activities, transit, and contingency; honor spend-on versus never-pay-for rules.
- Packing: tailor to destination weather, activities, lodging style, transport mode, and group health/access needs.
- On the ground: offer immediate alternatives within walking or short transit distance; preserve the mood before optimizing the itinerary.

## Procedure

### 1. Load the personal travel lens

Summarize the traveler profile and travel lessons in 3-5 working assumptions before planning. Include budget posture, pace, lodging preference, food constraints, access needs, and trip killers. Done when the plan can name what it is optimizing for and what it must avoid.

### 2. Select the operating mode

Classify the request as one primary mode, then state if a secondary mode is needed. Do not blend modes so broadly that the answer becomes mushy. Done when the user can see whether you are choosing, planning, budgeting, packing, or recovering.

### 3. Verify current facts

Use current retrieval for destination conditions, venue hours, closure notices, booking windows, weather, transit, realistic travel times, and prices. Prefer official or primary sources for official rules and high-stakes constraints. Done when each concrete recommendation that could be stale has a recent source or is labeled as an assumption.

### 4. Apply profile fit hard

Score or describe each option against the traveler's budget style, pace, lodging preference, food style, mobility/access needs, and trip killers. If an option clearly violates the profile, say so plainly and recommend against it before offering a salvage path. Done when no option is presented as good merely because it is popular.

### 5. Build the mode-specific output

- **Pick a Destination:** compare each candidate with pros, cons, budget fit, best season, logistics from the user's departure point, and trip-killer risks. End with a ranked recommendation.
- **Plan the Trip:** group activities by neighborhood or area, limit planned activities to the profile pace, protect wandering time, and flag advance bookings or reservation deadlines early.
- **Budget:** estimate accommodation, food, activities, transit, fees, and contingency. Explicitly map spend categories to the traveler's spend-on and never-pay-for rules.
- **Pack:** create an itemized list filtered by weather, activities, lodging, transport, laundry access, medical/access needs, and group members.
- **On the Ground:** give 2-4 nearby alternatives ranked by fit, with travel time, cost posture, weather suitability, and why each protects the trip from the disruption.

Done when the output directly answers the active mode without forcing the user to do extra sorting.

### 6. Preserve presence and decision quality

Separate must-do anchors from maybes and skips. Move open research into pre-trip lists rather than leaving the user to research during the trip. Done when the plan reduces on-trip decisions instead of creating a homework packet.

## Pitfalls

- Planning a generic tourist checklist after loading a highly specific traveler profile.
- Optimizing for maximum attractions instead of the traveler's sustainable pace.
- Recommending status luxury, unnecessary hotel upgrades, or expensive convenience that contradicts the budget style.
- Ignoring campground seasonality, booking windows, closures, laundry/showers, parking, or city-return logistics on camper or road trips.
- Sending the traveler across town for adjacent activities that could have been grouped by area.
- Treating a remembered price, schedule, or opening hour as current.
- Making the user research the next thing while they are supposed to be enjoying the current thing.

## Verification

- [ ] Traveler profile and travel lessons were used or missing pieces were explicitly requested.
- [ ] Current facts were verified for recommendations that can change.
- [ ] Activities are clustered by area and limited to the preferred pace.
- [ ] Spend recommendations match spend-on and never-pay-for rules.
- [ ] Advance bookings, reservation risks, closures, and seasonal constraints are flagged.
- [ ] Trip-killer risks are called out plainly, with a recommendation against poor-fit options when needed.
- [ ] The final plan names assumptions and immediate next actions.

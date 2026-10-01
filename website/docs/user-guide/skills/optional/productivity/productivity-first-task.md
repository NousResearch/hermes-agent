---
title: "First Task — Run the first task chat that setup hands off"
sidebar_label: "First Task"
description: "Run the first task chat that setup hands off"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# First Task

Run the first task chat that setup hands off.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/productivity/first-task` |
| Path | `optional-skills/productivity/first-task` |
| Version | `0.1.0` |
| Author | Siddharth Balyan (alt-glitch) + Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `onboarding`, `first-run`, `desktop`, `handoff` |
| Related skills | [`initiate-setup`](../../optional/productivity/productivity-initiate-setup.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# First Task Skill

Runs the first chat after setup, the one `start_chat` opened. The top of this message is the user's ask, then "What setup learned about me" (their picks and the scan), then these rules and a JSON block. The goal: they see work within one minute and hold a finished, useful result within five.

## When to Use

- This message carries the skill under a handoff from setup.
- Until the first result lands. After that it is a normal chat.

## Prerequisites

The task chat's own tools: `manage_connections`, `manage_catalog`, `clarify`, `terminal`, the file tools, the browser, `tool_search`, `skill_view`, and `desktop_preview` in the desktop app. Use only tools that are in your tool list. A tool named here but missing from your list, often `manage_catalog`, is deferred: run it through `tool_call`, as in the Quick Reference, never by its bare name.

## How to Run

The JSON block at the end holds `connect`, the connector ids of the apps they picked in setup, and `install`, the catalog plugin ids they picked or their first task brings. The ids are exact: skip search and status checks.

The "What setup learned" lines are your recon. Do not survey the machine first: no system scan, no update search, no inventory of installed tools. Look at one thing only when the next step needs it.

## Quick Reference

| Step | What | Call |
|---|---|---|
| 1 | One line on what you start with | none |
| 2 | Connect the picked apps | `manage_connections` connect, every id in `connect`, once |
| 3 | Install the picked plugins | `manage_catalog` install (through `tool_call` when deferred), every id in `install`, once |
| 4 | Specific ask: start it. Vague ask: three options | `clarify`, three choices |
| 5 | The first slice, finished in five minutes | the task's own tools |
| 6 | Show the result, offer the next step | `clarify`: Looks right, Change something, Take it further |

```
manage_connections  {"action":"connect","connectors":["<id>", ...]}
manage_catalog      {"action":"install","items":[{"kind":"plugin","id":"<id>"}, ...],"reason":"<one line>"}
tool_call           {"calls":[{"name":"manage_catalog","arguments":{"action":"install","items":[{"kind":"plugin","id":"<id>"}],"reason":"<one line>"}}]}   (when manage_catalog is deferred)
clarify             {"questions":[{"question":"<short question>","choices":["<option>","<option>","<option>"]}]}
```

## Procedure

### 1. Connect first

Your first reply is one short line on what you will start with, then steps 2 and 3 back to back, before any other work. Skip a step whose list is empty. A skipped or failed row never blocks: say in one line what it would have added, go on without it, and never offer it again in this chat.

Each step runs once. When a later step fails, fix that step and go on; never go back and redo an earlier one. The connect card's answer is final for this chat: use the apps that connected, and do not connect or ask about the others again (an app left unconnected after Continue counts as skipped). Say once, when you close the first slice, that they can connect the rest later.

### 2. Ask only when vague

- A specific ask (a named outcome such as "A daily brief from Linear and Slack", "Install a few apps for this Spark", a scene in Blender): start it now, with no confirming question.
- A vague ask ("I have something in mind", "Let's figure out a first task together", "help with my work"): one `clarify` card with three options, then start the pick. Text they type instead is the pick.

Options are outcomes of a few words, each finishable in five minutes, built from their picks and the scan. Name only apps they picked or the scan saw in use.

- Work apps picked (Linear, Slack, Gmail, Calendar): "A daily brief from Linear and Slack", "A summary of my week", "Learn how I work from my tools".
- An NVIDIA or Spark machine: "Install a few apps for this Spark", "Set up a local model".
- A plugin picked: one small thing in that app, such as "A simple scene in Blender".
- Nothing picked: "A small HTML page about &lt;something from the scan>", a quick useful script.

### 3. Time box

- Within one minute they see work: a file being written, a page opening, a first install.
- Within five minutes they hold a finished result: a page open in the preview, a brief in the chat, apps installed.
- A big ask gets cut to a first slice. Say so in one line ("Setting up the whole Spark is big; first I'll install a few apps you'll want."), finish that slice, then offer the next one.
- A step that runs long (a large download, an update search, a full build) is never the first slice: offer it as the next step.
- No written plan before work, except the machine interview below.

### 4. Machine setup is an interview

For "Help me set up this &lt;machine>" or "Install a few apps":

1. Two or three short `clarify` cards, one at a time, two to four choices each: "These apps?" (three to five everyday apps that fit their use and the apps they named), "Install the NVIDIA tools?" (NVIDIA machines only), "A local model?" (a Spark or a strong GPU only).
2. Then the smallest useful part: install the apps they chose with the official package manager, one line per install.
3. On Arm, check each install has a native arm64 build and say when only an x64 one exists.
4. Anything that needs a password, a licence or a payment goes on a short list for them. Never disable security settings or overwrite config without asking.

Finish with what changed and one offer for the next slice (drivers, a local model, developer tools).

### 5. Build rules

- Real data only: from connected apps (find their tools with `tool_search`) or tools already signed in on this computer, such as a logged-in `gh` (say so in one line). Never mock or sample data. Never route around a connector: no IMAP client, app password or scraping into the same account.
- Ask before sending, deleting or scheduling anything. Set up no recurring job unless they asked.
- A generated page is one self-contained HTML file, opened with `desktop_preview`.
- A plugin's tools: find them with `tool_search`, and read its skill with `skill_view` by its exact name. If its app is not running, say so plainly.
- When "What setup learned" says they are new to AI agent apps: explain a feature in one plain sentence when it first matters, with no jargon. The first time you act on the computer, say once that you ask for permission as you go and they can say no.

### 6. Close the first slice

One short line on what you made and where it is, then a `clarify` card with "Looks right", "Change something" and "Take it further". Act on the pick.

## Pitfalls

- A survey of the machine before any work. The facts block is the survey.
- A plan question when the ask was already specific.
- A 30-minute job as the first slice.
- Offering again an app they skipped or left unconnected.
- Running connect again after a later step fails.
- Calling a deferred tool by its bare name instead of through `tool_call`.
- Reading the facts block back to them.
- Stiff words: write short, plain, warm sentences, with no filler, em dashes or exclamation marks.

## Verification

- The first reply holds one line of text, then the connect and install calls.
- Visible work starts within one minute and the first result lands within five.
- A vague ask got one three-option card; a specific ask got none.
- The first slice ends with the Looks right, Change something, Take it further card.

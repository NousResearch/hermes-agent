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
| Version | `0.3.0` |
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

Runs the first chat after setup. This message holds the user's ask, then "What setup learned about me" (their picks and the machine basics), then these rules and a JSON block. Goal: visible work within one minute, a finished, useful result within five.

## When to Use

- This message carries the skill under a handoff from setup.
- Until the first result lands. After that it is a normal chat.

## Prerequisites

Tools: `manage_connections`, `manage_catalog`, `clarify`, `terminal`, the file tools, the browser, `tool_search`, `skill_view`, and `desktop_preview` in the desktop app. A tool named here but missing from your tool list, often `manage_catalog`, is deferred: run it through `tool_call`, never by its bare name.

## How to Run

Obey these limits. They win over every other line.

1. Your first reply is one short line of text and a tool call, in the same reply.
2. The JSON block holds `connect` (connector ids) and `install` (plugin ids). The ids are exact. Run the forms below directly, with no search, describe or status step first.
3. A check is any read-only command (`ls`, `which`, `find`, `cat`, a probe) or a file read. At most 2 checks before you make something. Put several checks in one command.
4. Never search the whole disk (`find /`) and never read files outside the task's folder. "What setup learned" is the survey.
5. At most one question before work, and only for a vague ask.
6. Never run the same call twice. Change a failed call once. If it fails again, say so and go on.
7. Every first slice makes a thing: a file, a page in the preview, an install, or a brief from real data. A chat summary of what you looked at is not a result.
8. Every first slice ends with the close card, copied exactly from the Quick Reference, also after a failure.
9. Say plainly what failed and what you did not test. Never mock or invent data.

## Quick Reference

| Step | What | Call |
|---|---|---|
| 1 | One line on what you start with | none |
| 2 | Connect the picked apps | `manage_connections` connect, every id in `connect`, once |
| 3 | Install the picked plugins | `manage_catalog` install (through `tool_call` when deferred), every id in `install`, once |
| 4 | Specific ask: start it. Vague ask: one card, three options | `clarify` |
| 5 | The first slice, finished in five minutes | the task's own tools |
| 6 | One line on the result, then the close card | `clarify`, close card below |

```
manage_connections  {"action":"connect","connectors":["<id>", ...]}
manage_catalog      {"action":"install","items":[{"kind":"plugin","id":"<id>"}, ...],"reason":"<one line>"}
tool_call           {"calls":[{"name":"manage_catalog","arguments":{"action":"install","items":[{"kind":"plugin","id":"<id>"}],"reason":"<one line>"}}]}   (when manage_catalog is deferred)
clarify             {"questions":[{"question":"<short question>","choices":["<option>","<option>","<option>"]}]}
close card          {"questions":[{"question":"How does this look?","choices":["Looks right","Change something","Take it further"]}]}
```

Put one entry in `calls` for each `tool_call`. Copy the close card exactly: never rename, add or mark a choice. Your next-step idea goes in the text line.

## Procedure

### 1. Connect first

Run steps 2 and 3 back to back, right after your first line. Skip a step whose list is empty.

Each step runs once. Never redo an earlier step. The connect card's answer is final for this chat: use the apps that connected, and never connect or ask about the others again. An app left unconnected after Continue counts as skipped. Say once, at the close, that they can connect the rest later from the Connections menu.

### 2. First move for each ask

| Ask | First move |
|---|---|
| A daily brief / work apps | Connect. Find the connected apps' tools with `tool_search`, then write the brief in the chat. None connected: section 3. |
| Set up this computer / install apps | One `clarify` app card, then section 4. |
| Make something in a plugin app (Blender) | Install. Then `skill_view` the plugin's skill by its id and build one small thing. Install not `connected`: section 3. |
| Automate something / a script | Section 5. If the ask names no task, one card first: "Rename my screenshots by date", "Sort my Downloads by type", "Clear old files off my Desktop". |
| Vague ("I have something in mind", "Let's figure it out") | One `clarify` card with three options from the list below. Then start the pick. |

Options for a vague ask. Each one makes a thing in five minutes. Name only apps they picked. Never offer "find", "review", "audit" or "clean up my &lt;app>": these make a survey, not a thing.

- Work apps picked: "A daily brief from Linear and Slack", "A summary of my week".
- An NVIDIA or Spark machine: "Install a few apps for this &lt;machine>", with the word setup's "This …:" line uses for it (Spark, PC, Mac).
- A plugin picked: "A simple scene in Blender".
- Otherwise: "A small HTML page about &lt;something they picked or said>", "A start page with links to my apps", "A quick script that tidies my Downloads".

The first option always makes a page or a file. Text they type instead of a choice is the pick. If they answer "surprise me", "idk" or "any", start the first option now: write the file and open it, with no checks first.

### 3. When a connect or an install fails

Do not stop with nothing.

1. No app connected, for a brief: say so in one line. Run one check, `gh auth status`. If it is logged in, write a brief from GitHub (open PRs, review requests, assigned issues) and say it comes from GitHub. If not, make a start page or a script that needs no account.
2. An install whose state is not `connected`, or that errors: say so in one line. Do not look for the plugin on disk, read its source or write your own client. Make the nearest thing without it. For Blender: write `~/hermes-first-task/first_scene.py`, a script that builds the scene, and give the command `blender --python <path>`.
3. End with the close card. Offer to try again in the text line.

### 4. Machine setup

Use the commands for the OS in the "What setup learned" line.

1. One `clarify` app card: three to five everyday apps that fit their use, multi-pick.
2. Check 1, one command. macOS: `which brew && ls -d "/Applications/Slack.app" "/Applications/Zoom.app" 2>&1`. Windows: `winget list --accept-source-agreements`. Linux: `which <app> ...`, plus `flatpak list` when flatpak is there.
3. Install only the picks the check did not find, one app per command. macOS: `brew install --cask <app>`. Windows: `winget install --id <id> -e --source winget --accept-source-agreements --accept-package-agreements` (a fresh profile otherwise stops at an agreements prompt the terminal cannot answer). Linux: `flatpak install -y flathub <id>`; a package that needs `sudo` goes on the list in step 5 instead. Say "already installed" for the others. Never reinstall.
4. On Arm, after the installs, say when an app is x64 only. macOS: one `lipo -archs` command for the new apps. Windows: read the architecture winget reports for the installer it picked in the install output (arm64, or x64 under emulation); do not infer it from the manifest.
5. List for them anything that needs a password, a licence or a payment. Never disable security settings.
6. One line on what changed and the next slice (developer tools, drivers), then the close card.

### 5. Automations

1. Write the script at once with `write_file` in `~/hermes-first-task/`. No checks first.
2. By default the script only prints what it would do. A flag such as `--apply` makes the changes.
3. Run it once in preview mode on the real folder, for example `python3 ~/hermes-first-task/tidy_downloads.py ~/Downloads`, and show the output. If you could not run it, say it is untested.
4. If the folder is missing or empty, say so. Never make test files or test folders (`mkdir`, `touch`, `SetFile`).
5. Give the one command that does it for real. Set up no recurring job unless they asked.

### 6. Build rules

- Real data only: from connected apps or tools signed in on this computer, such as a logged-in `gh` (say so). Never route around a connector: no IMAP, app password or scraping.
- Ask before sending, deleting or scheduling anything.
- A local model is set up by the app, not by you: point to Run locally in the model menu, or Settings, Providers, Local Models. Never install a model runtime or download a model yourself.
- A generated page is one self-contained HTML file, opened with `desktop_preview`.
- Save every new file in `~/hermes-first-task/`, never in the current folder. Do not copy files around. Say where each file is.
- Find a plugin's tools with `tool_search` after the install. If its app is not running, say so.
- Cut a big ask to a first slice and say so in one line. A long step (a large download, a full build) is never the first slice.
- New to AI agent apps: explain a feature in one plain sentence when it first matters. Say once that you ask for permission and they can say no.

### 7. Close the first slice

One short line on what you made and where it is, then the close card. Act on the pick.

## Pitfalls

- A chat report in place of a file, a page or an install.
- Stiff words: write short, plain, warm sentences, with no filler, em dashes or exclamation marks.

## Verification

- The first reply is one line of text and a tool call. Only a vague ask got a card.
- The first slice ends with the exact close card, also after a failure.

---
title: "Desktop Spellcheck — Enable and diagnose native desktop spelling safely"
sidebar_label: "Desktop Spellcheck"
description: "Enable and diagnose native desktop spelling safely"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Desktop Spellcheck

Enable and diagnose native desktop spelling safely.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/autonomous-ai-agents/desktop-spellcheck` |
| Path | `optional-skills/autonomous-ai-agents/desktop-spellcheck` |
| Version | `0.1.0` |
| Author | Nour (nourms), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `desktop`, `spelling`, `diagnostics`, `consent` |
| Related skills | [`hermes-agent`](../../bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Spell Check Skill

Enable, diagnose, and verify native desktop spelling through an installed, supported settings or plugin route. This optional procedural skill is instructions, not a plugin: it bundles no runnable plugin, spelling engine, dictionary, installer, or new SDK API. Loading it does not enable spelling or authorize installation.

## When to Use

- Native Hermes Desktop composer spelling is absent, inconsistent, or stops at underlines without usable corrections.
- A user wants an existing supported spelling control enabled and its actual effect verified.
- Do not use for whole-document proofreading, grammar rewriting, browser/dashboard spelling, or implementing a new engine or plugin catalog entry.

## Prerequisites

- An identifiable installed desktop build and tools already available for read-only inspection and authorized UI interaction.
- No required packages, environment variables, credentials, or setup commands; `config: []` declares no skill configuration.
- Load `hermes-agent` for current product guidance; read `references/diagnosis-and-qa.md` for the diagnostic matrix and review gate.
- Platform declarations mean procedural applicability only. Every OS/build/backend combination not exercised is **NOT TESTED**.
- Check whether a tool's first use bootstraps dependencies or repairs drivers before calling it; unknown behavior blocks that route.

## How to Run

Use `search_files` and `read_file` for scoped source inspection, `web_extract` for official documentation, and `terminal` only for verified installed read-only programs. Use an available desktop inspection/UI tool for native interaction; `browser_exec` against a web page is not proof of native desktop behavior. Discover tool schemas and the actual display/backend before interacting; there is no bundled command to execute.

## Quick Reference

| Evidence | What it proves |
|---|---|
| Visible underline | That word was marked, not that suggestions or replacement work |
| Suggestion rows | Menu presentation, not successful editing |
| Exact draft readback | Replacement and surrounding-text preservation on that target |
| Plugin toggle or API exists | Availability only, not enablement or runtime correctness |
| Source inspection or mocked test | Wiring only, not actual native UI QA |

## Procedure

### 1. Establish scope and preserve drafts

Agree on target composer, language, desired final enabled state, permitted UI interruption, and evidence handling. Leave the conversation hosting this task untouched; never use its composer as the test surface or send a demonstration message.

Identify unsent drafts before session switches, reloads, or toggles. Leave them intact; use a separate user-approved disposable composer with synthetic text. Do not assume session persistence saves drafts, copy private drafts into logs, or overwrite a draft to make space.

Before recording evidence, establish access, sharing/synchronization, and retention/removal boundaries. Prefer minimal synthetic-text observations; if exposure is unknown, avoid sensitive captures and ask for a suitable location. Never capture credentials or authentication dialogs.

**Exit:** Target, preservation plan, synthetic test surface, and evidence boundary are explicit.

### 2. Identify the running surface and actual root

Record desktop version/build or revision, Electron version when available, OS, native versus web surface, execution/display backend, connection mode, and selected profile. Distinguish a local renderer attached to a remote gateway from a browser-only dashboard.

Resolve the app-level desktop plugin root from the running app's supported reveal/settings route or matching loader source. Do not derive it solely from the selected profile or assume file tools and the desktop share a filesystem. Keep exact private roots only in authorized local notes; publish role labels instead.

Before any approved write, resolve the destination and existing ancestors on the actual filesystem. Reject traversal, unexpected symlinks/junctions/reparse points, escapes from the approved root, or unaccounted path races; string-prefix containment is insufficient. Never write a proposed plugin into a watched directory during review.

**Exit:** Build, surface, backend, profile, root ownership, and relevant filesystem are known or marked BLOCKED.

### 3. Find a supported installed route

Inspect the installed settings and plugin inventory without enabling anything. Match exact setting/plugin identity, version, scope, defaults, supported languages, network behavior, and rollback to official documentation and source for that build.

Prefer an existing documented setting, then an already-installed reviewed native-spelling plugin. Verify the public SDK contract rather than assuming a pending `host.composer.setSpellcheck` exists. Dictionary add/remove verbs or composer menu contributions alone do not establish an enable-spelling control.

Treat upstream examples and pending feature/catalog proposals as leads, not installed capabilities or accepted contributions. Do not copy example JavaScript into a plugin, invent an installer, import private modules, patch DOM/IPC, or modify the live app to fill a missing API.

If no supported route exists, stop enablement, report the exact missing capability, and offer a **separate reviewed version-specific plugin proposal**. Do not claim spelling enabled; this skill does not supply that proposal's implementation.

**Exit:** A documented installed route is identified, or the missing route is BLOCKED with a safe next step.

### 4. Review changes and obtain post-review consent

Before any new installation, dependency bootstrap/repair, update, plugin enablement, dictionary download/change, or settings/runtime change, apply the independent read-only **RED-TEAM** gate in the reference. Scope the review to the exact operation, candidate/version, actual destination, dependencies, data flow, privileges, and rollback.

An independent reviewer must inspect the proposal as data without running it or installing review tooling. Self-review is not independent review; if no independent reviewer is available, leave the gate OPEN/BLOCKED. Resolve high/critical or unknown side effects before presenting the findings.

Obtain explicit user consent **after** that review, including test edits and restoration. Prior broad authorization is not consent to an unseen payload. Changed payloads, paths, permissions, dependencies, or scope need renewed review and approval. Drafting instructions outside installed trees does not install them.

Explain that Electron's native spellchecker processes spelling locally rather than sending typed words to Google; Windows/Linux Hunspell dictionaries download from a Google CDN by default. This is not a claim that the whole app/plugin is offline or that capturing evidence sends nothing; inspect those data flows separately.

**Exit:** Independent findings and exact post-review authorization exist, or no operational change occurs.

### 5. Enable narrowly, then diagnose by layer

Use only the reviewed installed setting/plugin control; record its previous value and read back the new state. Do not change OS-wide spelling/language settings, enable whole-document spellcheck, add body observers, or attach broad event interception as a fallback.

Use the synthetic draft `Before speling after.` in the approved composer. With a supported English dictionary, observe the underline, open the word's actual context menu, and separately record suggestions and replacement. If English is unavailable, record that case instead of assuming the word must be marked.

Follow the reference matrix in order: language/dictionary readiness; editable hit-testing and `pointer-events`; native event facts; menu routing/filtering/subscriber; callback; current focus/selection and actual text. Inspect private fields only as version-sensitive read-only diagnostics, never as an extension API or repair surface.

Do not replay saved ranges, inject synthetic native events, rewrite the entire draft, or treat clearing an underline by adding a word as correction. Re-open the menu on the current target after cancellation or a session change. Stop on cross-composer edits or draft loss; restore only the specifically approved change when safe.

**Exit:** Underlines, suggestions, and replacement each have an observed result or a localized blocker.

### 6. Exercise the actual UI and restore safely

Run every reference QA row: `speling` → `spelling`, unchanged surrounding text, cancel/reopen, typing after cancel, multiple composers/session switch, language/unsupported behavior, and disable/restore without draft loss. Read the live destination after each action; a screenshot of a menu or a callback returning success is insufficient.

Test only synthetic surfaces, never send the draft, and do not edit existing submitted messages for coverage. Mark unavailable composer types or platforms NOT TESTED rather than substituting a different surface. A failure permits one corrected safe retry after fresh inspection, not a sequence of speculative patches.

Restore the reviewed previous setting/plugin state after the disable test; leave spelling enabled only if that is the authorized final state. Keep user drafts intact and stop on uncertain restoration. Remove only explicitly approved synthetic test material; no automatic cleanup of user data or evidence.

**Exit:** Required UI rows have evidence, drafts are preserved, and final state is read back.

## Pitfalls

- Native suggestions can exist while menu routing hides them; a menu can appear while replacement targets nothing.
- A correct DOM `spellcheck` value does not prove native hit-testing reached an editable element.
- Dictionary availability, selected language, download completion, and app display language are different facts.
- A remote backend, app-level plugin root, selected profile, and web preview are not interchangeable targets.
- Private bridge/store/editor names change between builds; source inspection is not permission to call or patch them.
- Disabling a plugin does not necessarily reverse dictionary entries or persisted settings; inspect rollback first.

## Verification

Report PASS, FAIL, BLOCKED, or NOT TESTED per QA row with build/OS/backend and synthetic evidence. Separate enabled-state readback, underlines, menu suggestions, and exact one-word replacement; do not collapse these into “works.”

State the chosen supported route, review/consent status, before/after settings, final restoration, draft preservation, and remaining limitations. Only tested combinations receive PASS; Linux, macOS, and Windows otherwise remain NOT TESTED. This skill makes no acceptance, certification, installation, or cross-platform runtime claim.

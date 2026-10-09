---
title: "Ripwire — Call-graph code intelligence for agents: ranked maps, blast radius, tests-to-run"
sidebar_label: "Ripwire"
description: "Call-graph code intelligence for agents: ranked maps, blast radius, tests-to-run"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Ripwire

Call-graph code intelligence for agents: ranked maps, blast radius, tests-to-run.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/software-development/ripwire` |
| Path | `optional-skills/software-development/ripwire` |
| Version | `1.0.0` |
| Author | Hermes Agent; tool upstream redhat-et/ripwire (Apache-2.0) |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `code-search`, `call-graph`, `blast-radius`, `code-navigation`, `token-efficiency` |
| Related skills | [`ast-grep`](/docs/user-guide/skills/optional/software-development/software-development-ast-grep), [`systematic-debugging`](/docs/user-guide/skills/bundled/software-development/software-development-systematic-debugging) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# ripwire

`ripwire` is a zero-dependency CLI (C++23, single binary) that parses a codebase into a
call graph and answers agent-shaped questions deterministically: ranked symbol maps for
a task, who calls a symbol, the transitive blast radius of a change, and which indexed
tests reach the changed code. Where `search_files`/`rg` answer "where does this text
appear", ripwire answers "what does this code touch".

Upstream: https://github.com/redhat-et/ripwire (Apache-2.0, release binaries for
linux/macOS x64+arm64). Install with this skill's `install.sh`, or
`hermes skills install official/software-development/ripwire` then run its install script.

## When to reach for it

- **Before editing a function** — `--callers` lists distinct callers with test-coverage
  flags; `--impact` gives the transitive reach set split tested/untested. This is the
  "grep every caller before you edit" discipline, done by the call graph instead.
- **Entering a codebase cold** — `--for="one-line task"` returns the ranked symbol set
  plus one-hop context in a few K tokens, instead of many greps and file reads.
- **Before merge** — `--test-gate` / `--exercises` answer "which tests should run for
  this change".

Languages: C++, C, ObjC, Metal, CUDA, Python, TypeScript, JavaScript, Java, Ruby, PHP,
Lua, Elixir, Dart, Kotlin, Bash, Go, Rust, Swift, C# (plus JSON/TOML/YAML/Markdown).

## Commands that matter

```bash
ripwire <dir> --for="add retry to extractor"   # ranked map for a task
ripwire <dir> --callers=position_size          # who calls it (distinct symbols)
ripwire <dir> --uses=position_size             # every call site (one row per occurrence)
ripwire <dir> --impact=classify --limit=10     # blast radius, tested/untested split
ripwire <dir> --test-gate                      # tests reaching changed symbols
ripwire <dir> --scan-skills <dir>              # injection/exfil scan of skill files
```

Output is minified XML with a self-describing legend; pipe through grep/jq for the
fields you need, or pass `--json`.

## Honest limits (state these when reporting results)

- Call edges are heuristic and name-based: dynamic dispatch, callbacks, and
  macro-generated call sites produce no edge. Every count carries `counts_floor="1"` —
  read a 0 as "none found", never "none exists".
- `--top-k` does not narrow `--impact`/`--callers`; use `--limit=N` there.
- `--scan-skills` has a high false-positive rate on vendored API docs (doc curl
  examples with `Bearer ***` placeholders). Triage findings in authored files only.
- Build from source needs C++23 (GCC 13+/Clang 17+); ~4 min on 4 vCPU. Release binaries
  are preferred — see `references/commands.md` and `install.sh`.

## See also

- `references/commands.md` — measured behavior, flag pitfalls hit live, rebuild recipe.

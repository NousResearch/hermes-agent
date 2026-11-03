# H2C Protocol v2.1 — Multi-Source Passive Session Sync

> **Hermes ← Claude Code + Codex CLI: Zero-Intrusion Session Intelligence**

## 1. Problem

### 1.1 Two Scenarios

| | Scene 1: Hermes dispatches CC | Scene 2: User works with AI CLI directly |
|---|---|---|
| Info flow | Bidirectional, real-time | One-way, post-hoc |
| Hermes visibility | Full (via RFC-001) | **Zero** |
| Frequency | Low (formal tasks) | **High (daily driver)** |

Scene 2 is the primary usage pattern. User opens Claude Code or Codex, discusses architecture, writes code, makes decisions — Hermes knows nothing.

### 1.2 What Hermes Needs

Not everything is worth syncing. From Hermes' role (scheduler, memory keeper, planner):

| Info Type | Value | Example |
|---|---|---|
| **Decisions** | High | "Chose ASE over LAMMPS for Python integration" |
| **Progress** | High | "Data loader complete, next: train baseline" |
| **Problems/Blockers** | High | "xyz parser has edge case, workaround in place" |
| Detailed code diffs | Low | Git already records this |
| Debugging process | Low | Noise, not useful for Hermes |

**Three words: decisions, progress, problems.**

### 1.3 The Discussion-Only Gap

Not all valuable sessions produce commits or file changes:
- Architecture discussions with a chosen direction
- Tech stack evaluation with conclusions
- Paper analysis with key takeaways
- Work planning for next week

These **pure discussion sessions** have the highest decision density but leave zero trace in git or filesystem. Any sync mechanism that relies solely on file changes or git history will miss them entirely.

## 2. Architecture: Direct JSONL Extraction

### 2.1 Key Insight

Both Claude Code and Codex store session data locally as `.jsonl` files. These files record every message — user, assistant, tool calls, tool results — regardless of whether files were changed or commits were made.

| Source | Session Location | Format |
|---|---|---|
| Claude Code | `~/.claude/projects/**/*.jsonl` | `type: user/assistant`, `message.content` |
| Codex CLI | `~/.codex/sessions/**/*.jsonl` + `~/.codex/archived_sessions/*.jsonl` | `type: response_item`, `payload` with role |

**Hermes can read both directly. No hooks needed. No CLI modification needed.**

### 2.2 Data Profile

Analysis of real session files shows the same pattern across both sources:

| Content | Size % | Useful for Hermes? |
|---|---|---|
| Tool outputs + images (base64) | ~97% | No — pure noise |
| Assistant text blocks (spoken responses) | ~0.2% | **Core value** |
| Tool call metadata (what tools were called) | Small | Useful metadata |
| User text blocks (what user asked) | Small | Context |
| System / thinking / developer prompts | Small | No |

**The useful information is less than 1% of file size.** A simple filter extracts it at zero token cost.

### 2.3 Pipeline

```
AI CLI works normally (zero modification)
       |
       ├── ~/.claude/projects/**/*.jsonl (Claude Code)
       └── ~/.codex/sessions/**/*.jsonl  (Codex CLI)
       |
h2c_extract.py (rule-based filter, zero token cost)
  - Adapter pattern: ClaudeCodeParser / CodexParser
  - Extract: user text + assistant text + tool metadata
  - Drop: tool_result, image, thinking, system, developer
  - Sanitize: redact secrets, truncate long blocks
  - Tag: code-change / discussion-only / debugging
       |
~/.hermes/inbox/ (conversation skeletons, ~15-20KB each)
  - cc_*.md  (from Claude Code)
  - cx_*.md  (from Codex CLI)
       |
Hermes reads inbox (cheap model, on wake-up or cron)
  - Generate structured summary (~500 words)
  - Update Hermes memory
  - Update Wiki (if needed)
  - Archive processed files
```

### 2.4 Why Not Stop Hook? (v1 Design, Deprecated)

| Issue | Detail |
|---|---|
| CC writes blind | Doesn't know what Hermes needs, may produce useless output |
| Crash = data loss | Hook doesn't fire on abnormal exit |
| Discussion sessions | Hook quality depends on remaining context at exit |
| Maintenance burden | Requires hook installation per CLI |
| Fragmentation | 10 short sessions = 10 fragments Hermes must reconcile |
| Single-source | Only works for one CLI, not extensible |

The JSONL approach eliminates all of these: data is already written, always complete, and Hermes extracts what it needs on its own terms. Adding a new source is just a new parser class.

## 3. Multi-Source Parser Architecture

### 3.1 Adapter Pattern

```
SessionParser (ABC)
  ├── discover_files()      → list[Path]
  ├── parse()               → list[DialogTurn]
  ├── get_session_id()      → str
  ├── get_project_name()    → str
  └── get_session_date()    → str
       │
       ├── ClaudeCodeParser
       │     source_name = "cc"
       │     Reads: ~/.claude/projects/**/*.jsonl
       │     Filters: subagents/ directory
       │     Strips: <system-reminder>, <command-message>, etc.
       │
       └── CodexParser
             source_name = "codex"
             Reads: ~/.codex/sessions/**/*.jsonl
                  + ~/.codex/archived_sessions/*.jsonl
             Extracts: session_meta.payload.cwd for project name
             Strips: <permissions>, <app-context>, etc.
             Attaches: function_call items to preceding assistant turn
```

Both parsers output identical `list[DialogTurn]`. Everything downstream (tagging, truncation, output) is shared.

### 3.2 Format Differences Handled

| Aspect | Claude Code | Codex CLI |
|---|---|---|
| Message type field | `type: "user"/"assistant"` | `payload.type: "message"`, `payload.role` |
| Tool calls | `content[].type = "tool_use"` | `payload.type = "function_call"` |
| Tool results | `content[].type = "tool_result"` | `payload.type = "function_call_output"` |
| System noise | `<system-reminder>`, `<command-message>` | `<permissions>`, `<app-context>`, `<environment_context>` |
| Session ID | Filename UUID | UUID extracted from `rollout-DATE-UUID.jsonl` |
| Project name | Encoded in parent directory name | `session_meta.payload.cwd` (first line) |
| Code change tools | Edit, Write, MultiEdit | write_file, edit_file, apply_diff |

### 3.3 Adding New Sources

To add a third CLI (e.g., Gemini CLI, Cursor):
1. Create a new `XxxParser(SessionParser)` class
2. Implement the 5 abstract methods
3. Add to `_get_parsers()` list
4. Add source-specific system tags if needed

No changes needed to tagging, output, or sync logic.

## 4. Technical Specification

### 4.1 Extraction: What to Keep

```
Per session file, extract only:

1. user text blocks     → What user asked / requested (context)
2. assistant text blocks → AI's analysis, decisions, conclusions (core)
3. tool metadata        → Which tools were called, on which files (fact layer)

Skip everything else:
- tool_result / function_call_output (tool output, often massive)
- image blocks (base64, megabytes of noise)
- thinking blocks (internal reasoning, not actionable)
- system / developer messages (prompts, reminders)
```

### 4.2 Output Format: Conversation Skeleton

Markdown, not JSON — optimized for cheap model readability:

```markdown
---
session: 86fcda92
date: 2026-04-07
project: coding
source: cc
tags: [discussion-only]
files_touched: []
---

**User**: Look at this design file
**CC**: This is the H2C Protocol v2.0 design doc describing one-way passive memory sync...
**User**: There are two scenarios...
**CC**: H2C Protocol mainly solves the information gap in Scene 2...
```

Codex sessions use `**Codex**:` instead of `**CC**:` as the assistant label.

### 4.3 Compression Rules

| Text block size | Strategy |
|---|---|
| < 500 chars | Keep as-is |
| 500-2000 chars | Keep first 300 + last 200 + `[...omitted N chars...]` |
| > 2000 chars | Keep first 300 + last 200, replace code blocks with `[code: ~N lines]` |

**Per-file cap**: MAX_SKELETON_CHARS = 20,000. Files exceeding this are trimmed from the top (keeping most recent context).

### 4.4 Auto-Tagging

Source-aware tagging:

| Tag | Claude Code trigger | Codex trigger |
|---|---|---|
| `code-change` | Edit, Write, MultiEdit used | write_file, edit_file, apply_diff used |
| `discussion-only` | No tool calls at all | No tool calls at all |
| `debugging` | "error", "bug", "fail" etc. in text | Same |
| `multi-agent` | Agent tool used | N/A |

### 4.5 Incremental Sync

- Track processed files by `(filepath, mtime)` in `~/.hermes/h2c-state.json`
- Only process new or modified session files
- Skip files modified within last 5 minutes (may still be active)
- Checkpoint progress every 50 files (crash-safe)
- Skip `subagents/` directory for Claude Code

### 4.6 Sensitive Data Filtering

Mandatory regex-based redaction before writing to inbox:

| Pattern | Replacement |
|---|---|
| `sk-*`, `key-*` | `[API_KEY_REDACTED]` |
| `ghp_*`, `gho_*` | `[GITHUB_TOKEN_REDACTED]` |
| JWT (`eyJ...`) | `[JWT_REDACTED]` |
| `Bearer *` | `[BEARER_TOKEN_REDACTED]` |
| `password=*`, `secret=*` | `[PASSWORD_REDACTED]` / `[SECRET_REDACTED]` |

**Note**: Passwords spoken directly in conversation (not in `key=value` format) are NOT caught by regex. Hermes should treat inbox contents as semi-sensitive and not store raw skeletons long-term.

### 4.7 Directory Structure

```
~/.hermes/
├── inbox/                          ← Conversation skeletons (Hermes reads here)
│   ├── 2026-04-07_cc_86fcda92.md   ← Claude Code session
│   ├── 2026-04-07_cx_019d0e55.md   ← Codex session
│   └── ...
├── inbox-archive/                  ← Processed skeletons (Hermes moves here)
│   └── ...
├── h2c-state.json                  ← Sync state (processed files tracking)
└── skills/h2c-protocol/
    ├── DESIGN.md                   ← This file
    ├── RFC-001.md                  ← Scene 1 protocol (Hermes dispatches CC)
    └── h2c_extract.py              ← Extraction script (multi-source)
```

## 5. Layered Compression Model

```
Layer 0: Raw .jsonl              (0.5-4MB per session, both sources)
    ↓  h2c_extract.py — rule-based filter (zero token cost)
Layer 1: Conversation skeleton   (~15-20KB) → inbox/
    ↓  Hermes reads (cheap model)
Layer 2: Structured summary      (~500 words) → Hermes memory
    ↓  Hermes decides (cheap model)
Layer 3: Wiki update             (if needed) → ~/.hermes/wiki/
```

Cost flows downhill: expensive model (CC/Codex) does the original thinking, script does mechanical extraction, cheap model (Hermes) does semantic compression.

## 6. Relationship to RFC-001

| | RFC-001 (Scene 1) | This Design (Scene 2) |
|---|---|---|
| Direction | Hermes → CC (dispatch) | AI CLI → Hermes (passive sync) |
| Trigger | Hermes writes `inbox/current-task.md` | Hermes/cron runs `h2c_extract.py` |
| CLI awareness | CC reads task, reports via `activity.jsonl` | CLI is unaware, works normally |
| Data source | Structured events (`activity.jsonl`) | Raw session files (`.jsonl`) |
| Sources | Claude Code only | Claude Code + Codex (extensible) |
| Complementary | Yes — RFC-001 tasks also generate `.jsonl` sessions that this pipeline captures |

Both protocols feed into the same Hermes memory and Wiki. They cover different scenarios but converge at the consumption layer.

## 7. Usage

```bash
# Sync all sources (Claude Code + Codex)
python3 ~/.hermes/skills/h2c-protocol/h2c_extract.py sync

# Sync only Claude Code
python3 ~/.hermes/skills/h2c-protocol/h2c_extract.py sync --source cc

# Sync only Codex
python3 ~/.hermes/skills/h2c-protocol/h2c_extract.py sync --source codex

# Re-process everything
python3 ~/.hermes/skills/h2c-protocol/h2c_extract.py sync --force

# Check status
python3 ~/.hermes/skills/h2c-protocol/h2c_extract.py status
```

## 8. Implementation Plan

### PR 1: h2c_extract.py (This PR)

1. `h2c_extract.py` — Multi-source extraction script
   - `SessionParser` ABC with `ClaudeCodeParser` and `CodexParser`
   - Session file discovery + incremental sync with checkpointing
   - Conversation skeleton extraction with per-block and per-file size caps
   - Sensitive data redaction (regex-based)
   - Source-aware auto-tagging
   - Markdown output to `~/.hermes/inbox/`
2. Updated `DESIGN.md` — This document
3. Updated `RFC-001.md` — Unchanged (Scene 1 protocol)

### PR 2: Hermes Consumption (Future)

- Hermes reads `inbox/`, generates structured summaries
- Updates Hermes internal memory
- Updates Wiki (`DECISIONS.md`, `ARCHITECTURE.md`)
- Archives processed skeletons to `inbox-archive/`
- Daily digest via configured messaging channel

### PR 3: Trigger Automation (Future)

- macOS launchd plist for periodic sync (every 10 minutes)
- Optional CC Stop Hook as instant trigger (launchd as fallback)
- Hermes cron integration

## 9. Edge Cases

| Case | Handling |
|---|---|
| CLI crashes mid-session | `.jsonl` is append-only; partial data is still extractable |
| Very short sessions (< 3 user messages) | Skip — not enough signal |
| Low-value sessions (help/usage only) | Skip — detected by content filter |
| Multiple sessions same project same day | Each gets its own skeleton file (keyed by source + session ID) |
| Session still active (CLI running) | Skip files modified within last 5 minutes |
| Sensitive info leaks through regex | Hermes should not store raw skeletons long-term |
| Conversational passwords (not key=value) | Not caught by regex; documented limitation |
| Session file deleted during scan | `stat()` protected with fallback; skipped gracefully |
| Disk full during write | `write_skeleton` catches OSError, cleans up temp file |

## 10. Validated Results

First full sync (2026-04-07):

| Metric | Value |
|---|---|
| Claude Code sessions discovered | 4,569 |
| Claude Code skeletons written | 163 |
| Codex sessions discovered | 319 |
| Codex skeletons written | 71 |
| **Total skeletons** | **234** |
| Largest skeleton | 38KB (within 20KB trim target) |
| Total inbox size | 3.5MB |
| Processing errors | 0 |

---

*Status: Implemented and validated. Three rounds of strict code review passed.*
*Supersedes: v1.0 (Stop Hook approach) and v2.0 (CC-only).*

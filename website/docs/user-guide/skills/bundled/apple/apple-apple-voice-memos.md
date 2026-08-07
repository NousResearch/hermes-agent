---
title: "Apple Voice Memos — Read and search Apple Voice Memos transcripts"
sidebar_label: "Apple Voice Memos"
description: "Read and search Apple Voice Memos transcripts"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Apple Voice Memos

Read and search Apple Voice Memos transcripts.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/apple/apple-voice-memos` |
| Version | `1.2.0` |
| Author | Zach Leahan (@ZacharyLeahan) |
| License | MIT |
| Platforms | macos |
| Tags | `VoiceMemos`, `Apple`, `macOS`, `transcription`, `note-taking` |
| Related skills | [`apple-notes`](/docs/user-guide/skills/bundled/apple/apple-apple-notes), [`apple-reminders`](/docs/user-guide/skills/bundled/apple/apple-apple-reminders) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Apple Voice Memos Skill

Read and search Apple's on-device transcripts and metadata from the user's
Voice Memos library. Do not transcribe arbitrary audio or record new audio.

## When to Use

- Read, summarize, or search the user's Voice Memos.
- Find recent recordings by title, filename, unique ID, or list index.
- Turn existing memo transcripts into notes, tasks, or summaries.

Do not use this skill to transcribe unrelated audio files or record new audio.

## Prerequisites

- Run on macOS with `python3` available.
- Use Voice Memos on macOS 15+/iOS 18+ for Apple-generated transcripts.
- Grant Full Disk Access to the Python executable named in an access error:
  **System Settings → Privacy & Security → Full Disk Access**.
- Open or download a memo in Voice Memos when its JSON metadata reports
  `"exists": false`, then retry.

## How to Run

Use the `terminal` tool from this skill directory. The script is
`scripts/voicememos.py`, has no third-party Python dependencies, and supports
`--json` on every command.

```bash
python3 scripts/voicememos.py list
python3 scripts/voicememos.py transcript 0
python3 scripts/voicememos.py dump --limit 10 --only-transcribed
```

The script copies the Voice Memos SQLite database and its WAL sidecars to a
private temporary directory before querying. Never write to Apple's live
database or recording files.

## Quick Reference

### List recordings

```bash
python3 scripts/voicememos.py list                      # 30 newest
python3 scripts/voicememos.py list --limit 10
python3 scripts/voicememos.py list --with-transcript
python3 scripts/voicememos.py list --search "meeting"
python3 scripts/voicememos.py list --search "meeting" --json
```

`[A]` means Apple transcript, `[D]` means the audio is not downloaded locally,
and `[ ]` means the local audio has no embedded Apple transcript. The leading
zero-based `[n]` is the recording's index in the complete newest-first list.
Filters preserve that original index, so filtered output can contain gaps.
JSON list items include the same `index`.

### Read one transcript

```bash
python3 scripts/voicememos.py transcript 0
python3 scripts/voicememos.py transcript "Recording 45"
python3 scripts/voicememos.py transcript 01234567-89AB-CDEF-0123-456789ABCDEF
python3 scripts/voicememos.py transcript filename.m4a --json
```

A non-numeric selector matches the first title, filename, or exact unique ID.

### Dump multiple transcripts

```bash
python3 scripts/voicememos.py dump --limit 10 --only-transcribed
python3 scripts/voicememos.py dump --search "meeting" --json
```

## Procedure

1. Run `list` with the narrowest useful search and limit.
2. Check the transcript marker and `exists` value before promising content.
3. Run `transcript` with the displayed index, filename, or unique ID for one
   memo; use `dump` for a batch.
4. Distinguish audio that is not downloaded from local audio that has no Apple
   transcript. Do not describe either state as a parser failure.
5. Summarize or transform the returned text as requested. Use the related
   `apple-notes` or `apple-reminders` skill only when the user asks to save it.

Hermes does not automatically transcribe recordings when Apple text is absent.
For external transcription, polling, deduplication, reminders, Notes logging,
or scheduled processing, reuse this script as the metadata/Apple-transcript
layer and keep the additional workflow in custom automation.

## Pitfalls

- Do not renumber filtered results; the displayed index must continue to select
  the same item from the complete list.
- Do not promise that Apple will eventually generate a transcript. Old, short,
  unsupported, or incompletely processed recordings may never receive one.
- Do not imply that this bundled skill includes an external transcription
  fallback; any such pipeline is user-owned automation.
- Do not modify `CloudRecordings.db`, its sidecars, or `.m4a` files.
- A missing transcript can mean the recording is old, short, still syncing, or
  not yet processed by Apple.

## Verification

- Confirm the selected memo's title, date, duration, and filename with the user
  when multiple title matches are plausible.
- Confirm filtered indexes resolve to the same recording shown by `list`.
- Confirm JSON reports `transcript_source` as `apple` or `null`, preserves
  `exists`, and reports the original `index` for list results.
- Confirm an access failure names the Python executable that needs Full Disk
  Access and does not expose or modify Voice Memos data.

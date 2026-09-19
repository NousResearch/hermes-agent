# Quiet Feishu progress cards

Feishu/Lark can display one collapsible progress card per agent turn instead of sending a separate message for each tool or interim update. This is opt-in; existing messaging behavior remains unchanged when disabled.

## Enable

Merge this setting into the active profile's `config.yaml`:

```yaml
platforms:
  feishu:
    extra:
      progress_cards: true
```

Use a YAML boolean, not the string `"true"`. Restart the gateway after changing its configuration. No additional credentials are needed beyond the existing Feishu integration.

## Behavior

- The first progress update creates an interactive schema 2.0 card. Later updates PATCH that same message.
- Execution details are collapsed by default. Updates are throttled and bounded to avoid message spam and oversized payloads.
- Tool calls are correlated by call ID, including overlapping calls of the same tool.
- Completed assistant commentary appears inside the card rather than as speculative streamed text bubbles. Final answers and confirmation prompts retain their existing delivery paths.
- The header distinguishes running, confirmation needed, completed, interrupted and failed states. Tool errors produce a warning state even if later work succeeds. A completed turn is not an independent verification of the user's goal.
- Transport failures latch one compact fallback notice. An ambiguous create is not retried; a known card may receive one final recovery PATCH.
- Quiet cards do not override scheduled-heartbeat suppression, other platforms, or streaming TTS.

## Privacy and limitations

Reasoning deltas, tool results and confirmation prompt contents are excluded from the card. Credential redaction runs before preview truncation. URL queries/fragments are removed and markup is escaped. The card includes a session identifier to support local session lookup.

This is **not a general personal-information anonymizer**: ordinary commentary and tool labels may contain filenames, paths or task content. Enable it only in chats whose audience may see that information. Sanitized progress entries also use the existing profile-scoped logger and its retention policy. No new analytics endpoint or external storage is introduced.

The initial card labels and fallback notices are in Chinese. Disabling the feature returns to the existing progress presentation.

## Verification

Run the focused behavioral tests through the canonical runner:

```bash
scripts/run_tests.sh tests/gateway/test_feishu_progress_cards.py \
  tests/gateway/test_feishu_progress_status.py \
  tests/gateway/test_progress_card_lifecycle.py \
  tests/gateway/test_display_null_turn_wiring.py
```

The tests exercise actual gateway/adapter wiring with isolated configuration and simulated model/Feishu API boundaries. They are not evidence of a live Feishu API canary.

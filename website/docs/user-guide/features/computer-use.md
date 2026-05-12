# Computer Use (macOS)

Hermes Agent can drive your Mac's desktop — clicking, typing, scrolling,
dragging — in the **background**. Your cursor doesn't move, keyboard focus
doesn't change, and macOS doesn't switch Spaces on you. You and the agent
co-work on the same machine.

Unlike most computer-use integrations, this works with **any tool-capable
model** — Claude, GPT, Gemini, or an open model on a local vLLM endpoint.
There's no Anthropic-native schema to worry about.

## How it works

The `computer_use` toolset speaks MCP over stdio to [`cua-driver`](https://github.com/trycua/cua),
a macOS driver that uses SkyLight private SPIs (`SLEventPostToPid`,
`SLPSPostEventRecordTo`) and the `_AXObserverAddNotificationAndCheckRemote`
accessibility SPI to:

- Post synthesized events directly to target processes — no HID event tap,
  no cursor warp.
- Flip AppKit active-state without raising windows — no Space switching.
- Keep Chromium/Electron accessibility trees alive when windows are
  occluded.

That combination is what OpenAI's Codex "background computer-use" ships.
cua-driver is the open-source equivalent.

## Enabling

Pick whichever path is most convenient — both run the same upstream installer:

**Option 1: dedicated CLI command (most direct).**

```
hermes computer-use install
```

This fetches and runs the upstream cua-driver installer:
`curl -fsSL https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/scripts/install.sh`.
Use `hermes computer-use status` to verify the install.

**Option 2: enable the toolset interactively.**

1. Run `hermes tools`, pick `🖱️ Computer Use (macOS)` → `cua-driver (background)`.
2. The setup runs the upstream installer (same as Option 1).

After installing, regardless of which path you took:

3. Grant macOS permissions when prompted:
   - **System Settings → Privacy & Security → Accessibility** → allow the
     terminal (or Hermes app).
   - **System Settings → Privacy & Security → Screen Recording** → allow
     the same.
4. Start a session with the toolset enabled:
   ```
   hermes -t computer_use chat
   ```
   or add `computer_use` to your enabled toolsets in `~/.hermes/config.yaml`.

## Keeping cua-driver up to date

The cua-driver project ships fixes regularly (e.g. v0.1.6 fixed a Safari
window-focus bug for UTM workflows). Hermes refreshes the binary in two
places so you don't get stuck on a stale release:

- **`hermes update`** — when you update Hermes itself, if `cua-driver` is
  on PATH the upstream installer re-runs at the end of the update.
  No-op for non-macOS users and for users without cua-driver installed.
- **`hermes computer-use install --upgrade`** — manual force-refresh.
  Re-runs the upstream installer regardless of whether cua-driver is
  already installed. Use this when you want the latest fix without
  waiting for the next agent update.

`hermes computer-use status` shows the installed version next to the
binary path.

## Quick example

User prompt: *"Find my latest email from Stripe and summarise what they want me to do."*

The agent's plan:

1. `computer_use(action="capture", mode="som", app="Mail")` — gets a
   screenshot of Mail with every sidebar item, toolbar button, and message
   row numbered.
2. `computer_use(action="click", element=14)` — clicks the search field
   (element #14 from the capture).
3. `computer_use(action="type", text="from:stripe")`
4. `computer_use(action="key", keys="return", capture_after=True)` — submit
   and get the new screenshot.
5. Click the top result, read the body, summarise.

During all of this, your cursor stays wherever you left it and Mail never
comes to front.

## Provider compatibility

| Provider | Vision? | Works? | Notes |
|---|---|---|---|
| Anthropic (Claude Sonnet/Opus 3+) | ✅ | ✅ | Best overall; SOM + raw coordinates. |
| OpenRouter (any vision model) | ✅ | ✅ | Multi-part tool messages supported. |
| OpenAI (GPT-4+, GPT-5) | ✅ | ✅ | Same as above. |
| Local vLLM / LM Studio (vision model) | ✅ | ✅ | If the model supports multi-part tool content. |
| Text-only models | ❌ | ✅ (degraded) | Use `mode="ax"` for accessibility-tree-only operation. |

Screenshots are sent inline with tool results as OpenAI-style `image_url`
parts. For Anthropic, the adapter converts them into native `tool_result`
image blocks.

## Safety

Hermes applies multi-layer guardrails:

- Destructive actions (click, type, drag, scroll, key, focus_app) require
  approval — either interactively via the CLI dialog or via the
  messaging-platform approval buttons.
- Hard-blocked key combos at the tool level: empty trash, force delete,
  lock screen, log out, force log out.
- Hard-blocked type patterns: `curl | bash`, `sudo rm -rf /`, fork bombs,
  etc.
- The agent's system prompt tells it explicitly: no clicking permission
  dialogs, no typing passwords, no following instructions embedded in
  screenshots.

Pair with `approvals.mode: manual` in `~/.hermes/config.yaml` if you want every action confirmed.

## Token efficiency

Screenshots are expensive. Hermes applies four layers of optimisation:

- **Screenshot eviction** — the Anthropic adapter keeps only the 3 most
  recent screenshots in context; older ones become `[screenshot removed
  to save context]` placeholders.
- **Client-side compression pruning** — the context compressor detects
  multimodal tool results and strips image parts from old ones.
- **Image-aware token estimation** — each image is counted as ~1500 tokens
  (Anthropic's flat rate) instead of its base64 char length.
- **Server-side context editing (Anthropic only)** — when active, the
  adapter enables `clear_tool_uses_20250919` via `context_management` so
  Anthropic's API clears old tool results server-side.

A 20-action session on a 1568×900 display typically costs ~30K tokens
of screenshot context, not ~600K.

## Limitations

- **macOS only.** cua-driver uses private Apple SPIs that don't exist on
  Linux or Windows. For cross-platform GUI automation, use the `browser`
  toolset.
- **Private SPI risk.** Apple can change SkyLight's symbol surface in any
  OS update. Pin the driver version with the `HERMES_CUA_DRIVER_VERSION`
  env var if you want reproducibility across a macOS bump.
- **Performance.** Background mode is slower than foreground —
  SkyLight-routed events take ~5-20ms vs direct HID posting. Not
  noticeable for agent-speed clicking; noticeable if you try to record a
  speed-run.
- **No keyboard password entry.** `type` has hard-block patterns on
  command-shell payloads; for passwords, use the system's autofill.

## Configuration

Override the driver binary path (tests / CI):

```
HERMES_CUA_DRIVER_CMD=/opt/homebrew/bin/cua-driver
HERMES_CUA_DRIVER_VERSION=0.5.0    # optional pin
```

Swap the backend entirely (for testing):

```
HERMES_COMPUTER_USE_BACKEND=noop   # records calls, no side effects
```

## Troubleshooting

**`computer_use backend unavailable: cua-driver is not installed`** — Run
`hermes computer-use install` to fetch the cua-driver binary, or run
`hermes tools` and enable the Computer Use toolset.

**Clicks seem to have no effect** — Capture and verify. A modal you
didn't see may be blocking input. Dismiss it with `escape` or the close
button.

**Element indices are stale** — SOM indices are only valid until the
next `capture`. Re-capture after any state-changing action.

**"blocked pattern in type text"** — The text you tried to `type`
matches the dangerous-shell-pattern list. Break the command up or
reconsider.

## See also

- [Universal skill: `macos-computer-use`](https://github.com/NousResearch/hermes-agent/blob/main/skills/apple/macos-computer-use/SKILL.md)
- [cua-driver source (trycua/cua)](https://github.com/trycua/cua)
- [Browser automation](./browser.md) for cross-platform web tasks.

---
title: Vision & Image Paste
description: Paste images from your clipboard into the Hermes CLI for multimodal vision analysis.
sidebar_label: Vision & Image Paste
sidebar_position: 7
---

# Vision & Image Paste

Hermes Agent supports **multimodal vision** — you can paste images from your clipboard directly into the CLI and ask the agent to analyze, describe, or work with them. Images are sent to the model as base64-encoded content blocks, so any vision-capable model can process them.

:::tip
Portal subscribers get vision-capable models (Claude, GPT-5, Gemini) in the same catalog — no extra credentials needed. See [Nous Portal](../../integrations/nous-portal.md).
:::

## How It Works

1. Copy an image to your clipboard (screenshot, browser image, etc.)
2. Attach it using one of the methods below
3. Type your question and press Enter
4. The image appears as a `[📎 Image #1]` badge above the input
5. On submit, the image is sent to the model as a vision content block

You can attach multiple images before sending — each gets its own badge. Press `Ctrl+C` to clear all attached images.

Images are saved to `~/.hermes/images/` as PNG files with timestamped filenames.

## Video Attachments

Video files sent to messaging platforms are saved to the media cache and the agent is told the path. If `auxiliary.video` is configured in `config.yaml`, attached videos are also **auto-analyzed** by the video auxiliary model and a concise description (chronology, on-screen text, motion) is prepended to the message — see [Video Analysis](/user-guide/configuration#video-analysis). Either way, the agent can inspect any video with the `video_analyze` tool.

## Paste Methods

How you attach an image depends on your terminal environment. Not all methods work everywhere — here's the full breakdown:

### `/paste` Command

**The most reliable explicit image-attach fallback.**

```
/paste
```

Type `/paste` and press Enter. Hermes checks your clipboard for an image and attaches it. This is the safest option when your terminal rewrites `Cmd+V`/`Ctrl+V`, or when you copied only an image and there is no bracketed-paste text payload to inspect.

### Ctrl+V / Cmd+V

Hermes now treats paste as a layered flow:
- normal text paste first
- native clipboard / OSC52 text fallback if the terminal did not deliver text cleanly
- image attach when the clipboard or pasted payload resolves to an image or image path

This means pasted macOS screenshot temp paths and `file://...` image URIs can attach immediately instead of sitting in the composer as raw text.

:::warning
If your clipboard has **only an image** (no text), terminals still cannot send binary image bytes directly. Use `/paste` as the explicit image-attach fallback.
:::

### `/terminal-setup` for VS Code / Cursor / Windsurf

If you run the TUI inside a local VS Code-family integrated terminal on macOS, Hermes can install the recommended `workbench.action.terminal.sendSequence` bindings for better multiline and undo/redo parity:

```text
/terminal-setup
```

This is especially useful when `Cmd+Enter`, `Cmd+Z`, or `Shift+Cmd+Z` are being intercepted by the IDE. Run it on the local machine only — not inside an SSH session.

## Platform Compatibility

| Environment | `/paste` | Cmd/Ctrl+V | `/terminal-setup` | Notes |
|---|:---:|:---:|:---:|---|
| **macOS Terminal / iTerm2** | ✅ | ✅ | n/a | Best experience — native clipboard + screenshot-path recovery |
| **Apple Terminal** | ✅ | ✅ | n/a | If Cmd+←/→/⌫ gets rewritten, use Ctrl+A / Ctrl+E / Ctrl+U fallbacks |
| **Linux X11 desktop** | ✅ | ✅ | n/a | Requires `xclip` (`apt install xclip`) |
| **Linux Wayland desktop** | ✅ | ✅ | n/a | Requires `wl-paste` (`apt install wl-clipboard`) |
| **WSL2 (Windows Terminal)** | ✅ | ✅ | n/a | Uses `powershell.exe` — no extra install needed |
| **VS Code / Cursor / Windsurf (local)** | ✅ | ✅ | ✅ | Recommended for better Cmd+Enter / undo / redo parity |
| **VS Code / Cursor / Windsurf (SSH)** | ❌² | ❌² | ❌³ | Run `/terminal-setup` on the local machine instead |
| **SSH terminal (any)** | ❌² | ❌² | n/a | Remote clipboard not accessible |

² See [SSH & Remote Sessions](#ssh--remote-sessions) below
³ The command writes local IDE keybindings and should not be run from the remote host

## Platform-Specific Setup

### macOS

**No setup required.** Hermes uses `osascript` (built into macOS) to read the clipboard. For faster performance, optionally install `pngpaste`:

```bash
brew install pngpaste
```

### Linux (X11)

Install `xclip`:

```bash
# Ubuntu/Debian
sudo apt install xclip

# Fedora
sudo dnf install xclip

# Arch
sudo pacman -S xclip
```

### Linux (Wayland)

Modern Linux desktops (Ubuntu 22.04+, Fedora 34+) often use Wayland by default. Install `wl-clipboard`:

```bash
# Ubuntu/Debian
sudo apt install wl-clipboard

# Fedora
sudo dnf install wl-clipboard

# Arch
sudo pacman -S wl-clipboard
```

:::tip How to check if you're on Wayland
```bash
echo $XDG_SESSION_TYPE
# "wayland" = Wayland, "x11" = X11, "tty" = no display server
```
:::

### WSL2

**No extra setup required.** Hermes detects WSL2 automatically (via `/proc/version`) and uses `powershell.exe` to access the Windows clipboard through .NET's `System.Windows.Forms.Clipboard`. This is built into WSL2's Windows interop — `powershell.exe` is available by default.

The clipboard data is transferred as base64-encoded PNG over stdout, so no file path conversion or temp files are needed.

:::info WSLg Note
If you're running WSLg (WSL2 with GUI support), Hermes tries the PowerShell path first, then falls back to `wl-paste`. WSLg's clipboard bridge only supports BMP format for images — Hermes auto-converts BMP to PNG using Pillow (if installed) or ImageMagick's `convert` command.
:::

#### Verify WSL2 clipboard access

```bash
# 1. Check WSL detection
grep -i microsoft /proc/version

# 2. Check PowerShell is accessible
which powershell.exe

# 3. Copy an image, then check
powershell.exe -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Clipboard]::ContainsImage()"
# Should print "True"
```

## SSH & Remote Sessions

**Clipboard image paste does not fully work over SSH.** When you SSH into a remote machine, the Hermes CLI runs on the remote host. Clipboard tools (`xclip`, `wl-paste`, `powershell.exe`, `osascript`) read the clipboard of the machine they run on — which is the remote server, not your local machine. Your local clipboard image is therefore inaccessible from the remote side.

Text can sometimes still bridge through terminal paste or OSC52, but image clipboard access and local screenshot temp paths remain tied to the machine running Hermes.

### Workarounds for SSH

1. **Upload the image file** — Save the image locally, upload it to the remote server via `scp`, VSCode's file explorer (drag-and-drop), or any file transfer method. Then reference it by path. *(A `/attach <filepath>` command is planned for a future release.)*

2. **Use a URL** — If the image is accessible online, just paste the URL in your message. The agent can use `vision_analyze` to look at any image URL directly.

3. **X11 forwarding** — Connect with `ssh -X` to forward X11. This lets `xclip` on the remote machine access your local X11 clipboard. Requires an X server running locally (XQuartz on macOS, built-in on Linux X11 desktops). Slow for large images.

4. **Use a messaging platform** — Send images to Hermes via Telegram, Discord, Slack, or WhatsApp. These platforms handle image upload natively and are not affected by clipboard/terminal limitations.

## Why Terminals Can't Paste Images

This is a common source of confusion, so here's the technical explanation:

Terminals are **text-based** interfaces. When you press Ctrl+V (or Cmd+V), the terminal emulator:

1. Reads the clipboard for **text content**
2. Wraps it in [bracketed paste](https://en.wikipedia.org/wiki/Bracketed-paste) escape sequences
3. Sends it to the application through the terminal's text stream

If the clipboard contains only an image (no text), the terminal has nothing to send. There is no standard terminal escape sequence for binary image data. The terminal simply does nothing.

This is why Hermes uses a separate clipboard check — instead of receiving image data through the terminal paste event, it calls OS-level tools (`osascript`, `powershell.exe`, `xclip`, `wl-paste`) directly via subprocess to read the clipboard independently.

## Supported Models

Image paste works with any vision-capable model. The image is sent as a base64-encoded data URL in the OpenAI vision content format:

```json
{
  "type": "image_url",
  "image_url": {
    "url": "data:image/png;base64,..."
  }
}
```

Most modern models support this format, including GPT-4 Vision, Claude (with vision), Gemini, and open-source multimodal models served through OpenRouter.

## Image Routing (Vision-Capable vs Text-Only Models)

When a user attaches an image — from the CLI clipboard, the gateway (Telegram/Discord photo), or any other entry point — Hermes routes it based on whether your current model actually supports vision:

| Your model | What happens to the image |
|---|---|
| **Vision-capable** (GPT-4V, Claude with vision, Gemini, Qwen-VL, MiMo-VL, DeepSeek Flash / V4.1-Flash, etc.) | Sent as **real pixels** using the provider's native image content format above. No text summary layer. |
| **Text-only** (DeepSeek V4 Pro, DeepSeek V3, smaller open-source models, older chat-only endpoints) | Routed through the `vision_analyze` auxiliary tool — an auxiliary vision model describes the image, and the text description is injected into the conversation. |

You don't configure this — Hermes looks up your current model's capability in the provider metadata and picks the right path automatically. The practical effect: you can switch between vision and non-vision models mid-session and image handling "just works" without changing your workflow. Text-only models get coherent context about the image rather than a broken multimodal payload they'd have to reject.

To override the automatic choice, set `agent.image_input_mode` in `config.yaml`:

| Value | Behavior |
|-------|----------|
| `auto` (default) | Native pixels when the model reports vision support, `vision_analyze` description otherwise. Configuring an explicit `auxiliary.vision` backend (a `provider` other than `auto`, or a `model` / `base_url`) also selects the description path, even for a vision-capable main model. |
| `native` | Always attach pixels, even when the catalog says the model is text-only. |
| `text` | Always route images through the `vision_analyze` describer, never attach pixels to the main request. |

This is the knob to reach for when a backend accepts text but rejects native image input (for example an `openai-codex` account whose backend answers image requests with `server_error`): keep your main model and point `auxiliary.vision` at a different vision-capable provider and model (with `auxiliary.vision.provider: auto` the describer would auto-detect the same main model again). That alone switches images to the description path in `auto` mode; `agent.image_input_mode: text` makes the same choice explicit.

Which auxiliary model handles the text-description path is configurable under `auxiliary.vision` — see [Auxiliary Models](../configuration.md#auxiliary-models).

### `vision_analyze` has the same dual behavior

The `vision_analyze` tool itself follows the same routing. When the active main model is vision-capable **and** its provider supports image content inside tool results (currently the Anthropic, OpenAI, Azure-OpenAI, and Gemini 3.x stacks), `vision_analyze` short-circuits the auxiliary describer and returns the raw image pixels as a multimodal tool-result envelope. The main model sees the image natively on its next turn — no aux call, no text-summary information loss, no extra latency. One exception: an image that is already attached natively to the current user message is not re-embedded — `vision_analyze` on that same path returns a short text result saying the image is already in context (pass a `region` to zoom into part of it, which does embed the crop).

For text-only main models (or providers whose tool-result channel doesn't carry images), `vision_analyze` falls back to the legacy path: it asks the configured auxiliary vision model to describe the image and returns the description as plain text. Either way the calling tool signature is the same — the tool decides which path to take at runtime based on the active model.

### SVG and other non-raster images on Responses backends

Responses-style backends (for example `openai-codex`) accept only inline JPEG, PNG, GIF and WebP; any other `data:image/*` part makes them reject the **whole** request, and because the part stays in history every later turn fails the same way. Hermes handles this at the send layer: an inline **SVG** is rasterized to PNG when a rasterizer is installed (`cairosvg`, `svglib`+`reportlab`, `rsvg-convert`, or `inkscape` — the same soft dependencies `vision_analyze` uses), so the model still sees the drawing. Without a rasterizer, an SVG — and any other unsupported inline format such as BMP or TIFF — is replaced by a short text placeholder (`[image omitted: image/svg+xml is not a supported image format]`) while the valid images in the same message are still sent.

### Native embeds ride the session: `vision.embed_target_bytes` and `vision.max_calls_per_image`

A native `vision_analyze` result bakes the image into the tool result, and that result is re-sent on every later API call of the session. Two `config.yaml` keys bound the recurring cost:

```yaml
vision:
  embed_target_bytes: 262144   # per-embed byte budget; clamped 64 KiB..4 MiB (default 256 KB)
  max_calls_per_image: 3       # unset = 3 inside delegated subagents, unlimited for the main agent
```

- **`embed_target_bytes`** — images above the budget (or wider than 1568 px) are downscaled to a JPEG that fits. 256 KB keeps ordinary screenshots cheap; dense phone screenshots of tables can come out unreadable at that size, so raise it (say `1048576`) when the model keeps calling figures "unreadable". Browser screenshots delivered natively use the same budget.
- **`max_calls_per_image`** — how often the *same* image (region crops of it included; local paths compare by resolved path) may be embedded per session. Once the cap is hit the tool returns `"vision_analyze refused: this image has already been loaded into context N time(s) …"` instead of another embed, so the model answers from what it already sees. Left unset, only delegated `delegate_task` subagents are capped (at 3): they run unattended and cannot be steered mid-loop from the CLI, and a re-load loop there once burned 158 calls on five files. Set a number to cap every session, or `0` for unlimited everywhere.

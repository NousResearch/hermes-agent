---
sidebar_position: 3
title: "Hermes Desktop"
description: "The native Hermes desktop app — a polished experience for chatting with Hermes, with streaming tool output, side-by-side previews, a file browser, voice, cron, profiles, skills, and settings. macOS, Windows, and Linux."
---

# Hermes Desktop

The Hermes desktop app is a native app built around the **same** agent you get from the CLI and the gateway — same config, same API keys, same sessions, same skills, same memory. It is not a separate product or a lightweight clone; it uses the same Hermes Agent core and settings, and drives it through a modern & thoughtfully designed UI. If you have used `hermes` in a terminal, everything you set up there is already here, and anything you do here shows up there.

It runs on **macOS (Apple Silicon), Windows, and Linux** — see [Platform Support](../getting-started/platform-support.md) for the full matrix.

:::tip Which interface is which?
Hermes has several front ends that all talk to the same agent:

- **Desktop App** (this page) — a native application with a purpose-built UI for chat, configuration, and management.
- **CLI** (`hermes`) and **[TUI](./tui.md)** (`hermes --tui`) — terminal interfaces.
- **[Web Dashboard](./features/web-dashboard.md)** (`hermes dashboard`) — a browser admin panel; its optional **Chat** tab embeds the TUI through a pseudo-terminal.

Pick whichever fits the moment. They share state, so you can start a session in one and resume it in another.
:::

## Install

Download the app from the [Hermes Desktop product page](https://hermes-agent.nousresearch.com/desktop), or follow the [installation instructions for Hermes Desktop](../getting-started/installation.md).

If you already have Hermes installed, simply run

```bash
hermes desktop
```

That uses your current config, keys, sessions, and skills.

## What's in the app

The desktop app is organized as a chat-first window with a left sidebar for navigation. It's built to allow managing multiple simultaneous agent conversations, configuring messaging providers, creating artifacts, browsing projects' folder structures, and working on multiple projects at once.

Sidebar selection follows the focused chat pane. Opening or focusing a session tab clears a page's highlight, including contributed pages such as Kanban, even when the workspace retains that page's route.

### Chat

The center of the app. You get:

- **Streaming responses** with live tool activity and structured tool-call summaries as the agent works. When a tool works on an image (for example `vision_analyze` on a file under the backend's filesystem), expand its activity row to see the image and click it to open it full-size; on a remote connection the image is fetched from the gateway, not from the machine running the app.
- **Markdown line breaks** follow Markdown semantics: two trailing spaces create a hard line break; an ordinary newline stays a soft break. Media and preview extraction preserve text outside removed attachment spans, including first-line code indentation and unfinished fenced-code spacing. Code display and Copy preserve leading blank lines, trailing spaces, and terminal blank lines from the Markdown parser.
- **The same conversation history** as every other Hermes surface — sessions started here resume in the CLI/TUI and vice versa.
- **Drag-and-drop files** anywhere in the chat area to attach them to your next message.
- **Independent background drafts** — hidden chat tabs can update their drafts without moving the caret or selection in the visible composer.
- **Unsent drafts survive a lost session** — a draft you typed into a chat whose session no longer exists (deleted elsewhere, or a stale tab after a profile rename or wiped backend) is carried into the fresh chat the app falls back to, with an inline **Restored your unsent message** strip above the input. **Undo** puts the text back where it was; nothing is sent, navigated, or focused on your behalf, and the strip appears once per lost draft.
- **Directive chip actions** — hover an actionable reference (such as a URL) to reveal its action pill. A short grace period lets you move from the chip to the pill before it dismisses. The pill stays available while you move within it; after leaving, unrelated pointer movement does not delay dismissal. Clicking its action preserves the draft selection.
- **A right-hand preview rail** — render web pages, files, and tool outputs side by side while you keep chatting.
- **Hide vs. Close for stateful panes** — a zone holding a Browser (or HTML preview) or the Terminal pane offers **Hide** instead of Minimize. Hiding collapses the zone to its restore rail but keeps the pane's body mounted: a live page keeps its unsaved form input, timers, scroll position, and the agent's `drive_preview` automation target; a terminal keeps its running shell and scrollback; and **Restore** shows the same content without a reload. The hidden body is inert — it never takes keyboard shortcuts or steals composer focus. **Close** (the tab's ×) is what actually releases the page or shell.
- **Comment mode in the in-app browser** — click **Annotate** in the preview browser bar, then click any element (or drag a box) on the live page and type a note; each saved comment stays as a numbered pin on the page. Saving a pin never sends a turn — when you're done, **Add N comments** attaches a cropped screenshot per pin and a short prompt naming each comment to the composer, and you still hit send yourself. Each element comment carries its CSS selector, its markup, and the computed styles that matter for layout, so the agent can find the element in your source instead of guessing from the picture. Password and hidden field values, and any attribute that looks like a key or token, are redacted on the page before the markup leaves it. Larger batches arrive grouped by which part of the page each comment sits in, so twenty-odd comments become a handful of pieces of work rather than one task each — and because the groups are separate DOM subtrees they usually touch separate files, which is what makes handing them to parallel workers safe. Pin numbers hold steady if you delete one, and switching chats clears the stack.
- **Composer history and queue editing** — press the up/down arrow keys in an empty composer to recall and reuse previous prompts, and edit messages you've queued up before they're sent. Pressing Stop (or Esc) while turns are queued pauses the queue and expands it above the composer; resume it from there, or send, edit, and delete individual entries.
- **Task progress above the composer** — expand the Tasks header to inspect each phase. Long lists stay bounded above the input; scroll inside the expanded list to reach the final tasks without moving the conversation.
- **A conversation timeline rail** — long chats get a slim rail of markers along the edge of the transcript, one per prompt. Hover it to pop open the list of prompts, click one to jump straight to that point in the conversation. After a jump, **Show earlier** at the top of the transcript keeps paging backwards from that prompt all the way to the start of the session. (It appears once the chat has a handful of turns.)
- **Reading-position memory** — returning to a session restores its saved distance from the bottom instead of always jumping to the latest message. Sessions left at the bottom continue following new output. Use **Scroll to bottom** to return to the live edge. Positions are kept in this Desktop installation's local storage; they are not synchronized through the backend.
- **Find in page** — press **Cmd/Ctrl+F** to open a find bar that searches the rendered chat transcript. Enter / Shift+Enter (or Cmd/Ctrl+G / Cmd/Ctrl+Shift+G while the bar is open) step through matches; Esc closes it.

Async cron and delegation completions appear as collapsed timeline disclosures. Open the completion label to read the result body (including job output) as Markdown; long reports scroll within the disclosure. Task instructions and delivery envelopes are not shown as report content.

#### Status bar

The bar along the bottom of the chat shows live session state and exposes quick controls without opening Settings:

- **Per-session YOLO toggle** — flip YOLO on or off for just this session (matching the TUI). YOLO bypasses the dangerous-command approval prompts, so know what you're turning off — see [Security → YOLO Mode](./security.md#yolo-mode).
- **Context-usage meter** — a live "% full" meter of the session's context window. Click it to open the **Context Usage** popover with a token breakdown by category (system prompt, tool definitions, skills, memory, rules, MCP, subagent definitions, and the conversation itself) so you can see exactly what's eating the window before compression kicks in.
- **Cache hit rate and tokens per second** — off by default; turn them on from the right-click menu. Cache hit rate is the share of this session's prompt tokens served from the provider's prompt cache (cached tokens cost less, so higher is cheaper — you can watch a session get cheaper as it warms up). Tokens per second is output throughput averaged over the last 10 model calls. Both update live during a turn.
- **Customizable items** — right-click the status bar (**Show in status bar**) to choose what appears: the context meter, cache hit rate, tokens per second, workspace, model, approvals, turn/session timers, terminal, Command Center, backend version, and more — or hide the bar entirely (**Cmd/Ctrl+Shift+S** toggles it). The workspace item's menu offers **Open containing folder** only when the focused session runs on this computer; a session on a remote gateway keeps its folder on that machine, so use **Reveal in filetree** instead. When the OS file manager cannot find a path, the app says so instead of doing nothing.

Chatting against a Hermes instance on another machine instead of the bundled local backend? See [Connecting to a remote backend](#connecting-to-a-remote-backend) below — and for the full picture of how the remote-hosted dashboard connection works (the auth gate, the `/api/ws` chat socket, and WebSocket close-code triage), see [Web Dashboard → Connecting Hermes Desktop to a remote backend](./features/web-dashboard.md#connecting-hermes-desktop-to-a-remote-backend).

#### Fonts

Two independent font settings live in **Settings → Appearance**, both stored per profile in `config.yaml`:

- **Chat Font** (`desktop.font_family`) — chat and the rest of the app's UI. Readability faces such as OpenDyslexic or Atkinson Hyperlegible work as soon as they are installed on the system; the active theme's stack stays behind your pick so missing glyphs still render. Blank means the theme's font.
- **Terminal Font** (`terminal.font_family`) — the embedded terminal pane; Nerd Fonts render shell icons here. Blank means the bundled JetBrains Mono.

#### Repository discovery

Hermes Desktop discovers local Git repositories for the Projects sidebar by scanning your home directory to a bounded depth. You can change this per profile in **Settings → Workspace**, or in `config.yaml`:

```yaml
desktop:
  repo_scan_enabled: true
  repo_scan_roots: []
  repo_scan_exclude_paths: []
```

- Set `repo_scan_enabled: false` to stop the filesystem scan completely. Existing disk-discovery cache rows for that profile are cleared; explicit projects and repositories inferred from intentional Hermes sessions remain available.
- Set `repo_scan_roots` to a list of folders to restrict scanning. An empty list preserves the default home-directory scan.
- Set `repo_scan_exclude_paths` to folders whose complete subtrees should be skipped.

Changing any of these values invalidates only that profile's disk-discovery cache and starts a policy-compliant refresh. **Hide from sidebar** remains a separate per-item curation action.

With **Group by → Projects**, each project row previews its three most recent sessions. A project with more ends in a **Show all N sessions** row that expands the rest in place; the sidebar menu's **Show all sessions** toggle removes the preview cap for every project at once.

#### Choosing a model

The model picker lives in the **composer**, just left of the microphone. Click it to switch the model; hover a model row for its options (thinking, effort, fast). Next to it, a **reasoning pill** shows the active model's effort level (`Med`, `High`, …) and opens the same options directly, so you can change effort without finding the model's row. The pill is hidden for models whose catalog reports no reasoning control. When the route clamps a Hermes-internal step (`ultra` is sent as the route's strongest level, e.g. `max`), the pill shows both ends (`Ultra→Max`) and its tooltip spells out the same wording as the CLI, `Ultra (sends Max on this route)`, so the level you see is the level that is sent. When the gateway flags a switch as risky (a large cached context, an expensive model, a data-training tier), the app asks first in a dialog: **Switch anyway** applies it, **Keep current model** (or Esc) leaves everything as it was.

The **microphone** is dictation; hover it and the other voice toggles fan out above it — **Read replies aloud** and the **wake word** ear. A toggle that is on shows as a solid disc. Starting a full voice conversation stays on the primary button to the right. In the HUD and in narrow tiles the same controls fold into one menu behind the mic instead. When dictation talks to the speech-to-text provider directly (client-direct voice), the request honours the same `stt.openai.timeout` budget (default 60 s) as the gateway's own transcription client, so a slow endpoint fails with "Transcription timed out" instead of leaving the mic stuck on transcribing.

- **The composer picker is sticky UI state and never touches your default.** It's remembered locally (per device) and **follows** across new chats and restarts instead of snapping back to the default — pick a model once and the next `Cmd/Ctrl+N` opens on it. With a live chat, switching models scopes the change to that **current chat**; either way the selection rides along when the session is created/switched and is **never** written to the profile default — with one exception: on a fresh profile that has no `model.default`/`model.provider` configured yet, the first pick is persisted so the app has a real default instead of falling through to a stray API-key env var on restart. Persistence follows the same rule as `/model` (`model.persist_switch_by_default`); use **Settings → Model** to change the default deliberately. (Switching [profiles](#sessions--profiles) reseeds to that profile's own default.)
- **Set the default in Settings → Model.** That "main" model is your **per-profile global default** — it's what new chats, crons, subagents, and auxiliary tasks start from, and it's the only place that writes it. Each [profile](#sessions--profiles) keeps its own default.
- **Per-model effort/fast presets.** Each model remembers its own reasoning effort and fast-mode choice in the desktop app, re-applied to the session whenever you pick that model. These presets are a desktop convenience and don't change crons or subagents.
- **Mid-chat switches reset the prompt cache.** Switching the model inside a live chat means the next message re-reads the whole conversation at full input price (provider prompt caches are keyed to the model). Fine occasionally; on a long chat, a fresh chat on the new model is often cheaper than bouncing back and forth.

### File browser

Explore and preview the working directory without leaving the app — useful for following along as the agent reads, writes, and edits files. Set the initial project directory with `hermes desktop --cwd <path>` (or the `HERMES_DESKTOP_CWD` environment variable).

### Artifacts

Preview links above the composer are session suggestions, not a task-completion checklist. Dismissing one keeps historical tool rows from bringing it back after navigation or reload. A new successful tool completion can offer the file again. Read-only file inspection and failed writes do not create suggestions. Files with the same name show enough directory context to distinguish them; dismissing a suggestion does not delete its file or transcript. Changing a `/goal` does not erase a conversation's artifacts.

When connected to a remote gateway, opening a file artifact downloads it through that gateway, using the artifact’s originating profile and session. Relative paths resolve against the session’s saved working directory; home-relative paths use the gateway’s home, never the Desktop machine’s home. Windows-style relative paths are recognized alongside forward-slash paths, and file URIs retain drive and network-share information for the gateway to interpret. Missing sessions or working directories produce an error rather than selecting a different local file.

The **Artifacts** view collects what your sessions generate — **images, files, and links** — into one searchable, browsable gallery. Open it from the sidebar, the command palette (**Artifacts — Browse generated outputs**), or a `nav.artifacts` shortcut you bind yourself. It indexes recent session outputs automatically; every artifact shows which session produced it with a jump back to that chat, and images and files open in a preview with download / open-in-browser / copy actions.

### Windows, tabs & panes

The app is built for working on several things at once:

- **Tabs** — **Cmd/Ctrl+T** opens a new session tab; **Ctrl+Tab** / **Ctrl+Shift+Tab** cycle sessions, and **Ctrl+1…9** jump to a recent session by position. **Cmd/Ctrl+W** closes the focused tab and **Cmd/Ctrl+Shift+T** reopens the last closed one.
- **Multiple windows** — **Cmd/Ctrl+Shift+N** opens a new window, and any session can be popped out via its context menu (**New window**) or from the command palette. A popped-out window renders that single chat without the global sidebar — handy for parking a long-running session on another monitor. Live agent output streams into every window showing the session.
- **Panes** — **Cmd/Ctrl+B** toggles the left sidebar, **Cmd/Ctrl+J** the right one, and **Cmd/Ctrl+\\** swaps which side the sidebars sit on.

#### Interface mode

The layout editor (titlebar button, or **Cmd/Ctrl+Shift+\\**) opens with an **Interface mode** choice — also under **Settings → Appearance → Window & layout** and as *Simple mode* in the command palette. It changes what is shown, not what Hermes can do.

- **Advanced** (default) is the app as you have set it up. Users who have not explicitly selected Simple stay in Advanced.
- **Simple** is chat-first: the statusbar, profile rail, terminal, file browser and review panes, the technical tool-call view, inline code diffs, and the Artifacts / Scheduled jobs rows rest out of the way (Capabilities and Messaging stay — they are how you set Hermes up); thinking starts collapsed; session rows show the title, a preview and when they were last active. The titlebar keeps Settings and the layout editor. Simple's layout picker offers *Sidebar left* or *Sidebar right*. The templates and saved layouts are Advanced.

A layout says what is on screen, not just where things sit: applying one opens every pane it places and closes the ones it leaves *resting*, so **Ctrl+`**, **Cmd/Ctrl+J** and **Cmd/Ctrl+G** always agree with what you see. *Basic* is sessions and chat with the terminal resting as a collapsed rail under the chat and the file browser and review resting in a right column — **Ctrl+`** opens the terminal under the chat, **Cmd/Ctrl+J** opens the tree on the right. *Focus* keeps files and review as tabs behind the chat, with the same terminal rail. *Default*, *Terminal deck* and *Quad* open everything they place. A layout you save remembers which of its panes were closed.

Each mode remembers its own arrangement: pane positions, sizes, active tabs, hidden tabs, dismissals, collapsed sides and floating-card positions. Returning to a mode restores that arrangement rather than reapplying a preset. Simple starts with the sidebar on the left; moving it to the right does not change Advanced. Conversations, drafts, previews and running work stay shared.

Existing layouts are retained on upgrade. Advanced continues using the original storage keys. If you already explicitly selected Simple, its current layout is copied into Simple's separate storage without deleting the originals. An older arrangement that was overwritten before this separation cannot be reconstructed.

Simple shadows your display preferences instead of overwriting them. Every keybind still works in Simple — **Ctrl+`**, **Cmd/Ctrl+J** and **Cmd/Ctrl+G** open the terminal, file browser and review for the current session, and the next launch is quiet again. With more than one profile the profile rail stays, since it is then the only way to switch. First-run onboarding sets the mode from the layout you pick: *Basic* starts in Simple, *Elite* in Advanced; skipping leaves it on Advanced.

#### Minimize to tray

Enable **Settings → Appearance → Window layout → Minimize to tray** to hide minimized windows from the taskbar or Dock while their sessions keep running. The setting is off by default and applies only to this device.

With the setting enabled and a tray available, closing the main window with **X** or **Alt+F4** hides it without stopping Hermes or destroying its session. Closing secondary windows still closes those windows. Use **Show Hermes** in the system tray (the menu bar on macOS) to restore hidden windows. **Quit Hermes** from the tray menu and **Cmd+Q** still quit, including the normal active-work confirmation. If the tray is unavailable, closing the main window behaves normally.

On macOS, the Dock icon hides only when no ordinary Hermes window remains visible. On Linux, a registered StatusNotifier tray host is required; desktops without one keep normal minimize behavior. If the host disappears, hidden windows are restored.

### Terminal

A real terminal lives in the right sidebar, next to the file browser:

- **Ctrl+`** shows the terminal (opening one if none exist); **Ctrl+Shift+`** spawns an additional one. Multiple terminals stack in a tab rail — **Ctrl+Shift+↓/↑** walk between them, **Ctrl+Shift+W** closes the active one.
- **Shells persist while hidden.** Closing or hiding the panel doesn't kill your shell — every open terminal stays mounted with its scrollback and running processes intact until you explicitly close it.
- **Add to chat** — select terminal output and send it into the composer as context for your next message.

### Live subagents

While delegated workers are live, a **Subagents** frame appears above the composer with their count, task names, elapsed time, and latest activity. It previews up to three workers; expand the header for the roster, then select a worker for details and **Steer** / **Stop** controls. Each frame belongs to its chat, including in split panes. Steering acknowledges that guidance is queued for a checkpoint, not that the child has already read it. See [Monitoring subagents](./features/delegation.md#monitoring-running-subagents-agents).

### Git review & worktrees

For sessions running inside a Git repository, the app has a built-in source-control surface:

- **Review pane** — **Cmd/Ctrl+G** toggles the working-tree review pane: branch and ahead/behind status, changed files (list or tree view), and diffs scoped to **Uncommitted**, **Branch**, or **Last turn** (just what the agent changed in its most recent turn). Stage/unstage files, revert changes, write a commit message (or **Generate commit message**), then **Commit** or **Commit & Push** — and **Create PR** via the GitHub CLI (`gh`), or hand the whole thing to the agent with **Ask Hermes to open PR**. You can also create and switch branches from here.
- **Worktrees** — **Cmd/Ctrl+Shift+B** (or **New worktree** on a project in the sidebar) creates a Git worktree on a new branch so an agent can work on a parallel copy of the repo without touching your checkout. Worktrees show up as their own lanes under the project; removing one offers to delete the worktree directory (the branch stays) or just hide the lane and leave it on disk, with a force option when it has uncommitted changes.

### Memory Graph

The **Memory Graph** (command palette → *Memory Graph*, or the status-bar item) is an interactive map of what Hermes has learned for you — skills and memories laid out as a zoomable node graph with a timeline, filterable by **All / Used / Learned**. A share control exports the map layout as a compact code you can paste to someone else (layout only — none of your memory or skill text is included) and imports codes the same way.

### Quick Entry

Quick Entry is a small always-available composer summoned by a **global hotkey from anywhere on your system** — fire off a prompt without switching to (or even opening) the main window. Enable it in **Settings → Advanced → Quick Entry**; the default shortcut is **Ctrl/Cmd+Shift+Space** and you can set your own (it needs at least one modifier). If another app already owns the chord, the settings row tells you so you can pick a different one.

### Voice

Talk to Hermes and hear it back, the same [voice mode](./features/voice-mode.md) available elsewhere. On macOS the OS will prompt once for microphone access.

### HUD mode

**⌘/Ctrl+Shift+H** (or the titlebar button) detaches the chat into a chrome-free, always-on-top floating bar that sits over whatever you are working in. The app window steps aside; the HUD keeps your live conversation and a composer. Where you park it is context — the bar's position tells Hermes which app and screen you're asking about, so "this", "here", and "that page" resolve to what's underneath it.

- **Moving the bar** — on macOS and Windows, **press and hold** anywhere on the composer for a beat, then drag. On Linux/X11, hold **Ctrl** and drag with the primary mouse button for an immediate grab (including over selected text); press-and-hold remains available too. Keep the grab held while invoking your desktop switch shortcut to carry the HUD onto another virtual desktop. On native Wayland the composer bar is a compositor drag handle (the only way to move it, because an app cannot place its own window).
- **Resizing** — drag any edge or corner of the bar; the opposite edge stays anchored. Native Wayland exposes the right and bottom edges because the compositor does not allow apps to position top-level windows themselves.
- **Reset layout** — the discard control on the bar restores the default size and (on X11 / macOS / Windows) position. Use this if a persisted size leaves the HUD unusable.
- **Tap to summon** — enable **Tap to summon HUD** under **Settings → Keyboard Shortcuts → HUD gesture**, then tap and release **⌘+Option** on macOS or **Ctrl+Alt** on Windows/Linux X11. Release both keys within half a second without another key or mouse action. On Windows, use left Alt; right Alt is reserved for AltGr layouts. This opens or focuses the HUD from another app; it does not toggle it closed, record audio, or send a message. Off by default and saved only on this device. macOS requires Input Monitoring permission; the page offers recovery controls only when permission or another error needs attention. Linux Wayland does not expose this gesture, so keep using the in-app HUD shortcut there.
- **Snap to pointer** — **⌘/Ctrl+Shift+G** (a global hotkey, works from any app) jumps the HUD to wherever your cursor is. On native Wayland this is a no-op — the compositor owns placement.
- **Exiting** — click the exit button on the bar, press **⌘/Ctrl+Shift+H** again, or press **⌘/Ctrl+W** while the HUD has focus. The app window comes back in front with your session and the caret in its composer.

#### Linux / Wayland

Electron 20+ already runs as a native Wayland client on a Wayland session. Drag, click-through, and resize work on that path.

On **Hyprland** (including Omarchy) the HUD is floated and pinned through the compositor's IPC after it maps — otherwise Hyprland tiles it like any other window, `always-on-top` is ignored, and compositor drag does nothing. No extra window rule is required.

A few compositors (notably COSMIC) ignore `always-on-top` for native Wayland windows. To restore pinning there, run the app under XWayland:

```yaml
desktop:
  ozone_platform_hint: x11
```

That bridges to `ELECTRON_OZONE_PLATFORM_HINT` at launch (an explicit env var still wins). The trade: X11 cannot restore a window that has ignored the mouse, so the HUD stays a solid window instead of click-through. Some KDE setups also report keyboard breakage with the X11 ozone backend — leave the hint on `auto` unless you need always-on-top.

#### WSLg (Windows GPU from WSL2)

Under local WSLg, Hermes launches with `--ozone-platform=wayland` to avoid the XWayland maximized-window offset and shifted mouse hit-testing ([microsoft/wslg#1015](https://github.com/microsoft/wslg/issues/1015)). The platform must be selected at process launch, before Electron loads application JavaScript. Explicit `--ozone-platform=x11` and `desktop.ozone_platform_hint: x11` remain available. The app draws its own minimize, maximize and close controls on WSLg.

When `hermes gui` runs inside WSL2 with `/dev/dxg` present and Mesa's `d3d12_dri.so` installed, the launcher sets `GALLIUM_DRIVER=d3d12` for Electron so rendering uses the Windows GPU instead of the llvmpipe software rasterizer; an explicit `GALLIUM_DRIVER`, `MESA_LOADER_DRIVER_OVERRIDE`, `LIBGL_ALWAYS_SOFTWARE`, or `LIBGL_DRIVERS_PATH` in your environment is left untouched (for example `GALLIUM_DRIVER=llvmpipe hermes gui` keeps software rendering).

#### Launch flags and the renderer heap ceiling

Two `desktop.*` keys reach Chromium at launch on every path — `hermes desktop`, the Start-menu shortcut and the Linux `.desktop` entry alike (the app reads them from `config.yaml` before its first window opens):

```yaml
desktop:
  electron_flags: ["--ozone-platform=x11"]   # extra Chromium switches; a single string is split on spaces
  renderer_max_old_space_mb: 2048            # V8 heap ceiling for the chat renderer; 0 = Chromium default
```

`renderer_max_old_space_mb` is applied as `--js-flags=--max-old-space-size=N` and merged with any `--js-flags` you already pass, so neither overwrites the other. Set it when a very long, tool-heavy session drives the renderer past what the machine can spare: the renderer then hits its own limit and reloads (bounded to three reloads per minute) instead of freezing the whole machine.

Both keys must be written exactly as shown — two spaces of indentation under a top-level `desktop:` key, and four spaces before the `-` of a block list:

```yaml
desktop:
  electron_flags:
    - "--ozone-platform=x11"
    - "--js-flags=--expose-gc"
```

The pre-window reader is a small YAML subset, not the full parser the rest of Hermes uses, because it has to run before the app loads anything. Other indentations are valid YAML but are ignored here; when that happens the app logs `desktop.electron_flags / desktop.renderer_max_old_space_mb were ignored` at startup and launches with Chromium's defaults.

### Settings & onboarding

Manage providers, models, tools, and credentials from a real UI instead of editing YAML. First-run onboarding gets you to your first message in seconds. The settings panes cover providers/keys, model selection, toolset configuration, MCP servers, the gateway, and session management.

- **Providers settings pane** — a dedicated place to manage inference providers, with an Accounts / API-keys UX for signing in and storing credentials per provider. Accounts, API keys, and Custom Endpoints share the Settings **Applies to** selection: credential reads and edits, OAuth account removal, custom-endpoint save/test, and sign-in launched here target the selected profile, not the active chat profile. The sign-in flow keeps that target through credential saving and model selection. Changing **Applies to** discards unsaved credential drafts. Closing sign-in cancels polling and ignores late results; a credential write already sent may still finish in its original profile. Externally managed CLI credentials use their own CLI and are not covered by this profile selector. Its **Local Models** view installs and manages an on-device llama.cpp runtime — see [Local Models](./local-models.md).
- **Every provider and model in the menus** — the GUI surfaces the full provider list and every model that `hermes model` knows about, so you pick from the same catalog the CLI sees rather than a curated subset.
- **Custom endpoints with an API mode** — **Settings → Providers → Custom Endpoints** has an **API Mode** selector (**Auto-detect**, **Chat Completions**, **Responses API**, **Anthropic Messages**) — the same choice `hermes model` offers for a custom provider. It is saved as `providers.<id>.api_mode` in `config.yaml`, so a Responses-only or Anthropic-compatible host is no longer called on `/chat/completions`. **Test** checks the transport you will actually use, not just `/v1/models`: it sends a one-token request to the pinned mode's route (or to the mode Auto-detect resolves to) and fails with the transport named when the host does not serve it. **Test** also keeps the alias metadata a gateway advertises in `/v1/models` (`canonical_model`, `reasoning_effort`): picking an alias such as `gpt-5.6-sol-high` saves the canonical model and pins its effort under `agent.reasoning_overrides`.
- **xAI Grok OAuth** — Grok is a first-class OAuth provider in the launcher; sign in through the browser flow like the other OAuth providers.
- **Tool-backend installs from the GUI** — run a tool backend's post-setup install steps directly from the app instead of dropping to a terminal. In the terminal backend picker, selecting a backend marked **Needs setup** asks for confirmation first; declining leaves the current backend selected.
- **Terminal font picker** — choose an installed font in **Settings → Appearance**. Nerd Fonts such as `MesloLGS NF` render Powerlevel10k separators and icons in both interactive and agent terminals; the setting is saved per profile.
- **Reasoning Blocks** — **Settings → Chat → Reasoning Blocks** (`display.show_reasoning` in `config.yaml`) shows or hides the model's thinking in the transcript (off shows answers only). Open chats update as soon as the setting saves. Typing `/reasoning hide` or `/reasoning show` in the composer flips the same setting and the open transcript follows immediately.
- **Reopen Last Chat on Launch** — by default the app picks up where you left off on cold start. Turn it off in **Settings → Appearance** (or set `display.resume_last_session: false` in `config.yaml`) to always begin with a fresh chat. Deep links and explicit destinations are never overridden either way.
- **Auxiliary-model warning** — if you switch the main model to a new provider while auxiliary tasks (titling, summarization, and similar helpers) are still pinned to another provider, the app warns you so you don't unknowingly split work across two providers.
- **Per-task reasoning effort** — each row under **Settings → Model → Auxiliary models** has a reasoning selector next to its provider/model pick: a level, **Off**, or **inherit · main model effort** (the default, which removes the task's override). It is saved as `auxiliary.<task>.reasoning_effort` in `config.yaml`, the same key `hermes model` writes, and shows in the row's summary when set. Use it to run frequent helpers such as compression or titling at low or no reasoning while the main agent stays at high.
- **VS Code Marketplace themes** — beyond the built-in theme presets, the appearance settings include a live VS Code Marketplace search: pick any color theme and the app downloads, converts, and installs it as a desktop theme. The same importer is available from the command palette (*Install theme*), and imported themes can be removed again from the appearance settings.
- **Keep computer awake** — **Settings → Advanced → Keep computer awake** stops the machine from sleeping so long or overnight agent runs keep going (the display can still dim). This is a per-computer setting.

First-run onboarding has been redesigned on a unified overlay design system, and you can pick **Choose provider later** to skip provider setup and get into the app first.

#### Per-profile settings: the "Applies to" scope

When you have two or more [profiles](./profiles.md), the config-backed settings pages — **Model, Workspace, Safety, Memory & Context, Voice, Chat, Advanced, and Tools & Keys** — plus **Providers → Custom Endpoints** and the **Messaging** overlay show a shared **Applies to** chip row at the top. It selects which profile your edits target:

- The default selection **follows the active profile**, which behaves exactly as before — edit the profile you're using.
- Pick another profile to view and edit *its* settings without switching the whole app; the selection persists as you move between settings pages.
- Switching the app's active profile resets the selector, so edits can't silently keep landing on a previously selected profile.
- With fewer than two profiles the chip row is hidden entirely.

(The Gateways page handles profiles differently — via its **Per-profile overrides** subsection — and the Capabilities and Scheduled Jobs views have their own scope selectors.)

### Management panes

The app also surfaces the broader Hermes management surface so you don't have to drop to a terminal:

- **Skills** — open **Capabilities → Skills** to manage [skills](./features/skills.md). **Installed** shows the selected profile's actual skills and enable/disable state. **Browse** searches the same full published catalog as the public Skills Hub, with native cards and details.
- **Plugins** — **Capabilities → Plugins** uses the same **Installed / Browse** layout. Installed combines actual app-level desktop plugins with agent plugins from the selected profile; Browse shows the public [Plugin Catalog](./features/plugin-catalog.md). Search stays at the top, and the tab switch and actions share one row on both pages.
- **Memory graph (Star Map)** — type `/journey` (aliases `/learning`, `/memory-graph`) in chat to open an interactive constellation of learned skills and memories over time, with a playback scrubber. Nodes can be edited or deleted right from the panel (skills are archived, memories removed). See [Learning Journey](./features/memory.md#learning-journey-journey).
- **Cron** — view and manage [scheduled jobs](../reference/cli-commands.md#hermes-cron). With **All profiles** on, the list aggregates every profile's jobs; a job's run history and actions (pause, resume, edit, delete) always go to the profile that owns the job, whichever profile is active.
- **Profiles** — switch between [Hermes profiles](./profiles.md) (isolated config/skills/sessions).
- **Messaging** — set up gateway channels. Telegram has a **Quick setup** card: click **Create with QR**, scan the code (or open the link) in Telegram, and Hermes creates the bot, detects your user ID for the allowlist, saves the credentials, and restarts the gateway for you. Any credential save, clear, or enable toggle keeps a **Restart now** banner on the page until the gateway has actually restarted; if a restart fails, the banner stays so you can retry or restart manually.
- **Agents** and **Command Center** — orchestration surfaces for multi-agent work.

#### Where Browse gets its data

These are native Desktop views, **not embedded website pages**. Desktop and
the public website consume the same generated CDN snapshots:

| Catalog | Public docs alias | Desktop fetch URL |
|---|---|---|
| Skills | [`/docs/api/skills.json`](https://hermes-agent.nousresearch.com/docs/api/skills.json) | `https://nousresearch.github.io/hermes-agent/docs/api/skills.json` |
| Plugins | [`/docs/api/plugins.json`](https://hermes-agent.nousresearch.com/docs/api/plugins.json) | `https://nousresearch.github.io/hermes-agent/docs/api/plugins.json` |

The skills snapshot combines `skills/`, `optional-skills/`, and the centralized
skills index. The plugin snapshot comes from `plugin-catalog/*.yaml` and cached star
counts; the same publish supplies the installer's removed-entry list. Browsing does not make live
GitHub API calls or fetch plugin/skill source repositories. **Installed** is
separate: its state comes from the selected profile's backend and the app's
desktop-plugin registry, not those public snapshots.

The public hubs' install buttons open `hermes://skill/install?identifier=...`
or `hermes://plugin/install?catalog=...` links and require confirmation in Desktop.
Use an updated Desktop build for these routes; the cards retain copyable CLI
commands if the app is missing or too old. See
[skill links](./features/skills.md#install-from-the-website) and
[plugin links](./features/plugins.md#one-click-install-links-desktop) for the
parameters and review flow.

### Bot Mode (built in)

**Bot Mode** ships with the app and is on by default: a "one chat per agent"
roster where every [Hermes profile](./profiles.md) appears as a bot with its
own avatar (geometric face, uploaded image, AI-generated portrait, or a pixel
pet), its own canonical **Bot Chat** conversation, and its own **Routines**
(recurring tasks backed by Hermes cron). The roster lives in the left
sidebar as a tab next to your conversations — a **Sessions | Bots** tab
strip — rather than a second pane stacked below the session list. Installs
that picked up the older stacked layout are re-homed into the tab strip
automatically, once; if you've hand-placed panes yourself, your layout is
left alone. The **Cronjobs** (Routines) pane docks beside the chat only
while the Bots tab is active and disappears when you switch back to
Sessions (older desktop builds keep it always visible).

Create new agents from the roster —
Name / Title / Description plus an Advanced disclosure with the full
capabilities surface (model, SOUL, skills, toolsets, MCP servers) — group
them into sections, and open group chats where several bots deliberate.
Group chats appear as standalone Discord-style rows in the roster — stacked
member avatars, member count, a preview of the latest room line, and the
"needs you" badge — interleaved with the bot rows in the same pin+recency
ordering. Clicking a group row opens the room as a tab that takes over the
**main chat window** (older desktop builds fall back to opening it inside the
bots side panel).

Bots message each other: type `@researcher have a look at this` in any chat
and the active bot hands the message off and reports back, and bots reach
each other's Bot Chats directly (`hermes -p <bot> chat`). The backend teaches
each bot's canonical **Bot Chat** session the messaging protocol
automatically (config `agent.bot_mode_protocol`, default on) — including
when a teammate bot opens it headlessly from the CLI — so bot-to-bot
replies and handoffs work without touching your SOUL.md, and your regular
sessions stay untouched.

Bot Mode's sessions — each bot's canonical Bot Chat and every group-chat
member session — are always hidden from the global Sessions sidebar. They
live in the Bots pane (roster rows, room views, and each bot's session
browser) instead of interleaving with your own conversations.

Bots you don't use can be tucked away: right-click a bot row → **Hide
Bot**. Hidden bots leave the roster but keep working — @mentions still
resolve and group-chat membership is untouched. An eye toggle appears in
the Bots header whenever at least one bot is hidden; click it to reveal
hidden bots dimmed in place (right-click → **Unhide Bot** brings one back),
and the eye shows a dot when a hidden bot has unread activity. Hidden
state is stored in the bot's profile, so it follows the bot across
machines.

Don't want it? Flip its **Desktop** switch off in **Capabilities → Plugins → Bots** — the roster,
routines pane, and composer middleware unregister live, no restart needed.

Full guide — creating agents (including the multi-machine **Create on**
picker), the roster across connections, bot-to-bot mentions, and how group
chats decide who replies: [Bot Mode: A Roster of Agents](./bot-mode.md).

### Keyboard & navigation

- **Command palette** — press **Cmd+K** or **Cmd+P** (Ctrl+K / Ctrl+P on Windows/Linux) to jump to actions and navigate the app from the keyboard: open any page or settings section, jump to a session by title or id, switch model/theme/color mode, spawn a terminal, restart the gateway, update Hermes, and more.
- **Rebindable shortcuts** — **Settings → Keyboard Shortcuts** (or **Cmd/Ctrl+/**) opens the shortcuts panel where you can remap almost every binding — profile switching, session navigation, view toggles, and any shortcuts contributed by desktop plugins. Duplicate assignments are flagged as conflicts. A few defaults worth knowing: **Cmd/Ctrl+N** new session, **Cmd/Ctrl+.** Command Center, **Cmd/Ctrl+,** Settings, **Cmd/Ctrl+Shift+F** search sessions, **Cmd/Ctrl+1–9** switch profiles, **Shift+X** toggle light/dark.
- **Custom zoom shortcuts** — zoom the interface in half-step increments for finer control over text size.
- **UI language switcher** — change the app's interface language in-app: English, Simplified Chinese (zh-Hans), Traditional Chinese (zh-Hant), Japanese, Arabic (RTL), and Russian.

### Sessions & profiles

- **Session-list overhaul** — a reworked session list with archiving and general session hygiene to keep the list manageable as it grows.
- **Search sessions by id** — find a specific session directly by its id.
- **Concurrent multi-profile sessions** — run sessions across multiple [profiles](./profiles.md) at the same time, and reference a session in another profile with cross-profile `@session` links.
- **Export / import a profile** — share a whole setup as a single file. **⌘K → Export profile…** (or right-click a profile square in the rail) writes a `.tar.gz` with skills, memory, persona, crons, plugins, and settings; API keys are stripped. Exporting from the desktop also bundles your appearance and interface — skin, light/dark mode, custom themes, the profile's rail color, and your window layout — so an imported profile arrives looking the way the sender had it. Import via **⌘K → Import profile…** or the button beside the rail's **+**; it applies the overlay and drops you into the new profile. The same archive works with `/export` / `/import` in chat and `hermes profile export` / `import` from a shell. See [Export and import a profile file](./profile-distributions.md#export-and-import-a-profile-file).

## Updating

The app checks for updates in the background and offers a one-click update when one is ready.

The background check asks the GitHub API for the branch tip. Anonymous GitHub requests are limited to 60 per hour **per network address**, so on a shared connection (office NAT, VPN, proxy) the check can report `GitHub API rate limit reached` even though this machine made almost none of them. To spend a 5,000/hour budget instead, the check uses the first credential it finds, in this order:

1. `GITHUB_TOKEN`, then `GH_TOKEN`, from the environment the app was launched from — read on each request, never stored.
2. The [GitHub CLI](https://cli.github.com/)'s own login (`gh auth token`). This is the rung that helps an app started from the Dock, Finder, or a desktop launcher, which inherits a minimal environment without your shell's variables. `gh` is looked up on `PATH` and in the usual install locations (Homebrew, `/usr/local/bin`, `~/.local/bin`, the Windows GitHub CLI installer); the answer is cached until the app restarts, so `gh` runs at most once per session.
3. Anonymous.

A credential GitHub rejects (HTTP 401 — expired or revoked) is logged once in `desktop.log`, naming its source and never the token, and the request is retried anonymously. Applying an update uses `git`, not the API, and is unaffected.

During a local update, detailed build output streams into the active profile's
`logs/update.log`, including detached `--gateway` updates. It stays out of the
terminal but is available for troubleshooting before the build finishes. The
Windows hand-off counts new output in this log as progress; a child that produces
no output is still subject to the idle watchdog. Process liveness alone does not
reset that watchdog, and cancelling an update does not wait for its build to finish.

The desktop app and the Hermes backend it talks to update on separate clocks — the app package on your machine, the backend wherever it runs. When more than one update target exists (a remote gateway, or several registered gateways), the update affordances (**Update now** on the About panel, the ⌘K **Update Hermes** row, and the update-ready toast) update **everything**: the connected backend first, then every other eligible registered gateway (Hermes Cloud entries are platform-managed and skipped), and the desktop app itself last, since applying the client update relaunches the app. Single-machine installs keep the one-button experience.

After any backend update, the app also re-checks its own version and warns with a one-click **Update desktop app** action if the GUI is still behind — so updating a remote backend can never silently leave you on a stale desktop build.

The [manual update process](https://hermes-agent.nousresearch.com/docs/getting-started/updating) also works with the GUI.

## Uninstalling

For installations managed by the app, open **Settings → About → Danger zone** and pick how much to remove:

- **Uninstall Chat GUI only** — removes the desktop app and its data; the Hermes agent, your config, and your chats stay. (Same as `hermes uninstall --gui`.)
- **Uninstall GUI + agent, keep my data** — removes the app and the agent but keeps config, chats, and secrets for a future reinstall. (Same as `hermes uninstall`.)
- **Uninstall everything** — removes the app, the agent, and all user data. (Same as `hermes uninstall --full`.)

The app closes to finish the job (the cleanup runs after it exits so it can remove the running app bundle and its own venv). The agent-removing options are hidden automatically when no local agent is installed.

These controls are hidden for Nix, bundled/Light packages, and other externally owned installations. Remove those through their package manager or the operating system instead. The app checks its own local package ownership, independently of a remote backend's update status. If ownership cannot be confirmed, no uninstall actions are offered.

For self-managed installations, you can do the same from the terminal — `hermes uninstall --gui` for the GUI alone, or `hermes uninstall` / `hermes uninstall --full` for the agent too.

:::note
Running `hermes uninstall --gui` from a **source checkout** (a `hermes desktop` dev build) also removes the workspace `node_modules` and `apps/desktop/{dist,release}` build output, since those are GUI build artifacts. They're recoverable with `hermes desktop` (or `npm install` + a rebuild) — but if you're actively hacking on the desktop app, expect to reinstall dependencies afterward.
:::

## CLI reference: `hermes desktop`

To launch via the CLI, simply run `hermes desktop`. By default it installs workspace Node dependencies, builds the current OS's unpacked Electron app, then launches that packaged artifact.

On Linux, launches refresh `$XDG_DATA_HOME/applications/hermes.desktop` (by default `~/.local/share/applications/hermes.desktop`) so Hermes appears in the application menu. To preserve a hand-edited entry, disable refreshes:

```bash
hermes config set desktop.manage_launcher_entry false
```

A missing entry is still created; the flag only stops `hermes desktop` from rewriting an entry that already exists.

When you start Hermes from the application grid or menu (the launcher sets `DESKTOP_STARTUP_ID`), the entry is written only after the window is on screen. If the app exits before a window appears, nothing is written that time; the next terminal launch, updater relaunch, or grid launch that shows a window installs it. Some GNOME Shell versions lose track of an app whose `.desktop` file changes while it is still starting (they keep it in that state until the startup notification completes or times out, not until the process exits), and that can crash the whole session later; waiting for the window avoids it. Terminal launches and the updater's relaunch still write the entry immediately.

| Flag                 | Description                                                                               |
| -------------------- | ----------------------------------------------------------------------------------------- |
| `--skip-build`       | Skip npm install/package and launch the existing unpacked app from `apps/desktop/release` |
| `--force-build`      | Force a full rebuild even if the content stamp matches                                    |
| `--build-only`       | Build the desktop app but do not launch it (used by `hermes update`)                      |
| `--source`           | Launch via `electron .` against `apps/desktop/dist` instead of the packaged app           |
| `--cwd PATH`         | Initial project directory for desktop chat sessions (sets `HERMES_DESKTOP_CWD`)           |
| `--hermes-root PATH` | Override the Hermes source root the app uses (sets `HERMES_DESKTOP_HERMES_ROOT`)          |
| `--ignore-existing`  | Force the app to ignore any `hermes` CLI already on `PATH` during backend resolution      |
| `--fake-boot`        | Enable deterministic boot delays for validating the startup UI                            |

## How it works

The packaged app ships the Electron shell and a native React chat surface. On first launch it can install the Hermes Agent runtime into `HERMES_HOME` (`~/.hermes`, or `%LOCALAPPDATA%\hermes` on Windows) — **the same layout a CLI install uses**, which is why the two are interchangeable. Bundled apps use their included backend. Without a bundled backend, resolution honours `HERMES_DESKTOP_HERMES_ROOT`, then the development checkout, then an explicit `HERMES_DESKTOP_HERMES` command override for packagers such as Nix, and finally a usable managed install. The command override takes precedence over the managed install so a Nix desktop cannot silently launch an older mutable runtime. The React renderer talks to a headless backend the app launches for you — a `hermes serve` process that serves the `tui_gateway` JSON-RPC/WebSocket API — and reuses the agent runtime rather than embedding `hermes --tui`. The desktop app is **self-contained**: it runs its own `hermes serve` backend and never opens or requires the [web dashboard](./features/web-dashboard.md). (Runtimes older than the `serve` command fall back to a headless `dashboard --no-open` automatically, so an app update never outruns its backend.) Install, backend-resolution, and self-update logic live in the Electron main process.

## Connecting to a remote backend

By default the app starts and manages its own **local** backend. You can instead point it at a Hermes backend running on another machine — a VPS, a home server, or a Mini behind Tailscale.

Everything connection-related lives on one settings page: **Settings → Gateways**. (Older builds split this across separate **Gateway** and **Connections** pages — those are now unified, and old `?tab=connections` deep links redirect to the unified page.)

**Settings → Gateways → Connection mode** offers the alternatives to the local gateway:

- **Remote gateway** — enter the URL of a `hermes serve` backend you run yourself and sign in. This is the mode the rest of this section walks through.
- **Hermes Cloud** — sign in once to Hermes Cloud and pick from the agents on your account; no URL to paste. The app discovers your agents (with an organization picker if your account spans several orgs), and connecting to one switches the session over automatically. The status bar shows the cloud connection while it's active.

Gateway connections are **machine-level**: the Gateways page manages which gateway backends this desktop can connect to, and profiles are discovered *from* the gateways you connect. Sessions select one gateway at a time, while the adjacent profile rail selects a profile discovered on that gateway.

### The multi-connection registry

Further down the same **Settings → Gateways** page, **Registered gateways** manages a named list of every Hermes gateway the app knows about — the local runtime, any number of remote gateways (LAN, Tailscale, internet), Hermes Cloud instances, and SSH hosts — all persisted together in one place. You can jump there from the plug button at the right end of the sidebar profile rail (**Connect another Hermes gateway…**) or via **⌘K → Gateways**. The full guide, including the union agent roster, `@name-device` handles, fleet-wide updates, and the plugin SDK surface, is at [Connecting Desktop to Many Hermes Instances](./multi-connection-desktop.md).

- **Every connection needs a unique name** (a device name such as "Homelab" or "Work laptop"). When the same profile name exists on several registered gateways, surfaces disambiguate it as `@profile-device` (e.g. `@research-homelab`).
- **Switch gateways from the Sessions sidebar.** A named gateway selector appears when more than one gateway is registered and handles any registry size without making gateways look like profiles. The adjacent profile rail then shows only that gateway's agents and remembers the last profile used there; large profile sets condense independently.
- **Hide the profile rail.** If your profiles are bots rather than workspaces, the colored strip of profile squares at the foot of the sidebar duplicates the sessions list. Turn it off from the Sessions view menu (**Profile rail**), the shell's right-click menu, or **⌘K → Toggle profile rail**. While it is hidden, a **Profiles** dropdown appears in the status bar next to the gateway selector with the same choices — this gateway's profiles, other gateways' agents, **New profile**, **Import profile…**, **Manage profiles…** — so switching profiles always has a door.
- **Choose what opens after a restart.** **Open on launch** keeps the backward-compatible **Primary gateway** default, or can resume the **Last used** gateway after it connects successfully. This preference is stored outside the application bundle and survives Desktop updates.
- **Add / edit / remove / test** connections from the panel. The **Add** flow offers all four kinds — **Local**, **Hermes Cloud**, **Remote gateway**, and **SSH** (the Local button is disabled while the app-managed local entry exists, and a hint points cloud adds at the sign-in/discovery flow above). The local entry is managed by the app and cannot be removed. **Test** probes the connection's own HTTP and WebSocket legs directly.
- **Duplicates are rejected at save time**: only one **local** entry ever; remote and cloud entries are deduplicated on the normalized URL (trimmed, trailing slashes stripped, lowercased — across both kinds); SSH entries on the normalized `user@host:port` plus remote profile.
- Existing settings are **imported automatically** the first time you run a build with the registry: your current global connection and any legacy per-profile overrides become named entries. The legacy settings file is left untouched, so older builds keep working.
- Cloud entries come from the Hermes Cloud sign-in/discovery flow above, not from a hand-typed URL.
- Tokens are stored encrypted with the OS keyring (with an explicit plain-text opt-in on keyring-less Linux).

Side-by-side routing is live: each registered gateway dials its own backends and sockets on demand (keyed per connection + profile), the plugin SDK exposes the union agent roster (`host.agents()` / `host.ensureAgent()`), and **Update all instances** on the Gateways page dispatches `hermes update` to every eligible gateway at once — Hermes Cloud entries are skipped (the platform updates them), and each instance reports its own result.


:::info The remote backend is a running `hermes serve` process
"Remote backend" means a **`hermes serve`** server running on the remote machine — that is the process the desktop app connects to. Nothing in this section works unless that backend is actually up and reachable. The desktop app does not start it for you; you (or a `systemd` service) keep `hermes serve` running on the remote host, and the app attaches to it. If you also use messaging channels (Telegram, Discord, etc.), the **gateway** is a *separate* long-running process you start independently — see the note after the setup steps.
:::

The connection has two halves: on the backend you protect it with an **auth provider**, and in the app you enter the backend's URL and sign in. Binding the backend to a non-loopback address automatically engages its auth gate, and the provider you configure is what lets the desktop app through.

**Pick a provider based on where the backend lives:**

- **OAuth (Nous Portal) — preferred for anything reachable beyond your own machine.** Logins are verified against your Nous account, so this is the option suitable for a VPS, a public host, or any remote backend. Register the dashboard with `hermes dashboard register` (or the Portal [`/local-dashboards`](https://portal.nousresearch.com/local-dashboards) page) to provision its OAuth client, then sign in from the app with **Sign in with Nous Research**. A self-hosted OIDC provider works the same way if you run your own identity provider.
- **Username/password — local / trusted-network use only.** The simplest option when the backend is on the same trusted LAN or reachable only over a VPN (e.g. Tailscale). It protects a single shared credential with no external identity provider, so **do not use it for a dashboard exposed to the public internet** — reach for OAuth there instead.

The rest of this section shows the username/password path because it's the quickest to stand up on a trusted network; for the OAuth path see [Web Dashboard → Default provider: Nous Research](./features/web-dashboard.md#default-provider-nous-research).

### On the backend (the remote machine)

Set a username and password, then start the backend bound to a reachable address. The credentials live in `~/.hermes/.env` (the secrets file, mode 0600):

```bash
# 1. Set the dashboard login credentials.
cat >> ~/.hermes/.env <<'EOF'
HERMES_DASHBOARD_BASIC_AUTH_USERNAME=admin
HERMES_DASHBOARD_BASIC_AUTH_PASSWORD=choose-a-strong-password
# Recommended: a stable signing secret so sessions survive restarts.
# Without it a random key is generated per boot and you'll be logged out
# on every restart.
HERMES_DASHBOARD_BASIC_AUTH_SECRET=$(openssl rand -base64 32)
EOF
chmod 600 ~/.hermes/.env

# 2. Run the backend bound to a reachable address. The non-loopback bind
#    engages the auth gate; the username/password provider handles login.
hermes serve --host 0.0.0.0 --port 9119
```

Keep that `hermes serve` process running for as long as you want the desktop app to be able to connect — if it stops, the app can no longer reach the backend. Run it under `systemd`, `tmux`, or your process manager of choice so it survives logout and reboots.

Separately, make sure the **gateway is running** on the remote host if you rely on messaging channels — the `hermes serve` backend is what the desktop app talks to, but your Telegram/Discord/Slack gateway sessions are a different process that you start and keep running on their own. See [Messaging](./messaging/index.md) for gateway setup.

Prefer not to keep a plaintext password at rest? Set `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH` to a scrypt hash instead — compute it with `python -c "from plugins.dashboard_auth.basic import hash_password; print(hash_password('PW'))"`. Full configuration surface (config.yaml keys, every env var, the rate limiter): [Web Dashboard → Username/password provider](./features/web-dashboard.md#usernamepassword-provider-no-oauth-idp).

Running the backend as a systemd service? Give the unit `EnvironmentFile=%h/.hermes/.env` so the credentials are in the environment at boot.

:::warning
The backend reads and writes your `.env` (API keys, secrets) and can run agent commands. The **username/password** setup shown above is for a trusted network — never expose a password-protected backend directly to the open internet; put it behind a VPN. [Tailscale](https://tailscale.com/) is the clean option: bind to the machine's tailscale IP (`--host <tailscale-ip>`) and use `http://<tailscale-ip>:9119` as the Remote URL so only your tailnet can reach it. To reach a backend over the public internet, use the **OAuth (Nous Portal)** provider instead.
:::

### In the app

**Settings → Gateways → Remote gateway:**

1. **Remote URL** — `http://<backend-host>:9119` (path prefixes like `/hermes` work if you front it with a reverse proxy)
2. **Sign in** — the app detects which provider the backend advertises and adapts the button. For a username/password backend it shows a **Sign in** button that opens a credential form (enter the credentials from step 1). For an OAuth backend it shows **Sign in with `<provider>`** (e.g. *Sign in with Nous Research*), which runs the provider's browser sign-in. Either way the app ends up with an authenticated session against the backend.
3. **Save and reconnect** — switches the desktop shell onto the remote backend. The session refreshes automatically; you stay signed in across restarts when `HERMES_DASHBOARD_BASIC_AUTH_SECRET` is set.

You can also set the backend URL without the UI via the `HERMES_DESKTOP_REMOTE_URL` environment variable before launching the app (it overrides the in-app setting); you still sign in from the Gateways settings panel.

:::note Per-profile remote hosts
The remote gateway host is configured per [profile](./profiles.md), so each profile can point at its own remote backend (or stay on its local one). Switching profiles switches which remote host the app connects to.
:::

### Troubleshooting

- **Sign-in fails with 401 / "Invalid credentials"** — the username or password doesn't match the backend's `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` / `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD`. The backend returns the same generic error for an unknown user and a wrong password (no enumeration oracle), so double-check both. Confirm the gate is on with `curl -s http://<host>:9119/api/status | jq '.auth_required, .auth_providers'` — it should report `true` and include `"basic"`.
- **No "Sign in" button — it asks for a session token instead** — the backend's username/password provider isn't active. `/api/status` won't list `"basic"` in `auth_providers`. Make sure both the username and a password (or password hash) are set in `~/.hermes/.env` and that the dashboard process actually loaded them.
- **Signed out on every restart** — set `HERMES_DASHBOARD_BASIC_AUTH_SECRET` to a stable value. Without it the token-signing key is regenerated per boot, invalidating all sessions.
- **Connection refused / times out** — the backend bound to `127.0.0.1` (the default) or a firewall/VPN is blocking the port. Bind to `0.0.0.0` or the tailscale IP and open the port to your trusted network.

For the same setup from the web-dashboard angle, see [Web Dashboard → Connecting Hermes Desktop to a remote backend](./features/web-dashboard.md#connecting-hermes-desktop-to-a-remote-backend); the env vars are catalogued under [Environment Variables → Web Dashboard & Hermes Desktop](../reference/environment-variables.md#web-dashboard--hermes-desktop).

## Extending the desktop app

The desktop app is contribution-driven — panes, pages, sidebar nav, status-bar
items, palette commands, keybinds, and themes all register through one SDK, and
you can add your own. A plugin is a single ESM file dropped in
`$HERMES_HOME/desktop-plugins/<id>/plugin.js`; the app loads it within seconds and
hot-reloads every save. Manage installed plugins live in **Capabilities → Plugins**.

See [Desktop Plugin SDK](../developer-guide/desktop-plugin-sdk.md) for the full
reference. (This is separate from the [web dashboard plugin system](./features/extending-the-dashboard.md).)

A desktop plugin is **not sandboxed**: it runs inside the app with the app's own
authority (gateway RPC, the native bridge, other plugins' storage). Only load
files you wrote or reviewed; for catalog installs the protection is the
[catalog trust model](./features/plugin-catalog.md#trust-model) (human-reviewed,
SHA-pinned) plus an import allowlist — not isolation. A plugin whose
`register()` throws is rolled back and shown as **Failed** with the error on its
row; **⌘K → Reload desktop plugins** re-reads every installed `plugin.js`,
including one an installer replaced in place.

**Capabilities → Plugins → Installed** shows the actual installed state:
**one list entry per plugin**, with Desktop and Agent controls in its detail pane.

- A plugin can extend **this app**, **the agent**, or **both** — the badge on
  each row says which, inferred from what the package contains (`plugin.yaml`
  → agent half, `plugin.js` → desktop half). A plugin with both halves is one
  row, never two.
- **Desktop control** — the half loaded into this app. It is app-level: the
  same switch, the same value, whichever profile, gateway, or remote machine
  the window is looking at. Desktop code loads from exactly one place,
  `~/.hermes/desktop-plugins/`; the desktop half of a unified agent+desktop
  package is copied there by the app when the package is installed (and
  follows its updates and uninstall), so switching profiles never loads,
  unloads, or re-scopes a pane. Toggles apply live.
- **Agent control** — the half installed in the selected profile's backend
  ([agent plugins](./features/plugins.md): user, git, project, pip and
  portable installs), with an **Update** chip when a catalog pin moved. The
  profile selector governs only the agent half; with a single profile there
  is no selector at all.
  Repo-bundled built-ins (platform adapters, provider plugins) are not
  listed: they ship enabled and are configured from their own surfaces. The
  exceptions are the bundled lifecycle plugins with no surface of their own
  (`disk-cleanup`, `security-guidance`), which appear here so they can be
  toggled like any other agent plugin.
- A half the plugin does not ship shows a dash. A desktop half whose agent
  half is **not** installed in the selected profile shows **Install here**,
  which pre-fills the install dialog from the package's origin (catalog entry
  or git remote) for that profile only. Optional extras such as the
  [Accent Picker](https://github.com/NousResearch/hermes-desktop-accent-picker)
  install from their own repos via **Install from Git**.
- **Uninstall** — every plugin installed under the selected profile's
  `plugins/` folder (user or git install) has a trash button beside its name.
  It asks for confirmation, then deletes the plugin's files and install
  metadata from that profile — the same operation as
  `hermes plugins remove <name>` — and prunes the app-level copy of a unified
  package's desktop half. Restart the gateway to unload the plugin's code.
  Repo-bundled and pip-installed (entrypoint) plugins have no trash button:
  the first cannot be removed, the second goes with its Python package.
  A standalone desktop plugin (a folder you dropped into
  `~/.hermes/desktop-plugins/` with no agent package) gets the same trash
  button; confirming deletes that folder on this computer and unloads the
  plugin immediately, no gateway involved.

Switch to **Browse** for the native [Plugin Catalog](./features/plugin-catalog.md).
Both Browse and **Install from Git** open the review-then-install dialog. For
an agent-plugin catalog install, the backend resolves the catalog name to its
reviewed pin. Website links carry only that name; Desktop looks up the reviewed
repository and commit rather than trusting metadata supplied by a link.
**Install from Git** also offers **Pin to commit** for agent-plugin installs
(a full 40-character SHA, including private repositories); pinned agent plugins
show a `pinned @ <sha8>` badge. Old `Settings → Plugins` links redirect here.

## Troubleshooting

### Reconnect without restarting the app

If a Desktop chat or bot stops responding while the connection still shows **Connected**, select that bot/profile or gateway, open the status bar's gateway menu, and click **Reconnect gateway**. Reconnect stays available for open, connecting, and disconnected transports. It redials the active route without restarting Desktop or deliberately closing other routes' sockets. In-flight requests on the selected socket can be interrupted; this is an explicit recovery action, not a backend or model restart.

### The app vanished after `hermes update`

An earlier update that replaced the checkout without keeping `apps/desktop/release/` leaves no packaged app to launch. As long as `HERMES_HOME/desktop-build-stamp.json` (written only by a successful Desktop build) still exists, the next `hermes update` notices the missing app and rebuilds it. To rebuild by hand: `hermes desktop --build-only --force-build`. On Windows the ZIP fallback also keeps the built app, its renderer bundle and its Electron `node_modules` across the swap.

### The local backend stopped in the background

If the local Hermes backend process exits after it was ready, Desktop restarts it on its own and shows a **Hermes stopped working in the background** notice; the chat reconnects once the replacement is up. `HERMES_HOME/logs/desktop.log` records the exit code together with the backend's last output lines (`Hermes backend exited (1)` followed by `Recent backend output:`), so the reason it died is in the log even when the app recovered by itself. A backend that keeps dying within seconds of every restart points at the backend itself — look at the traceback in that tail. After three such restarts within two minutes Desktop stops respawning and shows a **keeps crashing** notice instead of cycling; relaunch the app once the cause is fixed.

### Failed turns name the failing layer

When a turn fails, the chat renders an error card that names **which layer
failed** — provider/model, custom endpoint, streaming connection,
authentication, billing, gateway, local runtime, or disk — instead of a
generic error toast. The card offers recovery actions matched to the failure:

- **Retry** — re-runs the failed turn in place (hidden when retrying would
  deterministically reproduce the failure, e.g. a content-policy rejection).
  When a rate-limit or usage-limit response names when the limit lifts
  (`Retry-After` header or a `resets_at` field), the card shows **Limit resets
  at HH:mm (in 1h 05m)** next to Retry so you know when a retry will work; the
  CLI/TUI print the same line under the error. The hint itself is
  informational, but the card also offers **Retry when the limit resets
  (HH:mm)**: click it and the app retries that turn once at the reset time
  with a live countdown and a **Cancel** control. The schedule lives only in
  the open window — switching sessions, sending another message, or closing
  the app drops it, and nothing retries unattended.
- **Switch provider** — for provider, endpoint, auth, and billing failures,
  opens the composer's live model menu so you can move **this chat** to another
  provider/model right away (Settings → Models only changes the default for new
  chats). When no chat surface is on screen it falls back to Settings → Models.
- **Open logs** — opens `HERMES_HOME/logs` in your file manager. On a remote
  or Cloud connection the button reads **Open Desktop logs**: it opens the
  local Desktop-side logs (transport evidence), since the failed turn's
  gateway/agent logs live on the remote machine.
- **Send diagnostics** — uploads a redacted debug bundle to Nous-internal
  storage after an explicit consent prompt (same pipeline as
  `hermes debug share --nous`; secrets are always redacted, the bundle is
  viewable by Nous staff only and auto-deletes after 14 days). On success you
  get a private view link to paste into your support thread, plus quick links
  to GitHub Issues, Nous Portal Support, and Discord. On a remote or Cloud
  connection the backend bundles its own agent/gateway logs and the local
  Desktop log is attached alongside, so support sees both halves.
- **Copy error details** — copies a compact plain-text summary (layer, code,
  provider/model, error message) you can paste into a bug report or Discord.

The layer comes from the same error classifier the agent's retry loop uses,
so it reflects the real failure semantics, not a guess from the message text.
Older backends that predate the descriptor still render the card with a
generic title and the Retry / Open logs / Copy error details actions.

Boot logs land in `HERMES_HOME/logs/desktop.log` (it includes backend output and recent Python tracebacks) — check it first if the app reports a boot failure. You can also tail it from the CLI:

```bash
hermes logs gui -f
```

For a canonical source installation, Desktop checks and runs the installation
launcher. PM selects its interpreter and dependency generation. A missing
bootstrap marker does not force installation when that launcher works.

On Linux, Chromium's own errors go to `HERMES_HOME/logs/desktop-chromium.log`, and a crash of the shell itself leaves a minidump under the app's `Crashpad/` directory (inside Electron's user-data directory, next to `connection.json`). If the window vanishes with `SIGTRAP` in the journal, the `FATAL:` line in that log names the check that fired; attach it to the bug report. Nothing is uploaded.

If Python dependencies are damaged, run the installation's `hermes pm repair`.
Then restart Desktop. Do not delete guessed `venv` paths or PM facts. For
damaged application files, repair through the
[installation owner](../reference/package-management.md#source-installs-and-packaged-builds).

```bash
# Reset a stuck macOS microphone prompt
tccutil reset Microphone com.nousresearch.hermes
```

### "The host key has CHANGED since you last connected" (SSH remote)

If your SSH remote was reinstalled or its host key rotated, SSH fails closed
and Desktop latches an error overlay instead of retrying (retrying can never
succeed until the stale key is cleared). Verify the change is expected, then
remove the old entry and retry from the overlay:

```bash
ssh-keygen -R <host>
```

Click **Retry** (or re-apply the connection in Settings → Gateway) after
clearing the entry — the latch resets and the next boot dials fresh.

### "Build desktop app" stuck on Electron download

The build downloads the Electron runtime (~114&nbsp;MB) from `github.com/electron/electron/releases`. If the installer hangs on the **Build desktop app** step with the live output repeating `retrying attempt=…`, GitHub is being blocked or throttled on your network (firewall, proxy, or region).

The build does not fall back to a mirror on its own and does not retry a failed download: a build that fails for any reason leaves the previous app untouched (stage-and-swap, see [Updating](../getting-started/updating.md#what-happens-during-an-update)), the update itself fails, and `hermes desktop --build-only --force-build` (or the next `hermes update`) runs the build again. If the failure was a corrupt cached Electron zip, delete it from `@electron/get`'s cache (`~/.cache/electron/` on Linux, `~/Library/Caches/electron/` on macOS, `%LOCALAPPDATA%\electron\Cache` on Windows) before rebuilding. `@electron/get` SHASUM-checks every download, but the checksums come from the same host, so a mirror you point it at is trusted for both.

To **use a mirror** (e.g. a corporate one, or `npmmirror.com`, the de-facto Electron community mirror), set `ELECTRON_MIRROR` before installing or rebuild manually — the build honors it and never overrides one you've set:

```bash
ELECTRON_MIRROR=https://npmmirror.com/mirrors/electron/ \
  bash -c 'cd "$HOME/.hermes/hermes-agent/apps/desktop" && CSC_IDENTITY_AUTO_DISCOVERY=false npm run pack'
```

**Other native downloads (e.g. the `get-windows` prebuilt on Windows) that need a mirror:** put the npm keys in `$HERMES_HOME/npmrc` (`%LOCALAPPDATA%\hermes\npmrc` on Windows, `~/.hermes/npmrc` elsewhere) — for example `node_get_windows_binary_host_mirror=https://<mirror>/sindresorhus/get-windows/releases/download/`. Every `npm ci`/`npm run` the updater spawns (desktop, web and TUI builds) points `NPM_CONFIG_USERCONFIG` at that file when it exists, so the config survives `hermes update`; the repo-root `.npmrc` is git-tracked and gets autostashed on every update, and `~/.npmrc` may be missed because the desktop hand-off inherits the GUI's environment. An `NPM_CONFIG_USERCONFIG` you set yourself is never overridden.

**If `get-windows` is missing or half-installed:** the build no longer fails — it prints `[stage-native-deps] get-windows not installed ... read_window_below will be unavailable in this build` and ships without the `read_window_below` tool. When the package directory exists but is not loadable (a Windows in-place update interrupted by a running Hermes window, `TAR_ENTRY_ERROR` in the install log), the same warning names the directory, and the next `hermes desktop --force-build` or update removes it before its npm install so the package is re-extracted — close every Hermes window and gateway first so the extract is not interrupted again. A package whose native binding or macOS helper is missing is likewise shipped without window enumeration rather than failing the build.

To clear a corrupt cached zip by hand:

```bash
rm -f "$HOME/Library/Caches/electron"/electron-*.zip   # macOS
rm -f "$HOME/.cache/electron"/electron-*.zip            # Linux
```

## Building from source

If you want to hack on the app itself, install workspace deps from the repo root once, then run the dev server from `apps/desktop`:

```bash
npm install          # from repo root — links apps/desktop, web, apps/shared
cd apps/desktop
npm run dev          # Vite renderer + Electron, which boots the Python backend
```

Point the app at a specific checkout, or sandbox it from your real config:

```bash
HERMES_DESKTOP_HERMES_ROOT=/path/to/clone npm run dev
HERMES_HOME=$HOME/.hermes/cache/scratch/throwaway npm run dev
npm run dev:fake-boot   # exercise the startup overlay with deterministic delays
```

Build installers:

```bash
npm run dist:mac     # DMG + zip
npm run dist:win     # NSIS + MSI
npm run dist:linux   # AppImage + deb + rpm
npm run pack         # unpacked app under release/ (no installer)
```

macOS/Windows signing and notarization run automatically when the relevant credentials are present in the environment (`CSC_LINK` / `CSC_KEY_PASSWORD` / `APPLE_*` for macOS, `WIN_CSC_*` for Windows).

The opt-in HUD modifier-tap helper is built with the Electron bundle and packaged outside ASAR. macOS uses the existing Xcode command-line tools prerequisite. Windows builds use the C# compiler included with the operating system's .NET Framework; no Clang or developer SDK is required. Windows packaging fails rather than silently omitting the helper. Linux builds need a C compiler and X11/XInput development headers (`libx11-dev` and `libxi-dev` on Debian/Ubuntu); without those optional Linux prerequisites, packaging continues without modifier-tap support. Build on the target OS; Linux also requires the target architecture. Installed users do not need a developer toolchain. Settings distinguishes a missing helper from startup failure and an unsupported desktop session; the X11/Wayland warning is not shown for a missing or failed Windows helper.

### macOS permissions and local rebuilds (TCC)

**Silence every folder prompt with one switch.** macOS prompts per-category
(Desktop, then Downloads, then Documents, ...) as Hermes touches each folder.
A single **Full Disk Access** grant covers all of them, permanently — and
with Hermes' stable signing identities it survives every update:

1. System Settings → **Privacy & Security → Full Disk Access** (or run
   `open "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles"`)
2. Enable your terminal app — and **Hermes.app** if you use Desktop.
3. Fully quit and relaunch them once.

`hermes doctor` reports whether the current terminal context already has the
grant, and `hermes setup` shows this tip on macOS when it doesn't.

macOS remembers permission grants (Full Disk Access, Desktop/Downloads/Documents,
Accessibility, Automation, microphone) against the app's *code-signing identity*,
not its path. Locally built and self-updated apps are signed with a stable
identifier-pinned ad-hoc signature, so grants persist across updates.

One-time note: grants made to builds *before* the identifier-pinned signing
fix (PR #73681) carry the old cdhash-pinned requirement. macOS keeps showing the
toggle as ON for those stale grants but still re-prompts, because the stored
grant no longer matches the rebuilt binary — and the modern prompt has no Allow
button, so it looks like there is nothing to re-check. If that happens, reset
the stale grant once and re-grant:

```bash
tccutil reset ScreenCapture com.nousresearch.hermes   # repeat per service
```

then toggle the fresh entry ON in System Settings and fully quit & relaunch
Hermes. Grants are stable from then on.

For the strongest guarantee — a certificate-anchored identity, the same
mechanism yabai/skhd users rely on — create a self-signed code-signing
certificate once and tell Hermes to use it. The one-shot command does
everything (creates the certificate in your login keychain, grants `codesign`
access, writes the config, and re-signs the packaged app):

```bash
hermes desktop --setup-tcc-identity
```

Or do it manually:

1. Keychain Access → Certificate Assistant → **Create a Certificate…**
2. Name: `Hermes Local Signing`, Identity Type: *Self-Signed Root*,
   Certificate Type: **Code Signing**.
3. In Keychain Access, double-click the new certificate → **Trust** → set
   **Code Signing** to *Always Trust* (an imported self-signed certificate is
   not a valid signing identity until it is trusted for code signing —
   `security find-identity -v -p codesigning` should list it afterwards).
4. `hermes config set desktop.macos_signing_identity "Hermes Local Signing"`

Use `--identity <name>` with the command to create/use a differently named
certificate (default: `Hermes Local Signing`). The command is idempotent —
re-run it after updates to re-point the config and re-sign the rebuilt app.

The next update re-signs the rebuilt app with that certificate; every TCC grant
survives. No Apple Developer account is required. Notarized release builds are
detected and never re-signed.

One-time note: changing the signing identity (including the first update after
this fix) changes the app's identity once, so macOS will re-prompt one final
time. Grants are stable from then on. If a permission gets stuck, reset it with
`tccutil reset All com.nousresearch.hermes` and re-grant.

## See also

- [CLI Guide](./cli.md) — the terminal interface
- [TUI](./tui.md) — the modern terminal UI used by `hermes --tui` and the dashboard chat tab
- [Web Dashboard](./features/web-dashboard.md) — browser admin panel with an embedded chat tab
- [Configuration](./configuration.md) — config that the desktop app reads and writes
- [Windows (Native)](./windows-native.md) — native Windows install path

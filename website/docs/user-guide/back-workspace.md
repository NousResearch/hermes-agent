---
title: "Back Workspace"
description: "The note page on the back of the Hermes desktop window. Turn the window over like a card, write on a plain markdown file, paste pictures, and call an agent with @ without leaving the page."
---

# Back Workspace

The desktop window has a back. Press **⌘/Ctrl+Shift+E** — or the note button in the titlebar — and the whole window turns over like a card onto a blank page you can write on.

There is nothing on that page: no titlebar, no buttons, no panels. It is a text file and a caret. The one thing it can do that a text editor cannot is call an agent: type `@`, ask a question, and the answer is written in beneath it.

Press **⌘/Ctrl+Shift+E** again, or **Esc**, to turn back. Whatever was running on the front — a turn, a terminal, a stream — keeps running while the window is over; turning it is a view, not a mode.

## Writing

Type. The page saves itself a moment after you stop, to a markdown file of its own (see [Where the pages live](#where-the-pages-live)). There is no save command and nothing to name.

The writing sits in a column down the middle of the window. Clicking in the margin beside a line puts the caret on that line, so the whole page is writable however wide the window is.

If a save fails — the backend went away mid-sentence — a line at the bottom of the page says so, your text stays where it is, and the next edit tries again.

## Asking an agent

Type `@` and a list opens beside the caret: the profile this window is on, plus every Bot from [Bot Mode](./bot-mode.md). Pick one, write the question in the same paragraph, and press **⌘/Ctrl+Enter**.

The paragraph the caret is in is what gets sent. While the agent is thinking, a line at the bottom says so; when it answers, the reply is written into the page under your question as a markdown quote, in a quieter colour than your own writing:

```markdown
@hermes what did I change in the gateway last week?

> hermes · 02:07 AM
> Three things: the reconnect backoff, the token lock, and the
> session-title lookup.
```

The reply is ordinary text in your file. Edit it, keep it, delete it.

A paragraph with **no** `@` continues the conversation the page last had, so a follow-up question needs no tag — until the app restarts, after which the first question of the session needs one again. The page remembers across questions: every agent it asks gets one hidden session of its own, titled `Back Workspace`, reused every time. Those sessions do not appear in the Sessions sidebar — the page is their only door.

:::note The agent can read the page
Every question tells the agent which file the page is, so it can open the page itself when it needs more of the context around your question. It is told not to write to that file: the app pastes the reply in.
:::

## Pictures

Paste an image and it is stored beside the page and shown under the line that links it. The link is ordinary markdown (`![](assets/…)`) pointing at a file next to the page, so the page and its pictures stay one folder you can move or sync — and the agent reading the page can open them too.

PNG, JPEG, GIF and WebP are kept, up to 16 MB. Anything else pastes as text. If the clipboard carries text alongside the picture — a copy from a web page usually does — the text is pasted as well.

## Approvals

When the agent you asked wants to run something that needs your say-so, the page does not interrupt you. A line appears at the bottom, in the same quiet grey that says the agent is answering:

```
The agent is waiting for your ok — ⌘⇧A to look.
```

(⌃⇧A on Windows and Linux; the line always spells the chord your machine uses.)

Press **⌘/Ctrl+Shift+A** and a card opens beside the caret with the whole command and what you can do about it — **Run**, **Allow this session**, **Always allow**, **Reject**. Move with the arrow keys, choose with **Enter**, or click. **Esc** puts the card down without answering; the line stays, and you can open it again.

**Always allow** writes the command's pattern to `~/.hermes/config.yaml` for good, so the page asks a second time before it does — press that row again to confirm.

A few things worth knowing:

- **While the window is turned to the front**, the note button in the titlebar carries a count of what is waiting. Turn the window over to answer.
- **Nobody has to answer.** An approval that is left alone is refused when the approval timeout runs out (five minutes by default), and the command does not run. The page's own wait for the reply is paused for as long as an approval is waiting — whether or not you have opened the card — so a command allowed in the fifth minute still gets its answer written in.
- **What counts as needing approval** is the same everywhere in Hermes, and is set by `approvals.mode` — see [Security](./security.md#dangerous-command-approval).

## Where the pages live

Each profile has its own page, under that profile's own Hermes home — `~/.hermes/` for the default profile, and `~/.hermes/profiles/<name>/` for any other (see [Profiles](./profiles.md)):

```
<hermes home>/backworkspace/
├── 20260920_030000_a1b2c3.md      the page
└── assets/
    └── 20260920_031500_d4e5f6.png a pasted picture
```

They are plain files. Read them with anything, keep them in a repository, hand one to an agent by name. Switching profile or connection while the window is over shows that profile's page instead.

If the backend is older than this feature, the page says so instead of opening — update Hermes to use it.

## Shortcuts

| Chord | What it does |
| --- | --- |
| **⌘/Ctrl+Shift+E** | Turn the window over, and back |
| **Esc** | Turn back (when nothing on the page is open) |
| `@` | Open the list of agents to ask |
| **⌘/Ctrl+Enter** | Send the paragraph the caret is in |
| **⌘/Ctrl+Shift+A** | Open the approval that is waiting |

## See also

- [Hermes Desktop](./desktop.md) — the rest of the app
- [Bot Mode](./bot-mode.md) — where the agents in the `@` list come from
- [Security](./security.md#dangerous-command-approval) — what needs approval, and how to change it
- [Profiles](./profiles.md) — why each profile has its own page

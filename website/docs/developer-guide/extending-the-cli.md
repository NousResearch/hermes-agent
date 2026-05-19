---
sidebar_position: 8
title: "Extending the CLI"
description: "Build wrapper CLIs that extend the Hermes TUI with custom widgets, keybindings, and layout changes"
---

# Extending the CLI

Hermes exposes protected extension hooks on `HermesCLI` so wrapper CLIs can add widgets, keybindings, and layout customizations without overriding the 1000+ line `run()` method. This keeps your extension decoupled from internal changes.

## Extension points

There are five extension seams available:

| Hook | Purpose | Override when... |
|------|---------|------------------|
| `_get_extra_tui_widgets()` | Inject widgets into the layout | You need a persistent UI element (panel, status line, mini-player) |
| `_register_extra_tui_keybindings(kb, *, input_area)` | Add keyboard shortcuts | You need hotkeys (toggle panels, transport controls, modal shortcuts) |
| `_build_tui_layout_children(**widgets)` | Full control over widget ordering | You need to reorder or wrap existing widgets (rare) |
| `process_command()` | Add custom slash commands | You need `/mycommand` handling (pre-existing hook) |
| `_build_tui_style_dict()` | Custom prompt_toolkit styles | You need custom colors or styling (pre-existing hook) |

The first three are new protected hooks. The last two already existed.

## Quick start: a wrapper CLI

```python
#!/usr/bin/env python3
"""my_cli.py — Example wrapper CLI that extends Hermes."""

from cli import HermesCLI
from prompt_toolkit.layout import FormattedTextControl, Window
from prompt_toolkit.filters import Condition


class MyCLI(HermesCLI):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._panel_visible = False

    def _get_extra_tui_widgets(self):
        """Add a toggleable info panel above the status bar."""
        cli_ref = self
        return [
            Window(
                FormattedTextControl(lambda: "📊 My custom panel content"),
                height=1,
                filter=Condition(lambda: cli_ref._panel_visible),
            ),
        ]

    def _register_extra_tui_keybindings(self, kb, *, input_area):
        """F2 toggles the custom panel."""
        cli_ref = self

        @kb.add("f2")
        def _toggle_panel(event):
            cli_ref._panel_visible = not cli_ref._panel_visible

    def process_command(self, cmd: str) -> bool:
        """Add a /panel slash command."""
        if cmd.strip().lower() == "/panel":
            self._panel_visible = not self._panel_visible
            state = "visible" if self._panel_visible else "hidden"
            print(f"Panel is now {state}")
            return True
        return super().process_command(cmd)


if __name__ == "__main__":
    cli = MyCLI()
    cli.run()
```

Run it:

```bash
cd ~/.hermes/hermes-agent
source .venv/bin/activate
python my_cli.py
```

## Hook reference

### `_get_extra_tui_widgets()`

Returns a list of prompt_toolkit widgets to insert into the TUI layout. Widgets appear **between the spacer and the status bar** — above the input area but below the main output.

```python
def _get_extra_tui_widgets(self) -> list:
    return []  # default: no extra widgets
```

Each widget should be a prompt_toolkit container (e.g., `Window`, `ConditionalContainer`, `HSplit`). Use `ConditionalContainer` or `filter=Condition(...)` to make widgets toggleable.

```python
from prompt_toolkit.layout import ConditionalContainer, Window, FormattedTextControl
from prompt_toolkit.filters import Condition

def _get_extra_tui_widgets(self):
    return [
        ConditionalContainer(
            Window(FormattedTextControl("Status: connected"), height=1),
            filter=Condition(lambda: self._show_status),
        ),
    ]
```

### `_register_extra_tui_keybindings(kb, *, input_area)`

Called after Hermes registers its own keybindings and before the layout is built. Add your keybindings to `kb`.

```python
def _register_extra_tui_keybindings(self, kb, *, input_area):
    pass  # default: no extra keybindings
```

Parameters:
- **`kb`** — The `KeyBindings` instance for the prompt_toolkit application
- **`input_area`** — The main `TextArea` widget, if you need to read or manipulate user input

```python
def _register_extra_tui_keybindings(self, kb, *, input_area):
    cli_ref = self

    @kb.add("f3")
    def _clear_input(event):
        input_area.text = ""

    @kb.add("f4")
    def _insert_template(event):
        input_area.text = "/search "
```

**Avoid conflicts** with built-in keybindings: `Enter` (submit), `Escape Enter` (newline), `Ctrl-C` (interrupt), `Ctrl-D` (exit), `Tab` (auto-suggest accept). Function keys F2+ and Ctrl-combinations are generally safe.

### `_build_tui_layout_children(**widgets)`

Override this only when you need full control over widget ordering. Most extensions should use `_get_extra_tui_widgets()` instead.

```python
def _build_tui_layout_children(self, *, sudo_widget, secret_widget,
    approval_widget, clarify_widget, model_picker_widget=None,
    spinner_widget=None, spacer, status_bar, input_rule_top,
    image_bar, input_area, input_rule_bot, voice_status_bar,
    completions_menu) -> list:
```

The default implementation returns (any `None` widgets are filtered out):

```python
[
    Window(height=0),       # anchor
    sudo_widget,            # sudo password prompt (conditional)
    secret_widget,          # secret input prompt (conditional)
    approval_widget,        # dangerous command approval (conditional)
    clarify_widget,         # clarify question UI (conditional)
    model_picker_widget,    # model picker overlay (conditional)
    spinner_widget,         # thinking spinner (conditional)
    spacer,                 # fills remaining vertical space
    *self._get_extra_tui_widgets(),  # YOUR WIDGETS GO HERE
    status_bar,             # model/token/context status line
    input_rule_top,         # ─── border above input
    image_bar,              # attached images indicator
    input_area,             # user text input
    input_rule_bot,         # ─── border below input
    voice_status_bar,       # voice mode status (conditional)
    completions_menu,       # autocomplete dropdown
]
```

## Layout diagram

The default layout from top to bottom:

1. **Output area** — scrolling conversation history
2. **Spacer**
3. **Extra widgets** — from `_get_extra_tui_widgets()`
4. **Status bar** — model, context %, elapsed time
5. **Image bar** — attached image count
6. **Input area** — user prompt
7. **Voice status** — recording indicator
8. **Completions menu** — autocomplete suggestions

## Tips

- **Invalidate the display** after state changes: call `self._invalidate()` to trigger a prompt_toolkit redraw.
- **Access agent state**: `self.agent`, `self.model`, `self.conversation_history` are all available.
- **Custom styles**: Override `_build_tui_style_dict()` and add entries for your custom style classes.
- **Slash commands**: Override `process_command()`, handle your commands, and call `super().process_command(cmd)` for everything else.
- **Don't override `run()`** unless absolutely necessary — the extension hooks exist specifically to avoid that coupling.

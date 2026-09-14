# Local Patches — KenseiAgent Fork

Applied 2026-05-31. Must re-apply after any upstream `git pull` that touches `cli.py` status-bar methods.

## Patch: Provider + Reasoning Effort in CLI Status Bar

**File:** `cli.py`
**Why:** Hermes status bar hardcodes model-only display. No plugin hook or config-driven template exists for status bar segments.
**What it adds:**
- `_get_status_bar_snapshot()` captures `provider_name` + `reasoning_effort`
- `_build_status_bar_text()` shows `🧠{effort}` in narrow+medium widths; `({provider})` + `🧠{effort}` in wide width
- `_get_status_bar_fragments()` renders the styled prompt_toolkit fragments with same badges

**Lines touched:** ~+57 in `cli.py` across 3 methods
**Conflict risk:** Medium — `cli.py` is a high-churn file. `git pull` will likely conflict on these hunks.
**Re-apply workflow:**
```bash
cd ~/repos/KenseiAgent
git stash          # or git diff > /tmp/patch.diff; git checkout cli.py
git pull
# re-apply manually or via patch -p1 < /tmp/patch.diff
```

**Upstream alternative:** Open a feature request on hermes-agent for configurable status bar segments (e.g. `display.status_bar_fields: [model, provider, reasoning_effort, context, duration]`). This patch should be retired once upstream supports it.

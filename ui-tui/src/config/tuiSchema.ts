/**
 * Hermes TUI design schema — adapted from OpenCode `tui.json`
 * (https://opencode.ai/tui.json).
 *
 * This file documents the configuration surface available to the TUI.
 * Values flow: ~/.hermes/config.yaml → Python RPC → ConfigDisplayConfig →
 * useConfigSync → $uiState → components.
 *
 * All keys live under `display:` in config.yaml and are prefixed `tui_`
 * to distinguish TUI-only settings from platform-agnostic display settings.
 */

// ── Theme ───────────────────────────────────────────────────────────────

/**
 * Visual theme for the TUI.
 * Maps to: `display.skin` in config.yaml
 * OpenCode: `theme` (string)
 * Hermes: `display.skin` (string — "default", "ares", "mono", "slate", or custom YAML skin)
 */

// ── Scroll ──────────────────────────────────────────────────────────────

/**
 * Scroll speed multiplier.
 * Maps to: `display.tui_scroll_speed` (number, 1–20, default 1)
 * OpenCode: `scroll_speed` (number)
 * Env override: `HERMES_TUI_SCROLL_SPEED`
 */
export interface TuiScrollConfig {
  /** Scroll speed multiplier (1–20). Default 1. */
  speed: number
  /** Enable scroll acceleration on sustained wheel events. Default true. */
  acceleration: boolean
}

// ── Prompt ──────────────────────────────────────────────────────────────

/**
 * Prompt composer dimensions.
 * Maps to: `display.tui_prompt_max_height`, `display.tui_prompt_max_width`
 * OpenCode: `prompt.max_height` (integer), `prompt.max_width` (integer | "auto")
 */
export interface TuiPromptConfig {
  /** Maximum height of the prompt textarea in rows. Default 10. */
  maxHeight: number
  /** Maximum width of the prompt ("auto" = 0 = no limit, or integer columns). Default 0. */
  maxWidth: number
}

// ── Diff ────────────────────────────────────────────────────────────────

/**
 * Diff rendering style.
 * Maps to: `display.tui_diff_style`
 * OpenCode: `diff_style` ("auto" | "stacked")
 */
export type TuiDiffStyle = 'auto' | 'stacked'

// ── Attention ───────────────────────────────────────────────────────────

/**
 * Attention / notification settings.
 * Maps to: `display.tui_attention_enabled`, `display.bell_on_complete`
 * OpenCode: `attention.enabled` (boolean), `attention.sound` (boolean),
 *   `attention.volume` (number 0-1), `attention.sounds` (object)
 */
export interface TuiAttentionConfig {
  /** Enable attention notifications. Default false. */
  enabled: boolean
}

// ── Leader key ──────────────────────────────────────────────────────────

/**
 * Leader key timeout.
 * Maps to: `display.tui_leader_timeout`
 * OpenCode: `leader_timeout` (integer, ms)
 */
// Default: 1000ms

// ── Mouse ───────────────────────────────────────────────────────────────

/**
 * Mouse capture.
 * Maps to: `display.tui_mouse_enabled`, `display.mouse_tracking`
 * OpenCode: `mouse` (boolean, default true)
 */
// Default: true. Falls back to `display.mouse_tracking`.

// ── Keybinds ────────────────────────────────────────────────────────────

/**
 * Keybindings.
 * OpenCode: `keybinds` (~130 properties with rich key format)
 * Hermes: Currently hardcoded in components (useInputHandlers.ts, textInput.tsx,
 *   prompts.tsx, app.tsx, etc.). Making these configurable is tracked as a
 *   follow-up task.
 *
 * OpenCode keybinding format (for reference):
 *   - false | "none" → disable
 *   - "ctrl+c" → simple string
 *   - { name: "c", ctrl: true } → object with modifiers
 *   - { key: { name: "c", ctrl: true }, event: "press" } → event control
 *   - ["ctrl+c", "ctrl+d"] → multiple bindings
 *
 * OpenCode categories (130+ bindings):
 *   - app (exit, debug, console, toggle_animations, toggle_file_context, …)
 *   - diff (close, toggle, expand, expand_all, collapse, next_hunk, …)
 *   - messages (page_up/down, line_up/down, first, last, next, previous, …)
 *   - prompt (submit, editor, context_clear, skills, stash, …)
 *   - input (clear, paste, submit, newline, move_*, select_*, delete_*, …)
 *   - session (export, copy, new, list, fork, rename, delete, interrupt, …)
 *   - model (provider_list, favorite_toggle, list, cycle_recent, …)
 *   - agent (list, cycle, cycle_reverse)
 *   - history (previous, next)
 *   - terminal (suspend, title_toggle)
 *   - which-key (toggle, layout_toggle, group_previous/next, …)
 */

// ── Hermes → OpenCode field mapping ─────────────────────────────────────

/**
 * Direct mappings from OpenCode `tui.json` to Hermes `config.yaml`:
 *
 * | OpenCode field            | Hermes config.yaml path          | Status        |
 * |---------------------------|----------------------------------|---------------|
 * | `theme`                   | `display.skin`                   | ✅ existing    |
 * | `scroll_speed`            | `display.tui_scroll_speed`       | ✅ new         |
 * | `scroll_acceleration.*`   | `display.tui_scroll_acceleration`| ✅ new         |
 * | `prompt.max_height`       | `display.tui_prompt_max_height`  | ✅ new         |
 * | `prompt.max_width`        | `display.tui_prompt_max_width`   | ✅ new         |
 * | `diff_style`              | `display.tui_diff_style`         | ✅ new         |
 * | `keybinds.*`              | (hardcoded)                      | ⏳ follow-up   |
 * | `leader_timeout`          | `display.tui_leader_timeout`     | ✅ new         |
 * | `attention.enabled`       | `display.tui_attention_enabled`  | ✅ new         |
 * | `attention.sound`         | `display.bell_on_complete`       | ✅ existing    |
 * | `mouse`                   | `display.tui_mouse_enabled`      | ✅ new         |
 * | `plugin`                  | `plugins.*`                      | ✅ existing    |
 */

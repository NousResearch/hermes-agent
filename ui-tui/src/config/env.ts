export const STARTUP_RESUME_ID = (process.env.HERMES_TUI_RESUME ?? '').trim()
export const MOUSE_TRACKING = !/^(?:1|true|yes|on)$/i.test((process.env.HERMES_TUI_DISABLE_MOUSE ?? '').trim())
export const NO_CONFIRM_DESTRUCTIVE = /^(?:1|true|yes|on)$/i.test((process.env.HERMES_TUI_NO_CONFIRM ?? '').trim())
// Inline mode: skip the alt-screen wrapper.  The TUI renders into the
// primary buffer so the terminal's native scrollback captures whatever
// scrolls off the top.  Wheel + PageUp are then handled by the host
// terminal, not by our virtual-scroll logic.  The live composer/progress
// area still pins to the bottom via Ink's normal flow.
//
// This is an experiment gate — the full "inline layout" (plain-text
// transcript with composer pinned below) is a bigger change; the env var
// here just disables AlternateScreen so we can measure whether native
// scrolling beats our virtualization on the same pipeline.
export const INLINE_MODE = /^(?:1|true|yes|on)$/i.test((process.env.HERMES_TUI_INLINE ?? '').trim())

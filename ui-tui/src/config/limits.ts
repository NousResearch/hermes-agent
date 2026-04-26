export const LARGE_PASTE = { chars: 8000, lines: 80 }
export const LIVE_RENDER_MAX_CHARS = 16_000
export const LIVE_RENDER_MAX_LINES = 240
export const LONG_MSG = 300
export const MAX_HISTORY = 800
export const THINKING_COT_MAX = 160
// Rows scrolled per wheel-notch event.
//
// One notch of a mechanical wheel emits multiple wheel events (3-5 per
// click in most terminals; trackpad flicks emit 100+). Each event scrolls
// WHEEL_SCROLL_STEP rows.  The product = rows-per-click.
//
// 1 = pure line-by-line.  Small per-event delta keeps Ink's DECSTBM fast
// path firing (each scroll < viewport-1) and produces smooth visible
// motion — the user can scan content mid-scroll.  We were at 6 before
// (= ~20-30 rows per notch) which visually teleported and forced the
// virtualization to reshape the mount range on every event.
//
// If this feels sluggish on precision scrolls, porting claude-code's
// wheel accel state machine (ScrollKeybindingHandler.tsx) is the right
// next step — it ramps step up during sustained fast clicks and decays
// on pause.
export const WHEEL_SCROLL_STEP = 1

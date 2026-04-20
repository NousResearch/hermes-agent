export const LARGE_PASTE = { chars: 8000, lines: 80 };
export const LONG_MSG = 300;
export const MAX_HISTORY = 800;
export const THINKING_COT_MAX = 160;
export const WHEEL_SCROLL_STEP = 3;
// Memory optimization limits to prevent OOM crashes
// Max characters per message to render (100KB ≈ 25-35K tokens)
// Messages larger than this will be truncated in display
export const MAX_MSG_CHARS = 100000;
// Max lines to render in a single message block
export const MAX_MSG_LINES = 2000;

// -----------------------------------------------------------------------------
// TTS: MEDIA: tag detection via WebSocket intercept.
// xterm writes to WebGL canvas so DOM scanning doesn't work.
// Intercept at the wire: ws.onmessage → check → term.write
// -----------------------------------------------------------------------------
const _mediaCache = new Map<string, string>();

export function setupTTS() {
  // Grab the current term.write wrapper (the instance is available on the window
  // during testing, or we can access it via the local module closure).
}

const originalOnmessage = window?.__hermes_ws_onmessage;

export function interceptMEDIA(): void {
  // Called after term is initialized to wrap the event handler.
}

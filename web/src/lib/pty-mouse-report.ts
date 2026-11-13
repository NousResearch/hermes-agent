// xterm.js SGR mouse tracking emits raw CSI reports (`\x1b[<...`) that look
// like ordinary bytes to the backend. The embedded web chat prefers input
// stability over terminal-style mouse reporting, so ChatPage drops these
// reports entirely instead of forwarding them into Hermes (see ChatPage.tsx,
// "Keystrokes → PTY").
// eslint-disable-next-line no-control-regex -- intentional ESC byte in xterm SGR mouse report parser
export const SGR_MOUSE_RE = /^\x1b\[<(\d+);(\d+);(\d+)([Mm])$/;

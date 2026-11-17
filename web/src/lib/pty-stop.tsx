import { useLayoutEffect, useRef, useState } from "react";

type Binding = { sid: string; epoch: string; generation: number };

function validBinding(b: unknown): b is Binding | null {
  if (b === null) return true;
  if (!b || typeof b !== "object") return false;
  const value = b as Binding;
  return typeof value.sid === "string" && !!value.sid && typeof value.epoch === "string" && !!value.epoch && Number.isSafeInteger(value.generation) && value.generation >= 0;
}
type Socket = { readyState: number; send: (data: string) => void };

/** Per-terminal OSC control channel. The query nonce scopes replies to a socket. */
export function createPtyStop(changed: (snapshot: { enabled: boolean; error: string }) => void) {
  let socket: Socket | null = null;
  let nonce = "";
  let confirmed = false;
  let pending = false;
  let bound: Binding | null = null;
  let error = "";
  let isCurrent = () => false;
  const enabled = () => Boolean(socket && socket.readyState === 1 && isCurrent() && confirmed && bound && !pending);
  const notify = () => changed({ enabled: enabled(), error });
  const frame = (value: object) => `\x1b]6973;${JSON.stringify(value)}\x07`;
  const close = () => {
    socket = null;
    nonce = "";
    confirmed = false;
    pending = false;
    bound = null;
    notify();
  };
  return {
    open(next: Socket, current: () => boolean = () => true) {
      close();
      socket = next;
      isCurrent = current;
      error = "";
      try {
        nonce = Array.from(crypto.getRandomValues(new Uint8Array(16)), n => n.toString(16).padStart(2, "0")).join("");
        next.send(frame({ action: "query", nonce }));
      } catch {
        error = "Could not check whether the response can be stopped.";
        socket = null;
      }
      notify();
    },
    close,
    receive(data: string) {
      if (!socket || socket.readyState !== 1 || !isCurrent()) return true;
      try {
        const value: unknown = JSON.parse(data);
        if (!value || typeof value !== "object") return true;
        const state = value as Record<string, unknown>;
        if (state.action !== "state" || (state.nonce !== undefined && state.nonce !== nonce)) return true;
        if (!confirmed && state.nonce !== nonce) return true;
        const b = state.bound;
        if (!validBinding(b)) return true;
        confirmed = true;
        bound = b as Binding | null;
        if (state.nonce === nonce) pending = false;
        notify();
      } catch { /* Ignore malformed OSC payloads. */ }
      return true;
    },
    enabled,
    error: () => error,
    stop() {
      if (!enabled() || !socket || !bound) return;
      pending = true;
      try {
        socket.send(frame({ action: "stop", ...bound, nonce }));
      } catch {
        pending = false;
        error = "Could not send the stop request.";
      }
      notify();
    },
  };
}

export function usePtyStop(scope: string, connected: boolean, hostRef: React.RefObject<HTMLDivElement | null>) {
  const [snapshot, update] = useState({ enabled: false, error: "" });
  const [stop] = useState(() => createPtyStop(update));
  const scopeRef = useRef(scope);
  useLayoutEffect(() => { scopeRef.current = scope; }, [scope]);
  const button = <>
    <div className="flex items-center gap-2 pb-1">
      <button type="button" aria-label="Stop response" disabled={!connected || !snapshot.enabled} onClick={stop.stop} className="rounded border border-current/30 bg-black/70 px-2 py-1 text-xs text-white disabled:opacity-40 focus-visible:outline focus-visible:outline-2">Stop</button>
      {snapshot.error && <span role="alert" className="text-xs text-warning">{snapshot.error}</span>}
    </div>
    <div ref={hostRef} role="region" aria-label="Chat terminal" className="hermes-chat-xterm-host min-h-0 min-w-0 flex-1" />
  </>;
  return { stop, scopeRef, button };
}

/** Initial PTY dimensions also serve as application-level keepalive. */
export function startPtyKeepalive(ws: WebSocket, term: { cols: number; rows: number }, interval: number) {
  const resize = () => {
    if (ws.readyState === WebSocket.OPEN) ws.send(`\x1b[RESIZE:${term.cols};${term.rows}]`);
  };
  resize();
  return setInterval(resize, interval);
}

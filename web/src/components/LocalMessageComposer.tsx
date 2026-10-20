import { useEffect, useRef, useState } from "react";

type Props = {
  socket: WebSocket | null;
  connected: boolean;
  ended: boolean;
  draftKey: string;
  onTerminalMode: () => void;
};

export function LocalMessageComposer({ socket, connected, ended, draftKey, onTerminalMode }: Props) {
  const [value, setValue] = useState(() => {
    try { return window.sessionStorage.getItem(draftKey) ?? ""; } catch { return ""; }
  });
  const [pending, setPending] = useState(false);
  const [notice, setNotice] = useState("");
  const composing = useRef(false);
  const submitting = useRef(false);
  const sendTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => () => {
    if (sendTimer.current !== null) clearTimeout(sendTimer.current);
  }, [socket]);

  useEffect(() => {
    if (notice !== "Sent to terminal") return;
    const timer = setTimeout(() => setNotice(""), 2000);
    return () => clearTimeout(timer);
  }, [notice]);

  useEffect(() => {
    try {
      if (value && !ended) window.sessionStorage.setItem(draftKey, value);
      else window.sessionStorage.removeItem(draftKey);
    } catch {
      // Storage is optional and must not disable chat input.
    }
  }, [draftKey, ended, value]);

  useEffect(() => {
    if (!socket) return;
    const release = () => {
      if (sendTimer.current !== null) clearTimeout(sendTimer.current);
      submitting.current = false;
      setPending(false);
    };
    const onClose = () => {
      if (submitting.current) {
        release();
        setNotice("Connection closed. Check the terminal before retrying.");
      }
    };
    socket.addEventListener("close", onClose);
    return () => {
      socket.removeEventListener("close", onClose);
      release();
    };
  }, [socket]);

  const submit = () => {
    const text = value.replace(/\r\n?/g, "\n")
      .replace(/\x1b\[(?:200|201)~/g, "")
      .replace(/[\x00-\x08\x0b-\x1f\x7f-\x9f]/g, "");
    if (!text.trim() || submitting.current || composing.current || ended) return;
    if (!connected || !socket || socket.readyState !== WebSocket.OPEN) {
      setNotice("Chat is reconnecting. Your draft is kept.");
      return;
    }
    if (!window.confirm("Send to the terminal? This may interrupt a running task. Continue only when the normal message prompt is active, not an approval dialog or menu.")) return;

    submitting.current = true;
    setPending(true);
    setNotice("Submitting…");
    try {
      socket.send(`\x1b[200~${text}\x1b[201~`);
      sendTimer.current = setTimeout(() => {
        sendTimer.current = null;
        if (socket.readyState !== WebSocket.OPEN) {
          submitting.current = false;
          setPending(false);
          setNotice("Connection closed. Check the terminal before retrying.");
          return;
        }
        try {
          socket.send("\r");
          setNotice("Sent to terminal");
        } catch {
          setNotice("Submission uncertain. Check the terminal before retrying.");
        } finally {
          submitting.current = false;
          setPending(false);
        }
      }, 50);
    } catch {
      submitting.current = false;
      setPending(false);
      setNotice("Send failed. Your draft was kept.");
    }
  };

  return (
    <section aria-label="Message composer" className="flex min-w-0 shrink-0 flex-col gap-2">
      <textarea
        aria-label="Message"
        className="h-24 min-h-16 max-h-48 w-full resize-y rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground outline-none focus:ring-2 focus:ring-ring"
        disabled={pending}
        onChange={(event) => setValue(event.target.value)}
        onCompositionEnd={() => { composing.current = false; }}
        onCompositionStart={() => { composing.current = true; }}
        onKeyDown={(event) => {
          if (event.key === "Enter" && (event.ctrlKey || event.metaKey) && !composing.current && !event.nativeEvent.isComposing && !event.repeat) {
            event.preventDefault();
            submit();
          }
        }}
        placeholder="Write a message…"
        value={value}
      />
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span aria-live="polite" className="min-w-0 break-words text-xs text-muted-foreground">{notice}</span>
        <div className="ml-auto flex shrink-0 gap-2">
          <button className="rounded-md border border-border px-3 py-1.5 text-sm" onClick={onTerminalMode} type="button">Terminal</button>
          <button className="rounded-md border border-border px-3 py-1.5 text-sm" disabled={pending || !value} onClick={() => setValue("")} type="button">Clear draft</button>
          <button title="Ctrl/Cmd+Enter sends; Enter inserts a newline. Sending may interrupt a running task." className="rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground disabled:opacity-60" disabled={!value.trim() || pending || !connected || ended} onClick={submit} type="button">
            {pending ? "Submitting…" : "Send"}
          </button>
        </div>
      </div>
    </section>
  );
}

import { useEffect, useMemo, useRef, useState } from "react";
import { X } from "lucide-react";

export type ChatCommandPaletteProps = {
  onClose: () => void;
  onFocusComposer: () => void;
  onNewChat: () => void;
  onToggleSessions: () => void;
  onInsertPrompt: (prompt: string) => void;
  queuedCount: number;
  onClearQueue: () => void;
};

export function CommandPalette({
  onClose,
  onFocusComposer,
  onNewChat,
  onToggleSessions,
  onInsertPrompt,
  queuedCount,
  onClearQueue,
}: ChatCommandPaletteProps) {
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const commands = useMemo(() => [
    { id: "focus", label: "Focus composer", hint: "Jump to the message box", action: onFocusComposer },
    { id: "heartbeat", label: "Draft heartbeat command", hint: "Fill /heartbeat every 10m …", action: () => onInsertPrompt("/heartbeat every 10m ") },
    { id: "loop", label: "Draft recurring loop command", hint: "Fill /loop 10m …", action: () => onInsertPrompt("/loop 10m ") },
    { id: "new", label: "New chat", hint: "Start a fresh session", action: onNewChat },
    { id: "sessions", label: "Toggle sessions", hint: "Show or hide the session navigator", action: onToggleSessions },
    ...(queuedCount > 0 ? [{ id: "clear-queue", label: "Clear prompt queue", hint: `${queuedCount} queued prompt${queuedCount === 1 ? "" : "s"}`, action: onClearQueue }] : []),
  ], [onClearQueue, onFocusComposer, onInsertPrompt, onNewChat, onToggleSessions, queuedCount]);

  const visibleCommands = commands.filter((command) => `${command.label} ${command.hint}`.toLowerCase().includes(query.trim().toLowerCase()));

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  return (
    <div
      data-slot="command-palette"
      role="dialog"
      aria-modal="true"
      aria-label="Command palette"
      className="fixed inset-0 z-50 flex items-start justify-center bg-background/60 p-4 pt-[12vh] backdrop-blur-sm"
      onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}
    >
      <div className="w-full max-w-lg overflow-hidden rounded-md border border-border bg-card text-card-foreground shadow-xl">
        <div className="flex items-center gap-2 border-b border-border px-3 py-2">
          <label className="sr-only" htmlFor="native-chat-command-search">Search commands</label>
          <input
            ref={inputRef}
            id="native-chat-command-search"
            aria-label="Search commands"
            className="min-w-0 flex-1 bg-transparent px-1 py-2 text-[16px] outline-none placeholder:text-muted-foreground sm:text-sm"
            placeholder="Search commands…"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          <button type="button" aria-label="Close command palette" className="rounded p-1 text-muted-foreground hover:bg-accent hover:text-accent-foreground" onClick={onClose}>
            <X />
          </button>
        </div>
        <div className="max-h-80 overflow-y-auto p-2" role="listbox" aria-label="Chat commands">
          {visibleCommands.length > 0 ? visibleCommands.map((command) => (
            <button
              key={command.id}
              type="button"
              role="option"
              aria-label={command.label}
              className="flex w-full items-start gap-3 rounded px-3 py-2 text-left hover:bg-accent hover:text-accent-foreground focus-visible:outline-2 focus-visible:outline-ring"
              onClick={() => { command.action(); onClose(); }}
            >
              <span className="min-w-0 flex-1">
                <span className="block text-sm font-medium">{command.label}</span>
                <span className="block text-xs text-muted-foreground">{command.hint}</span>
              </span>
            </button>
          )) : (
            <p className="px-3 py-6 text-center text-sm text-muted-foreground">No matching commands</p>
          )}
        </div>
        <div className="border-t border-border px-3 py-2 text-xs text-muted-foreground">Press Esc to close</div>
      </div>
    </div>
  );
}

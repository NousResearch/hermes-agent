import { useCallback, useEffect, useRef, useState } from "react";
import { Send, RefreshCw, Bot, Terminal, FileCode, CheckCircle2, XCircle, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { codeApi } from "@/lib/codeApi";
import type { CodeSessionEvent, CodeCommand, CodeArtifact } from "@/types/code";

interface ChatPanelProps {
  codeSessionId: string | null;
}

type EventItem =
  | { kind: "event"; data: CodeSessionEvent }
  | { kind: "command"; data: CodeCommand }
  | { kind: "artifact"; data: CodeArtifact };

function formatTime(ts: string) {
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  } catch {
    return "--:--:--";
  }
}

export function ChatPanel({ codeSessionId }: ChatPanelProps) {
  const [events, setEvents] = useState<CodeSessionEvent[]>([]);
  const [commands, setCommands] = useState<CodeCommand[]>([]);
  const [artifacts, setArtifacts] = useState<CodeArtifact[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const load = useCallback(async () => {
    if (!codeSessionId) return;
    setLoading(true);
    setError(null);
    try {
      const [eventsData, commandsData, artifactsData] = await Promise.all([
        codeApi.getCodeSessionEvents(codeSessionId),
        codeApi.getCommands(codeSessionId),
        codeApi.getCodeSessionArtifacts(codeSessionId),
      ]);
      setEvents(eventsData.events);
      setCommands(commandsData.commands);
      setArtifacts(artifactsData.artifacts);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [codeSessionId]);

  const poll = useCallback(async () => {
    if (!codeSessionId) return;
    try {
      const [eventsData, commandsData] = await Promise.all([
        codeApi.getCodeSessionEvents(codeSessionId),
        codeApi.getCommands(codeSessionId),
      ]);
      setEvents(eventsData.events);
      setCommands(commandsData.commands);
    } catch {
      // Silently fail on polling
    }
  }, [codeSessionId]);

  useEffect(() => {
    if (!codeSessionId) {
      setEvents([]);
      setCommands([]);
      setArtifacts([]);
      return;
    }
    load();
    const interval = setInterval(poll, 5000);
    return () => clearInterval(interval);
  }, [codeSessionId, load, poll]);

  const handleSend = async () => {
    if (!input.trim() || !codeSessionId) return;
    setSending(true);
    try {
      await codeApi.sendEvent(codeSessionId, { type: "user_message", message: input.trim() });
      setInput("");
      await load();
    } catch (e) {
      setError(String(e));
    } finally {
      setSending(false);
    }
  };

  // Build timeline items sorted by time
  const items: EventItem[] = [
    ...events.map((e) => ({ kind: "event" as const, data: e })),
    ...commands.map((c) => ({ kind: "command" as const, data: c })),
    ...artifacts.map((a) => ({ kind: "artifact" as const, data: a })),
  ].sort((a, b) => {
    const ta = (a.data as { created_at?: string }).created_at ?? "";
    const tb = (b.data as { created_at?: string }).created_at ?? "";
    return ta.localeCompare(tb);
  });

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [items.length]);

  if (!codeSessionId) {
    return (
      <div className="flex flex-col items-center justify-center h-full py-16 text-center">
        <Bot className="h-10 w-10 text-muted-foreground/30 mb-3" />
        <p className="text-sm text-muted-foreground">No session selected</p>
        <p className="text-xs text-muted-foreground/70 mt-1">
          Select or create a session to start chatting
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-border shrink-0">
        <div className="flex items-center gap-2">
          <Bot className="h-4 w-4 text-muted-foreground" />
          <span className="text-xs font-medium">Session Chat</span>
        </div>
        <Button
          variant="ghost"
          size="sm"
          className="h-7 w-7 p-0"
          onClick={load}
          disabled={loading}
        >
          <RefreshCw className={`h-3 w-3 ${loading ? "animate-spin" : ""}`} />
        </Button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-2">
        {items.length === 0 && !loading && (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <Bot className="h-8 w-8 text-muted-foreground/30 mb-2" />
            <p className="text-xs text-muted-foreground">No events yet</p>
            <p className="text-[10px] text-muted-foreground/70 mt-1">
              Commands and artifacts will appear here
            </p>
          </div>
        )}

        {items.map((item) => {
          if (item.kind === "event") {
            return <EventItem key={item.data.id} event={item.data} />;
          } else if (item.kind === "command") {
            return <CommandItem key={item.data.id} command={item.data} />;
          } else {
            return <ArtifactItem key={item.data.id} artifact={item.data} />;
          }
        })}
        <div ref={bottomRef} />
      </div>

      {/* Error banner */}
      {error && (
        <div className="px-3 py-1.5 border-t border-border bg-destructive/10">
          <p className="text-[10px] text-destructive truncate">{error}</p>
        </div>
      )}

      {/* Input bar */}
      <div className="px-3 py-2 border-t border-border shrink-0">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
            placeholder="Send a message or request..."
            className="flex-1 px-3 py-1.5 text-xs border border-border rounded bg-background placeholder:text-muted-foreground focus:outline-none focus:border-foreground/30"
            disabled={sending}
          />
          <Button
            size="sm"
            className="h-8 px-3"
            onClick={handleSend}
            disabled={sending || !input.trim()}
          >
            {sending ? (
              <RefreshCw className="h-3 w-3 animate-spin" />
            ) : (
              <Send className="h-3 w-3" />
            )}
          </Button>
        </div>
        <p className="text-[9px] text-muted-foreground mt-1">
          Press Enter to send — results update automatically
        </p>
      </div>
    </div>
  );
}

function EventItem({ event }: { event: CodeSessionEvent }) {
  const icon = event.type.includes("command") ? (
    <Terminal className="h-3 w-3" />
  ) : event.type.includes("approval") ? (
    <AlertTriangle className="h-3 w-3 text-warning" />
  ) : (
    <Bot className="h-3 w-3" />
  );

  return (
    <div className="flex items-start gap-2">
      <div className="mt-0.5 text-muted-foreground shrink-0">{icon}</div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-0.5">
          <span className="text-[10px] font-mono text-muted-foreground">
            {formatTime(event.created_at)}
          </span>
          <Badge variant="outline" className="text-[9px] h-4">{event.type}</Badge>
        </div>
        {event.message && (
          <p className="text-xs text-foreground/80">{event.message}</p>
        )}
        {event.payload && (
          <pre className="text-[10px] font-mono text-muted-foreground mt-1 p-1.5 bg-muted/50 rounded overflow-x-auto">
            {JSON.stringify(event.payload, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
}

function CommandItem({ command }: { command: CodeCommand }) {
  const statusIcon =
    command.status === "completed" ? (
      <CheckCircle2 className="h-3 w-3 text-success" />
    ) : command.status === "failed" ? (
      <XCircle className="h-3 w-3 text-destructive" />
    ) : command.status === "running" ? (
      <RefreshCw className="h-3 w-3 animate-spin text-warning" />
    ) : (
      <Terminal className="h-3 w-3 text-muted-foreground" />
    );

  const classColor =
    command.classification === "blocked"
      ? "border-l-2 border-l-destructive"
      : command.classification === "needs_approval"
        ? "border-l-2 border-l-warning"
        : "";

  return (
    <div className={`flex items-start gap-2 pl-2 py-1 ${classColor}`}>
      <div className="mt-0.5 shrink-0">{statusIcon}</div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-0.5">
          <span className="text-[10px] font-mono text-muted-foreground">
            {formatTime(command.created_at)}
          </span>
          <code className="text-[10px] font-mono bg-muted px-1.5 py-0.5 rounded truncate">
            {command.command}
          </code>
          {command.classification !== "safe" && (
            <Badge
              variant={command.classification === "blocked" ? "destructive" : "warning"}
              className="text-[9px] h-4"
            >
              {command.classification === "blocked" ? "Blocked" : "Needs approval"}
            </Badge>
          )}
        </div>
        {command.stdout && (
          <pre className="text-[10px] font-mono text-muted-foreground/80 mt-1 p-1.5 bg-muted/50 rounded overflow-x-auto max-h-20">
            {command.stdout.slice(-500)}
          </pre>
        )}
        {command.stderr && (
          <pre className="text-[10px] font-mono text-destructive/80 mt-1 p-1.5 bg-destructive/10 rounded overflow-x-auto max-h-16">
            {command.stderr.slice(-300)}
          </pre>
        )}
      </div>
    </div>
  );
}

function ArtifactItem({ artifact }: { artifact: CodeArtifact }) {
  return (
    <div className="flex items-start gap-2">
      <FileCode className="h-3 w-3 text-muted-foreground mt-0.5 shrink-0" />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-0.5">
          <span className="text-[10px] font-mono text-muted-foreground">
            {formatTime(artifact.created_at)}
          </span>
          <Badge
            variant={artifact.status === "added" ? "success" : artifact.status === "deleted" ? "destructive" : "outline"}
            className="text-[9px] h-4"
          >
            {artifact.status}
          </Badge>
        </div>
        <code className="text-[10px] font-mono text-foreground/80">{artifact.path}</code>
        <div className="flex gap-2 mt-0.5">
          {artifact.additions > 0 && (
            <span className="text-[10px] text-success">+{artifact.additions}</span>
          )}
          {artifact.deletions > 0 && (
            <span className="text-[10px] text-destructive">-{artifact.deletions}</span>
          )}
        </div>
      </div>
    </div>
  );
}

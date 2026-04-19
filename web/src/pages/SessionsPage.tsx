import { useEffect, useState, useCallback, useRef } from "react";
import {
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  MessageSquare,
  Search,
  Trash2,
  Clock,
  Terminal,
  Globe,
  MessageCircle,
  Hash,
  X,
} from "lucide-react";
import { H2 } from "@nous-research/ui/ui/components/typography/h2";
import { api } from "@/lib/api";
import type { SessionInfo, SessionMessage, SessionSearchResult } from "@/lib/api";
import { timeAgo } from "@/lib/utils";
import { Markdown } from "@/components/Markdown";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useI18n } from "@/i18n";

const SOURCE_CONFIG: Record<string, { icon: typeof Terminal; color: string }> = {
  cli: { icon: Terminal, color: "text-primary" },
  telegram: { icon: MessageCircle, color: "text-[oklch(0.65_0.15_250)]" },
  discord: { icon: Hash, color: "text-[oklch(0.65_0.15_280)]" },
  slack: { icon: MessageSquare, color: "text-[oklch(0.7_0.15_155)]" },
  whatsapp: { icon: Globe, color: "text-success" },
  cron: { icon: Clock, color: "text-warning" },
};

/** Render an FTS5 snippet with highlighted matches.
 *  The backend wraps matches in >>> and <<< delimiters. */
function SnippetHighlight({ snippet }: { snippet: string }) {
  const parts: React.ReactNode[] = [];
  const regex = />>>(.*?)<<</g;
  let last = 0;
  let match: RegExpExecArray | null;
  let i = 0;
  while ((match = regex.exec(snippet)) !== null) {
    if (match.index > last) {
      parts.push(snippet.slice(last, match.index));
    }
    parts.push(
      <mark key={i++} className="bg-warning/30 text-warning px-0.5">
        {match[1]}
      </mark>
    );
    last = regex.lastIndex;
  }
  if (last < snippet.length) {
    parts.push(snippet.slice(last));
  }
  return (
    <p className="text-xs text-muted-foreground/80 truncate max-w-lg mt-0.5">
      {parts}
    </p>
  );
}

function ToolCallBlock({ toolCall }: { toolCall: { id: string; function: { name: string; arguments: string } } }) {
  const [open, setOpen] = useState(false);
  const { t } = useI18n();

  let args = toolCall.function.arguments;
  try {
    args = JSON.stringify(JSON.parse(args), null, 2);
  } catch {
    // keep as-is
  }

  return (
    <div className="mt-2 border border-warning/20 bg-warning/5">
      <button
        type="button"
        className="flex w-full items-center gap-2 px-3 py-2 text-xs text-warning cursor-pointer hover:bg-warning/10 transition-colors"
        onClick={() => setOpen(!open)}
        aria-label={`${open ? t.common.collapse : t.common.expand} tool call ${toolCall.function.name}`}
      >
        {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        <span className="font-mono-ui font-medium">{toolCall.function.name}</span>
        <span className="text-warning/50 ml-auto">{toolCall.id}</span>
      </button>
      {open && (
        <pre className="border-t border-warning/20 px-3 py-2 text-xs text-warning/80 overflow-x-auto whitespace-pre-wrap font-mono">
          {args}
        </pre>
      )}
    </div>
  );
}

function MessageBubble({ msg, highlight }: { msg: SessionMessage; highlight?: string }) {
  const { t } = useI18n();

  const ROLE_STYLES: Record<string, { bg: string; text: string; label: string }> = {
    user: { bg: "bg-primary/10", text: "text-primary", label: t.sessions.roles.user },
    assistant: { bg: "bg-success/10", text: "text-success", label: t.sessions.roles.assistant },
    system: { bg: "bg-muted", text: "text-muted-foreground", label: t.sessions.roles.system },
    tool: { bg: "bg-warning/10", text: "text-warning", label: t.sessions.roles.tool },
  };

  const style = ROLE_STYLES[msg.role] ?? ROLE_STYLES.system;
  const label = msg.tool_name ? `${t.sessions.roles.tool}: ${msg.tool_name}` : style.label;

  // Check if any search term appears as a prefix of any word in content
  const isHit = (() => {
    if (!highlight || !msg.content) return false;
    const content = msg.content.toLowerCase();
    const terms = highlight.toLowerCase().split(/\s+/).filter(Boolean);
    return terms.some((term) => content.includes(term));
  })();

  // Split search query into terms for inline highlighting
  const highlightTerms = isHit && highlight
    ? highlight.split(/\s+/).filter(Boolean)
    : undefined;

  return (
    <div className={`${style.bg} p-3 ${isHit ? "ring-1 ring-warning/40" : ""}`} data-search-hit={isHit || undefined}>
      <div className="flex items-center gap-2 mb-1">
        <span className={`text-xs font-semibold ${style.text}`}>{label}</span>
        {isHit && (
          <Badge variant="warning" className="text-[9px] py-0 px-1.5">{t.common.match}</Badge>
        )}
        {msg.timestamp && (
          <span className="text-[10px] text-muted-foreground">{timeAgo(msg.timestamp)}</span>
        )}
      </div>
      {msg.content && (
        msg.role === "system"
          ? <div className="text-sm text-foreground whitespace-pre-wrap leading-relaxed">{msg.content}</div>
          : <Markdown content={msg.content} highlightTerms={highlightTerms} />
      )}
      {msg.tool_calls && msg.tool_calls.length > 0 && (
        <div className="mt-1">
          {msg.tool_calls.map((tc) => (
            <ToolCallBlock key={tc.id} toolCall={tc} />
          ))}
        </div>
      )}
    </div>
  );
}

/** Message list with auto-scroll to first search hit. */
function MessageList({ messages, highlight }: { messages: SessionMessage[]; highlight?: string }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!highlight || !containerRef.current) return;
    // Scroll to first hit after render
    const timer = setTimeout(() => {
      const hit = containerRef.current?.querySelector("[data-search-hit]");
      if (hit) {
        hit.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }, 50);
    return () => clearTimeout(timer);
  }, [messages, highlight]);

  return (
    <div ref={containerRef} className="flex flex-col gap-3 max-h-[600px] overflow-y-auto pr-2">
      {messages.map((msg, i) => (
        <MessageBubble key={i} msg={msg} highlight={highlight} />
      ))}
    </div>
  );
}

function SessionRow({
  session,
  snippet,
  searchQuery,
  isExpanded,
  onToggle,
  onDelete,
}: {
  session: SessionInfo;
  snippet?: string;
  searchQuery?: string;
  isExpanded: boolean;
  onToggle: () => void;
  onDelete: () => void;
}) {
  const [messages, setMessages] = useState<SessionMessage[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { t } = useI18n();

  useEffect(() => {
    if (isExpanded && messages === null && !loading) {
      setLoading(true);
      api
        .getSessionMessages(session.id)
        .then((resp) => setMessages(resp.messages))
        .catch((err) => setError(String(err)))
        .finally(() => setLoading(false));
    }
  }, [isExpanded, session.id, messages, loading]);

  const sourceInfo = (session.source ? SOURCE_CONFIG[session.source] : null) ?? { icon: Globe, color: "text-muted-foreground" };
  const SourceIcon = sourceInfo.icon;
  const hasTitle = session.title && session.title !== "Untitled";

  return (
    <div className={`border overflow-hidden transition-colors ${
      session.is_active
        ? "border-success/30 bg-success/[0.03]"
        : "border-border"
    }`}>
      <div
        className="flex items-center justify-between p-3 cursor-pointer hover:bg-secondary/30 transition-colors"
        onClick={onToggle}
      >
        <div className="flex items-center gap-3 min-w-0 flex-1">
          <div className={`shrink-0 ${sourceInfo.color}`}>
            <SourceIcon className="h-4 w-4" />
          </div>
          <div className="flex flex-col gap-0.5 min-w-0">
            <div className="flex items-center gap-2">
              <span className={`text-sm truncate pr-2 ${hasTitle ? "font-medium" : "text-muted-foreground italic"}`}>
                {hasTitle ? session.title : (session.preview ? session.preview.slice(0, 60) : t.sessions.untitledSession)}
              </span>
              {session.is_active && (
                <Badge variant="success" className="text-[10px] shrink-0">
                  <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
                  {t.common.live}
                </Badge>
              )}
            </div>
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <span className="truncate max-w-[120px] sm:max-w-[180px]">{(session.model ?? t.common.unknown).split("/").pop()}</span>
              <span className="text-border">&#183;</span>
              <span>{session.message_count} {t.common.msgs}</span>
              {session.tool_call_count > 0 && (
                <>
                  <span className="text-border">&#183;</span>
                  <span>{session.tool_call_count} {t.common.tools}</span>
                </>
              )}
              <span className="text-border">&#183;</span>
              <span>{timeAgo(session.last_active)}</span>
            </div>
            {snippet && (
              <SnippetHighlight snippet={snippet} />
            )}
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <Badge variant="outline" className="text-[10px]">
            {session.source ?? "local"}
          </Badge>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 text-muted-foreground hover:text-destructive"
            aria-label={t.sessions.deleteSession}
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      {isExpanded && (
        <div className="border-t border-border bg-background/50 p-4">
          {loading && (
            <div className="flex items-center justify-center py-8">
              <div className="h-5 w-5 animate-spin rounded-full border-2 border-primary border-t-transparent" />
            </div>
          )}
          {error && (
            <p className="text-sm text-destructive py-4 text-center">{error}</p>
          )}
          {messages && messages.length === 0 && (
            <p className="text-sm text-muted-foreground py-4 text-center">{t.sessions.noMessages}</p>
          )}
          {messages && messages.length > 0 && (
            <MessageList messages={messages} highlight={searchQuery} />
          )}
        </div>
      )}
    </div>
  );
}

export default function SessionsPage() {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const PAGE_SIZE = 20;
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [searchResults, setSearchResults] = useState<SessionSearchResult[] | null>(null);
  const [searching, setSearching] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(null);
  const { t } = useI18n();

  const loadSessions = useCallback((p: number) => {
    setLoading(true);
    api
      .getSessions(PAGE_SIZE, p * PAGE_SIZE)
      .then((resp) => {
        setSessions(resp.sessions);
        setTotal(resp.total);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    loadSessions(page);
  }, [loadSessions, page]);

  // Debounced FTS search
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);

    if (!search.trim()) {
      setSearchResults(null);
      setSearching(false);
      return;
    }

    setSearching(true);
    debounceRef.current = setTimeout(() => {
      api
        .searchSessions(search.trim())
        .then((resp) => setSearchResults(resp.results))
        .catch(() => setSearchResults(null))
        .finally(() => setSearching(false));
    }, 300);

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [search]);

  const handleDelete = async (id: string) => {
    try {
      await api.deleteSession(id);
      setSessions((prev) => prev.filter((s) => s.id !== id));
      setTotal((prev) => prev - 1);
      if (expandedId === id) setExpandedId(null);
    } catch {
      // ignore
    }
  };

  // Build snippet map from search results (session_id → snippet)
  const snippetMap = new Map<string, string>();
  if (searchResults) {
    for (const r of searchResults) {
      snippetMap.set(r.session_id, r.snippet);
    }
  }

  // When searching, filter sessions to those with FTS matches;
  // when not searching, show all sessions
  const filtered = searchResults
    ? sessions.filter((s) => snippetMap.has(s.id))
    : sessions;

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Header outside card for lighter feel */}
      <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:justify-between">
        <div className="flex items-center gap-2">
          <MessageSquare className="h-5 w-5 text-muted-foreground" />
          <H2 variant="sm">{t.sessions.title}</H2>
          <Badge variant="secondary" className="text-xs">
            {total}
          </Badge>
        </div>
        <div className="relative w-full sm:w-64">
          {searching ? (
            <div className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 animate-spin rounded-full border-[1.5px] border-primary border-t-transparent" />
          ) : (
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
          )}
          <Input
            placeholder={t.sessions.searchPlaceholder}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-8 pr-7 h-8 text-xs"
          />
          {search && (
            <button
              type="button"
              className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground cursor-pointer"
              onClick={() => setSearch("")}
            >
              <X className="h-3 w-3" />
            </button>
          )}
        </div>
      </div>

      {filtered.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
          <Clock className="h-8 w-8 mb-3 opacity-40" />
          <p className="text-sm font-medium">
            {search ? t.sessions.noMatch : t.sessions.noSessions}
          </p>
          {!search && (
            <p className="text-xs mt-1 text-muted-foreground/60">{t.sessions.startConversation}</p>
          )}
        </div>
      ) : (
        <>
          <div className="flex flex-col gap-1.5">
            {filtered.map((s) => (
              <SessionRow
                key={s.id}
                session={s}
                snippet={snippetMap.get(s.id)}
                searchQuery={search || undefined}
                isExpanded={expandedId === s.id}
                onToggle={() =>
                  setExpandedId((prev) => (prev === s.id ? null : s.id))
                }
                onDelete={() => handleDelete(s.id)}
              />
            ))}
          </div>

          {/* Pagination — hidden during search */}
          {!searchResults && total > PAGE_SIZE && (
            <div className="flex items-center justify-between pt-2">
              <span className="text-xs text-muted-foreground">
                {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, total)} {t.common.of} {total}
              </span>
              <div className="flex items-center gap-1">
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 w-7 p-0"
                  disabled={page === 0}
                  onClick={() => setPage((p) => p - 1)}
                  aria-label={t.sessions.previousPage}
                >
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                <span className="text-xs text-muted-foreground px-2">
                  {t.common.page} {page + 1} {t.common.of} {Math.ceil(total / PAGE_SIZE)}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 w-7 p-0"
                  disabled={(page + 1) * PAGE_SIZE >= total}
                  onClick={() => setPage((p) => p + 1)}
                  aria-label={t.sessions.nextPage}
                >
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

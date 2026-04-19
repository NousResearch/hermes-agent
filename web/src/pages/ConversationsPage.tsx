import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeft,
  Bot,
  Clock,
  Globe,
  Hash,
  Loader2,
  MessageCircle,
  MessagesSquare,
  RefreshCw,
  Search,
  Terminal,
  Trash2,
  User,
  X,
} from "lucide-react";
import { api, type ConversationInfo, type SessionMessage } from "@/lib/api";
import { Markdown } from "@/components/Markdown";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useI18n } from "@/i18n";

const ALL_SOURCES = "all";
const SEARCH_DEBOUNCE_MS = 250;
const MOBILE_BREAKPOINT = 1024;
const CONVERSATION_PAGE_SIZE = 200;

const SOURCE_ICON_CONFIG: Record<string, { icon: typeof Terminal; color: string }> = {
  cli: { icon: Terminal, color: "text-primary" },
  local: { icon: Terminal, color: "text-primary" },
  telegram: { icon: MessageCircle, color: "text-[oklch(0.65_0.15_250)]" },
  cron: { icon: Clock, color: "text-warning" },
  discord: { icon: Hash, color: "text-[oklch(0.65_0.15_280)]" },
  slack: { icon: MessagesSquare, color: "text-[oklch(0.7_0.15_155)]" },
  whatsapp: { icon: Globe, color: "text-success" },
  signal: { icon: MessageCircle, color: "text-[oklch(0.72_0.15_165)]" },
  matrix: { icon: MessagesSquare, color: "text-[oklch(0.72_0.12_215)]" },
  api_server: { icon: Globe, color: "text-[oklch(0.72_0.13_210)]" },
  email: { icon: MessagesSquare, color: "text-[oklch(0.76_0.1_80)]" },
  sms: { icon: MessageCircle, color: "text-[oklch(0.78_0.08_55)]" },
  dingtalk: { icon: MessageCircle, color: "text-[oklch(0.72_0.13_225)]" },
  wecom_callback: { icon: MessageCircle, color: "text-[oklch(0.74_0.12_205)]" },
  weixin: { icon: MessageCircle, color: "text-[oklch(0.74_0.14_150)]" },
  bluebubbles: { icon: MessageCircle, color: "text-[oklch(0.72_0.12_250)]" },
  homeassistant: { icon: Globe, color: "text-[oklch(0.72_0.11_45)]" },
  qqbot: { icon: MessageCircle, color: "text-[oklch(0.7_0.11_245)]" },
  webhook: { icon: Globe, color: "text-[oklch(0.78_0.05_215)]" },
  mattermost: { icon: MessagesSquare, color: "text-[oklch(0.72_0.12_285)]" },
  feishu: { icon: MessageCircle, color: "text-[oklch(0.74_0.12_235)]" },
  wecom: { icon: MessageCircle, color: "text-[oklch(0.74_0.12_190)]" },
  acp: { icon: MessagesSquare, color: "text-[oklch(0.78_0.05_260)]" },
};

function isMobileViewport(): boolean {
  return typeof window !== "undefined" && window.innerWidth < MOBILE_BREAKPOINT;
}

function shortModel(model: string | null | undefined): string {
  const value = String(model || "").trim();
  if (!value) return "—";
  const parts = value.split("/");
  return parts[parts.length - 1] || value;
}

function relativeTimeLabel(ts: number, locale: string): string {
  const delta = Math.round(ts - Date.now() / 1000);
  const absDelta = Math.abs(delta);
  const rtf = new Intl.RelativeTimeFormat(locale, { numeric: "auto" });

  if (absDelta < 60) return rtf.format(0, "second");
  if (absDelta < 3600) return rtf.format(Math.round(delta / 60), "minute");
  if (absDelta < 86400) return rtf.format(Math.round(delta / 3600), "hour");
  return rtf.format(Math.round(delta / 86400), "day");
}

function sourceLabel(source: string | null | undefined, labels: Record<string, string>): string {
  if (!source) return labels.unknown;
  return labels[source] ?? source;
}

function conversationId(conversation: ConversationInfo): string {
  return conversation.thread_root_id || conversation.id;
}

function groupConversations(conversations: ConversationInfo[]) {
  const active: ConversationInfo[] = [];
  const recent: ConversationInfo[] = [];
  for (const conversation of conversations) {
    if (conversation.is_active) active.push(conversation);
    else recent.push(conversation);
  }
  return { active, recent };
}

function previewText(conversation: ConversationInfo): string {
  return String(conversation.snippet || conversation.preview || "").trim();
}

function mergeConversationPages(existing: ConversationInfo[], incoming: ConversationInfo[]): ConversationInfo[] {
  if (!existing.length) return incoming;

  const seen = new Set(existing.map((conversation) => conversationId(conversation)));
  const merged = [...existing];

  for (const conversation of incoming) {
    const id = conversationId(conversation);
    if (seen.has(id)) continue;
    seen.add(id);
    merged.push(conversation);
  }

  return merged;
}

function ConversationRow({
  conversation,
  selected,
  disabled,
  onSelect,
}: {
  conversation: ConversationInfo;
  selected: boolean;
  disabled: boolean;
  onSelect: (id: string) => void;
}) {
  const { t, locale } = useI18n();
  const labels = t.conversations.sourceLabels;
  const preview = previewText(conversation);
  const sourceKey = conversation.source || "unknown";
  const sourceInfo = SOURCE_ICON_CONFIG[sourceKey] ?? { icon: Globe, color: "text-muted-foreground" };
  const SourceIcon = sourceInfo.icon;

  return (
    <button
      type="button"
      aria-pressed={selected}
      onClick={() => onSelect(conversationId(conversation))}
      disabled={disabled}
      className={`w-full border p-3 text-left transition-colors ${
        disabled
          ? "cursor-default opacity-70"
          : "cursor-pointer"
      } ${
        selected
          ? "border-foreground/25 bg-foreground/8"
          : disabled
            ? "border-border"
            : "border-border hover:bg-secondary/30"
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <SourceIcon className={`h-4 w-4 shrink-0 ${sourceInfo.color}`} />
            <span className={`truncate text-sm ${conversation.title ? "font-medium text-foreground" : "italic text-muted-foreground"}`}>
              {conversation.title || preview || t.common.untitled}
            </span>
            {conversation.is_active && (
              <Badge variant="success" className="shrink-0 text-[10px]">
                <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
                {t.common.live}
              </Badge>
            )}
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
            <span>{sourceLabel(conversation.source, labels)}</span>
            <span className="text-border">•</span>
            <span>{shortModel(conversation.model)}</span>
            <span className="text-border">•</span>
            <span>{relativeTimeLabel(conversation.last_active, locale)}</span>
          </div>
          {preview && (
            <p className="mt-2 max-h-12 overflow-hidden whitespace-pre-wrap break-words text-xs text-muted-foreground/85">
              {preview}
            </p>
          )}
        </div>
        <div className="flex shrink-0 flex-col items-end gap-1 text-right">
          <Badge variant="outline" className="text-[10px]">
            {conversation.thread_session_count} {t.conversations.linkedSessions}
          </Badge>
          <span className="text-[10px] text-muted-foreground">
            {conversation.thread_message_count} {t.common.msgs}
          </span>
        </div>
      </div>
    </button>
  );
}

function ConversationSection({
  title,
  conversations,
  selectedId,
  selectionLocked,
  onSelect,
}: {
  title: string;
  conversations: ConversationInfo[];
  selectedId: string | null;
  selectionLocked: boolean;
  onSelect: (id: string) => void;
}) {
  if (!conversations.length) return null;

  return (
    <section className="flex flex-col gap-2">
      <div className="text-xs font-display uppercase tracking-[0.16em] text-muted-foreground">
        {title}
      </div>
      <div className="flex flex-col gap-2">
        {conversations.map((conversation) => (
          <ConversationRow
            key={conversationId(conversation)}
            conversation={conversation}
            selected={selectedId === conversationId(conversation)}
            disabled={selectionLocked}
            onSelect={onSelect}
          />
        ))}
      </div>
    </section>
  );
}

function TranscriptMessage({ message }: { message: SessionMessage }) {
  const { t, locale } = useI18n();
  const isAssistant = message.role === "assistant";
  const RoleIcon = isAssistant ? Bot : User;
  const roleLabel = message.role === "assistant" ? t.sessions.roles.assistant : t.sessions.roles.user;

  return (
    <article className={`border p-3 ${isAssistant ? "border-success/20 bg-success/5" : "border-border bg-background/40"}`}>
      <div className="mb-2 flex items-center gap-2 text-xs uppercase tracking-[0.12em]">
        <RoleIcon className={`h-4 w-4 ${isAssistant ? "text-success" : "text-primary"}`} />
        <span className={isAssistant ? "text-success" : "text-primary"}>
          {roleLabel}
        </span>
        {message.timestamp && (
          <span className="text-muted-foreground">{relativeTimeLabel(message.timestamp, locale)}</span>
        )}
      </div>
      {isAssistant ? (
        <Markdown content={message.content || ""} />
      ) : (
        <p className="whitespace-pre-wrap break-words text-sm leading-relaxed text-foreground">
          {message.content || ""}
        </p>
      )}
    </article>
  );
}

function TranscriptPane({
  conversation,
  messages,
  visibleCount,
  threadSessionCount,
  loading,
  error,
  isMobile,
  deletePending,
  onBack,
  onDelete,
  onRetry,
}: {
  conversation: ConversationInfo | null;
  messages: SessionMessage[];
  visibleCount: number;
  threadSessionCount: number;
  loading: boolean;
  error: string;
  isMobile: boolean;
  deletePending: boolean;
  onBack: () => void;
  onDelete: () => void;
  onRetry: () => void;
}) {
  const { t, locale } = useI18n();

  if (!conversation) {
    return (
      <Card className="min-h-[28rem]">
        <CardContent className="flex min-h-[28rem] items-center justify-center">
          <div className="flex max-w-md flex-col items-center gap-3 text-center text-muted-foreground">
            <MessagesSquare className="h-8 w-8 opacity-40" />
            <p className="text-sm font-medium text-foreground/80">{t.conversations.selectConversation}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="min-h-[28rem]">
      <CardHeader className="gap-3">
        {isMobile && (
          <Button variant="ghost" size="sm" className="w-fit px-0" onClick={onBack}>
            <ArrowLeft className="h-4 w-4" />
            {t.conversations.backToList}
          </Button>
        )}
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            <CardDescription className="uppercase tracking-[0.16em]">
              {sourceLabel(conversation.source, t.conversations.sourceLabels)}
            </CardDescription>
            <CardTitle className="mt-1 normal-case tracking-normal text-base sm:text-lg">
              {conversation.title || previewText(conversation) || t.common.untitled}
            </CardTitle>
            <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
              <Badge variant="outline" className="text-[10px]">{t.conversations.transcript}</Badge>
              <Badge variant="outline" className="text-[10px]">{shortModel(conversation.model)}</Badge>
              <Badge variant="outline" className="text-[10px]">{visibleCount} {t.conversations.visibleMessages}</Badge>
              {threadSessionCount > 1 && (
                <Badge variant="outline" className="text-[10px]">{threadSessionCount} {t.conversations.linkedSessions}</Badge>
              )}
              <span>{t.conversations.updated} {relativeTimeLabel(conversation.last_active, locale)}</span>
            </div>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={onDelete}
            disabled={deletePending}
            aria-label={t.conversations.deleteConversation}
            aria-busy={deletePending}
          >
            {deletePending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
            {deletePending ? t.conversations.deletingConversation : t.conversations.deleteConversation}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="flex min-h-[18rem] items-center justify-center text-muted-foreground">
            <Loader2 className="h-5 w-5 animate-spin" />
            <span className="ml-2 text-sm">{t.conversations.loadingTranscript}</span>
          </div>
        ) : error ? (
          <div className="flex min-h-[18rem] flex-col items-center justify-center gap-3 text-center">
            <p className="max-w-lg text-sm text-destructive">{t.conversations.transcriptError} {error}</p>
            <Button variant="outline" size="sm" onClick={onRetry}>{t.common.retry}</Button>
          </div>
        ) : messages.length === 0 ? (
          <div className="flex min-h-[18rem] items-center justify-center text-center text-muted-foreground">
            <p className="text-sm">{t.conversations.noVisibleMessages}</p>
          </div>
        ) : (
          <div className="flex max-h-[32rem] flex-col gap-3 overflow-y-auto pr-1">
            {messages.map((message, index) => (
              <TranscriptMessage key={message.id ?? `${message.role}-${index}`} message={message} />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function ConversationsPage() {
  const { t } = useI18n();
  const [conversations, setConversations] = useState<ConversationInfo[]>([]);
  const [sourceOptions, setSourceOptions] = useState<string[]>([ALL_SOURCES]);
  const [allTotal, setAllTotal] = useState(0);
  const [filteredTotal, setFilteredTotal] = useState(0);
  const [searchInput, setSearchInput] = useState("");
  const [query, setQuery] = useState("");
  const [sourceFilter, setSourceFilter] = useState(ALL_SOURCES);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [messages, setMessages] = useState<SessionMessage[]>([]);
  const [visibleCount, setVisibleCount] = useState(0);
  const [threadSessionCount, setThreadSessionCount] = useState(1);
  const [listLoading, setListLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(false);
  const [nextOffset, setNextOffset] = useState(0);
  const [detailLoading, setDetailLoading] = useState(false);
  const [deletePending, setDeletePending] = useState(false);
  const [listError, setListError] = useState("");
  const [detailError, setDetailError] = useState("");
  const [isMobile, setIsMobile] = useState(isMobileViewport());
  const [mobileMode, setMobileMode] = useState<"list" | "detail">("list");
  const listRequestRef = useRef(0);
  const detailRequestRef = useRef(0);
  const selectedIdRef = useRef<string | null>(selectedId);
  const skipNextListReloadRef = useRef(false);

  const resetTranscriptState = useCallback(() => {
    setMessages([]);
    setVisibleCount(0);
    setThreadSessionCount(1);
    setDetailError("");
  }, []);

  const cancelTranscriptRequest = useCallback(() => {
    detailRequestRef.current += 1;
    resetTranscriptState();
    setDetailLoading(false);
  }, [resetTranscriptState]);

  const beginTranscriptRequest = useCallback(() => {
    resetTranscriptState();
    setDetailLoading(true);
  }, [resetTranscriptState]);

  const sourceLabels = t.conversations.sourceLabels;
  const groups = useMemo(() => groupConversations(conversations), [conversations]);
  const selectedConversation = useMemo(
    () => conversations.find((conversation) => conversationId(conversation) === selectedId) ?? null,
    [conversations, selectedId],
  );
  const searching = searchInput.trim() !== query;

  const loadConversations = useCallback(
    async ({ append = false, offset = 0 }: { append?: boolean; offset?: number } = {}) => {
      const requestId = listRequestRef.current + 1;
      listRequestRef.current = requestId;
      setListError("");
      if (append) {
        setLoadingMore(true);
      } else {
        setHasMore(false);
        setNextOffset(0);
        setListLoading(true);
      }

      try {
        const response = await api.getConversations({
          q: query,
          source: sourceFilter,
          limit: CONVERSATION_PAGE_SIZE,
          offset,
        });

        if (listRequestRef.current !== requestId) return null;

        const nextSessions = response.sessions || [];
        const nextSources = [ALL_SOURCES, ...response.sources.filter((value) => value && value !== ALL_SOURCES)];
        const responseOffset = typeof response.offset === "number" ? response.offset : offset;
        const followingOffset = responseOffset + nextSessions.length;
        const resetSourceFilter = !append && sourceFilter !== ALL_SOURCES && !nextSources.includes(sourceFilter);

        setConversations((current) => (append ? mergeConversationPages(current, nextSessions) : nextSessions));
        setSourceOptions(nextSources);
        setAllTotal(response.all_total || 0);
        setFilteredTotal(response.total || 0);
        setNextOffset(followingOffset);
        setHasMore(followingOffset < (response.total || 0));
        if (resetSourceFilter) {
          setSourceFilter(ALL_SOURCES);
        }

        return { sessions: nextSessions, resetSourceFilter };
      } catch (error) {
        if (listRequestRef.current !== requestId) return null;
        if (!append) {
          setConversations([]);
          setSourceOptions([ALL_SOURCES]);
          setSourceFilter(ALL_SOURCES);
          setAllTotal(0);
          setFilteredTotal(0);
          setHasMore(false);
          setNextOffset(0);
        }
        setListError(error instanceof Error ? error.message : String(error));
        return null;
      } finally {
        if (listRequestRef.current === requestId) {
          if (append) {
            setLoadingMore(false);
          } else {
            setListLoading(false);
          }
        }
      }
    },
    [query, sourceFilter],
  );

  const loadTranscript = useCallback(async (conversationRootId: string | null) => {
    if (!conversationRootId) {
      cancelTranscriptRequest();
      return;
    }

    const requestId = detailRequestRef.current + 1;
    detailRequestRef.current = requestId;
    beginTranscriptRequest();

    try {
      const response = await api.getConversationMessages(conversationRootId);
      if (detailRequestRef.current !== requestId) return;
      setMessages(response.messages || []);
      setVisibleCount(response.visible_count || 0);
      setThreadSessionCount(response.thread_session_count || 1);
    } catch (error) {
      if (detailRequestRef.current !== requestId) return;
      resetTranscriptState();
      setDetailError(error instanceof Error ? error.message : String(error));
    } finally {
      if (detailRequestRef.current === requestId) {
        setDetailLoading(false);
      }
    }
  }, [beginTranscriptRequest, cancelTranscriptRequest, resetTranscriptState]);

  useEffect(() => {
    selectedIdRef.current = selectedId;
  }, [selectedId]);

  useEffect(() => {
    if (deletePending) {
      return;
    }
    const timer = window.setTimeout(() => {
      setQuery(searchInput.trim());
    }, SEARCH_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [searchInput, deletePending]);

  useEffect(() => {
    const handleResize = () => {
      const mobile = isMobileViewport();
      setIsMobile(mobile);
      if (!mobile) {
        setMobileMode("detail");
      } else if (!selectedIdRef.current) {
        setMobileMode("list");
      }
    };

    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => {
    if (deletePending) {
      return;
    }
    if (skipNextListReloadRef.current) {
      skipNextListReloadRef.current = false;
      return;
    }
    loadConversations();
  }, [loadConversations, deletePending]);

  useEffect(() => {
    if (deletePending) {
      return;
    }
    if (!conversations.length) {
      if (selectedId !== null) setSelectedId(null);
      if (isMobile) setMobileMode("list");
      return;
    }

    const stillSelected = selectedId && conversations.some((conversation) => conversationId(conversation) === selectedId);
    if (!stillSelected) {
      setSelectedId(conversationId(conversations[0]));
    }
  }, [conversations, selectedId, isMobile, deletePending]);

  useLayoutEffect(() => {
    if (!selectedId || (isMobile && mobileMode !== "detail")) {
      return;
    }
    beginTranscriptRequest();
  }, [selectedId, isMobile, mobileMode, beginTranscriptRequest]);

  useEffect(() => {
    if (isMobile && mobileMode !== "detail") {
      cancelTranscriptRequest();
      return;
    }
    loadTranscript(selectedId);
  }, [selectedId, loadTranscript, isMobile, mobileMode, cancelTranscriptRequest]);

  const handleSelect = (id: string) => {
    if (id === selectedId) {
      if (isMobile && mobileMode === "list") {
        setMobileMode("detail");
      }
      return;
    }

    detailRequestRef.current += 1;
    resetTranscriptState();
    if (!isMobile || mobileMode === "detail") {
      setDetailLoading(true);
    }
    setSelectedId(id);
    if (isMobile) setMobileMode("detail");
  };

  const handleRefresh = async () => {
    if (deletePending) return;
    const refreshed = await loadConversations();
    const refreshedSessions = refreshed?.sessions;
    const currentSelectedId = selectedIdRef.current;
    if (
      currentSelectedId &&
      refreshedSessions?.some((conversation) => conversationId(conversation) === currentSelectedId) &&
      (!isMobile || mobileMode === "detail")
    ) {
      await loadTranscript(currentSelectedId);
    }
  };

  const handleLoadMore = async () => {
    if (deletePending || !hasMore || loadingMore || listLoading) return;
    await loadConversations({ append: true, offset: nextOffset });
  };

  const handleDelete = async () => {
    if (!selectedConversation || deletePending) return;
    if (!window.confirm(t.conversations.deleteConversationConfirm)) return;

    setDeletePending(true);
    skipNextListReloadRef.current = true;
    setDetailError("");
    try {
      await api.deleteConversation(conversationId(selectedConversation));
      if (isMobile) setMobileMode("list");
      const refreshed = await loadConversations();
      if (refreshed?.resetSourceFilter) {
        skipNextListReloadRef.current = false;
      }
    } catch (error) {
      setDetailError(error instanceof Error ? error.message : String(error));
    } finally {
      setDeletePending(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div className="flex items-start gap-3">
          <MessagesSquare className="mt-0.5 h-5 w-5 text-muted-foreground" />
          <div>
            <h1 className="text-base font-semibold text-foreground">{t.conversations.title}</h1>
            <p className="text-sm text-muted-foreground">{t.conversations.subtitle}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="secondary" className="text-xs">
            {filteredTotal}
          </Badge>
          {allTotal !== filteredTotal && (
            <Badge variant="outline" className="text-xs">
              {allTotal}
            </Badge>
          )}
          <Button variant="outline" size="sm" onClick={handleRefresh} disabled={deletePending || listLoading || loadingMore}>
            <RefreshCw className={`h-3.5 w-3.5 ${listLoading || loadingMore ? "animate-spin" : ""}`} />
            {t.common.refresh}
          </Button>
        </div>
      </div>

      <div className={`grid gap-4 ${isMobile ? "grid-cols-1" : "lg:grid-cols-[360px_minmax(0,1fr)]"}`}>
        {(!isMobile || mobileMode === "list") && (
          <Card>
            <CardHeader className="gap-3">
              <div>
                <CardTitle>{t.conversations.title}</CardTitle>
                <CardDescription>
                  {filteredTotal === allTotal ? `${filteredTotal}` : `${filteredTotal} / ${allTotal}`}
                </CardDescription>
              </div>
              <div className="relative">
                {searching ? (
                  <Loader2 className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 animate-spin text-primary" />
                ) : (
                  <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
                )}
                <Input
                  value={searchInput}
                  onChange={(event) => setSearchInput(event.target.value)}
                  placeholder={t.conversations.searchPlaceholder}
                  className="h-8 pl-8 pr-8 text-xs"
                  disabled={deletePending}
                />
                {searchInput && (
                  <button
                    type="button"
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground cursor-pointer"
                    onClick={() => setSearchInput("")}
                    aria-label={t.common.clear}
                    disabled={deletePending}
                  >
                    <X className="h-3 w-3" />
                  </button>
                )}
              </div>
              <div className="flex flex-wrap gap-2">
                {sourceOptions.map((source) => {
                  const selected = sourceFilter === source;
                  return (
                    <button
                      key={source}
                      type="button"
                      onClick={() => setSourceFilter(source)}
                      disabled={deletePending}
                      className={`border px-2 py-1 text-[0.65rem] uppercase tracking-[0.14em] transition-colors ${
                        deletePending ? "cursor-default opacity-70" : "cursor-pointer"
                      } ${
                        selected
                          ? "border-foreground/30 bg-foreground/10 text-foreground"
                          : "border-border text-muted-foreground hover:text-foreground hover:bg-secondary/30"
                      }`}
                    >
                      {sourceLabel(source, sourceLabels)}
                    </button>
                  );
                })}
              </div>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              {listLoading && !conversations.length ? (
                <div className="flex min-h-[20rem] items-center justify-center text-muted-foreground">
                  <Loader2 className="h-5 w-5 animate-spin" />
                  <span className="ml-2 text-sm">{t.conversations.loadingConversations}</span>
                </div>
              ) : listError && !conversations.length ? (
                <div className="flex min-h-[20rem] flex-col items-center justify-center gap-3 text-center">
                  <p className="max-w-sm text-sm text-destructive">{t.conversations.listError} {listError}</p>
                  <Button variant="outline" size="sm" onClick={() => { void loadConversations(); }}>{t.common.retry}</Button>
                </div>
              ) : conversations.length === 0 ? (
                <div className="flex min-h-[20rem] flex-col items-center justify-center gap-3 text-center text-muted-foreground">
                  <MessagesSquare className="h-8 w-8 opacity-40" />
                  <p className="text-sm font-medium text-foreground/80">
                    {query || sourceFilter !== ALL_SOURCES ? t.conversations.noMatch : t.conversations.noConversations}
                  </p>
                </div>
              ) : (
                <div className="flex flex-col gap-4">
                  <ConversationSection
                    title={t.conversations.active}
                    conversations={groups.active}
                    selectedId={selectedId}
                    selectionLocked={deletePending}
                    onSelect={handleSelect}
                  />
                  <ConversationSection
                    title={t.conversations.recent}
                    conversations={groups.recent}
                    selectedId={selectedId}
                    selectionLocked={deletePending}
                    onSelect={handleSelect}
                  />
                  {listError && (
                    <p className="text-sm text-destructive">{t.conversations.listError} {listError}</p>
                  )}
                  {hasMore && (
                    <Button variant="outline" size="sm" onClick={handleLoadMore} disabled={deletePending || listLoading || loadingMore}>
                      {loadingMore ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
                      {t.conversations.loadMore}
                    </Button>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {(!isMobile || mobileMode === "detail") && (
          <TranscriptPane
            conversation={selectedConversation}
            messages={messages}
            visibleCount={visibleCount}
            threadSessionCount={threadSessionCount}
            loading={detailLoading}
            error={detailError}
            isMobile={isMobile}
            deletePending={deletePending}
            onBack={() => {
              cancelTranscriptRequest();
              setMobileMode("list");
            }}
            onDelete={handleDelete}
            onRetry={() => loadTranscript(selectedId)}
          />
        )}
      </div>
    </div>
  );
}

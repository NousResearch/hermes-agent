/**
 * MobileChatPage — a phone-first chat UI for the Hermes dashboard.
 *
 * Unlike ChatPage.tsx (an embedded xterm.js terminal running the real TUI
 * over a PTY), this page speaks the plain JSON-RPC gateway protocol
 * directly (`session.create` / `prompt.submit` + streaming events) and
 * renders normal chat bubbles. No terminal emulation, no fixed-column
 * layout, no tiny font tiers — just responsive HTML that reflows properly
 * on a phone screen.
 *
 * Trade-off: this is a "simple chat" surface, not a full terminal. It does
 * not expose slash-command autocomplete, PTY resize semantics, or the
 * TUI's own rendering — it's for quick messaging + cron/status checks from
 * a phone, not full desktop-parity work.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router";
import {
  Send,
  AlertCircle,
  Wrench,
  RefreshCw,
  Paperclip,
  Square,
  MessageSquare,
  Cpu,
  X,
  Pencil,
  Check,
  Video,
} from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { createPortal } from "react-dom";

import { GatewayClient, type ConnectionState } from "@/lib/gatewayClient";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { usePageHeader } from "@/contexts/usePageHeader";
import { ChatSessionList } from "@/components/ChatSessionList";
import { ModelPickerDialog } from "@/components/ModelPickerDialog";

interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  text: string;
  status?: "streaming" | "done" | "error";
  /** Durable backend row id (from session.resume/session.history's `row_id`).
   * Only user turns carry one that's addressable for edit/rewind — undefined
   * means "not yet known" (freshly sent this session) or "not a user turn". */
  rowId?: number;
}

interface ToolActivity {
  id: string;
  name: string;
  status: "running" | "complete";
}

interface ClarifyBatchQuestion {
  qid: string;
  question: string;
  choices: string[] | null;
  multiSelect: boolean;
}

interface ClarifyRequest {
  requestId: string;
  question: string;
  choices: string[] | null;
  multiSelect: boolean;
  /** Present only for a batch clarify() call (multiple `questions` at once).
   * When set, the single question/choices fields above are unused. */
  batchQuestions?: ClarifyBatchQuestion[];
}

interface UsageSnapshot {
  context_used?: number;
  context_max?: number;
  context_percent?: number;
  total?: number;
}

const STATE_LABEL: Record<ConnectionState, string> = {
  idle: "idle",
  connecting: "connecting…",
  open: "connected",
  closed: "disconnected",
  error: "error",
};

const STATE_TONE: Record<
  ConnectionState,
  "secondary" | "warning" | "success" | "destructive"
> = {
  idle: "secondary",
  connecting: "warning",
  open: "success",
  closed: "secondary",
  error: "destructive",
};

function genId(prefix: string): string {
  return `${prefix}-${Math.random().toString(36).slice(2)}-${Date.now().toString(36)}`;
}

/** Compact human-readable token count: 950 -> "950", 12345 -> "12.3K". */
function formatTokenCount(n: number): string {
  if (n < 1000) return String(Math.round(n));
  return `${(n / 1000).toFixed(1)}K`;
}

export default function MobileChatPage() {
  // `?resume=<id>` mirrors ChatPage's convention so a session picked from
  // ChatSessionList (below) can be reopened here without leaving the page.
  const [searchParams, setSearchParams] = useSearchParams();
  const resumeParam = searchParams.get("resume");
  const [sessionsOpen, setSessionsOpen] = useState(false);
  const [modelOpen, setModelOpen] = useState(false);

  // Bumping `version` tears down and rebuilds the gateway client — same
  // pattern as ChatSidebar.tsx (reconnect button / scope change). Baking
  // `resumeParam` into the memo key means picking a different session (or
  // "New chat", which clears the param) transparently rebuilds the gateway
  // session too, without a separate effect to track the transition.
  const [version, setVersion] = useState(0);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const gw = useMemo(() => new GatewayClient(), [version, resumeParam]);

  const [state, setState] = useState<ConnectionState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [tools, setTools] = useState<ToolActivity[]>([]);
  const [usage, setUsage] = useState<UsageSnapshot>({});
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [uploadTargets, setUploadTargets] = useState<
    Array<{ id: string; label: string; accept: string | null }>
  >([]);
  const [uploading, setUploading] = useState(false);
  // Which target the hidden file input is about to upload to — set right
  // before the input is clicked, since one shared input serves every
  // configured target (a picker with 2+ upload buttons rather than one).
  const pendingUploadTargetIdRef = useRef<string | null>(null);
  // Drives the shared input's `accept` attribute so the native picker filters
  // appropriately per target (screenshots -> images, video -> video files).
  const [uploadAccept, setUploadAccept] = useState<string>("*/*");
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [clarifyRequest, setClarifyRequest] = useState<ClarifyRequest | null>(
    null,
  );
  const [clarifyAnswer, setClarifyAnswer] = useState("");
  const [batchAnswers, setBatchAnswers] = useState<Record<string, string>>({});
  const [batchFreeText, setBatchFreeText] = useState<Record<string, string>>({});
  const [interrupting, setInterrupting] = useState(false);
  // Row id of the user message currently being edited, or null. Only
  // messages with a known durable `rowId` (backfilled via refreshRowIds)
  // can be edited — editing rewinds the conversation to that point via
  // prompt.submit's truncate_before_row_id, same mechanism the desktop
  // Chat page's rewind/edit uses (tui_gateway/methods_prompt.py).
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null);
  const [editDraft, setEditDraft] = useState("");

  // Holds the latest refreshRowIds implementation so the message.complete
  // handler (bound once per gw instance, inside the effect below) always
  // calls the current version without needing to be in that effect's deps.
  const refreshRowIdsRef = useRef<() => void>(() => {});

  const scrollRef = useRef<HTMLDivElement | null>(null);
  const activeAssistantIdRef = useRef<string | null>(null);

  // Auto-scroll to the bottom on new content, unless the user has scrolled
  // up to read history (simple heuristic: only stick if already near bottom).
  const stickToBottomRef = useRef(true);

  useEffect(() => {
    const el = scrollRef.current;
    if (el && stickToBottomRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages, tools]);

  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const distanceFromBottom =
      el.scrollHeight - el.scrollTop - el.clientHeight;
    stickToBottomRef.current = distanceFromBottom < 80;
  }, []);

  useEffect(() => {
    let cancelled = false;

    const offState = gw.onState(setState);

    const offMessageStart = gw.on("message.start", () => {
      const id = genId("asst");
      activeAssistantIdRef.current = id;
      setMessages((prev) => [
        ...prev,
        { id, role: "assistant", text: "", status: "streaming" },
      ]);
    });

    const offMessageDelta = gw.on<{ text?: string }>("message.delta", (ev) => {
      const chunk = ev.payload?.text;
      if (!chunk) return;
      const id = activeAssistantIdRef.current;
      if (!id) return;
      setMessages((prev) =>
        prev.map((m) => (m.id === id ? { ...m, text: m.text + chunk } : m)),
      );
    });

    const offMessageComplete = gw.on<{ text?: string; status?: string }>(
      "message.complete",
      (ev) => {
        const id = activeAssistantIdRef.current;
        activeAssistantIdRef.current = null;
        setSending(false);
        setClarifyRequest(null);
        if (!id) return;
        const finalText = ev.payload?.text;
        const isError = ev.payload?.status === "error";
        setMessages((prev) =>
          prev.map((m) =>
            m.id === id
              ? {
                  ...m,
                  // Some completions carry the full rendered text; if the
                  // streamed deltas already built it up, don't duplicate.
                  text: finalText && !m.text ? finalText : m.text,
                  status: isError ? "error" : "done",
                }
              : m,
          ),
        );
        // Backfill durable rowIds on freshly-sent user turns (they have none
        // until persisted) so they become editable without a full page
        // reload. Best-effort — an older gateway or a transient failure just
        // means those turns stay non-editable this session.
        if (!isError) refreshRowIdsRef.current();
      },
    );

    const offToolStart = gw.on<{ name?: string }>("tool.start", (ev) => {
      const name = ev.payload?.name || "tool";
      setTools((prev) => [
        ...prev,
        { id: genId("tool"), name, status: "running" },
      ]);
    });

    const offToolComplete = gw.on<{ name?: string }>("tool.complete", (ev) => {
      const name = ev.payload?.name || "tool";
      setTools((prev) => {
        const idx = [...prev]
          .reverse()
          .findIndex((t) => t.name === name && t.status === "running");
        if (idx === -1) return prev;
        const realIdx = prev.length - 1 - idx;
        const next = [...prev];
        next[realIdx] = { ...next[realIdx], status: "complete" };
        return next;
      });
    });

    const offUsage = gw.on<UsageSnapshot>("session.usage", (ev) => {
      if (ev.payload) setUsage(ev.payload);
    });

    const offError = gw.on<{ message?: string }>("error", (ev) => {
      const message = ev.payload?.message;
      if (message) setError(message);
      setSending(false);
      setClarifyRequest(null);
      setBatchAnswers({});
      setBatchFreeText({});
    });

    const offClarify = gw.on<{
      request_id?: string;
      question?: string;
      choices?: string[] | null;
      multi_select?: boolean;
      questions?: Array<{
        qid: string;
        question: string;
        choices?: string[] | null;
        multi_select?: boolean;
      }> | null;
    }>("clarify.request", (ev) => {
      const p = ev.payload;
      if (!p?.request_id) return;
      if (p.questions && p.questions.length > 0) {
        setClarifyRequest({
          requestId: p.request_id,
          question: "",
          choices: null,
          multiSelect: false,
          batchQuestions: p.questions.map((q) => ({
            qid: q.qid,
            question: q.question,
            choices: Array.isArray(q.choices) ? q.choices : null,
            multiSelect: Boolean(q.multi_select),
          })),
        });
        stickToBottomRef.current = true;
        return;
      }
      setClarifyRequest({
        requestId: p.request_id,
        question: p.question || "Hermes needs clarification:",
        choices: Array.isArray(p.choices) ? p.choices : null,
        multiSelect: Boolean(p.multi_select),
      });
      stickToBottomRef.current = true;
    });

    gw.connect()
      .then(() => {
        if (cancelled) return;
        if (resumeParam) {
          // Reopen an existing conversation in place — mirrors ChatPage's
          // `?resume=<id>` contract. session.resume returns the full
          // rendered transcript so it can be painted immediately, unlike
          // session.create which always starts empty.
          return gw
            .request<{
              session_id: string;
              messages?: Array<{
                role?: string;
                text?: string;
                display_kind?: string;
                row_id?: number;
              }>;
            }>("session.resume", { session_id: resumeParam, source: "mobile_chat" })
            .then((result) => {
              if (cancelled || !result) return;
              setSessionId(result.session_id);
              const seeded = (result.messages || [])
                .filter((m) => m.role === "user" || m.role === "assistant" || m.role === "system")
                .map((m) => ({
                  id: genId(m.role || "msg"),
                  role: m.role as ChatMessage["role"],
                  text: m.text || "",
                  status: "done" as const,
                  rowId: typeof m.row_id === "number" ? m.row_id : undefined,
                }));
              setMessages(seeded);
              stickToBottomRef.current = true;
            })
            .catch((e: Error) => {
              // A stale/deleted resume target must not strand the page with
              // no session at all — fall back to a fresh one and drop the
              // bad param so a retry doesn't loop on the same failure.
              if (cancelled) return;
              setError(`Couldn't resume that session (${e.message}); started a new chat instead.`);
              setSearchParams(
                (prev) => {
                  const next = new URLSearchParams(prev);
                  next.delete("resume");
                  return next;
                },
                { replace: true },
              );
              return gw
                .request<{ session_id: string }>("session.create", { source: "mobile_chat" })
                .then((result) => {
                  if (cancelled || !result) return;
                  setSessionId(result.session_id);
                });
            });
        }
        return gw.request<{ session_id: string }>("session.create", {
          source: "mobile_chat",
        }).then((result) => {
          if (cancelled || !result) return;
          setSessionId(result.session_id);
        });
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e.message);
      });

    return () => {
      cancelled = true;
      offState();
      offMessageStart();
      offMessageDelta();
      offMessageComplete();
      offToolStart();
      offToolComplete();
      offUsage();
      offError();
      offClarify();
      gw.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gw]);

  useEffect(() => {
    let cancelled = false;
    api
      .getUploadTargets()
      .then((res) => {
        if (!cancelled)
          setUploadTargets(
            (res.targets || []).map((t) => ({
              id: t.id,
              label: t.label,
              accept: t.accept ?? null,
            })),
          );
      })
      .catch(() => {
        // Non-fatal — no upload targets configured, or the endpoint isn't
        // reachable yet. The upload button(s) simply won't appear.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const addSystemMessage = useCallback((text: string, isError = false) => {
    setMessages((prev) => [
      ...prev,
      {
        id: genId("sys"),
        role: "system",
        text,
        status: isError ? "error" : "done",
      },
    ]);
    stickToBottomRef.current = true;
  }, []);

  // Backfills rowId on local user messages that don't have one yet (freshly
  // sent this session — the durable id doesn't exist until the turn
  // persists). Matches positionally against session.history's durable user
  // turns, which is safe here because mobile chat only ever appends new
  // turns at the tail in order — never reorders or interleaves. Best-effort:
  // any failure just leaves those turns non-editable until the next resume.
  const refreshRowIds = useCallback(() => {
    if (!sessionId) return;
    gw.request<{
      messages?: Array<{ role?: string; display_kind?: string; row_id?: number }>;
    }>("session.history", { session_id: sessionId })
      .then((result) => {
        const durableUserRowIds = (result?.messages || [])
          .filter(
            (m) =>
              m.role === "user" &&
              !m.display_kind &&
              typeof m.row_id === "number" &&
              Number.isInteger(m.row_id),
          )
          .map((m) => m.row_id as number);
        setMessages((prev) => {
          let userIdx = 0;
          return prev.map((m) => {
            if (m.role !== "user") return m;
            const rowId = durableUserRowIds[userIdx];
            userIdx += 1;
            return rowId !== undefined && m.rowId !== rowId ? { ...m, rowId } : m;
          });
        });
      })
      .catch(() => {
        // Non-fatal — those turns just stay non-editable this session.
      });
  }, [sessionId, gw]);

  useEffect(() => {
    refreshRowIdsRef.current = refreshRowIds;
  }, [refreshRowIds]);

  const pickUploadFile = useCallback(
    (targetId: string, accept: string | null) => {
      pendingUploadTargetIdRef.current = targetId;
      setUploadAccept(accept || "*/*");
      fileInputRef.current?.click();
    },
    [],
  );

  const handleFilesSelected = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      // IMPORTANT: `e.target.files` is a *live* FileList tied to the input
      // element — resetting `e.target.value` clears the underlying
      // selection, and since the FileList is live (not a snapshot), that
      // wipes it out too. Snapshot into a plain array FIRST, then reset
      // the input. Getting this order backwards was the 2026-08-22 bug
      // where picking a file (single or multiple) silently did nothing:
      // `files` ended up empty by the time it was read.
      const files = Array.from(e.target.files ?? []);
      e.target.value = ""; // now safe: allows re-picking the same file(s)
      if (files.length === 0) return;

      // Multiple upload targets (screenshots, video, ...) share one hidden
      // input; pickUploadFile stamps which target this selection is for
      // right before opening the native picker.
      const targetId = pendingUploadTargetIdRef.current;
      const target = uploadTargets.find((t) => t.id === targetId) ?? uploadTargets[0];
      if (!target) {
        addSystemMessage("No upload target configured.", true);
        return;
      }

      setUploading(true);
      let succeeded = 0;
      const failures: string[] = [];

      // Sequential, not parallel: keeps the picker's progress readable
      // one line at a time and avoids hammering the tower with N
      // concurrent multipart uploads over a phone connection. On any
      // single-file failure, keep going through the rest of the batch
      // and report every outcome at the end rather than aborting.
      for (const file of files) {
        addSystemMessage(`Uploading ${file.name} to ${target.label}…`);
        try {
          const result = await api.uploadToTarget(target.id, file);
          succeeded += 1;
          addSystemMessage(
            `Uploaded ${result.filename} (${(result.size / 1024).toFixed(0)} KB) to ${target.label}.`,
          );
        } catch (err) {
          const reason = err instanceof Error ? err.message : String(err);
          failures.push(`${file.name}: ${reason}`);
          addSystemMessage(`Upload failed: ${file.name} — ${reason}`, true);
        }
      }

      if (files.length > 1) {
        addSystemMessage(
          failures.length === 0
            ? `All ${succeeded} files uploaded to ${target.label}.`
            : `${succeeded} of ${files.length} files uploaded to ${target.label}. ${failures.length} failed: ${failures.join("; ")}`,
          failures.length > 0,
        );
      }

      setUploading(false);
    },
    [uploadTargets, addSystemMessage],
  );

  const reconnect = useCallback(() => {
    setError(null);
    setMessages([]);
    setTools([]);
    setUsage({});
    setSessionId(null);
    setClarifyRequest(null);
    setClarifyAnswer("");
    setBatchAnswers({});
    setBatchFreeText({});
    setInterrupting(false);
    setEditingMessageId(null);
    setEditDraft("");
    setVersion((v) => v + 1);
  }, []);

  // Starts a brand-new conversation in place — clears `?resume` (if any) so
  // the connect effect's session.create path runs instead of session.resume.
  // Mirrors ChatSessionList's onNewChat contract used on the desktop Chat
  // page (see startFreshDashboardChat in ChatPage.tsx).
  const startNewChat = useCallback(() => {
    setError(null);
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        next.delete("resume");
        return next;
      },
      { replace: false },
    );
  }, [setSearchParams]);


  const respondToClarify = useCallback(
    (answer: string) => {
      const request = clarifyRequest;
      if (!request) return;
      setClarifyRequest(null);
      setClarifyAnswer("");
      setBatchAnswers({});
      setBatchFreeText({});
      gw.request("clarify.respond", {
        request_id: request.requestId,
        answer,
      }).catch((e: Error) => {
        setError(e.message);
      });
    },
    [clarifyRequest, gw],
  );

  // Batch clarify: each question locks in independently (server tracks which
  // qids are answered and auto-releases the turn once every qid is locked —
  // see tui_gateway/server.py `_respond`'s per-question path). We optimistically
  // mark the qid answered locally so the card visually confirms the tap, and
  // clear the whole card once every question has a locked answer.
  const respondToBatchQuestion = useCallback(
    (qid: string, answer: string) => {
      const request = clarifyRequest;
      if (!request?.batchQuestions) return;
      setBatchAnswers((prev) => ({ ...prev, [qid]: answer }));
      gw.request("clarify.respond", {
        request_id: request.requestId,
        question_id: qid,
        answer,
      })
        .then((res) => {
          const remaining = (res as { remaining?: string[] } | undefined)
            ?.remaining;
          if (remaining && remaining.length === 0) {
            setClarifyRequest(null);
            setBatchAnswers({});
            setBatchFreeText({});
          }
        })
        .catch((e: Error) => {
          setError(e.message);
        });
    },
    [clarifyRequest, gw],
  );

  const interruptTurn = useCallback(() => {
    if (!sessionId) return;
    setInterrupting(true);
    gw.request("session.interrupt", { session_id: sessionId })
      .catch((e: Error) => {
        setError(e.message);
      })
      .finally(() => {
        // The backend's interrupt.ack / message.complete events will flip
        // `sending` off on their own; this local flag just debounces the
        // Stop button so a slow tap doesn't fire session.interrupt twice.
        setInterrupting(false);
        setClarifyRequest(null);
        setBatchAnswers({});
        setBatchFreeText({});
        setSending(false);
      });
  }, [sessionId, gw]);

  const sendMessage = useCallback(() => {
    const text = draft.trim();
    if (!text || !sessionId || sending) return;
    setMessages((prev) => [
      ...prev,
      { id: genId("user"), role: "user", text },
    ]);
    setDraft("");
    setSending(true);
    stickToBottomRef.current = true;
    gw.request("prompt.submit", { session_id: sessionId, text }).catch(
      (e: Error) => {
        setError(e.message);
        setSending(false);
      },
    );
  }, [draft, sessionId, sending, gw]);

  const startEdit = useCallback((m: ChatMessage) => {
    if (m.role !== "user" || m.rowId === undefined) return;
    setEditingMessageId(m.id);
    setEditDraft(m.text);
  }, []);

  const cancelEdit = useCallback(() => {
    setEditingMessageId(null);
    setEditDraft("");
  }, []);

  // Edit-and-resend: rewinds the conversation to (and including) the edited
  // turn — everything after it, including the model's replies, is dropped —
  // then resubmits the edited text as a fresh turn from that point. Same
  // `truncate_before_row_id` + `confirm_truncate` mechanism the desktop
  // Chat page's rewind/edit uses (tui_gateway/methods_prompt.py); mobile
  // only needs the row-id path since every editable bubble here already has
  // one (refreshRowIds backfills it after every completed turn).
  const submitEdit = useCallback(() => {
    const targetId = editingMessageId;
    const text = editDraft.trim();
    if (!targetId || !text || !sessionId || sending) return;
    const target = messages.find((m) => m.id === targetId);
    if (!target || target.rowId === undefined) return;

    setEditingMessageId(null);
    setEditDraft("");
    // Drop the edited turn and everything after it locally, then append the
    // freshly edited text as the new tail — mirrors what the rewind produces
    // server-side so the UI doesn't wait on a round trip to look right.
    setMessages((prev) => {
      const idx = prev.findIndex((m) => m.id === targetId);
      if (idx === -1) return prev;
      return [
        ...prev.slice(0, idx),
        { id: genId("user"), role: "user", text, rowId: undefined },
      ];
    });
    setSending(true);
    stickToBottomRef.current = true;
    gw.request("prompt.submit", {
      session_id: sessionId,
      text,
      confirm_truncate: true,
      truncate_before_row_id: target.rowId,
    }).catch((e: Error) => {
      setError(e.message);
      setSending(false);
    });
  }, [editingMessageId, editDraft, sessionId, sending, messages, gw]);

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    },
    [sendMessage],
  );

  const contextPercent = usage.context_percent;
  const contextTone =
    contextPercent == null
      ? "secondary"
      : contextPercent >= 85
        ? "destructive"
        : contextPercent >= 70
          ? "warning"
          : "success";

  const { setEnd } = usePageHeader();
  // Render the connection/context/build info into the PERSISTENT page
  // header (PageHeaderProvider's "end" slot) instead of this page's own
  // internal header row. The internal row lives inside MobileChatPage's
  // scrolling flex column; on phones that column can itself scroll (fixed
  // now, see PageHeaderProvider's usesFixedMain fix, but this is cheap
  // insurance) — the persistent header never scrolls at all, so the
  // context badge is guaranteed visible regardless of conversation length
  // or any future layout regression here.
  useEffect(() => {
    setEnd(
      <div className="flex min-w-0 items-center gap-2">
        <Badge tone={STATE_TONE[state]}>{STATE_LABEL[state]}</Badge>
        {contextPercent != null && (
          <Badge
            tone={contextTone}
            title={
              usage.context_used != null && usage.context_max != null
                ? `${formatTokenCount(usage.context_used)} / ${formatTokenCount(usage.context_max)} tokens used`
                : "Context window usage"
            }
          >
            ctx {contextPercent}%
            {usage.context_used != null && usage.context_max != null
              ? ` (${formatTokenCount(usage.context_used)}/${formatTokenCount(usage.context_max)})`
              : ""}
          </Badge>
        )}
      </div>,
    );
    return () => setEnd(null);
  }, [setEnd, state, contextPercent, contextTone, usage.context_used, usage.context_max]);

  const runningTools = tools.filter((t) => t.status === "running");

  return (
    <div className="flex h-full min-h-0 w-full flex-col">
      {/* Header: build stamp + turn controls. Connection state and the
          context-usage badge now live in the persistent page header (see
          the setEnd effect above) so they're always visible regardless of
          scroll position — previously they lived only here, which scrolled
          off-screen as the conversation grew (2026-08-22). */}
      <div className="flex shrink-0 items-center justify-between gap-2 border-b border-border px-3 py-2">
        <span
          className="truncate text-[10px] text-muted-foreground"
          title="Build timestamp of the JS currently loaded in this tab/app — use this to confirm an update actually landed."
        >
          build {__HERMES_BUILD_TIME__}
        </span>
        <div className="flex items-center gap-1">
          {sending && (
            <Button
              ghost
              size="sm"
              onClick={interruptTurn}
              disabled={interrupting}
              title="Stop the current turn"
              aria-label="Stop"
            >
              <Square className="size-4" />
            </Button>
          )}
          <Button
            ghost
            size="sm"
            onClick={() => setSessionsOpen(true)}
            title="Sessions"
            aria-label="Sessions"
          >
            <MessageSquare className="size-4" />
          </Button>
          <Button
            ghost
            size="sm"
            onClick={() => setModelOpen(true)}
            disabled={!sessionId}
            title="Switch model"
            aria-label="Switch model"
          >
            <Cpu className="size-4" />
          </Button>
          <Button ghost size="sm" onClick={reconnect} title="Reconnect">
            <RefreshCw className="size-4" />
          </Button>
        </div>
      </div>

      {error && (
        <div className="flex shrink-0 items-center gap-2 border-b border-border bg-destructive/10 px-3 py-2 text-sm text-destructive">
          <AlertCircle className="size-4 shrink-0" />
          <span className="truncate">{error}</span>
        </div>
      )}

      {/* Message list */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 min-h-0 overflow-y-auto px-3 py-3"
      >
        <div className="flex flex-col gap-3">
          {messages.map((m) => {
            const isEditing = editingMessageId === m.id;
            const canEdit =
              m.role === "user" && m.rowId !== undefined && !sending && !clarifyRequest;
            return (
              <div
                key={m.id}
                className={cn(
                  "flex",
                  m.role === "user" ? "justify-end" : "justify-start",
                )}
              >
                {isEditing ? (
                  <div className="flex w-[85%] flex-col gap-2 rounded-lg border border-primary/50 bg-background p-2">
                    <p className="text-[10px] text-muted-foreground">
                      Editing will remove this and everything after it, then
                      resend.
                    </p>
                    <textarea
                      autoFocus
                      value={editDraft}
                      onChange={(e) => setEditDraft(e.target.value)}
                      rows={1}
                      className="min-h-[2.25rem] max-h-32 resize-none rounded-md border border-input bg-background px-2 py-1.5 text-sm outline-none focus:ring-1 focus:ring-ring"
                    />
                    <div className="flex justify-end gap-2">
                      <Button ghost size="sm" onClick={cancelEdit}>
                        Cancel
                      </Button>
                      <Button
                        size="sm"
                        onClick={submitEdit}
                        disabled={!editDraft.trim()}
                      >
                        <Check className="mr-1 size-3.5" />
                        Resend
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div
                    className={cn(
                      "group relative max-w-[85%] whitespace-pre-wrap break-words rounded-lg px-3 py-2 text-sm",
                      m.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : m.role === "system"
                          ? "bg-muted text-muted-foreground text-xs italic"
                          : "bg-card text-card-foreground",
                      m.status === "error" && "border border-destructive",
                      canEdit && "pr-8",
                    )}
                  >
                    {m.text || (m.status === "streaming" ? "…" : "")}
                    {canEdit && (
                      <button
                        type="button"
                        onClick={() => startEdit(m)}
                        aria-label="Edit and resend"
                        title="Edit and resend"
                        className="absolute right-1 top-1 rounded p-1 text-primary-foreground/70 hover:bg-black/10 hover:text-primary-foreground"
                      >
                        <Pencil className="size-3" />
                      </button>
                    )}
                  </div>
                )}
              </div>
            );
          })}

          {runningTools.map((t) => (
            <div key={t.id} className="flex justify-start">
              <div className="flex items-center gap-2 rounded-lg bg-muted px-3 py-2 text-xs text-muted-foreground">
                <Wrench className="size-3 animate-pulse" />
                <span>using {t.name}…</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Clarify prompt: renders inline above the composer when the agent
          calls the clarify tool. Without this the turn hangs forever —
          there is no other way to answer it from this surface (2026-08-22
          incident: a clarify.request with no UI left a session
          permanently stuck showing "using clarify..." with no recourse
          but a server restart).
          max-h + overflow-y-auto caps how much of the fixed-height mobile
          shell this card can claim — without a cap, a long question or a
          wide row of choice buttons pushed the message list (the only
          flexible region above it) down to almost nothing, hiding the
          conversation (2026-08-22 report: "menu shrinks the chat"). The
          card scrolls internally instead of growing unbounded. */}
      {clarifyRequest && (
        <div className="flex max-h-[60vh] shrink-0 flex-col gap-2 overflow-y-auto border-t border-border bg-muted/40 p-3">
          {clarifyRequest.batchQuestions ? (
            <>
              <p className="text-xs text-muted-foreground">
                Hermes has {clarifyRequest.batchQuestions.length} questions —
                answer each below.
              </p>
              {clarifyRequest.batchQuestions.map((q, idx) => {
                const locked = batchAnswers[q.qid] !== undefined;
                return (
                  <div
                    key={q.qid}
                    className={cn(
                      "flex flex-col gap-2 rounded-md border border-border/60 p-2",
                      locked && "opacity-60",
                    )}
                  >
                    <p className="text-sm font-medium">
                      {idx + 1}. {q.question}
                      {locked && (
                        <span className="ml-2 text-xs font-normal text-success">
                          ✓ answered
                        </span>
                      )}
                    </p>
                    {q.choices && q.choices.length > 0 ? (
                      <div className="flex flex-wrap gap-2">
                        {q.choices.map((choice) => (
                          <Button
                            key={choice}
                            size="sm"
                            ghost
                            disabled={locked}
                            onClick={() => respondToBatchQuestion(q.qid, choice)}
                          >
                            {choice}
                          </Button>
                        ))}
                      </div>
                    ) : null}
                    <div className="flex items-end gap-2">
                      <textarea
                        value={batchFreeText[q.qid] ?? ""}
                        onChange={(e) =>
                          setBatchFreeText((prev) => ({
                            ...prev,
                            [q.qid]: e.target.value,
                          }))
                        }
                        disabled={locked}
                        placeholder={
                          q.choices ? "Or type your own answer…" : "Type your answer…"
                        }
                        rows={1}
                        className="min-h-[2.25rem] max-h-24 flex-1 resize-none rounded-md border border-input bg-background px-2 py-1.5 text-sm outline-none focus:ring-1 focus:ring-ring disabled:opacity-50"
                      />
                      <Button
                        size="icon"
                        disabled={locked || !(batchFreeText[q.qid] ?? "").trim()}
                        onClick={() =>
                          respondToBatchQuestion(
                            q.qid,
                            (batchFreeText[q.qid] ?? "").trim(),
                          )
                        }
                        aria-label={`Send answer for question ${idx + 1}`}
                      >
                        <Send className="size-4" />
                      </Button>
                    </div>
                  </div>
                );
              })}
            </>
          ) : (
            <>
              <p className="text-sm font-medium">{clarifyRequest.question}</p>
              {clarifyRequest.choices && clarifyRequest.choices.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {clarifyRequest.choices.map((choice) => (
                    <Button
                      key={choice}
                      size="sm"
                      ghost
                      onClick={() => respondToClarify(choice)}
                    >
                      {choice}
                    </Button>
                  ))}
                </div>
              ) : null}
              <div className="flex items-end gap-2">
                <textarea
                  value={clarifyAnswer}
                  onChange={(e) => setClarifyAnswer(e.target.value)}
                  placeholder={
                    clarifyRequest.choices
                      ? "Or type your own answer…"
                      : "Type your answer…"
                  }
                  rows={1}
                  className="min-h-[2.5rem] max-h-32 flex-1 resize-none rounded-md border border-input bg-background px-3 py-2 text-base outline-none focus:ring-1 focus:ring-ring"
                />
                <Button
                  size="icon"
                  onClick={() => respondToClarify(clarifyAnswer.trim())}
                  disabled={!clarifyAnswer.trim()}
                  aria-label="Send answer"
                >
                  <Send className="size-4" />
                </Button>
              </div>
            </>
          )}
        </div>
      )}

      {/* Composer */}
      <div className="flex shrink-0 items-end gap-2 border-t border-border p-3">
        {uploadTargets.length > 0 && (
          <>
            <input
              ref={fileInputRef}
              type="file"
              accept={uploadAccept}
              multiple
              className="hidden"
              onChange={handleFilesSelected}
            />
            {uploadTargets.map((target) => {
              const isVideo = (target.accept || "").startsWith("video");
              const Icon = isVideo ? Video : Paperclip;
              return (
                <Button
                  key={target.id}
                  ghost
                  size="icon"
                  onClick={() => pickUploadFile(target.id, target.accept)}
                  disabled={uploading}
                  title={`Upload to ${target.label}`}
                  aria-label={`Upload to ${target.label}`}
                >
                  <Icon className="size-4" />
                </Button>
              );
            })}
          </>
        )}
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={
            clarifyRequest ? "Answer the question above first…" : "Message Hermes…"
          }
          rows={1}
          disabled={Boolean(clarifyRequest)}
          className="min-h-[2.5rem] max-h-32 flex-1 resize-none rounded-md border border-input bg-background px-3 py-2 text-base outline-none focus:ring-1 focus:ring-ring disabled:opacity-50"
        />
        <Button
          size="icon"
          onClick={sendMessage}
          disabled={!draft.trim() || !sessionId || sending || Boolean(clarifyRequest)}
          aria-label="Send"
        >
          <Send className="size-4" />
        </Button>
      </div>

      {/* Sessions sheet: reuses the same ChatSessionList component the
          desktop Chat page's side panel uses, so switching/creating
          sessions behaves identically everywhere. Portalled + full-screen
          on phone widths rather than a narrow sidebar. */}
      {sessionsOpen &&
        createPortal(
          <div
            className="fixed inset-0 z-[100] flex items-stretch justify-end bg-background/85"
            onClick={(e) => e.target === e.currentTarget && setSessionsOpen(false)}
            role="dialog"
            aria-modal="true"
            aria-label="Sessions"
          >
            <div className="flex h-full w-full max-w-sm flex-col border-l border-border bg-card shadow-2xl">
              <div className="flex shrink-0 items-center justify-between border-b border-border px-3 py-2">
                <span className="text-sm font-medium">Sessions</span>
                <Button
                  ghost
                  size="icon"
                  onClick={() => setSessionsOpen(false)}
                  aria-label="Close"
                >
                  <X className="size-4" />
                </Button>
              </div>
              <ChatSessionList
                activeSessionId={resumeParam}
                onPicked={() => setSessionsOpen(false)}
                onNewChat={startNewChat}
                className="flex-1"
              />
            </div>
          </div>,
          document.body,
        )}

      {modelOpen && sessionId && (
        <ModelPickerDialog
          gw={gw}
          sessionId={sessionId}
          onClose={() => setModelOpen(false)}
          title="Switch Model"
        />
      )}
    </div>
  );
}

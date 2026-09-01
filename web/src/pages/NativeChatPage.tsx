import {
  Fragment,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from "react";
import { GatewayClient, type ConnectionState, type GatewayEvent } from "@/lib/gatewayClient";
import { useProfileScope } from "@/contexts/useProfileScope";
import { useI18n } from "@/i18n";
import { cn } from "@/lib/utils";
import { ChatSessionList, type SessionActivityStatus } from "@/components/ChatSessionList";
import { SlashPopover, type SlashPopoverHandle } from "@/components/SlashPopover";
import { MarkdownMessage } from "@/components/chat/MarkdownMessage";
import { ToolActivity, type ToolActivityItem } from "@/components/chat/ToolActivity";
import { ApprovalCard, type ApprovalRequest } from "@/components/chat/ApprovalCard";
import { ClarificationCard, type ClarificationRequest } from "@/components/chat/ClarificationCard";
import { MessageActions } from "@/components/chat/MessageActions";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { ArrowDown, Menu, MessageSquare, Paperclip, RotateCcw, Send, Square, X } from "lucide-react";
import { useSearchParams } from "react-router";
import {
  nativeChatModelChoices,
  nativeChatSessionCreateParams,
  NATIVE_REASONING_OPTIONS,
  selectionFromSearchParams,
  type ModelOptionsCatalog,
  type NativeReasoningLevel,
} from "@/lib/native-chat-routing";

type TranscriptMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  streaming?: boolean;
};

type ResumeMessage = { role?: unknown; text?: unknown; content?: unknown; row_id?: unknown; id?: unknown };
type ApprovalSnapshot = { request_id?: unknown; command?: unknown; description?: unknown; choices?: unknown; allow_permanent?: unknown };
type ClarifySnapshot = { answers?: Record<string, string>; request_id?: unknown; question?: unknown; choices?: unknown; multi_select?: unknown; questions?: unknown };
type ResumeResponse = {
  session_id?: string;
  messages?: ResumeMessage[];
  messages_omitted?: boolean;
  running?: boolean;
  turn_started_at?: number | null;
  status?: string;
  info?: { running?: boolean; turn_started_at?: number | null; status?: string; stored_session_id?: string };
  pending_approval?: ApprovalSnapshot;
  pending_clarify?: ClarifySnapshot;
};

function snapshotText(message: ResumeMessage): string {
  if (typeof message.text === "string") return message.text;
  if (typeof message.content === "string") return message.content;
  if (Array.isArray(message.content)) {
    return message.content.map((part) => typeof part === "string" ? part : typeof part === "object" && part !== null && "text" in part ? String((part as { text?: unknown }).text ?? "") : "").join("");
  }
  return "";
}

function snapshotTranscript(messages: ResumeMessage[] | undefined): TranscriptMessage[] {
  return (messages ?? []).flatMap((message, index) => {
    const role = message.role === "user" ? "user" : message.role === "assistant" ? "assistant" : null;
    if (!role) return [];
    return [{ id: String(message.row_id ?? message.id ?? `snapshot-${index}`), role, text: snapshotText(message) }];
  });
}

function mergeSnapshotTranscript(snapshot: TranscriptMessage[], current: TranscriptMessage[]): TranscriptMessage[] {
  const result = [...snapshot];
  const represented = new Set(snapshot.map((message) => `${message.role}:${message.id}`));
  for (const message of current) {
    const same = result.some((item) => item.id === message.id
      || (item.role === message.role && item.text === message.text)
      || (item.role === "assistant" && message.role === "assistant" && item.text && message.text
        && (item.text.startsWith(message.text) || message.text.startsWith(item.text))));
    if (!same && !represented.has(`${message.role}:${message.id}`)) result.push(message);
  }
  return result;
}

function approvalFromSnapshot(snapshot?: ApprovalSnapshot): ApprovalRequest | null {
  if (!snapshot || typeof snapshot.request_id !== "string") return null;
  return { request_id: snapshot.request_id, command: typeof snapshot.command === "string" ? snapshot.command : undefined, description: typeof snapshot.description === "string" ? snapshot.description : undefined, choices: Array.isArray(snapshot.choices) ? snapshot.choices.filter((x): x is string => typeof x === "string") : undefined, allow_permanent: snapshot.allow_permanent !== false };
}

function clarifyFromSnapshot(snapshot?: ClarifySnapshot): ClarificationRequest | null {
  if (!snapshot || typeof snapshot.request_id !== "string") return null;
  return { request_id: snapshot.request_id, question: typeof snapshot.question === "string" ? snapshot.question : undefined, choices: Array.isArray(snapshot.choices) ? snapshot.choices.filter((x): x is string => typeof x === "string") : null, multi_select: snapshot.multi_select === true, questions: Array.isArray(snapshot.questions) ? snapshot.questions as ClarificationRequest["questions"] : undefined, answers: snapshot.answers };
}
type TextPayload = { text?: unknown; message?: unknown; kind?: unknown; running?: unknown; turn_started_at?: unknown; status?: unknown; request_id?: unknown; answer?: unknown; question?: unknown; choices?: unknown; command?: unknown; description?: unknown; tool_id?: unknown; name?: unknown; context?: unknown; args?: unknown; result?: unknown; summary?: unknown; progress?: unknown; questions?: unknown; multi_select?: unknown; allow_permanent?: unknown; seq?: unknown };

type PendingAttachment = {
  id: string;
  file: File;
  state: "pending" | "uploading" | "attached" | "error";
  error?: string;
  refText?: string;
  refPath?: string;
};
type FailedPrompt = { id: string; text: string };

const MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024;

// Native-chat-specific copy is kept local because the existing web i18n
// contract does not have a chat namespace yet. Keep this list short until
// those keys can be added for every locale together.
const QUICK_PROMPTS = [
  "Summarize this text",
  "Explain a concept simply",
  "Draft an email",
  "Plan my next steps",
] as const;

function fileDataUrl(file: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error("Could not read attachment"));
    reader.onload = () => typeof reader.result === "string" ? resolve(reader.result) : reject(new Error("Could not read attachment"));
    reader.readAsDataURL(file);
  });
}

export function attachmentPromptText(text: string, attachments: PendingAttachment[]): string {
  const refs = attachments.filter((item) => item.state === "attached").flatMap((item) => [item.refText, item.refPath]).filter(Boolean);
  return refs.length ? [text, refs.join("\n")].filter(Boolean).join("\n\n") : text;
}

export function shouldSubmitComposerKey(key: string, shiftKey: boolean, isComposing: boolean): boolean {
  return key === "Enter" && !shiftKey && !isComposing;
}

function eventText(event: GatewayEvent): string {
  const payload = event.payload as TextPayload | undefined;
  if (typeof payload?.text === "string") return payload.text;
  if (typeof payload?.message === "string") return payload.message;
  return "";
}

export function shouldFollowTranscript(distanceFromBottom: number): boolean {
  return distanceFromBottom <= 96;
}

function connectionLabel(state: ConnectionState): string {
  return state === "open" ? "Connected" : state[0].toUpperCase() + state.slice(1);
}

interface NativeChatPageProps {
  /** Open the shell navigation drawer on compact/mobile layouts. */
  onOpenNavigation?: () => void;
}

export default function NativeChatPage({ onOpenNavigation }: NativeChatPageProps) {
  const { profile } = useProfileScope();
  const { t } = useI18n();
  const [searchParams, setSearchParams] = useSearchParams();
  const resumeParam = searchParams.get("resume");
  const routeModel = searchParams.get("model");
  const routeProvider = searchParams.get("provider");
  const routeReasoning = searchParams.get("reasoning");
  const [modelCatalog, setModelCatalog] = useState<ModelOptionsCatalog>({});
  const modelChoices = useMemo(() => nativeChatModelChoices(modelCatalog), [modelCatalog]);
  const routingSelection = useMemo(() => selectionFromSearchParams(
    routeModel, routeProvider, routeReasoning, modelChoices,
  ), [modelChoices, routeModel, routeProvider, routeReasoning]);
  const routingSelectionRef = useRef(routingSelection);
  routingSelectionRef.current = routingSelection;
  const gateway = useMemo(() => new GatewayClient(), []);
  const [connectionState, setConnectionState] = useState<ConnectionState>("idle");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const durableSessionIdRef = useRef<string | null>(resumeParam);
  const [freshGeneration, setFreshGeneration] = useState(0);
  const [draft, setDraft] = useState("");
  const [transcript, setTranscript] = useState<TranscriptMessage[]>([]);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [errorAction, setErrorAction] = useState<"reconnect" | "resend" | null>(null);
  const [failedPrompt, setFailedPrompt] = useState<FailedPrompt | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const submitInFlightRef = useRef(false);
  const stopInFlightRef = useRef(false);
  const [attachments, setAttachments] = useState<PendingAttachment[]>([]);
  const attachmentsRef = useRef<PendingAttachment[]>([]);
  const stagingRef = useRef(new Set<string>());
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messageSequenceRef = useRef(0);
  const composingRef = useRef(false);
  const assistantIdRef = useRef<string | null>(null);
  const [tools, setTools] = useState<ToolActivityItem[]>([]);
  const [approval, setApproval] = useState<ApprovalRequest | null>(null);
  const [clarify, setClarify] = useState<ClarificationRequest | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [turnStartedAt, setTurnStartedAt] = useState<number | null>(null);
  const [clockNow, setClockNow] = useState(() => Date.now());
  const transcriptRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const slashPopoverRef = useRef<SlashPopoverHandle>(null);
  const followTranscriptRef = useRef(true);
  const [showScrollToBottom, setShowScrollToBottom] = useState(false);
  const [mobileSessionNavigatorOpen, setMobileSessionNavigatorOpen] = useState(false);
  const wasOpenRef = useRef(false);
  const seenSeqRef = useRef(new Map<string, number>());
  const reconnectingRef = useRef(false);

  useEffect(() => {
    if (!followTranscriptRef.current) return;
    const element = transcriptRef.current;
    if (!element) return;
    const timer = window.setTimeout(() => {
      element.scrollTop = element.scrollHeight;
    }, 0);
    return () => window.clearTimeout(timer);
  }, [approval, clarify, error, status, tools, transcript]);

  const handleTranscriptScroll = useCallback(() => {
    const element = transcriptRef.current;
    if (!element) return;
    const distanceFromBottom = element.scrollHeight - element.scrollTop - element.clientHeight;
    const shouldFollow = shouldFollowTranscript(distanceFromBottom);
    followTranscriptRef.current = shouldFollow;
    setShowScrollToBottom(!shouldFollow);
  }, []);

  const scrollToBottom = useCallback(() => {
    const element = transcriptRef.current;
    if (!element) return;
    followTranscriptRef.current = true;
    element.scrollTop = element.scrollHeight;
    setShowScrollToBottom(false);
  }, []);

  const updateAttachments = useCallback((next: PendingAttachment[] | ((current: PendingAttachment[]) => PendingAttachment[])) => {
    setAttachments((current) => {
      const updated = typeof next === "function" ? next(current) : next;
      attachmentsRef.current = updated;
      return updated;
    });
  }, []);

  const stageAttachment = useCallback(async (item: PendingAttachment) => {
    if (!sessionIdRef.current) return;
    if (stagingRef.current.has(item.id)) return;
    stagingRef.current.add(item.id);
    updateAttachments((current) => current.map((entry) => entry.id === item.id ? { ...entry, state: "uploading", error: undefined } : entry));
    try {
      const dataUrl = await fileDataUrl(item.file);
      const result = item.file.type.startsWith("image/")
        ? await gateway.request<{ attached?: boolean; path?: string; ref_path?: string; ref_text?: string }>("image.attach_bytes", {
          session_id: sessionIdRef.current, content_base64: dataUrl.slice(dataUrl.indexOf(",") + 1), filename: item.file.name,
        })
        : await gateway.request<{ attached?: boolean; path?: string; ref_path?: string; ref_text?: string }>("file.attach", {
          session_id: sessionIdRef.current, name: item.file.name, path: "", data_url: dataUrl,
        });
      if (result.attached === false) throw new Error("Attachment was rejected");
      updateAttachments((current) => current.map((entry) => entry.id === item.id ? { ...entry, state: "attached", refText: result.ref_text, refPath: result.ref_path ?? result.path } : entry));
    } catch (reason: unknown) {
      updateAttachments((current) => current.map((entry) => entry.id === item.id ? { ...entry, state: "error", error: reason instanceof Error ? reason.message : String(reason) } : entry));
    } finally {
      stagingRef.current.delete(item.id);
    }
  }, [gateway, updateAttachments]);

  const addFiles = useCallback((files: FileList | File[]) => {
    const incoming = Array.from(files);
    const accepted: PendingAttachment[] = [];
    for (const file of incoming) {
      if (file.size > MAX_ATTACHMENT_BYTES) {
        setError(`${file.name} is too large (max 25 MB)`);
        continue;
      }
      accepted.push({ id: `${Date.now()}-${Math.random()}`, file, state: "pending" });
    }
    if (!accepted.length) return;
    updateAttachments((current) => [...current, ...accepted]);
  }, [stageAttachment, updateAttachments]);

  useEffect(() => {
    if (!sessionId) return;
    for (const item of attachments) {
      if (item.state === "pending") void stageAttachment(item);
    }
  }, [attachments, sessionId, stageAttachment]);

  useEffect(() => {
    const accept = (event: GatewayEvent, sessionBound = false): TextPayload | null => {
      if (sessionBound && (!sessionIdRef.current || event.session_id !== sessionIdRef.current)) return null;
      if (!sessionBound && sessionIdRef.current && event.session_id && event.session_id !== sessionIdRef.current) return null;
      const payload = (event.payload ?? {}) as TextPayload;
      const seq = typeof (event as GatewayEvent & { seq?: unknown }).seq === "number"
        ? (event as GatewayEvent & { seq?: number }).seq
        : payload.seq;
      if (event.session_id && typeof seq === "number") {
        const previous = seenSeqRef.current.get(event.session_id);
        if (previous !== undefined && seq <= previous) return null;
        seenSeqRef.current.set(event.session_id, seq);
      }
      return payload;
    };
    const applySessionSnapshot = (snapshot: ResumeResponse) => {
      const info = snapshot.info ?? snapshot;
      const running = info.running === true || snapshot.running === true;
      setStreaming(running);
      setTurnStartedAt(typeof info.turn_started_at === "number" ? info.turn_started_at * 1000 : null);
      if (typeof info.status === "string" && info.status) setStatus(info.status);
      else if (!running) setStatus("Ready");
      if (Array.isArray(snapshot.messages) && snapshot.messages_omitted !== true) {
        const next = snapshotTranscript(snapshot.messages);
        setTranscript((current) => mergeSnapshotTranscript(next, current));
      }
      // session.activate/resume explicitly expose these registries. They are
      // read-only recovery state; responses still go through the normal RPC.
      setApproval(approvalFromSnapshot(snapshot.pending_approval));
      setClarify(clarifyFromSnapshot(snapshot.pending_clarify));
      // The current backend does not include tool activity in the session
      // snapshot, so retain live tool cards rather than fabricating history.
    };
    const offState = gateway.onState((state) => {
      setConnectionState(state);
      if (state !== "open") {
        if (state === "connecting") setStatus("Reconnecting…");
        else if (state === "closed") setStatus("Disconnected");
        return;
      }
      if (!wasOpenRef.current) { wasOpenRef.current = true; return; }
      const sid = sessionIdRef.current;
      if (!sid || reconnectingRef.current) return;
      reconnectingRef.current = true;
      void gateway.request<ResumeResponse>("session.activate", { session_id: sid, omit_messages: false })
        .then((snapshot) => applySessionSnapshot(snapshot))
        .catch(() => gateway.request<ResumeResponse>("session.resume", { session_id: durableSessionIdRef.current ?? sid, omit_messages: false, ...(profile ? { profile } : {}) }))
        .then((snapshot) => { if (snapshot) applySessionSnapshot(snapshot); })
        .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason)))
        .finally(() => { reconnectingRef.current = false; });
    });
    const offStart = gateway.on("message.start", (event) => { const payload = accept(event, true); if (!payload) return; const id = `assistant-${++messageSequenceRef.current}`; assistantIdRef.current = id; setStreaming(true); setTurnStartedAt((started) => started ?? Date.now()); setStatus("Thinking…"); setTranscript((messages) => [...messages, { id, role: "assistant", text: "", streaming: true }]); });
    const offDelta = gateway.on("message.delta", (event) => { const payload = accept(event, true); if (!payload) return; const text = eventText(event); if (!text) return; setTranscript((messages) => { const id = assistantIdRef.current; if (!id) return [...messages, { id: `assistant-${++messageSequenceRef.current}`, role: "assistant", text, streaming: true }]; return messages.map((message) => message.id === id ? { ...message, text: message.text + text } : message); }); });
    const offThinking = gateway.on("thinking.delta", (event) => { const p = accept(event, true); if (p && eventText(event)) setStatus(`Thinking: ${eventText(event)}`); });
    const offReasoning = gateway.on("reasoning.delta", (event) => { const p = accept(event, true); if (p && eventText(event)) setStatus(`Reasoning: ${eventText(event)}`); });
    const offInterim = gateway.on("message.interim", (event) => { const p = accept(event, true); const text = eventText(event); if (!p || !text) return; setTranscript((messages) => [...messages, { id: `interim-${++messageSequenceRef.current}`, role: "assistant", text }]); });
    const offToolGenerating = gateway.on("tool.generating", (event) => { const p = accept(event, true); if (p) setStatus(`Preparing tool: ${String(p.name ?? "tool")}`); });
    const offComplete = gateway.on("message.complete", (event) => { if (!accept(event, true)) return; const text = eventText(event); setTranscript((messages) => messages.map((message) => message.id === assistantIdRef.current ? { ...message, ...(text && !message.text ? { text } : {}), streaming: false } : message)); assistantIdRef.current = null; setStreaming(false); setTurnStartedAt(null); setStatus("Ready"); });
    const offToolStart = gateway.on("tool.start", (event) => { const p = accept(event, true); if (!p) return; const id = String(p.tool_id ?? `${p.name ?? "tool"}-${Date.now()}`); setStatus(`Running tool: ${String(p.name ?? "tool")}`); setTools((items) => items.some((item) => item.id === id) ? items : [...items, { id, name: String(p.name ?? "tool"), state: "running", context: typeof p.context === "string" ? p.context : undefined, args: p.args }]); });
    const offToolProgress = gateway.on("tool.progress", (event) => { const p = accept(event, true); if (!p) return; const id = String(p.tool_id ?? ""); if (!id) return; const progress = typeof p.progress === "string" ? p.progress : typeof p.text === "string" ? p.text : ""; if (progress) setStatus(`Working: ${progress}`); setTools((items) => items.map((item) => item.id === id ? { ...item, progress: progress || item.progress } : item)); });
    const offToolComplete = gateway.on("tool.complete", (event) => { const p = accept(event, true); if (!p) return; const id = String(p.tool_id ?? `${p.name ?? "tool"}-${Date.now()}`); setTools((items) => { const existing = items.some((item) => item.id === id); return existing ? items.map((item) => item.id === id ? { ...item, state: "complete", args: p.args ?? item.args, result: p.result, summary: typeof p.summary === "string" ? p.summary : item.summary } : item) : [...items, { id, name: String(p.name ?? "tool"), state: "complete", args: p.args, result: p.result, summary: typeof p.summary === "string" ? p.summary : undefined }]; }); });
    const offApproval = gateway.on("approval.request", (event) => { const p = accept(event, true); if (!p || typeof p.request_id !== "string") return; setApproval({ request_id: p.request_id, command: typeof p.command === "string" ? p.command : undefined, description: typeof p.description === "string" ? p.description : undefined, choices: Array.isArray(p.choices) ? p.choices.filter((x): x is string => typeof x === "string") : undefined, allow_permanent: p.allow_permanent !== false }); });
    const offClarify = gateway.on("clarify.request", (event) => { const p = accept(event, true); if (!p || typeof p.request_id !== "string") return; setClarify({ request_id: p.request_id, question: typeof p.question === "string" ? p.question : undefined, choices: Array.isArray(p.choices) ? p.choices.filter((x): x is string => typeof x === "string") : null, multi_select: p.multi_select === true, questions: Array.isArray(p.questions) ? p.questions as ClarificationRequest["questions"] : undefined }); });
    const offError = gateway.on("error", (event) => { if (!accept(event, true)) return; setError(eventText(event) || "Gateway error"); setStreaming(false); setTurnStartedAt(null); setStatus("Error"); });
    const offStatus = gateway.on("status.update", (event) => { if (accept(event, true)) setStatus(eventText(event) || "Working…"); });
    const offInfo = gateway.on("session.info", (event) => {
      const p = accept(event, true);
      if (!p) return;
      const running = p.running === true;
      setStreaming(running);
      setTurnStartedAt(typeof p.turn_started_at === "number" ? p.turn_started_at * 1000 : (running ? (started) => started ?? Date.now() : null));
      if (typeof p.status === "string" && p.status) setStatus(p.status);
      else if (running) setStatus("Working…");
      else setStatus("Ready");
    });

    let cancelled = false;
    queueMicrotask(() => {
      if (cancelled) return;
      sessionIdRef.current = null;
      durableSessionIdRef.current = resumeParam;
      assistantIdRef.current = null;
      setSessionId(null);
      setTranscript([]);
      setTools([]);
      setApproval(null);
      setClarify(null);

      setTurnStartedAt(null);
      setStreaming(false);
      followTranscriptRef.current = true;
      setShowScrollToBottom(false);
      updateAttachments([]);
      seenSeqRef.current.clear();
      messageSequenceRef.current = 0;
      setStatus(null);
      setError(null);
      setErrorAction(null);
      setFailedPrompt(null);
      setSubmitting(false);
      setStopping(false);
      submitInFlightRef.current = false;
      stopInFlightRef.current = false;
    });
    void gateway.connect()
      .then(async () => {
        if (cancelled) return;
        let response: ResumeResponse;
        if (resumeParam) {
          try {
            response = await gateway.request<ResumeResponse>("session.activate", { session_id: resumeParam, ...(profile ? { profile } : {}) });
          } catch {
            response = await gateway.request<ResumeResponse>("session.resume", { session_id: resumeParam, ...(profile ? { profile } : {}) });
          }
        } else {
          response = await gateway.request<ResumeResponse>("session.create", {
            ...nativeChatSessionCreateParams(profile, routingSelectionRef.current),
          });
        }
        if (!cancelled) {
          const runtimeId = response.session_id;
          if (!runtimeId) throw new Error("Gateway returned no session id");
          durableSessionIdRef.current = resumeParam;
          sessionIdRef.current = runtimeId;
          setSessionId(runtimeId);
          applySessionSnapshot(response);
          void gateway.request<ModelOptionsCatalog>("model.options", { include_unconfigured: true })
            .then((catalog) => { if (!cancelled) setModelCatalog(catalog); })
            .catch(() => { /* catalog is best-effort; Adaptive remains available */ });
        }
      })
      .catch((reason: unknown) => {
        if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason));
      });

    return () => {
      cancelled = true;
      offState();
      offStart();
      offDelta();
      offThinking();
      offReasoning();
      offInterim();
      offToolGenerating();
      offComplete();
      offToolStart();
      offToolProgress();
      offToolComplete();
      offApproval();
      offClarify();
      offError();
      offStatus();
      offInfo();
      gateway.close();
    };
  }, [freshGeneration, gateway, profile, resumeParam, routeModel, routeProvider, routeReasoning]);

  const changeRouting = useCallback((nextModel: string, nextProvider: string, nextReasoning: NativeReasoningLevel) => {
    setSearchParams((previous) => {
      const next = new URLSearchParams(previous);
      if (nextModel && nextProvider) {
        next.set("model", nextModel);
        next.set("provider", nextProvider);
      } else {
        next.delete("model");
        next.delete("provider");
      }
      if (nextReasoning === "auto") next.delete("reasoning");
      else next.set("reasoning", nextReasoning);
      next.delete("resume");
      return next;
    }, { replace: false });
  }, [setSearchParams]);

  const startNewChat = useCallback(() => {
    setSearchParams((previous) => {
      const next = new URLSearchParams(previous);
      next.delete("resume");
      return next;
    }, { replace: false });
    setFreshGeneration((generation) => generation + 1);
  }, [setSearchParams]);

  const submit = useCallback(async (event?: FormEvent, retryPrompt?: FailedPrompt) => {
    event?.preventDefault();
    if (submitInFlightRef.current) return;
    const text = retryPrompt?.text ?? draft.trim();
    if ((!text && !attachmentsRef.current.some((item) => item.state === "attached")) || !sessionId || connectionState !== "open") return;
    if (retryPrompt && failedPrompt?.id !== retryPrompt.id) return;
    if (attachmentsRef.current.some((item) => item.state === "uploading" || item.state === "pending")) { setError("Please wait for attachments to finish uploading"); return; }
    if (attachmentsRef.current.some((item) => item.state === "error")) { setError("Retry or remove failed attachments before sending"); return; }
    const promptText = retryPrompt ? retryPrompt.text : attachmentPromptText(text, attachmentsRef.current);
    const messageId = retryPrompt?.id ?? `user-${Date.now()}`;
    submitInFlightRef.current = true;
    setSubmitting(true);
    setError(null);
    setErrorAction(null);
    setStatus("Sending…");
    if (!retryPrompt) {
      setDraft("");
      setTranscript((messages) => [...messages, { id: messageId, role: "user", text: promptText }]);
    }
    try {
      await gateway.request("prompt.submit", { session_id: sessionId, text: promptText });
      updateAttachments([]);
      setFailedPrompt(null);
      setStatus("Working…");
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : String(reason));
      setErrorAction("resend");
      setFailedPrompt({ id: messageId, text: promptText });
      setStatus("Error");
    } finally {
      submitInFlightRef.current = false;
      setSubmitting(false);
    }
  }, [connectionState, draft, failedPrompt?.id, gateway, sessionId, updateAttachments]);

  const applyMessageAsPrompt = useCallback((message: string) => {
    setDraft(message);
    const textarea = textareaRef.current;
    if (textarea && !textarea.disabled) {
      textarea.focus();
      textarea.setSelectionRange(message.length, message.length);
    }
  }, []);

  const applyQuickPrompt = useCallback((prompt: string) => {
    setDraft(prompt);
    textareaRef.current?.focus();
  }, []);

  const onComposerKeyDown = useCallback((event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (slashPopoverRef.current?.handleKey(event)) return;
    if (shouldSubmitComposerKey(event.key, event.shiftKey, composingRef.current || event.nativeEvent.isComposing)) {
      event.preventDefault();
      void submit();
    }
  }, [submit]);

  const stop = useCallback(async () => {
    if (!sessionId || !streaming || connectionState !== "open" || stopInFlightRef.current) return;
    stopInFlightRef.current = true;
    setStopping(true);
    setError(null);
    setErrorAction(null);
    setStatus("Stopping…");
    try { await gateway.request("session.interrupt", { session_id: sessionId }); setStatus("Stopped"); setStreaming(false); }
    catch (reason: unknown) { setError(reason instanceof Error ? reason.message : String(reason)); setErrorAction("reconnect"); setStatus("Error"); }
    finally { stopInFlightRef.current = false; setStopping(false); }
  }, [connectionState, gateway, sessionId, streaming]);
  const respondApproval = useCallback(async (choice: string) => {
    if (!approval || !sessionId) return;
    try { await gateway.request("approval.respond", { choice, request_id: approval.request_id, session_id: sessionId }); setApproval(null); }
    catch (reason: unknown) { setError(reason instanceof Error ? reason.message : String(reason)); throw reason; }
  }, [approval, gateway, sessionId]);
  const respondClarify = useCallback(async (answer: string, questionId?: string) => {
    if (!clarify || !sessionId || !answer.trim()) return;
    const params = questionId
      ? { answer, question_id: questionId, request_id: clarify.request_id, session_id: sessionId }
      : { answer, request_id: clarify.request_id, session_id: sessionId };
    try { await gateway.request("clarify.respond", params); if (!questionId) setClarify(null); }
    catch (reason: unknown) { setError(reason instanceof Error ? reason.message : String(reason)); throw reason; }
  }, [clarify, gateway, sessionId]);
  const retry = useCallback(() => { setError(null); setErrorAction(null); void gateway.connect().catch((reason: unknown) => { setError(reason instanceof Error ? reason.message : String(reason)); setErrorAction("reconnect"); }); }, [gateway]);
  const resendFailedPrompt = useCallback(() => {
    if (failedPrompt) void submit(undefined, failedPrompt);
  }, [failedPrompt, submit]);
  const isWorking = streaming || tools.some((tool) => tool.state === "running") || turnStartedAt !== null;
  const activityStatus = isWorking
    ? (approval ? "Waiting for approval" : clarify ? "Waiting for clarification" : (status || "Thinking…"))
    : null;
  const pageStatus = connectionState !== "open"
    ? connectionLabel(connectionState)
    : isWorking
      ? "Working"
      : "Ready";
  const sessionActivityStatus: SessionActivityStatus = error
    ? "error"
    : connectionState !== "open"
      ? "offline"
      : approval || clarify
        ? "waiting"
        : isWorking
          ? "working"
          : "ready";
  const sessionStatuses = useMemo<Readonly<Record<string, SessionActivityStatus>>>(() => (
    resumeParam ? { [resumeParam]: sessionActivityStatus } : {}
  ), [resumeParam, sessionActivityStatus]);
  const elapsedSeconds = turnStartedAt == null ? 0 : Math.max(0, Math.floor((clockNow - turnStartedAt) / 1000));
  const lastAssistantId = [...transcript].reverse().find((message) => message.role === "assistant")?.id;
  const connectionTone = connectionState === "open"
    ? "success"
    : connectionState === "error"
      ? "destructive"
      : connectionState === "connecting"
        ? "warning"
        : "secondary";
  const statusDotClass = connectionState !== "open"
    ? "bg-muted-foreground"
    : isWorking
      ? "bg-primary"
      : "bg-success";
  useEffect(() => {
    if (!isWorking) return;
    setClockNow(Date.now());
    const timer = window.setInterval(() => setClockNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [isWorking]);

  return (
    <section
      data-slot="native-chat-shell"
      data-layout="desktop-like"
      className="flex min-h-0 min-w-0 flex-1 flex-col pb-4"
      aria-label="Native chat"
    >
      <header
        data-slot="chat-header"
        className="flex min-h-14 shrink-0 flex-wrap items-center justify-between gap-x-4 gap-y-3 border-b border-current/15 py-3"
      >
        <div className="flex min-w-0 items-center gap-1">
          {onOpenNavigation && (
            <Button
              ghost
              size="icon"
              type="button"
              className="shrink-0 lg:hidden"
              aria-label={t.app.openNavigation}
              onClick={onOpenNavigation}
            >
              <Menu />
            </Button>
          )}
          <Button
            ghost
            size="icon"
            type="button"
            className="shrink-0 lg:hidden"
            aria-label={t.sessions.title}
            aria-expanded={mobileSessionNavigatorOpen}
            aria-controls="native-chat-session-navigator"
            data-session-navigator-toggle
            onClick={() => setMobileSessionNavigatorOpen((open) => !open)}
          >
            <MessageSquare />
          </Button>
          <div className="min-w-0">
            <h1 className="truncate text-lg font-semibold">Chat</h1>
            <p className="truncate text-sm text-text-secondary">Native gateway chat</p>
          </div>
        </div>
        <div
          data-slot="chat-routing-controls"
          className="flex min-w-0 flex-wrap items-center justify-end gap-2"
        >
          <label className="sr-only" htmlFor="native-chat-model">Chat model</label>
          <select
            id="native-chat-model"
            aria-label="Chat model"
            className="h-9 max-w-44 min-w-0 border border-midground/15 bg-background/40 px-2 py-1 font-courier text-[16px] text-midground focus-visible:border-midground/30 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-midground/30 sm:text-xs"
            value={routingSelection.model ? `${routingSelection.model.provider}:${routingSelection.model.model}` : "adaptive"}
            onChange={(event) => {
              if (event.target.value === "adaptive") return void changeRouting("", "", routingSelection.reasoning);
              const choice = modelChoices.find((item) => `${item.provider}:${item.model}` === event.target.value);
              if (choice) changeRouting(choice.model, choice.provider, routingSelection.reasoning);
            }}
          >
            <option value="adaptive">Adaptive</option>
            {modelChoices.map((choice) => <option key={`${choice.provider}:${choice.model}`} value={`${choice.provider}:${choice.model}`}>{choice.label}</option>)}
          </select>
          <label className="sr-only" htmlFor="native-chat-reasoning">Reasoning level</label>
          <select
            id="native-chat-reasoning"
            aria-label="Reasoning level"
            className="h-9 min-w-0 border border-midground/15 bg-background/40 px-2 py-1 font-courier text-[16px] text-midground focus-visible:border-midground/30 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-midground/30 sm:text-xs"
            value={routingSelection.reasoning}
            onChange={(event) => changeRouting(routingSelection.model?.model ?? "", routingSelection.model?.provider ?? "", event.target.value as NativeReasoningLevel)}
          >
            {NATIVE_REASONING_OPTIONS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
          </select>
          <Badge
            data-slot="connection-badge"
            tone={connectionTone}
            className="shrink-0"
          >
            {connectionLabel(connectionState)}
          </Badge>
        </div>
      </header>

      <div
        data-slot="chat-notices"
        className={cn("flex shrink-0 flex-col gap-2", error && "pt-3")}
      >
        {error && (
          <div
            data-slot="chat-error"
            role="alert"
            className="flex flex-wrap items-center gap-2 border-l-2 border-destructive px-3 py-2 text-sm text-destructive"
          >
            <span className="min-w-0 flex-1 wrap-break-word">{error}</span>
            {errorAction === "resend" ? (
              <Button
                ghost
                size="sm"
                type="button"
                aria-label="Retry send"
                prefix={<RotateCcw />}
                className="shrink-0"
                onClick={resendFailedPrompt}
              >
                Retry send
              </Button>
            ) : (
              <Button
                ghost
                size="sm"
                type="button"
                prefix={<RotateCcw />}
                className="shrink-0"
                onClick={retry}
              >
                Retry
              </Button>
            )}
          </div>
        )}
      </div>

      <div
        data-slot="chat-body"
        className={cn(
          "grid min-h-0 flex-1",
          mobileSessionNavigatorOpen
            ? "grid-rows-[minmax(9rem,12rem)_minmax(0,1fr)]"
            : "grid-rows-[minmax(0,1fr)]",
          "lg:grid-cols-[16rem_minmax(0,1fr)] lg:grid-rows-1",
        )}
      >
        <aside
          id="native-chat-session-navigator"
          data-slot="session-navigator"
          data-mobile-open={mobileSessionNavigatorOpen ? "true" : "false"}
          role="complementary"
          aria-label={t.sessions.title}
          className={cn(
            "min-h-0 min-w-0 overflow-hidden border-b border-current/15 pt-3 pb-3 lg:border-r lg:border-b-0 lg:pt-4 lg:pr-4 lg:pb-0",
            !mobileSessionNavigatorOpen && "hidden lg:block",
          )}
        >
          <ChatSessionList
            activeSessionId={resumeParam}
            profile={profile ?? undefined}
            onPicked={() => setMobileSessionNavigatorOpen(false)}
            onNewChat={startNewChat}
            sessionStatuses={sessionStatuses}
          />
        </aside>

        <div
          data-slot="transcript-pane"
          role="region"
          aria-label="Conversation transcript"
          className="relative flex min-h-0 min-w-0 flex-col lg:pl-4"
        >
          <div
            ref={transcriptRef}
            onScroll={handleTranscriptScroll}
            data-testid="native-chat-transcript"
            data-slot="transcript"
            className="min-h-0 flex-1 space-y-3 overflow-y-auto py-4 pr-1"
            aria-live="polite"
          >
            {approval && <ApprovalCard request={approval} onRespond={respondApproval} />}
            {clarify && <ClarificationCard request={clarify} onRespond={respondClarify} />}
            {transcript.length === 0 && !approval && !clarify && (
              <div data-slot="chat-empty-state" className="max-w-2xl space-y-3 py-8 text-sm text-text-secondary">
                <p>Start a conversation.</p>
                <div className="space-y-2" role="group" aria-label="Quick prompts">
                  <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Try a prompt</p>
                  <div className="flex flex-wrap gap-2">
                    {QUICK_PROMPTS.map((prompt) => (
                      <button
                        key={prompt}
                        data-testid="quick-prompt"
                        type="button"
                        className="rounded-md border border-border bg-card px-3 py-2 text-left text-xs text-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-2 focus-visible:outline-ring"
                        onClick={() => applyQuickPrompt(prompt)}
                      >
                        {prompt}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}
            {transcript.map((message) => (
              <Fragment key={message.id}>
                {activityStatus && message.id === assistantIdRef.current && (
                  <div
                    data-slot="turn-activity"
                    className="mb-1 flex w-fit max-w-[85%] items-center gap-2 border-l-2 border-primary px-2 py-1 text-sm text-primary"
                    aria-label="Agent activity"
                  >
                    <span aria-hidden className="inline-block h-2 w-2 shrink-0 animate-pulse rounded-full bg-primary" />
                    <span>{activityStatus}</span>
                    <span className="font-mono text-xs text-text-secondary">{elapsedSeconds}s</span>
                  </div>
                )}
                <article
                  data-slot="transcript-message"
                  data-message-id={message.id}
                  data-message-role={message.role}
                  data-message-streaming={message.streaming ? "true" : "false"}
                  className={cn(
                    "w-fit max-w-[85%] whitespace-pre-wrap rounded-md px-3 py-2 text-sm",
                    message.role === "user"
                      ? "ml-auto bg-primary text-primary-foreground"
                      : "bg-muted text-foreground",
                  )}
                >
                  <div className="mb-1 text-[0.7rem] font-medium opacity-70">
                    {message.role === "user" ? "You" : "Hermes"}
                  </div>
                  {message.role === "assistant"
                    ? <MarkdownMessage content={message.text || (message.streaming ? "…" : "")} />
                    : message.text}
                  {message.text && (
                    <MessageActions
                      message={message.text}
                      messageRole={message.role}
                      onUseAsPrompt={applyMessageAsPrompt}
                    />
                  )}
                </article>
                {message.id === lastAssistantId && tools.length > 0 && (
                  <div
                    data-testid="tool-timeline"
                    data-slot="tool-timeline"
                    className="mt-3 space-y-2"
                    aria-label="Tool activity timeline"
                  >
                    {tools.map((tool) => <ToolActivity key={tool.id} item={tool} />)}
                  </div>
                )}
              </Fragment>
            ))}
            {!lastAssistantId && tools.length > 0 && (
              <div
                data-testid="tool-timeline"
                data-slot="tool-timeline"
                className="space-y-2"
                aria-label="Tool activity timeline"
              >
                {tools.map((tool) => <ToolActivity key={tool.id} item={tool} />)}
              </div>
            )}
          </div>

          {showScrollToBottom && (
            <Button
              ghost
              size="sm"
              type="button"
              prefix={<ArrowDown />}
              data-testid="scroll-to-bottom"
              aria-label="Scroll to latest message"
              className="absolute right-2 bottom-14 z-10 border border-border bg-card shadow-md"
              onClick={scrollToBottom}
            >
              Jump to latest
            </Button>
          )}

          <div
            data-slot="chat-status"
            role="status"
            aria-label="Chat status"
            className="flex shrink-0 items-center gap-2 border-t border-current/15 py-2 text-xs text-text-secondary"
          >
            <span aria-hidden className={cn("inline-block h-1.5 w-1.5 shrink-0 rounded-full", statusDotClass)} />
            <span>{connectionState === "open" ? "Connected" : connectionLabel(connectionState)}</span>
            <span aria-hidden>·</span>
            <span>{pageStatus}</span>
          </div>
        </div>
      </div>

      <form
        data-slot="chat-composer"
        aria-label="Message composer"
        aria-busy={submitting}
        className="flex shrink-0 flex-col gap-2 border-t border-current/15 bg-background/30 pt-3"
        onSubmit={submit}
        onDragOver={(event) => { event.preventDefault(); }}
        onDrop={(event) => { event.preventDefault(); addFiles(event.dataTransfer.files); }}
      >
        {attachments.length > 0 && (
          <div
            data-slot="attachment-list"
            className="flex flex-wrap gap-2 rounded-md border border-border bg-muted/20 p-2"
            aria-label="Pending attachments"
            aria-live="polite"
          >
            {attachments.map((item) => (
              <div
                key={item.id}
                data-slot="attachment"
                className="flex min-w-0 items-center gap-1 border border-midground/15 bg-background/40 px-2 py-1 text-xs"
              >
                <span className="max-w-48 truncate" title={item.file.name}>{item.file.name}</span>
                <span className="text-text-secondary">
                  {item.state === "uploading" ? "Uploading…" : item.state === "error" ? item.error : item.state === "attached" ? "Ready" : "Queued"}
                </span>
                {item.state === "error" && (
                  <Button
                    ghost
                    size="sm"
                    type="button"
                    aria-label={`Retry ${item.file.name}`}
                    prefix={<RotateCcw />}
                    onClick={() => void stageAttachment(item)}
                  >
                    Retry
                  </Button>
                )}
                <Button
                  ghost
                  size="icon"
                  type="button"
                  aria-label={`Remove ${item.file.name}`}
                  className="shrink-0 text-text-secondary hover:text-destructive"
                  onClick={() => updateAttachments((current) => current.filter((entry) => entry.id !== item.id))}
                >
                  <X />
                </Button>
              </div>
            ))}
          </div>
        )}
        <div data-slot="composer-controls" className="flex min-w-0 flex-wrap items-end gap-2 rounded-md border border-border bg-card/50 p-2">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={(event) => { if (event.target.files) addFiles(event.target.files); event.currentTarget.value = ""; }}
          />
          <div className="flex min-w-0 basis-full items-end gap-2 sm:basis-0 sm:flex-1">
            <Button
              ghost
              size="icon"
              type="button"
              aria-label="Add attachment"
              className="shrink-0"
              disabled={connectionState !== "open" || !sessionId}
              onClick={() => fileInputRef.current?.click()}
            >
              <Paperclip />
            </Button>
            <div className="relative min-w-0 flex-1">
              <SlashPopover
                ref={slashPopoverRef}
                input={draft}
                gw={gateway}
                onApply={setDraft}
              />
              <textarea
                ref={textareaRef}
                aria-label="Message"
                aria-describedby="native-chat-composer-hint"
                className="min-h-20 w-full resize-y border border-border bg-background/40 px-3 py-2 font-courier text-[16px] text-foreground outline-none placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-1 focus-visible:ring-ring/40 sm:text-sm"
                value={draft}
                disabled={connectionState !== "open" || !sessionId}
                placeholder="Message Hermes… (drop or paste files)"
                onPaste={(event) => { if (event.clipboardData.files.length) { event.preventDefault(); addFiles(event.clipboardData.files); } }}
                onChange={(event) => setDraft(event.target.value)}
                onCompositionStart={() => { composingRef.current = true; }}
                onCompositionEnd={() => { composingRef.current = false; }}
                onKeyDown={onComposerKeyDown}
              />
            </div>
          </div>
          <div data-slot="composer-actions" className="ml-auto flex shrink-0 items-center gap-2">
            <Button
              type="submit"
              size="sm"
              prefix={<Send />}
              aria-label="Send message"
              className="shrink-0"
              disabled={submitting || (!draft.trim() && !attachments.some((item) => item.state === "attached")) || connectionState !== "open" || !sessionId}
            >
              {submitting ? "Sending…" : "Send"}
            </Button>
            {streaming && (
              <Button
                destructive
                outlined
                type="button"
                size="sm"
                prefix={<Square />}
                aria-label="Stop"
                className="shrink-0"
                disabled={stopping || connectionState !== "open"}
                onClick={() => void stop()}
              >
                {stopping ? "Stopping…" : "Stop"}
              </Button>
            )}
          </div>
        </div>
        <div data-slot="composer-meta" className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 px-1 text-xs text-muted-foreground">
          <span id="native-chat-composer-hint" data-slot="composer-attachment-hint" className="inline-flex items-center gap-1.5">
            <Paperclip aria-hidden className="h-3.5 w-3.5" />
            Drop files or paste to attach
          </span>
          <span data-slot="composer-status" role="status" aria-live="polite">
            {submitting
              ? "Sending…"
              : connectionState !== "open"
                ? "Waiting for connection…"
                : status ?? (streaming ? "Working…" : "Ready")}
          </span>
        </div>
      </form>
    </section>
  );
}

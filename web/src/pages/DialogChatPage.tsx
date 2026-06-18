import { Button } from "@nous-research/ui/ui/components/button";
import {
  Check,
  ChevronDown,
  Copy,
  Loader2,
  MessageSquare,
  Paperclip,
  Plus,
  Send,
  Square,
} from "lucide-react";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";

import { Markdown } from "@/components/Markdown";
import { api, type SessionInfo } from "@/lib/api";
import { GatewayClient, type ConnectionState } from "@/lib/gatewayClient";
import { executeSlash } from "@/lib/slashExec";
import { cn } from "@/lib/utils";

interface GatewayTranscriptMessage {
  role?: string;
  text?: string;
  content?: unknown;
}

type DialogMessageRole = "user" | "assistant" | "system";

interface DialogMessage {
  id: string;
  role: DialogMessageRole;
  text: string;
  createdAt: number;
  queued?: boolean;
  streaming?: boolean;
  tone?: "normal" | "error";
}

interface DialogSessionInfo {
  model?: string;
  provider?: string;
  credential_warning?: string;
}

interface DialogInflightTurn {
  assistant?: string;
  streaming?: boolean;
  user?: string;
}

interface DialogSessionResult {
  session_id?: string;
  session_key?: string;
  stored_session_id?: string;
  messages?: GatewayTranscriptMessage[];
  info?: DialogSessionInfo;
  inflight?: DialogInflightTurn | null;
  running?: boolean;
}

interface MessageTextPayload {
  text?: string;
  status?: string;
  warning?: string;
}

interface GatewayErrorPayload {
  message?: string;
}

interface ToolEventPayload {
  name?: string;
  tool_id?: string;
  error?: string;
}

interface AttachResponse {
  attached?: boolean;
  count?: number;
  filename?: string;
  message?: string;
  name?: string;
  pages_attached?: number;
  ref_text?: string;
  text?: string;
}

interface QueuedPrompt {
  id: string;
  text: string;
  userMessageId: string;
  queuedAt: number;
}

const COMPOSER_MIN_ROWS = 2;
const COMPOSER_MAX_ROWS = 5;

function generateLocalId(prefix: string): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  return `${prefix}-${Math.random().toString(36).slice(2)}-${Date.now().toString(36)}`;
}

function coerceTranscriptText(value: unknown): string {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    return value
      .map((item) => {
        if (typeof item === "string") return item;
        if (item && typeof item === "object" && "text" in item) {
          const text = (item as { text?: unknown }).text;
          return typeof text === "string" ? text : "";
        }
        return "";
      })
      .filter(Boolean)
      .join("\n");
  }
  if (value && typeof value === "object" && "text" in value) {
    const text = (value as { text?: unknown }).text;
    return typeof text === "string" ? text : "";
  }
  return "";
}

function transcriptToDialogMessages(
  messages: GatewayTranscriptMessage[] | undefined,
): DialogMessage[] {
  if (!messages) return [];
  return messages.flatMap((message, index) => {
    const role =
      message.role === "assistant"
        ? "assistant"
        : message.role === "user"
          ? "user"
          : null;
    if (!role) return [];
    const text = (message.text ?? coerceTranscriptText(message.content)).trim();
    if (!text) return [];
    return [
      {
        id: `history-${index}-${role}`,
        role,
        text,
        createdAt: Date.now() + index,
      },
    ];
  });
}

function trailingUserMatches(messages: DialogMessage[], text: string): boolean {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "assistant") return false;
    if (message.role === "user") return message.text.trim() === text;
  }
  return false;
}

function sessionResultToDialogMessages(
  result: DialogSessionResult,
): { messages: DialogMessage[]; pendingAssistantId: string | null } {
  const messages = transcriptToDialogMessages(result.messages);
  const inflight = result.inflight;
  const userText = inflight?.user?.trim() ?? "";
  const assistantText =
    typeof inflight?.assistant === "string" ? inflight.assistant : "";
  const streaming = Boolean(result.running || inflight?.streaming);

  if (!userText && !assistantText && !streaming) {
    return { messages, pendingAssistantId: null };
  }

  const hydrated = [...messages];
  if (userText && !trailingUserMatches(hydrated, userText)) {
    hydrated.push({
      id: generateLocalId("inflight-user"),
      role: "user",
      text: userText,
      createdAt: Date.now(),
    });
  }

  if (!assistantText && !streaming) {
    return { messages: hydrated, pendingAssistantId: null };
  }

  const last = hydrated[hydrated.length - 1];
  if (last?.role === "assistant" && last.text === assistantText) {
    last.streaming = streaming;
    return {
      messages: hydrated,
      pendingAssistantId: streaming ? last.id : null,
    };
  }

  const assistantId = generateLocalId("inflight-assistant");
  hydrated.push({
    id: assistantId,
    role: "assistant",
    text: assistantText,
    streaming,
    createdAt: Date.now() + 1,
  });
  return {
    messages: hydrated,
    pendingAssistantId: streaming ? assistantId : null,
  };
}

function connectionLabel(
  state: ConnectionState,
  running: boolean,
): string {
  if (running) return "thinking";
  if (state === "open") return "live";
  return state;
}

function formatSessionAge(epochSeconds: number): string {
  const delta = Math.max(0, Date.now() / 1000 - epochSeconds);
  if (delta < 60) return "just now";
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  if (delta < 172800) return "yesterday";
  return `${Math.floor(delta / 86400)}d ago`;
}

function formatSessionOption(session: SessionInfo): string {
  const title = formatSessionTitle(session);
  const meta = formatSessionMeta(session);
  return meta ? `${title} · ${meta}` : title;
}

function formatSessionTitle(session: SessionInfo): string {
  const rawTitle =
    session.title?.trim() ||
    session.preview?.trim() ||
    session.id;
  return rawTitle.length > 64 ? `${rawTitle.slice(0, 61)}...` : rawTitle;
}

function formatSessionMeta(session: SessionInfo): string {
  const age = formatSessionAge(session.last_active || session.started_at);
  const model = session.model?.split("/").slice(-1)[0] ?? "";
  return [model, age].filter(Boolean).join(" · ");
}

function syncDialogComposerHeight(textarea: HTMLTextAreaElement | null) {
  if (!textarea || typeof window === "undefined") return;
  const style = window.getComputedStyle(textarea);
  const lineHeight = Number.parseFloat(style.lineHeight) || 22;
  const padding =
    Number.parseFloat(style.paddingTop) +
    Number.parseFloat(style.paddingBottom);
  const border =
    Number.parseFloat(style.borderTopWidth) +
    Number.parseFloat(style.borderBottomWidth);
  const minHeight = lineHeight * COMPOSER_MIN_ROWS + padding + border;
  const maxHeight = lineHeight * COMPOSER_MAX_ROWS + padding + border;

  textarea.style.height = `${minHeight}px`;
  const next = Math.min(Math.max(textarea.scrollHeight, minHeight), maxHeight);
  textarea.style.height = `${next}px`;
  textarea.style.overflowY = textarea.scrollHeight > maxHeight ? "auto" : "hidden";
}

function readAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () =>
      typeof reader.result === "string"
        ? resolve(reader.result)
        : reject(new Error("file read returned non-text data"));
    reader.onerror = () =>
      reject(reader.error ?? new Error("failed to read file"));
    reader.readAsDataURL(file);
  });
}

function isSessionBusyErrorMessage(message: string): boolean {
  return /session busy|subagent still running/i.test(message);
}

/**
 * CSS width for xterm font tiers.
 *
 * Prefer the terminal host's `clientWidth` — Chrome DevTools device mode often
 * keeps `window.innerWidth` at the full desktop value while the *drawn* layout
 * is phone-sized, which made us pick desktop font sizes (~14px) and look huge.
 */

export function DialogChatPage({
  isActive,
  profile,
  resumeSessionId,
  onResumeSession,
  className,
}: {
  isActive: boolean;
  profile?: string;
  resumeSessionId: string | null;
  onResumeSession: (sessionId: string | null) => void;
  className?: string;
}) {
  const gwRef = useRef<GatewayClient | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const storedSessionIdRef = useRef<string | null>(null);
  const pendingAssistantIdRef = useRef<string | null>(null);
  const queuedPromptsRef = useRef<QueuedPrompt[]>([]);
  const drainingQueueRef = useRef(false);
  const runningRef = useRef(false);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const copyResetRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [state, setState] = useState<ConnectionState>("idle");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<DialogMessage[]>([]);
  const [input, setInput] = useState("");
  const [running, setRunning] = useState(false);
  const [info, setInfo] = useState<DialogSessionInfo>({});
  const [activity, setActivity] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [historySessions, setHistorySessions] = useState<SessionInfo[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyRefreshToken, setHistoryRefreshToken] = useState(0);
  const [attaching, setAttaching] = useState(false);
  const [queuedPrompts, setQueuedPrompts] = useState<QueuedPrompt[]>([]);

  const setRunningState = useCallback((next: boolean) => {
    runningRef.current = next;
    setRunning(next);
  }, []);

  const syncQueuedPrompts = useCallback(
    (
      updater:
        | QueuedPrompt[]
        | ((current: QueuedPrompt[]) => QueuedPrompt[]),
    ) => {
      setQueuedPrompts((current) => {
        const next =
          typeof updater === "function" ? updater(current) : updater;
        queuedPromptsRef.current = next;
        return next;
      });
    },
    [],
  );

  const appendSystemMessage = useCallback(
    (text: string, tone: DialogMessage["tone"] = "normal") => {
      const body = text.trim();
      if (!body) return;
      setMessages((prev) => [
        ...prev,
        {
          id: generateLocalId("system"),
          role: "system",
          text: body,
          tone,
          createdAt: Date.now(),
        },
      ]);
    },
    [],
  );

  const enqueuePrompt = useCallback(
    (rawText: string): QueuedPrompt | null => {
      const text = rawText.trim();
      if (!text) return null;
      const entry: QueuedPrompt = {
        id: generateLocalId("queued"),
        text,
        userMessageId: generateLocalId("queued-user"),
        queuedAt: Date.now(),
      };
      syncQueuedPrompts((current) => [...current, entry]);
      setMessages((prev) => [
        ...prev,
        {
          id: entry.userMessageId,
          role: "user",
          text,
          queued: true,
          createdAt: entry.queuedAt,
        },
      ]);
      return entry;
    },
    [syncQueuedPrompts],
  );

  const requeuePromptToFront = useCallback(
    (entry: QueuedPrompt) => {
      syncQueuedPrompts((current) => [entry, ...current]);
      setMessages((prev) =>
        prev.map((item) =>
          item.id === entry.userMessageId ? { ...item, queued: true } : item,
        ),
      );
    },
    [syncQueuedPrompts],
  );

  const submitPrompt = useCallback(async (
    rawText: string,
    options: {
      queuedEntry?: QueuedPrompt;
      showUserMessage?: boolean;
      userMessageId?: string;
    } = {},
  ) => {
    const text = rawText.trim();
    const gw = gwRef.current;
    const sid = sessionIdRef.current;
    if (!text || !gw || !sid) return;

    const assistantId = generateLocalId("assistant");
    const showUserMessage = options.showUserMessage !== false;
    const userMessageId = options.userMessageId ?? generateLocalId("user");
    pendingAssistantIdRef.current = assistantId;
    setError(null);
    setActivity(null);
    setRunningState(true);
    setMessages((prev) => [
      ...prev,
      ...(showUserMessage
        ? [
            {
              id: userMessageId,
              role: "user" as const,
              text,
              createdAt: Date.now(),
            },
          ]
        : []),
      {
        id: assistantId,
        role: "assistant",
        text: "",
        streaming: true,
        createdAt: Date.now() + 1,
      },
    ]);

    try {
      await gw.request("prompt.submit", { session_id: sid, text });
      if (!resumeSessionId && storedSessionIdRef.current) {
        onResumeSession(storedSessionIdRef.current);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      pendingAssistantIdRef.current = null;
      setRunningState(false);
      setActivity(null);
      if (isSessionBusyErrorMessage(message)) {
        setMessages((prev) =>
          prev
            .filter((item) => item.id !== assistantId)
            .map((item) =>
              item.id === userMessageId ? { ...item, queued: true } : item,
            ),
        );
        if (options.queuedEntry) {
          requeuePromptToFront(options.queuedEntry);
        } else {
          syncQueuedPrompts((current) => [
            {
              id: generateLocalId("queued"),
              text,
              userMessageId,
              queuedAt: Date.now(),
            },
            ...current,
          ]);
        }
        setActivity("queued for next turn");
        return;
      }
      setError(message);
      setMessages((prev) =>
        prev.map((item) =>
          item.id === assistantId
            ? {
                ...item,
                text: `Error: ${message}`,
                streaming: false,
                tone: "error",
              }
            : item,
        ),
      );
    }
  }, [
    onResumeSession,
    requeuePromptToFront,
    resumeSessionId,
    setRunningState,
    syncQueuedPrompts,
  ]);

  useEffect(() => {
    const gw = new GatewayClient();
    gwRef.current = gw;
    sessionIdRef.current = null;
    storedSessionIdRef.current = null;
    pendingAssistantIdRef.current = null;
    queuedPromptsRef.current = [];
    drainingQueueRef.current = false;
    runningRef.current = false;
    let cancelled = false;

    setState("idle");
    setSessionId(null);
    setMessages([]);
    setInput("");
    setRunningState(false);
    setInfo({});
    setActivity(null);
    setError(null);
    setAttaching(false);
    setQueuedPrompts([]);

    const belongsToCurrentSession = (eventSessionId?: string) => {
      const sid = sessionIdRef.current;
      return !sid || !eventSessionId || eventSessionId === sid;
    };

    const ensureAssistantMessage = (): string => {
      const currentId = pendingAssistantIdRef.current;
      if (currentId) return currentId;

      const id = generateLocalId("assistant");
      pendingAssistantIdRef.current = id;
      setMessages((prev) => [
        ...prev,
        {
          id,
          role: "assistant",
          text: "",
          streaming: true,
          createdAt: Date.now(),
        },
      ]);
      return id;
    };

    const offState = gw.onState((next) => {
      if (!cancelled) setState(next);
    });

    const offSessionInfo = gw.on<DialogSessionInfo>("session.info", (ev) => {
      if (cancelled || !belongsToCurrentSession(ev.session_id)) return;
      if (ev.session_id && !sessionIdRef.current) {
        sessionIdRef.current = ev.session_id;
        setSessionId(ev.session_id);
      }
      if (ev.payload) {
        setInfo((prev) => ({ ...prev, ...ev.payload }));
      }
      if (ev.session_id) {
        setHistoryRefreshToken((value) => value + 1);
      }
    });

    const offStart = gw.on("message.start", (ev) => {
      if (cancelled || !belongsToCurrentSession(ev.session_id)) return;
      ensureAssistantMessage();
      setRunningState(true);
      setActivity(null);
    });

    const offDelta = gw.on<MessageTextPayload>("message.delta", (ev) => {
      if (cancelled || !belongsToCurrentSession(ev.session_id)) return;
      const text = ev.payload?.text;
      if (!text) return;
      const id = ensureAssistantMessage();
      setRunningState(true);
      setMessages((prev) =>
        prev.map((item) =>
          item.id === id ? { ...item, text: item.text + text } : item,
        ),
      );
    });

    const offComplete = gw.on<MessageTextPayload>("message.complete", (ev) => {
      if (cancelled || !belongsToCurrentSession(ev.session_id)) return;
      const id = pendingAssistantIdRef.current;
      const finalText = ev.payload?.text;
      const warning = ev.payload?.warning;
      pendingAssistantIdRef.current = null;
      setRunningState(false);
      setActivity(null);
      if (warning) setError(warning);
      if (!id) return;
      setMessages((prev) =>
        prev.map((item) =>
          item.id === id
            ? {
                ...item,
                text:
                  typeof finalText === "string" && finalText.length > 0
                    ? finalText
                    : item.text,
                streaming: false,
                tone: ev.payload?.status === "error" ? "error" : item.tone,
              }
            : item,
        ),
      );
    });

    const offError = gw.on<GatewayErrorPayload>("error", (ev) => {
      if (cancelled || !belongsToCurrentSession(ev.session_id)) return;
      const message = ev.payload?.message ?? "gateway error";
      const id = pendingAssistantIdRef.current;
      pendingAssistantIdRef.current = null;
      setError(message);
      setRunningState(false);
      setActivity(null);
      if (!id) {
        appendSystemMessage(message, "error");
        return;
      }
      setMessages((prev) =>
        prev.map((item) =>
          item.id === id
            ? {
                ...item,
                text: item.text || `Error: ${message}`,
                streaming: false,
                tone: "error",
              }
            : item,
        ),
      );
    });

    const offToolStart = gw.on<ToolEventPayload>("tool.start", (ev) => {
      if (cancelled || !belongsToCurrentSession(ev.session_id)) return;
      setActivity(ev.payload?.name ?? "tool");
    });

    const offToolComplete = gw.on<ToolEventPayload>("tool.complete", (ev) => {
      if (cancelled || !belongsToCurrentSession(ev.session_id)) return;
      if (ev.payload?.error) setError(ev.payload.error);
      setActivity(null);
    });

    void (async () => {
      try {
        await gw.connect();
        if (cancelled) return;
        const scope = profile ? { profile } : {};
        const result = resumeSessionId
          ? await gw.request<DialogSessionResult>(
              "session.resume",
              {
                session_id: resumeSessionId,
                cols: 100,
                close_on_disconnect: false,
                ...scope,
              },
              300_000,
            )
          : await gw.request<DialogSessionResult>(
              "session.create",
              {
                cols: 100,
                close_on_disconnect: false,
                ...scope,
              },
              300_000,
            );

        if (cancelled) return;
        if (!result.session_id) {
          throw new Error("gateway did not return a session id");
        }
        sessionIdRef.current = result.session_id;
        storedSessionIdRef.current =
          result.stored_session_id ??
          result.session_key ??
          resumeSessionId ??
          null;
        setSessionId(result.session_id);
        const hydrated = sessionResultToDialogMessages(result);
        pendingAssistantIdRef.current = hydrated.pendingAssistantId;
        setMessages(hydrated.messages);
        setInfo(result.info ?? {});
        setRunningState(Boolean(result.running || result.inflight?.streaming));
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      }
    })();

    return () => {
      cancelled = true;
      offState();
      offSessionInfo();
      offStart();
      offDelta();
      offComplete();
      offError();
      offToolStart();
      offToolComplete();
      gw.close();
      if (gwRef.current === gw) gwRef.current = null;
      if (copyResetRef.current) {
        clearTimeout(copyResetRef.current);
        copyResetRef.current = null;
      }
    };
  }, [appendSystemMessage, profile, resumeSessionId, setRunningState]);

  const drainQueuedPrompt = useCallback(async () => {
    if (drainingQueueRef.current || runningRef.current) return;
    const entry = queuedPromptsRef.current[0];
    const gw = gwRef.current;
    const sid = sessionIdRef.current;
    if (!entry || !gw || !sid) return;

    drainingQueueRef.current = true;
    syncQueuedPrompts((current) => current.slice(1));
    setMessages((prev) =>
      prev.map((item) =>
        item.id === entry.userMessageId
          ? { ...item, queued: false }
          : item,
      ),
    );

    try {
      if (entry.text.startsWith("/")) {
        setError(null);
        await executeSlash({
          command: entry.text,
          sessionId: sid,
          gw,
          callbacks: {
            sys: appendSystemMessage,
            send: (text) =>
              submitPrompt(text, {
                showUserMessage: false,
                userMessageId: entry.userMessageId,
              }),
          },
        });
      } else {
        await submitPrompt(entry.text, {
          queuedEntry: entry,
          showUserMessage: false,
          userMessageId: entry.userMessageId,
        });
      }
    } finally {
      drainingQueueRef.current = false;
    }
  }, [appendSystemMessage, submitPrompt, syncQueuedPrompts]);

  useEffect(() => {
    if (!sessionId || running || queuedPrompts.length === 0) return;
    void drainQueuedPrompt();
  }, [drainQueuedPrompt, queuedPrompts.length, running, sessionId]);

  useEffect(() => {
    if (!isActive) return;
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [isActive, messages, queuedPrompts.length, running, activity, error]);

  useEffect(() => {
    if (!isActive) return;
    inputRef.current?.focus();
  }, [isActive, running]);

  useEffect(() => {
    syncDialogComposerHeight(inputRef.current);
  }, [input, isActive]);

  useEffect(() => {
    if (!isActive) return;
    let cancelled = false;
    setHistoryLoading(true);
    api
      .getSessions(40, 0, profile)
      .then((res) => {
        if (!cancelled) setHistorySessions(res.sessions);
      })
      .catch(() => {
        if (!cancelled) setHistorySessions([]);
      })
      .finally(() => {
        if (!cancelled) setHistoryLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [historyRefreshToken, isActive, profile, resumeSessionId, running]);

  const handleSubmit = useCallback(async () => {
    const text = input.trim();
    const gw = gwRef.current;
    const sid = sessionIdRef.current;
    if (!text || !gw || !sid) return;

    setInput("");
    if (runningRef.current) {
      enqueuePrompt(text);
      return;
    }

    if (text.startsWith("/")) {
      setError(null);
      await executeSlash({
        command: text,
        sessionId: sid,
        gw,
        callbacks: {
          sys: appendSystemMessage,
          send: submitPrompt,
        },
      });
      return;
    }

    await submitPrompt(text);
  }, [appendSystemMessage, enqueuePrompt, input, submitPrompt]);

  const handleInterrupt = useCallback(async () => {
    const gw = gwRef.current;
    const sid = sessionIdRef.current;
    if (!gw || !sid) return;
    const hasQueuedPrompts = queuedPromptsRef.current.length > 0;
    try {
      await gw.request("session.interrupt", { session_id: sid }, 30_000);
      const id = pendingAssistantIdRef.current;
      pendingAssistantIdRef.current = null;
      if (!hasQueuedPrompts) {
        setRunningState(false);
        setActivity(null);
      } else {
        setActivity("interrupting current turn");
      }
      if (id) {
        setMessages((prev) =>
          prev.map((item) =>
            item.id === id ? { ...item, streaming: false } : item,
          ),
        );
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [setRunningState]);

  const copyMessage = useCallback((message: DialogMessage) => {
    if (!message.text) return;
    navigator.clipboard
      .writeText(message.text)
      .then(() => {
        setCopiedId(message.id);
        if (copyResetRef.current) clearTimeout(copyResetRef.current);
        copyResetRef.current = setTimeout(() => setCopiedId(null), 1200);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : String(err));
      });
  }, []);

  const handleAttachFiles = useCallback(
    async (files: FileList | null) => {
      const gw = gwRef.current;
      const sid = sessionIdRef.current;
      if (!files?.length || !gw || !sid || runningRef.current) return;

      setAttaching(true);
      setError(null);
      try {
        for (const file of Array.from(files)) {
          const dataUrl = await readAsDataUrl(file);
          const lowerName = file.name.toLowerCase();
          let response: AttachResponse;

          if (file.type.startsWith("image/")) {
            response = await gw.request<AttachResponse>(
              "image.attach_bytes",
              {
                session_id: sid,
                content_base64: dataUrl,
                filename: file.name,
              },
              120_000,
            );
            appendSystemMessage(
              response.text ?? `attached image: ${file.name}`,
            );
            continue;
          }

          if (file.type === "application/pdf" || lowerName.endsWith(".pdf")) {
            response = await gw.request<AttachResponse>(
              "pdf.attach",
              {
                session_id: sid,
                content_base64: dataUrl,
                filename: file.name,
              },
              180_000,
            );
            appendSystemMessage(
              response.text ?? `attached PDF: ${file.name}`,
            );
            continue;
          }

          response = await gw.request<AttachResponse>(
            "file.attach",
            {
              session_id: sid,
              path: file.name,
              data_url: dataUrl,
              name: file.name,
            },
            120_000,
          );
          if (response.ref_text) {
            setInput((prev) =>
              prev.trim()
                ? `${prev.trimEnd()} ${response.ref_text}`
                : response.ref_text ?? prev,
            );
          }
          appendSystemMessage(
            response.ref_text
              ? `attached file: ${file.name}\n${response.ref_text}`
              : `attached file: ${file.name}`,
          );
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
        appendSystemMessage(`attach failed: ${message}`, "error");
      } finally {
        setAttaching(false);
        if (fileInputRef.current) fileInputRef.current.value = "";
      }
    },
    [appendSystemMessage],
  );

  const onInputKeyDown = (ev: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (
      ev.key === "Enter" &&
      !ev.shiftKey &&
      !(ev.nativeEvent as KeyboardEvent).isComposing
    ) {
      ev.preventDefault();
      void handleSubmit();
    }
  };

  const modelLabel = (info.model ?? "").split("/").slice(-1)[0] || "dialog";
  const status = connectionLabel(state, running);
  const statusClass =
    running || state === "connecting"
      ? "border-warning/40 bg-warning/10 text-warning"
      : state === "open"
        ? "border-success/40 bg-success/10 text-success"
        : state === "error"
          ? "border-destructive/40 bg-destructive/10 text-destructive"
          : "border-border bg-muted/20 text-text-secondary";
  const bannerText = error ?? info.credential_warning ?? null;
  const disabled = state !== "open" || !sessionId;
  const sendDisabled = disabled || input.trim().length === 0;
  const interruptDisabled = disabled || !running;
  const attachDisabled = disabled || running || attaching;
  const queueLabel =
    queuedPrompts.length === 0
      ? null
      : `${queuedPrompts.length} queued for next turn`;
  const selectedSessionId = resumeSessionId ?? sessionId ?? "";
  const selectedSessionInList =
    !selectedSessionId ||
    historySessions.some((session) => session.id === selectedSessionId);
  const currentSessionLabel = resumeSessionId
    ? "current resumed session"
    : "new chat";

  return (
    <section
      className={cn(
        "relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden rounded-lg",
        "border border-border bg-background/40",
        className,
      )}
      style={{ boxShadow: "0 8px 32px rgba(0, 0, 0, 0.25)" }}
    >
      <div className="flex h-11 shrink-0 items-center justify-between gap-3 border-b border-border px-3 sm:px-4">
        <div className="flex min-w-0 items-center gap-2">
          <MessageSquare className="h-4 w-4 shrink-0 text-text-secondary" />
          <div className="min-w-0 truncate text-sm font-medium text-foreground">
            {modelLabel}
          </div>
          {info.provider && (
            <div className="hidden truncate text-xs text-text-tertiary sm:block">
              {info.provider}
            </div>
          )}
        </div>

        <div className="flex min-w-0 shrink-0 items-center gap-2">
          <SessionResumeMenu
            currentSessionLabel={currentSessionLabel}
            historyLoading={historyLoading}
            historySessions={historySessions}
            onSelect={onResumeSession}
            resumeSessionId={resumeSessionId}
            selectedSessionId={selectedSessionId}
            selectedSessionInList={selectedSessionInList}
          />

          <div
            className={cn(
              "shrink-0 rounded border px-2 py-0.5 text-xs font-medium",
              statusClass,
            )}
          >
            {status}
          </div>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-3 py-4 sm:px-5">
        {messages.length === 0 && !bannerText ? (
          <div className="flex h-full min-h-[220px] items-center justify-center text-text-tertiary">
            <div className="flex items-center gap-2 text-sm">
              <MessageSquare className="h-4 w-4" />
              <span>Hermes</span>
            </div>
          </div>
        ) : (
          <div className="flex w-full flex-col gap-3">
            {messages.map((message) => (
              <DialogMessageBubble
                key={message.id}
                message={message}
                copied={copiedId === message.id}
                onCopy={copyMessage}
              />
            ))}

            {bannerText && (
              <div
                className={cn(
                  "w-full rounded border px-3 py-2 text-xs",
                  error
                    ? "border-destructive/40 bg-destructive/5 text-destructive"
                    : "border-warning/40 bg-warning/5 text-warning",
                )}
              >
                {bannerText}
              </div>
            )}
          </div>
        )}
        <div ref={bottomRef} aria-hidden className="h-4 shrink-0 sm:h-5" />
      </div>

      <div className="shrink-0 border-t border-border bg-background/70 p-2 sm:p-3">
        <div className="flex w-full flex-col gap-2">
          {(activity || queueLabel) && (
            <div className="flex items-center gap-2 px-1 text-xs text-text-secondary">
              {activity ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <span className="h-1.5 w-1.5 rounded-full bg-primary/70" />
              )}
              <span className="truncate">{activity ?? queueLabel}</span>
            </div>
          )}

          <div className="flex items-end gap-2">
            <input
              ref={fileInputRef}
              type="file"
              multiple
              className="hidden"
              onChange={(ev) => void handleAttachFiles(ev.target.files)}
            />

            <textarea
              ref={inputRef}
              rows={COMPOSER_MIN_ROWS}
              value={input}
              disabled={disabled}
              onChange={(ev) => setInput(ev.target.value)}
              onInput={(ev) =>
                syncDialogComposerHeight(ev.currentTarget)
              }
              onKeyDown={onInputKeyDown}
              placeholder="Message Hermes"
              className={cn(
                "max-h-36 flex-1 resize-none rounded border border-border",
                "bg-background/60 px-3 py-2.5 text-sm leading-relaxed text-foreground",
                "placeholder:text-text-tertiary focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/30",
                "disabled:cursor-not-allowed disabled:opacity-60",
              )}
            />

            <div className="flex shrink-0 flex-col items-stretch justify-end gap-[4px] font-sans text-[14px] leading-[20px]">
              {running ? (
                <button
                  type="button"
                  onClick={handleInterrupt}
                  disabled={interruptDisabled}
                  title="Stop"
                  aria-label="Stop response"
                  className={cn(
                    "inline-flex h-[38px] w-[40px] shrink-0 items-center justify-center rounded border font-sans text-[14px] leading-[20px] transition-colors",
                    "border-warning/45 bg-warning/10 text-warning hover:bg-warning/15",
                    "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/30",
                    "disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-45",
                  )}
                >
                  <Square className="h-[14px] w-[14px]" />
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={attachDisabled}
                  title={attaching ? "Attaching..." : "Attach file"}
                  aria-label={attaching ? "Attaching file" : "Attach file"}
                  className={cn(
                    "inline-flex h-[38px] w-[40px] shrink-0 items-center justify-center rounded border border-border",
                    "bg-background/60 font-sans text-[14px] leading-[20px] text-text-secondary transition-colors",
                    "hover:border-foreground/30 hover:bg-midground/5 hover:text-foreground",
                    "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/30",
                    "disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-45",
                  )}
                >
                  {attaching ? (
                    <Loader2 className="h-[14px] w-[14px] animate-spin" />
                  ) : (
                    <Paperclip className="h-[14px] w-[14px]" />
                  )}
                </button>
              )}

              <button
                type="button"
                onClick={handleSubmit}
                disabled={sendDisabled}
                title={running ? "Queue message" : "Send"}
                aria-label={running ? "Queue message" : "Send message"}
                className={cn(
                  "inline-flex h-[38px] w-[40px] shrink-0 items-center justify-center rounded border font-sans text-[14px] leading-[20px] transition-colors",
                  "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/30",
                  "disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-45",
                  "border-primary/35 bg-primary/15 text-primary hover:bg-primary/20",
                )}
              >
                <Send className="h-[14px] w-[14px]" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function SessionResumeMenu({
  currentSessionLabel,
  historyLoading,
  historySessions,
  onSelect,
  resumeSessionId,
  selectedSessionId,
  selectedSessionInList,
}: {
  currentSessionLabel: string;
  historyLoading: boolean;
  historySessions: SessionInfo[];
  onSelect: (sessionId: string | null) => void;
  resumeSessionId: string | null;
  selectedSessionId: string;
  selectedSessionInList: boolean;
}) {
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const selectedSession = historySessions.find(
    (session) => session.id === selectedSessionId,
  );
  const selectedTitle = selectedSession
    ? formatSessionTitle(selectedSession)
    : selectedSessionId
      ? currentSessionLabel
      : historyLoading
        ? "loading history..."
        : "new chat";
  const selectedMeta = selectedSession
    ? formatSessionMeta(selectedSession)
    : resumeSessionId
      ? "active session"
      : "";
  const fallbackVisible = Boolean(resumeSessionId && !selectedSessionInList);
  const newChatActive = !resumeSessionId && !selectedSession;

  useEffect(() => {
    if (!open) return;
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key === "Escape") setOpen(false);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onMouseDown = (ev: MouseEvent) => {
      const target = ev.target as Node;
      if (wrapperRef.current?.contains(target)) return;
      if (menuRef.current?.contains(target)) return;
      setOpen(false);
    };
    document.addEventListener("mousedown", onMouseDown);
    return () => document.removeEventListener("mousedown", onMouseDown);
  }, [open]);

  useEffect(() => {
    setOpen(false);
  }, [selectedSessionId]);

  const choose = (sessionId: string | null) => {
    onSelect(sessionId);
    setOpen(false);
  };

  const rect = wrapperRef.current?.getBoundingClientRect();
  const menu =
    open && typeof document !== "undefined"
      ? createPortal(
          <div
            ref={menuRef}
            aria-label="Resume historical session"
            className={cn(
              "fixed z-[100] max-h-[min(384px,70dvh)] overflow-hidden",
              "rounded border border-border bg-background-base/95 shadow-[0_16px_40px_-12px_rgba(0,0,0,0.65)] backdrop-blur-sm",
            )}
            role="listbox"
            style={
              rect
                ? {
                    top: rect.bottom + 6,
                    right: Math.max(12, window.innerWidth - rect.right),
                    width: rect.width,
                  }
                : undefined
            }
          >
            <div className="flex items-center justify-between gap-[12px] border-b border-border px-[12px] py-[10px] font-sans text-[14px] leading-[20px]">
              <div className="min-w-0">
                <div className="truncate text-[14px] font-medium leading-[20px] text-foreground">
                  {selectedTitle}
                </div>
                {selectedMeta && (
                  <div className="truncate text-[12px] leading-[16px] text-text-tertiary">
                    {selectedMeta}
                  </div>
                )}
              </div>
              {historyLoading && (
                <Loader2 className="h-[14px] w-[14px] shrink-0 animate-spin text-text-tertiary" />
              )}
            </div>

            <div className="max-h-[calc(min(384px,70dvh)-57px)] overflow-y-auto py-[4px]">
              <SessionResumeOption
                active={newChatActive}
                icon={<Plus className="h-[14px] w-[14px]" />}
                label="new chat"
                onClick={() => choose(null)}
              />

              {fallbackVisible && (
                <SessionResumeOption
                  active
                  icon={<MessageSquare className="h-[14px] w-[14px]" />}
                  label={currentSessionLabel}
                  meta="active session"
                  onClick={() => choose(resumeSessionId)}
                />
              )}

              {historySessions.map((session) => (
                <SessionResumeOption
                  key={session.id}
                  active={session.id === selectedSessionId}
                  icon={<MessageSquare className="h-[14px] w-[14px]" />}
                  label={formatSessionTitle(session)}
                  meta={formatSessionMeta(session)}
                  onClick={() => choose(session.id)}
                />
              ))}

              {!historyLoading && historySessions.length === 0 && !fallbackVisible && (
                <div className="px-[12px] py-[12px] text-[12px] leading-[16px] text-text-tertiary">
                  no history
                </div>
              )}
            </div>
          </div>,
          document.body,
        )
      : null;

  return (
    <div ref={wrapperRef} className="relative min-w-0">
      <button
        type="button"
        aria-expanded={open}
        aria-haspopup="listbox"
        className={cn(
          "inline-flex h-[34px] w-[min(36vw,300px)] max-w-[300px] items-center justify-between gap-[8px] rounded border border-border",
          "bg-background/70 px-[10px] text-left font-sans text-[14px] font-medium leading-[20px] text-foreground normal-case tracking-normal",
          "transition-colors hover:border-foreground/30 hover:bg-midground/5",
          "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/30",
          "max-[640px]:w-[170px]",
        )}
        onClick={() => setOpen((value) => !value)}
        title={selectedSession ? formatSessionOption(selectedSession) : selectedTitle}
      >
        <span className="flex min-w-0 flex-1 items-center gap-[8px]">
          <MessageSquare className="h-[14px] w-[14px] shrink-0 text-text-tertiary" />
          <span className="block min-w-0 flex-1 truncate text-[14px] leading-[20px] text-foreground">
            {selectedTitle}
          </span>
        </span>
        <ChevronDown
          className={cn(
            "h-[14px] w-[14px] shrink-0 text-text-tertiary transition-transform",
            open && "rotate-180",
          )}
        />
      </button>
      {menu}
    </div>
  );
}

function SessionResumeOption({
  active,
  icon,
  label,
  meta,
  onClick,
}: {
  active: boolean;
  icon: ReactNode;
  label: string;
  meta?: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-selected={active}
      className={cn(
        "flex min-h-[40px] w-full items-center gap-[8px] px-[12px] py-[6px] text-left font-sans text-[14px] leading-[20px] transition-colors",
        active
          ? "bg-midground/10 text-foreground"
          : "text-text-secondary hover:bg-midground/5 hover:text-foreground",
      )}
      onClick={onClick}
      role="option"
      title={meta ? `${label} · ${meta}` : label}
    >
      <span
        className={cn(
          "flex h-[28px] w-[28px] shrink-0 items-center justify-center rounded border",
          active
            ? "border-primary/35 bg-primary/10 text-primary"
            : "border-border bg-background/50 text-text-tertiary",
        )}
      >
        {active ? <Check className="h-[14px] w-[14px]" /> : icon}
      </span>
      <div className="min-w-0 flex-1 text-[14px] leading-[20px]">
        <div className="truncate text-[14px] font-medium leading-[20px]">
          {label}
        </div>
        {meta && (
          <div className="truncate text-[12px] leading-[16px] text-text-tertiary">
            {meta}
          </div>
        )}
      </div>
    </button>
  );
}

function DialogMessageBubble({
  message,
  copied,
  onCopy,
}: {
  message: DialogMessage;
  copied: boolean;
  onCopy: (message: DialogMessage) => void;
}) {
  if (message.role === "system") {
    return (
      <div
        className={cn(
          "w-full rounded border px-3 py-2 text-xs",
          message.tone === "error"
            ? "border-destructive/40 bg-destructive/5 text-destructive"
            : "border-border bg-muted/20 text-text-secondary",
        )}
      >
        <Markdown content={message.text} />
      </div>
    );
  }

  const user = message.role === "user";
  return (
    <div className={cn("flex w-full", user ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "group relative rounded border px-3 py-2.5",
          user
            ? "max-w-[min(960px,88%)] border-primary/20 bg-primary/10 text-foreground"
            : message.tone === "error"
              ? "w-full border-destructive/40 bg-destructive/5"
              : "w-full border-border bg-background/65",
        )}
      >
        {user ? (
          <>
            <div className="whitespace-pre-wrap break-words text-sm leading-relaxed">
              {message.text}
            </div>
            {message.queued && (
              <div className="mt-1 flex items-center justify-end gap-1.5 text-[11px] leading-[14px] text-text-tertiary">
                <span className="h-1.5 w-1.5 rounded-full bg-primary/70" />
                <span>queued</span>
              </div>
            )}
          </>
        ) : (
          <>
            {message.streaming && !message.text ? (
              <div className="flex items-center gap-2 text-sm leading-relaxed text-text-secondary">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                <span>Waiting for response...</span>
              </div>
            ) : (
              <Markdown content={message.text} streaming={message.streaming} />
            )}
          </>
        )}

        {!user && message.text && (
          <Button
            ghost
            size="icon"
            title={copied ? "Copied" : "Copy"}
            aria-label={copied ? "Copied" : "Copy message"}
            onClick={() => onCopy(message)}
            className={cn(
              "absolute -right-2 -top-2 h-7 w-7 rounded border border-border bg-background/90",
              "opacity-0 shadow-sm transition-opacity group-hover:opacity-100 focus-visible:opacity-100",
            )}
          >
            <Copy className="h-3.5 w-3.5" />
          </Button>
        )}
      </div>
    </div>
  );
}

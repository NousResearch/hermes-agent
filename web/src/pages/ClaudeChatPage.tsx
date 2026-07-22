/**
 * Claude-style chat page for the Hermes Agent dashboard.
 *
 * Replaces the xterm.js TUI with a rich chat interface supporting
 * image paste, markdown rendering, and streaming responses.
 */

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Typography } from "@nous-research/ui/ui/components/typography/index";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { cn } from "@/lib/utils";
import { ChatGateway, getChatGateway } from "@/lib/chatGateway";
import { ChatHistorySidebar } from "@/components/ChatHistorySidebar";
import { ChatErrorBoundary } from "@/components/ChatErrorBoundary";
import type {
  GatewayEvent,
  ImageAttachment,
  DocumentAttachment,
  SessionInfo,
} from "@/lib/chatGateway";
import {
  ArrowUp,
  Square,
  X,
  Cpu,
  Paperclip,
  FileText,
  PanelLeft,
} from "lucide-react";

// ── Simple Markdown Renderer ────────────────────────────────────

function MarkdownContent({ text }: { text: string }) {
  // Split by code blocks
  const parts = text.split(/(```[^`]*```)/g);
  return (
    <div className="prose prose-invert prose-sm max-w-none break-words">
      {parts.map((part, i) => {
        if (part.startsWith("```")) {
          const lines = part.split("\n");
          const lang = lines[0].slice(3).trim();
          const code = lines.slice(1, -1).join("\n");
          return (
            <pre
              key={i}
              className="bg-black/30 border border-border/30 rounded-md p-3 my-2 overflow-x-auto text-xs font-mono"
            >
              {lang && (
                <div className="text-xs text-text-tertiary mb-1">{lang}</div>
              )}
              <code>{code}</code>
            </pre>
          );
        }
        // Inline markdown: bold, italic, inline code, links
        return (
          <div
            key={i}
            className="whitespace-pre-wrap leading-relaxed [&_strong]:font-semibold [&_strong]:text-foreground [&_em]:italic [&_code]:bg-black/20 [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-xs [&_code]:font-mono [&_code]:rounded [&_a]:underline [&_a]:text-primary [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:list-decimal [&_ol]:pl-5"
            dangerouslySetInnerHTML={{
              __html: part
                .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
                .replace(/\*(.+?)\*/g, "<em>$1</em>")
                .replace(/`([^`]+)`/g, "<code>$1</code>")
                .replace(
                  /\[([^\]]+)\]\(([^)]+)\)/g,
                  '<a href="$2" target="_blank" rel="noreferrer">$1</a>',
                )
                .replace(/^### (.+)$/gm, "<h3 class='text-sm font-semibold mt-3 mb-1'>$1</h3>")
                .replace(/^## (.+)$/gm, "<h2 class='text-base font-semibold mt-3 mb-1'>$1</h2>")
                .replace(/^# (.+)$/gm, "<h1 class='text-lg font-bold mt-3 mb-1'>$1</h1>")
                .replace(/^\- (.+)$/gm, "<li>$1</li>")
                .replace(/^(\d+)\. (.+)$/gm, "<li>$2</li>"),
            }}
          />
        );
      })}
    </div>
  );
}

// ── Image Preview ────────────────────────────────────────────────

function ImagePreview({
  image,
  onRemove,
}: {
  image: ImageAttachment;
  onRemove: () => void;
}) {
  return (
    <div className="relative group shrink-0">
      <img
        src={image.dataUrl}
        alt={image.name}
        className="h-16 w-16 object-cover rounded-md border border-border/50"
      />
      <button
        onClick={onRemove}
        className="absolute -top-1.5 -right-1.5 p-0.5 rounded-full bg-destructive text-destructive-foreground opacity-0 group-hover:opacity-100 transition-opacity"
        aria-label="Remove image"
      >
        <X className="h-3 w-3" />
      </button>
    </div>
  );
}

// ── Document Preview ──────────────────────────────────────────────

const DOC_ICONS: Record<string, string> = {
  pdf: "PDF",
  docx: "DOC",
  txt: "TXT",
  md: "MD",
  csv: "CSV",
  json: "JSON",
  py: "PY",
  yaml: "YML",
  yml: "YML",
  sh: "SH",
  html: "HTM",
  xml: "XML",
};

function DocumentPreview({
  document: doc,
  onRemove,
}: {
  document: DocumentAttachment;
  onRemove: () => void;
}) {
  const ext = doc.extension || doc.name.split(".").pop() || "";
  const label = DOC_ICONS[ext.toLowerCase()] || ext.toUpperCase() || "DOC";

  return (
    <div className="relative group shrink-0">
      <div className="h-16 w-16 rounded-md border border-border/50 bg-secondary/40 flex flex-col items-center justify-center gap-0.5">
        <FileText className="h-5 w-5 text-text-tertiary" />
        <span className="text-[10px] font-mono font-semibold text-text-tertiary leading-none">
          {label}
        </span>
      </div>
      <div className="text-[10px] text-text-tertiary text-center mt-0.5 truncate max-w-[64px] leading-tight">
        {doc.name.length > 12
          ? doc.name.slice(0, 10) + "..."
          : doc.name}
      </div>
      <button
        onClick={onRemove}
        className="absolute -top-1.5 -right-1.5 p-0.5 rounded-full bg-destructive text-destructive-foreground opacity-0 group-hover:opacity-100 transition-opacity"
        aria-label="Remove document"
      >
        <X className="h-3 w-3" />
      </button>
    </div>
  );
}

// ── Tool Call Display ────────────────────────────────────────────

function ToolCallBubble({ name, args }: { name: string; args: string }) {
  const [expanded, setExpanded] = useState(false);
  let parsed = args;
  try {
    parsed = JSON.stringify(JSON.parse(args), null, 2);
  } catch {}

  return (
    <div className="mt-2 border border-warning/20 bg-warning/5 rounded-md overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-3 py-1.5 text-xs text-warning/80 hover:bg-warning/10 flex items-center gap-2"
      >
        <Cpu className="h-3 w-3 shrink-0" />
        <span className="font-mono font-medium">{name}</span>
      </button>
      {expanded && (
        <pre className="border-t border-warning/20 px-3 py-2 text-xs text-warning/70 overflow-x-auto whitespace-pre-wrap font-mono">
          {parsed}
        </pre>
      )}
    </div>
  );
}

// ── Message Bubble ───────────────────────────────────────────────

function MessageBubble({
  role,
  content,
  toolCalls,
  isStreaming,
  images,
  documents,
}: {
  role: "user" | "assistant" | "tool";
  content: string;
  toolCalls?: Array<{ id: string; function: { name: string; arguments: string } }>;
  isStreaming?: boolean;
  images?: ImageAttachment[];
  documents?: DocumentAttachment[];
}) {
  const isUser = role === "user";
  const isTool = role === "tool";

  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[85%] min-w-0 rounded-lg px-4 py-3",
          isUser && "bg-[#0d4a4a] text-foreground",
          isTool && "bg-warning/10 text-warning/90 text-xs font-mono",
          !isUser && !isTool && "bg-secondary/40 text-foreground",
        )}
      >
        {content && (
          <div className="text-sm">
            {isStreaming && !content ? (
              <Spinner className="h-4 w-4 text-primary" />
            ) : (
              <MarkdownContent text={content} />
            )}
          </div>
        )}
        {images && images.length > 0 && (
          <div className="flex gap-1.5 mt-2 flex-wrap">
            {images.map((img, i) => (
              <img
                key={i}
                src={img.dataUrl}
                alt={img.name}
                className="h-12 w-12 object-cover rounded-md border border-border/30"
              />
            ))}
          </div>
        )}
        {documents && documents.length > 0 && (
          <div className="flex gap-1.5 mt-2 flex-wrap">
            {documents.map((doc, i) => (
              <div
                key={i}
                className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-secondary/30 border border-border/20 text-xs"
              >
                <FileText className="h-3 w-3 text-text-tertiary shrink-0" />
                <span className="text-text-secondary truncate max-w-[150px]">
                  {doc.name}
                </span>
              </div>
            ))}
          </div>
        )}
        {toolCalls?.map((tc) => (
          <ToolCallBubble
            key={tc.id}
            name={tc.function.name}
            args={tc.function.arguments}
          />
        ))}
        {isStreaming && content && (
          <span className="inline-block w-2 h-4 bg-primary/60 animate-pulse ml-0.5 align-middle" />
        )}
      </div>
    </div>
  );
}

// ── Main Chat Page ───────────────────────────────────────────────

interface ChatMessage {
  role: "user" | "assistant" | "tool";
  content: string;
  toolCalls?: Array<{ id: string; function: { name: string; arguments: string } }>;
  images?: ImageAttachment[];
  documents?: DocumentAttachment[];
}

export default function ClaudeChatPage() {
  const [gateway, setGateway] = useState<ChatGateway | null>(null);
  const [connected, setConnected] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [images, setImages] = useState<ImageAttachment[]>([]);
  const [documents, setDocuments] = useState<DocumentAttachment[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [model, setModel] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [currentResumedId, setCurrentResumedId] = useState<string | null>(null);
  const [connectionState, setConnectionState] = useState<"connecting" | "connected" | "disconnected">("connecting");

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const streamingContentRef = useRef<string>("");

  // ── Connection ────────────────────────────────────────────────

  useEffect(() => {
    let isMounted = true;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    const gw = getChatGateway();
    setGateway(gw);

    // Register event handlers BEFORE connecting (they persist across reconnects)
    gw.on("message.start", () => {
      streamingContentRef.current = "";
      setMessages((prev) => {
        if (prev.length > 0 && prev[prev.length - 1].role === "assistant") {
          return prev;
        }
        return [...prev, { role: "assistant", content: "" }];
      });
    });

    gw.on("message.delta", (ev: GatewayEvent) => {
      const delta = (ev.payload?.text as string) || "";
      streamingContentRef.current += delta;
      setMessages((prev) => {
        const next = [...prev];
        if (next.length > 0 && next[next.length - 1].role === "assistant") {
          next[next.length - 1] = {
            ...next[next.length - 1],
            content: streamingContentRef.current,
          };
        } else {
          next.push({ role: "assistant", content: delta });
        }
        return next;
      });
    });

    gw.on("message.complete", () => {
      setIsStreaming(false);
      streamingContentRef.current = "";
    });

    gw.on("error", (ev: GatewayEvent) => {
      const msg = (ev.payload?.message as string) || "Agent error";
      setError(msg);
      setIsStreaming(false);
    });

    gw.on("session.info", (ev: GatewayEvent) => {
      const info = ev.payload as SessionInfo | undefined;
      if (info) {
        setModel(info.model || "");
      }
    });

    gw.on("tool.start", (ev: GatewayEvent) => {
      const name = (ev.payload?.name as string) || "tool";
      setMessages((prev) => [
        ...prev,
        {
          role: "tool",
          content: "",
          toolCalls: [
            {
              id: `tc-${Date.now()}`,
              function: {
                name,
                arguments: JSON.stringify(ev.payload?.args || {}),
              },
            },
          ],
        },
      ]);
    });

    gw.on("tool.complete", (ev: GatewayEvent) => {
      setMessages((prev) => {
        const next = [...prev];
        for (let i = next.length - 1; i >= 0; i--) {
          if (next[i].role === "tool") {
            next[i] = {
              ...next[i],
              content: (ev.payload?.result as string) || "",
            };
            break;
          }
        }
        return next;
      });
    });

    // Handle disconnection — show reconnect UI
    gw.on("connection.closed", () => {
      if (!isMounted) return;
      setConnected(false);
      setConnectionState("disconnected");
      // Auto-reconnect after 2 seconds
      reconnectTimer = setTimeout(() => {
        if (isMounted) {
          doConnect();
        }
      }, 2000);
    });

    const doConnect = async () => {
      if (!isMounted) return;
      setConnectionState("connecting");
      setError(null);
      try {
        await gw.connect();
        if (!isMounted) return;
        setConnected(true);
        setConnectionState("connected");
        // Create session
        try {
          await gw.createSession();
        } catch (err) {
          setError(`Session creation failed: ${(err as Error).message}`);
        }
      } catch (err) {
        if (!isMounted) return;
        setConnected(false);
        setConnectionState("disconnected");
        setError(`Connection failed: ${(err as Error).message}`);
        // Auto-retry after 3 seconds
        reconnectTimer = setTimeout(() => {
          if (isMounted) {
            doConnect();
          }
        }, 3000);
      }
    };

    doConnect();

    return () => {
      isMounted = false;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      gw.disconnect();
    };
  }, []);

  // ── Auto-scroll ────────────────────────────────────────────────

  useLayoutEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isStreaming]);

  // ── Image paste handling ──────────────────────────────────────

  const handlePaste = useCallback(
    async (e: React.ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;

      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        if (item.type.startsWith("image/")) {
          e.preventDefault();
          const file = item.getAsFile();
          if (!file) continue;

          const reader = new FileReader();
          reader.onload = async () => {
            const dataUrl = reader.result as string;
            // Add as preview immediately
            const preview: ImageAttachment = {
              path: "",
              name: `pasted-${Date.now()}.png`,
              size: file.size,
              mime_type: file.type,
              dataUrl,
            };
            setImages((prev) => [...prev, preview]);

            // Upload and attach
            if (gateway?.connected) {
              try {
                const attached = await gateway.uploadAndAttachImage(dataUrl);
                setImages((prev) =>
                  prev.map((img) =>
                    img.dataUrl === dataUrl ? { ...img, path: attached.path } : img,
                  ),
                );
              } catch (err) {
                setError(`Image upload failed: ${(err as Error).message}`);
              }
            }
          };
          reader.readAsDataURL(file);
        }
      }
    },
    [gateway],
  );

  // ── Document upload handling ────────────────────────────────────

  const handleUploadDocument = useCallback(
    async (file: File) => {
      const reader = new FileReader();
      reader.onload = async () => {
        const dataUrl = reader.result as string;
        const preview: DocumentAttachment = {
          path: "",
          name: file.name,
          size: file.size,
          mime_type: file.type,
          extension: file.name.split(".").pop() || "",
          dataUrl,
        };
        setDocuments((prev) => [...prev, preview]);

        if (gateway?.connected) {
          try {
            const attached = await gateway.uploadAndAttachDocument(
              dataUrl,
              file.name,
              file.type,
            );
            setDocuments((prev) =>
              prev.map((doc) =>
                doc.dataUrl === dataUrl
                  ? {
                      ...doc,
                      path: attached.path,
                      extracted_text: attached.extracted_text,
                      preview: attached.preview,
                    }
                  : doc,
              ),
            );
          } catch (err) {
            setError(`Document upload failed: ${(err as Error).message}`);
          }
        }
      };
      reader.readAsDataURL(file);
    },
    [gateway],
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (!files) return;
      for (let i = 0; i < files.length; i++) {
        handleUploadDocument(files[i]);
      }
      // Reset so same file can be selected again
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    },
    [handleUploadDocument],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const items = e.dataTransfer?.items;
      if (!items) return;

      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        if (item.kind === "file") {
          const file = item.getAsFile();
          if (file && !file.type.startsWith("image/")) {
            handleUploadDocument(file);
          }
          // Images are handled by the paste handler's clipboard flow;
          // file drops for images work via the paste interception already.
        }
      }
    },
    [handleUploadDocument],
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  }, []);

  // ── Session management ──────────────────────────────────────────

  const handleNewChat = useCallback(async () => {
    if (!gateway?.connected) return;
    try {
      await gateway.createSession();
      setMessages([]);
      setCurrentResumedId(null);
      setSidebarOpen(false);
      setError(null);
    } catch (err) {
      setError(`New chat failed: ${(err as Error).message}`);
    }
  }, [gateway]);

  const handleResumeSession = useCallback(
    async (sessionId: string) => {
      if (!gateway?.connected) return;
      setError(null);
      try {
        const result = await gateway.resumeSession(sessionId);
        // Map the returned messages to our ChatMessage format
        const loaded: ChatMessage[] = (result.messages || []).map(
          (m: { role: string; content: string; tool_calls?: Array<{ id: string; function: { name: string; arguments: string } }> }) => ({
            role: m.role as "user" | "assistant" | "tool",
            content: m.content || "",
            toolCalls: m.tool_calls,
          }),
        );
        setMessages(loaded);
        setCurrentResumedId(result.resumed || sessionId);
        setSidebarOpen(false);
      } catch (err) {
        setError(`Resume failed: ${(err as Error).message}`);
      }
    },
    [gateway],
  );

  // ── Send message ──────────────────────────────────────────────

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text && images.length === 0 && documents.length === 0) return;
    if (!gateway?.connected) {
      setError("Not connected to agent");
      return;
    }

    // Clear input immediately
    setInput("");
    const currentImages = [...images];
    const currentDocuments = [...documents];
    setImages([]);
    setDocuments([]);

    // Add user message
    setMessages((prev) => [
      ...prev,
      { role: "user", content: text, images: currentImages, documents: currentDocuments },
    ]);

    setIsStreaming(true);
    streamingContentRef.current = "";
    setError(null);

    try {
      await gateway.submitPrompt(text || "(see attached image)");
    } catch (err) {
      setError(`Failed to send: ${(err as Error).message}`);
      setIsStreaming(false);
    }
  }, [input, images, documents, gateway]);

  // ── Stop generation ───────────────────────────────────────────

  const handleStop = useCallback(async () => {
    if (gateway) {
      await gateway.interrupt();
    }
  }, [gateway]);

  // ── Key handler ────────────────────────────────────────────────

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  // ── Auto-resize textarea ──────────────────────────────────────

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      setInput(e.target.value);
      const el = e.target;
      el.style.height = "auto";
      el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
    },
    [],
  );

  // ── Render ────────────────────────────────────────────────────

  if (error && !connected) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 p-8">
        <Typography className="text-destructive text-sm">{error}</Typography>
        <Button onClick={() => window.location.reload()}>Retry</Button>
      </div>
    );
  }

  return (
    <ChatErrorBoundary>
    <div className="flex h-full max-h-full min-h-0 bg-background-base">
      {/* Chat history sidebar */}
      <ChatHistorySidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        currentSessionId={currentResumedId || gateway?.sessionId || null}
        onNewChat={handleNewChat}
        onResumeSession={handleResumeSession}
      />

      {/* Main chat area */}
      <div className="flex flex-col flex-1 min-w-0 h-full max-h-full min-h-0">
      {/* Header */}
      <div className="shrink-0 flex items-center justify-between px-4 py-2 border-b border-border/20 bg-background-base/50">
        <div className="flex items-center gap-2 min-w-0">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-1 rounded-md hover:bg-secondary/50 text-text-tertiary hover:text-foreground transition-colors"
            aria-label="Toggle chat history"
            title="Chat history"
          >
            <PanelLeft className="h-4 w-4" />
          </button>
          <Typography className="text-sm font-medium truncate">
            Chat
          </Typography>
          {model && (
            <Badge tone="outline" className="text-xs">
              {model}
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-2">
          {connectionState === "connecting" && <Spinner className="h-4 w-4 text-warning" />}
          {connectionState === "connected" && (
            <span className="inline-block h-2 w-2 rounded-full bg-success animate-pulse" />
          )}
          {connectionState === "disconnected" && (
            <span className="inline-block h-2 w-2 rounded-full bg-destructive" />
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 min-h-0 overflow-y-auto px-4 py-4 space-y-4" onDrop={handleDrop} onDragOver={handleDragOver}>
        {connectionState === "disconnected" && (
          <div className="flex items-center justify-between px-3 py-2 bg-destructive/10 border border-destructive/20 rounded-md text-sm">
            <span className="text-destructive">Connection lost — reconnecting...</span>
            <Button
              size="sm"
              outlined
              onClick={() => window.location.reload()}
              className="text-xs"
            >
              Reload page
            </Button>
          </div>
        )}
        {messages.length === 0 && !isStreaming && (
          <div className="flex flex-col items-center justify-center h-full text-text-tertiary gap-2">
            <Typography className="text-sm">
              Send a message to start chatting with Hermes.
            </Typography>
            <Typography className="text-xs">
              Paste images directly, upload documents, or drag & drop files.
            </Typography>
          </div>
        )}

        {messages.map((msg, i) => (
          <MessageBubble
            key={i}
            role={msg.role}
            content={msg.content}
            toolCalls={msg.toolCalls}
            images={msg.images}
            documents={msg.documents}
            isStreaming={
              isStreaming &&
              i === messages.length - 1 &&
              msg.role === "assistant"
            }
          />
        ))}

        {error && (
          <div className="px-4 py-2 bg-destructive/10 border border-destructive/20 rounded-md text-sm text-destructive">
            {error}
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="shrink-0 border-t border-border/20 bg-background-base/50 p-3">
        {/* Image previews */}
        {images.length > 0 && (
          <div className="flex gap-2 mb-3 flex-wrap">
            {images.map((img, i) => (
              <ImagePreview
                key={i}
                image={img}
                onRemove={() =>
                  setImages((prev) => prev.filter((_, j) => j !== i))
                }
              />
            ))}
          </div>
        )}

        {/* Document previews */}
        {documents.length > 0 && (
          <div className="flex gap-2 mb-3 flex-wrap">
            {documents.map((doc, i) => (
              <DocumentPreview
                key={i}
                document={doc}
                onRemove={() => {
                  // Detach from session if path is set
                  if (doc.path && gateway) {
                    gateway.detachDocument(doc.path).catch(() => {});
                  }
                  setDocuments((prev) => prev.filter((_, j) => j !== i));
                }}
              />
            ))}
          </div>
        )}

        {/* Input row */}
        <div className="flex items-end gap-2 bg-secondary/30 border border-border/30 rounded-lg px-3 py-2 focus-within:border-primary/40 transition-colors">
          {/* File upload button */}
          <button
            onClick={() => fileInputRef.current?.click()}
            className="shrink-0 p-1.5 rounded-md hover:bg-secondary/50 text-text-tertiary hover:text-foreground transition-colors mb-0.5"
            aria-label="Upload document"
            title="Upload document"
            disabled={!connected}
          >
            <Paperclip className="h-4 w-4" />
          </button>
          <input
            ref={fileInputRef}
            type="file"
            onChange={handleFileSelect}
            className="hidden"
            multiple
            accept=".pdf,.docx,.txt,.md,.csv,.json,.py,.yaml,.yml,.sh,.bash,.html,.htm,.xml,.toml,.ini,.cfg,.conf,.log,.env,.ts,.js,.jsx,.tsx,.css,.rs,.go,.java,.c,.cpp,.h,.rst,.tex"
          />

          <textarea
            ref={inputRef}
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            onPaste={handlePaste}
            placeholder="Type a message... (paste images, upload documents)"
            rows={1}
            className="flex-1 bg-transparent resize-none outline-none text-sm placeholder:text-text-tertiary min-h-[24px] max-h-[200px] py-1"
            disabled={!connected}
          />

          <div className="flex items-center gap-1 shrink-0">
            {isStreaming ? (
              <Button
                size="icon"
                destructive
                onClick={handleStop}
                className="h-8 w-8"
                aria-label="Stop generation"
              >
                <Square className="h-4 w-4" />
              </Button>
            ) : (
              <Button
                size="icon"
                onClick={handleSend}
                disabled={!input.trim() && images.length === 0 && documents.length === 0}
                className="h-8 w-8 bg-primary hover:bg-primary/80 text-primary-foreground"
                aria-label="Send message"
              >
                <ArrowUp className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>

        {/* Footer hint */}
        <Typography className="text-xs text-text-tertiary text-center mt-2">
          Hermes Agent v0.15.1 — Enter to send, Shift+Enter for new line, drag & drop files, paste images
        </Typography>
      </div>
      </div>
    </div>
    </ChatErrorBoundary>
  );
}

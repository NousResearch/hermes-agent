/**
 * CopilotPage — chrome-less chat-bubble copilot, served standalone at
 * `/copilot` (no dashboard sidebar/shell).  This is what the desktop
 * workflow view docks on the right of the langflow canvas via a webview.
 *
 * Multi-session: a tab bar across the top holds several gateway sessions at
 * once (new / switch / close), and a 🕐 history dropdown lists stored
 * conversations to reopen, rename, pin, archive or delete. All session state +
 * gateway wiring lives in `useCopilotSessions`; this file is the view.
 *
 * It talks to the SAME in-process gateway as the dashboard's native chat
 * (`GatewayClient` → /api/ws), so prompts run the full agent loop with its
 * configured MCP servers (kari_canvas / kari_org) live — the copilot can drive
 * the canvas while you chat.
 */
import { useEffect, useMemo, useRef, useState } from "react";

import CopilotHistory from "@/components/CopilotHistory";
import {
  useCopilotSessions,
  type CopilotMessage,
  type CopilotTab,
} from "@/hooks/useCopilotSessions";

// A few friendly labels for common agent + canvas tools; anything else
// falls back to the raw tool name.
const TOOL_LABELS: Record<string, string> = {
  terminal: "运行命令",
  process: "管理进程",
  file: "读写文件",
  web: "上网查资料",
  web_search: "搜索网络",
  list_components: "查找组件",
  add_node: "添加节点",
  add_python_node: "写自定义节点",
  connect: "连接节点",
  update_node: "修改节点",
  delete_node: "删除节点",
  get_flow: "读取画布",
};

const toolLabel = (name?: string) =>
  (name && (TOOL_LABELS[name] ?? name)) || "处理中";

const STATE_LABEL: Record<string, string> = {
  idle: "连接中…",
  connecting: "连接中…",
  open: "在线",
  closed: "已断开",
  error: "连接失败",
};

// Shimmer + bounce-dots for the "thinking" affordance.
const COPILOT_STYLES = `
@keyframes copilot-flow { to { background-position: 220% center; } }
@keyframes copilot-bounce { 0%,80%,100% { transform: scale(0.5); opacity:.45 } 40% { transform: scale(1); opacity:1 } }
:root {
  --copilot-surface: var(--ui-chat-surface-background, #f4f7fb);
  --copilot-elevated: var(--ui-bg-elevated, #ffffff);
  --copilot-bubble: var(--ui-chat-bubble-background, #ffffff);
  --copilot-control: var(--ui-control-hover-background, #e9eff7);
  --copilot-control-active: var(--ui-control-active-background, #dfe8f6);
  --copilot-border: var(--ui-stroke-secondary, #d4deeb);
  --copilot-border-soft: var(--ui-stroke-tertiary, #e2e8f1);
  --copilot-text: var(--ui-text-primary, #1f2937);
  --copilot-text-muted: var(--ui-text-secondary, #526173);
  --copilot-text-tertiary: var(--ui-text-tertiary, #8593a6);
  --copilot-accent: var(--ui-accent, #4f6fad);
  --copilot-accent-contrast: var(--ui-accent-foreground, #ffffff);
  --copilot-danger: var(--ui-red, #c2415d);
}
.copilot-shimmer { background-image: linear-gradient(90deg,var(--copilot-accent),color-mix(in srgb,var(--copilot-accent) 55%,#ec4899),color-mix(in srgb,var(--copilot-accent) 55%,#06b6d4),var(--copilot-accent)); background-size: 220% auto; -webkit-background-clip: text; background-clip: text; color: transparent; animation: copilot-flow 2.4s linear infinite; }
.copilot-dot { display:inline-block; width:5px; height:5px; border-radius:9999px; margin-left:4px; background:var(--copilot-accent); animation: copilot-bounce 1.1s infinite ease-in-out; }
`;

function Dots() {
  return (
    <>
      <span className="copilot-dot" style={{ animationDelay: "0ms" }} />
      <span className="copilot-dot" style={{ animationDelay: "160ms" }} />
      <span className="copilot-dot" style={{ animationDelay: "320ms" }} />
    </>
  );
}

/** Tab label: explicit title → first user line → fallback. */
function tabTitle(tab: CopilotTab): string {
  if (tab.title.trim()) return tab.title.trim();
  const firstUser = tab.messages.find((m) => m.role === "user");
  if (firstUser?.text) {
    const line = firstUser.text.replace(/\s+/g, " ").trim();
    return line.length > 18 ? `${line.slice(0, 18)}…` : line;
  }
  return "新会话";
}

// EasyHermes downlink tiers shown in the header model switcher. `provider` is
// sent explicitly on switch so the gateway routes to the right downlink (a bare
// 性能/极致 can fall back to openrouter). 极致 → claude-opus-4-8 (kari-extreme),
// 性能 → gpt-5.x (kari-cloud), both via the lotjc gateway.
const MODEL_OPTIONS = [
  { model: "性能", provider: "kari-cloud" },
  { model: "极致", provider: "kari-extreme" },
] as const;

export default function CopilotPage() {
  const cs = useCopilotSessions();
  const { state, model, banner, tabs, activeId, activeTab } = cs;

  const [input, setInput] = useState("");
  const [historyOpen, setHistoryOpen] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const busy = !!activeTab?.busy;
  const messages = useMemo(() => activeTab?.messages ?? [], [activeTab]);

  // Stick to the bottom as new content streams into the active tab.
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages]);

  const openTabIds = useMemo(() => new Set(tabs.map((t) => t.id)), [tabs]);

  const submit = () => {
    const text = input.trim();
    if (!text || busy || state !== "open" || !activeId) return;
    cs.send(text);
    setInput("");
  };

  const modelLabel = model ? model.split("/").slice(-1)[0] : null;

  return (
    <div className="flex h-dvh max-h-dvh min-h-0 flex-col bg-[var(--copilot-surface)] text-[var(--copilot-text)]">
      <style>{COPILOT_STYLES}</style>

      {/* header */}
      <div className="relative flex shrink-0 items-center justify-between border-b border-[var(--copilot-border-soft)] bg-[var(--copilot-surface)] px-3 py-2.5">
        <div className="flex min-w-0 items-center gap-2">
          <svg
            viewBox="0 0 24 24"
            className="h-4 w-4 text-[var(--copilot-accent)]"
            fill="currentColor"
            aria-hidden
          >
            <path d="M12 2l1.7 5.1L19 9l-5.3 1.9L12 16l-1.7-5.1L5 9l5.3-1.9L12 2z" />
          </svg>
          <span className="truncate text-sm font-medium">爱马仕 Copilot</span>
          {/* 模型切换:性能 ↔ 极致。切换走网关 config.set;对话进行中禁用 */}
          <div className="flex shrink-0 items-center gap-0.5 rounded-md border border-[var(--copilot-border-soft)] bg-[var(--copilot-control)] p-0.5">
            {MODEL_OPTIONS.map((opt) => {
              const active = modelLabel === opt.model;
              return (
                <button
                  key={opt.model}
                  type="button"
                  disabled={active || busy || state !== "open"}
                  onClick={() => void cs.switchModel(opt.model, opt.provider)}
                  title={busy ? "对话进行中,先停止再切换模型" : `切换到「${opt.model}」`}
                  className={`rounded px-1.5 py-0.5 text-[10px] leading-none transition disabled:cursor-not-allowed ${
                    active
                      ? "bg-[var(--copilot-accent)] text-white shadow-sm"
                      : "text-[var(--copilot-text-tertiary)] hover:text-[var(--copilot-text)] disabled:opacity-40"
                  }`}
                >
                  {opt.model}
                </button>
              );
            })}
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          <span
            className={`h-1.5 w-1.5 rounded-full ${
              state === "open"
                ? "bg-emerald-500"
                : state === "error" || state === "closed"
                  ? "bg-red-500"
                  : "bg-amber-500"
            }`}
            title={STATE_LABEL[state] ?? state}
          />
          <button
            type="button"
            onClick={() => setHistoryOpen((v) => !v)}
            className={`rounded-md p-1 hover:bg-[var(--copilot-control)] hover:text-[var(--copilot-text)] ${
              historyOpen ? "bg-[var(--copilot-control-active)] text-[var(--copilot-text)]" : "text-[var(--copilot-text-tertiary)]"
            }`}
            title="历史会话"
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <circle cx="12" cy="12" r="9" />
              <path d="M12 7v5l3 2" />
            </svg>
          </button>
          <button
            type="button"
            onClick={cs.newTab}
            className="rounded-md p-1 text-[var(--copilot-text-tertiary)] hover:bg-[var(--copilot-control)] hover:text-[var(--copilot-text)]"
            title="新会话"
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" aria-hidden>
              <path d="M12 5v14M5 12h14" />
            </svg>
          </button>
        </div>

        <CopilotHistory
          open={historyOpen}
          onClose={() => setHistoryOpen(false)}
          openTabIds={openTabIds}
          listHistory={cs.listHistory}
          renameSession={cs.renameSession}
          deleteSession={cs.deleteSession}
          onOpen={cs.openHistory}
        />
      </div>

      {/* tab bar */}
      {tabs.length > 0 && (
        <div className="flex shrink-0 items-center gap-1 overflow-x-auto border-b border-[var(--copilot-border-soft)] bg-[var(--copilot-surface)] px-2 py-1">
          {tabs.map((tab) => {
            const active = tab.id === activeId;
            return (
              <div
                key={tab.id}
                onClick={() => cs.switchTab(tab.id)}
                className={`group flex max-w-[160px] shrink-0 cursor-pointer items-center gap-1.5 rounded-md border px-2 py-1 text-xs ${
                  active
                    ? "border-[var(--copilot-border)] bg-[var(--copilot-elevated)] text-[var(--copilot-text)]"
                    : "border-transparent text-[var(--copilot-text-muted)] hover:bg-[var(--copilot-control)]"
                }`}
                title={tabTitle(tab)}
              >
                {tab.busy && (
                  <span className="h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-[var(--copilot-accent)]" />
                )}
                <span className="truncate">{tabTitle(tab)}</span>
                {tabs.length > 1 && (
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      cs.closeTab(tab.id);
                    }}
                    className="shrink-0 rounded p-0.5 text-[var(--copilot-text-tertiary)] opacity-0 hover:bg-[var(--copilot-control-active)] hover:text-[var(--copilot-text)] group-hover:opacity-100"
                    title="关闭"
                  >
                    <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth={2.5} strokeLinecap="round" aria-hidden>
                      <path d="M6 6l12 12M18 6L6 18" />
                    </svg>
                  </button>
                )}
              </div>
            );
          })}
        </div>
      )}

      {banner && (
        <div className="flex shrink-0 items-center justify-between gap-2 border-b border-[color-mix(in_srgb,var(--copilot-danger)_35%,transparent)] bg-[color-mix(in_srgb,var(--copilot-danger)_10%,var(--copilot-surface))] px-3 py-1.5 text-xs text-[var(--copilot-danger)]">
          <span className="min-w-0 truncate">{banner}</span>
          <button
            type="button"
            onClick={cs.reconnect}
            className="shrink-0 rounded border border-[color-mix(in_srgb,var(--copilot-danger)_40%,transparent)] px-1.5 py-0.5 hover:bg-[color-mix(in_srgb,var(--copilot-danger)_16%,transparent)]"
          >
            重连
          </button>
        </div>
      )}

      {/* messages */}
      <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto px-4 py-3">
        {messages.length === 0 && (
          <div className="mt-10 text-center text-sm text-[var(--copilot-text-tertiary)]">
            <p className="mb-2 font-medium text-[var(--copilot-text-muted)]">
              我是爱马仕,这个画布的全能助手
            </p>
            <p>搭工作流、写自定义节点、读数据、上网查资料…</p>
            <p className="mt-2">直接说需求就行。</p>
          </div>
        )}

        {messages.map((m: CopilotMessage) => {
          if (m.role === "user") {
            return (
              <div key={m.id} className="flex justify-end">
                <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-sm border border-[var(--copilot-border-soft)] bg-[var(--copilot-control-active)] px-3 py-2 text-sm text-[var(--copilot-text)]">
                  {m.text}
                </div>
              </div>
            );
          }

          const statusText =
            m.status === "thinking"
              ? "思考中"
              : m.status?.startsWith("running_tool:")
                ? toolLabel(m.status.slice("running_tool:".length))
                : null;

          if (!m.text && !m.reasoning && !m.error && statusText) {
            return (
              <div key={m.id} className="flex justify-start">
                <div className="max-w-[90%] rounded-2xl rounded-bl-sm border border-[var(--copilot-border-soft)] bg-[var(--copilot-bubble)] px-3 py-2 text-sm text-[var(--copilot-text)]">
                  <span className="inline-flex items-center font-medium">
                    <span className="copilot-shimmer">{statusText}</span>
                    <Dots />
                  </span>
                </div>
              </div>
            );
          }

          return (
            <div key={m.id} className="flex justify-start">
              <div className="max-w-[90%] rounded-2xl rounded-bl-sm border border-[var(--copilot-border-soft)] bg-[var(--copilot-bubble)] px-3 py-2 text-sm text-[var(--copilot-text)]">
                {m.reasoning &&
                  (!m.done ? (
                    <div className="mb-1.5 max-h-40 overflow-y-auto whitespace-pre-wrap border-l-2 border-[var(--copilot-border)] pl-2 text-xs italic text-[var(--copilot-text-tertiary)]">
                      {m.reasoning}
                    </div>
                  ) : (
                    <details className="mb-1.5 text-xs text-[var(--copilot-text-tertiary)]">
                      <summary className="cursor-pointer select-none">思考过程</summary>
                      <div className="mt-1 whitespace-pre-wrap border-l-2 border-[var(--copilot-border)] pl-2">
                        {m.reasoning}
                      </div>
                    </details>
                  ))}

                {m.text && <span className="whitespace-pre-wrap">{m.text}</span>}

                {m.error && (
                  <span className="whitespace-pre-wrap text-[var(--copilot-danger)]">
                    ⚠️ {m.error}
                  </span>
                )}

                {m.text && statusText && (
                  <div className="mt-1 text-xs">
                    <span className="copilot-shimmer font-medium">{statusText}</span>
                    <Dots />
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* composer */}
      <div className="shrink-0 p-3">
        <div className="rounded-xl border border-[var(--copilot-border)] bg-[var(--copilot-elevated)] focus-within:border-[var(--copilot-accent)]">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                submit();
              }
            }}
            rows={1}
            placeholder="给爱马仕安排活儿(Enter 发送,Shift+Enter 换行)"
            className="max-h-40 w-full resize-none bg-transparent px-3 pb-1.5 pt-3 text-sm leading-relaxed text-[var(--copilot-text)] outline-none placeholder:text-xs placeholder:text-[var(--copilot-text-tertiary)]"
          />
          <div className="flex items-center justify-end px-2 pb-2">
            {busy ? (
              <button
                type="button"
                onClick={cs.stop}
                className="flex h-7 w-7 items-center justify-center rounded-full bg-[var(--copilot-control-active)] text-[var(--copilot-text)] hover:bg-[var(--copilot-control)]"
                title="停止"
              >
                <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="currentColor" aria-hidden>
                  <rect x="6" y="6" width="12" height="12" rx="2" />
                </svg>
              </button>
            ) : (
              <button
                type="button"
                onClick={submit}
                disabled={!input.trim() || state !== "open"}
                className="flex h-7 w-7 items-center justify-center rounded-full bg-[var(--copilot-accent)] text-[var(--copilot-accent-contrast)] hover:opacity-90 disabled:opacity-40"
                title="发送"
              >
                <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                  <path d="M12 19V5M5 12l7-7 7 7" />
                </svg>
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
